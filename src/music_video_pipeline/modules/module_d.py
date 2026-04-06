"""
文件用途：实现模块 D（视频合成）的 MVP 版本。
核心流程：先将每个静态帧渲染为小视频片段，再按顺序拼接并混入原音轨输出 MP4。
输入输出：输入 RuntimeContext，输出最终视频文件路径。
依赖说明：依赖标准库 subprocess 调用 FFmpeg/FFprobe。
维护说明：本文件只保留“段视频->总拼接”方案，不再保留旧的图片直接 concat 方案。
"""

# 标准库：用于并发调度
from concurrent.futures import ProcessPoolExecutor, as_completed
# 标准库：用于子进程命令执行
import subprocess
# 标准库：用于函数结果缓存
from functools import lru_cache
# 标准库：用于日志对象类型提示
import logging
# 标准库：用于多进程上下文
import multiprocessing
# 标准库：用于阶段计时
import time
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json


@lru_cache(maxsize=8)
def _probe_ffmpeg_encoder_capabilities(ffmpeg_bin: str) -> dict[str, bool]:
    """
    功能说明：探测 ffmpeg 编码器能力并缓存结果。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    返回值：
    - dict[str, bool]: 编码器可用性字典（键为编码器名）。
    异常说明：探测失败时返回空能力集合，不在此处抛错。
    边界条件：缓存按 ffmpeg_bin 维度生效，避免重复探测开销。
    """
    command = [ffmpeg_bin, "-hide_banner", "-encoders"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except Exception:  # noqa: BLE001
        return {}

    output_text = str(result.stdout)
    supported = {
        "h264_nvenc": "h264_nvenc" in output_text,
        "hevc_nvenc": "hevc_nvenc" in output_text,
    }
    return supported


def _normalize_video_accel_mode(video_accel_mode: str) -> str:
    """
    功能说明：归一化视频加速模式配置。
    参数说明：
    - video_accel_mode: 原始配置值。
    返回值：
    - str: 合法模式（auto/cpu_only/gpu_only）。
    异常说明：无。
    边界条件：非法值回退为 auto。
    """
    normalized = str(video_accel_mode).strip().lower()
    if normalized in {"auto", "cpu_only", "gpu_only"}:
        return normalized
    return "auto"


def _normalize_concat_video_mode(concat_video_mode: str) -> str:
    """
    功能说明：归一化最终拼接视频模式配置。
    参数说明：
    - concat_video_mode: 原始配置值。
    返回值：
    - str: 合法模式（copy/reencode）。
    异常说明：无。
    边界条件：非法值回退为 copy。
    """
    normalized = str(concat_video_mode).strip().lower()
    if normalized in {"copy", "reencode"}:
        return normalized
    return "copy"


def _normalize_render_batch_size(render_batch_size: int, logger: logging.Logger) -> int:
    """
    功能说明：归一化兼容字段 render_batch_size（当前固定单段渲染）。
    参数说明：
    - render_batch_size: 原始批大小配置值（兼容旧配置）。
    - logger: 日志对象，用于记录配置纠正信息。
    返回值：
    - int: 固定返回 1。
    异常说明：无。
    边界条件：当值非法或不为 1 时记录 warning，并统一按 1 执行。
    """
    try:
        normalized = int(render_batch_size)
    except (TypeError, ValueError):
        logger.warning("模块D兼容字段 render_batch_size 非法，已回退为1，raw_value=%s", render_batch_size)
        return 1
    if normalized < 1:
        logger.warning("模块D兼容字段 render_batch_size 非法（<1），已回退为1，raw_value=%s", render_batch_size)
        return 1
    if normalized != 1:
        logger.warning(
            "模块D兼容字段 render_batch_size=%s 在新渲染路径下将固定按1执行，请改用 render_workers 控制并发。",
            normalized,
        )
    return 1


def _normalize_render_workers(render_workers: int, logger: logging.Logger) -> int:
    """
    功能说明：归一化受控并行 worker 数量。
    参数说明：
    - render_workers: 原始 worker 配置值。
    - logger: 日志对象，用于记录配置纠正信息。
    返回值：
    - int: 合法 worker 数量（范围 1~4）。
    异常说明：无。
    边界条件：非法值统一回退为 3，并写 warning。
    """
    try:
        normalized = int(render_workers)
    except (TypeError, ValueError):
        logger.warning("模块D渲染并发配置非法，已回退默认值3，raw_value=%s", render_workers)
        return 3
    if normalized < 1 or normalized > 4:
        logger.warning("模块D渲染并发超出建议范围[1,4]，已回退默认值3，raw_value=%s", render_workers)
        return 3
    return normalized


def _clamp_nvenc_cq(gpu_cq: int | None, fallback_crf: int) -> int:
    """
    功能说明：获取 NVENC CQ 参数，未配置时从 CRF 近似映射。
    参数说明：
    - gpu_cq: 配置中的 GPU CQ（可选）。
    - fallback_crf: CPU 路径 CRF，作为 CQ 近似值来源。
    返回值：
    - int: 归一化后的 CQ 值。
    异常说明：无。
    边界条件：CQ 限制在 [0, 51]。
    """
    if gpu_cq is None:
        candidate = int(fallback_crf)
    else:
        candidate = int(gpu_cq)
    return max(0, min(51, candidate))


def _normalize_nvenc_rc_mode_for_preset(gpu_rc_mode: str, gpu_preset: str) -> str:
    """
    功能说明：在旧版 NVENC 约束下归一化 RC 模式，避免与 p1~p7 预设冲突。
    参数说明：
    - gpu_rc_mode: 原始 RC 模式配置。
    - gpu_preset: 原始 GPU 预设配置。
    返回值：
    - str: 兼容后的 RC 模式。
    异常说明：无。
    边界条件：当 preset 为 p1~p7 且 rc 为 vbr_hq/cbr_hq 时自动降级为 vbr/cbr。
    """
    normalized_rc = str(gpu_rc_mode).strip().lower() or "vbr"
    normalized_preset = str(gpu_preset).strip().lower()
    preset_is_p_series = normalized_preset.startswith("p") and normalized_preset[1:].isdigit()
    if preset_is_p_series and normalized_rc == "vbr_hq":
        return "vbr"
    if preset_is_p_series and normalized_rc == "cbr_hq":
        return "cbr"
    return normalized_rc


def _resolve_video_encoder_profile(
    ffmpeg_bin: str,
    video_accel_mode: str,
    cpu_video_codec: str,
    cpu_video_preset: str,
    cpu_video_crf: int,
    gpu_video_codec: str,
    gpu_preset: str,
    gpu_rc_mode: str,
    gpu_cq: int | None,
    gpu_bitrate: str | None,
) -> dict[str, Any]:
    """
    功能说明：根据配置与能力探测结果选择视频编码方案。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - cpu_video_codec/cpu_video_preset/cpu_video_crf: CPU 编码配置。
    - gpu_video_codec/gpu_preset/gpu_rc_mode/gpu_cq/gpu_bitrate: GPU 编码配置。
    返回值：
    - dict[str, Any]: 编码方案（含 `use_gpu`、`command_args`、`fallback_cpu_profile`）。
    异常说明：gpu_only 且编码器不可用时抛 RuntimeError。
    边界条件：auto 模式下 GPU 不可用自动回退 CPU。
    """
    normalized_mode = _normalize_video_accel_mode(video_accel_mode=video_accel_mode)
    normalized_gpu_codec = str(gpu_video_codec).strip().lower() or "h264_nvenc"
    capabilities = _probe_ffmpeg_encoder_capabilities(ffmpeg_bin=ffmpeg_bin)
    gpu_available = bool(capabilities.get(normalized_gpu_codec, False))

    cpu_profile = {
        "use_gpu": False,
        "name": "cpu",
        "codec": cpu_video_codec,
        "command_args": [
            "-c:v",
            cpu_video_codec,
            "-preset",
            cpu_video_preset,
            "-crf",
            str(cpu_video_crf),
        ],
        "fallback_cpu_profile": None,
    }
    normalized_gpu_rc_mode = _normalize_nvenc_rc_mode_for_preset(
        gpu_rc_mode=gpu_rc_mode,
        gpu_preset=gpu_preset,
    )
    gpu_profile = {
        "use_gpu": True,
        "name": "gpu",
        "codec": normalized_gpu_codec,
        "command_args": [
            "-c:v",
            normalized_gpu_codec,
            "-preset",
            str(gpu_preset),
            "-rc",
            str(normalized_gpu_rc_mode),
            "-cq",
            str(_clamp_nvenc_cq(gpu_cq=gpu_cq, fallback_crf=cpu_video_crf)),
        ],
        "fallback_cpu_profile": cpu_profile,
    }
    if gpu_bitrate:
        gpu_profile["command_args"].extend(["-b:v", str(gpu_bitrate)])
    else:
        gpu_profile["command_args"].extend(["-b:v", "0"])

    if normalized_mode == "cpu_only":
        return cpu_profile
    if normalized_mode == "gpu_only":
        if not gpu_available:
            raise RuntimeError(f"模块D-GPU编码不可用：未检测到编码器 {normalized_gpu_codec}")
        return {**gpu_profile, "fallback_cpu_profile": None}

    if gpu_available:
        return gpu_profile
    return cpu_profile


def _build_single_segment_command(
    ffmpeg_bin: str,
    frame_path: str,
    exact_frames: int,
    fps: int,
    encoder_command_args: list[str],
    output_path: str,
) -> list[str]:
    """
    功能说明：构建单段渲染 ffmpeg 命令（单输入单输出）。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - frame_path: 输入帧路径。
    - exact_frames: 输出帧数。
    - fps: 输出帧率。
    - encoder_command_args: 编码参数数组。
    - output_path: 输出视频路径。
    返回值：
    - list[str]: 可执行的 ffmpeg 命令数组。
    异常说明：无。
    边界条件：固定禁用音频流并输出 yuv420p。
    """
    return [
        ffmpeg_bin,
        "-y",
        "-loop",
        "1",
        "-i",
        str(frame_path),
        "-frames:v",
        str(exact_frames),
        "-r",
        str(fps),
        *list(encoder_command_args),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]


def _render_single_segment_worker(
    ffmpeg_bin: str,
    frame_path: str,
    exact_frames: int,
    fps: int,
    encoder_command_args: list[str],
    segment_index: int,
    temp_output_path: str,
    final_output_path: str,
    profile_name: str,
) -> dict[str, Any]:
    """
    功能说明：执行单段渲染并以原子替换方式提交最终产物。
    参数说明：
    - ffmpeg_bin/frame_path/exact_frames/fps/encoder_command_args: 渲染命令参数。
    - segment_index: 片段序号（用于日志上下文）。
    - temp_output_path: 临时输出路径。
    - final_output_path: 最终输出路径。
    - profile_name: 渲染 profile 名称（gpu/cpu）。
    返回值：
    - dict[str, Any]: 渲染结果摘要（segment_index/segment_path/elapsed/profile_name）。
    异常说明：渲染失败或原子替换失败时抛 RuntimeError。
    边界条件：失败时会清理临时文件，避免残留半成品。
    """
    stage_start = time.perf_counter()
    temp_path = Path(temp_output_path)
    final_path = Path(final_output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass

    command = _build_single_segment_command(
        ffmpeg_bin=ffmpeg_bin,
        frame_path=frame_path,
        exact_frames=exact_frames,
        fps=fps,
        encoder_command_args=encoder_command_args,
        output_path=str(temp_path),
    )
    try:
        _run_ffmpeg_command(
            command=command,
            command_name=f"渲染小片段 segment_{segment_index:03d}（{profile_name}）",
        )
        temp_path.replace(final_path)
    except Exception:  # noqa: BLE001
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise
    elapsed = time.perf_counter() - stage_start
    return {
        "segment_index": int(segment_index),
        "segment_path": str(final_path),
        "elapsed": float(elapsed),
        "profile_name": str(profile_name),
    }


def run_module_d(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 D，输出最终成片视频。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 最终视频路径。
    异常说明：FFmpeg/FFprobe 调用失败时抛 RuntimeError。
    边界条件：当关键帧清单为空时直接抛错，避免生成空视频。
    """
    context.logger.info("模块D开始执行，task_id=%s", context.task_id)
    stage_total_start = time.perf_counter()

    module_c_path = context.artifacts_dir / "module_c_output.json"
    module_c_output = read_json(module_c_path)
    frame_items = module_c_output.get("frame_items", [])
    if not frame_items:
        raise RuntimeError("模块D无法执行：模块C输出的 frame_items 为空。")

    segments_dir = context.artifacts_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    audio_duration = _probe_media_duration(
        media_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
    )

    stage_render_start = time.perf_counter()
    segment_paths = _render_segment_videos(
        frame_items=frame_items,
        segments_dir=segments_dir,
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        fps=context.config.ffmpeg.fps,
        video_codec=context.config.ffmpeg.video_codec,
        video_preset=context.config.ffmpeg.video_preset,
        video_crf=context.config.ffmpeg.video_crf,
        render_batch_size=context.config.ffmpeg.render_batch_size,
        render_workers=context.config.ffmpeg.render_workers,
        video_accel_mode=context.config.ffmpeg.video_accel_mode,
        gpu_video_codec=context.config.ffmpeg.gpu_video_codec,
        gpu_preset=context.config.ffmpeg.gpu_preset,
        gpu_rc_mode=context.config.ffmpeg.gpu_rc_mode,
        gpu_cq=context.config.ffmpeg.gpu_cq,
        gpu_bitrate=context.config.ffmpeg.gpu_bitrate,
        audio_duration=audio_duration,
        logger=context.logger,
    )
    render_elapsed = time.perf_counter() - stage_render_start

    output_video_path = context.task_dir / "final_output.mp4"
    stage_concat_start = time.perf_counter()
    concat_result = _concat_segment_videos(
        segment_paths=segment_paths,
        concat_file_path=context.artifacts_dir / "segments_concat.txt",
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        audio_path=context.audio_path,
        output_video_path=output_video_path,
        audio_duration=audio_duration,
        fps=context.config.ffmpeg.fps,
        video_codec=context.config.ffmpeg.video_codec,
        audio_codec=context.config.ffmpeg.audio_codec,
        video_preset=context.config.ffmpeg.video_preset,
        video_crf=context.config.ffmpeg.video_crf,
        video_accel_mode=context.config.ffmpeg.video_accel_mode,
        gpu_video_codec=context.config.ffmpeg.gpu_video_codec,
        gpu_preset=context.config.ffmpeg.gpu_preset,
        gpu_rc_mode=context.config.ffmpeg.gpu_rc_mode,
        gpu_cq=context.config.ffmpeg.gpu_cq,
        gpu_bitrate=context.config.ffmpeg.gpu_bitrate,
        concat_video_mode=context.config.ffmpeg.concat_video_mode,
        concat_copy_fallback_reencode=context.config.ffmpeg.concat_copy_fallback_reencode,
        logger=context.logger,
    )
    concat_elapsed = time.perf_counter() - stage_concat_start
    total_elapsed = time.perf_counter() - stage_total_start

    context.logger.info(
        "模块D耗时统计，render_segments_elapsed=%.3fs，concat_elapsed=%.3fs，total_elapsed=%.3fs，concat_mode=%s，copy_fallback=%s",
        render_elapsed,
        concat_elapsed,
        total_elapsed,
        concat_result.get("mode", "unknown"),
        bool(concat_result.get("copy_fallback_triggered", False)),
    )

    context.logger.info("模块D执行完成，task_id=%s，输出=%s", context.task_id, output_video_path)
    return output_video_path


def _render_segment_videos(
    frame_items: list[dict[str, Any]],
    segments_dir: Path,
    ffmpeg_bin: str,
    fps: int,
    video_codec: str,
    video_preset: str,
    video_crf: int,
    render_batch_size: int,
    render_workers: int,
    audio_duration: float,
    logger: logging.Logger,
    *,
    video_accel_mode: str = "auto",
    gpu_video_codec: str = "h264_nvenc",
    gpu_preset: str = "p1",
    gpu_rc_mode: str = "vbr",
    gpu_cq: int | None = 34,
    gpu_bitrate: str | None = None,
) -> list[Path]:
    """
    功能说明：将每个静态帧渲染成独立的小视频片段（受控并行单段渲染）。
    参数说明：
    - frame_items: 帧清单数组，需含 frame_path 与 duration。
    - segments_dir: 小视频片段输出目录。
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - fps: 输出帧率。
    - video_codec: 视频编码器。
    - video_preset: 视频编码预设。
    - video_crf: 视频 CRF 参数。
    - render_batch_size: 兼容旧配置字段（当前固定按 1 执行）。
    - render_workers: 并行渲染 worker 数量（建议 1~4）。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - gpu_video_codec: GPU 视频编码器（如 h264_nvenc）。
    - gpu_preset: GPU 编码预设（如 p1~p7）。
    - gpu_rc_mode: GPU 码率控制模式（如 vbr_hq）。
    - gpu_cq: GPU CQ 参数（可选）。
    - gpu_bitrate: GPU 目标码率（可选）。
    - audio_duration: 原音轨时长（秒）。
    - logger: 日志对象。
    返回值：
    - list[Path]: 已生成的小视频片段路径数组。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：使用全局帧分配，保证各片段总帧数与音频目标帧数一致；输出顺序固定按 segment_index。
    """
    allocated_frames = _allocate_segment_frames_by_timeline(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=fps,
    )
    target_total_frames = max(1, round(audio_duration * fps))
    allocated_total_frames = sum(allocated_frames)
    logger.info(
        "模块D帧分配汇总，目标总帧=%s，分配总帧=%s，片段数=%s",
        target_total_frames,
        allocated_total_frames,
        len(allocated_frames),
    )
    normalized_batch_size = _normalize_render_batch_size(render_batch_size=render_batch_size, logger=logger)
    normalized_workers = _normalize_render_workers(render_workers=render_workers, logger=logger)

    profile = _resolve_video_encoder_profile(
        ffmpeg_bin=ffmpeg_bin,
        video_accel_mode=video_accel_mode,
        cpu_video_codec=video_codec,
        cpu_video_preset=video_preset,
        cpu_video_crf=video_crf,
        gpu_video_codec=gpu_video_codec,
        gpu_preset=gpu_preset,
        gpu_rc_mode=gpu_rc_mode,
        gpu_cq=gpu_cq,
        gpu_bitrate=gpu_bitrate,
    )
    logger.info(
        "模块D渲染编码策略，mode=%s，encoder=%s，render_workers=%s",
        _normalize_video_accel_mode(video_accel_mode=video_accel_mode),
        str(profile["codec"]),
        normalized_workers,
    )
    render_jobs = [
        {
            "segment_index": index,
            "frame_path": str(item["frame_path"]),
            "exact_frames": allocated_frames[index - 1],
            "segment_path": segments_dir / f"segment_{index:03d}.mp4",
            "segment_temp_path": segments_dir / f"segment_{index:03d}.tmp.mp4",
        }
        for index, item in enumerate(frame_items, start=1)
    ]
    logger.info(
        "模块D受控并行渲染计划，segment_count=%s，render_workers=%s，effective_batch_size=%s",
        len(render_jobs),
        normalized_workers,
        normalized_batch_size,
    )
    detail_lines = _build_frame_allocation_detail_lines(frame_items=frame_items, allocated_frames=allocated_frames, fps=fps)
    detail_text = "\n".join(detail_lines)
    segment_paths_by_index: dict[int, Path] = {}
    gpu_attempts = len(render_jobs) if bool(profile.get("use_gpu", False)) else 0
    gpu_to_cpu_fallback_count = 0
    hard_fail_messages: list[str] = []

    def _execute_job(job: dict[str, Any], active_profile: dict[str, Any]) -> dict[str, Any]:
        return _render_single_segment_worker(
            ffmpeg_bin=ffmpeg_bin,
            frame_path=str(job["frame_path"]),
            exact_frames=int(job["exact_frames"]),
            fps=fps,
            encoder_command_args=list(active_profile["command_args"]),
            segment_index=int(job["segment_index"]),
            temp_output_path=str(job["segment_temp_path"]),
            final_output_path=str(job["segment_path"]),
            profile_name=str(active_profile.get("name", "unknown")),
        )

    def _record_success(result: dict[str, Any]) -> None:
        segment_index = int(result["segment_index"])
        segment_path = Path(str(result["segment_path"]))
        elapsed = float(result["elapsed"])
        profile_name = str(result.get("profile_name", "unknown"))
        segment_paths_by_index[segment_index] = segment_path
        logger.info(
            "模块D单段渲染完成，segment=segment_%03d，profile=%s，elapsed=%.3fs",
            segment_index,
            profile_name,
            elapsed,
        )

    def _render_with_current_process(
        jobs: list[dict[str, Any]],
        active_profile: dict[str, Any],
    ) -> list[tuple[dict[str, Any], Exception]]:
        failed_jobs: list[tuple[dict[str, Any], Exception]] = []
        for job in jobs:
            try:
                result = _execute_job(job=job, active_profile=active_profile)
                _record_success(result=result)
            except Exception as error:  # noqa: BLE001
                failed_jobs.append((job, error))
        return failed_jobs

    def _render_with_process_pool(
        jobs: list[dict[str, Any]],
        active_profile: dict[str, Any],
    ) -> list[tuple[dict[str, Any], Exception]]:
        failed_jobs: list[tuple[dict[str, Any], Exception]] = []
        with ProcessPoolExecutor(
            max_workers=normalized_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            future_to_job = {
                executor.submit(
                    _render_single_segment_worker,
                    ffmpeg_bin,
                    str(job["frame_path"]),
                    int(job["exact_frames"]),
                    fps,
                    list(active_profile["command_args"]),
                    int(job["segment_index"]),
                    str(job["segment_temp_path"]),
                    str(job["segment_path"]),
                    str(active_profile.get("name", "unknown")),
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    _record_success(result=result)
                except Exception as error:  # noqa: BLE001
                    failed_jobs.append((job, error))
        return failed_jobs

    if normalized_workers == 1:
        primary_failed_jobs = _render_with_current_process(jobs=render_jobs, active_profile=profile)
    else:
        try:
            primary_failed_jobs = _render_with_process_pool(jobs=render_jobs, active_profile=profile)
        except Exception as error:  # noqa: BLE001
            logger.warning("模块D并行调度异常，已回退串行执行，错误=%s", error)
            remaining_jobs = [
                job for job in render_jobs if int(job["segment_index"]) not in segment_paths_by_index
            ]
            primary_failed_jobs = _render_with_current_process(jobs=remaining_jobs, active_profile=profile)

    fallback_profile = profile.get("fallback_cpu_profile")
    if primary_failed_jobs and fallback_profile is not None:
        for failed_job, failed_error in primary_failed_jobs:
            segment_index = int(failed_job["segment_index"])
            logger.warning(
                "模块D单段GPU渲染失败，已回退CPU重试，segment=segment_%03d，错误=%s",
                segment_index,
                failed_error,
            )
            cpu_retry_failed = _render_with_current_process(jobs=[failed_job], active_profile=fallback_profile)
            if cpu_retry_failed:
                final_error = cpu_retry_failed[0][1]
                hard_fail_messages.append(f"segment_{segment_index:03d}: {final_error}")
            else:
                gpu_to_cpu_fallback_count += 1
    else:
        for failed_job, failed_error in primary_failed_jobs:
            segment_index = int(failed_job["segment_index"])
            hard_fail_messages.append(f"segment_{segment_index:03d}: {failed_error}")

    hard_fail_count = len(hard_fail_messages)
    logger.info(
        "模块D受控并行渲染汇总，segment_count=%s，render_workers=%s，gpu_attempts=%s，gpu_to_cpu_fallback_count=%s，hard_fail_count=%s",
        len(render_jobs),
        normalized_workers,
        gpu_attempts,
        gpu_to_cpu_fallback_count,
        hard_fail_count,
    )

    if hard_fail_count > 0:
        error_text = "\n".join(hard_fail_messages)
        raise RuntimeError(f"模块D小片段渲染失败，共{hard_fail_count}段失败。\n{error_text}\n模块D逐段帧分配明细：\n{detail_text}")

    ordered_segment_paths: list[Path] = []
    for segment_index in range(1, len(render_jobs) + 1):
        segment_path = segment_paths_by_index.get(segment_index)
        if segment_path is None:
            raise RuntimeError(f"模块D渲染失败：segment_{segment_index:03d} 缺失输出文件。")
        ordered_segment_paths.append(segment_path)
    return ordered_segment_paths


def _allocate_segment_frames_by_timeline(
    frame_items: list[dict[str, Any]],
    audio_duration: float,
    fps: int,
) -> list[int]:
    """
    功能说明：根据全局时间轴为每个片段分配绝对帧数，消除累积舍入误差。
    参数说明：
    - frame_items: 模块 C 产出的帧清单，需包含 start_time/end_time。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    返回值：
    - list[int]: 与 frame_items 一一对应的片段帧数。
    异常说明：输入非法或无法满足最小帧分配时抛 RuntimeError。
    边界条件：每个片段至少分配 1 帧，总帧数严格等于 round(audio_duration * fps)。
    """
    if not frame_items:
        raise RuntimeError("模块D帧分配失败：frame_items 为空。")
    if fps <= 0:
        raise RuntimeError(f"模块D帧分配失败：fps 非法，fps={fps}")

    safe_audio_duration = max(0.1, float(audio_duration))
    target_total_frames = max(1, round(safe_audio_duration * fps))
    segment_count = len(frame_items)
    if segment_count > target_total_frames:
        raise RuntimeError(
            f"模块D帧分配失败：片段数大于目标总帧数，segment_count={segment_count}, target_total_frames={target_total_frames}"
        )

    normalized_end_frames: list[int] = []
    last_start_time = 0.0
    for index, item in enumerate(frame_items, start=1):
        start_time = float(item.get("start_time", 0.0))
        if "end_time" in item:
            end_time = float(item["end_time"])
        else:
            end_time = start_time + max(0.1, float(item.get("duration", 0.1)))

        if end_time < start_time:
            raise RuntimeError(f"模块D帧分配失败：片段时间区间非法，index={index}, start={start_time}, end={end_time}")
        if start_time < last_start_time:
            raise RuntimeError(
                f"模块D帧分配失败：片段开始时间未按升序，index={index}, previous_start={last_start_time}, start={start_time}"
            )
        last_start_time = start_time

        clamped_end = max(0.0, min(safe_audio_duration, end_time))
        normalized_end_frames.append(round(clamped_end * fps))

    allocated_frames: list[int] = []
    previous_end_frame = 0
    for index, raw_end_frame in enumerate(normalized_end_frames, start=1):
        remaining_segments = segment_count - index
        min_end_frame = previous_end_frame + 1
        max_end_frame = target_total_frames - remaining_segments
        clamped_end_frame = min(max(raw_end_frame, min_end_frame), max_end_frame)
        current_frames = clamped_end_frame - previous_end_frame
        allocated_frames.append(current_frames)
        previous_end_frame = clamped_end_frame

    if sum(allocated_frames) != target_total_frames:
        raise RuntimeError(
            f"模块D帧分配失败：总帧数不一致，allocated={sum(allocated_frames)}, target={target_total_frames}"
        )
    if any(frame_count <= 0 for frame_count in allocated_frames):
        raise RuntimeError("模块D帧分配失败：存在非正帧片段。")
    return allocated_frames


def _build_frame_allocation_detail_lines(frame_items: list[dict[str, Any]], allocated_frames: list[int], fps: int) -> list[str]:
    """
    功能说明：构建逐段帧分配明细文本，用于失败排障。
    参数说明：
    - frame_items: 帧清单数组。
    - allocated_frames: 已分配帧数组。
    - fps: 输出帧率。
    返回值：
    - list[str]: 可直接拼接输出的明细行。
    异常说明：无。
    边界条件：若字段缺失则使用默认值，不中断明细输出。
    """
    lines: list[str] = []
    cumulative_frames = 0
    for index, (item, frame_count) in enumerate(zip(frame_items, allocated_frames, strict=True), start=1):
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        cumulative_frames += frame_count
        lines.append(
            (
                f"- segment_{index:03d}: start={start_time:.3f}s, end={end_time:.3f}s, "
                f"frames={frame_count}, cumulative_frames={cumulative_frames}, fps={fps}"
            )
        )
    return lines


def _concat_segment_videos(
    segment_paths: list[Path],
    concat_file_path: Path,
    ffmpeg_bin: str,
    audio_path: Path,
    output_video_path: Path,
    audio_duration: float,
    fps: int,
    video_codec: str,
    audio_codec: str,
    video_preset: str,
    video_crf: int,
    *,
    video_accel_mode: str = "auto",
    gpu_video_codec: str = "h264_nvenc",
    gpu_preset: str = "p1",
    gpu_rc_mode: str = "vbr",
    gpu_cq: int | None = 34,
    gpu_bitrate: str | None = None,
    concat_video_mode: str = "copy",
    concat_copy_fallback_reencode: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    功能说明：拼接小视频片段并混入原音轨，生成最终成片。
    参数说明：
    - segment_paths: 小视频片段路径数组。
    - concat_file_path: concat 清单文件路径。
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - audio_path: 原音轨路径。
    - output_video_path: 输出视频路径。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    - video_codec: 视频编码器。
    - audio_codec: 音频编码器。
    - video_preset: 视频编码预设。
    - video_crf: 视频 CRF 参数。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - gpu_video_codec: GPU 视频编码器。
    - gpu_preset: GPU 编码预设。
    - gpu_rc_mode: GPU 码率控制模式。
    - gpu_cq: GPU CQ 参数（可选）。
    - gpu_bitrate: GPU 目标码率（可选）。
    - concat_video_mode: 最终拼接模式（copy/reencode）。
    - concat_copy_fallback_reencode: copy 失败时是否回退重编码。
    - logger: 日志对象（可选）。
    返回值：
    - dict[str, Any]: 拼接阶段执行信息（mode/copy_fallback_triggered）。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：显式使用 -t 音频时长，避免最终视频长于音频。
    """
    active_logger = logger or logging.getLogger("music_video_pipeline")
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{_escape_concat_path(str(path))}'" for path in segment_paths]
    concat_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    normalized_concat_mode = _normalize_concat_video_mode(concat_video_mode=concat_video_mode)
    copy_command = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-t",
        f"{audio_duration:.3f}",
        str(output_video_path),
    ]

    profile = _resolve_video_encoder_profile(
        ffmpeg_bin=ffmpeg_bin,
        video_accel_mode=video_accel_mode,
        cpu_video_codec=video_codec,
        cpu_video_preset=video_preset,
        cpu_video_crf=video_crf,
        gpu_video_codec=gpu_video_codec,
        gpu_preset=gpu_preset,
        gpu_rc_mode=gpu_rc_mode,
        gpu_cq=gpu_cq,
        gpu_bitrate=gpu_bitrate,
    )
    reencode_command = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file_path),
        "-i",
        str(audio_path),
        *list(profile["command_args"]),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        audio_codec,
        "-t",
        f"{audio_duration:.3f}",
        str(output_video_path),
    ]

    if normalized_concat_mode == "reencode":
        _run_ffmpeg_command(command=reencode_command, command_name="拼接小片段并混音（reencode）")
        return {"mode": "reencode", "copy_fallback_triggered": False}

    try:
        _run_ffmpeg_command(command=copy_command, command_name="拼接小片段并混音（copy）")
        return {"mode": "copy", "copy_fallback_triggered": False}
    except RuntimeError as copy_error:
        if not bool(concat_copy_fallback_reencode):
            raise
        active_logger.warning("模块D-concat copy 失败，已回退 reencode，错误=%s", copy_error)
        _run_ffmpeg_command(command=reencode_command, command_name="拼接小片段并混音（copy回退reencode）")
        return {"mode": "copy_with_reencode_fallback", "copy_fallback_triggered": True}


def _probe_media_duration(media_path: Path, ffprobe_bin: str) -> float:
    """
    功能说明：使用 ffprobe 获取媒体时长（秒）。
    参数说明：
    - media_path: 媒体文件路径。
    - ffprobe_bin: ffprobe 可执行文件名或路径。
    返回值：
    - float: 媒体时长秒数。
    异常说明：ffprobe 执行失败或解析失败时抛 RuntimeError。
    边界条件：返回值最小为 0.1 秒，避免无效时长。
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"找不到 ffprobe 可执行文件，请检查 ffprobe_bin={ffprobe_bin}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffprobe 执行失败，stderr={error.stderr}") from error

    output_text = result.stdout.strip()
    try:
        duration = float(output_text)
    except ValueError as error:
        raise RuntimeError(f"ffprobe 时长解析失败，输出内容={output_text!r}") from error
    return max(0.1, duration)


def _run_ffmpeg_command(command: list[str], command_name: str) -> None:
    """
    功能说明：统一执行 ffmpeg 命令并抛出带上下文的错误信息。
    参数说明：
    - command: ffmpeg 命令参数数组。
    - command_name: 命令用途说明。
    返回值：无。
    异常说明：命令执行失败时抛 RuntimeError。
    边界条件：stderr 按 utf-8 replace 解码，避免 Windows 编码报错中断。
    """
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{command_name}失败：找不到 ffmpeg 可执行文件。") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{command_name}失败：ffmpeg 返回非零状态。\n命令：{' '.join(command)}\nstderr: {error.stderr}"
        ) from error


def _escape_concat_path(path_text: str) -> str:
    """
    功能说明：转义 concat 文件中的路径文本。
    参数说明：
    - path_text: 原始路径字符串。
    返回值：
    - str: 适合写入 concat 文件的路径文本。
    异常说明：无。
    边界条件：仅处理单引号转义，其他字符按原样保留。
    """
    return path_text.replace("'", "'\\''")
