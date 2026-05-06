"""
文件用途：提供模块 D 的拼接与 FFmpeg/FFprobe 通用执行能力。
核心流程：按两阶段终拼策略执行 concat，并在需要时从 copy 回退 reencode。
输入输出：输入片段路径与编码配置，输出最终拼接执行结果。
依赖说明：依赖标准库 subprocess/logging/pathlib/functools。
维护说明：本文件只负责终拼与工具函数，不承担单元调度职责。
"""

# 标准库：用于函数结果缓存
from functools import lru_cache
# 标准库：用于日志类型
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于子进程执行
import subprocess
# 标准库：用于类型提示
from typing import Any


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
    return {
        "h264_nvenc": "h264_nvenc" in output_text,
        "hevc_nvenc": "hevc_nvenc" in output_text,
    }


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
    - dict[str, Any]: 编码方案（含 use_gpu/command_args/fallback_cpu_profile）。
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


def _concat_segment_videos(
    segment_paths: list[Path],
    concat_file_path: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
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
    transition_plans: list[dict[str, Any]] | None = None,
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
    - gpu_video_codec/gpu_preset/gpu_rc_mode/gpu_cq/gpu_bitrate: GPU 编码配置。
    - concat_video_mode: 最终拼接模式（copy/reencode）。
    - concat_copy_fallback_reencode: copy 失败时是否回退重编码。
    - logger: 日志对象（可选）。
    返回值：
    - dict[str, Any]: 拼接阶段执行信息（mode/copy_fallback_triggered）。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：显式使用 -t 音频时长，避免最终视频长于音频。
    """
    active_logger = logger or logging.getLogger("D")
    normalized_transition_plans = transition_plans or []
    if _has_nontrivial_transitions(normalized_transition_plans):
        return _concat_segment_videos_with_transitions(
            segment_paths=segment_paths,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            audio_path=audio_path,
            output_video_path=output_video_path,
            audio_duration=audio_duration,
            fps=fps,
            video_codec=video_codec,
            audio_codec=audio_codec,
            video_preset=video_preset,
            video_crf=video_crf,
            video_accel_mode=video_accel_mode,
            gpu_video_codec=gpu_video_codec,
            gpu_preset=gpu_preset,
            gpu_rc_mode=gpu_rc_mode,
            gpu_cq=gpu_cq,
            gpu_bitrate=gpu_bitrate,
            transition_plans=normalized_transition_plans,
            logger=active_logger,
        )
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{_escape_concat_path(str(path))}'" for path in segment_paths]
    concat_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    normalized_concat_mode = _normalize_concat_video_mode(concat_video_mode=concat_video_mode)
    copy_command = [
        ffmpeg_bin,
        "-nostdin",
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
        "-nostdin",
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


def apply_camera_plan_to_segment(
    *,
    segment_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    fps: int,
    video_codec: str,
    video_preset: str,
    video_crf: int,
    camera_plan: dict[str, Any],
) -> bool:
    """
    功能说明：对单个片段应用 FFmpeg 运镜后处理。
    参数说明：
    - segment_path: 原始片段路径。
    - output_path: 输出片段路径。
    - ffmpeg_bin/ffprobe_bin: FFmpeg/FFprobe 可执行文件。
    - fps: 目标帧率。
    - video_codec/video_preset/video_crf: 编码参数。
    - camera_plan: 模块B透传的运镜计划。
    返回值：
    - bool: true 表示执行了后处理，false 表示按 none 跳过。
    异常说明：ffprobe/ffmpeg 失败时抛 RuntimeError。
    边界条件：mode=none 时跳过处理。
    """
    normalized_mode = str(camera_plan.get("mode", "none")).strip().lower()
    if normalized_mode == "none":
        return False
    width, height = _probe_video_resolution(media_path=segment_path, ffprobe_bin=ffprobe_bin)
    duration = _probe_media_duration(media_path=segment_path, ffprobe_bin=ffprobe_bin)
    filter_text = _build_camera_filter(
            width=width,
            height=height,
            duration=duration,
            camera_plan=camera_plan,
        )
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-i",
        str(segment_path),
        "-vf",
        filter_text,
        "-r",
        str(int(fps)),
        "-c:v",
        str(video_codec),
        "-preset",
        str(video_preset),
        "-crf",
        str(int(video_crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    _run_ffmpeg_command(command=command, command_name="模块D单段运镜后处理")
    return True


def _concat_segment_videos_with_transitions(
    *,
    segment_paths: list[Path],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    audio_path: Path,
    output_video_path: Path,
    audio_duration: float,
    fps: int,
    video_codec: str,
    audio_codec: str,
    video_preset: str,
    video_crf: int,
    video_accel_mode: str,
    gpu_video_codec: str,
    gpu_preset: str,
    gpu_rc_mode: str,
    gpu_cq: int | None,
    gpu_bitrate: str | None,
    transition_plans: list[dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    功能说明：使用 filter_complex + xfade 路径执行带转场的终拼。
    参数说明：同上层 concat 函数。
    返回值：
    - dict[str, Any]: 拼接阶段执行信息。
    异常说明：ffmpeg 失败时抛 RuntimeError。
    边界条件：通过对每段尾部做 clone padding，保证转场存在时仍保持总视觉时长不被压短。
    """
    if not segment_paths:
        raise RuntimeError("模块D终拼失败：segment_paths 为空。")
    input_args: list[str] = []
    durations = [_probe_media_duration(media_path=path, ffprobe_bin=ffprobe_bin) for path in segment_paths]
    for segment_path in segment_paths:
        input_args.extend(["-i", str(segment_path)])
    input_args.extend(["-i", str(audio_path)])

    transition_specs = [
        _resolve_xfade_transition(
            transition_plan=transition_plans[index] if index < len(transition_plans) else {},
        )
        for index in range(len(segment_paths) - 1)
    ]
    outgoing_pad_durations = [spec[1] for spec in transition_specs] + [0.0]

    filter_parts: list[str] = []
    for index in range(len(segment_paths)):
        base_filter = f"[{index}:v]fps={int(fps)},format=yuv420p,setsar=1"
        pad_duration = float(outgoing_pad_durations[index])
        if pad_duration > 0.0:
            base_filter += f",tpad=stop_mode=clone:stop_duration={pad_duration:.3f}"
        filter_parts.append(f"{base_filter}[v{index}]")
    current_label = "v0"
    current_duration = float(durations[0]) + float(outgoing_pad_durations[0])
    current_tail_pad = float(outgoing_pad_durations[0])
    for index in range(1, len(segment_paths)):
        transition_name, transition_duration = transition_specs[index - 1]
        offset = max(0.0, current_duration - current_tail_pad)
        next_label = f"x{index}"
        filter_parts.append(
            f"[{current_label}][v{index}]xfade=transition={transition_name}:duration={transition_duration:.3f}:offset={offset:.3f}[{next_label}]"
        )
        current_label = next_label
        next_tail_pad = float(outgoing_pad_durations[index])
        current_duration = current_duration + float(durations[index]) + next_tail_pad - current_tail_pad
        current_tail_pad = next_tail_pad

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
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        *input_args,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        f"[{current_label}]",
        "-map",
        f"{len(segment_paths)}:a:0",
        *list(profile["command_args"]),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(fps)),
        "-c:a",
        audio_codec,
        "-t",
        f"{audio_duration:.3f}",
        str(output_video_path),
    ]
    logger.info("模块D检测到非 none 转场，切换到 filter_complex + xfade 终拼路径（已启用尾帧 padding 保持总时长）。")
    _run_ffmpeg_command(command=command, command_name="拼接小片段并混音（xfade）")
    return {"mode": "xfade_reencode", "copy_fallback_triggered": False}


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


def _probe_video_resolution(media_path: Path, ffprobe_bin: str) -> tuple[int, int]:
    """
    功能说明：使用 ffprobe 获取视频主流宽高。
    参数说明：
    - media_path: 视频路径。
    - ffprobe_bin: ffprobe 可执行文件。
    返回值：
    - tuple[int, int]: 宽度与高度。
    异常说明：ffprobe 失败或解析失败时抛 RuntimeError。
    边界条件：最小回退为 2x2，避免无效尺寸。
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
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
        raise RuntimeError(f"ffprobe 分辨率探测失败，stderr={error.stderr}") from error
    output_text = result.stdout.strip()
    try:
        width_text, height_text = output_text.split("x", maxsplit=1)
        width = max(2, int(width_text))
        height = max(2, int(height_text))
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"ffprobe 分辨率解析失败，输出内容={output_text!r}") from error
    return width, height


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


def _has_nontrivial_transitions(transition_plans: list[dict[str, Any]]) -> bool:
    """
    功能说明：判断转场列表中是否存在需要切换到 xfade 路径的条目。
    参数说明：
    - transition_plans: 转场计划数组。
    返回值：
    - bool: 是否存在非 none 转场。
    异常说明：无。
    边界条件：空数组返回 false。
    """
    for item in transition_plans:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind", "none")).strip().lower() != "none":
            return True
    return False


def _resolve_xfade_transition(transition_plan: dict[str, Any]) -> tuple[str, float]:
    """
    功能说明：将 transition_plan 转换为 xfade 参数。
    参数说明：
    - transition_plan: 转场计划对象。
    返回值：
    - tuple[str, float]: xfade transition 名与持续秒数。
    异常说明：无。
    边界条件：none/hard_cut 在混合链路中会被近似为极短 fade。
    """
    kind = str(transition_plan.get("kind", "none")).strip().lower()
    try:
        duration_ms = int(transition_plan.get("duration_ms", 0))
    except (TypeError, ValueError):
        duration_ms = 0
    duration_seconds = max(0.001, float(duration_ms) / 1000.0)
    mapping = {
        "none": "fade",
        "hard_cut": "fade",
        "crossfade": "fade",
        "fade_black": "fadeblack",
        "fade_white": "fadewhite",
        "wipe_left": "wipeleft",
        "wipe_right": "wiperight",
    }
    if kind in {"none", "hard_cut"}:
        duration_seconds = 0.001
    return mapping.get(kind, "fade"), duration_seconds


def _build_camera_filter(width: int, height: int, duration: float, camera_plan: dict[str, Any]) -> str:
    """
    功能说明：按 camera_plan 构建 FFmpeg 视频滤镜表达式。
    参数说明：
    - width/height: 原始视频尺寸。
    - duration: 片段时长。
    - camera_plan: 运镜计划对象。
    返回值：
    - str: 可直接传给 -vf 的滤镜表达式。
    异常说明：无。
    边界条件：未知模式回退为空运镜。
    """
    mode = str(camera_plan.get("mode", "none")).strip().lower()
    easing = str(camera_plan.get("easing", "linear")).strip().lower()
    strength = str(camera_plan.get("strength", "none")).strip().lower()
    progress = _build_progress_expression(duration=max(0.1, float(duration)), easing=easing)
    if mode == "zoom":
        zoom_delta = 0.14 if strength == "small" else 0.24
        preset_id = str(camera_plan.get("preset_id", "")).strip().lower()
        if preset_id.startswith("zoom_out"):
            start_scale = 1.0 + zoom_delta
            end_scale = 1.0
        else:
            start_scale = 1.0
            end_scale = 1.0 + zoom_delta
        scale_expr = f"{start_scale:.4f}+({end_scale - start_scale:.4f})*({progress})"
        return (
            f"scale=w='ceil(iw*({scale_expr})/2)*2':"
            f"h='ceil(ih*({scale_expr})/2)*2':"
            f"eval=frame:flags=lanczos,"
            f"crop={int(width)}:{int(height)}:'(iw-ow)/2':'(ih-oh)/2'"
        )
    if mode == "pan":
        overscan = 1.18 if strength == "small" else 1.30
        scaled_width = max(int(width), int(round(float(width) * overscan)))
        scaled_height = max(int(height), int(round(float(height) * overscan)))
        max_x = max(0, scaled_width - int(width))
        max_y = max(0, scaled_height - int(height))
        x_start, x_end, y_start, y_end = _resolve_pan_endpoints(
            direction=str(camera_plan.get("direction", "center")).strip().lower(),
            max_x=max_x,
            max_y=max_y,
        )
        x_expr = f"{x_start:.3f}+({x_end - x_start:.3f})*({progress})"
        y_expr = f"{y_start:.3f}+({y_end - y_start:.3f})*({progress})"
        return (
            f"scale={scaled_width}:{scaled_height}:flags=lanczos,"
            f"crop={int(width)}:{int(height)}:'{x_expr}':'{y_expr}'"
        )
    return "null"


def _resolve_pan_endpoints(direction: str, max_x: int, max_y: int) -> tuple[float, float, float, float]:
    """
    功能说明：根据方向枚举解析 pan 起止坐标。
    参数说明：
    - direction: center/left/right/up/down/diagonal。
    - max_x/max_y: 可移动范围。
    返回值：
    - tuple[float, float, float, float]: x_start, x_end, y_start, y_end。
    异常说明：无。
    边界条件：未知方向回退到静止中心。
    """
    center_x = float(max_x) / 2.0
    center_y = float(max_y) / 2.0
    mapping = {
        "center": (center_x, center_x, center_y, center_y),
        "left": (float(max_x), 0.0, center_y, center_y),
        "right": (0.0, float(max_x), center_y, center_y),
        "up": (center_x, center_x, float(max_y), 0.0),
        "down": (center_x, center_x, 0.0, float(max_y)),
        "up_left": (float(max_x), 0.0, float(max_y), 0.0),
        "up_right": (0.0, float(max_x), float(max_y), 0.0),
        "down_left": (float(max_x), 0.0, 0.0, float(max_y)),
        "down_right": (0.0, float(max_x), 0.0, float(max_y)),
    }
    return mapping.get(direction, mapping["center"])


def _build_progress_expression(duration: float, easing: str) -> str:
    """
    功能说明：构建 FFmpeg 可执行的 0~1 进度表达式。
    参数说明：
    - duration: 片段时长。
    - easing: 缓动名。
    返回值：
    - str: 表达式文本。
    异常说明：无。
    边界条件：非法 easing 回退 linear。
    """
    progress = f"max(0,min(1,t/{max(0.1, float(duration)):.6f}))"
    if easing == "ease_in":
        return f"pow({progress},2)"
    if easing == "ease_out":
        return f"(1-pow(1-{progress},2))"
    if easing == "ease_in_out":
        return f"if(lt({progress},0.5),2*pow({progress},2),1-pow(-2*{progress}+2,2)/2)"
    return progress


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
