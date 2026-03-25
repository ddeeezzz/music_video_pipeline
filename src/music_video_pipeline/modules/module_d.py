"""
文件用途：实现模块 D（视频合成）的 MVP 版本。
核心流程：先将每个静态帧渲染为小视频片段，再按顺序拼接并混入原音轨输出 MP4。
输入输出：输入 RuntimeContext，输出最终视频文件路径。
依赖说明：依赖标准库 subprocess 调用 FFmpeg/FFprobe。
维护说明：本文件只保留“段视频->总拼接”方案，不再保留旧的图片直接 concat 方案。
"""

# 标准库：用于子进程命令执行
import subprocess
# 标准库：用于日志对象类型提示
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json


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

    segment_paths = _render_segment_videos(
        frame_items=frame_items,
        segments_dir=segments_dir,
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        fps=context.config.ffmpeg.fps,
        video_codec=context.config.ffmpeg.video_codec,
        video_preset=context.config.ffmpeg.video_preset,
        video_crf=context.config.ffmpeg.video_crf,
        audio_duration=audio_duration,
        logger=context.logger,
    )

    output_video_path = context.task_dir / "final_output.mp4"
    _concat_segment_videos(
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
    audio_duration: float,
    logger: logging.Logger,
) -> list[Path]:
    """
    功能说明：将每个静态帧渲染成独立的小视频片段。
    参数说明：
    - frame_items: 帧清单数组，需含 frame_path 与 duration。
    - segments_dir: 小视频片段输出目录。
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - fps: 输出帧率。
    - video_codec: 视频编码器。
    - video_preset: 视频编码预设。
    - video_crf: 视频 CRF 参数。
    - audio_duration: 原音轨时长（秒）。
    - logger: 日志对象。
    返回值：
    - list[Path]: 已生成的小视频片段路径数组。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：使用全局帧分配，保证各片段总帧数与音频目标帧数一致。
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

    segment_paths: list[Path] = []
    for index, item in enumerate(frame_items, start=1):
        frame_path = Path(str(item["frame_path"]))
        exact_frames = allocated_frames[index - 1]
        segment_path = segments_dir / f"segment_{index:03d}.mp4"

        command = [
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
            "-c:v",
            video_codec,
            "-preset",
            video_preset,
            "-crf",
            str(video_crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(segment_path),
        ]
        try:
            _run_ffmpeg_command(command=command, command_name=f"渲染小片段 segment_{index:03d}")
        except RuntimeError as error:
            detail_lines = _build_frame_allocation_detail_lines(frame_items=frame_items, allocated_frames=allocated_frames, fps=fps)
            detail_text = "\n".join(detail_lines)
            raise RuntimeError(f"{error}\n模块D逐段帧分配明细：\n{detail_text}") from error
        segment_paths.append(segment_path)
    return segment_paths


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
) -> None:
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
    返回值：无。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：显式使用 -t 音频时长，避免最终视频长于音频。
    """
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{_escape_concat_path(str(path))}'" for path in segment_paths]
    concat_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
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
        "-c:v",
        video_codec,
        "-preset",
        video_preset,
        "-crf",
        str(video_crf),
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
    _run_ffmpeg_command(command=command, command_name="拼接小片段并混音")


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
