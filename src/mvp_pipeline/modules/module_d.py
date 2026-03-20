"""
文件用途：实现模块 D（视频合成）的 MVP 版本。
核心流程：先将每个静态帧渲染为小视频片段，再按顺序拼接并混入原音轨输出 MP4。
输入输出：输入 RuntimeContext，输出最终视频文件路径。
依赖说明：依赖标准库 subprocess 调用 FFmpeg/FFprobe。
维护说明：本文件只保留“段视频->总拼接”方案，不再保留旧的图片直接 concat 方案。
"""

# 标准库：用于子进程命令执行
import subprocess
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from mvp_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from mvp_pipeline.io_utils import read_json


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

    segment_paths = _render_segment_videos(
        frame_items=frame_items,
        segments_dir=segments_dir,
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        fps=context.config.ffmpeg.fps,
        video_codec=context.config.ffmpeg.video_codec,
        video_preset=context.config.ffmpeg.video_preset,
        video_crf=context.config.ffmpeg.video_crf,
    )

    audio_duration = _probe_media_duration(
        media_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
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
    返回值：
    - list[Path]: 已生成的小视频片段路径数组。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：duration 最低限制为 0.1 秒，避免无效片段。
    """
    segment_paths: list[Path] = []
    for index, item in enumerate(frame_items, start=1):
        frame_path = Path(str(item["frame_path"]))
        duration = round(max(0.1, float(item.get("duration", 0.1))), 3)
        segment_path = segments_dir / f"segment_{index:03d}.mp4"

        command = [
            ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            str(frame_path),
            "-t",
            f"{duration:.3f}",
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
        _run_ffmpeg_command(command=command, command_name=f"渲染小片段 segment_{index:03d}")
        segment_paths.append(segment_path)
    return segment_paths


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
