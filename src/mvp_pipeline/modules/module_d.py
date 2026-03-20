"""
文件用途：实现模块 D（视频合成）的 MVP 版本。
核心流程：将模块 C 关键帧按时长拼接，并与原音轨混流生成 MP4。
输入输出：输入 RuntimeContext，输出最终视频文件路径。
依赖说明：依赖标准库 subprocess 调用 FFmpeg。
维护说明：当引入更复杂转场时仍应保持绝对时间轴约束。
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
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：当关键帧清单为空时直接抛错，避免生成空视频。
    """
    context.logger.info("模块D开始执行，task_id=%s", context.task_id)

    module_c_path = context.artifacts_dir / "module_c_output.json"
    module_c_output = read_json(module_c_path)
    frame_items = module_c_output.get("frame_items", [])
    if not frame_items:
        raise RuntimeError("模块D无法执行：模块C输出的 frame_items 为空。")

    concat_file_path = context.artifacts_dir / "ffmpeg_concat.txt"
    _build_ffmpeg_concat_file(frame_items=frame_items, concat_file_path=concat_file_path)

    output_video_path = context.task_dir / "final_output.mp4"
    command = [
        context.config.ffmpeg.ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file_path),
        "-i",
        str(context.audio_path),
        "-c:v",
        context.config.ffmpeg.video_codec,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(context.config.ffmpeg.fps),
        "-c:a",
        context.config.ffmpeg.audio_codec,
        "-shortest",
        str(output_video_path),
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"模块D执行失败：找不到 ffmpeg 可执行文件，请检查配置 ffmpeg_bin={context.config.ffmpeg.ffmpeg_bin}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"模块D执行失败：ffmpeg 返回非零状态，stderr={error.stderr}") from error

    context.logger.info("模块D执行完成，task_id=%s，输出=%s", context.task_id, output_video_path)
    return output_video_path


def _build_ffmpeg_concat_file(frame_items: list[dict[str, Any]], concat_file_path: Path) -> None:
    """
    功能说明：生成 FFmpeg concat demuxer 所需清单文件。
    参数说明：
    - frame_items: 帧清单数组，需含 frame_path 与 duration。
    - concat_file_path: concat 清单输出路径。
    返回值：无。
    异常说明：写入失败时抛 OSError。
    边界条件：最后一帧需重复一次 file 行，防止最后 duration 丢失。
    """
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for item in frame_items:
        frame_path = _escape_concat_path(path_text=str(item["frame_path"]))
        duration = max(0.1, float(item.get("duration", 0.1)))
        lines.append(f"file '{frame_path}'")
        lines.append(f"duration {duration:.3f}")

    last_frame_path = _escape_concat_path(path_text=str(frame_items[-1]["frame_path"]))
    lines.append(f"file '{last_frame_path}'")

    concat_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
