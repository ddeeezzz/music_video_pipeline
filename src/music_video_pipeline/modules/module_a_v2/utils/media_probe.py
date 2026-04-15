"""
文件用途：提供模块A V2音频媒体信息探测能力。
核心流程：优先使用 mutagen 读取时长，失败后回退 ffprobe，再回退默认值。
输入输出：输入音频路径与 ffprobe 命令，输出音频总时长（秒）。
依赖说明：依赖 mutagen 与 subprocess。
维护说明：本文件仅负责时长探测，不承载编排流程。
"""

# 标准库：子进程调用
import subprocess
# 标准库：路径类型
from pathlib import Path

# 第三方库：读取音频时长
from mutagen import File as MutagenFile


def probe_audio_duration(audio_path: Path, ffprobe_bin: str, logger) -> float:
    """
    功能说明：读取音频时长，优先 mutagen，失败时 ffprobe。
    参数说明：
    - audio_path: 输入音频文件路径。
    - ffprobe_bin: ffprobe可执行命令路径或命令名。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - float: 音频时长（秒）。
    异常说明：异常在函数内吞并并按兜底策略处理。
    边界条件：双探测均失败时回退默认 20 秒。
    """
    try:
        media_obj = MutagenFile(audio_path)
        if media_obj is not None and media_obj.info is not None and getattr(media_obj.info, "length", None):
            return max(0.1, float(media_obj.info.length))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2使用 mutagen 读取时长失败，已尝试 ffprobe，错误=%s", error)

    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return max(0.1, float(result.stdout.strip()))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2使用 ffprobe 读取时长失败，已降级默认20秒，错误=%s", error)
        return 20.0
