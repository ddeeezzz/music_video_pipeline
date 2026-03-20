"""
文件用途：实现模块 A（音乐理解）的 MVP 版本。
核心流程：读取音频时长并生成分段、节拍、能量特征与歌词占位结构。
输入输出：输入 RuntimeContext，输出 ModuleAOutput JSON 文件路径。
依赖说明：依赖 mutagen（时长读取）与标准库 subprocess（ffprobe 兜底）。
维护说明：后续接入 librosa/allin1 时应保持输出字段兼容。
"""

# 标准库：用于命令执行（ffprobe 兜底）
import subprocess
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于读取音频元信息（时长）
from mutagen import File as MutagenFile

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 写入工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：契约校验函数
from music_video_pipeline.types import validate_module_a_output


def run_module_a(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 A 并落盘标准输出 JSON。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 A 输出 JSON 路径。
    异常说明：写文件失败时抛 OSError，契约不合规时抛异常。
    边界条件：歌词不可用时返回空 lyric_units。
    """
    context.logger.info("模块A开始执行，task_id=%s，输入音频=%s", context.task_id, context.audio_path)

    duration_seconds = _probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )
    segments = _build_segments(duration_seconds=duration_seconds)
    beats = _build_beats(duration_seconds=duration_seconds, interval_seconds=context.config.mock.beat_interval_seconds)
    energy_features = _build_energy_features(segments=segments)
    lyric_units: list[dict[str, Any]] = []

    output_data = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "segments": segments,
        "beats": beats,
        "lyric_units": lyric_units,
        "energy_features": energy_features,
    }
    validate_module_a_output(output_data)

    output_path = context.artifacts_dir / "module_a_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块A执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


def _probe_audio_duration(audio_path: Path, ffprobe_bin: str, logger) -> float:
    """
    功能说明：读取音频时长，优先 mutagen，失败时 ffprobe，最后兜底 20 秒。
    参数说明：
    - audio_path: 音频文件路径。
    - ffprobe_bin: ffprobe 可执行名或路径。
    - logger: 日志对象。
    返回值：
    - float: 音频时长（秒）。
    异常说明：内部捕获解析异常并降级，不向外抛。
    边界条件：返回值最小为 20.0 秒，避免下游出现 0 时长。
    """
    try:
        media_obj = MutagenFile(audio_path)
        if media_obj is not None and media_obj.info is not None and getattr(media_obj.info, "length", None):
            duration = float(media_obj.info.length)
            return max(20.0, duration)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A使用 mutagen 读取时长失败，已尝试 ffprobe，错误=%s", error)

    try:
        command = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return max(20.0, duration)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A使用 ffprobe 读取时长失败，已降级为默认20秒，错误=%s", error)
        return 20.0


def _build_segments(duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：根据音频时长构建宏观段落结构。
    参数说明：
    - duration_seconds: 音频时长（秒）。
    返回值：
    - list[dict[str, Any]]: 段落数组。
    异常说明：无。
    边界条件：总时长不足时至少返回一个段落。
    """
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    segment_count = min(len(labels), max(1, int(duration_seconds // 4)))
    segment_duration = duration_seconds / segment_count
    segments: list[dict[str, Any]] = []

    current_start = 0.0
    for index in range(segment_count):
        end_time = duration_seconds if index == segment_count - 1 else round(current_start + segment_duration, 3)
        segments.append(
            {
                "segment_id": f"seg_{index + 1:03d}",
                "start_time": round(current_start, 3),
                "end_time": round(end_time, 3),
                "label": labels[index % len(labels)],
            }
        )
        current_start = end_time
    return segments


def _build_beats(duration_seconds: float, interval_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：按固定间隔生成节拍占位点。
    参数说明：
    - duration_seconds: 音频时长（秒）。
    - interval_seconds: 节拍间隔（秒）。
    返回值：
    - list[dict[str, Any]]: 节拍数组。
    异常说明：无。
    边界条件：间隔非法时自动回退到 0.5 秒。
    """
    safe_interval = interval_seconds if interval_seconds > 0 else 0.5
    beats: list[dict[str, Any]] = []
    current_time = 0.0
    while current_time <= duration_seconds:
        beats.append(
            {
                "time": round(current_time, 3),
                "type": "major" if int(current_time / safe_interval) % 4 == 0 else "minor",
                "source": "beat",
            }
        )
        current_time += safe_interval
    return beats


def _build_energy_features(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：为每个段落生成规则化能量特征。
    参数说明：
    - segments: 段落数组。
    返回值：
    - list[dict[str, Any]]: 能量特征数组。
    异常说明：无。
    边界条件：segments 为空时返回空数组。
    """
    patterns = [
        ("low", "up", 0.30),
        ("mid", "up", 0.55),
        ("high", "flat", 0.85),
        ("mid", "down", 0.50),
        ("low", "flat", 0.25),
    ]
    features: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        energy_level, trend, tension = patterns[index % len(patterns)]
        features.append(
            {
                "start_time": float(segment["start_time"]),
                "end_time": float(segment["end_time"]),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": tension,
            }
        )
    return features
