"""
文件用途：统一加载与管理 MVP 配置。
核心流程：读取 JSON 配置文件并映射到 dataclass。
输入输出：输入配置路径，输出 AppConfig 实例。
依赖说明：依赖标准库 json/pathlib/dataclasses。
维护说明：新增配置项时需保持默认值与向后兼容。
"""

# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于 JSON 解析
import json
# 标准库：用于路径处理
from pathlib import Path


@dataclass(frozen=True)
class ModeConfig:
    """
    功能说明：定义模块 B/C 的生成器模式。
    参数说明：
    - script_generator: 分镜生成器模式（mock/llm）。
    - frame_generator: 帧生成器模式（mock/diffusion）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：未知模式在工厂层做降级处理。
    """

    script_generator: str
    frame_generator: str


@dataclass(frozen=True)
class PathsConfig:
    """
    功能说明：定义运行目录与默认输入路径。
    参数说明：
    - runs_dir: 运行输出根目录。
    - default_audio_path: 默认音频路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：路径可为相对路径，最终由调用方解析。
    """

    runs_dir: str
    default_audio_path: str


@dataclass(frozen=True)
class FfmpegConfig:
    """
    功能说明：定义 FFmpeg 相关参数。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行名或绝对路径。
    - ffprobe_bin: ffprobe 可执行名或绝对路径。
    - video_codec: 输出视频编码。
    - audio_codec: 输出音频编码。
    - fps: 输出帧率。
    - video_preset: 视频编码预设（速度/压缩率权衡）。
    - video_crf: 视频 CRF 质量参数（越大越快、体积越小）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：ffmpeg 缺失时在模块 D 抛出运行错误。
    """

    ffmpeg_bin: str
    ffprobe_bin: str
    video_codec: str
    audio_codec: str
    fps: int
    video_preset: str
    video_crf: int


@dataclass(frozen=True)
class LoggingConfig:
    """
    功能说明：定义日志级别配置。
    参数说明：
    - level: 标准 logging 级别字符串。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法级别由 logging 模块自动兜底为 INFO。
    """

    level: str


@dataclass(frozen=True)
class MockConfig:
    """
    功能说明：定义 Mock 模块参数。
    参数说明：
    - beat_interval_seconds: 默认节拍间隔。
    - video_width: 占位图宽度。
    - video_height: 占位图高度。
    返回值：不适用。
    异常说明：不适用。
    边界条件：宽高建议为偶数，便于编码器处理。
    """

    beat_interval_seconds: float
    video_width: int
    video_height: int


@dataclass(frozen=True)
class AppConfig:
    """
    功能说明：聚合全局应用配置。
    参数说明：
    - mode: 生成器模式配置。
    - paths: 路径配置。
    - ffmpeg: FFmpeg 配置。
    - logging: 日志配置。
    - mock: Mock 参数配置。
    返回值：不适用。
    异常说明：不适用。
    边界条件：配置文件缺字段时走默认值。
    """

    mode: ModeConfig
    paths: PathsConfig
    ffmpeg: FfmpegConfig
    logging: LoggingConfig
    mock: MockConfig


def _read_json_config(config_path: Path) -> dict:
    """
    功能说明：读取 JSON 配置并返回字典。
    参数说明：
    - config_path: JSON 配置文件路径。
    返回值：
    - dict: 解析后的配置字典。
    异常说明：
    - FileNotFoundError: 文件不存在。
    - json.JSONDecodeError: 配置不是合法 JSON。
    边界条件：路径必须指向文件而非目录。
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _merge_defaults(raw_data: dict) -> dict:
    """
    功能说明：为缺失配置项补齐默认值。
    参数说明：
    - raw_data: 原始配置字典。
    返回值：
    - dict: 含默认值的完整配置字典。
    异常说明：无。
    边界条件：调用方应保证 raw_data 为字典。
    """
    default_data = {
        "mode": {"script_generator": "mock", "frame_generator": "mock"},
        "paths": {"runs_dir": "runs", "default_audio_path": "resources/juebieshu20s.mp3"},
        "ffmpeg": {
            "ffmpeg_bin": "ffmpeg",
            "ffprobe_bin": "ffprobe",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "fps": 24,
            "video_preset": "veryfast",
            "video_crf": 30,
        },
        "logging": {"level": "INFO"},
        "mock": {"beat_interval_seconds": 0.5, "video_width": 960, "video_height": 540},
    }

    merged = default_data
    for top_key, top_value in raw_data.items():
        if isinstance(top_value, dict) and isinstance(merged.get(top_key), dict):
            merged[top_key] = {**merged[top_key], **top_value}
        else:
            merged[top_key] = top_value
    return merged


def load_config(config_path: Path) -> AppConfig:
    """
    功能说明：加载配置文件并转换为 AppConfig。
    参数说明：
    - config_path: 配置文件路径。
    返回值：
    - AppConfig: 应用配置对象。
    异常说明：
    - FileNotFoundError/json.JSONDecodeError: 由读取函数抛出。
    边界条件：当配置项缺失时自动补默认值。
    """
    raw_data = _read_json_config(config_path)
    merged = _merge_defaults(raw_data)
    return AppConfig(
        mode=ModeConfig(**merged["mode"]),
        paths=PathsConfig(**merged["paths"]),
        ffmpeg=FfmpegConfig(**merged["ffmpeg"]),
        logging=LoggingConfig(**merged["logging"]),
        mock=MockConfig(**merged["mock"]),
    )
