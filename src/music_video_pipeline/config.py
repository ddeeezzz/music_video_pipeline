"""
文件用途：统一加载与管理项目配置。
核心流程：读取 JSON 配置并映射为 dataclass，提供默认值与向后兼容。
输入输出：输入配置文件路径，输出 AppConfig 对象。
依赖说明：依赖标准库 dataclasses/json/pathlib。
维护说明：新增配置项时必须同步默认值与文档说明。
"""

# 标准库：用于声明不可变数据类
from dataclasses import dataclass, field
# 标准库：用于 JSON 解析
import json
# 标准库：用于路径处理
from pathlib import Path


@dataclass(frozen=True)
class ModeConfig:
    """
    功能说明：定义模块 B/C 的生成器模式。
    参数说明：
    - script_generator: 分镜生成模式（mock/llm）。
    - frame_generator: 关键帧生成模式（mock/diffusion）。
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
    - video_preset: 编码预设。
    - video_crf: 视频质量参数。
    - render_batch_size: 兼容旧配置字段，当前固定按单段渲染语义使用。
    - render_workers: 受控并行渲染 worker 数量。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - gpu_video_codec: GPU 视频编码器（如 h264_nvenc/hevc_nvenc）。
    - gpu_preset: GPU 编码预设（如 p1~p7）。
    - gpu_rc_mode: GPU 码率控制模式（如 vbr_hq/cbr）。
    - gpu_cq: GPU 常质量参数（可选）。
    - gpu_bitrate: GPU 目标码率（可选，如 8M）。
    - concat_video_mode: 最终拼接视频处理模式（copy/reencode）。
    - concat_copy_fallback_reencode: copy 失败时是否自动回退重编码。
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
    render_batch_size: int = 1
    render_workers: int = 3
    video_accel_mode: str = "auto"
    gpu_video_codec: str = "h264_nvenc"
    gpu_preset: str = "p1"
    gpu_rc_mode: str = "vbr"
    gpu_cq: int | None = 34
    gpu_bitrate: str | None = None
    concat_video_mode: str = "copy"
    concat_copy_fallback_reencode: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """
    功能说明：定义日志等级配置。
    参数说明：
    - level: 标准 logging 级别字符串。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法级别由 logging 兜底为 INFO。
    """

    level: str


@dataclass(frozen=True)
class MockConfig:
    """
    功能说明：定义 Mock 链路参数。
    参数说明：
    - beat_interval_seconds: 默认节拍间隔。
    - video_width: 占位帧宽度。
    - video_height: 占位帧高度。
    返回值：不适用。
    异常说明：不适用。
    边界条件：宽高建议为偶数。
    """

    beat_interval_seconds: float
    video_width: int
    video_height: int


@dataclass(frozen=True)
class ModuleAConfig:
    """
    功能说明：定义模块 A 的真实链路配置。
    参数说明：
    - funasr_language: FunASR 语言策略（auto 或语言代码，如 zh/en/ja）。
    - lyric_segment_policy: 歌词视觉单元策略（sentence_strict/adaptive_phrase）。
    - comma_pause_seconds: 逗号断句停顿阈值（秒）。
    - long_pause_seconds: 长停顿断句阈值（秒）。
    - merge_gap_seconds: 相邻短句合并阈值（秒）。
    - max_visual_unit_seconds: 单个歌词视觉单元最大时长（秒）。
    - mode: 模式（real_auto/real_strict/fallback_only）。
    - lyric_beat_snap_threshold_ms: 歌词到节拍的吸附阈值（毫秒）。
    - instrumental_labels: 视为器乐段的标签集合。
    - fallback_enabled: 真实模型失败时是否降级到规则链。
    - device: 设备策略（auto/cpu/cuda）。
    - funasr_model: FunASR 模型名称。
    - demucs_model: Demucs 模型名称。
    - vocal_energy_enter_quantile: 人声音量进入阈值分位点。
    - vocal_energy_exit_quantile: 人声音量退出阈值分位点。
    - mid_segment_min_duration_seconds: 人声中间段最小时长（秒）。
    - short_vocal_non_lyric_merge_seconds: 人声“无歌词/短歌词”短段合并阈值（秒）。
    - instrumental_single_split_min_seconds: 器乐单次切分触发最小时长（秒）。
    - accent_delta_trigger_ratio: 首重音检测的能量突变触发比例（0~1）。
    - skip_funasr_when_vocals_silent: 当人声音轨能量极低时是否跳过 FunASR。
    - vocal_skip_peak_rms_threshold: “极低人声”判定的峰值 RMS 阈值。
    - vocal_skip_active_ratio_threshold: “极低人声”判定的活跃帧占比阈值。
    返回值：不适用。
    异常说明：不适用。
    边界条件：阈值建议大于等于 0。
    """

    funasr_language: str
    lyric_segment_policy: str = "sentence_strict"
    comma_pause_seconds: float = 0.45
    long_pause_seconds: float = 0.8
    merge_gap_seconds: float = 0.25
    max_visual_unit_seconds: float = 6.0
    mode: str = "real_auto"
    lyric_beat_snap_threshold_ms: int = 200
    instrumental_labels: list[str] = field(default_factory=lambda: ["intro", "outro", "inst"])
    fallback_enabled: bool = True
    device: str = "auto"
    funasr_model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    demucs_model: str = "htdemucs"
    vocal_energy_enter_quantile: float = 0.70
    vocal_energy_exit_quantile: float = 0.45
    mid_segment_min_duration_seconds: float = 0.8
    short_vocal_non_lyric_merge_seconds: float = 1.2
    instrumental_single_split_min_seconds: float = 4.0
    accent_delta_trigger_ratio: float = 0.35
    skip_funasr_when_vocals_silent: bool = True
    vocal_skip_peak_rms_threshold: float = 0.010
    vocal_skip_active_ratio_threshold: float = 0.020


@dataclass(frozen=True)
class AppConfig:
    """
    功能说明：聚合应用全局配置。
    参数说明：
    - mode: 生成器模式配置。
    - paths: 路径配置。
    - ffmpeg: FFmpeg 配置。
    - logging: 日志配置。
    - mock: Mock 参数配置。
    - module_a: 模块 A 参数配置。
    返回值：不适用。
    异常说明：不适用。
    边界条件：缺少字段时走默认值。
    """

    mode: ModeConfig
    paths: PathsConfig
    ffmpeg: FfmpegConfig
    logging: LoggingConfig
    mock: MockConfig
    module_a: ModuleAConfig = field(default_factory=lambda: ModuleAConfig(funasr_language="auto"))


def _read_json_config(config_path: Path) -> dict:
    """
    功能说明：读取 JSON 配置并返回字典。
    参数说明：
    - config_path: JSON 配置文件路径。
    返回值：
    - dict: 解析后的配置字典。
    异常说明：
    - FileNotFoundError: 文件不存在。
    - json.JSONDecodeError: 不是合法 JSON。
    边界条件：路径必须指向文件。
    """
    with config_path.open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def _merge_defaults(raw_data: dict) -> dict:
    """
    功能说明：为缺失配置项补齐默认值。
    参数说明：
    - raw_data: 原始配置字典。
    返回值：
    - dict: 合并默认值后的配置字典。
    异常说明：无。
    边界条件：调用方需保证 raw_data 为字典。
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
            "render_batch_size": 1,
            "render_workers": 3,
            "video_accel_mode": "auto",
            "gpu_video_codec": "h264_nvenc",
            "gpu_preset": "p1",
            "gpu_rc_mode": "vbr",
            "gpu_cq": 34,
            "gpu_bitrate": None,
            "concat_video_mode": "copy",
            "concat_copy_fallback_reencode": True,
        },
        "logging": {"level": "INFO"},
        "mock": {"beat_interval_seconds": 0.5, "video_width": 960, "video_height": 540},
        "module_a": {
            "mode": "real_auto",
            "lyric_beat_snap_threshold_ms": 200,
            "instrumental_labels": ["intro", "outro", "inst"],
            "fallback_enabled": True,
            "device": "auto",
            "lyric_segment_policy": "sentence_strict",
            "comma_pause_seconds": 0.45,
            "long_pause_seconds": 0.8,
            "merge_gap_seconds": 0.25,
            "max_visual_unit_seconds": 6.0,
            "funasr_model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "demucs_model": "htdemucs",
            "vocal_energy_enter_quantile": 0.70,
            "vocal_energy_exit_quantile": 0.45,
            "mid_segment_min_duration_seconds": 0.8,
            "short_vocal_non_lyric_merge_seconds": 1.2,
            "instrumental_single_split_min_seconds": 4.0,
            "accent_delta_trigger_ratio": 0.35,
            "skip_funasr_when_vocals_silent": True,
            "vocal_skip_peak_rms_threshold": 0.010,
            "vocal_skip_active_ratio_threshold": 0.020,
        },
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
    边界条件：当配置缺失字段时自动补默认值。
    """
    raw_data = _read_json_config(config_path)
    merged = _merge_defaults(raw_data)
    return AppConfig(
        mode=ModeConfig(**merged["mode"]),
        paths=PathsConfig(**merged["paths"]),
        ffmpeg=FfmpegConfig(**merged["ffmpeg"]),
        logging=LoggingConfig(**merged["logging"]),
        mock=MockConfig(**merged["mock"]),
        module_a=ModuleAConfig(**merged["module_a"]),
    )
