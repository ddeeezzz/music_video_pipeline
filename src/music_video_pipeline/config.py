"""
文件用途：统一加载与管理项目配置。
核心流程：读取 JSON 配置并映射为 dataclass，提供默认值与向后兼容。
输入输出：输入配置文件路径，输出 AppConfig 对象。
依赖说明：依赖标准库 dataclasses/json/pathlib。
维护说明：新增配置项时必须同步默认值与文档说明。
"""

# 标准库：用于声明不可变数据类
from dataclasses import dataclass, field
# 标准库：用于配置兼容告警
import logging
# 标准库：用于 JSON 解析
import json
# 标准库：用于路径处理
from pathlib import Path

# 常量：配置模块日志器（用于兼容键清理告警）
LOGGER = logging.getLogger(__name__)


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
class ModuleBLlmConfig:
    """
    功能说明：定义模块 B 真实 LLM 分镜生成参数。
    参数说明：
    - provider: LLM 服务提供商标识（当前支持 siliconflow）。
    - base_url: Chat Completions 接口根地址。
    - model: 目标模型名称。
    - api_key_file: API Key 文件路径（支持相对路径）。
    - timeout_seconds: 单次请求超时（秒）。
    - request_retry_times: HTTP 请求重试次数。
    - json_retry_times: JSON 解析失败后的补救重试次数。
    - temperature: 采样温度。
    - top_p: 采样 top_p。
    - max_tokens: 单次输出 token 上限。
    - use_response_format_json_object: 是否启用 response_format=json_object。
    - scene_desc_max_chars: scene_desc 最大字符数。
    - keyframe_prompt_max_chars: keyframe_prompt 最大字符数。
    - video_prompt_max_chars: video_prompt 最大字符数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：当 script_generator=llm 时由模块 B 调用层读取并执行。
    """

    provider: str = "siliconflow"
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "deepseek-ai/DeepSeek-V3.2"
    api_key_file: str = ".secrets/siliconflow_api_key.txt"
    timeout_seconds: float = 60.0
    request_retry_times: int = 2
    json_retry_times: int = 2
    temperature: float = 0.30
    top_p: float = 0.90
    max_tokens: int = 350
    use_response_format_json_object: bool = False
    scene_desc_max_chars: int = 120
    keyframe_prompt_max_chars: int = 400
    video_prompt_max_chars: int = 500


@dataclass(frozen=True)
class ModuleBConfig:
    """
    功能说明：定义模块 B 的并行与重试参数。
    参数说明：
    - script_workers: 分镜最小单元并行生成 worker 数量。
    - unit_retry_times: 单元失败后的重试次数。
    - llm: 模块 B 真实 LLM 分镜参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法值由模块 B 执行层归一化兜底。
    """

    script_workers: int = 3
    unit_retry_times: int = 1
    llm: ModuleBLlmConfig = field(default_factory=ModuleBLlmConfig)


@dataclass(frozen=True)
class ModuleCConfig:
    """
    功能说明：定义模块 C 的并行与重试参数。
    参数说明：
    - render_workers: 最小视觉单元并行生成 worker 数量。
    - unit_retry_times: 单元失败后的重试次数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法值由模块 C 执行层归一化兜底。
    """

    render_workers: int = 3
    unit_retry_times: int = 1


@dataclass(frozen=True)
class ModuleDConfig:
    """
    功能说明：定义模块 D 的并行与重试参数。
    参数说明：
    - segment_workers: 片段最小单元并行渲染 worker 数量。
    - unit_retry_times: 单元失败后的重试次数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法值由模块 D 执行层归一化兜底。
    """

    segment_workers: int = 3
    unit_retry_times: int = 1


@dataclass(frozen=True)
class CrossModuleConfig:
    """
    功能说明：定义跨模块（B/C/D）并行调度参数。
    参数说明：
    - global_render_limit: 模块 C 与模块 D 的共享并发上限。
    - scheduler_tick_ms: 调度器轮询间隔（毫秒）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：非法值由跨模块调度层归一化兜底。
    """

    global_render_limit: int = 3
    scheduler_tick_ms: int = 50


@dataclass(frozen=True)
class MonitoringConfig:
    """
    功能说明：定义运行时任务监督服务参数。
    参数说明：
    - max_wait_after_terminal_minutes: 任务进入终态后，CLI等待监督服务退出的最长分钟数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：默认20分钟，超时后CLI会强制关闭监督服务。
    """

    max_wait_after_terminal_minutes: float = 20.0


@dataclass(frozen=True)
class ModuleAConfig:
    """
    功能说明：定义模块 A 的真实链路配置。
    参数说明：
    - funasr_language: FunASR 语言策略（auto 或语言代码，如 zh/en/ja）。
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
    - implementation: 模块A实现版本（v1/v2），用于迁移期开关切换。
    - lyric_head_offset_seconds: 歌词句首小时间戳固定后移量（秒，v2歌词主链使用）。
    - long_instrumental_gap_seconds: 人声段内长器乐补检触发阈值（秒，v2歌词主链使用）。
    - lyric_boundary_near_anchor_seconds: 大段边界后“近锚点冲突”判定阈值（秒，v2歌词主链使用）。
    - content_role_tiny_merge_bars: 内容角色tiny并段阈值（小节，v2四分类清理使用）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：阈值建议大于等于 0。
    """

    funasr_language: str
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
    implementation: str = "v1"
    lyric_head_offset_seconds: float = 0.02
    long_instrumental_gap_seconds: float = 5.0
    lyric_boundary_near_anchor_seconds: float = 1.5
    content_role_tiny_merge_bars: float = 0.9


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
    - module_b: 模块 B 参数配置。
    - module_c: 模块 C 参数配置。
    - module_d: 模块 D 参数配置。
    - cross_module: 跨模块并行调度参数配置。
    - monitoring: 运行时监督服务配置。
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
    module_b: ModuleBConfig = field(default_factory=ModuleBConfig)
    module_c: ModuleCConfig = field(default_factory=ModuleCConfig)
    module_d: ModuleDConfig = field(default_factory=ModuleDConfig)
    cross_module: CrossModuleConfig = field(default_factory=CrossModuleConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
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
        "module_b": {
            "script_workers": 3,
            "unit_retry_times": 1,
            "llm": {
                "provider": "siliconflow",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-V3.2",
                "api_key_file": ".secrets/siliconflow_api_key.txt",
                "timeout_seconds": 60.0,
                "request_retry_times": 2,
                "json_retry_times": 2,
                "temperature": 0.30,
                "top_p": 0.90,
                "max_tokens": 350,
                "use_response_format_json_object": False,
                "scene_desc_max_chars": 120,
                "keyframe_prompt_max_chars": 400,
                "video_prompt_max_chars": 500,
            },
        },
        "module_c": {
            "render_workers": 3,
            "unit_retry_times": 1,
        },
        "module_d": {
            "segment_workers": 3,
            "unit_retry_times": 1,
        },
        "cross_module": {
            "global_render_limit": 3,
            "scheduler_tick_ms": 50,
        },
        "monitoring": {
            "max_wait_after_terminal_minutes": 20.0,
        },
        "module_a": {
            "mode": "real_auto",
            "lyric_beat_snap_threshold_ms": 200,
            "instrumental_labels": ["intro", "outro", "inst"],
            "fallback_enabled": True,
            "device": "auto",
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
            "implementation": "v1",
            "lyric_head_offset_seconds": 0.02,
            "long_instrumental_gap_seconds": 5.0,
            "lyric_boundary_near_anchor_seconds": 1.5,
            "content_role_tiny_merge_bars": 0.9,
        },
    }

    merged = default_data
    for top_key, top_value in raw_data.items():
        if top_key == "module_b" and isinstance(top_value, dict) and isinstance(merged.get(top_key), dict):
            merged_module_b = {**merged[top_key], **top_value}
            default_llm = merged[top_key].get("llm", {})
            override_llm = top_value.get("llm", {})
            if isinstance(default_llm, dict) and isinstance(override_llm, dict):
                merged_module_b["llm"] = {**default_llm, **override_llm}
            merged[top_key] = merged_module_b
        elif isinstance(top_value, dict) and isinstance(merged.get(top_key), dict):
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
    module_b_data = dict(merged["module_b"])
    module_b_llm_data = module_b_data.pop("llm", {})
    if not isinstance(module_b_llm_data, dict):
        raise TypeError("配置错误：module_b.llm 必须是对象。")

    module_a_data = dict(merged["module_a"])
    raw_module_a_data = raw_data.get("module_a", {}) if isinstance(raw_data.get("module_a", {}), dict) else {}
    if "lyric_segment_policy" in module_a_data:
        LOGGER.warning("配置键 module_a.lyric_segment_policy 已移除并忽略，请删除该配置项。")
        module_a_data.pop("lyric_segment_policy", None)
    # 兼容说明：历史配置使用 content_role_tiny_merge_seconds，现统一为按小节阈值。
    if (
        "content_role_tiny_merge_bars" not in raw_module_a_data
        and "content_role_tiny_merge_seconds" in raw_module_a_data
    ):
        module_a_data["content_role_tiny_merge_bars"] = raw_module_a_data.get("content_role_tiny_merge_seconds")
    module_a_data.pop("content_role_tiny_merge_seconds", None)
    if "english_head_pullback_window_seconds" in module_a_data:
        LOGGER.warning("配置键 module_a.english_head_pullback_window_seconds 已移除并忽略，请删除该配置项。")
        module_a_data.pop("english_head_pullback_window_seconds", None)

    return AppConfig(
        mode=ModeConfig(**merged["mode"]),
        paths=PathsConfig(**merged["paths"]),
        ffmpeg=FfmpegConfig(**merged["ffmpeg"]),
        logging=LoggingConfig(**merged["logging"]),
        mock=MockConfig(**merged["mock"]),
        module_b=ModuleBConfig(llm=ModuleBLlmConfig(**module_b_llm_data), **module_b_data),
        module_c=ModuleCConfig(**merged["module_c"]),
        module_d=ModuleDConfig(**merged["module_d"]),
        cross_module=CrossModuleConfig(**merged["cross_module"]),
        monitoring=MonitoringConfig(**merged["monitoring"]),
        module_a=ModuleAConfig(**module_a_data),
    )
