"""
文件用途：验证配置加载逻辑中的 module_a.funasr_language 行为。
核心流程：构造临时配置文件，覆盖必填字段缺失与显式赋值场景。
输入输出：输入 JSON 配置文件，输出 AppConfig 断言结果。
依赖说明：依赖 pytest 与项目内 load_config 实现。
维护说明：当配置结构变更时需同步更新本测试。
"""

# 标准库：用于 JSON 写入
import json
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置加载入口
from music_video_pipeline.config import load_config


def test_load_config_should_raise_when_funasr_language_missing(tmp_path: Path) -> None:
    """
    功能说明：验证 module_a.funasr_language 缺失时配置加载应失败。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本测试要求非兼容策略生效，不允许自动补默认值。
    """
    config_path = tmp_path / "config_missing_funasr_language.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": {"script_generator": "mock", "frame_generator": "mock"},
                "paths": {"runs_dir": "runs", "default_audio_path": "resources/demo.mp3"},
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
                "module_a": {
                    "mode": "real_auto",
                    "lyric_beat_snap_threshold_ms": 200,
                    "instrumental_labels": ["intro", "outro", "inst"],
                    "fallback_enabled": True,
                    "device": "auto",
                    "funasr_model": "FunAudioLLM/Fun-ASR-Nano-2512",
                    "demucs_model": "htdemucs",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        load_config(config_path=config_path)


def test_load_config_should_accept_explicit_funasr_language(tmp_path: Path) -> None:
    """
    功能说明：验证显式 funasr_language 能正确映射到配置对象。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖 auto 与指定语言两种合法场景。
    """
    for language_value in ["auto", "zh"]:
        config_path = tmp_path / f"config_with_{language_value}.json"
        config_path.write_text(
            json.dumps(
                {
                    "module_a": {
                        "funasr_language": language_value,
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        app_config = load_config(config_path=config_path)
        assert app_config.module_a.funasr_language == language_value


def test_load_config_should_fail_when_legacy_whisper_fields_present(tmp_path: Path) -> None:
    """
    功能说明：验证旧版 whisper 字段在新配置结构下会触发构造失败。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本测试验证破坏性升级，不做向后兼容。
    """
    config_path = tmp_path / "config_with_legacy_whisper_fields.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                    "whisper_language": "auto",
                    "whisper_model": "base",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        load_config(config_path=config_path)


def test_load_config_should_ignore_removed_lyric_segment_policy_key_with_warning(tmp_path: Path, caplog) -> None:
    """
    功能说明：验证已移除键 lyric_segment_policy 会被忽略并输出兼容告警。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - caplog: pytest 日志捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧键存在时配置加载仍应成功。
    """
    config_path = tmp_path / "config_with_removed_lyric_segment_policy.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                    "lyric_segment_policy": "adaptive_phrase",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    caplog.set_level("WARNING")
    app_config = load_config(config_path=config_path)
    assert "lyric_segment_policy 已移除并忽略" in caplog.text
    assert not hasattr(app_config.module_a, "lyric_segment_policy")
    assert app_config.module_a.comma_pause_seconds == 0.45
    assert app_config.module_a.long_pause_seconds == 0.8
    assert app_config.module_a.merge_gap_seconds == 0.25
    assert app_config.module_a.max_visual_unit_seconds == 6.0
    assert app_config.module_a.vocal_energy_enter_quantile == 0.70
    assert app_config.module_a.vocal_energy_exit_quantile == 0.45
    assert app_config.module_a.mid_segment_min_duration_seconds == 0.8
    assert app_config.module_a.short_vocal_non_lyric_merge_seconds == 1.2
    assert app_config.module_a.instrumental_single_split_min_seconds == 4.0
    assert app_config.module_a.accent_delta_trigger_ratio == 0.35
    assert app_config.module_a.skip_funasr_when_vocals_silent is True
    assert app_config.module_a.vocal_skip_peak_rms_threshold == 0.010
    assert app_config.module_a.vocal_skip_active_ratio_threshold == 0.020
    assert app_config.module_a.implementation == "v1"
    assert app_config.module_a.lyric_head_offset_seconds == 0.02
    assert app_config.module_a.long_instrumental_gap_seconds == 5.0
    assert app_config.module_a.lyric_boundary_near_anchor_seconds == 1.5
    assert app_config.module_a.content_role_tiny_merge_bars == 0.9


def test_load_config_should_accept_explicit_module_a_segmentation_tuning(tmp_path: Path) -> None:
    """
    功能说明：验证 module_a 新增分段调参字段可显式加载并覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅覆盖本次新增字段，不影响其他默认项。
    """
    config_path = tmp_path / "config_with_module_a_tuning.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                    "vocal_energy_enter_quantile": 0.75,
                    "vocal_energy_exit_quantile": 0.40,
                    "mid_segment_min_duration_seconds": 0.9,
                    "short_vocal_non_lyric_merge_seconds": 0.7,
                    "instrumental_single_split_min_seconds": 5.0,
                    "accent_delta_trigger_ratio": 0.25,
                    "skip_funasr_when_vocals_silent": False,
                    "vocal_skip_peak_rms_threshold": 0.02,
                    "vocal_skip_active_ratio_threshold": 0.05,
                    "implementation": "v2",
                    "lyric_head_offset_seconds": 0.05,
                    "long_instrumental_gap_seconds": 6.0,
                    "lyric_boundary_near_anchor_seconds": 2.0,
                    "content_role_tiny_merge_bars": 0.6,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_a.vocal_energy_enter_quantile == 0.75
    assert app_config.module_a.vocal_energy_exit_quantile == 0.40
    assert app_config.module_a.mid_segment_min_duration_seconds == 0.9
    assert app_config.module_a.short_vocal_non_lyric_merge_seconds == 0.7
    assert app_config.module_a.instrumental_single_split_min_seconds == 5.0
    assert app_config.module_a.accent_delta_trigger_ratio == 0.25
    assert app_config.module_a.skip_funasr_when_vocals_silent is False
    assert app_config.module_a.vocal_skip_peak_rms_threshold == 0.02
    assert app_config.module_a.vocal_skip_active_ratio_threshold == 0.05
    assert app_config.module_a.implementation == "v2"
    assert app_config.module_a.lyric_head_offset_seconds == 0.05
    assert app_config.module_a.long_instrumental_gap_seconds == 6.0
    assert app_config.module_a.lyric_boundary_near_anchor_seconds == 2.0
    assert app_config.module_a.content_role_tiny_merge_bars == 0.6


def test_load_config_should_compat_old_content_role_tiny_merge_seconds_key(tmp_path: Path) -> None:
    """
    功能说明：验证旧配置键 content_role_tiny_merge_seconds 仍可兼容映射到小节阈值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当新旧键同时存在时，新键优先级应更高（该用例仅覆盖旧键单独存在）。
    """
    config_path = tmp_path / "config_with_old_tiny_seconds_key.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                    "content_role_tiny_merge_seconds": 0.9,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_a.content_role_tiny_merge_bars == 0.9


def test_load_config_should_fill_ffmpeg_gpu_accel_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 ffmpeg GPU 与 concat 新字段在缺省配置下可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验新增字段，不影响既有 ffmpeg 基础字段行为。
    """
    config_path = tmp_path / "config_ffmpeg_gpu_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.ffmpeg.render_batch_size == 1
    assert app_config.ffmpeg.render_workers == 3
    assert app_config.ffmpeg.video_accel_mode == "auto"
    assert app_config.ffmpeg.gpu_video_codec == "h264_nvenc"
    assert app_config.ffmpeg.gpu_preset == "p1"
    assert app_config.ffmpeg.gpu_rc_mode == "vbr"
    assert app_config.ffmpeg.gpu_cq == 34
    assert app_config.ffmpeg.gpu_bitrate is None


def test_load_config_should_ignore_removed_english_head_pullback_key_with_warning(tmp_path: Path, caplog) -> None:
    """
    功能说明：验证旧键 english_head_pullback_window_seconds 会被忽略并输出兼容告警。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - caplog: pytest 日志捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧键存在时配置加载仍应成功。
    """
    config_path = tmp_path / "config_with_removed_english_pullback_key.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                    "implementation": "v2",
                    "english_head_pullback_window_seconds": 0.35,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    caplog.set_level("WARNING")
    app_config = load_config(config_path=config_path)
    assert app_config.module_a.implementation == "v2"
    assert "english_head_pullback_window_seconds 已移除并忽略" in caplog.text
    assert not hasattr(app_config.module_a, "english_head_pullback_window_seconds")
    assert app_config.ffmpeg.concat_video_mode == "copy"
    assert app_config.ffmpeg.concat_copy_fallback_reencode is True


def test_load_config_should_fill_module_c_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 module_c 配置缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认值来自代码，不依赖 default.json。
    """
    config_path = tmp_path / "config_module_c_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_c.render_workers == 3
    assert app_config.module_c.unit_retry_times == 1


def test_load_config_should_fill_module_b_llm_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 module_b.llm 缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验 module_b.llm 新增字段，不影响其他配置。
    """
    config_path = tmp_path / "config_module_b_llm_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_b.llm.provider == "siliconflow"
    assert app_config.module_b.llm.base_url == "https://api.siliconflow.cn/v1"
    assert app_config.module_b.llm.model == "deepseek-ai/DeepSeek-V3.2"
    assert app_config.module_b.llm.api_key_file == ".secrets/siliconflow_api_key.txt"
    assert app_config.module_b.llm.timeout_seconds == 60.0
    assert app_config.module_b.llm.request_retry_times == 2
    assert app_config.module_b.llm.json_retry_times == 2
    assert app_config.module_b.llm.temperature == 0.30
    assert app_config.module_b.llm.top_p == 0.90
    assert app_config.module_b.llm.max_tokens == 350
    assert app_config.module_b.llm.use_response_format_json_object is False
    assert app_config.module_b.llm.scene_desc_max_chars == 120
    assert app_config.module_b.llm.keyframe_prompt_max_chars == 400
    assert app_config.module_b.llm.video_prompt_max_chars == 500


def test_load_config_should_accept_module_b_llm_overrides(tmp_path: Path) -> None:
    """
    功能说明：验证 module_b.llm 支持显式覆盖并正确映射。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当仅覆盖部分字段时，未覆盖项继续使用默认值。
    """
    config_path = tmp_path / "config_module_b_llm_overrides.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "module_b": {
                    "script_workers": 4,
                    "unit_retry_times": 2,
                    "llm": {
                        "provider": "siliconflow",
                        "model": "deepseek-ai/DeepSeek-V3.2",
                        "api_key_file": ".secrets/custom_key.txt",
                        "temperature": 0.15,
                        "top_p": 0.85,
                        "json_retry_times": 4,
                        "scene_desc_max_chars": 80,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_b.script_workers == 4
    assert app_config.module_b.unit_retry_times == 2
    assert app_config.module_b.llm.provider == "siliconflow"
    assert app_config.module_b.llm.model == "deepseek-ai/DeepSeek-V3.2"
    assert app_config.module_b.llm.api_key_file == ".secrets/custom_key.txt"
    assert app_config.module_b.llm.temperature == 0.15
    assert app_config.module_b.llm.top_p == 0.85
    assert app_config.module_b.llm.json_retry_times == 4
    assert app_config.module_b.llm.scene_desc_max_chars == 80
    assert app_config.module_b.llm.video_prompt_max_chars == 500


def test_load_config_should_fill_cross_module_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 cross_module 配置缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认值来自代码，不依赖 default.json。
    """
    config_path = tmp_path / "config_cross_module_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.cross_module.global_render_limit == 3
    assert app_config.cross_module.scheduler_tick_ms == 50


def test_load_config_should_accept_explicit_cross_module_values(tmp_path: Path) -> None:
    """
    功能说明：验证 cross_module 配置可显式覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅覆盖跨模块并行字段，不影响其他配置。
    """
    config_path = tmp_path / "config_cross_module_explicit.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "cross_module": {
                    "global_render_limit": 4,
                    "scheduler_tick_ms": 80,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.cross_module.global_render_limit == 4
    assert app_config.cross_module.scheduler_tick_ms == 80


def test_load_config_should_accept_explicit_module_c_config(tmp_path: Path) -> None:
    """
    功能说明：验证 module_c 配置支持显式覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只覆盖 module_c，不影响其他配置默认值。
    """
    config_path = tmp_path / "config_module_c_explicit.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "module_c": {
                    "render_workers": 4,
                    "unit_retry_times": 2,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_c.render_workers == 4
    assert app_config.module_c.unit_retry_times == 2


def test_load_config_should_fill_module_b_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 module_b 配置缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认值来自代码，不依赖 default.json。
    """
    config_path = tmp_path / "config_module_b_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_b.script_workers == 3
    assert app_config.module_b.unit_retry_times == 1


def test_load_config_should_accept_explicit_module_b_config(tmp_path: Path) -> None:
    """
    功能说明：验证 module_b 配置支持显式覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只覆盖 module_b，不影响其他配置默认值。
    """
    config_path = tmp_path / "config_module_b_explicit.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "module_b": {
                    "script_workers": 4,
                    "unit_retry_times": 2,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_b.script_workers == 4
    assert app_config.module_b.unit_retry_times == 2


def test_load_config_should_fill_module_d_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 module_d 配置缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认值来自代码，不依赖 default.json。
    """
    config_path = tmp_path / "config_module_d_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_d.segment_workers == 3
    assert app_config.module_d.unit_retry_times == 1


def test_load_config_should_accept_explicit_module_d_config(tmp_path: Path) -> None:
    """
    功能说明：验证 module_d 配置支持显式覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只覆盖 module_d，不影响其他配置默认值。
    """
    config_path = tmp_path / "config_module_d_explicit.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "module_d": {
                    "segment_workers": 2,
                    "unit_retry_times": 3,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.module_d.segment_workers == 2
    assert app_config.module_d.unit_retry_times == 3


def test_load_config_should_fill_monitoring_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证 monitoring 配置缺省时可自动补齐默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认值来自代码，不依赖 default.json。
    """
    config_path = tmp_path / "config_monitoring_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.monitoring.max_wait_after_terminal_minutes == 20.0


def test_load_config_should_accept_explicit_monitoring_config(tmp_path: Path) -> None:
    """
    功能说明：验证 monitoring 配置支持显式覆盖默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只覆盖 monitoring，不影响其他配置默认值。
    """
    config_path = tmp_path / "config_monitoring_explicit.json"
    config_path.write_text(
        json.dumps(
            {
                "module_a": {
                    "funasr_language": "auto",
                },
                "monitoring": {
                    "max_wait_after_terminal_minutes": 7.5,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    app_config = load_config(config_path=config_path)
    assert app_config.monitoring.max_wait_after_terminal_minutes == 7.5
