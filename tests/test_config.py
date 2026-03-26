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
        assert app_config.module_a.lyric_segment_policy == "sentence_strict"


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


def test_load_config_should_accept_adaptive_lyric_segment_policy(tmp_path: Path) -> None:
    """
    功能说明：验证 module_a.lyric_segment_policy 可显式设置为 adaptive_phrase。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：阈值字段缺失时应走默认值。
    """
    config_path = tmp_path / "config_with_adaptive_policy.json"
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
    app_config = load_config(config_path=config_path)
    assert app_config.module_a.lyric_segment_policy == "adaptive_phrase"
    assert app_config.module_a.comma_pause_seconds == 0.45
    assert app_config.module_a.long_pause_seconds == 0.8
    assert app_config.module_a.merge_gap_seconds == 0.25
    assert app_config.module_a.max_visual_unit_seconds == 6.0
