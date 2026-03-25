"""
文件用途：验证模块 A/B 输出契约最低字段。
核心流程：在临时目录执行模块函数并检查输出结构。
输入输出：输入测试上下文，输出契约断言结果。
依赖说明：依赖 pytest 与项目内模块实现。
维护说明：契约字段变更时需同步更新测试断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：目录工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块实现
from music_video_pipeline.modules.module_a import run_module_a
# 项目内模块：模块实现
from music_video_pipeline.modules.module_b import run_module_b
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def test_module_a_and_b_outputs_should_match_contracts(tmp_path: Path) -> None:
    """
    功能说明：执行模块 A/B 并验证输出满足最低契约。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入音频为临时占位文件，仅用于流程验证。
    """
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake-audio-content")

    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("contract_test")
    logger.setLevel(logging.INFO)
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")

    context = RuntimeContext(
        task_id="contract_task",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "contract_task",
        artifacts_dir=tmp_path / "runs" / "contract_task" / "artifacts",
        config=config,
        logger=logger,
        state_store=state_store,
    )
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)

    module_a_path = run_module_a(context)
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)
    assert module_a_output["task_id"] == "contract_task"
    assert isinstance(module_a_output["big_segments"], list)
    assert len(module_a_output["big_segments"]) > 0
    assert isinstance(module_a_output["segments"], list)
    assert len(module_a_output["segments"]) > 0
    assert "big_segment_id" in module_a_output["segments"][0]

    module_b_path = run_module_b(context)
    module_b_output = read_json(module_b_path)
    validate_module_b_output(module_b_output)
    assert len(module_b_output) == len(module_a_output["segments"])
    assert "lyric_text" in module_b_output[0]
    assert "lyric_units" in module_b_output[0]
    assert isinstance(module_b_output[0]["lyric_text"], str)
    assert isinstance(module_b_output[0]["lyric_units"], list)
    assert "big_segment_id" in module_b_output[0]
    assert "big_segment_label" in module_b_output[0]
    assert "segment_label" in module_b_output[0]
    assert "audio_role" in module_b_output[0]

    instrumental_set = {item.lower() for item in config.module_a.instrumental_labels}
    for shot in module_b_output:
        assert isinstance(shot["big_segment_id"], str)
        assert isinstance(shot["big_segment_label"], str)
        assert isinstance(shot["segment_label"], str)
        assert shot["audio_role"] in {"instrumental", "vocal"}
        if str(shot["segment_label"]).lower() in instrumental_set:
            assert shot["audio_role"] == "instrumental"
        else:
            assert shot["audio_role"] == "vocal"


def test_validate_module_b_output_should_be_forward_compatible_for_legacy_items() -> None:
    """
    功能说明：验证模块B契约校验兼容旧版无歌词字段分镜。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅保留最低必填字段。
    """
    legacy_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "image_prompt": "default prompt",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
        }
    ]
    validate_module_b_output(legacy_output)


def test_validate_module_b_output_should_validate_lyrics_fields() -> None:
    """
    功能说明：验证模块B契约校验可识别歌词扩展字段的合法性。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：lyric_text 非字符串时应抛出 TypeError。
    """
    enhanced_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "image_prompt": "default prompt",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "lyric_text": "第一句 第二句",
            "lyric_units": [
                {"start_time": 0.1, "end_time": 0.6, "text": "第一句", "confidence": 0.9},
                {"start_time": 0.7, "end_time": 1.1, "text": "第二句", "confidence": 0.8},
            ],
            "big_segment_id": "big_001",
            "big_segment_label": "verse",
            "segment_label": "verse",
            "audio_role": "vocal",
        }
    ]
    validate_module_b_output(enhanced_output)

    invalid_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "image_prompt": "default prompt",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "lyric_text": 123,
        }
    ]
    with pytest.raises(TypeError):
        validate_module_b_output(invalid_output)

    invalid_audio_role_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "image_prompt": "default prompt",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "audio_role": "unknown",
        }
    ]
    with pytest.raises(ValueError):
        validate_module_b_output(invalid_audio_role_output)

    invalid_big_segment_id_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "image_prompt": "default prompt",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "big_segment_id": 123,
        }
    ]
    with pytest.raises(TypeError):
        validate_module_b_output(invalid_big_segment_id_output)


def _build_test_config(tmp_path: Path) -> AppConfig:
    """
    功能说明：构建用于测试的最小配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - AppConfig: 测试配置对象。
    异常说明：无。
    边界条件：ffmpeg 配置在本测试中不会被实际调用。
    """
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path="input.mp3"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        mock=MockConfig(beat_interval_seconds=0.5, video_width=640, video_height=360),
        module_a=ModuleAConfig(whisper_language="auto", mode="fallback_only"),
    )
