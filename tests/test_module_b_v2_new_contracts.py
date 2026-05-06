"""
文件用途：验证模块B v2 模板、规则与新契约字段的核心行为。
核心流程：加载正式模板、校验新 ModuleBOutput 契约，并检查旧 mock 入口已切到新字段。
输入输出：输入 pytest 测试上下文，输出断言结果。
依赖说明：依赖 pytest 与项目内模块实现。
维护说明：当 camera_plan/transition_plan 契约调整时需同步更新本测试。
"""

# 标准库：用于日志对象。
import logging
# 标准库：用于路径处理。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest

# 项目内模块：模块B配置。
from music_video_pipeline.config import ModuleBConfig
# 项目内模块：分镜生成器。
from music_video_pipeline.generators.script_generator import MockScriptGenerator
# 项目内模块：模块B v2 音频规则。
from music_video_pipeline.modules.module_b_v2.audio_rules import build_segment_audio_features_v2
# 项目内模块：模块B v2 歌词上下文裁剪。
from music_video_pipeline.modules.module_b_v2.lyric_context import (
    build_big_segment_lyric_context,
    build_role3_big_segment_lyric_context,
)
# 项目内模块：模块D转场/运镜辅助。
from music_video_pipeline.modules.module_d.finalizer import _build_camera_filter, _has_nontrivial_transitions, _resolve_xfade_transition
# 项目内模块：模板加载器。
from music_video_pipeline.modules.module_b_v2.template_loader import load_storyboard_template
# 项目内模块：契约校验。
from music_video_pipeline.types import validate_module_b_output


def test_storyboard_template_v1_should_load_and_cover_9x9_rules() -> None:
    """
    功能说明：验证正式模板文件可被加载，且包含 9 条运镜映射与 9 条转场映射。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模板路径固定为 configs/prompts/storyboard_template.v1.md。
    """
    project_root = Path(__file__).resolve().parents[1]
    template = load_storyboard_template(project_root=project_root)
    assert template["template_id"] == "storyboard_template_v1_monochrome_cat_hide_seek"
    assert len(template["composition_catalog"]) == 9
    assert len(template["camera_mapping"]) == 9
    assert len(template["transition_mapping"]) == 9
    assert any(item["preset_id"] == "zoom_in_s" for item in template["camera_plan_presets"])
    assert any(item["preset_id"] == "wipe_left_200" for item in template["transition_presets"])


def test_build_segment_audio_features_v2_should_produce_candidate_plans() -> None:
    """
    功能说明：验证规则层会为每个 segment 补齐增强字段与候选运镜/转场计划。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：按大段中位分布离散 tension_band。
    """
    project_root = Path(__file__).resolve().parents[1]
    template = load_storyboard_template(project_root=project_root)
    module_a_output = {
        "segments": [
            {"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"},
            {"segment_id": "seg_002", "big_segment_id": "big_001", "start_time": 1.0, "end_time": 2.0, "label": "verse"},
        ],
        "energy_features": [
            {"start_time": 0.0, "end_time": 1.0, "energy_level": "low", "trend": "up", "rhythm_tension": 0.2},
            {"start_time": 1.0, "end_time": 2.0, "energy_level": "high", "trend": "down", "rhythm_tension": 0.8},
        ],
    }
    result = build_segment_audio_features_v2(module_a_output=module_a_output, storyboard_template=template)
    assert result["seg_001"]["segment_rank_in_big_segment"] == 1
    assert result["seg_001"]["segment_count_in_big_segment"] == 2
    assert result["seg_001"]["default_camera_plan"]["preset_id"] == "zoom_in_s"
    assert result["seg_001"]["camera_plan_candidates"][0]["preset_id"] == "none"
    assert result["seg_002"]["default_transition_plan"]["preset_id"] in {"fade_black_240", "none", "hard_cut_0", "crossfade_160"}


def test_lyric_context_should_strip_token_level_and_mount_tree_fields() -> None:
    """
    功能说明：验证角色2/角色3歌词上下文只保留裁剪后的摘要，不再透传 token 级或整棵挂载树。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：角色3仍需保留按 segment 的歌词挂载概览，但不保留时间戳与 confidence。
    """
    module_a_output = {
        "big_segments": [
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"},
        ],
        "segments": [
            {"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
            {"segment_id": "seg_002", "big_segment_id": "big_001", "start_time": 2.0, "end_time": 4.0, "label": "verse"},
        ],
        "lyric_units": [
            {
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "第一句歌词",
                "confidence": 0.9,
                "token_units": [{"text": "第", "start_time": 0.0, "end_time": 0.1}],
            },
            {
                "segment_id": "seg_002",
                "start_time": 2.0,
                "end_time": 3.0,
                "text": "第二句歌词",
                "confidence": 0.8,
                "token_units": [{"text": "二", "start_time": 2.0, "end_time": 2.1}],
            },
        ],
    }

    role2_context = build_big_segment_lyric_context(module_a_output=module_a_output)
    assert role2_context == [
        {
            "big_segment_id": "big_001",
            "lyric_line_count": 2,
            "lyric_excerpt": "第一句歌词 / 第二句歌词",
        }
    ]

    role3_context = build_role3_big_segment_lyric_context(module_a_output=module_a_output)
    assert role3_context["big_001"]["lyric_excerpt"] == "第一句歌词 / 第二句歌词"
    assert role3_context["big_001"]["segment_lyrics"] == [
        {"segment_id": "seg_001", "lyric_count": 1, "lyric_lines": ["第一句歌词"]},
        {"segment_id": "seg_002", "lyric_count": 1, "lyric_lines": ["第二句歌词"]},
    ]


def test_validate_module_b_output_should_require_camera_plan_and_transition_plan() -> None:
    """
    功能说明：验证新 ModuleBOutput 契约要求 camera_plan/transition_plan，且拒绝旧字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧字段 camera_motion/transition 出现即判非法。
    """
    valid_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.2,
            "scene_desc": "默认场景",
            "keyframe_prompt_start_zh": "关键帧起始中文",
            "keyframe_prompt_start_en": "keyframe start en",
            "keyframe_negative_prompt_start_zh": "负面起始中文",
            "keyframe_negative_prompt_start_en": "negative start en",
            "keyframe_prompt_end_zh": "关键帧结束中文",
            "keyframe_prompt_end_en": "keyframe end en",
            "keyframe_negative_prompt_end_zh": "负面结束中文",
            "keyframe_negative_prompt_end_en": "negative end en",
            "video_prompt_zh": "视频提示词中文",
            "video_prompt_en": "video prompt en",
            "camera_plan": {
                "preset_id": "zoom_in_s",
                "mode": "zoom",
                "direction": "center",
                "strength": "small",
                "easing": "ease_in_out",
            },
            "transition_plan": {
                "preset_id": "crossfade_160",
                "kind": "crossfade",
                "duration_ms": 160,
                "easing": "ease_in_out",
            },
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
        }
    ]
    validate_module_b_output(valid_output)

    invalid_output = [
        {
            **valid_output[0],
            "camera_motion": "slow_pan",
        }
    ]
    with pytest.raises(KeyError):
        validate_module_b_output(invalid_output)


def test_mock_script_generator_should_emit_new_plan_fields() -> None:
    """
    功能说明：验证旧 mock 分镜入口已经切到新 plan 字段，而不是旧 camera_motion/transition。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该行为保证旧模式仍可进入模块 C/D 新链路。
    """
    generator = MockScriptGenerator()
    shot = generator.generate_one(
        module_a_output={
            "big_segments": [{"segment_id": "big_001", "label": "verse"}],
            "segments": [{"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"}],
            "energy_features": [{"energy_level": "mid", "trend": "flat"}],
            "lyric_units": [],
        },
        segment={"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
        segment_index=0,
    )
    assert "camera_plan" in shot
    assert "transition_plan" in shot
    assert "camera_motion" not in shot
    assert "transition" not in shot
    validate_module_b_output([shot])


def test_multi_role_mode_should_be_removed_from_legacy_factory() -> None:
    """
    功能说明：验证新模式名 multi_role_llm_v2 已从旧 ScriptGenerator 工厂摘除。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：真正入口由 module_b.orchestrator 直接路由到 run_module_b_v2。
    """
    from music_video_pipeline.generators.script_generator import build_script_generator

    with pytest.raises(ValueError, match="run_module_b_v2"):
        build_script_generator(
            mode="multi_role_llm_v2",
            logger=logging.getLogger("test_module_b_v2_new_contracts"),
            module_b_config=ModuleBConfig(),
        )


def test_module_d_transition_and_camera_helpers_should_match_new_schema() -> None:
    """
    功能说明：验证模块D对新 camera_plan/transition_plan 的基础辅助逻辑可用。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：此处只测字符串/枚举级逻辑，不执行真实 ffmpeg。
    """
    assert _has_nontrivial_transitions(
        [{"kind": "none"}, {"kind": "crossfade", "duration_ms": 160, "easing": "ease_in_out"}]
    )
    transition_name, duration_seconds = _resolve_xfade_transition(
        {"kind": "wipe_left", "duration_ms": 200, "easing": "ease_in_out"}
    )
    assert transition_name == "wipeleft"
    assert duration_seconds == pytest.approx(0.2, rel=1e-6)

    filter_text = _build_camera_filter(
        width=848,
        height=480,
        duration=2.0,
        camera_plan={
            "preset_id": "pan_ur_s",
            "mode": "pan",
            "direction": "up_right",
            "strength": "small",
            "easing": "ease_in_out",
        },
    )
    assert "scale=" in filter_text
    assert "crop=" in filter_text
