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
# 项目内模块：模块B v2 解析与校验。
from music_video_pipeline.modules.module_b_v2.parser import (
    validate_role1_visual_catalog_output,
    validate_role2_big_segment_story_output,
    validate_role3_segment_directing_output,
    validate_role4_prompt_output,
)
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


def test_storyboard_template_v1_should_load_and_cover_current_catalogs() -> None:
    """
    功能说明：验证正式模板文件可被加载，且包含当前最小构图库与计划预设集。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模板路径固定为 configs/storyboard_templates/storyboard_template.v1.md。
    """
    project_root = Path(__file__).resolve().parents[1]
    template = load_storyboard_template(project_root=project_root)
    assert template["template_id"] == "storyboard_template_v1_monochrome_cat_hide_seek"
    assert len(template["composition_catalog"]) == 3
    assert len(template["camera_plan_presets"]) == 4
    assert len(template["transition_presets"]) == 8
    assert any(item["preset_id"] == "zoom_in_s" for item in template["camera_plan_presets"])
    assert any(item["preset_id"] == "wipe_left_200" for item in template["transition_presets"])


def test_build_segment_audio_features_v2_should_produce_audio_semantics() -> None:
    """
    功能说明：验证规则层会为每个 segment 补齐增强音频语义字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前规则层只产出确定性音频语义，不直接产出运镜/转场候选。
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
    assert result["seg_001"]["energy_level"] == "low"
    assert result["seg_001"]["tension_band"] == "low"
    assert result["seg_001"]["position_in_big_segment"] == "start"
    assert result["seg_002"]["energy_level"] == "high"
    assert result["seg_002"]["tension_band"] == "high"
    assert result["seg_002"]["tension_delta"] == "up"
    assert result["seg_002"]["is_local_peak"] is True


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


def test_role2_should_accept_none_ids_and_normalize_to_empty_lists() -> None:
    """
    功能说明：验证角色2允许 scene/character/prop 使用 none，并标准化为空数组。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：none 语义仅在角色输出层生效，不要求真实目录包含 none。
    """
    result = validate_role2_big_segment_story_output(
        data={
            "big_segments": [
                {
                    "big_segment_id": "big_001",
                    "title_zh": "空镜停顿",
                    "story_outline_zh": "画面暂时留白，节奏停顿后再推进。",
                    "selected_scene_ids": [],
                    "selected_character_ids": [],
                    "selected_prop_ids": [],
                }
            ]
        },
        big_segment_ids=["big_001"],
        scene_ids=["scene_alley_dim"],
        prop_ids=["prop_cage_wire"],
        character_ids=["character_black_cat"],
    )
    assert result["big_segments"][0]["selected_scene_ids"] == []
    assert result["big_segments"][0]["selected_character_ids"] == []
    assert result["big_segments"][0]["selected_prop_ids"] == []


def test_role3_should_accept_none_scene_and_resolve_preset_ids() -> None:
    """
    功能说明：验证角色3允许空场景，并使用 preset_id 回填完整运镜/转场计划。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：selected_scene_id 为空时标准化为 ""。
    """
    result = validate_role3_segment_directing_output(
        data={
            "shots": [
                {
                    "shot_id": "shot_001",
                    "scene_desc_zh": "黑猫停在留白里，短暂停顿后轻微抬头。",
                    "selected_scene_id": "",
                    "selected_character_ids": [],
                    "selected_prop_ids": [],
                    "composition_id": "comp_sym_center",
                    "camera_plan_preset_id": "zoom_in_s",
                    "transition_plan_preset_id": "crossfade_160",
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
                    "motion_delta_label": "tiny",
                    "motion_speed_label": "slow",
                    "composition_stability": "stable",
                }
            ]
        },
        shot_ids=["shot_001"],
        scene_ids=["scene_alley_dim"],
        prop_ids=["prop_cage_wire"],
        character_ids=["character_black_cat"],
        composition_ids=["comp_sym_center"],
    )
    assert result["shots"][0]["selected_scene_id"] == ""
    assert result["shots"][0]["camera_plan_preset_id"] == "zoom_in_s"
    assert result["shots"][0]["transition_plan_preset_id"] == "crossfade_160"


def test_role1_should_normalize_tokens_and_merge_negative_template() -> None:
    """
    功能说明：验证角色1会将宽松 tag 文本标准化为 token，并把负面增量与固定模板合并。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：正向英文 token 超过上限时应被截断到 18 个以内。
    """
    long_pos_en = ", ".join([f"tag_{index:02d}" for index in range(1, 25)])
    result = validate_role1_visual_catalog_output(
        data={
            "scene_refs": [
                {
                    "item_id": "scene_alley_dim",
                    "refs": [
                        {
                            "ref_id": "ref_1",
                            "pos_zh": "黑白, 单色, 小巷, 潮湿地面, 斑驳砖墙",
                            "pos_en": long_pos_en,
                            "neg_zh": "额外人物，彩色污染",
                            "neg_en": "extra people, color contamination",
                        }
                    ],
                }
            ],
            "prop_refs": [],
            "character_refs": [],
        },
        scene_ids=["scene_alley_dim"],
        prop_ids=[],
        character_ids=[],
    )
    ref_item = result["scene_refs"][0]["refs"][0]
    assert len(ref_item["pos_tokens_en"]) <= 18
    assert any(str(item.get("text", "")) == "black and white" for item in ref_item["pos_tokens_en"])
    assert any("extra people" == str(item.get("text", "")) for item in ref_item["neg_tokens_en_increment"])
    assert "realistic" in ref_item["neg_en"]


def test_role4_should_compile_tokens_and_merge_fixed_negative_prompt() -> None:
    """
    功能说明：验证角色4会把宽松 tag 文本编译为 token，并自动拼入固定风格与固定负面模板。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：style_color_mode/style_render_style 作为内部上下文输入，不属于最终公开字段要求。
    """
    result = validate_role4_prompt_output(
        data={
            "shots": [
                {
                    "shot_id": "shot_001",
                    "scene_desc": "黑猫停住后轻轻抬头。",
                    "keyframe_prompt_start_zh": "黑猫, 窗边, 留白",
                    "keyframe_prompt_start_en": "black cat, windowsill, negative space",
                    "keyframe_negative_prompt_start_zh": "彩色污染, 额外人物",
                    "keyframe_negative_prompt_start_en": "color contamination, extra people",
                    "keyframe_prompt_end_zh": "黑猫抬头, 窗边, 留白",
                    "keyframe_prompt_end_en": "black cat raises head, windowsill, negative space",
                    "keyframe_negative_prompt_end_zh": "彩色污染, 额外人物",
                    "keyframe_negative_prompt_end_en": "color contamination, extra people",
                    "video_prompt_zh": "黑猫慢慢抬头, 构图稳定",
                    "video_prompt_en": "black cat slowly raises head, stable composition",
                    "style_color_mode": "黑白",
                    "style_render_style": "日本漫画风插图",
                }
            ]
        },
        shot_ids=["shot_001"],
    )
    item = result["shots"][0]
    assert any(str(token.get("text", "")) == "black and white" for token in item["keyframe_prompt_start_tokens_en"])
    assert any(str(token.get("text", "")) == "anime limited animation" for token in item["video_prompt_tokens_en"])
    assert "realistic" in item["keyframe_negative_prompt_start_en"]
    assert any(str(token.get("text", "")) == "extra people" for token in item["keyframe_negative_prompt_start_tokens_en_increment"])


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
