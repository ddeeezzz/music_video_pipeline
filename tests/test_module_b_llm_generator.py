"""
文件用途：验证模块B LLM分镜生成器的双语提示词回填与失败行为。
核心流程：打桩真实LLM调用函数，检查 generate_one 返回结构。
输入输出：输入伪造模块A与segment数据，输出shot断言结果。
依赖说明：依赖 pytest 与 LlmScriptGenerator。
维护说明：若分镜字段契约调整，需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging

# 第三方库：用于异常断言
import pytest

# 项目内模块：模块B配置类型
from music_video_pipeline.config import ModuleBConfig
# 项目内模块：模块B LLM生成异常类型
from music_video_pipeline.modules.module_b.llm_generator import ModuleBLlmGenerationError
# 项目内模块：分镜生成器
from music_video_pipeline.generators import script_generator as script_generator_module


def test_llm_script_generator_should_fill_bilingual_prompt_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能说明：验证LLM分镜生成器可回填 scene_desc 与 keyframe/video 的中英文字段。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：camera_motion/transition 仍由规则逻辑生成。
    """

    def _fake_generate_module_b_prompts(logger, llm_config, llm_input_payload, project_root):
        _ = (logger, llm_config, project_root)
        assert llm_input_payload["segment_id"] == "seg_0001"
        return {
            "scene_desc": "雨夜街口，人物在霓虹下停步回望，镜头缓慢推进。",
            "keyframe_prompt_zh": "电影感关键帧，雨夜霓虹街道，孤立主体，中景，构图稳定",
            "keyframe_prompt_en": "cinematic keyframe, rainy neon street, lone figure, soft rim light, medium shot, film still",
            "video_prompt_zh": "电影感视频提示词，雨夜街道，慢速推进，风雨细微运动，忧郁氛围",
            "video_prompt_en": "cinematic video prompt, rainy neon street, slow push-in, gentle wind and rain motion, melancholic mood",
            "keyframe_prompt": "cinematic keyframe, rainy neon street, lone figure, soft rim light, medium shot, film still",
            "video_prompt": "cinematic video prompt, rainy neon street, slow push-in, gentle wind and rain motion, melancholic mood",
        }

    monkeypatch.setattr(script_generator_module, "generate_module_b_prompts", _fake_generate_module_b_prompts)
    generator = script_generator_module.LlmScriptGenerator(
        logger=logging.getLogger("test_llm_script_generator"),
        module_b_config=ModuleBConfig(),
    )

    shot = generator.generate_one(
        module_a_output={
            "big_segments": [{"segment_id": "big_001", "label": "verse"}],
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "big_segment_id": "big_001",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "label": "verse",
                }
            ],
            "energy_features": [{"energy_level": "mid", "trend": "flat"}],
            "lyric_units": [{"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.5, "text": "第一句", "confidence": 0.8}],
        },
        segment={
            "segment_id": "seg_0001",
            "big_segment_id": "big_001",
            "start_time": 0.0,
            "end_time": 2.0,
            "label": "verse",
        },
        segment_index=0,
    )

    assert shot["scene_desc"].startswith("雨夜街口")
    assert "雨夜霓虹街道" in shot["keyframe_prompt_zh"]
    assert "rainy neon street" in shot["keyframe_prompt_en"]
    assert "雨夜街道" in shot["video_prompt_zh"]
    assert "rainy neon street" in shot["video_prompt_en"]
    # 兼容字段默认指向英文版本。
    assert "neon street" in shot["keyframe_prompt"]
    assert "video prompt" in shot["video_prompt"]
    assert shot["camera_motion"] in {"none", "slow_pan", "zoom_in", "shake", "push_pull"}
    assert shot["transition"] in {"crossfade", "hard_cut"}


def test_llm_script_generator_should_raise_when_llm_generation_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能说明：验证LLM双语提示词生成失败时会抛出可定位异常。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：错误信息应包含 segment_id，便于模块B单元排障。
    """

    def _fake_generate_module_b_prompts(logger, llm_config, llm_input_payload, project_root):
        _ = (logger, llm_config, llm_input_payload, project_root)
        raise ModuleBLlmGenerationError("mock llm failed")

    monkeypatch.setattr(script_generator_module, "generate_module_b_prompts", _fake_generate_module_b_prompts)
    generator = script_generator_module.LlmScriptGenerator(
        logger=logging.getLogger("test_llm_script_generator_fail"),
        module_b_config=ModuleBConfig(),
    )

    with pytest.raises(RuntimeError, match="segment_id=seg_0001"):
        generator.generate_one(
            module_a_output={
                "big_segments": [{"segment_id": "big_001", "label": "verse"}],
                "segments": [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"}],
                "energy_features": [{"energy_level": "mid", "trend": "flat"}],
                "lyric_units": [],
            },
            segment={
                "segment_id": "seg_0001",
                "big_segment_id": "big_001",
                "start_time": 0.0,
                "end_time": 1.0,
                "label": "verse",
            },
            segment_index=0,
        )
