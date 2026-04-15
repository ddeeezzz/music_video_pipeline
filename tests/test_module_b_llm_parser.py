"""
文件用途：验证模块B真实LLM返回解析器的稳健性与字段校验。
核心流程：构造合法与非法输出样本，断言 parse_module_b_llm_output 行为。
输入输出：输入LLM文本，输出解析结果或异常。
依赖说明：依赖 pytest 与模块B LLM解析器实现。
维护说明：若字段契约调整，需同步更新本测试。
"""

# 第三方库：用于异常断言
import pytest

# 项目内模块：模块B LLM解析器
from music_video_pipeline.modules.module_b.llm_parser import ModuleBLlmParseError, parse_module_b_llm_output


def test_parse_module_b_llm_output_should_parse_json_with_noise_prefix() -> None:
    """
    功能说明：验证解析器可从前置噪声中提取首个JSON对象。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：尾部附加文本不应影响首对象解析。
    """
    raw_text = (
        "noise\n"
        "{\"scene_desc\":\"中文描述\","
        "\"keyframe_prompt_zh\":\"中文关键帧提示词\","
        "\"keyframe_prompt_en\":\"key prompt\","
        "\"video_prompt_zh\":\"中文视频提示词\","
        "\"video_prompt_en\":\"video prompt\"}\n"
        "trailing"
    )
    parsed = parse_module_b_llm_output(
        llm_output_text=raw_text,
        scene_desc_max_chars=120,
        keyframe_prompt_max_chars=400,
        video_prompt_max_chars=500,
    )
    assert parsed["scene_desc"] == "中文描述"
    assert parsed["keyframe_prompt_zh"] == "中文关键帧提示词"
    assert parsed["keyframe_prompt_en"] == "key prompt"
    assert parsed["video_prompt_zh"] == "中文视频提示词"
    assert parsed["video_prompt_en"] == "video prompt"
    # 兼容字段默认映射到英文版本。
    assert parsed["keyframe_prompt"] == "key prompt"
    assert parsed["video_prompt"] == "video prompt"


def test_parse_module_b_llm_output_should_reject_extra_keys() -> None:
    """
    功能说明：验证解析器会拒绝超出白名单的字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：字段集合必须与契约完全一致。
    """
    raw_text = (
        "{"
        "\"scene_desc\":\"中文描述\","
        "\"keyframe_prompt_zh\":\"中文关键帧提示词\","
        "\"keyframe_prompt_en\":\"key prompt\","
        "\"video_prompt_zh\":\"中文视频提示词\","
        "\"video_prompt_en\":\"video prompt\","
        "\"extra\":\"x\""
        "}"
    )
    with pytest.raises(ModuleBLlmParseError, match="字段不匹配"):
        parse_module_b_llm_output(
            llm_output_text=raw_text,
            scene_desc_max_chars=120,
            keyframe_prompt_max_chars=400,
            video_prompt_max_chars=500,
        )


def test_parse_module_b_llm_output_should_reject_empty_field() -> None:
    """
    功能说明：验证解析器会拒绝空字符串字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：scene_desc 与 keyframe/video 的中英文字段均不可为空。
    """
    raw_text = (
        "{"
        "\"scene_desc\":\" \","
        "\"keyframe_prompt_zh\":\"中文关键帧提示词\","
        "\"keyframe_prompt_en\":\"key prompt\","
        "\"video_prompt_zh\":\"中文视频提示词\","
        "\"video_prompt_en\":\"video prompt\""
        "}"
    )
    with pytest.raises(ModuleBLlmParseError, match="字段为空"):
        parse_module_b_llm_output(
            llm_output_text=raw_text,
            scene_desc_max_chars=120,
            keyframe_prompt_max_chars=400,
            video_prompt_max_chars=500,
        )


def test_parse_module_b_llm_output_should_reject_length_overflow() -> None:
    """
    功能说明：验证解析器会拒绝超过长度上限的字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用极小上限触发越界路径。
    """
    raw_text = (
        "{"
        "\"scene_desc\":\"中文描述\","
        "\"keyframe_prompt_zh\":\"中文关键帧提示词\","
        "\"keyframe_prompt_en\":\"key prompt\","
        "\"video_prompt_zh\":\"中文视频提示词\","
        "\"video_prompt_en\":\"video prompt\""
        "}"
    )
    with pytest.raises(ModuleBLlmParseError, match="video_prompt_"):
        parse_module_b_llm_output(
            llm_output_text=raw_text,
            scene_desc_max_chars=120,
            keyframe_prompt_max_chars=400,
            video_prompt_max_chars=5,
        )
