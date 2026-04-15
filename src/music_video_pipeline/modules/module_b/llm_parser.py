"""
文件用途：解析并校验模块B真实LLM返回内容。
核心流程：提取首个JSON对象 -> 字段白名单校验 -> 字符串与长度校验。
输入输出：输入LLM文本输出，输出标准化的双语提示词字典。
依赖说明：依赖标准库 json 实现解析。
维护说明：字段契约变更时需同步更新本文件与提示词。
"""

# 标准库：用于JSON解码
import json


class ModuleBLlmParseError(ValueError):
    """模块B LLM 解析失败异常。"""


def parse_module_b_llm_output(
    llm_output_text: str,
    scene_desc_max_chars: int,
    keyframe_prompt_max_chars: int,
    video_prompt_max_chars: int,
) -> dict[str, str]:
    """
    功能说明：解析并校验模块B LLM返回文本。
    参数说明：
    - llm_output_text: 模型返回原始文本。
    - scene_desc_max_chars: scene_desc 最大字符数。
    - keyframe_prompt_max_chars: keyframe_prompt 最大字符数。
    - video_prompt_max_chars: video_prompt 最大字符数。
    返回值：
    - dict[str, str]: 标准化后的 scene_desc 与 keyframe/video 中英文提示词。
    异常说明：
    - ModuleBLlmParseError: 解析失败或字段非法时抛出。
    边界条件：允许JSON对象前存在噪声文本，但只提取首个对象。
    """
    json_obj = _extract_first_json_object(llm_output_text=llm_output_text)
    expected_keys = {
        "scene_desc",
        "keyframe_prompt_zh",
        "keyframe_prompt_en",
        "video_prompt_zh",
        "video_prompt_en",
    }
    actual_keys = set(json_obj.keys())
    if actual_keys != expected_keys:
        raise ModuleBLlmParseError(
            "模块B LLM 输出字段不匹配："
            f"期望={sorted(expected_keys)}，实际={sorted(actual_keys)}"
        )

    scene_desc = _normalize_non_empty_text(field_name="scene_desc", value=json_obj["scene_desc"])
    keyframe_prompt_zh = _normalize_non_empty_text(field_name="keyframe_prompt_zh", value=json_obj["keyframe_prompt_zh"])
    keyframe_prompt_en = _normalize_non_empty_text(field_name="keyframe_prompt_en", value=json_obj["keyframe_prompt_en"])
    video_prompt_zh = _normalize_non_empty_text(field_name="video_prompt_zh", value=json_obj["video_prompt_zh"])
    video_prompt_en = _normalize_non_empty_text(field_name="video_prompt_en", value=json_obj["video_prompt_en"])

    if len(scene_desc) > int(scene_desc_max_chars):
        raise ModuleBLlmParseError(
            f"模块B LLM 输出过长：scene_desc 长度={len(scene_desc)}，上限={int(scene_desc_max_chars)}"
        )
    if len(keyframe_prompt_zh) > int(keyframe_prompt_max_chars):
        raise ModuleBLlmParseError(
            "模块B LLM 输出过长："
            f"keyframe_prompt_zh 长度={len(keyframe_prompt_zh)}，上限={int(keyframe_prompt_max_chars)}"
        )
    if len(keyframe_prompt_en) > int(keyframe_prompt_max_chars):
        raise ModuleBLlmParseError(
            "模块B LLM 输出过长："
            f"keyframe_prompt_en 长度={len(keyframe_prompt_en)}，上限={int(keyframe_prompt_max_chars)}"
        )
    if len(video_prompt_zh) > int(video_prompt_max_chars):
        raise ModuleBLlmParseError(
            f"模块B LLM 输出过长：video_prompt_zh 长度={len(video_prompt_zh)}，上限={int(video_prompt_max_chars)}"
        )
    if len(video_prompt_en) > int(video_prompt_max_chars):
        raise ModuleBLlmParseError(
            f"模块B LLM 输出过长：video_prompt_en 长度={len(video_prompt_en)}，上限={int(video_prompt_max_chars)}"
        )

    return {
        "scene_desc": scene_desc,
        "keyframe_prompt_zh": keyframe_prompt_zh,
        "keyframe_prompt_en": keyframe_prompt_en,
        "video_prompt_zh": video_prompt_zh,
        "video_prompt_en": video_prompt_en,
        # 兼容字段：保持下游当前契约不变，默认使用英文版本。
        "keyframe_prompt": keyframe_prompt_en,
        "video_prompt": video_prompt_en,
    }


def _extract_first_json_object(llm_output_text: str) -> dict:
    """
    功能说明：从LLM文本中提取首个JSON对象。
    参数说明：
    - llm_output_text: 模型输出文本。
    返回值：
    - dict: 解析出的JSON对象。
    异常说明：
    - ModuleBLlmParseError: 未找到对象或解析失败时抛出。
    边界条件：从首个 "{" 开始尝试 raw_decode，忽略尾部噪声。
    """
    raw_text = str(llm_output_text or "")
    start_index = raw_text.find("{")
    if start_index < 0:
        raise ModuleBLlmParseError("模块B LLM 输出中未找到JSON对象起始符号 '{'。")

    decoder = json.JSONDecoder()
    try:
        parsed_obj, _ = decoder.raw_decode(raw_text[start_index:])
    except json.JSONDecodeError as error:
        raise ModuleBLlmParseError(f"模块B LLM JSON解析失败：{error}") from error

    if not isinstance(parsed_obj, dict):
        raise ModuleBLlmParseError("模块B LLM 输出不是JSON对象。")
    return parsed_obj


def _normalize_non_empty_text(field_name: str, value: object) -> str:
    """
    功能说明：校验并标准化非空文本字段。
    参数说明：
    - field_name: 字段名。
    - value: 字段值。
    返回值：
    - str: 去首尾空白后的文本。
    异常说明：
    - ModuleBLlmParseError: 类型错误或空文本时抛出。
    边界条件：仅接受字符串，其他类型直接报错。
    """
    if not isinstance(value, str):
        raise ModuleBLlmParseError(f"模块B LLM 字段类型非法：{field_name} 必须是字符串。")
    normalized = value.strip()
    if not normalized:
        raise ModuleBLlmParseError(f"模块B LLM 字段为空：{field_name}。")
    return normalized
