"""
文件用途：构建模块B真实LLM分镜生成的系统与用户提示词。
核心流程：将单段输入结构化为JSON文本，拼接严格输出约束。
输入输出：输入分镜上下文字典，输出 chat completions messages 数组。
依赖说明：依赖标准库 json 进行稳定序列化。
维护说明：若输出字段契约变更，需同步更新本文件与解析器。
"""

# 标准库：用于JSON序列化
import json
# 标准库：用于类型提示
from typing import Any


# 常量：模块B LLM 系统提示词，要求输出 scene_desc 与 keyframe/video 中英文双版本。
SYSTEM_PROMPT = (
    "你是音乐视频分镜生成助手。"
    "你的任务是根据给定的单个音乐片段信息，生成适合当前片段的画面描述与提示词。\n"
    "严格遵守以下规则：\n"
    "1. 只返回一个合法 JSON 对象。\n"
    "2. 不要输出任何解释、前后缀、标题、Markdown、代码块。\n"
    "3. 返回 JSON 只能包含五个字段：scene_desc、keyframe_prompt_zh、keyframe_prompt_en、video_prompt_zh、video_prompt_en。\n"
    "4. scene_desc 必须是中文，1 到 2 句话，描述当前片段主体、场景、氛围、情绪、镜头感。\n"
    "5. keyframe_prompt_zh 必须是中文，用于关键帧图像生成，必须具体且有镜头感。\n"
    "6. keyframe_prompt_en 必须是英文，用于关键帧图像生成，必须具体且有镜头感。\n"
    "7. video_prompt_zh 必须是中文，用于文图生视频文本提示，需体现运动与镜头节奏。\n"
    "8. video_prompt_en 必须是英文，用于文图生视频文本提示，需体现运动与镜头节奏。\n"
    "9. 两种语言版本语义必须一致，不允许互相矛盾。\n"
    "10. 英文提示词优先采用逗号分隔短语标签风格，避免空泛句子。\n"
    "11. 中文提示词应精炼、可执行，不要口语化废话。\n"
    "12. 必须结合 lyric_text、lyric_units、segment_label、big_segment_label、energy_level、trend。\n"
    "13. 画面内容约束：除非输入中显式要求，否则尽量避免生成人类、歌手、角色等AIGC复杂元素；尽量避免强调复杂的灯光、光照等容易导致视频前后不一致或场景闪烁的元素。\n"
    "14. camera_motion_rule 和 transition_rule 只作为节奏参考，不要作为输出字段返回。\n"
    "15. 不要改写输入中的时间戳、segment_id、歌词文本与结构信息。\n"
    "16. 输出必须能被 json.loads 直接解析。"
)


def build_module_b_prompt_messages(input_payload: dict[str, Any], retry_hint: str = "") -> list[dict[str, str]]:
    """
    功能说明：构建模块B真实LLM请求 messages。
    参数说明：
    - input_payload: 当前分镜输入上下文字典。
    - retry_hint: 解析失败后的补救提示（可选）。
    返回值：
    - list[dict[str, str]]: chat completions 标准消息数组。
    异常说明：无。
    边界条件：输入字段由上游构建，当前函数不做字段完整性校验。
    """
    payload_text = json.dumps(input_payload, ensure_ascii=False, separators=(",", ":"))
    user_prompt_lines = [
        "请根据以下 JSON 信息，为当前音乐分镜生成结果。请只返回 JSON。",
        payload_text,
        "返回要求：",
        "1. 只返回一个 JSON 对象。",
        "2. 只能包含以下五个字段：scene_desc、keyframe_prompt_zh、keyframe_prompt_en、video_prompt_zh、video_prompt_en。",
        "3. scene_desc 要贴合歌词内容、段落功能、能量等级和趋势。",
        "4. keyframe_prompt_zh 与 keyframe_prompt_en 分别是中文/英文关键帧提示词，语义保持一致。",
        "5. video_prompt_zh 与 video_prompt_en 分别是中文/英文文图生视频提示词，语义保持一致。",
        "6. 英文提示词建议使用逗号分隔短语标签，中文提示词保持简洁可执行。",
        "7. 画面安全约束：除非输入明确要求，否则绝对避免人类/歌手/角色等复杂实体，避免复杂的灯光/光照/光效等易导致画面闪烁的元素。",
        "8. 不要输出任何额外说明，不要输出 Markdown，不要输出代码块，不要输出旧字段 keyframe_prompt/video_prompt。",
    ]
    normalized_retry_hint = str(retry_hint).strip()
    if normalized_retry_hint:
        user_prompt_lines.append(f"补救要求：{normalized_retry_hint}")
    user_prompt = "\n".join(user_prompt_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
