"""
文件用途：整理模块B v2 角色2/角色3使用的歌词挂载上下文。
核心流程：读取模块A的 segments/lyric_units -> 生成角色级裁剪后的歌词摘要载荷。
输入输出：输入模块A输出，输出可直接喂给角色编导的结构化歌词上下文字典。
依赖说明：依赖标准库 typing。
维护说明：歌词在本链路中只作为情感与节奏参考，不作为具体视觉意象来源。
"""

# 标准库：用于类型提示。
from typing import Any


# 常量：角色2大段歌词摘要的最大字符数，避免上下文过长。
ROLE2_LYRIC_EXCERPT_MAX_CHARS = 48
# 常量：角色3大段歌词摘要的最大字符数，供镜头编排理解段落语气。
ROLE3_LYRIC_EXCERPT_MAX_CHARS = 72


def build_big_segment_lyric_context(module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
    """
    功能说明：构建角色2使用的“大段歌词摘要上下文”。
    参数说明：
    - module_a_output: 模块A输出。
    返回值：
    - list[dict[str, Any]]: 按大段组织的短歌词摘要。
    异常说明：无。
    边界条件：无歌词时仍返回各大段的空挂载结构，便于模型理解“此段无人声”。
    """
    big_segments = [dict(item) for item in module_a_output.get("big_segments", []) if isinstance(item, dict)]
    segments = [dict(item) for item in module_a_output.get("segments", []) if isinstance(item, dict)]
    lyric_lines_by_segment = _build_segment_lyric_lines_map(module_a_output=module_a_output)

    segments_by_big_segment: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        big_segment_id = str(segment.get("big_segment_id", "")).strip()
        if big_segment_id:
            segments_by_big_segment.setdefault(big_segment_id, []).append(segment)

    results: list[dict[str, Any]] = []
    for big_segment in big_segments:
        big_segment_id = str(big_segment.get("segment_id", "")).strip()
        segment_items = sorted(
            segments_by_big_segment.get(big_segment_id, []),
            key=lambda item: float(item.get("start_time", 0.0)),
        )
        all_lyric_lines: list[str] = []
        for segment in segment_items:
            segment_id = str(segment.get("segment_id", "")).strip()
            lyric_text_lines = list(lyric_lines_by_segment.get(segment_id, []))
            all_lyric_lines.extend(lyric_text_lines)
        results.append(
            {
                "big_segment_id": big_segment_id,
                "lyric_line_count": len(all_lyric_lines),
                "lyric_excerpt": _build_excerpt(
                    lyric_lines=all_lyric_lines,
                    max_chars=ROLE2_LYRIC_EXCERPT_MAX_CHARS,
                ),
            }
        )
    return results


def build_role3_big_segment_lyric_context(module_a_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    功能说明：构建角色3使用的“大段级歌词挂载索引”。
    参数说明：
    - module_a_output: 模块A输出。
    返回值：
    - dict[str, dict[str, Any]]: big_segment_id 到歌词挂载上下文的映射。
    异常说明：无。
    边界条件：无歌词时保留空列表字段。
    """
    big_segments = [dict(item) for item in module_a_output.get("big_segments", []) if isinstance(item, dict)]
    segments = [dict(item) for item in module_a_output.get("segments", []) if isinstance(item, dict)]
    lyric_lines_by_segment = _build_segment_lyric_lines_map(module_a_output=module_a_output)

    segments_by_big_segment: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        big_segment_id = str(segment.get("big_segment_id", "")).strip()
        if big_segment_id:
            segments_by_big_segment.setdefault(big_segment_id, []).append(segment)

    results: dict[str, dict[str, Any]] = {}
    for big_segment in big_segments:
        big_segment_id = str(big_segment.get("segment_id", "")).strip()
        if not big_segment_id:
            continue
        segment_items = sorted(
            segments_by_big_segment.get(big_segment_id, []),
            key=lambda item: float(item.get("start_time", 0.0)),
        )
        segment_lyrics: list[dict[str, Any]] = []
        all_lyric_lines: list[str] = []
        for segment in segment_items:
            segment_id = str(segment.get("segment_id", "")).strip()
            lyric_lines = list(lyric_lines_by_segment.get(segment_id, []))
            all_lyric_lines.extend(lyric_lines)
            segment_lyrics.append(
                {
                    "segment_id": segment_id,
                    "lyric_count": len(lyric_lines),
                    "lyric_lines": lyric_lines,
                }
            )
        results[big_segment_id] = {
            "big_segment_id": big_segment_id,
            "lyric_line_count": len(all_lyric_lines),
            "lyric_excerpt": _build_excerpt(
                lyric_lines=all_lyric_lines,
                max_chars=ROLE3_LYRIC_EXCERPT_MAX_CHARS,
            ),
            "segment_lyrics": segment_lyrics,
        }
    return results


def _build_segment_lyric_lines_map(module_a_output: dict[str, Any]) -> dict[str, list[str]]:
    """
    功能说明：按 segment_id 聚合歌词文本行，并过滤角色编排不需要的细粒度字段。
    参数说明：
    - module_a_output: 模块A输出。
    返回值：
    - dict[str, list[str]]: 小段 ID 到歌词文本行列表的映射。
    异常说明：无。
    边界条件：空文本与无效对象会被忽略。
    """
    lyric_units = [dict(item) for item in module_a_output.get("lyric_units", []) if isinstance(item, dict)]
    lyric_lines_by_segment: dict[str, list[str]] = {}
    for lyric_unit in lyric_units:
        segment_id = str(lyric_unit.get("segment_id", "")).strip()
        lyric_text = str(lyric_unit.get("text", "")).strip()
        if not segment_id or not lyric_text:
            continue
        lyric_lines_by_segment.setdefault(segment_id, []).append(lyric_text)
    return lyric_lines_by_segment


def _build_excerpt(*, lyric_lines: list[str], max_chars: int) -> str:
    """
    功能说明：将多行歌词压缩为短摘要，避免把完整挂载树送入 LLM。
    参数说明：
    - lyric_lines: 原始歌词文本行列表。
    - max_chars: 摘要最大字符数。
    返回值：
    - str: 截断后的短摘要文本。
    异常说明：无。
    边界条件：空输入返回空字符串。
    """
    normalized_lines = [str(item).strip() for item in lyric_lines if str(item).strip()]
    if not normalized_lines:
        return ""
    excerpt = " / ".join(normalized_lines)
    if len(excerpt) <= max(0, int(max_chars)):
        return excerpt
    safe_max_chars = max(1, int(max_chars) - 1)
    return f"{excerpt[:safe_max_chars]}…"
