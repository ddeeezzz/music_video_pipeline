"""
文件用途：跨来源歌词候选合并——当不同来源返回相同歌曲的同步歌词时，
          拼凑翻译、罗马音、词级时间戳等补充字段，产生更完整的歌词候选。
核心流程：收集所有来源候选 → 按清洗后的歌词文本分组 → 同组内合并补充字段
         → 返回合成候选列表。
输入输出：输入 provider_groups 列表，输出合并候选列表。
维护说明：本文件为自包含模块，不依赖 pipeline.py，避免循环导入。
"""

from difflib import SequenceMatcher
import re
from typing import Any

# 常量：LRC 行时间戳提取规则（同 lrc_parser.py）
LRC_TIMESTAMP_PATTERN = re.compile(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]")
# 常量：非单词字符归一化规则（同 pipeline.py）
NON_WORD_NORMALIZE_PATTERN = re.compile(r"[\s\-_/\|:：·,，.。!！?？'\"]+")

# 常量：歌词文本相似度阈值（字符级 SequenceMatcher ratio）
LYRICS_SIMILARITY_THRESHOLD = 0.8

# 常量：合并来源识别名
MERGED_PROVIDER_NAME = "merged"
MERGED_PROVIDER_DISPLAY = "合并补全"


def _strip_lrc_timestamps(lrc_text: str) -> str:
    """
    功能说明：从 LRC 文本中移除所有时间戳标记，仅保留纯文本行。
    参数说明：
    - lrc_text: LRC 歌词原文。
    返回值：
    - str: 去时间戳后的纯文本行，每行用换行符分隔。
    """
    normalized_text = str(lrc_text).strip()
    if not normalized_text:
        return ""
    text_lines: list[str] = []
    for raw_line in normalized_text.splitlines():
        line_text = raw_line.strip()
        if not line_text:
            continue
        clean_line = LRC_TIMESTAMP_PATTERN.sub("", line_text).strip()
        if clean_line:
            text_lines.append(clean_line)
    return "\n".join(text_lines)


def _normalize_lyrics_for_comparison(text: str) -> str:
    """
    功能说明：将歌词文本标准化为可比对形式——移时间戳、去标点、小写、去空格。
    参数说明：
    - text: LRC 歌词原文。
    返回值：
    - str: 标准化后的纯文本（仅保留字母数字）。
    边界条件：空输入返回空字符串。
    """
    plain_text = _strip_lrc_timestamps(text).strip().lower()
    if not plain_text:
        return ""
    return NON_WORD_NORMALIZE_PATTERN.sub("", plain_text)


def _compute_lyrics_similarity(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> float:
    """
    功能说明：计算两个歌词候选的同步歌词文本相似度。
    参数说明：
    - candidate_a: 候选 A。
    - candidate_b: 候选 B。
    返回值：
    - float: 0~1 的相似度分数。
    边界条件：任一候选缺少 synced_lyrics 或文本过短时返回 0。
    """
    text_a = str(candidate_a.get("synced_lyrics", "")).strip()
    text_b = str(candidate_b.get("synced_lyrics", "")).strip()
    if not text_a or not text_b:
        return 0.0
    normalized_a = _normalize_lyrics_for_comparison(text_a)
    normalized_b = _normalize_lyrics_for_comparison(text_b)
    if not normalized_a or not normalized_b:
        return 0.0
    if len(normalized_a) < 10 or len(normalized_b) < 10:
        return 0.0
    return float(SequenceMatcher(None, normalized_a, normalized_b).ratio())


def _count_completeness(candidate: dict[str, Any]) -> int:
    """
    功能说明：统计候选的补充字段完整度，用于选择合并基准。
    参数说明：
    - candidate: 歌词候选。
    返回值：
    - int: 携带的非空补充字段数量（word_timed / translated / romanized）。
    """
    return sum(
        1 for key in ("word_timed_lyrics", "translated_lyrics", "romanized_lyrics")
        if bool(str(candidate.get(key, "")).strip())
    )


def _merge_two_candidates(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """
    功能说明：将副候选的缺失补充字段合并到主候选。
    参数说明：
    - primary: 主候选（作为合并基准；synced_lyrics 不做替换）。
    - secondary: 副候选（提供主候选缺失的字段）。
    返回值：
    - dict[str, Any]: 合并后的候选对象（操作原对象副本）。
    边界条件：仅合并主候选为空而副候选非空的字段。
    """
    merged = dict(primary)
    merged["provider_id"] = f'{str(primary.get("provider_id", "")).strip()}+{str(secondary.get("provider_id", "")).strip()}'
    if not str(merged.get("provider_song_id", "")).strip():
        merged["provider_song_id"] = str(secondary.get("provider_song_id", "")).strip()
    for key in ("word_timed_lyrics", "translated_lyrics", "romanized_lyrics"):
        if not str(merged.get(key, "")).strip():
            secondary_value = str(secondary.get(key, "")).strip()
            if secondary_value:
                merged[key] = secondary_value
    merged["score"] = max(
        float(primary.get("score", 0.0) or 0.0),
        float(secondary.get("score", 0.0) or 0.0),
    )
    merged["has_word_timed_lyrics"] = bool(str(merged.get("word_timed_lyrics", "")).strip())
    merged["has_translated_lyrics"] = bool(str(merged.get("translated_lyrics", "")).strip())
    merged["has_romanized_lyrics"] = bool(str(merged.get("romanized_lyrics", "")).strip())
    merged["provider"] = MERGED_PROVIDER_NAME
    return merged


def merge_lyrics_candidates(
    provider_groups: list[dict[str, Any]],
    logger,
) -> list[dict[str, Any]]:
    """
    功能说明：跨来源合并候选歌词——比较各来源候选的歌词正文相似度，
              将同歌曲的不同补充字段合并到同一候选中。
    参数说明：
    - provider_groups: 来源分组候选数组。
    - logger: 日志对象。
    返回值：
    - list[dict[str, Any]]: 合并后的候选列表。如果没有任何可合并场景，返回空列表。
    边界条件：仅对至少有两个不同来源、且存在歌词相似度>=阈值候选的场景生效。
    """
    if not isinstance(provider_groups, list):
        return []
    valid_groups = [
        g for g in provider_groups
        if isinstance(g, dict) and isinstance(g.get("candidates", []), list)
    ]
    if len(valid_groups) < 2:
        return []
    all_candidates: list[dict[str, Any]] = []
    for group in valid_groups:
        for candidate in group.get("candidates", []):
            if isinstance(candidate, dict) and str(candidate.get("synced_lyrics", "")).strip():
                all_candidates.append(candidate)
    if len(all_candidates) < 2:
        return []
    merged_results: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for i in range(len(all_candidates)):
        if i in used_indices:
            continue
        current_cluster: list[int] = [i]
        for j in range(i + 1, len(all_candidates)):
            if j in used_indices:
                continue
            similarity = _compute_lyrics_similarity(all_candidates[i], all_candidates[j])
            if similarity >= LYRICS_SIMILARITY_THRESHOLD:
                current_cluster.append(j)
        if len(current_cluster) < 2:
            continue
        cluster_members = [all_candidates[idx] for idx in current_cluster]
        cluster_members.sort(key=_count_completeness, reverse=True)
        primary_candidate = cluster_members[0]
        current_merged = dict(primary_candidate)
        for secondary_candidate in cluster_members[1:]:
            current_merged = _merge_two_candidates(primary=current_merged, secondary=secondary_candidate)
            used_indices.add(all_candidates.index(secondary_candidate))
        merged_results.append(current_merged)
        used_indices.add(all_candidates.index(cluster_members[0]))
    if merged_results:
        logger.info(
            "模块A V2-歌词合并完成：共 %d 组可合并，生成 %d 个合并候选",
            len(merged_results),
            len(merged_results),
        )
    return merged_results
