"""
文件用途：提供模块A V2歌词与小段挂载能力。
核心流程：按时间重叠与 token 裁剪将歌词绑定到最终 segments。
输入输出：输入句级歌词与 segments，输出带 segment_id 的歌词单元。
依赖说明：依赖正则与 v2 时间工具。
维护说明：本文件仅负责挂载，不承担歌词清洗逻辑。
"""

# 标准库：正则处理
import re
# 标准库：类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time


# 常量：用于清理歌词文本句首标点（含中英文常见符号）的正则。
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+")
# 常量：用于识别“纯标点文本”，避免将其作为可展示歌词。
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+$")
# 常量：跨段 token 前段残片归后段阈值（秒，V2 专用策略）。
SMALL_BOUNDARY_TOKEN_FRAGMENT_SECONDS = 0.021


def attach_lyrics_to_segments(
    lyric_units_raw: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    prefer_next_segment_for_small_boundary_token: bool = False,
    small_boundary_token_fragment_seconds: float = SMALL_BOUNDARY_TOKEN_FRAGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """
    功能说明：将歌词单元按时间绑定到小段落。
    参数说明：
    - lyric_units_raw: ASR原始歌词单元列表。
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    - prefer_next_segment_for_small_boundary_token: 是否启用“跨段小残片归后段”策略。
    - small_boundary_token_fragment_seconds: 小残片阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 挂载了 segment_id 的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：输入为空时返回空列表。
    """
    if not lyric_units_raw or not segments:
        return []

    safe_small_fragment_seconds = max(0.0, float(small_boundary_token_fragment_seconds))
    sorted_segments = sorted(segments, key=lambda item: float(item.get("start_time", 0.0)))
    output_items: list[dict[str, Any]] = []
    for item in lyric_units_raw:
        start_time = round_time(float(item.get("start_time", 0.0)))
        end_time = round_time(max(start_time, float(item.get("end_time", start_time))))
        if end_time - start_time <= 1e-6:
            continue

        token_units = _normalize_token_units(item.get("token_units", []))
        base_text = _strip_edge_punctuation(str(item.get("text", "")).strip())
        if token_units:
            base_text = _build_text_from_token_units(token_units) or base_text
        if not base_text or _is_punctuation_only_text(base_text):
            continue

        source_sentence_index = item.get("source_sentence_index", None)
        unit_transform = str(item.get("unit_transform", "")).strip().lower()
        confidence = round(float(item.get("confidence", 0.0)), 3)
        overlapped_segments = _collect_overlapped_segments(
            start_time=start_time,
            end_time=end_time,
            segments=sorted_segments,
        )

        # 没有 token 级时间戳时，无法安全切片，保持“单句挂单段”的兜底语义。
        if not token_units or not overlapped_segments:
            segment = _select_best_overlap_segment(
                start_time=start_time,
                end_time=end_time,
                segments=sorted_segments,
            )
            attached_item = {
                "segment_id": str(segment["segment_id"]),
                "start_time": start_time,
                "end_time": end_time,
                "text": base_text,
                "confidence": confidence,
            }
            if isinstance(source_sentence_index, int) and source_sentence_index >= 0:
                attached_item["source_sentence_index"] = source_sentence_index
            if unit_transform in {"original", "split", "merged"}:
                attached_item["unit_transform"] = unit_transform
            if token_units:
                attached_item["token_units"] = token_units
            output_items.append(attached_item)
            continue

        split_items: list[dict[str, Any]] = []
        covered_duration = 0.0
        for segment in overlapped_segments:
            seg_start = float(segment.get("start_time", 0.0))
            seg_end = max(seg_start, float(segment.get("end_time", seg_start)))
            clip_start = round_time(max(start_time, seg_start))
            clip_end = round_time(min(end_time, seg_end))
            if clip_end - clip_start <= 1e-6:
                continue
            should_drop_small_right_fragment = (
                prefer_next_segment_for_small_boundary_token
                and safe_small_fragment_seconds > 0.0
                and seg_end < end_time - 1e-6
            )
            clipped_tokens = _clip_token_units_by_range(
                token_units=token_units,
                start_time=clip_start,
                end_time=clip_end,
                drop_right_boundary_fragment_seconds=(
                    safe_small_fragment_seconds if should_drop_small_right_fragment else None
                ),
            )
            clipped_text = _build_text_from_token_units(clipped_tokens)
            if not clipped_text:
                continue
            if _is_punctuation_only_text(clipped_text):
                continue

            attached_item = {
                "segment_id": str(segment["segment_id"]),
                "start_time": clip_start,
                "end_time": clip_end,
                "text": clipped_text,
                "confidence": confidence,
                "token_units": clipped_tokens,
            }
            if isinstance(source_sentence_index, int) and source_sentence_index >= 0:
                attached_item["source_sentence_index"] = source_sentence_index
            if unit_transform in {"original", "split", "merged"}:
                attached_item["unit_transform"] = unit_transform
            split_items.append(attached_item)
            covered_duration += max(0.0, clip_end - clip_start)

        lyric_duration = max(1e-6, end_time - start_time)
        coverage_ratio = covered_duration / lyric_duration
        if split_items and coverage_ratio >= 0.5:
            output_items.extend(split_items)
            continue

        # token 切片覆盖不足时回退到最大重叠挂载，避免“只挂到标点附近小片段”。
        segment = _select_best_overlap_segment(
            start_time=start_time,
            end_time=end_time,
            segments=sorted_segments,
        )
        attached_item = {
            "segment_id": str(segment["segment_id"]),
            "start_time": start_time,
            "end_time": end_time,
            "text": base_text,
            "confidence": confidence,
        }
        if token_units:
            attached_item["token_units"] = token_units
        if isinstance(source_sentence_index, int) and source_sentence_index >= 0:
            attached_item["source_sentence_index"] = source_sentence_index
        if unit_transform in {"original", "split", "merged"}:
            attached_item["unit_transform"] = unit_transform
        output_items.append(attached_item)
    return output_items


def _normalize_token_units(token_units: Any) -> list[dict[str, Any]]:
    """
    功能说明：清洗 token_units，确保结构满足 text/start/end。
    参数说明：
    - token_units: token级时间单元列表。
    返回值：
    - list[dict[str, Any]]: 归一化后的 token 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：输入非法或空内容时返回空列表。
    """
    if not isinstance(token_units, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in token_units:
        if not isinstance(item, dict):
            continue
        start_raw = item.get("start_time", item.get("start", 0.0))
        end_raw = item.get("end_time", item.get("end", start_raw))
        if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            continue
        start_time = round_time(float(start_raw))
        end_time = round_time(max(start_time, float(end_raw)))
        token_text = str(item.get("text", ""))
        if not token_text.strip():
            continue
        normalized.append(
            {
                "text": token_text,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return normalized


def _strip_edge_punctuation(text: str) -> str:
    """
    功能说明：去除文本句首的标点与空白，避免句首孤立标点进入歌词展示内容。
    参数说明：
    - text: 原始文本。
    返回值：
    - str: 清洗后的文本。
    异常说明：无。
    边界条件：仅清理句首标点，不移除句中或句尾标点。
    """
    raw_text = str(text).strip()
    if not raw_text:
        return ""
    cleaned_text = EDGE_PUNCTUATION_PATTERN.sub("", raw_text)
    return cleaned_text.strip()


def _is_punctuation_only_text(text: str) -> bool:
    """
    功能说明：判断文本是否仅由标点和空白组成。
    参数说明：
    - text: 待判断文本。
    返回值：
    - bool: 仅标点/空白返回 True，否则 False。
    异常说明：无。
    边界条件：空字符串返回 False。
    """
    raw_text = str(text).strip()
    if not raw_text:
        return False
    return bool(PUNCTUATION_ONLY_PATTERN.fullmatch(raw_text))


def _build_text_from_token_units(token_units: list[dict[str, Any]]) -> str:
    """
    功能说明：根据 token 列表组装可展示歌词文本，并去除句首标点。
    参数说明：
    - token_units: token级时间单元列表。
    返回值：
    - str: 组装后的歌词文本。
    异常说明：无。
    边界条件：当 token 为空或仅标点时返回空字符串。
    """
    if not token_units:
        return ""
    token_text_parts = [str(item.get("text", "")) for item in token_units if str(item.get("text", "")).strip()]
    if not token_text_parts:
        return ""
    joined_text = "".join(token_text_parts)
    cleaned_text = _strip_edge_punctuation(joined_text)
    if _is_punctuation_only_text(cleaned_text):
        return ""
    return cleaned_text


def _clip_token_units_by_range(
    token_units: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    drop_right_boundary_fragment_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：将 token 列表裁剪到指定时间窗口，便于歌词按 segment 精确挂载。
    参数说明：
    - token_units: token级时间单元列表。
    - start_time: 裁剪窗口起点（秒）。
    - end_time: 裁剪窗口终点（秒）。
    - drop_right_boundary_fragment_seconds: 丢弃跨越右边界极短残片阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 裁剪后的 token 列表。
    异常说明：无。
    边界条件：窗口无交集时返回空列表。
    """
    if not token_units or end_time <= start_time:
        return []
    safe_drop_threshold = None
    if isinstance(drop_right_boundary_fragment_seconds, (int, float)):
        safe_drop_threshold = max(0.0, float(drop_right_boundary_fragment_seconds))
    clipped_units: list[dict[str, Any]] = []
    for item in token_units:
        token_start = float(item.get("start_time", 0.0))
        token_end = max(token_start, float(item.get("end_time", token_start)))
        overlap_start = max(start_time, token_start)
        overlap_end = min(end_time, token_end)
        clipped_duration = overlap_end - overlap_start
        if clipped_duration <= 1e-6:
            continue
        if (
            safe_drop_threshold is not None
            and token_end > end_time + 1e-6
            and overlap_end >= end_time - 1e-6
            and clipped_duration <= safe_drop_threshold + 1e-6
        ):
            # 将跨越右边界的极短残片交给后续 segment，避免“前段多挂字”。
            continue
        clipped_units.append(
            {
                "text": str(item.get("text", "")),
                "start_time": round_time(overlap_start),
                "end_time": round_time(overlap_end),
            }
        )
    return clipped_units


def _collect_overlapped_segments(start_time: float, end_time: float, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：收集与歌词时间区间有交集的小段落列表。
    参数说明：
    - start_time: 歌词区间起点（秒）。
    - end_time: 歌词区间终点（秒）。
    - segments: 小段落列表。
    返回值：
    - list[dict[str, Any]]: 与歌词有交集的小段落，按起点升序返回。
    异常说明：无。
    边界条件：无交集时返回空列表。
    """
    overlapped_segments: list[dict[str, Any]] = []
    for segment in segments:
        seg_start = float(segment.get("start_time", 0.0))
        seg_end = max(seg_start, float(segment.get("end_time", seg_start)))
        overlap = max(0.0, min(end_time, seg_end) - max(start_time, seg_start))
        if overlap <= 1e-6:
            continue
        overlapped_segments.append(segment)
    return overlapped_segments


def _select_best_overlap_segment(start_time: float, end_time: float, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：根据时间重叠度选择最匹配的小段落，避免边界歌词挂错片段。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - segments: 小段落列表。
    返回值：
    - dict[str, Any]: 与目标区间重叠最优的分段对象。
    异常说明：无。
    边界条件：segments 为空时返回 unknown 占位段。
    """
    if not segments:
        return {"segment_id": "seg_0000", "start_time": 0.0, "end_time": 0.0, "label": "unknown"}

    lyric_end = max(start_time, end_time)
    best_segment = segments[0]
    best_overlap = -1.0
    for segment in segments:
        seg_start = float(segment["start_time"])
        seg_end = float(segment["end_time"])
        overlap = max(0.0, min(lyric_end, seg_end) - max(start_time, seg_start))
        if overlap > best_overlap + 1e-6:
            best_segment = segment
            best_overlap = overlap
            continue
        if abs(overlap - best_overlap) <= 1e-6 and seg_start <= start_time <= seg_end:
            best_segment = segment
    return best_segment
