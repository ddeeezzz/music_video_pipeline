"""
文件用途：提供模块A的歌词清洗、视觉单元与挂载逻辑。
核心流程：清洗句级歌词，生成视觉歌词单元并挂载到小段落。
输入输出：输入 ASR 原始单元与 segments，输出清洗后 lyric_units。
依赖说明：依赖标准库 re 与项目内时间工具。
维护说明：保持“正常/未识别/吟唱”三态语义。
"""

# 标准库：正则处理
import re
# 标准库：类型提示
from typing import Any

# 项目内模块：时间工具
from music_video_pipeline.modules.module_a.timing_energy import _find_big_segment, _round_time


def _clean_lyric_units(
    lyric_units_raw: list[dict[str, Any]],
    big_segments: list[dict[str, Any]],
    instrumental_labels: list[str],
    logger,
    min_confidence: float = 0.25,
) -> list[dict[str, Any]]:
    """
    功能说明：清洗 ASR 原始结果：输出可靠歌词/未识别歌词/吟唱三态。
    参数说明：
    - lyric_units_raw: ASR原始歌词单元列表。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - logger: 日志记录器，用于输出过程与异常信息。
    - min_confidence: 最低可接受置信度阈值。
    返回值：
    - list[dict[str, Any]]: 清洗后的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not lyric_units_raw:
        return []

    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    cleaned_units: list[dict[str, Any]] = []
    unknown_text = "[未识别歌词]"
    for item in lyric_units_raw:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        start_time = _round_time(start_time)
        end_time = _round_time(max(start_time, end_time))
        text = str(item.get("text", "")).strip()
        confidence = round(float(item.get("confidence", 0.0)), 3)
        no_speech_prob = max(0.0, min(1.0, float(item.get("no_speech_prob", 0.35))))
        token_units = _normalize_token_units(item.get("token_units", []))
        if not text:
            continue

        mid_time = (start_time + end_time) / 2.0
        big_segment = _find_big_segment(mid_time, big_segments)
        big_label = str(big_segment.get("label", "unknown")).lower().strip()
        if big_label in instrumental_set:
            continue

        if _is_vocalise_text(text):
            cleaned_units.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": "吟唱",
                    "confidence": max(0.5, confidence),
                    "token_units": [],
                }
            )
            continue

        if _is_obvious_noise_text(text=text, confidence=confidence, no_speech_prob=no_speech_prob):
            continue

        suspected_vocal = _is_probable_vocal_presence(
            confidence=confidence,
            no_speech_prob=no_speech_prob,
            start_time=start_time,
            end_time=end_time,
        )
        if _is_placeholder_text(text):
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                        "token_units": [],
                    }
                )
            continue

        if confidence < min_confidence:
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                        "token_units": [],
                    }
                )
            continue

        cleaned_units.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "confidence": confidence,
                "token_units": token_units,
            }
        )

    cleaned_units.sort(key=lambda item: float(item["start_time"]))
    logger.info("模块A-歌词清洗完成，原始=%s，保留=%s", len(lyric_units_raw), len(cleaned_units))
    return cleaned_units


def _build_visual_lyric_units(
    sentence_units: list[dict[str, Any]],
    big_segments: list[dict[str, Any]],
    instrumental_labels: list[str],
    lyric_segment_policy: str,
    comma_pause_seconds: float,
    long_pause_seconds: float,
    merge_gap_seconds: float,
    max_visual_unit_seconds: float,
    logger,
) -> list[dict[str, Any]]:
    """
    功能说明：将句级歌词单元转换为视觉歌词单元（支持句级严格/自适应短句并拆）。
    参数说明：
    - sentence_units: 句级歌词单元列表。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - lyric_segment_policy: 歌词视觉切分策略（如 sentence_strict/adaptive_phrase）。
    - comma_pause_seconds: 逗号停顿触发切分阈值（秒）。
    - long_pause_seconds: 长停顿触发切分阈值（秒）。
    - merge_gap_seconds: 相邻短单元允许合并的最大间隔（秒）。
    - max_visual_unit_seconds: 单个视觉歌词单元允许的最大时长（秒）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - list[dict[str, Any]]: 用于视觉分段的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not sentence_units:
        return []

    policy = _normalize_lyric_segment_policy(lyric_segment_policy=lyric_segment_policy, logger=logger)
    comma_pause_seconds = _normalize_positive_threshold(
        threshold_name="comma_pause_seconds",
        value=comma_pause_seconds,
        fallback=0.45,
        logger=logger,
    )
    long_pause_seconds = _normalize_positive_threshold(
        threshold_name="long_pause_seconds",
        value=long_pause_seconds,
        fallback=0.8,
        logger=logger,
    )
    merge_gap_seconds = _normalize_non_negative_threshold(
        threshold_name="merge_gap_seconds",
        value=merge_gap_seconds,
        fallback=0.25,
        logger=logger,
    )
    max_visual_unit_seconds = _normalize_positive_threshold(
        threshold_name="max_visual_unit_seconds",
        value=max_visual_unit_seconds,
        fallback=6.0,
        logger=logger,
    )

    normalized_sentence_units: list[dict[str, Any]] = []
    sorted_sentence_units = sorted(sentence_units, key=lambda item: float(item.get("start_time", 0.0)))
    for sentence_index, item in enumerate(sorted_sentence_units):
        normalized_item = {
            "start_time": _round_time(float(item.get("start_time", 0.0))),
            "end_time": _round_time(max(float(item.get("start_time", 0.0)), float(item.get("end_time", 0.0)))),
            "text": str(item.get("text", "")).strip(),
            "confidence": round(float(item.get("confidence", 0.0)), 3),
            "token_units": _normalize_token_units(item.get("token_units", [])),
            "source_sentence_index": sentence_index,
            "unit_transform": "original",
        }
        if normalized_item["text"]:
            normalized_sentence_units.append(normalized_item)

    if policy == "sentence_strict":
        logger.info("模块A-歌词视觉单元策略=%s，句级单元=%s，视觉单元=%s", policy, len(normalized_sentence_units), len(normalized_sentence_units))
        return normalized_sentence_units

    split_units: list[dict[str, Any]] = []
    for item in normalized_sentence_units:
        split_units.extend(
            _split_sentence_unit_for_adaptive_policy(
                sentence_unit=item,
                comma_pause_seconds=comma_pause_seconds,
                long_pause_seconds=long_pause_seconds,
                max_visual_unit_seconds=max_visual_unit_seconds,
            )
        )

    merged_units = _merge_adjacent_visual_lyric_units(
        visual_units=split_units,
        big_segments=big_segments,
        instrumental_labels=instrumental_labels,
        merge_gap_seconds=merge_gap_seconds,
        max_visual_unit_seconds=max_visual_unit_seconds,
    )
    logger.info("模块A-歌词视觉单元策略=%s，句级单元=%s，视觉单元=%s", policy, len(normalized_sentence_units), len(merged_units))
    return merged_units


def _normalize_lyric_segment_policy(lyric_segment_policy: str, logger) -> str:
    """
    功能说明：归一化歌词视觉单元策略，不合法时回退 sentence_strict。
    参数说明：
    - lyric_segment_policy: 歌词视觉切分策略（如 sentence_strict/adaptive_phrase）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - str: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized = str(lyric_segment_policy).strip().lower()
    if normalized in {"sentence_strict", "adaptive_phrase"}:
        return normalized
    logger.warning("模块A-歌词视觉单元策略非法，已回退 sentence_strict，原始值=%s", lyric_segment_policy)
    return "sentence_strict"


def _normalize_positive_threshold(threshold_name: str, value: float, fallback: float, logger) -> float:
    """
    功能说明：将阈值归一化为正数，不合法时回退默认值。
    参数说明：
    - threshold_name: 阈值配置项名称。
    - value: 待归一化的输入值。
    - fallback: 输入不合法时采用的默认值。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - float: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    try:
        normalized = float(value)
    except Exception:  # noqa: BLE001
        logger.warning("模块A-%s 配置非法，已回退默认值=%s，原始值=%s", threshold_name, fallback, value)
        return fallback
    if normalized <= 0.0:
        logger.warning("模块A-%s 必须大于0，已回退默认值=%s，原始值=%s", threshold_name, fallback, value)
        return fallback
    return normalized


def _normalize_non_negative_threshold(threshold_name: str, value: float, fallback: float, logger) -> float:
    """
    功能说明：将阈值归一化为非负数，不合法时回退默认值。
    参数说明：
    - threshold_name: 阈值配置项名称。
    - value: 待归一化的输入值。
    - fallback: 输入不合法时采用的默认值。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - float: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    try:
        normalized = float(value)
    except Exception:  # noqa: BLE001
        logger.warning("模块A-%s 配置非法，已回退默认值=%s，原始值=%s", threshold_name, fallback, value)
        return fallback
    if normalized < 0.0:
        logger.warning("模块A-%s 不能为负，已回退默认值=%s，原始值=%s", threshold_name, fallback, value)
        return fallback
    return normalized


def _split_sentence_unit_for_adaptive_policy(
    sentence_unit: dict[str, Any],
    comma_pause_seconds: float,
    long_pause_seconds: float,
    max_visual_unit_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：将单句歌词按标点+停顿拆分为更稳定的视觉歌词单元。
    参数说明：
    - sentence_unit: 业务处理所需输入参数。
    - comma_pause_seconds: 逗号停顿触发切分阈值（秒）。
    - long_pause_seconds: 长停顿触发切分阈值（秒）。
    - max_visual_unit_seconds: 单个视觉歌词单元允许的最大时长（秒）。
    返回值：
    - list[dict[str, Any]]: 切分后的区间或单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    token_units = _normalize_token_units(sentence_unit.get("token_units", []))
    if len(token_units) <= 1:
        return [dict(sentence_unit)]

    split_slices: list[tuple[int, int]] = []
    chunk_start_index = 0
    for token_index in range(len(token_units) - 1):
        current_token = token_units[token_index]
        next_token = token_units[token_index + 1]
        current_text = str(current_token.get("text", ""))
        gap_after = float(next_token.get("start_time", 0.0)) - float(current_token.get("end_time", 0.0))
        chunk_duration = float(current_token.get("end_time", 0.0)) - float(token_units[chunk_start_index].get("start_time", 0.0))
        should_split = False

        if re.search(r"[。！？!?；;]", current_text):
            should_split = True
        elif "、" in current_text:
            prev_end = float(token_units[token_index - 1].get("end_time", current_token.get("start_time", 0.0))) if token_index > 0 else float(current_token.get("start_time", 0.0))
            gap_before = float(current_token.get("start_time", 0.0)) - prev_end
            if gap_before >= comma_pause_seconds or gap_after >= comma_pause_seconds:
                should_split = True

        if not should_split and gap_after >= long_pause_seconds:
            should_split = True

        if not should_split and chunk_duration >= max_visual_unit_seconds:
            should_split = True

        if should_split:
            split_slices.append((chunk_start_index, token_index))
            chunk_start_index = token_index + 1

    if chunk_start_index <= len(token_units) - 1:
        split_slices.append((chunk_start_index, len(token_units) - 1))

    if len(split_slices) <= 1:
        return [dict(sentence_unit)]

    split_units: list[dict[str, Any]] = []
    for left_index, right_index in split_slices:
        token_slice = token_units[left_index : right_index + 1]
        split_unit = _build_visual_unit_from_token_slice(
            token_slice=token_slice,
            source_unit=sentence_unit,
            unit_transform="split",
        )
        if split_unit:
            split_units.append(split_unit)
    return split_units if split_units else [dict(sentence_unit)]


def _build_visual_unit_from_token_slice(
    token_slice: list[dict[str, Any]],
    source_unit: dict[str, Any],
    unit_transform: str,
) -> dict[str, Any]:
    """
    功能说明：将 token 切片构建为视觉歌词单元。
    参数说明：
    - token_slice: 待构建视觉单元的token切片。
    - source_unit: 原始句级单元。
    - unit_transform: 单元变换来源标记（如 split/merge）。
    返回值：
    - dict[str, Any]: 类型为 `dict[str, Any]` 的处理结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not token_slice:
        return {}

    start_time = _round_time(float(token_slice[0]["start_time"]))
    end_time = _round_time(float(token_slice[-1]["end_time"]))
    text_parts = [str(item.get("text", "")) for item in token_slice]
    has_word_granularity = any(str(item.get("granularity", "")).strip().lower() == "word" for item in token_slice)
    text = " ".join([item for item in text_parts if item]).strip() if has_word_granularity else "".join(text_parts).strip()
    if not text:
        return {}

    return {
        "start_time": start_time,
        "end_time": max(start_time, end_time),
        "text": text,
        "confidence": round(float(source_unit.get("confidence", 0.0)), 3),
        "token_units": _normalize_token_units(token_slice),
        "source_sentence_index": int(source_unit.get("source_sentence_index", 0)),
        "unit_transform": unit_transform,
    }


def _merge_adjacent_visual_lyric_units(
    visual_units: list[dict[str, Any]],
    big_segments: list[dict[str, Any]],
    instrumental_labels: list[str],
    merge_gap_seconds: float,
    max_visual_unit_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：合并相邻短句视觉单元（仅在同段落类型且间隔较小时生效）。
    参数说明：
    - visual_units: 视觉歌词单元列表。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - merge_gap_seconds: 相邻短单元允许合并的最大间隔（秒）。
    - max_visual_unit_seconds: 单个视觉歌词单元允许的最大时长（秒）。
    返回值：
    - list[dict[str, Any]]: 合并后的单元或区间结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not visual_units:
        return []

    sorted_units = sorted(visual_units, key=lambda item: float(item.get("start_time", 0.0)))
    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    short_duration_threshold = max(1.0, max_visual_unit_seconds / 2.0)
    merged_output: list[dict[str, Any]] = []
    current_unit = dict(sorted_units[0])

    for next_unit in sorted_units[1:]:
        if _can_merge_visual_units(
            left_unit=current_unit,
            right_unit=next_unit,
            big_segments=big_segments,
            instrumental_set=instrumental_set,
            merge_gap_seconds=merge_gap_seconds,
            max_visual_unit_seconds=max_visual_unit_seconds,
            short_duration_threshold=short_duration_threshold,
        ):
            current_unit = _merge_two_visual_units(left_unit=current_unit, right_unit=next_unit)
            continue
        merged_output.append(current_unit)
        current_unit = dict(next_unit)
    merged_output.append(current_unit)

    return merged_output


def _can_merge_visual_units(
    left_unit: dict[str, Any],
    right_unit: dict[str, Any],
    big_segments: list[dict[str, Any]],
    instrumental_set: set[str],
    merge_gap_seconds: float,
    max_visual_unit_seconds: float,
    short_duration_threshold: float,
) -> bool:
    """
    功能说明：判断两条视觉歌词单元是否可以合并。
    参数说明：
    - left_unit: 左侧待合并单元。
    - right_unit: 右侧待合并单元。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - instrumental_set: 业务处理所需输入参数。
    - merge_gap_seconds: 相邻短单元允许合并的最大间隔（秒）。
    - max_visual_unit_seconds: 单个视觉歌词单元允许的最大时长（秒）。
    - short_duration_threshold: 业务处理所需输入参数。
    返回值：
    - bool: 类型为 `bool` 的处理结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    left_source = left_unit.get("source_sentence_index", None)
    right_source = right_unit.get("source_sentence_index", None)
    if isinstance(left_source, int) and isinstance(right_source, int) and left_source == right_source:
        return False

    left_text = str(left_unit.get("text", "")).strip()
    right_text = str(right_unit.get("text", "")).strip()
    if not left_text or not right_text:
        return False
    if left_text in {"[未识别歌词]", "吟唱"} or right_text in {"[未识别歌词]", "吟唱"}:
        return False

    left_start = float(left_unit.get("start_time", 0.0))
    left_end = float(left_unit.get("end_time", left_start))
    right_start = float(right_unit.get("start_time", left_end))
    right_end = float(right_unit.get("end_time", right_start))
    left_duration = max(0.0, left_end - left_start)
    right_duration = max(0.0, right_end - right_start)
    gap = right_start - left_end
    merged_duration = right_end - left_start

    if gap < 0.0 or gap > merge_gap_seconds:
        return False
    if left_duration > short_duration_threshold or right_duration > short_duration_threshold:
        return False
    if merged_duration > max_visual_unit_seconds:
        return False

    left_mid = (left_start + left_end) / 2.0
    right_mid = (right_start + right_end) / 2.0
    left_big_segment = _find_big_segment(left_mid, big_segments)
    right_big_segment = _find_big_segment(right_mid, big_segments)
    if str(left_big_segment.get("segment_id", "")) != str(right_big_segment.get("segment_id", "")):
        return False

    left_label = str(left_big_segment.get("label", "unknown")).lower().strip()
    right_label = str(right_big_segment.get("label", "unknown")).lower().strip()
    left_audio_role = "instrumental" if left_label in instrumental_set else "vocal"
    right_audio_role = "instrumental" if right_label in instrumental_set else "vocal"
    return left_audio_role == right_audio_role


def _merge_two_visual_units(left_unit: dict[str, Any], right_unit: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：合并两条视觉歌词单元并保留可追溯元信息。
    参数说明：
    - left_unit: 左侧待合并单元。
    - right_unit: 右侧待合并单元。
    返回值：
    - dict[str, Any]: 合并后的单元或区间结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    left_text = str(left_unit.get("text", "")).strip()
    right_text = str(right_unit.get("text", "")).strip()
    if re.search(r"[。！？!?；;、，,]$", left_text):
        merged_text = f"{left_text}{right_text}"
    else:
        merged_text = f"{left_text} {right_text}".strip()

    left_tokens = _normalize_token_units(left_unit.get("token_units", []))
    right_tokens = _normalize_token_units(right_unit.get("token_units", []))
    merged_tokens = left_tokens + right_tokens
    left_source = int(left_unit.get("source_sentence_index", 0))
    right_source = int(right_unit.get("source_sentence_index", left_source))

    return {
        "start_time": _round_time(float(left_unit.get("start_time", 0.0))),
        "end_time": _round_time(max(float(left_unit.get("start_time", 0.0)), float(right_unit.get("end_time", 0.0)))),
        "text": merged_text,
        "confidence": round(max(float(left_unit.get("confidence", 0.0)), float(right_unit.get("confidence", 0.0))), 3),
        "token_units": merged_tokens,
        "source_sentence_index": min(left_source, right_source),
        "unit_transform": "merged",
    }


def _normalize_token_units(token_units: Any) -> list[dict[str, Any]]:
    """
    功能说明：清洗 token_units，确保结构满足 text/start/end/granularity。
    参数说明：
    - token_units: token级时间单元列表。
    返回值：
    - list[dict[str, Any]]: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
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
        start_time = _round_time(float(start_raw))
        end_time = _round_time(max(start_time, float(end_raw)))
        granularity_raw = str(item.get("granularity", "char")).strip().lower()
        granularity = "word" if granularity_raw == "word" else "char"
        normalized.append(
            {
                "text": str(item.get("text", "")).strip(),
                "start_time": start_time,
                "end_time": end_time,
                "granularity": granularity,
            }
        )
    return normalized


def _is_placeholder_text(text: str) -> bool:
    """
    功能说明：判断是否为歌词占位词（用于转写不可靠标记）。
    参数说明：
    - text: 文本内容。
    返回值：
    - bool: 布尔判断结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return False
    placeholder_set = {"lyrics", "lyric", "歌詞", "歌词", "歌詩", "lyrix", "music"}
    return normalized in placeholder_set


def _is_obvious_noise_text(text: str, confidence: float, no_speech_prob: float) -> bool:
    """
    功能说明：判断明显噪声文本（用于直接过滤，避免幻觉词显示）。
    参数说明：
    - text: 文本内容。
    - confidence: 业务处理所需输入参数。
    - no_speech_prob: 无语音概率估计值。
    返回值：
    - bool: 布尔判断结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return True
    if len(normalized) <= 2 and re.fullmatch(r"[a-z]+", normalized):
        return True
    if len(normalized) <= 3 and re.fullmatch(r"[a-z]+", normalized) and confidence < 0.35:
        return True
    if no_speech_prob >= 0.85 and confidence < 0.4:
        return True
    return False


def _is_probable_vocal_presence(confidence: float, no_speech_prob: float, start_time: float, end_time: float) -> bool:
    """
    功能说明：基于置信度+无声概率估计是否存在人声。
    参数说明：
    - confidence: 业务处理所需输入参数。
    - no_speech_prob: 无语音概率估计值。
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    返回值：
    - bool: 布尔判断结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    duration = max(0.0, end_time - start_time)
    if no_speech_prob <= 0.45:
        return True
    if no_speech_prob <= 0.6 and (confidence >= 0.12 or duration >= 0.35):
        return True
    return False


def _is_vocalise_text(text: str) -> bool:
    """
    功能说明：判断是否为无语义吟唱片段（lalala/dadada/啦啦啦 等）。
    参数说明：
    - text: 文本内容。
    返回值：
    - bool: 布尔判断结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    raw_text = text.strip().lower()
    compact_text = re.sub(r"[\s\-\_\,\.\!\?]+", "", raw_text)
    if not compact_text:
        return False

    if re.fullmatch(r"(?:la|da|na|ra){3,}", compact_text):
        return True
    if re.fullmatch(r"[らラ啦らぁラァ]{3,}", compact_text):
        return True
    if re.fullmatch(r"(?:は|ハ|啊|あ){3,}", compact_text):
        return True
    if re.fullmatch(r"(?:吟唱)+", compact_text):
        return True
    return False


def _attach_lyrics_to_segments(lyric_units_raw: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：将歌词单元按时间绑定到小段落。
    参数说明：
    - lyric_units_raw: ASR原始歌词单元列表。
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    返回值：
    - list[dict[str, Any]]: 挂载了 segment_id 的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not lyric_units_raw or not segments:
        return []

    output_items: list[dict[str, Any]] = []
    for item in lyric_units_raw:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        segment = _select_best_overlap_segment(start_time=start_time, end_time=end_time, segments=segments)
        output_items.append(
            {
                "segment_id": str(segment["segment_id"]),
                "start_time": _round_time(start_time),
                "end_time": _round_time(max(start_time, end_time)),
                "text": str(item.get("text", "")).strip(),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
            }
        )
        source_sentence_index = item.get("source_sentence_index", None)
        if isinstance(source_sentence_index, int) and source_sentence_index >= 0:
            output_items[-1]["source_sentence_index"] = source_sentence_index
        unit_transform = str(item.get("unit_transform", "")).strip().lower()
        if unit_transform in {"original", "split", "merged"}:
            output_items[-1]["unit_transform"] = unit_transform
        token_units = _normalize_token_units(item.get("token_units", []))
        if token_units:
            output_items[-1]["token_units"] = token_units
    return output_items


def _select_best_overlap_segment(start_time: float, end_time: float, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：根据时间重叠度选择最匹配的小段落，避免边界歌词挂错片段。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    返回值：
    - dict[str, Any]: 与目标区间重叠最优的分段对象。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
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
