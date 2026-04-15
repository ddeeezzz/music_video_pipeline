"""
文件用途：执行大时间戳解析（A0->A1单次矫正）与最终S/BIG对齐。
核心流程：歌词占比校正 -> 其他窗口占比校正 -> 近锚点冲突裁决 -> 生成最终segments。
输入输出：输入A0与并段后窗口，输出A1、小时戳、最终segments与边界报告。
依赖说明：依赖 window_builder 句级归一能力。
维护说明：本文件只负责时间轴求解，不负责窗口构造与角色分类。
"""

# 标准库：用于类型提示
from typing import Any

# 项目内模块：复用句级归一
from music_video_pipeline.modules.module_a_v2.timeline.window_builder import normalize_sentence_units


# 常量：浮点比较容差
EPSILON_SECONDS = 1e-6


# 常量：边界最小间隔（秒），避免零时长big
BOUNDARY_MIN_STEP_SECONDS = 0.002


# 常量：后段优先偏置系数（激进版：只要后段占比达到前段45%即优先后一个big）
NEXT_BIG_BIAS_RATIO = 0.45


# 常量：人声静音阈值下界
RMS_SILENCE_FLOOR = 0.003


def _prefer_next_big_with_bias(left_span: float, right_span: float) -> bool:
    """
    功能说明：判断占比决策是否应优先后一个big。
    参数说明：
    - left_span: 边界到跨界段左侧跨度（秒）。
    - right_span: 边界到跨界段右侧跨度（秒）。
    返回值：
    - bool: True 表示边界优先给后一个big。
    异常说明：无。
    边界条件：当右侧跨度 >= 左侧跨度*0.45 时判定为后段优先。
    """
    safe_left = max(0.0, float(left_span))
    safe_right = max(0.0, float(right_span))
    if safe_left <= EPSILON_SECONDS:
        return True
    return safe_right + EPSILON_SECONDS >= safe_left * NEXT_BIG_BIAS_RATIO


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：安全转换为浮点数。
    参数说明：
    - value: 输入对象。
    - default: 回退值。
    返回值：
    - float: 转换结果。
    异常说明：异常内部吞并。
    边界条件：NaN/inf 回退 default。
    """
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if number != number or number in {float("inf"), float("-inf")}:
        return float(default)
    return number


def _round_time(value: float) -> float:
    """
    功能说明：统一时间精度。
    参数说明：
    - value: 原始秒数。
    返回值：
    - float: 6位小数秒。
    异常说明：无。
    边界条件：无。
    """
    return round(float(value), 6)


def _clamp_time(value: float, duration_seconds: float) -> float:
    """
    功能说明：将时间裁剪到合法区间。
    参数说明：
    - value: 原始时间。
    - duration_seconds: 音频总时长。
    返回值：
    - float: 裁剪后时间。
    异常说明：无。
    边界条件：duration 小于0时按0处理。
    """
    safe_duration = max(0.0, float(duration_seconds))
    return max(0.0, min(float(value), safe_duration))


def _quantile(values: list[float], quantile: float) -> float:
    """
    功能说明：计算线性插值分位数。
    参数说明：
    - values: 样本列表。
    - quantile: 分位点（0~1）。
    返回值：
    - float: 分位值。
    异常说明：无。
    边界条件：空输入返回0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    target = max(0.0, min(1.0, float(quantile))) * (len(sorted_values) - 1)
    left_index = int(target)
    right_index = min(len(sorted_values) - 1, left_index + 1)
    if left_index == right_index:
        return sorted_values[left_index]
    weight = target - left_index
    return sorted_values[left_index] * (1.0 - weight) + sorted_values[right_index] * weight


def _rms_value_at(timestamp: float, rms_times: list[float], rms_values: list[float]) -> float:
    """
    功能说明：获取最邻近采样RMS值。
    参数说明：
    - timestamp: 目标时间。
    - rms_times/rms_values: RMS序列。
    返回值：
    - float: RMS值。
    异常说明：无。
    边界条件：空序列返回0。
    """
    if not rms_times or not rms_values:
        return 0.0
    pair_count = min(len(rms_times), len(rms_values))
    if pair_count <= 0:
        return 0.0
    clipped_times = rms_times[:pair_count]
    clipped_values = rms_values[:pair_count]
    nearest_index = min(
        range(pair_count),
        key=lambda index: abs(_safe_float(clipped_times[index], 0.0) - float(timestamp)),
    )
    return max(0.0, _safe_float(clipped_values[nearest_index], 0.0))


def _normalize_big_segments(
    big_segments_a0: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：规范化A0大段列表。
    参数说明：
    - big_segments_a0: 原始A0列表。
    - duration_seconds: 音频总时长。
    返回值：
    - list[dict[str, Any]]: 连续且合法的大段列表。
    异常说明：无。
    边界条件：保持 segment_id 与 label 不变。
    """
    safe_duration = max(0.0, float(duration_seconds))
    sorted_items = sorted(big_segments_a0, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))
    output: list[dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(sorted_items):
        end_time_raw = _safe_float(item.get("end_time", cursor), cursor)
        end_time = _clamp_time(max(cursor, end_time_raw), safe_duration)
        output.append(
            {
                "segment_id": str(item.get("segment_id", f"big_{index + 1:03d}")),
                "start_time": _round_time(cursor),
                "end_time": _round_time(end_time),
                "label": str(item.get("label", "unknown")),
            }
        )
        cursor = end_time
    if output:
        output[0]["start_time"] = 0.0
        output[-1]["end_time"] = _round_time(safe_duration)
        for index in range(1, len(output)):
            output[index]["start_time"] = _round_time(_safe_float(output[index - 1].get("end_time", 0.0), 0.0))
            if _safe_float(output[index].get("end_time", 0.0), 0.0) < _safe_float(output[index].get("start_time", 0.0), 0.0):
                output[index]["end_time"] = output[index]["start_time"]
    return output


def _extract_sentence_tail_time(sentence_item: dict[str, Any], duration_seconds: float) -> float:
    """
    功能说明：提取句尾时间（优先最后token右边界）。
    参数说明：
    - sentence_item: 句级歌词。
    - duration_seconds: 音频总时长。
    返回值：
    - float: 句尾时间。
    异常说明：无。
    边界条件：token缺失回退句级 end_time。
    """
    token_units = sentence_item.get("token_units", [])
    if isinstance(token_units, list) and token_units:
        token_end = _safe_float(token_units[-1].get("end_time", sentence_item.get("end_time", 0.0)), sentence_item.get("end_time", 0.0))
        return _round_time(_clamp_time(token_end, duration_seconds))
    return _round_time(_clamp_time(_safe_float(sentence_item.get("end_time", 0.0), 0.0), duration_seconds))


def _extract_sentence_head_time(sentence_item: dict[str, Any], duration_seconds: float, head_offset_seconds: float) -> float:
    """
    功能说明：提取句首锚点（首token起点 + 固定后移）。
    参数说明：
    - sentence_item: 句级歌词。
    - duration_seconds: 音频总时长。
    - head_offset_seconds: 固定后移秒数。
    返回值：
    - float: 句首锚点。
    异常说明：无。
    边界条件：token缺失回退句 start_time。
    """
    base_start = _safe_float(sentence_item.get("start_time", 0.0), 0.0)
    token_units = sentence_item.get("token_units", [])
    if isinstance(token_units, list) and token_units:
        base_start = _safe_float(token_units[0].get("start_time", base_start), base_start)
    head_time = _clamp_time(base_start + max(0.0, float(head_offset_seconds)), duration_seconds)
    sentence_end = _clamp_time(_safe_float(sentence_item.get("end_time", head_time), head_time), duration_seconds)
    if head_time > sentence_end - EPSILON_SECONDS:
        head_time = _clamp_time(_safe_float(sentence_item.get("start_time", 0.0), 0.0), duration_seconds)
    return _round_time(head_time)


def _resolve_boundaries_with_monotonic_guard(
    base_boundaries: list[float],
    duration_seconds: float,
) -> list[float]:
    """
    功能说明：在边界调整后强制保证边界单调递增。
    参数说明：
    - base_boundaries: 原始边界列表（长度=big段数+1）。
    - duration_seconds: 音频总时长。
    返回值：
    - list[float]: 修正后的边界列表。
    异常说明：无。
    边界条件：首边界固定0，末边界固定duration。
    """
    big_count = max(0, len(base_boundaries) - 1)
    if big_count <= 0:
        return [0.0, _round_time(max(0.0, float(duration_seconds)))]

    fixed = [0.0 for _ in range(big_count + 1)]
    fixed[0] = 0.0
    fixed[-1] = _round_time(max(0.0, float(duration_seconds)))
    for boundary_index in range(1, big_count):
        candidate = _safe_float(base_boundaries[boundary_index], fixed[boundary_index - 1])
        min_allowed = fixed[boundary_index - 1] + BOUNDARY_MIN_STEP_SECONDS
        max_allowed = fixed[-1] - BOUNDARY_MIN_STEP_SECONDS * (big_count - boundary_index)
        fixed[boundary_index] = _round_time(max(min_allowed, min(candidate, max_allowed)))
    return fixed


def _build_a1_from_a0_and_sentences(
    big_segments_a0: list[dict[str, Any]],
    sentence_units: list[dict[str, Any]],
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    功能说明：保留原AL规则，执行 A0->A1 初次边界校正。
    参数说明：
    - big_segments_a0: A0大段。
    - sentence_units: 句级歌词。
    - duration_seconds: 音频总时长。
    返回值：
    - tuple[list[dict], list[dict]]: (A1大段, 边界轨迹)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：只改边界，不改 big 段数与ID。
    """
    normalized_big = _normalize_big_segments(big_segments_a0=big_segments_a0, duration_seconds=duration_seconds)
    if not normalized_big:
        return [], []

    normalized_sentences = normalize_sentence_units(sentence_units=sentence_units, duration_seconds=duration_seconds)
    sentence_tail_ranges: list[tuple[float, float]] = []
    for sentence_item in normalized_sentences:
        sentence_start = _safe_float(sentence_item.get("start_time", 0.0), 0.0)
        sentence_tail = _extract_sentence_tail_time(sentence_item, duration_seconds=duration_seconds)
        sentence_tail_ranges.append((sentence_start, sentence_tail))

    raw_boundaries = [0.0]
    traces: list[dict[str, Any]] = []
    for index, big_item in enumerate(normalized_big[:-1]):
        original_boundary = _safe_float(big_item.get("end_time", 0.0), 0.0)
        candidate_boundary = original_boundary
        hit_sentence = False
        for sentence_start, sentence_tail in sentence_tail_ranges:
            if sentence_start + EPSILON_SECONDS < original_boundary < sentence_tail - EPSILON_SECONDS:
                hit_sentence = True
                left_span = max(0.0, original_boundary - sentence_start)
                right_span = max(0.0, sentence_tail - original_boundary)
                if _prefer_next_big_with_bias(left_span=left_span, right_span=right_span):
                    candidate_boundary = min(candidate_boundary, sentence_start)
                else:
                    candidate_boundary = max(candidate_boundary, sentence_tail)
        raw_boundaries.append(_round_time(candidate_boundary))
        traces.append(
            {
                "boundary_index": index,
                "original_time": _round_time(original_boundary),
                "after_lyric_overlap_time": _round_time(candidate_boundary),
                "hit_cross_sentence": bool(hit_sentence),
            }
        )
    raw_boundaries.append(_round_time(duration_seconds))

    fixed_boundaries = _resolve_boundaries_with_monotonic_guard(raw_boundaries, duration_seconds=duration_seconds)
    output_big: list[dict[str, Any]] = []
    for index, base_item in enumerate(normalized_big):
        output_big.append(
            {
                "segment_id": str(base_item.get("segment_id", f"big_{index + 1:03d}")),
                "start_time": _round_time(fixed_boundaries[index]),
                "end_time": _round_time(fixed_boundaries[index + 1]),
                "label": str(base_item.get("label", "unknown")),
            }
        )
    return output_big, traces


def _apply_other_window_ratio_correction(
    big_segments_a1: list[dict[str, Any]],
    windows_merged: list[dict[str, Any]],
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    功能说明：对跨big的“其他窗口”执行占比边界校正。
    参数说明：
    - big_segments_a1: A1大段（歌词重叠校正后）。
    - windows_merged: 并段后的窗口列表。
    - duration_seconds: 音频总时长。
    返回值：
    - tuple[list[dict], list[dict]]: (校正后A1, 事件列表)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅处理 role in {chant,inst,silence} 且跨边界窗口。
    """
    if len(big_segments_a1) <= 1:
        return list(big_segments_a1), []

    boundaries = [0.0]
    for item in big_segments_a1[:-1]:
        boundaries.append(_safe_float(item.get("end_time", 0.0), 0.0))
    boundaries.append(_round_time(duration_seconds))

    events: list[dict[str, Any]] = []
    for boundary_index in range(1, len(boundaries) - 1):
        boundary_time = _safe_float(boundaries[boundary_index], 0.0)
        crossing_candidates = []
        for window_item in windows_merged:
            role = str(window_item.get("role", "silence")).lower().strip()
            if role not in {"chant", "inst", "silence"}:
                continue
            start_time = _safe_float(window_item.get("start_time", 0.0), 0.0)
            end_time = _safe_float(window_item.get("end_time", start_time), start_time)
            if start_time + EPSILON_SECONDS < boundary_time < end_time - EPSILON_SECONDS:
                crossing_candidates.append(window_item)
        if not crossing_candidates:
            continue

        chosen_window = max(
            crossing_candidates,
            key=lambda item: _safe_float(item.get("end_time", 0.0), 0.0) - _safe_float(item.get("start_time", 0.0), 0.0),
        )
        win_start = _safe_float(chosen_window.get("start_time", 0.0), 0.0)
        win_end = _safe_float(chosen_window.get("end_time", win_start), win_start)
        left_span = max(0.0, boundary_time - win_start)
        right_span = max(0.0, win_end - boundary_time)
        if _prefer_next_big_with_bias(left_span=left_span, right_span=right_span):
            target_boundary = win_start
            action = "move_next_left_to_window_start"
        else:
            target_boundary = win_end
            action = "move_prev_right_to_window_end"

        min_allowed = boundaries[boundary_index - 1] + BOUNDARY_MIN_STEP_SECONDS
        max_allowed = boundaries[boundary_index + 1] - BOUNDARY_MIN_STEP_SECONDS
        if max_allowed <= min_allowed:
            continue
        resolved_boundary = _round_time(max(min_allowed, min(target_boundary, max_allowed)))
        boundaries[boundary_index] = resolved_boundary
        events.append(
            {
                "boundary_index": boundary_index - 1,
                "action": action,
                "window_id": str(chosen_window.get("window_id", "")),
                "window_role": str(chosen_window.get("role", "")),
                "window_start_time": _round_time(win_start),
                "window_end_time": _round_time(win_end),
                "old_boundary_time": _round_time(boundary_time),
                "new_boundary_time": _round_time(resolved_boundary),
                "left_span_seconds": _round_time(left_span),
                "right_span_seconds": _round_time(right_span),
            }
        )

    fixed_boundaries = _resolve_boundaries_with_monotonic_guard(boundaries, duration_seconds=duration_seconds)
    rewritten_big: list[dict[str, Any]] = []
    for index, base_item in enumerate(big_segments_a1):
        rewritten_big.append(
            {
                "segment_id": str(base_item.get("segment_id", f"big_{index + 1:03d}")),
                "start_time": _round_time(fixed_boundaries[index]),
                "end_time": _round_time(fixed_boundaries[index + 1]),
                "label": str(base_item.get("label", "unknown")),
            }
        )
    return rewritten_big, events


def _has_effective_lyric_tokens(sentence_item: dict[str, Any]) -> bool:
    """
    功能说明：判断句级单元是否包含有效歌词token。
    参数说明：
    - sentence_item: 句级歌词对象。
    返回值：
    - bool: 包含有效token返回True。
    异常说明：无。
    边界条件：纯空白/纯标点视为无效token。
    """
    token_units = sentence_item.get("token_units", [])
    if not isinstance(token_units, list) or not token_units:
        return False
    for token_item in token_units:
        token_text = str(token_item.get("text", "")).strip()
        if not token_text:
            continue
        if any(char.isalnum() or ("\u4e00" <= char <= "\u9fff") for char in token_text):
            return True
    return False


def _estimate_vocal_energy_threshold(vocal_rms_values: list[float]) -> float:
    """
    功能说明：估计近锚点判定的人声能量阈值。
    参数说明：
    - vocal_rms_values: 人声RMS序列。
    返回值：
    - float: 阈值。
    异常说明：无。
    边界条件：空输入回退静音下界。
    """
    safe_values = [max(0.0, _safe_float(item, 0.0)) for item in vocal_rms_values]
    if not safe_values:
        return RMS_SILENCE_FLOOR
    return max(RMS_SILENCE_FLOOR, _quantile(safe_values, 0.35))


def _has_vocal_energy_in_window(
    window_start: float,
    window_end: float,
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    vocal_energy_threshold: float,
) -> bool:
    """
    功能说明：判断窗口内是否有人声能量。
    参数说明：
    - window_start/window_end: 窗口起止。
    - vocal_rms_times/vocal_rms_values: 人声RMS序列。
    - vocal_energy_threshold: 阈值。
    返回值：
    - bool: 命中返回True。
    异常说明：无。
    边界条件：无样本时回退中点采样。
    """
    if window_end - window_start <= EPSILON_SECONDS:
        return False
    samples: list[float] = []
    for index, time_item in enumerate(vocal_rms_times):
        time_value = _safe_float(time_item, window_start)
        if window_start <= time_value <= window_end and index < len(vocal_rms_values):
            samples.append(max(0.0, _safe_float(vocal_rms_values[index], 0.0)))
    if not samples:
        mid_time = (window_start + window_end) / 2.0
        samples = [_rms_value_at(mid_time, vocal_rms_times, vocal_rms_values)]
    peak_rms = max(samples) if samples else 0.0
    return peak_rms > max(RMS_SILENCE_FLOOR, float(vocal_energy_threshold)) + EPSILON_SECONDS


def _apply_near_anchor_conflict_resolution(
    big_segments_a1: list[dict[str, Any]],
    sentence_units: list[dict[str, Any]],
    duration_seconds: float,
    head_offset_seconds: float,
    near_anchor_seconds: float,
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：在 A1 后执行近锚点冲突裁决。
    参数说明：
    - big_segments_a1: A1大段。
    - sentence_units: 句级歌词。
    - duration_seconds: 音频总时长。
    - head_offset_seconds: 句首锚点偏移。
    - near_anchor_seconds: 近锚点阈值。
    - vocal_rms_times/vocal_rms_values: 人声RMS序列。
    返回值：
    - tuple[list[dict], dict]: (裁决后A1, 裁决报告)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅处理“边界后最近句首锚点”。
    """
    if len(big_segments_a1) <= 1:
        return list(big_segments_a1), {
            "near_anchor_seconds": _round_time(max(0.0, float(near_anchor_seconds))),
            "decisions": [],
            "dropped_head_sentence_indexes": [],
            "summary": {"checked_boundaries": 0, "drop_head_count": 0, "push_boundary_count": 0},
        }

    safe_near_anchor = max(0.0, float(near_anchor_seconds))
    normalized_sentences = normalize_sentence_units(sentence_units=sentence_units, duration_seconds=duration_seconds)
    sentence_heads: list[dict[str, Any]] = []
    for sentence_index, sentence_item in enumerate(normalized_sentences):
        sentence_heads.append(
            {
                "sentence_index": sentence_index,
                "head_time": _extract_sentence_head_time(sentence_item, duration_seconds=duration_seconds, head_offset_seconds=head_offset_seconds),
                "has_effective_lyric_tokens": _has_effective_lyric_tokens(sentence_item),
            }
        )
    sentence_heads.sort(key=lambda item: _safe_float(item.get("head_time", 0.0), 0.0))

    boundaries = [0.0]
    for item in big_segments_a1[:-1]:
        boundaries.append(_safe_float(item.get("end_time", 0.0), 0.0))
    boundaries.append(_round_time(duration_seconds))

    dropped_sentence_indexes: set[int] = set()
    decisions: list[dict[str, Any]] = []
    vocal_threshold = _estimate_vocal_energy_threshold(vocal_rms_values=vocal_rms_values)

    for boundary_index in range(1, len(boundaries) - 1):
        boundary_time = _safe_float(boundaries[boundary_index], 0.0)
        candidate_head: dict[str, Any] | None = None
        for head_item in sentence_heads:
            sentence_index = int(head_item.get("sentence_index", -1))
            if sentence_index in dropped_sentence_indexes:
                continue
            head_time = _safe_float(head_item.get("head_time", 0.0), 0.0)
            gap_seconds = head_time - boundary_time
            if gap_seconds <= EPSILON_SECONDS:
                continue
            if gap_seconds <= safe_near_anchor + EPSILON_SECONDS:
                candidate_head = head_item
            break
        if candidate_head is None:
            continue

        sentence_index = int(candidate_head.get("sentence_index", -1))
        candidate_head_time = _safe_float(candidate_head.get("head_time", boundary_time), boundary_time)
        has_token = bool(candidate_head.get("has_effective_lyric_tokens", False))
        has_vocal = _has_vocal_energy_in_window(
            window_start=boundary_time,
            window_end=candidate_head_time,
            vocal_rms_times=vocal_rms_times,
            vocal_rms_values=vocal_rms_values,
            vocal_energy_threshold=vocal_threshold,
        )

        decision = {
            "boundary_index": boundary_index - 1,
            "old_boundary_time": _round_time(boundary_time),
            "candidate_head_time": _round_time(candidate_head_time),
            "gap_seconds": _round_time(max(0.0, candidate_head_time - boundary_time)),
            "has_effective_lyric_tokens": has_token,
            "has_vocal_energy": has_vocal,
        }

        if has_token and has_vocal:
            dropped_sentence_indexes.add(sentence_index)
            decision["action"] = "drop_near_anchor"
            decision["new_boundary_time"] = _round_time(boundary_time)
            decisions.append(decision)
            continue

        min_allowed = boundaries[boundary_index - 1] + BOUNDARY_MIN_STEP_SECONDS
        max_allowed = boundaries[boundary_index + 1] - BOUNDARY_MIN_STEP_SECONDS
        if max_allowed <= min_allowed:
            decision["action"] = "keep_original_invalid_room"
            decision["new_boundary_time"] = _round_time(boundary_time)
            decisions.append(decision)
            continue

        moved_boundary = _round_time(max(min_allowed, min(candidate_head_time, max_allowed)))
        boundaries[boundary_index] = moved_boundary
        decision["action"] = "push_big_boundary_to_head"
        decision["new_boundary_time"] = moved_boundary
        decisions.append(decision)

    fixed_boundaries = _resolve_boundaries_with_monotonic_guard(boundaries, duration_seconds=duration_seconds)
    output_big: list[dict[str, Any]] = []
    for index, base_item in enumerate(big_segments_a1):
        output_big.append(
            {
                "segment_id": str(base_item.get("segment_id", f"big_{index + 1:03d}")),
                "start_time": _round_time(fixed_boundaries[index]),
                "end_time": _round_time(fixed_boundaries[index + 1]),
                "label": str(base_item.get("label", "unknown")),
            }
        )

    return output_big, {
        "near_anchor_seconds": _round_time(safe_near_anchor),
        "vocal_energy_threshold": _round_time(vocal_threshold),
        "decisions": decisions,
        "dropped_head_sentence_indexes": sorted(dropped_sentence_indexes),
        "summary": {
            "checked_boundaries": len(decisions),
            "drop_head_count": sum(1 for item in decisions if str(item.get("action", "")) == "drop_near_anchor"),
            "push_boundary_count": sum(1 for item in decisions if str(item.get("action", "")) == "push_big_boundary_to_head"),
        },
    }


def _resolve_big_segment_id(timestamp: float, big_segments: list[dict[str, Any]]) -> str:
    """
    功能说明：按时间点定位所属big段ID。
    参数说明：
    - timestamp: 时间点。
    - big_segments: big段列表。
    返回值：
    - str: big段ID。
    异常说明：无。
    边界条件：未命中时回退最近big段。
    """
    if not big_segments:
        return ""
    for item in big_segments:
        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = _safe_float(item.get("end_time", start_time), start_time)
        if start_time - EPSILON_SECONDS <= timestamp <= end_time + EPSILON_SECONDS:
            return str(item.get("segment_id", ""))
    nearest = min(big_segments, key=lambda item: abs(_safe_float(item.get("start_time", 0.0), 0.0) - timestamp))
    return str(nearest.get("segment_id", ""))


def _resolve_big_label(timestamp: float, big_segments: list[dict[str, Any]]) -> str:
    """
    功能说明：按时间点定位所属big段label。
    参数说明：
    - timestamp: 时间点。
    - big_segments: big段列表。
    返回值：
    - str: big段label。
    异常说明：无。
    边界条件：未命中时回退最近big段。
    """
    if not big_segments:
        return "unknown"
    for item in big_segments:
        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = _safe_float(item.get("end_time", start_time), start_time)
        if start_time - EPSILON_SECONDS <= timestamp <= end_time + EPSILON_SECONDS:
            return str(item.get("label", "unknown"))
    nearest = min(big_segments, key=lambda item: abs(_safe_float(item.get("start_time", 0.0), 0.0) - timestamp))
    return str(nearest.get("label", "unknown"))


def _resolve_window_role(timestamp: float, windows: list[dict[str, Any]]) -> str:
    """
    功能说明：按时间点定位所属窗口角色。
    参数说明：
    - timestamp: 时间点。
    - windows: 窗口列表。
    返回值：
    - str: 窗口角色。
    异常说明：无。
    边界条件：未命中时回退 silence。
    """
    for item in windows:
        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = _safe_float(item.get("end_time", start_time), start_time)
        if start_time - EPSILON_SECONDS <= timestamp <= end_time + EPSILON_SECONDS:
            return str(item.get("role", "silence")).lower().strip() or "silence"
    return "silence"


def build_small_timestamps(
    windows_merged: list[dict[str, Any]],
    duration_seconds: float,
) -> list[float]:
    """
    功能说明：由窗口边界生成小时戳集合。
    参数说明：
    - windows_merged: 并段后窗口。
    - duration_seconds: 音频总时长。
    返回值：
    - list[float]: 归一化小时戳。
    异常说明：无。
    边界条件：必含0与duration，且不再额外并入big边界。
    """
    points = {0.0, _round_time(duration_seconds)}
    for item in windows_merged:
        points.add(_round_time(_safe_float(item.get("start_time", 0.0), 0.0)))
        points.add(_round_time(_safe_float(item.get("end_time", 0.0), 0.0)))
    normalized = sorted(point for point in points if 0.0 <= point <= _round_time(duration_seconds))
    if not normalized:
        return [0.0, _round_time(duration_seconds)]
    return normalized


def build_final_segments(
    windows_merged: list[dict[str, Any]],
    big_segments_a1: list[dict[str, Any]],
    small_timestamps: list[float],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：按小时戳切分并映射到最终S段。
    参数说明：
    - windows_merged: 并段后窗口。
    - big_segments_a1: A1大段。
    - small_timestamps: 小时戳集合。
    - duration_seconds: 音频总时长。
    返回值：
    - list[dict[str, Any]]: 最终segments（含 role）。
    异常说明：无。
    边界条件：segments覆盖全时长且无重叠倒序。
    """
    if not small_timestamps:
        small_timestamps = [0.0, _round_time(duration_seconds)]
    cuts = sorted({_round_time(_clamp_time(_safe_float(item, 0.0), duration_seconds)) for item in small_timestamps})
    if len(cuts) < 2:
        cuts = [0.0, _round_time(duration_seconds)]

    segments: list[dict[str, Any]] = []
    for index in range(len(cuts) - 1):
        start_time = _safe_float(cuts[index], 0.0)
        end_time = max(start_time, _safe_float(cuts[index + 1], start_time))
        if end_time - start_time <= EPSILON_SECONDS:
            continue
        mid_time = (start_time + end_time) / 2.0
        segments.append(
            {
                "segment_id": f"seg_{len(segments) + 1:04d}",
                "big_segment_id": _resolve_big_segment_id(mid_time, big_segments_a1),
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "label": _resolve_big_label(mid_time, big_segments_a1),
                "role": _resolve_window_role(mid_time, windows_merged),
            }
        )

    if not segments:
        return []
    segments[0]["start_time"] = 0.0
    for index in range(1, len(segments)):
        segments[index]["start_time"] = _round_time(_safe_float(segments[index - 1].get("end_time", 0.0), 0.0))
    segments[-1]["end_time"] = _round_time(duration_seconds)
    return segments


def _snap_big_boundaries_to_segment_cuts(
    big_segments: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    功能说明：将 big 内部边界吸附到最终 S 段切点，确保 big/seg 时间对齐。
    参数说明：
    - big_segments: 原 big 段列表。
    - segments: 最终 S 段列表。
    - duration_seconds: 音频总时长。
    返回值：
    - tuple[list[dict], list[dict]]: (吸附后 big 段, 吸附事件列表)。
    异常说明：无。
    边界条件：仅在合法区间存在可用切点时才移动边界。
    """
    if len(big_segments) <= 1 or not segments:
        return list(big_segments), []

    safe_duration = _round_time(max(0.0, float(duration_seconds)))
    segment_cuts = sorted(
        {
            _round_time(_clamp_time(_safe_float(item.get("start_time", 0.0), 0.0), safe_duration))
            for item in segments
        }
        | {
            _round_time(_clamp_time(_safe_float(item.get("end_time", 0.0), 0.0), safe_duration))
            for item in segments
        }
    )
    if len(segment_cuts) <= 2:
        return list(big_segments), []

    boundaries = [0.0]
    for item in big_segments[:-1]:
        boundaries.append(_safe_float(item.get("end_time", 0.0), 0.0))
    boundaries.append(safe_duration)

    events: list[dict[str, Any]] = []
    for boundary_index in range(1, len(boundaries) - 1):
        old_boundary = _safe_float(boundaries[boundary_index], 0.0)
        lower_bound = boundaries[boundary_index - 1] + BOUNDARY_MIN_STEP_SECONDS
        upper_bound = boundaries[boundary_index + 1] - BOUNDARY_MIN_STEP_SECONDS
        if upper_bound <= lower_bound:
            continue
        candidates = [point for point in segment_cuts if lower_bound < point < upper_bound]
        if not candidates:
            continue
        new_boundary = min(candidates, key=lambda point: abs(point - old_boundary))
        if abs(new_boundary - old_boundary) <= EPSILON_SECONDS:
            continue
        boundaries[boundary_index] = _round_time(new_boundary)
        events.append(
            {
                "boundary_index": boundary_index - 1,
                "old_boundary_time": _round_time(old_boundary),
                "new_boundary_time": _round_time(new_boundary),
                "action": "snap_to_segment_cut",
            }
        )

    fixed_boundaries = _resolve_boundaries_with_monotonic_guard(boundaries, duration_seconds=safe_duration)
    snapped_big: list[dict[str, Any]] = []
    for index, base_item in enumerate(big_segments):
        snapped_big.append(
            {
                "segment_id": str(base_item.get("segment_id", f"big_{index + 1:03d}")),
                "start_time": _round_time(fixed_boundaries[index]),
                "end_time": _round_time(fixed_boundaries[index + 1]),
                "label": str(base_item.get("label", "unknown")),
            }
        )
    return snapped_big, events


def resolve_big_timestamps_and_segments(
    big_segments_a0: list[dict[str, Any]],
    windows_merged: list[dict[str, Any]],
    sentence_units: list[dict[str, Any]],
    duration_seconds: float,
    head_offset_seconds: float,
    near_anchor_seconds: float,
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    功能说明：统一完成 A1 求解、近锚点裁决与最终S段构建。
    参数说明：
    - big_segments_a0: A0大段列表。
    - windows_merged: 并段后窗口列表。
    - sentence_units: 句级歌词列表。
    - duration_seconds: 音频总时长。
    - head_offset_seconds: 句首锚点偏移。
    - near_anchor_seconds: 近锚点阈值。
    - vocal_rms_times/vocal_rms_values: 人声RMS序列。
    返回值：
    - tuple[list[dict], list[float], list[dict], dict, dict]:
      (A1大段, 小时戳, 最终S段, near_anchor报告, boundary_moves报告)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：A1只做单次矫正，不做二次big联动回写。
    """
    a1_by_lyric, lyric_overlap_traces = _build_a1_from_a0_and_sentences(
        big_segments_a0=big_segments_a0,
        sentence_units=sentence_units,
        duration_seconds=duration_seconds,
    )
    a1_by_other, other_ratio_events = _apply_other_window_ratio_correction(
        big_segments_a1=a1_by_lyric,
        windows_merged=windows_merged,
        duration_seconds=duration_seconds,
    )
    a1_final, near_anchor_report = _apply_near_anchor_conflict_resolution(
        big_segments_a1=a1_by_other,
        sentence_units=sentence_units,
        duration_seconds=duration_seconds,
        head_offset_seconds=head_offset_seconds,
        near_anchor_seconds=near_anchor_seconds,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
    )
    small_timestamps = build_small_timestamps(
        windows_merged=windows_merged,
        duration_seconds=duration_seconds,
    )
    segments_final = build_final_segments(
        windows_merged=windows_merged,
        big_segments_a1=a1_final,
        small_timestamps=small_timestamps,
        duration_seconds=duration_seconds,
    )
    a1_snapped, seg_cut_snap_events = _snap_big_boundaries_to_segment_cuts(
        big_segments=a1_final,
        segments=segments_final,
        duration_seconds=duration_seconds,
    )
    if seg_cut_snap_events:
        segments_final = build_final_segments(
            windows_merged=windows_merged,
            big_segments_a1=a1_snapped,
            small_timestamps=small_timestamps,
            duration_seconds=duration_seconds,
        )
    else:
        a1_snapped = a1_final

    boundary_moves = {
        "lyric_overlap_traces": lyric_overlap_traces,
        "other_window_ratio_moves": other_ratio_events,
        "seg_cut_snap_moves": seg_cut_snap_events,
        "summary": {
            "lyric_overlap_boundary_count": len(lyric_overlap_traces),
            "other_window_ratio_count": len(other_ratio_events),
            "seg_cut_snap_count": len(seg_cut_snap_events),
        },
    }
    return a1_snapped, small_timestamps, segments_final, near_anchor_report, boundary_moves
