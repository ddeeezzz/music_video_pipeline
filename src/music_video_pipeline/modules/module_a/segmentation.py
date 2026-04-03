"""
文件用途：提供模块A的小时戳筛选与分段构建逻辑。
核心流程：根据段落类型、歌词与声学候选生成连续小段落。
输入输出：输入大段落和候选时间戳，输出 segments 与 beats 结构。
依赖说明：依赖项目内 timing_energy 时间工具。
维护说明：保持节拍驱动和时间轴连续性约束。
"""

# 标准库：类型提示
from typing import Any

# 项目内模块：时间与能量工具
from music_video_pipeline.modules.module_a.timing_energy import (
    _clamp_time,
    _find_big_segment,
    _normalize_timestamp_list,
    _rms_delta_at,
    _rms_value_at,
    _round_time,
    _snap_to_nearest_beat,
)


def _select_small_timestamps(
    duration_seconds: float,
    big_segments: list[dict[str, Any]],
    beat_candidates: list[float],
    onset_candidates: list[float],
    rms_times: list[float],
    rms_values: list[float],
    lyric_sentence_starts: list[float],
    instrumental_labels: list[str],
    snap_threshold_ms: int,
) -> list[float]:
    """
    功能说明：按段落类型筛选最终小时间戳。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - beat_candidates: 节拍候选时间戳列表（秒）。
    - onset_candidates: 起音候选时间戳列表（秒）。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    - lyric_sentence_starts: 歌词句级起始时间列表（秒）。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - snap_threshold_ms: 歌词吸附到节拍的阈值（毫秒）。
    返回值：
    - list[float]: 筛选后的小时间戳列表（秒）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    snap_threshold_seconds = max(0.0, snap_threshold_ms / 1000.0)
    instrumental_set = {label.lower().strip() for label in instrumental_labels}

    beat_pool = _normalize_timestamp_list(beat_candidates + [0.0, duration_seconds], duration_seconds)
    onset_pool = _normalize_timestamp_list(onset_candidates + [0.0, duration_seconds], duration_seconds)
    timestamps: list[float] = [0.0, duration_seconds]

    for big_segment in big_segments:
        start_time = float(big_segment["start_time"])
        end_time = float(big_segment["end_time"])
        label = str(big_segment.get("label", "")).lower().strip()

        beat_in_segment = [value for value in beat_pool if start_time < value < end_time]
        onset_in_segment = [value for value in onset_pool if start_time < value < end_time]

        if label in instrumental_set:
            if onset_in_segment:
                peak_onset = max(onset_in_segment, key=lambda item: _rms_delta_at(item, rms_times, rms_values))
                peak_delta = _rms_delta_at(peak_onset, rms_times, rms_values)
                if peak_delta <= 1e-6:
                    peak_onset = max(onset_in_segment, key=lambda item: _rms_value_at(item, rms_times, rms_values))
                timestamps.append(peak_onset)
            elif beat_in_segment:
                timestamps.append(beat_in_segment[len(beat_in_segment) // 2])
            else:
                timestamps.append((start_time + end_time) / 2.0)
            timestamps.extend(beat_in_segment[::2] if len(beat_in_segment) > 4 else beat_in_segment)
            continue

        lyric_in_segment = [value for value in lyric_sentence_starts if start_time <= value <= end_time]
        if lyric_in_segment:
            for lyric_time in lyric_in_segment:
                timestamps.append(_snap_to_nearest_beat(lyric_time, beat_pool, snap_threshold_seconds))
        elif beat_in_segment:
            timestamps.extend(beat_in_segment[::4] if len(beat_in_segment) > 3 else beat_in_segment)
        elif onset_in_segment:
            timestamps.append(onset_in_segment[len(onset_in_segment) // 2])
        else:
            timestamps.append((start_time + end_time) / 2.0)

    return _normalize_timestamp_list(timestamps, duration_seconds)


def _build_small_segments(timestamps: list[float], big_segments: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：由相邻小时间戳构建最小视觉单元。
    参数说明：
    - timestamps: 时间戳列表（秒）。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 构建后的小分段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized_times = _normalize_timestamp_list(timestamps, duration_seconds)
    if len(normalized_times) < 2:
        normalized_times = [0.0, _round_time(duration_seconds)]

    segments: list[dict[str, Any]] = []
    for index in range(len(normalized_times) - 1):
        start_time = normalized_times[index]
        end_time = normalized_times[index + 1]
        if end_time - start_time < 0.1:
            continue
        mid_time = (start_time + end_time) / 2.0
        big_segment = _find_big_segment(mid_time, big_segments)
        segments.append(
            {
                "segment_id": f"seg_{len(segments) + 1:04d}",
                "big_segment_id": str(big_segment["segment_id"]),
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "label": str(big_segment.get("label", "unknown")),
            }
        )

    if not segments:
        fallback_big_segment = big_segments[0]
        segments.append(
            {
                "segment_id": "seg_0001",
                "big_segment_id": str(fallback_big_segment["segment_id"]),
                "start_time": 0.0,
                "end_time": _round_time(duration_seconds),
                "label": str(fallback_big_segment.get("label", "unknown")),
            }
        )

    segments[0]["start_time"] = 0.0
    segments[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(segments)):
        segments[index]["start_time"] = segments[index - 1]["end_time"]
    return segments


def _build_segments_with_lyric_priority(
    duration_seconds: float,
    big_segments: list[dict[str, Any]],
    beat_candidates: list[float],
    onset_candidates: list[float],
    lyric_units: list[dict[str, Any]],
    instrumental_labels: list[str],
    rms_times: list[float] | None = None,
    rms_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：在保持节拍驱动的前提下，对人声段应用“歌词句优先”分段。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - beat_candidates: 节拍候选时间戳列表（秒）。
    - onset_candidates: 起音候选时间戳列表（秒）。
    - lyric_units: 歌词单元列表。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    返回值：
    - list[dict[str, Any]]: 构建后的小分段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not big_segments:
        return _build_small_segments([0.0, duration_seconds], big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": duration_seconds, "label": "unknown"}], duration_seconds=duration_seconds)  # noqa: E501

    beat_pool = _normalize_timestamp_list(beat_candidates + [0.0, duration_seconds], duration_seconds)
    onset_pool = _normalize_timestamp_list(onset_candidates + [0.0, duration_seconds], duration_seconds)
    safe_rms_times = rms_times or []
    safe_rms_values = rms_values or []
    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    small_gap_merge_seconds = 0.35

    range_items: list[dict[str, Any]] = []
    spill_end = 0.0
    for big_segment in big_segments:
        start_time = max(float(big_segment["start_time"]), spill_end)
        end_time = float(big_segment["end_time"])
        label = str(big_segment.get("label", "unknown")).lower().strip()
        big_segment_id = str(big_segment["segment_id"])
        if end_time - start_time <= 1e-6:
            continue

        lyric_in_segment = [
            item
            for item in lyric_units
            if start_time <= float(item.get("start_time", 0.0)) < end_time
        ]
        lyric_in_segment.sort(key=lambda item: float(item.get("start_time", 0.0)))

        if label in instrumental_set:
            for left_time, right_time in _split_instrumental_range_once_by_energy(
                start_time=start_time,
                end_time=end_time,
                beat_pool=beat_pool,
                onset_pool=onset_pool,
                rms_times=safe_rms_times,
                rms_values=safe_rms_values,
            ):
                range_items.append(
                    {
                        "start_time": left_time,
                        "end_time": right_time,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": False,
                    }
                )
            continue

        if not lyric_in_segment:
            for left_time, right_time in _split_range_by_rhythm(
                start_time=start_time,
                end_time=end_time,
                beat_pool=beat_pool,
                onset_pool=onset_pool,
                dense_mode=False,
            ):
                range_items.append(
                    {
                        "start_time": left_time,
                        "end_time": right_time,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": False,
                    }
                )
            continue

        cursor = start_time
        for lyric_item in lyric_in_segment:
            lyric_start = _clamp_time(float(lyric_item.get("start_time", cursor)), duration_seconds)
            lyric_end = _clamp_time(float(lyric_item.get("end_time", lyric_start)), duration_seconds)
            lyric_start = max(cursor, lyric_start)
            lyric_end = max(lyric_start, lyric_end)

            if lyric_start > cursor + 1e-6:
                # 句间极短空档不单独拆镜头，避免“歌词尚未唱完即跳图”的闪切观感。
                if lyric_start - cursor <= small_gap_merge_seconds:
                    lyric_start = cursor
                else:
                    for left_time, right_time in _split_range_by_rhythm(
                        start_time=cursor,
                        end_time=lyric_start,
                        beat_pool=beat_pool,
                        onset_pool=onset_pool,
                        dense_mode=False,
                    ):
                        range_items.append(
                            {
                                "start_time": left_time,
                                "end_time": right_time,
                                "big_segment_id": big_segment_id,
                                "label": label,
                                "lyric_anchor": False,
                            }
                        )

            if lyric_end > lyric_start + 1e-6:
                # 关键约束：一句歌词独占一个最小视觉单元，禁止句内再切段。
                range_items.append(
                    {
                        "start_time": lyric_start,
                        "end_time": lyric_end,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": True,
                    }
                )
            cursor = max(cursor, lyric_end)
            spill_end = max(spill_end, lyric_end)

        if cursor < end_time - 1e-6:
            if end_time - cursor <= small_gap_merge_seconds and range_items:
                range_items[-1]["end_time"] = end_time
            else:
                for left_time, right_time in _split_range_by_rhythm(
                    start_time=cursor,
                    end_time=end_time,
                    beat_pool=beat_pool,
                    onset_pool=onset_pool,
                    dense_mode=False,
                ):
                    range_items.append(
                        {
                            "start_time": left_time,
                            "end_time": right_time,
                            "big_segment_id": big_segment_id,
                            "label": label,
                            "lyric_anchor": False,
                        }
                    )

    normalized_ranges = _normalize_segment_ranges(range_items=range_items, duration_seconds=duration_seconds)
    normalized_ranges = _merge_short_vocal_non_lyric_ranges(
        range_items=normalized_ranges,
        duration_seconds=duration_seconds,
        instrumental_set=instrumental_set,
        min_duration_seconds=0.8,
    )
    if not normalized_ranges:
        return _build_small_segments(
            timestamps=[0.0, duration_seconds],
            big_segments=big_segments,
            duration_seconds=duration_seconds,
        )

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(normalized_ranges):
        segments.append(
            {
                "segment_id": f"seg_{index + 1:04d}",
                "big_segment_id": str(item["big_segment_id"]),
                "start_time": _round_time(float(item["start_time"])),
                "end_time": _round_time(float(item["end_time"])),
                "label": str(item["label"]),
            }
        )

    segments[0]["start_time"] = 0.0
    segments[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(segments)):
        segments[index]["start_time"] = segments[index - 1]["end_time"]
    return segments


def _split_instrumental_range_once_by_energy(
    start_time: float,
    end_time: float,
    beat_pool: list[float],
    onset_pool: list[float],
    rms_times: list[float],
    rms_values: list[float],
    long_segment_threshold_seconds: float = 4.0,
    min_side_duration_seconds: float = 0.8,
) -> list[tuple[float, float]]:
    """
    功能说明：器乐长段只切一次，在能量落差最大处生成两个分段。
    参数说明：
    - start_time/end_time: 当前器乐段起止时间。
    - beat_pool/onset_pool: 节拍与起音候选点。
    - rms_times/rms_values: RMS 时序，用于能量落差计算。
    - long_segment_threshold_seconds: 触发单次切分的最小时长阈值。
    - min_side_duration_seconds: 切点两侧最小保留时长。
    返回值：
    - list[tuple[float, float]]: 分段区间列表（短段1个，长段最多2个）。
    异常说明：无。
    边界条件：候选点不可用时回退到区间中点，仍保证不按节拍连续切分。
    """
    duration = max(0.0, end_time - start_time)
    if duration <= 0.12:
        return [(start_time, end_time)] if end_time > start_time else []
    if duration < long_segment_threshold_seconds:
        return [(start_time, end_time)]

    left_bound = start_time + min_side_duration_seconds
    right_bound = end_time - min_side_duration_seconds
    if right_bound <= left_bound:
        split_time = (start_time + end_time) / 2.0
    else:
        onset_candidates = [value for value in onset_pool if left_bound <= value <= right_bound]
        beat_candidates = [value for value in beat_pool if left_bound <= value <= right_bound]
        candidate_pool = onset_candidates if onset_candidates else beat_candidates
        if candidate_pool:
            split_time = max(candidate_pool, key=lambda value: _rms_delta_at(value, rms_times, rms_values))
            peak_delta = _rms_delta_at(split_time, rms_times, rms_values)
            if peak_delta <= 1e-6:
                split_time = max(candidate_pool, key=lambda value: _rms_value_at(value, rms_times, rms_values))
        else:
            split_time = (start_time + end_time) / 2.0

    split_time = _round_time(split_time)
    if split_time - start_time < 0.12 or end_time - split_time < 0.12:
        return [(start_time, end_time)]
    return [(start_time, split_time), (split_time, end_time)]


def _split_range_by_rhythm(
    start_time: float,
    end_time: float,
    beat_pool: list[float],
    onset_pool: list[float],
    dense_mode: bool,
) -> list[tuple[float, float]]:
    """
    功能说明：在无歌词区间按节拍/起音构建子区间。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - beat_pool: 归一化后的节拍候选池（秒）。
    - onset_pool: 归一化后的起音候选池（秒）。
    - dense_mode: 是否使用更密集的节拍切分模式。
    返回值：
    - list[tuple[float, float]]: 切分后的区间或单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if end_time - start_time <= 0.12:
        return [(start_time, end_time)] if end_time > start_time else []

    beat_in_range = [value for value in beat_pool if start_time < value < end_time]
    onset_in_range = [value for value in onset_pool if start_time < value < end_time]

    if beat_in_range:
        step = 2 if dense_mode else 4
        if len(beat_in_range) <= step:
            selected = beat_in_range
        else:
            selected = beat_in_range[::step]
    elif onset_in_range:
        selected = [onset_in_range[len(onset_in_range) // 2]]
    else:
        selected = []

    boundaries = [start_time, *selected, end_time]
    boundaries = sorted(set(_round_time(item) for item in boundaries if start_time <= item <= end_time))
    if boundaries[0] > start_time:
        boundaries.insert(0, _round_time(start_time))
    if boundaries[-1] < end_time:
        boundaries.append(_round_time(end_time))

    ranges: list[tuple[float, float]] = []
    for index in range(len(boundaries) - 1):
        left_time = float(boundaries[index])
        right_time = float(boundaries[index + 1])
        if right_time - left_time < 0.08:
            continue
        ranges.append((left_time, right_time))

    if not ranges:
        return [(start_time, end_time)]
    return ranges


def _normalize_segment_ranges(range_items: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：规范化区间列表，保证覆盖连续且无非法时长。
    参数说明：
    - range_items: 待归一化或合并的分段区间列表。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not range_items:
        return []

    sorted_items = sorted(range_items, key=lambda item: float(item.get("start_time", 0.0)))
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(sorted_items):
        original_end = _clamp_time(float(item.get("end_time", cursor)), duration_seconds)
        if index == len(sorted_items) - 1:
            end_time = _round_time(duration_seconds)
        else:
            end_time = _round_time(max(cursor, original_end))
        if end_time - cursor < 0.05:
            continue
        normalized.append(
            {
                "start_time": _round_time(cursor),
                "end_time": _round_time(end_time),
                "big_segment_id": str(item.get("big_segment_id", "")),
                "label": str(item.get("label", "unknown")),
                "lyric_anchor": bool(item.get("lyric_anchor", False)),
            }
        )
        cursor = end_time
        if cursor >= duration_seconds - 1e-6:
            break

    if not normalized:
        return []

    normalized[0]["start_time"] = 0.0
    normalized[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(normalized)):
        normalized[index]["start_time"] = normalized[index - 1]["end_time"]
    return normalized


def _merge_short_vocal_non_lyric_ranges(
    range_items: list[dict[str, Any]],
    duration_seconds: float,
    instrumental_set: set[str],
    min_duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：合并人声区过短的非歌词段，降低镜头闪切。
    参数说明：
    - range_items: 待归一化或合并的分段区间列表。
    - duration_seconds: 音频总时长（秒）。
    - instrumental_set: 业务处理所需输入参数。
    - min_duration_seconds: 最小时长阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 合并后的单元或区间结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not range_items:
        return []

    merged_ranges = [dict(item) for item in range_items]
    index = 0
    while index < len(merged_ranges):
        item = merged_ranges[index]
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        duration = max(0.0, end_time - start_time)
        label = str(item.get("label", "unknown")).lower().strip()
        is_lyric = bool(item.get("lyric_anchor", False))

        should_merge = (
            label not in instrumental_set
            and not is_lyric
            and duration < min_duration_seconds
            and len(merged_ranges) > 1
        )
        if not should_merge:
            index += 1
            continue

        if index <= 0:
            merged_ranges[1]["start_time"] = start_time
            del merged_ranges[index]
            continue
        if index >= len(merged_ranges) - 1:
            merged_ranges[index - 1]["end_time"] = end_time
            del merged_ranges[index]
            index = max(0, index - 1)
            continue

        prev_item = merged_ranges[index - 1]
        next_item = merged_ranges[index + 1]
        prev_duration = float(prev_item.get("end_time", 0.0)) - float(prev_item.get("start_time", 0.0))
        next_duration = float(next_item.get("end_time", 0.0)) - float(next_item.get("start_time", 0.0))
        prev_same_label = str(prev_item.get("label", "")).lower().strip() == label
        next_same_label = str(next_item.get("label", "")).lower().strip() == label

        merge_to_prev = False
        if prev_same_label and not next_same_label:
            merge_to_prev = True
        elif next_same_label and not prev_same_label:
            merge_to_prev = False
        else:
            merge_to_prev = prev_duration >= next_duration

        if merge_to_prev:
            prev_item["end_time"] = end_time
            del merged_ranges[index]
            index = max(0, index - 1)
            continue

        next_item["start_time"] = start_time
        del merged_ranges[index]

    return _normalize_segment_ranges(range_items=merged_ranges, duration_seconds=duration_seconds)
