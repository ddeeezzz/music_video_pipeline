"""
文件用途：提供模块A的小时戳筛选与分段构建逻辑。
核心流程：根据段落类型、歌词与声学候选生成连续小段落。
输入输出：输入大段落和候选时间戳，输出 segments 与 beats 结构。
依赖说明：依赖项目内 timing_energy 时间工具。
维护说明：保持节拍驱动和时间轴连续性约束。
"""

# 标准库：类型提示
from bisect import bisect_left, bisect_right
# 标准库：数据类定义
from dataclasses import dataclass
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

# 常量：inst 边界歌词保护窗口大小（秒），窗口内歌词可触发局部人声微段切分。
BOUNDARY_LYRIC_PROTECTION_WINDOW_SECONDS = 0.35
# 常量：边界歌词保护切出的最小微段时长（秒），避免生成噪声级碎片。
BOUNDARY_LYRIC_MIN_MICRO_DURATION_SECONDS = 0.06
# 常量：跨标签/跨大段时，vocal-vocal 间短 inst 空挡并合阈值（秒）。
INST_GAP_MERGE_CROSS_THRESHOLD_SECONDS = 1.2
# 常量：同标签且同大段时，vocal-vocal 间短 inst 空挡并合阈值（秒）。
INST_GAP_MERGE_SAME_GROUP_THRESHOLD_SECONDS = 1.4


@dataclass(frozen=True)
class SegmentationTuning:
    """
    功能说明：承载模块A分段链路的调参参数，统一由单一入口归一化。
    参数说明：
    - vocal_energy_enter_quantile: 人声音量进入阈值分位点（0~1）。
    - vocal_energy_exit_quantile: 人声音量退出阈值分位点（0~1，且不高于 enter）。
    - mid_segment_min_duration_seconds: 人声中间段最小时长阈值（秒）。
    - short_vocal_non_lyric_merge_seconds: 人声“无歌词/短歌词”短段合并阈值（秒）。
    - instrumental_single_split_min_seconds: 器乐单次切分触发最小时长（秒）。
    - accent_delta_trigger_ratio: 首重音检测的能量突变触发比例（0~1）。
    - lyric_sentence_gap_merge_seconds: 歌词句间空档并入阈值（秒）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：本对象中的值默认已完成归一化与安全裁剪。
    """

    vocal_energy_enter_quantile: float
    vocal_energy_exit_quantile: float
    mid_segment_min_duration_seconds: float
    short_vocal_non_lyric_merge_seconds: float
    instrumental_single_split_min_seconds: float
    accent_delta_trigger_ratio: float
    lyric_sentence_gap_merge_seconds: float


def _build_segmentation_tuning(
    vocal_energy_enter_quantile: float = 0.70,
    vocal_energy_exit_quantile: float = 0.45,
    mid_segment_min_duration_seconds: float = 0.8,
    short_vocal_non_lyric_merge_seconds: float = 1.2,
    instrumental_single_split_min_seconds: float = 4.0,
    accent_delta_trigger_ratio: float = 0.35,
    lyric_sentence_gap_merge_seconds: float = 0.35,
) -> SegmentationTuning:
    """
    功能说明：构建并归一化分段调参对象，作为阈值生效的唯一入口。
    参数说明：
    - vocal_energy_enter_quantile: 人声音量进入阈值分位点。
    - vocal_energy_exit_quantile: 人声音量退出阈值分位点。
    - mid_segment_min_duration_seconds: 人声中间段最小时长阈值（秒）。
    - short_vocal_non_lyric_merge_seconds: 人声“无歌词/短歌词”短段合并阈值（秒）。
    - instrumental_single_split_min_seconds: 器乐单次切分触发最小时长（秒）。
    - accent_delta_trigger_ratio: 首重音检测的能量突变触发比例（0~1）。
    - lyric_sentence_gap_merge_seconds: 歌词句间空档并入阈值（秒）。
    返回值：
    - SegmentationTuning: 归一化后的分段调参对象。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：对非法值进行安全裁剪，确保算法内部无需重复裁剪。
    """
    safe_enter_quantile = max(0.0, min(1.0, float(vocal_energy_enter_quantile)))
    safe_exit_quantile = max(0.0, min(safe_enter_quantile, float(vocal_energy_exit_quantile)))
    safe_mid_segment_min_duration = max(0.1, float(mid_segment_min_duration_seconds))
    safe_short_vocal_non_lyric_merge = max(0.1, float(short_vocal_non_lyric_merge_seconds))
    safe_instrumental_single_split_min = max(0.1, float(instrumental_single_split_min_seconds))
    safe_accent_delta_trigger_ratio = max(0.0, min(1.0, float(accent_delta_trigger_ratio)))
    safe_lyric_sentence_gap_merge = max(0.0, float(lyric_sentence_gap_merge_seconds))
    return SegmentationTuning(
        vocal_energy_enter_quantile=safe_enter_quantile,
        vocal_energy_exit_quantile=safe_exit_quantile,
        mid_segment_min_duration_seconds=safe_mid_segment_min_duration,
        short_vocal_non_lyric_merge_seconds=safe_short_vocal_non_lyric_merge,
        instrumental_single_split_min_seconds=safe_instrumental_single_split_min,
        accent_delta_trigger_ratio=safe_accent_delta_trigger_ratio,
        lyric_sentence_gap_merge_seconds=safe_lyric_sentence_gap_merge,
    )


def _slice_sorted_window(
    sorted_values: list[float],
    start_time: float,
    end_time: float,
    include_left: bool = False,
    include_right: bool = False,
) -> list[float]:
    """
    功能说明：在有序时间轴上使用二分法截取窗口数据。
    参数说明：
    - sorted_values: 升序时间戳数组。
    - start_time: 窗口起点（秒）。
    - end_time: 窗口终点（秒）。
    - include_left: 是否包含左边界。
    - include_right: 是否包含右边界。
    返回值：
    - list[float]: 落入窗口的时间戳列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当输入为空或窗口非法时返回空列表。
    """
    if not sorted_values or end_time < start_time:
        return []
    left_index = bisect_left(sorted_values, start_time) if include_left else bisect_right(sorted_values, start_time)
    right_index = bisect_right(sorted_values, end_time) if include_right else bisect_left(sorted_values, end_time)
    if right_index <= left_index:
        return []
    return sorted_values[left_index:right_index]


def _build_rms_index(rms_times: list[float], rms_values: list[float]) -> dict[str, list[float]]:
    """
    功能说明：构建 RMS 查询索引（有序时间轴 + 前缀和）。
    参数说明：
    - rms_times: RMS 时间轴（秒）。
    - rms_values: RMS 能量值序列。
    返回值：
    - dict[str, list[float]]: 包含 times/values/prefix 三个键的索引字典。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空输入返回空索引。
    """
    points = [
        (float(time_value), float(rms_values[index]))
        for index, time_value in enumerate(rms_times)
        if index < len(rms_values)
    ]
    points.sort(key=lambda item: item[0])
    times = [item[0] for item in points]
    values = [item[1] for item in points]
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return {"times": times, "values": values, "prefix": prefix}


def _slice_rms_points_by_time(
    rms_index: dict[str, list[float]],
    start_time: float,
    end_time: float,
    include_left: bool = True,
    include_right: bool = True,
) -> list[tuple[float, float]]:
    """
    功能说明：在 RMS 索引中按时间窗口抽取采样点。
    参数说明：
    - rms_index: 由 _build_rms_index 构建的索引字典。
    - start_time: 窗口起点（秒）。
    - end_time: 窗口终点（秒）。
    - include_left: 是否包含左边界。
    - include_right: 是否包含右边界。
    返回值：
    - list[tuple[float, float]]: `(time, value)` 采样点列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：窗口无数据时返回空列表。
    """
    times = rms_index.get("times", [])
    values = rms_index.get("values", [])
    if not times or end_time < start_time:
        return []
    left_index = bisect_left(times, start_time) if include_left else bisect_right(times, start_time)
    right_index = bisect_right(times, end_time) if include_right else bisect_left(times, end_time)
    if right_index <= left_index:
        return []
    return [(times[index], values[index]) for index in range(left_index, right_index)]


def _slice_lyric_units_by_start(
    lyric_units_sorted: list[dict[str, Any]],
    lyric_start_times: list[float],
    start_time: float,
    end_time: float,
) -> list[dict[str, Any]]:
    """
    功能说明：按歌词起点时间窗口截取歌词单元，避免全量扫描。
    参数说明：
    - lyric_units_sorted: 按 start_time 升序后的歌词单元。
    - lyric_start_times: 与 lyric_units_sorted 对齐的起点数组。
    - start_time: 窗口起点（秒，包含）。
    - end_time: 窗口终点（秒，不包含）。
    返回值：
    - list[dict[str, Any]]: 窗口内歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无匹配时返回空列表。
    """
    if not lyric_units_sorted or not lyric_start_times or end_time <= start_time:
        return []
    left_index = bisect_left(lyric_start_times, start_time)
    right_index = bisect_left(lyric_start_times, end_time)
    if right_index <= left_index:
        return []
    return lyric_units_sorted[left_index:right_index]


def _slice_lyric_units_by_overlap(
    lyric_units_sorted: list[dict[str, Any]],
    lyric_start_times: list[float],
    start_time: float,
    end_time: float,
) -> list[dict[str, Any]]:
    """
    功能说明：按时间重叠截取歌词单元，支持“歌词早起点跨区间”场景。
    参数说明：
    - lyric_units_sorted: 按 start_time 升序后的歌词单元。
    - lyric_start_times: 与 lyric_units_sorted 对齐的起点数组。
    - start_time: 窗口起点（秒，包含）。
    - end_time: 窗口终点（秒，不包含）。
    返回值：
    - list[dict[str, Any]]: 与窗口有时间重叠的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无匹配时返回空列表。
    """
    if not lyric_units_sorted or not lyric_start_times or end_time <= start_time:
        return []
    right_index = bisect_left(lyric_start_times, end_time)
    if right_index <= 0:
        return []
    left_index = max(0, bisect_left(lyric_start_times, start_time) - 1)
    overlap_units: list[dict[str, Any]] = []
    for index in range(left_index, right_index):
        item = lyric_units_sorted[index]
        lyric_start = float(item.get("start_time", 0.0))
        lyric_end = max(lyric_start, float(item.get("end_time", lyric_start)))
        if lyric_end <= start_time or lyric_start >= end_time:
            continue
        overlap_units.append(item)
    return overlap_units


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
    lyric_pool = sorted(float(item) for item in lyric_sentence_starts)
    timestamps: list[float] = [0.0, duration_seconds]

    for big_segment in big_segments:
        start_time = float(big_segment["start_time"])
        end_time = float(big_segment["end_time"])
        label = str(big_segment.get("label", "")).lower().strip()

        beat_in_segment = _slice_sorted_window(
            sorted_values=beat_pool,
            start_time=start_time,
            end_time=end_time,
            include_left=False,
            include_right=False,
        )
        onset_in_segment = _slice_sorted_window(
            sorted_values=onset_pool,
            start_time=start_time,
            end_time=end_time,
            include_left=False,
            include_right=False,
        )

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

        lyric_in_segment = _slice_sorted_window(
            sorted_values=lyric_pool,
            start_time=start_time,
            end_time=end_time,
            include_left=True,
            include_right=True,
        )
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


def _prepare_segmentation_indexes(
    duration_seconds: float,
    beat_candidates: list[float],
    onset_candidates: list[float],
    lyric_units: list[dict[str, Any]],
    instrumental_labels: list[str],
    rms_times: list[float] | None,
    rms_values: list[float] | None,
    vocal_onset_candidates: list[float] | None,
    vocal_rms_times: list[float] | None,
    vocal_rms_values: list[float] | None,
) -> dict[str, Any]:
    """
    功能说明：准备分段阶段所需索引与候选池，集中处理一次性预计算。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - beat_candidates: 节拍候选时间戳列表（秒）。
    - onset_candidates: 起音候选时间戳列表（秒）。
    - lyric_units: 歌词单元列表。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - rms_times/rms_values: 伴奏侧 RMS 时间轴与能量值。
    - vocal_onset_candidates: 人声音轨起音候选时间戳列表（秒）。
    - vocal_rms_times/vocal_rms_values: 人声音轨 RMS 时间轴与能量值。
    返回值：
    - dict[str, Any]: 包含分段全流程所需索引对象与候选池。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：候选缺失时使用链路内已存在的回退逻辑。
    """
    beat_pool = _normalize_timestamp_list(beat_candidates + [0.0, duration_seconds], duration_seconds)
    onset_pool = _normalize_timestamp_list(onset_candidates + [0.0, duration_seconds], duration_seconds)
    safe_rms_times = rms_times or []
    safe_rms_values = rms_values or []
    rms_index = _build_rms_index(safe_rms_times, safe_rms_values)
    safe_vocal_onset = vocal_onset_candidates or onset_candidates
    safe_vocal_rms_times = vocal_rms_times or safe_rms_times
    safe_vocal_rms_values = vocal_rms_values or safe_rms_values
    vocal_rms_index = _build_rms_index(safe_vocal_rms_times, safe_vocal_rms_values)
    vocal_onset_pool = _normalize_timestamp_list(safe_vocal_onset + [0.0, duration_seconds], duration_seconds)
    lyric_units_sorted = sorted(lyric_units, key=lambda item: float(item.get("start_time", 0.0)))
    lyric_start_times = [float(item.get("start_time", 0.0)) for item in lyric_units_sorted]
    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    instrumental_set.add("inst")
    return {
        "beat_pool": beat_pool,
        "onset_pool": onset_pool,
        "safe_rms_times": safe_rms_times,
        "safe_rms_values": safe_rms_values,
        "rms_index": rms_index,
        "safe_vocal_rms_times": safe_vocal_rms_times,
        "safe_vocal_rms_values": safe_vocal_rms_values,
        "vocal_rms_index": vocal_rms_index,
        "vocal_onset_pool": vocal_onset_pool,
        "lyric_units_sorted": lyric_units_sorted,
        "lyric_start_times": lyric_start_times,
        "instrumental_set": instrumental_set,
    }


def _build_mid_segments_stage(
    big_segments: list[dict[str, Any]],
    indexes: dict[str, Any],
    tuning: SegmentationTuning,
) -> list[dict[str, Any]]:
    """
    功能说明：分段阶段二：基于人声音量从 big_segment 构建 vocal/inst 中间段。
    参数说明：
    - big_segments: 大段落列表。
    - indexes: 预计算索引字典。
    - tuning: 分段调参对象。
    返回值：
    - list[dict[str, Any]]: 中间段列表（mid_segment，仅内部使用）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：中间段不会写入对外契约字段。
    """
    return _build_mid_segments_by_vocal_energy(
        big_segments=big_segments,
        instrumental_set=indexes["instrumental_set"],
        vocal_rms_times=indexes["safe_vocal_rms_times"],
        vocal_rms_values=indexes["safe_vocal_rms_values"],
        vocal_rms_index=indexes["vocal_rms_index"],
        min_mid_duration_seconds=tuning.mid_segment_min_duration_seconds,
        enter_quantile=tuning.vocal_energy_enter_quantile,
        exit_quantile=tuning.vocal_energy_exit_quantile,
    )


def _append_instrumental_range_items(
    range_items: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    big_segment_id: str,
    beat_pool: list[float],
    onset_pool: list[float],
    rms_times: list[float],
    rms_values: list[float],
    long_segment_threshold_seconds: float,
) -> None:
    """
    功能说明：向候选区间追加器乐段分段结果（长段单次能量切分）。
    参数说明：
    - range_items: 待写入的候选区间列表。
    - start_time/end_time: 当前待切分区间起止时间（秒）。
    - big_segment_id: 归属大段落ID。
    - beat_pool/onset_pool: 节拍与起音候选池。
    - rms_times/rms_values: 能量时序，用于单次能量切分。
    - long_segment_threshold_seconds: 触发单次切分的最小时长阈值。
    返回值：无。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：区间过短时不追加，避免产生噪声碎片。
    """
    if end_time - start_time <= 0.12:
        return
    for left_time, right_time in _split_instrumental_range_once_by_energy(
        start_time=start_time,
        end_time=end_time,
        beat_pool=beat_pool,
        onset_pool=onset_pool,
        rms_times=rms_times,
        rms_values=rms_values,
        long_segment_threshold_seconds=long_segment_threshold_seconds,
    ):
        range_items.append(
            {
                "start_time": left_time,
                "end_time": right_time,
                "big_segment_id": big_segment_id,
                "label": "inst",
                "lyric_anchor": False,
                "lyric_text": "",
            }
        )


def _build_big_segment_vocal_label_map(
    mid_segments: list[dict[str, Any]],
    instrumental_set: set[str],
) -> dict[str, str]:
    """
    功能说明：构建 big_segment 到“可复用人声标签”的映射，用于边界短词保护。
    参数说明：
    - mid_segments: 中间段列表。
    - instrumental_set: 器乐标签集合。
    返回值：
    - dict[str, str]: 键为 big_segment_id，值为人声标签。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：同一 big_segment 命中多个人声标签时取最早出现者。
    """
    big_segment_vocal_labels: dict[str, str] = {}
    for item in mid_segments:
        if not bool(item.get("is_vocal", False)):
            continue
        label = str(item.get("label", "unknown")).lower().strip()
        if not label or label in instrumental_set:
            continue
        big_segment_id = str(item.get("big_segment_id", ""))
        if big_segment_id and big_segment_id not in big_segment_vocal_labels:
            big_segment_vocal_labels[big_segment_id] = label
    return big_segment_vocal_labels


def _collect_boundary_lyric_micro_ranges(
    start_time: float,
    end_time: float,
    lyric_units: list[dict[str, Any]],
    boundary_window_seconds: float = BOUNDARY_LYRIC_PROTECTION_WINDOW_SECONDS,
    min_micro_duration_seconds: float = BOUNDARY_LYRIC_MIN_MICRO_DURATION_SECONDS,
) -> list[tuple[float, float]]:
    """
    功能说明：提取 inst 边界窗口内的歌词命中区间，供强制切出 vocal 微段。
    参数说明：
    - start_time: inst 区间起点（秒）。
    - end_time: inst 区间终点（秒）。
    - lyric_units: 与 inst 区间重叠的歌词单元列表。
    - boundary_window_seconds: 边界保护窗口大小（秒）。
    - min_micro_duration_seconds: 最小保护微段时长（秒）。
    返回值：
    - list[tuple[float, float]]: 合并后的边界歌词微段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅保护边界窗口命中的歌词，区间内部歌词不在本步骤处理。
    """
    if end_time - start_time <= 1e-6 or not lyric_units:
        return []
    safe_window = max(0.1, float(boundary_window_seconds))
    safe_min_duration = max(0.03, float(min_micro_duration_seconds))
    left_window_end = min(end_time, start_time + safe_window)
    right_window_start = max(start_time, end_time - safe_window)
    raw_ranges: list[tuple[float, float]] = []

    for lyric_item in lyric_units:
        lyric_start = max(start_time, float(lyric_item.get("start_time", start_time)))
        lyric_end = min(end_time, float(lyric_item.get("end_time", lyric_start)))
        if lyric_end - lyric_start <= 1e-6:
            continue

        left_start = max(start_time, lyric_start)
        left_end = min(left_window_end, lyric_end)
        if left_end - left_start >= safe_min_duration:
            raw_ranges.append((_round_time(left_start), _round_time(left_end)))

        right_start = max(right_window_start, lyric_start)
        right_end = min(end_time, lyric_end)
        if right_end - right_start >= safe_min_duration:
            raw_ranges.append((_round_time(right_start), _round_time(right_end)))

    if not raw_ranges:
        return []

    sorted_raw_ranges = sorted(raw_ranges, key=lambda item: item[0])
    merged_ranges: list[list[float]] = [[float(sorted_raw_ranges[0][0]), float(sorted_raw_ranges[0][1])]]
    for current_start, current_end in sorted_raw_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        if current_start <= prev_end + 1e-6:
            merged_ranges[-1][1] = max(prev_end, float(current_end))
            continue
        merged_ranges.append([float(current_start), float(current_end)])
    return [(item[0], item[1]) for item in merged_ranges if item[1] - item[0] >= safe_min_duration]


def _split_inst_mid_by_boundary_lyric_protection(
    start_time: float,
    end_time: float,
    big_segment_id: str,
    lyric_units: list[dict[str, Any]],
    fallback_vocal_label: str,
    boundary_window_seconds: float = BOUNDARY_LYRIC_PROTECTION_WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    """
    功能说明：对 inst 中间段执行边界歌词保护，必要时切出 vocal 微段。
    参数说明：
    - start_time: inst 区间起点（秒）。
    - end_time: inst 区间终点（秒）。
    - big_segment_id: 归属大段落ID。
    - lyric_units: 与当前 inst 区间重叠的歌词单元。
    - fallback_vocal_label: 切出 vocal 微段时使用的人声标签。
    - boundary_window_seconds: 边界保护窗口大小（秒）。
    返回值：
    - list[dict[str, Any]]: 保护后中间段列表（含 is_vocal 与 label）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当边界窗口无歌词命中时，返回原 inst 区间。
    """
    if end_time - start_time <= 0.12:
        return [
            {
                "start_time": start_time,
                "end_time": end_time,
                "big_segment_id": big_segment_id,
                "label": "inst",
                "is_vocal": False,
            }
        ]
    boundary_ranges = _collect_boundary_lyric_micro_ranges(
        start_time=start_time,
        end_time=end_time,
        lyric_units=lyric_units,
        boundary_window_seconds=boundary_window_seconds,
    )
    if not boundary_ranges:
        return [
            {
                "start_time": start_time,
                "end_time": end_time,
                "big_segment_id": big_segment_id,
                "label": "inst",
                "is_vocal": False,
            }
        ]

    boundaries = [start_time, end_time]
    for left_time, right_time in boundary_ranges:
        boundaries.extend([left_time, right_time])
    sorted_boundaries = sorted(set(_round_time(value) for value in boundaries if start_time <= value <= end_time))
    if sorted_boundaries[0] > start_time:
        sorted_boundaries.insert(0, _round_time(start_time))
    if sorted_boundaries[-1] < end_time:
        sorted_boundaries.append(_round_time(end_time))

    protected_mid_segments: list[dict[str, Any]] = []
    for index in range(len(sorted_boundaries) - 1):
        left_time = float(sorted_boundaries[index])
        right_time = float(sorted_boundaries[index + 1])
        if right_time - left_time < 0.05:
            continue
        center_time = (left_time + right_time) / 2.0
        is_vocal = any(vocal_left <= center_time <= vocal_right for vocal_left, vocal_right in boundary_ranges)
        protected_mid_segments.append(
            {
                "start_time": left_time,
                "end_time": right_time,
                "big_segment_id": big_segment_id,
                "label": fallback_vocal_label if is_vocal else "inst",
                "is_vocal": is_vocal,
            }
        )
    if not protected_mid_segments:
        return [
            {
                "start_time": start_time,
                "end_time": end_time,
                "big_segment_id": big_segment_id,
                "label": "inst",
                "is_vocal": False,
            }
        ]
    return _merge_adjacent_mid_segments_by_role(mid_segments=protected_mid_segments)


def _build_range_items_stage(
    duration_seconds: float,
    mid_segments: list[dict[str, Any]],
    indexes: dict[str, Any],
    tuning: SegmentationTuning,
) -> list[dict[str, Any]]:
    """
    功能说明：分段阶段三：根据 mid_segment 生成候选区间（歌词优先+节奏补全）。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - mid_segments: 中间段列表。
    - indexes: 预计算索引字典。
    - tuning: 分段调参对象。
    返回值：
    - list[dict[str, Any]]: 待归一化区间列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：句内仍遵循“一句歌词独占一个最小视觉单元”约束。
    """
    beat_pool = indexes["beat_pool"]
    onset_pool = indexes["onset_pool"]
    safe_rms_times = indexes["safe_rms_times"]
    safe_rms_values = indexes["safe_rms_values"]
    lyric_units_sorted = indexes["lyric_units_sorted"]
    lyric_start_times = indexes["lyric_start_times"]
    instrumental_set = indexes["instrumental_set"]
    big_segment_vocal_labels = _build_big_segment_vocal_label_map(
        mid_segments=mid_segments,
        instrumental_set=instrumental_set,
    )
    range_items: list[dict[str, Any]] = []
    spill_end = 0.0
    for mid_segment in mid_segments:
        start_time = max(float(mid_segment["start_time"]), spill_end)
        end_time = float(mid_segment["end_time"])
        label = str(mid_segment.get("label", "unknown")).lower().strip()
        big_segment_id = str(mid_segment["big_segment_id"])
        is_vocal_mid = bool(mid_segment.get("is_vocal", False))
        if end_time - start_time <= 1e-6:
            continue

        lyric_in_segment = _slice_lyric_units_by_overlap(
            lyric_units_sorted=lyric_units_sorted,
            lyric_start_times=lyric_start_times,
            start_time=start_time,
            end_time=end_time,
        )

        if not is_vocal_mid:
            mid_big_label = str(mid_segment.get("big_label", label)).lower().strip()
            fallback_vocal_label = big_segment_vocal_labels.get(big_segment_id, "")
            if not fallback_vocal_label:
                fallback_vocal_label = mid_big_label if mid_big_label and mid_big_label not in instrumental_set else "verse"
            protected_mid_segments = _split_inst_mid_by_boundary_lyric_protection(
                start_time=start_time,
                end_time=end_time,
                big_segment_id=big_segment_id,
                lyric_units=lyric_in_segment,
                fallback_vocal_label=fallback_vocal_label,
            )
            for protected_item in protected_mid_segments:
                protected_start = float(protected_item.get("start_time", start_time))
                protected_end = float(protected_item.get("end_time", protected_start))
                if protected_end - protected_start <= 1e-6:
                    continue
                if bool(protected_item.get("is_vocal", False)):
                    range_items.append(
                        {
                            "start_time": protected_start,
                            "end_time": protected_end,
                            "big_segment_id": big_segment_id,
                            "label": str(protected_item.get("label", fallback_vocal_label)),
                            "lyric_anchor": True,
                            "lyric_text": "",
                        }
                    )
                    continue
                _append_instrumental_range_items(
                    range_items=range_items,
                    start_time=protected_start,
                    end_time=protected_end,
                    big_segment_id=big_segment_id,
                    beat_pool=beat_pool,
                    onset_pool=onset_pool,
                    rms_times=safe_rms_times,
                    rms_values=safe_rms_values,
                    long_segment_threshold_seconds=tuning.instrumental_single_split_min_seconds,
                )
            continue

        if not lyric_in_segment:
            # 高召回策略：有人声音量但歌词缺失时，仍保留人声段标签，避免“唱段被误标器乐”。
            range_items.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "big_segment_id": big_segment_id,
                    "label": label,
                    "lyric_anchor": False,
                    "lyric_text": "",
                }
            )
            continue

        cursor = start_time
        appended_vocal_count = 0
        for lyric_item in lyric_in_segment:
            lyric_start = _clamp_time(float(lyric_item.get("start_time", cursor)), duration_seconds)
            lyric_end = _clamp_time(float(lyric_item.get("end_time", lyric_start)), duration_seconds)
            lyric_start = max(start_time, lyric_start)
            lyric_end = max(lyric_start, lyric_end)
            # 设计决策：vocal mid 内的无歌词空档不再转为 inst，直接并入相邻 vocal 片段。
            lyric_start = cursor
            if lyric_end > lyric_start + 1e-6:
                range_items.append(
                    {
                        "start_time": lyric_start,
                        "end_time": lyric_end,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": True,
                        "lyric_text": str(lyric_item.get("text", "")),
                    }
                )
                appended_vocal_count += 1
            cursor = max(cursor, lyric_end)
        if appended_vocal_count > 0:
            last_item = range_items[-1]
            if (
                str(last_item.get("big_segment_id", "")) == big_segment_id
                and str(last_item.get("label", "")).lower().strip() == label
            ):
                last_item["end_time"] = max(float(last_item.get("end_time", cursor)), end_time)
            spill_end = max(spill_end, end_time)
            continue

        # 兜底：若歌词单元全部无效（时长为0等），仍回退为整段 vocal，避免误切 inst。
        range_items.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "big_segment_id": big_segment_id,
                "label": label,
                "lyric_anchor": False,
                "lyric_text": "",
            }
        )
        spill_end = max(spill_end, end_time)
    return range_items


def _normalize_and_merge_ranges_stage(
    range_items: list[dict[str, Any]],
    duration_seconds: float,
    instrumental_set: set[str],
    tuning: SegmentationTuning,
) -> list[dict[str, Any]]:
    """
    功能说明：分段阶段四：区间归一化并合并短段，输出连续可用片段。
    参数说明：
    - range_items: 待归一化区间列表。
    - duration_seconds: 音频总时长（秒）。
    - instrumental_set: 器乐标签集合。
    - tuning: 分段调参对象。
    返回值：
    - list[dict[str, Any]]: 归一化且合并后的区间列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空输入时返回空列表，由上层回退策略兜底。
    """
    normalized_ranges = _normalize_segment_ranges(range_items=range_items, duration_seconds=duration_seconds)
    merged_vocal_ranges = _merge_short_vocal_non_lyric_ranges(
        range_items=normalized_ranges,
        duration_seconds=duration_seconds,
        instrumental_set=instrumental_set,
        min_duration_seconds=tuning.short_vocal_non_lyric_merge_seconds,
    )
    return _merge_short_inst_gaps_between_vocal_ranges(
        range_items=merged_vocal_ranges,
        duration_seconds=duration_seconds,
        instrumental_set=instrumental_set,
        cross_threshold_seconds=INST_GAP_MERGE_CROSS_THRESHOLD_SECONDS,
        same_group_threshold_seconds=INST_GAP_MERGE_SAME_GROUP_THRESHOLD_SECONDS,
    )


def _is_meaningful_lyric_evidence(text: Any) -> bool:
    """
    功能说明：判断歌词文本是否可作为跨 big 边界调整的有效证据。
    参数说明：
    - text: 待判断歌词文本。
    返回值：
    - bool: 若为有效歌词证据返回 True，否则返回 False。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：未识别占位词与吟唱占位词不作为边界调整证据。
    """
    raw_text = str(text).strip()
    if not raw_text:
        return False
    if raw_text in {"[未识别歌词]", "吟唱"}:
        return False
    return True


def _overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    """
    功能说明：计算两个时间区间的重叠时长。
    参数说明：
    - left_start/left_end: 区间1起止时间（秒）。
    - right_start/right_end: 区间2起止时间（秒）。
    返回值：
    - float: 重叠时长（秒）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无交集时返回 0。
    """
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _build_big_segments_v2_by_lyric_overlap(
    big_segments_stage1: list[dict[str, Any]],
    lyric_units: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：基于跨界歌词占比重算 big 边界，生成 big_v2。
    参数说明：
    - big_segments_stage1: 原始 stage1 big 段列表。
    - lyric_units: 歌词单元列表（用于跨界证据计算）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 重算后的 big_v2 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无歌词跨界证据时保持 stage1 边界不变。
    """
    if not big_segments_stage1:
        return []

    sorted_stage1 = sorted(big_segments_stage1, key=lambda item: float(item.get("start_time", 0.0)))
    big_v2 = [
        {
            "segment_id": str(item.get("segment_id", "")),
            "start_time": _round_time(max(0.0, float(item.get("start_time", 0.0)))),
            "end_time": _round_time(max(float(item.get("start_time", 0.0)), float(item.get("end_time", 0.0)))),
            "label": str(item.get("label", "unknown")),
        }
        for item in sorted_stage1
    ]
    if not lyric_units or len(big_v2) <= 1:
        return big_v2

    evidence_units = []
    for item in lyric_units:
        if not isinstance(item, dict):
            continue
        if not _is_meaningful_lyric_evidence(item.get("text", "")):
            continue
        start_time = float(item.get("start_time", 0.0))
        end_time = max(start_time, float(item.get("end_time", start_time)))
        if end_time - start_time <= 1e-6:
            continue
        evidence_units.append({"start_time": start_time, "end_time": end_time, "text": str(item.get("text", "")).strip()})
    if not evidence_units:
        return big_v2

    for boundary_index in range(len(big_v2) - 1):
        left_big = big_v2[boundary_index]
        right_big = big_v2[boundary_index + 1]
        left_start = float(left_big.get("start_time", 0.0))
        right_end = float(right_big.get("end_time", left_start))
        current_boundary = float(left_big.get("end_time", left_start))
        if right_end - left_start <= 1e-6:
            continue

        crossing_units = []
        total_left_overlap = 0.0
        total_right_overlap = 0.0
        for lyric_item in evidence_units:
            lyric_start = float(lyric_item["start_time"])
            lyric_end = float(lyric_item["end_time"])
            if not (lyric_start < current_boundary < lyric_end):
                continue
            left_overlap = _overlap_seconds(lyric_start, lyric_end, left_start, current_boundary)
            right_overlap = _overlap_seconds(lyric_start, lyric_end, current_boundary, right_end)
            if left_overlap <= 1e-6 and right_overlap <= 1e-6:
                continue
            crossing_units.append((lyric_start, lyric_end, left_overlap, right_overlap))
            total_left_overlap += left_overlap
            total_right_overlap += right_overlap
        if not crossing_units:
            continue

        prefer_right = total_right_overlap >= total_left_overlap
        if prefer_right:
            candidate_boundary = min(item[0] for item in crossing_units)
        else:
            candidate_boundary = max(item[1] for item in crossing_units)

        # 仅保证边界严格落在前后 big 之间，不叠加额外吸附/碎片约束。
        candidate_boundary = max(left_start + 1e-6, min(right_end - 1e-6, candidate_boundary))
        candidate_boundary = _round_time(candidate_boundary)
        if abs(candidate_boundary - current_boundary) <= 1e-6:
            continue
        left_big["end_time"] = candidate_boundary
        right_big["start_time"] = candidate_boundary

    # 归一化时间轴，保证 big_v2 连续覆盖。
    big_v2[0]["start_time"] = 0.0
    big_v2[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(big_v2)):
        big_v2[index]["start_time"] = _round_time(float(big_v2[index - 1]["end_time"]))
    return big_v2


def _build_segments_from_ranges(normalized_ranges: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：将归一化区间转换为标准 segments 输出结构。
    参数说明：
    - normalized_ranges: 归一化区间列表。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 标准化 segments 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：调用前需保证 normalized_ranges 非空。
    """
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


def _build_segments_with_lyric_priority(
    duration_seconds: float,
    big_segments: list[dict[str, Any]],
    beat_candidates: list[float],
    onset_candidates: list[float],
    lyric_units: list[dict[str, Any]],
    instrumental_labels: list[str],
    tuning: SegmentationTuning | None = None,
    rms_times: list[float] | None = None,
    rms_values: list[float] | None = None,
    vocal_onset_candidates: list[float] | None = None,
    vocal_rms_times: list[float] | None = None,
    vocal_rms_values: list[float] | None = None,
    vocal_energy_enter_quantile: float = 0.70,
    vocal_energy_exit_quantile: float = 0.45,
    mid_segment_min_duration_seconds: float = 0.8,
    short_vocal_non_lyric_merge_seconds: float = 1.2,
    instrumental_single_split_min_seconds: float = 4.0,
    accent_delta_trigger_ratio: float = 0.35,
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
    - tuning: 分段调参对象（优先使用）。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    - vocal_onset_candidates: 人声音轨起音候选时间戳列表（秒）。
    - vocal_rms_times: 人声音轨 RMS 时间轴（秒）。
    - vocal_rms_values: 人声音轨 RMS 能量值序列。
    - vocal_energy_enter_quantile: 人声音量进入阈值分位点。
    - vocal_energy_exit_quantile: 人声音量退出阈值分位点。
    - mid_segment_min_duration_seconds: 人声中间段最小时长阈值（秒）。
    - short_vocal_non_lyric_merge_seconds: 人声“无歌词/短歌词”短段合并阈值（秒）。
    - instrumental_single_split_min_seconds: 器乐单次切分触发最小时长（秒）。
    - accent_delta_trigger_ratio: 首重音检测的能量突变触发比例（0~1）。
    返回值：
    - list[dict[str, Any]]: 构建后的小分段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not big_segments:
        return _build_small_segments([0.0, duration_seconds], big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": duration_seconds, "label": "unknown"}], duration_seconds=duration_seconds)  # noqa: E501

    safe_tuning = tuning or _build_segmentation_tuning(
        vocal_energy_enter_quantile=vocal_energy_enter_quantile,
        vocal_energy_exit_quantile=vocal_energy_exit_quantile,
        mid_segment_min_duration_seconds=mid_segment_min_duration_seconds,
        short_vocal_non_lyric_merge_seconds=short_vocal_non_lyric_merge_seconds,
        instrumental_single_split_min_seconds=instrumental_single_split_min_seconds,
        accent_delta_trigger_ratio=accent_delta_trigger_ratio,
    )
    indexes = _prepare_segmentation_indexes(
        duration_seconds=duration_seconds,
        beat_candidates=beat_candidates,
        onset_candidates=onset_candidates,
        lyric_units=lyric_units,
        instrumental_labels=instrumental_labels,
        rms_times=rms_times,
        rms_values=rms_values,
        vocal_onset_candidates=vocal_onset_candidates,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
    )
    mid_segments = _build_mid_segments_stage(
        big_segments=big_segments,
        indexes=indexes,
        tuning=safe_tuning,
    )
    range_items = _build_range_items_stage(
        duration_seconds=duration_seconds,
        mid_segments=mid_segments,
        indexes=indexes,
        tuning=safe_tuning,
    )
    normalized_ranges = _normalize_and_merge_ranges_stage(
        range_items=range_items,
        duration_seconds=duration_seconds,
        instrumental_set=indexes["instrumental_set"],
        tuning=safe_tuning,
    )
    if not normalized_ranges:
        return _build_small_segments(
            timestamps=[0.0, duration_seconds],
            big_segments=big_segments,
            duration_seconds=duration_seconds,
        )
    return _build_segments_from_ranges(normalized_ranges=normalized_ranges, duration_seconds=duration_seconds)


def _build_mid_segments_by_vocal_energy(
    big_segments: list[dict[str, Any]],
    instrumental_set: set[str],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    vocal_rms_index: dict[str, list[float]] | None = None,
    min_mid_duration_seconds: float = 0.8,
    enter_quantile: float = 0.70,
    exit_quantile: float = 0.45,
) -> list[dict[str, Any]]:
    """
    功能说明：在 big_segment 内按“全局静音地板 + 固定阈值”切出 vocal/inst 中间段。
    参数说明：
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    - instrumental_set: 器乐标签集合。
    - vocal_rms_times: 人声音轨 RMS 时间轴（秒）。
    - vocal_rms_values: 人声音轨 RMS 能量值序列。
    - vocal_rms_index: 人声音轨 RMS 查询索引（可选，传入可避免重复构建）。
    - min_mid_duration_seconds: 中间段最小时长阈值（秒）。
    - enter_quantile: 兼容参数（当前仅用于静音地板估计的低分位参考）。
    - exit_quantile: 兼容参数（当前仅用于静音地板估计的低分位参考）。
    返回值：
    - list[dict[str, Any]]: 中间段列表（含 is_vocal 与 label）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：判定目标偏高召回，仅“绝对静音”优先判为器乐段。
    """
    safe_rms_index = vocal_rms_index or _build_rms_index(vocal_rms_times, vocal_rms_values)
    silence_floor_rms = _estimate_vocal_silence_floor_rms(
        vocal_rms_values=vocal_rms_values,
        enter_quantile=enter_quantile,
        exit_quantile=exit_quantile,
    )
    epsilon = max(0.0008, silence_floor_rms * 0.25)
    vocal_threshold_rms = min(0.08, silence_floor_rms + epsilon)
    min_vocal_duration_seconds = max(0.06, float(min_mid_duration_seconds) * 0.15)
    min_silence_duration_seconds = max(0.08, float(min_mid_duration_seconds) * 0.35)
    mid_segments: list[dict[str, Any]] = []
    for item in big_segments:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        if end_time - start_time <= 1e-6:
            continue
        big_segment_id = str(item.get("segment_id", ""))
        big_label = str(item.get("label", "unknown")).lower().strip()

        if big_label in instrumental_set:
            mid_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "big_segment_id": big_segment_id,
                    "big_label": big_label,
                    "label": "inst",
                    "is_vocal": False,
                }
            )
            continue

        segment_points = _slice_rms_points_by_time(
            rms_index=safe_rms_index,
            start_time=start_time,
            end_time=end_time,
            include_left=True,
            include_right=True,
        )
        if not segment_points:
            is_vocal_fallback = big_label not in instrumental_set
            mid_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "big_segment_id": big_segment_id,
                    "big_label": big_label,
                    "label": big_label if is_vocal_fallback else "inst",
                    "is_vocal": is_vocal_fallback,
                }
            )
            continue

        segment_values = [float(value) for _, value in segment_points]
        segment_peak_linear = max(segment_values) if segment_values else 0.0
        if segment_peak_linear <= vocal_threshold_rms:
            mid_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "big_segment_id": big_segment_id,
                    "big_label": big_label,
                    "label": "inst",
                    "is_vocal": False,
                }
            )
            continue

        vocal_ranges = _build_vocal_ranges_by_single_threshold(
            segment_points=segment_points,
            start_time=start_time,
            end_time=end_time,
            vocal_threshold_rms=vocal_threshold_rms,
        )
        vocal_ranges = _merge_vocal_ranges_with_short_silence(
            vocal_ranges=vocal_ranges,
            max_silence_duration_seconds=min_silence_duration_seconds,
        )

        vocal_ranges = [
            (_round_time(max(start_time, left_time)), _round_time(min(end_time, right_time)))
            for left_time, right_time in vocal_ranges
            if right_time - left_time >= min_vocal_duration_seconds
        ]

        if not vocal_ranges:
            mid_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "big_segment_id": big_segment_id,
                    "big_label": big_label,
                    "label": "inst",
                    "is_vocal": False,
                }
            )
            continue

        boundaries = [start_time, end_time]
        for left_time, right_time in vocal_ranges:
            boundaries.extend([left_time, right_time])
        boundaries = sorted(set(_round_time(value) for value in boundaries if start_time <= value <= end_time))
        if boundaries[0] > start_time:
            boundaries.insert(0, _round_time(start_time))
        if boundaries[-1] < end_time:
            boundaries.append(_round_time(end_time))

        local_mid_segments: list[dict[str, Any]] = []
        for index in range(len(boundaries) - 1):
            left_time = float(boundaries[index])
            right_time = float(boundaries[index + 1])
            if right_time - left_time < 0.05:
                continue
            mid_time = (left_time + right_time) / 2.0
            is_vocal_sub = any(vocal_start <= mid_time <= vocal_end for vocal_start, vocal_end in vocal_ranges)
            local_mid_segments.append(
                {
                    "start_time": left_time,
                    "end_time": right_time,
                    "big_segment_id": big_segment_id,
                    "big_label": big_label,
                    "label": big_label if is_vocal_sub else "inst",
                    "is_vocal": is_vocal_sub,
                }
            )
        local_mid_segments = _smooth_mid_segments_by_duration(
            mid_segments=local_mid_segments,
            min_vocal_duration_seconds=min_vocal_duration_seconds,
            min_silence_duration_seconds=min_silence_duration_seconds,
        )
        mid_segments.extend(local_mid_segments)

    return sorted(mid_segments, key=lambda item: float(item.get("start_time", 0.0)))


def _estimate_vocal_silence_floor_rms(
    vocal_rms_values: list[float],
    enter_quantile: float = 0.70,
    exit_quantile: float = 0.45,
) -> float:
    """
    功能说明：估计人声音轨的全局静音地板（RMS 线性能量）。
    参数说明：
    - vocal_rms_values: 人声音轨 RMS 序列。
    - enter_quantile: 兼容参数，用于保护低分位边界。
    - exit_quantile: 兼容参数，用于保护低分位边界。
    返回值：
    - float: 静音地板 RMS 值。
    异常说明：无。
    边界条件：返回值始终在安全范围 `[0.0015, 0.02]`。
    """
    if not vocal_rms_values:
        return 0.0030
    safe_values = [max(0.0, float(value)) for value in vocal_rms_values]
    low_q = max(0.02, min(0.30, min(float(enter_quantile), float(exit_quantile), 0.10)))
    noise_floor = _quantile(safe_values, low_q)
    floor_rms = max(0.0015, min(0.02, noise_floor))
    return floor_rms


def _build_vocal_ranges_by_single_threshold(
    segment_points: list[tuple[float, float]],
    start_time: float,
    end_time: float,
    vocal_threshold_rms: float,
) -> list[tuple[float, float]]:
    """
    功能说明：按固定 RMS 阈值提取人声区间（高召回，阈值主导）。
    参数说明：
    - segment_points: 当前区间采样点 `(time, rms)`。
    - start_time: 区间起点。
    - end_time: 区间终点。
    - vocal_threshold_rms: 人声判定阈值（RMS）。
    返回值：
    - list[tuple[float, float]]: 粗粒度人声区间列表。
    异常说明：无。
    边界条件：区间开头快速进入时会吸附到起点，尽量保留字头。
    """
    if not segment_points or end_time - start_time <= 1e-6:
        return []
    vocal_ranges: list[tuple[float, float]] = []
    is_vocal = False
    vocal_start = start_time
    for point_time, point_value in segment_points:
        point_rms = max(0.0, float(point_value))
        if not is_vocal and point_rms > vocal_threshold_rms:
            is_vocal = True
            vocal_start = start_time if point_time - start_time <= 0.12 else point_time
            continue
        if is_vocal and point_rms <= vocal_threshold_rms:
            vocal_end = max(vocal_start, point_time)
            if vocal_end - vocal_start > 1e-6:
                vocal_ranges.append((vocal_start, vocal_end))
            is_vocal = False
    if is_vocal:
        vocal_end = max(vocal_start, end_time)
        if vocal_end - vocal_start > 1e-6:
            vocal_ranges.append((vocal_start, vocal_end))
    return vocal_ranges


def _smooth_mid_segments_by_duration(
    mid_segments: list[dict[str, Any]],
    min_vocal_duration_seconds: float,
    min_silence_duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：按时长平滑 mid_segment，优先消除短静音空洞和极短毛刺段。
    参数说明：
    - mid_segments: 原始中间段列表。
    - min_vocal_duration_seconds: 人声最小时长阈值。
    - min_silence_duration_seconds: 器乐最小时长阈值。
    返回值：
    - list[dict[str, Any]]: 平滑后的中间段列表。
    异常说明：无。
    边界条件：当邻段角色一致时优先执行跨段并合。
    """
    if not mid_segments:
        return []
    merged = _merge_adjacent_mid_segments_by_role(mid_segments=mid_segments)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        index = 0
        while index < len(merged):
            item = merged[index]
            start_time = float(item.get("start_time", 0.0))
            end_time = float(item.get("end_time", start_time))
            duration = max(0.0, end_time - start_time)
            is_vocal = bool(item.get("is_vocal", False))
            min_duration = min_vocal_duration_seconds if is_vocal else min_silence_duration_seconds
            if duration >= min_duration:
                index += 1
                continue

            if 0 < index < len(merged) - 1:
                prev_item = merged[index - 1]
                next_item = merged[index + 1]
                prev_role = bool(prev_item.get("is_vocal", False))
                next_role = bool(next_item.get("is_vocal", False))
                if prev_role == next_role:
                    prev_item["end_time"] = next_item.get("end_time", prev_item.get("end_time", 0.0))
                    prev_item["label"] = prev_item.get("label", "unknown") if prev_role else "inst"
                    del merged[index : index + 2]
                    changed = True
                    break

            if index == 0:
                merged[1]["start_time"] = start_time
                del merged[index]
                changed = True
                break
            if index == len(merged) - 1:
                merged[index - 1]["end_time"] = end_time
                del merged[index]
                changed = True
                break

            prev_item = merged[index - 1]
            next_item = merged[index + 1]
            prev_duration = float(prev_item.get("end_time", 0.0)) - float(prev_item.get("start_time", 0.0))
            next_duration = float(next_item.get("end_time", 0.0)) - float(next_item.get("start_time", 0.0))
            merge_to_prev = prev_duration >= next_duration
            if merge_to_prev:
                prev_item["end_time"] = end_time
                del merged[index]
            else:
                next_item["start_time"] = start_time
                del merged[index]
            changed = True
            break

    return _merge_adjacent_mid_segments_by_role(mid_segments=merged)


def _merge_adjacent_mid_segments_by_role(mid_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：将相邻同角色的中间段合并，减少重复边界。
    参数说明：
    - mid_segments: 中间段列表。
    返回值：
    - list[dict[str, Any]]: 合并后的中间段列表。
    异常说明：无。
    边界条件：按 start_time 升序处理，保持时间连续。
    """
    if not mid_segments:
        return []
    normalized = [
        dict(item)
        for item in sorted(mid_segments, key=lambda value: float(value.get("start_time", 0.0)))
        if float(item.get("end_time", item.get("start_time", 0.0))) - float(item.get("start_time", 0.0)) > 1e-6
    ]
    if not normalized:
        return []
    merged: list[dict[str, Any]] = [normalized[0]]
    for item in normalized[1:]:
        prev_item = merged[-1]
        prev_role = bool(prev_item.get("is_vocal", False))
        curr_role = bool(item.get("is_vocal", False))
        if prev_role == curr_role:
            prev_item["end_time"] = item.get("end_time", prev_item.get("end_time", 0.0))
            prev_item["label"] = prev_item.get("label", "unknown") if prev_role else "inst"
            continue
        merged.append(item)
    return merged


def _merge_vocal_ranges_with_short_silence(
    vocal_ranges: list[tuple[float, float]],
    max_silence_duration_seconds: float,
) -> list[tuple[float, float]]:
    """
    功能说明：合并人声区间之间的短静音空洞，减少“人声-器乐-人声”抖动误切。
    参数说明：
    - vocal_ranges: 人声区间列表（按时间升序）。
    - max_silence_duration_seconds: 允许回填的最大静音时长（秒）。
    返回值：
    - list[tuple[float, float]]: 回填后的稳定人声区间列表。
    异常说明：无。
    边界条件：当输入为空时返回空列表。
    """
    if not vocal_ranges:
        return []
    max_gap = max(0.0, float(max_silence_duration_seconds))
    merged_ranges: list[list[float]] = [[float(vocal_ranges[0][0]), float(vocal_ranges[0][1])]]
    for left_time, right_time in vocal_ranges[1:]:
        prev_left, prev_right = merged_ranges[-1]
        gap = float(left_time) - prev_right
        if gap <= max_gap:
            merged_ranges[-1][1] = max(prev_right, float(right_time))
            continue
        merged_ranges.append([float(left_time), float(right_time)])
    return [(item[0], item[1]) for item in merged_ranges if item[1] - item[0] > 1e-6]


def _merge_short_mid_segments_by_neighbor_energy(
    mid_segments: list[dict[str, Any]],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    min_duration_seconds: float,
    vocal_rms_index: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：将过短 mid_segment 按邻段平均能量相近原则合并，减少抖动碎片。
    参数说明：
    - mid_segments: 中间段列表。
    - vocal_rms_times: 人声音轨 RMS 时间轴（秒）。
    - vocal_rms_values: 人声音轨 RMS 能量值序列。
    - vocal_rms_index: 人声音轨 RMS 查询索引（可选，传入可避免重复构建）。
    - min_duration_seconds: 触发合并的最小时长阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 合并后的中间段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：首尾短段分别并入唯一邻段。
    """
    if not mid_segments:
        return []
    safe_rms_index = vocal_rms_index or _build_rms_index(vocal_rms_times, vocal_rms_values)

    merged = [dict(item) for item in sorted(mid_segments, key=lambda item: float(item.get("start_time", 0.0)))]
    index = 0
    while index < len(merged):
        item = merged[index]
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        duration = end_time - start_time
        if duration >= min_duration_seconds or len(merged) <= 1:
            index += 1
            continue

        if index == 0:
            merged[1]["start_time"] = start_time
            del merged[index]
            continue
        if index == len(merged) - 1:
            merged[index - 1]["end_time"] = end_time
            del merged[index]
            index = max(0, index - 1)
            continue

        prev_item = merged[index - 1]
        next_item = merged[index + 1]
        current_energy = _mean_rms_in_range(
            start_time=start_time,
            end_time=end_time,
            rms_times=vocal_rms_times,
            rms_values=vocal_rms_values,
            rms_index=safe_rms_index,
        )
        prev_energy = _mean_rms_in_range(
            float(prev_item.get("start_time", 0.0)),
            float(prev_item.get("end_time", 0.0)),
            vocal_rms_times,
            vocal_rms_values,
            rms_index=safe_rms_index,
        )
        next_energy = _mean_rms_in_range(
            float(next_item.get("start_time", 0.0)),
            float(next_item.get("end_time", 0.0)),
            vocal_rms_times,
            vocal_rms_values,
            rms_index=safe_rms_index,
        )
        prev_gap = abs(prev_energy - current_energy)
        next_gap = abs(next_energy - current_energy)
        merge_to_prev = prev_gap <= next_gap
        if merge_to_prev:
            prev_item["end_time"] = end_time
            del merged[index]
            index = max(0, index - 1)
            continue
        next_item["start_time"] = start_time
        del merged[index]

    normalized: list[dict[str, Any]] = []
    for item in merged:
        if not normalized:
            normalized.append(item)
            continue
        prev_item = normalized[-1]
        prev_role = bool(prev_item.get("is_vocal", False))
        curr_role = bool(item.get("is_vocal", False))
        if prev_role == curr_role:
            prev_item["end_time"] = item.get("end_time", prev_item.get("end_time", 0.0))
            prev_item["label"] = prev_item.get("label", "inst") if prev_role else "inst"
            continue
        normalized.append(item)
    return normalized


def _detect_first_accent_in_vocal_segment(
    start_time: float,
    end_time: float,
    vocal_onset_pool: list[float],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accent_delta_trigger_ratio: float = 0.35,
) -> float:
    """
    功能说明：在 vocal mid_segment 内检测首个重音时间点（纯能量算法）。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - vocal_onset_pool: 人声起音候选池（秒）。
    - vocal_rms_times: 人声音轨 RMS 时间轴（秒）。
    - vocal_rms_values: 人声音轨 RMS 能量值序列。
    - accent_delta_trigger_ratio: 能量突变触发比例（0~1）。
    返回值：
    - float: 首重音时间（秒），失败时回退区间起点。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当候选不存在或区间极短时直接回退起点。
    """
    if end_time - start_time <= 0.12:
        return start_time

    onset_candidates = _slice_sorted_window(
        sorted_values=vocal_onset_pool,
        start_time=start_time,
        end_time=end_time,
        include_left=True,
        include_right=True,
    )
    if not onset_candidates:
        return start_time

    delta_values = [_rms_delta_at(value, vocal_rms_times, vocal_rms_values) for value in onset_candidates]
    peak_delta = max(delta_values) if delta_values else 0.0
    if peak_delta > 1e-6:
        trigger_threshold = peak_delta * accent_delta_trigger_ratio
        for index, value in enumerate(onset_candidates):
            if delta_values[index] >= trigger_threshold:
                return _round_time(value)

    fallback_peak = max(onset_candidates, key=lambda value: _rms_value_at(value, vocal_rms_times, vocal_rms_values))
    if fallback_peak >= start_time + 1e-3:
        return _round_time(fallback_peak)
    return _round_time(onset_candidates[0])


def _mean_rms_in_range(
    start_time: float,
    end_time: float,
    rms_times: list[float],
    rms_values: list[float],
    rms_index: dict[str, list[float]] | None = None,
) -> float:
    """
    功能说明：计算时间区间内 RMS 平均值。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - rms_times: RMS 时间轴（秒）。
    - rms_values: RMS 能量值序列。
    - rms_index: RMS 查询索引（可选，传入可使用前缀和快速求均值）。
    返回值：
    - float: 区间平均 RMS。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当区间采样为空时回退到区间起点 RMS。
    """
    safe_index = rms_index or _build_rms_index(rms_times, rms_values)
    times = safe_index.get("times", [])
    values = safe_index.get("values", [])
    prefix = safe_index.get("prefix", [])
    if times and prefix and len(prefix) == len(times) + 1:
        left_index = bisect_left(times, start_time)
        right_index = bisect_right(times, end_time)
        if right_index > left_index:
            segment_sum = prefix[right_index] - prefix[left_index]
            return segment_sum / float(right_index - left_index)
    if values:
        return _rms_value_at(start_time, times, values)
    return _rms_value_at(start_time, rms_times, rms_values)


def _quantile(values: list[float], quantile: float) -> float:
    """
    功能说明：计算分位数（线性插值）。
    参数说明：
    - values: 数值序列。
    - quantile: 分位点（0~1）。
    返回值：
    - float: 分位数结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空序列返回 0.0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    target = max(0.0, min(1.0, quantile)) * (len(sorted_values) - 1)
    left_index = int(target)
    right_index = min(len(sorted_values) - 1, left_index + 1)
    if left_index == right_index:
        return sorted_values[left_index]
    weight = target - left_index
    return sorted_values[left_index] * (1.0 - weight) + sorted_values[right_index] * weight


def _split_instrumental_range_once_by_energy(
    start_time: float,
    end_time: float,
    beat_pool: list[float],
    onset_pool: list[float],
    rms_times: list[float],
    rms_values: list[float],
    long_segment_threshold_seconds: float = 4.0,
    min_side_duration_seconds: float = 1.2,
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
        onset_candidates = _slice_sorted_window(
            sorted_values=onset_pool,
            start_time=left_bound,
            end_time=right_bound,
            include_left=True,
            include_right=True,
        )
        beat_candidates = _slice_sorted_window(
            sorted_values=beat_pool,
            start_time=left_bound,
            end_time=right_bound,
            include_left=True,
            include_right=True,
        )
        candidate_pool = onset_candidates if onset_candidates else beat_candidates
        if candidate_pool:
            split_time = max(candidate_pool, key=lambda value: _rms_delta_at(value, rms_times, rms_values))
            peak_delta = _rms_delta_at(split_time, rms_times, rms_values)
            if peak_delta <= 1e-6:
                split_time = max(candidate_pool, key=lambda value: _rms_value_at(value, rms_times, rms_values))
        else:
            split_time = (start_time + end_time) / 2.0

    split_time = _round_time(split_time)
    safe_min_side_duration = max(0.12, float(min_side_duration_seconds))
    if split_time - start_time < safe_min_side_duration or end_time - split_time < safe_min_side_duration:
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

    beat_in_range = _slice_sorted_window(
        sorted_values=beat_pool,
        start_time=start_time,
        end_time=end_time,
        include_left=False,
        include_right=False,
    )
    onset_in_range = _slice_sorted_window(
        sorted_values=onset_pool,
        start_time=start_time,
        end_time=end_time,
        include_left=False,
        include_right=False,
    )

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
                "lyric_text": str(item.get("lyric_text", "")),
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


def _normalized_lyric_text_length(text: Any) -> int:
    """
    功能说明：计算歌词文本去标点后的有效字符长度（中文/字母/数字）。
    参数说明：
    - text: 歌词文本。
    返回值：
    - int: 去标点后的有效字符数。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空文本返回 0。
    """
    raw_text = str(text or "").strip()
    if not raw_text:
        return 0
    normalized = "".join(
        char
        for char in raw_text
        if char.isalnum() or ("\u4e00" <= char <= "\u9fff")
    )
    return len(normalized)


def _is_short_vocal_target(
    item: dict[str, Any],
    instrumental_set: set[str],
    min_duration_seconds: float,
) -> bool:
    """
    功能说明：判断分段项是否属于“短 vocal 收敛目标”（无歌词或短歌词）。
    参数说明：
    - item: 待判断分段项。
    - instrumental_set: 器乐标签集合。
    - min_duration_seconds: 短段阈值（秒）。
    返回值：
    - bool: 若为短 vocal 收敛目标则返回 True。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅在非 inst 且时长小于阈值时才可能命中。
    """
    label = str(item.get("label", "unknown")).lower().strip()
    if label in instrumental_set:
        return False
    start_time = float(item.get("start_time", 0.0))
    end_time = float(item.get("end_time", start_time))
    duration = max(0.0, end_time - start_time)
    if duration >= min_duration_seconds:
        return False
    if not bool(item.get("lyric_anchor", False)):
        return True
    lyric_length = _normalized_lyric_text_length(item.get("lyric_text", ""))
    return lyric_length <= 2


def _is_vocal_merge_target(
    item: dict[str, Any] | None,
    instrumental_set: set[str],
    min_duration_seconds: float,
) -> bool:
    """
    功能说明：判断分段项是否可作为短 vocal 并合目标段。
    参数说明：
    - item: 候选目标分段项。
    - instrumental_set: 器乐标签集合。
    - min_duration_seconds: 短段阈值（秒）。
    返回值：
    - bool: 若可并合则返回 True。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：目标段必须为非 inst 且不属于“短 vocal 收敛目标”。
    """
    if not item:
        return False
    label = str(item.get("label", "unknown")).lower().strip()
    if label in instrumental_set:
        return False
    return not _is_short_vocal_target(item, instrumental_set=instrumental_set, min_duration_seconds=min_duration_seconds)


def _merge_short_vocal_non_lyric_ranges(
    range_items: list[dict[str, Any]],
    duration_seconds: float,
    instrumental_set: set[str],
    min_duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：合并人声区过短的“无歌词/短歌词”段，降低镜头闪切。
    参数说明：
    - range_items: 待归一化或合并的分段区间列表。
    - duration_seconds: 音频总时长（秒）。
    - instrumental_set: 业务处理所需输入参数。
    - min_duration_seconds: 最小时长阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 合并后的单元或区间结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：连续短段先合并成簇；簇时长达到阈值时保持独立，否则并入相邻合格人声段。
    """
    if not range_items:
        return []

    merged_ranges = [dict(item) for item in range_items]
    index = 0
    while index < len(merged_ranges):
        if not _is_short_vocal_target(merged_ranges[index], instrumental_set=instrumental_set, min_duration_seconds=min_duration_seconds):
            index += 1
            continue

        cluster_start_index = index
        cluster_end_index = index
        while (
            cluster_end_index + 1 < len(merged_ranges)
            and _is_short_vocal_target(
                merged_ranges[cluster_end_index + 1],
                instrumental_set=instrumental_set,
                min_duration_seconds=min_duration_seconds,
            )
        ):
            cluster_end_index += 1

        has_continuous_cluster = cluster_end_index > cluster_start_index
        if has_continuous_cluster:
            merged_ranges[cluster_start_index]["end_time"] = merged_ranges[cluster_end_index]["end_time"]
            merged_ranges[cluster_start_index]["lyric_anchor"] = any(
                bool(merged_ranges[item_index].get("lyric_anchor", False))
                for item_index in range(cluster_start_index, cluster_end_index + 1)
            )
            merged_ranges[cluster_start_index]["lyric_text"] = "".join(
                str(merged_ranges[item_index].get("lyric_text", ""))
                for item_index in range(cluster_start_index, cluster_end_index + 1)
            )
            del merged_ranges[cluster_start_index + 1 : cluster_end_index + 1]

        cluster_item = merged_ranges[cluster_start_index]
        cluster_duration = max(
            0.0,
            float(cluster_item.get("end_time", cluster_item.get("start_time", 0.0)))
            - float(cluster_item.get("start_time", 0.0)),
        )
        if has_continuous_cluster and cluster_duration >= min_duration_seconds:
            index = cluster_start_index + 1
            continue

        prev_item = merged_ranges[cluster_start_index - 1] if cluster_start_index > 0 else None
        next_item = merged_ranges[cluster_start_index + 1] if cluster_start_index + 1 < len(merged_ranges) else None
        prev_is_target = _is_vocal_merge_target(prev_item, instrumental_set=instrumental_set, min_duration_seconds=min_duration_seconds)
        next_is_target = _is_vocal_merge_target(next_item, instrumental_set=instrumental_set, min_duration_seconds=min_duration_seconds)

        if prev_is_target:
            merged_ranges[cluster_start_index - 1]["end_time"] = cluster_item.get("end_time", cluster_item.get("start_time", 0.0))
            del merged_ranges[cluster_start_index]
            index = max(0, cluster_start_index - 1)
            continue

        if next_is_target:
            merged_ranges[cluster_start_index + 1]["start_time"] = cluster_item.get("start_time", 0.0)
            del merged_ranges[cluster_start_index]
            index = max(0, cluster_start_index - 1)
            continue

        index = cluster_start_index + 1

    return _normalize_segment_ranges(range_items=merged_ranges, duration_seconds=duration_seconds)


def _merge_short_inst_gaps_between_vocal_ranges(
    range_items: list[dict[str, Any]],
    duration_seconds: float,
    instrumental_set: set[str],
    cross_threshold_seconds: float = INST_GAP_MERGE_CROSS_THRESHOLD_SECONDS,
    same_group_threshold_seconds: float = INST_GAP_MERGE_SAME_GROUP_THRESHOLD_SECONDS,
) -> list[dict[str, Any]]:
    """
    功能说明：合并人声段之间过短的 inst 空挡，减少“人声-器乐-人声”细碎抖动。
    参数说明：
    - range_items: 待归一化或合并的分段区间列表。
    - duration_seconds: 音频总时长（秒）。
    - instrumental_set: 器乐标签集合。
    - cross_threshold_seconds: 跨标签/跨大段并合阈值（秒）。
    - same_group_threshold_seconds: 同标签同大段并合阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 合并后的分段区间列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：默认并入左侧人声；若位于开头且无左侧可并时并入右侧。
    """
    if not range_items:
        return []
    safe_cross_threshold = max(0.0, float(cross_threshold_seconds))
    safe_same_group_threshold = max(safe_cross_threshold, float(same_group_threshold_seconds))
    merged_ranges = [dict(item) for item in range_items]
    index = 0
    while index < len(merged_ranges):
        item = merged_ranges[index]
        label = str(item.get("label", "unknown")).lower().strip()
        if label not in instrumental_set:
            index += 1
            continue
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        duration = max(0.0, end_time - start_time)

        prev_item = merged_ranges[index - 1] if index > 0 else None
        next_item = merged_ranges[index + 1] if index + 1 < len(merged_ranges) else None
        prev_is_vocal = bool(prev_item) and str(prev_item.get("label", "unknown")).lower().strip() not in instrumental_set
        next_is_vocal = bool(next_item) and str(next_item.get("label", "unknown")).lower().strip() not in instrumental_set
        if not (prev_is_vocal or next_is_vocal):
            index += 1
            continue

        if prev_is_vocal and next_is_vocal:
            same_big_segment = (
                str(prev_item.get("big_segment_id", "")).strip()
                == str(item.get("big_segment_id", "")).strip()
                == str(next_item.get("big_segment_id", "")).strip()
            )
            same_label = (
                str(prev_item.get("label", "")).lower().strip()
                == str(next_item.get("label", "")).lower().strip()
            )
            threshold = safe_same_group_threshold if (same_big_segment and same_label) else safe_cross_threshold
        else:
            threshold = safe_cross_threshold

        if duration > threshold:
            index += 1
            continue

        if prev_is_vocal:
            prev_item["end_time"] = end_time
            del merged_ranges[index]
            index = max(0, index - 1)
            continue
        if next_is_vocal:
            next_item["start_time"] = start_time
            del merged_ranges[index]
            continue
        index += 1

    return _normalize_segment_ranges(range_items=merged_ranges, duration_seconds=duration_seconds)
