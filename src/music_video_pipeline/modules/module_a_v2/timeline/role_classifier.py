"""
文件用途：对窗口执行四分类（lyric/chant/inst/silence）。
核心流程：先从双路 RMS 提取活动时间窗，再按“歌词 > 人声活动 > 伴奏活动”判定角色。
输入输出：输入窗口与双路活动窗，输出带 role 的窗口列表。
依赖说明：仅依赖标准库。
维护说明：本文件只负责活动窗提取与分类，不负责并段与边界矫正。
"""

# 标准库：用于类型提示
from typing import Any


# 常量：浮点比较容差
EPSILON_SECONDS = 1e-6


# 常量：RMS静音下界
RMS_SILENCE_FLOOR = 0.003


# 常量：伴奏活动进入阈值分位点
ACCOMPANIMENT_ENTER_QUANTILE = 0.20
# 常量：伴奏活动退出阈值分位点
ACCOMPANIMENT_EXIT_QUANTILE = 0.10
# 常量：伴奏活动最小保留时长（秒）
ACCOMPANIMENT_MIN_INTERVAL_SECONDS = 0.18
# 常量：伴奏活动短间隙闭合阈值（秒）
ACCOMPANIMENT_MERGE_GAP_SECONDS = 0.12


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：安全转换为浮点数。
    参数说明：
    - value: 待转换对象。
    - default: 失败回退值。
    返回值：
    - float: 有效浮点数。
    异常说明：异常内部吞并。
    边界条件：NaN/inf 回退默认值。
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


def _estimate_active_threshold(values: list[float], quantile: float) -> float:
    """
    功能说明：估计“有能量”阈值。
    参数说明：
    - values: RMS序列。
    - quantile: 分位点。
    返回值：
    - float: 能量阈值。
    异常说明：无。
    边界条件：空序列回退静音下界。
    """
    safe_values = [max(0.0, _safe_float(item, 0.0)) for item in values]
    if not safe_values:
        return RMS_SILENCE_FLOOR
    return max(RMS_SILENCE_FLOOR, _quantile(safe_values, quantile))


def _normalize_rms_pairs(
    rms_times: list[float],
    rms_values: list[float],
    duration_seconds: float,
) -> list[tuple[float, float]]:
    """
    功能说明：规范化 RMS 时间和值序列。
    参数说明：
    - rms_times/rms_values: RMS 时间轴与数值序列。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[tuple[float, float]]: 去裁剪、去异常后的 `(time, value)` 列表。
    异常说明：无。
    边界条件：长度不一致时按最短长度截断。
    """
    pair_count = min(len(rms_times), len(rms_values))
    if pair_count <= 0:
        return []
    safe_duration = max(0.0, float(duration_seconds))
    pairs: list[tuple[float, float]] = []
    for index in range(pair_count):
        time_value = max(0.0, min(safe_duration, _safe_float(rms_times[index], 0.0)))
        rms_value = max(0.0, _safe_float(rms_values[index], 0.0))
        pairs.append((time_value, rms_value))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _merge_close_intervals(
    intervals: list[dict[str, float]],
    merge_gap_seconds: float,
) -> list[dict[str, float]]:
    """
    功能说明：合并间距很短的相邻活动区间。
    参数说明：
    - intervals: 原始活动区间列表。
    - merge_gap_seconds: 允许闭合的短间隙阈值（秒）。
    返回值：
    - list[dict[str, float]]: 合并后的活动区间。
    异常说明：无。
    边界条件：空列表返回空。
    """
    if not intervals:
        return []
    safe_gap = max(0.0, float(merge_gap_seconds))
    merged: list[dict[str, float]] = [dict(intervals[0])]
    for item in intervals[1:]:
        current = dict(item)
        previous = merged[-1]
        gap_seconds = _safe_float(current.get("start_time", 0.0), 0.0) - _safe_float(previous.get("end_time", 0.0), 0.0)
        if gap_seconds <= safe_gap + EPSILON_SECONDS:
            previous["end_time"] = _round_time(
                max(
                    _safe_float(previous.get("end_time", 0.0), 0.0),
                    _safe_float(current.get("end_time", 0.0), 0.0),
                )
            )
            previous["duration"] = _round_time(
                max(
                    0.0,
                    _safe_float(previous.get("end_time", 0.0), 0.0)
                    - _safe_float(previous.get("start_time", 0.0), 0.0),
                )
            )
            continue
        merged.append(current)
    return merged


def _filter_short_intervals(
    intervals: list[dict[str, float]],
    min_interval_seconds: float,
) -> list[dict[str, float]]:
    """
    功能说明：过滤极短活动区间，降低分离噪声带来的假阳性。
    参数说明：
    - intervals: 活动区间列表。
    - min_interval_seconds: 最小时长阈值（秒）。
    返回值：
    - list[dict[str, float]]: 过滤后的活动区间。
    异常说明：无。
    边界条件：阈值小于等于0时直接返回原列表。
    """
    safe_min_interval = max(0.0, float(min_interval_seconds))
    if safe_min_interval <= EPSILON_SECONDS:
        return [dict(item) for item in intervals]
    filtered: list[dict[str, float]] = []
    for item in intervals:
        duration_seconds = max(
            0.0,
            _safe_float(item.get("end_time", 0.0), 0.0) - _safe_float(item.get("start_time", 0.0), 0.0),
        )
        if duration_seconds + EPSILON_SECONDS < safe_min_interval:
            continue
        filtered.append(dict(item))
    return filtered


def build_track_activity_intervals(
    rms_times: list[float],
    rms_values: list[float],
    duration_seconds: float,
    *,
    enter_quantile: float,
    exit_quantile: float,
    min_interval_seconds: float,
    merge_gap_seconds: float,
    activity_name: str,
) -> dict[str, Any]:
    """
    功能说明：基于单路 RMS 提取活动时间窗。
    参数说明：
    - rms_times/rms_values: RMS 时间轴与数值序列。
    - duration_seconds: 音频总时长（秒）。
    - enter_quantile: 活动进入阈值分位点。
    - exit_quantile: 活动退出阈值分位点。
    - min_interval_seconds: 最小保留时长（秒）。
    - merge_gap_seconds: 短间隙闭合阈值（秒）。
    - activity_name: 活动名称，仅用于结果可读性。
    返回值：
    - dict[str, Any]: 包含阈值、区间列表与统计信息的结果。
    异常说明：无。
    边界条件：输入为空时返回空区间。
    """
    pairs = _normalize_rms_pairs(
        rms_times=rms_times,
        rms_values=rms_values,
        duration_seconds=duration_seconds,
    )
    safe_enter_quantile = max(0.0, min(1.0, float(enter_quantile)))
    safe_exit_quantile = max(0.0, min(safe_enter_quantile, float(exit_quantile)))
    if not pairs:
        return {
            "activity_name": str(activity_name),
            "enter_threshold": RMS_SILENCE_FLOOR,
            "exit_threshold": RMS_SILENCE_FLOOR,
            "intervals": [],
        }

    values = [item[1] for item in pairs]
    enter_threshold = _estimate_active_threshold(values, quantile=safe_enter_quantile)
    exit_threshold = _estimate_active_threshold(values, quantile=safe_exit_quantile)
    exit_threshold = min(enter_threshold, exit_threshold)

    raw_intervals: list[dict[str, float]] = []
    active_start: float | None = None
    for time_value, rms_value in pairs:
        if active_start is None:
            if rms_value >= enter_threshold - EPSILON_SECONDS:
                active_start = time_value
            continue
        if rms_value < exit_threshold - EPSILON_SECONDS:
            raw_intervals.append(
                {
                    "start_time": _round_time(active_start),
                    "end_time": _round_time(max(active_start, time_value)),
                    "duration": _round_time(max(0.0, time_value - active_start)),
                }
            )
            active_start = None
    if active_start is not None:
        safe_duration = max(0.0, float(duration_seconds))
        raw_intervals.append(
            {
                "start_time": _round_time(active_start),
                "end_time": _round_time(max(active_start, safe_duration)),
                "duration": _round_time(max(0.0, safe_duration - active_start)),
            }
        )

    merged_intervals = _merge_close_intervals(
        intervals=raw_intervals,
        merge_gap_seconds=merge_gap_seconds,
    )
    filtered_intervals = _filter_short_intervals(
        intervals=merged_intervals,
        min_interval_seconds=min_interval_seconds,
    )
    return {
        "activity_name": str(activity_name),
        "enter_threshold": _round_time(enter_threshold),
        "exit_threshold": _round_time(exit_threshold),
        "intervals": filtered_intervals,
    }


def build_dual_track_activity_windows(
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    duration_seconds: float,
    *,
    vocal_enter_quantile: float,
    vocal_exit_quantile: float,
    vocal_min_interval_seconds: float,
    vocal_merge_gap_seconds: float,
) -> dict[str, Any]:
    """
    功能说明：统一提取人声与伴奏两路活动时间窗。
    参数说明：
    - vocal_rms_times/vocal_rms_values: 人声 RMS 序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏 RMS 序列。
    - duration_seconds: 音频总时长（秒）。
    - vocal_enter_quantile/vocal_exit_quantile: 人声活动阈值分位点。
    - vocal_min_interval_seconds: 人声最小保留时长（秒）。
    - vocal_merge_gap_seconds: 人声短间隙闭合阈值（秒）。
    返回值：
    - dict[str, Any]: 双路活动窗与阈值信息。
    异常说明：无。
    边界条件：任何一路为空时，该路返回空区间。
    """
    vocal_activity = build_track_activity_intervals(
        rms_times=vocal_rms_times,
        rms_values=vocal_rms_values,
        duration_seconds=duration_seconds,
        enter_quantile=vocal_enter_quantile,
        exit_quantile=vocal_exit_quantile,
        min_interval_seconds=vocal_min_interval_seconds,
        merge_gap_seconds=vocal_merge_gap_seconds,
        activity_name="vocal",
    )
    accompaniment_activity = build_track_activity_intervals(
        rms_times=accompaniment_rms_times,
        rms_values=accompaniment_rms_values,
        duration_seconds=duration_seconds,
        enter_quantile=ACCOMPANIMENT_ENTER_QUANTILE,
        exit_quantile=ACCOMPANIMENT_EXIT_QUANTILE,
        min_interval_seconds=ACCOMPANIMENT_MIN_INTERVAL_SECONDS,
        merge_gap_seconds=ACCOMPANIMENT_MERGE_GAP_SECONDS,
        activity_name="accompaniment",
    )
    return {
        "vocal": vocal_activity,
        "accompaniment": accompaniment_activity,
    }


def _window_overlap_seconds(
    window_start: float,
    window_end: float,
    intervals: list[dict[str, Any]],
) -> float:
    """
    功能说明：计算窗口与活动区间集合的总重叠时长。
    参数说明：
    - window_start/window_end: 目标窗口边界。
    - intervals: 活动区间集合。
    返回值：
    - float: 总重叠时长（秒）。
    异常说明：无。
    边界条件：区间为空时返回0。
    """
    overlap_seconds = 0.0
    for item in intervals:
        interval_start = _safe_float(item.get("start_time", 0.0), 0.0)
        interval_end = max(interval_start, _safe_float(item.get("end_time", interval_start), interval_start))
        overlap_seconds += max(0.0, min(window_end, interval_end) - max(window_start, interval_start))
    return overlap_seconds


def classify_window_roles(
    windows: list[dict[str, Any]],
    vocal_activity_intervals: list[dict[str, Any]],
    accompaniment_activity_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    功能说明：为窗口添加 role 分类。
    参数说明：
    - windows: 窗口列表。
    - vocal_activity_intervals: 人声活动区间列表。
    - accompaniment_activity_intervals: 伴奏活动区间列表。
    返回值：
    - list[dict[str, Any]]: 含 role 的窗口列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：歌词句窗口固定 role=lyric；其他窗口按活动区间重叠判定。
    """
    if not windows:
        return []

    output: list[dict[str, Any]] = []
    for window_item in sorted(windows, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0)):
        rewritten = dict(window_item)
        start_time = _safe_float(rewritten.get("start_time", 0.0), 0.0)
        end_time = max(start_time, _safe_float(rewritten.get("end_time", start_time), start_time))
        rewritten["start_time"] = _round_time(start_time)
        rewritten["end_time"] = _round_time(end_time)
        rewritten["duration"] = _round_time(max(0.0, end_time - start_time))

        if str(rewritten.get("window_role_hint", "other")).lower().strip() == "lyric":
            rewritten["role"] = "lyric"
            rewritten["merge_action"] = "keep_original"
            rewritten["vocal_active_overlap_seconds"] = _round_time(
                _window_overlap_seconds(start_time, end_time, vocal_activity_intervals)
            )
            rewritten["accompaniment_active_overlap_seconds"] = _round_time(
                _window_overlap_seconds(start_time, end_time, accompaniment_activity_intervals)
            )
            output.append(rewritten)
            continue

        vocal_overlap_seconds = _window_overlap_seconds(start_time, end_time, vocal_activity_intervals)
        accompaniment_overlap_seconds = _window_overlap_seconds(
            start_time,
            end_time,
            accompaniment_activity_intervals,
        )
        vocal_active = vocal_overlap_seconds > EPSILON_SECONDS
        accompaniment_active = accompaniment_overlap_seconds > EPSILON_SECONDS

        role = "silence"
        if vocal_active:
            role = "chant"
        elif accompaniment_active:
            role = "inst"

        rewritten["role"] = role
        rewritten["vocal_active_overlap_seconds"] = _round_time(vocal_overlap_seconds)
        rewritten["accompaniment_active_overlap_seconds"] = _round_time(accompaniment_overlap_seconds)
        rewritten["merge_action"] = "keep_original"
        output.append(rewritten)
    return output
