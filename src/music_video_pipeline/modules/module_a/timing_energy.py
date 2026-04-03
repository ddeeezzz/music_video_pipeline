"""
文件用途：提供模块A的时间戳与能量计算工具函数。
核心流程：计算分段能量特征、节拍构建与时间戳归一化。
输入输出：输入分段与声学序列，输出能量特征与时间工具结果。
依赖说明：依赖标准库 bisect 与类型提示。
维护说明：时间戳单位统一为秒并保留3位小数。
"""

# 标准库：二分检索
import bisect
# 标准库：类型提示
from typing import Any


def _build_energy_features(
    segments: list[dict[str, Any]],
    rms_times: list[float],
    rms_values: list[float],
    beat_candidates: list[float],
) -> list[dict[str, Any]]:
    """
    功能说明：按小段落计算能量等级、趋势与节奏紧张度。
    参数说明：
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    - beat_candidates: 节拍候选时间戳列表（秒）。
    返回值：
    - list[dict[str, Any]]: 每个分段对应的能量特征列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not segments:
        return []
    if not rms_times or not rms_values:
        return _build_fallback_energy_features(segments)

    safe_max = max(max(rms_values), 1e-6)
    features: list[dict[str, Any]] = []
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        duration = max(0.1, end_time - start_time)

        value_list = _slice_rms(start_time, end_time, rms_times, rms_values)
        if not value_list:
            value_list = [_rms_value_at(start_time, rms_times, rms_values), _rms_value_at(end_time, rms_times, rms_values)]

        mean_energy = sum(value_list) / len(value_list)
        normalized = max(0.0, min(1.0, mean_energy / safe_max))

        first_half = value_list[: max(1, len(value_list) // 2)]
        second_half = value_list[max(1, len(value_list) // 2) :]
        trend_delta = (sum(second_half) / max(1, len(second_half))) - (sum(first_half) / max(1, len(first_half)))

        if normalized < 0.33:
            energy_level = "low"
        elif normalized < 0.66:
            energy_level = "mid"
        else:
            energy_level = "high"

        if trend_delta > 0.02:
            trend = "up"
        elif trend_delta < -0.02:
            trend = "down"
        else:
            trend = "flat"

        beat_count = sum(1 for beat in beat_candidates if start_time <= beat <= end_time)
        rhythm_tension = round(max(0.0, min(1.0, (beat_count / duration) / 4.0)), 3)

        features.append(
            {
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": rhythm_tension,
            }
        )
    return features


def _build_beats_from_timestamps(timestamps: list[float]) -> list[dict[str, Any]]:
    """
    功能说明：将最终小时戳映射为 beats 契约结构。
    参数说明：
    - timestamps: 时间戳列表（秒）。
    返回值：
    - list[dict[str, Any]]: 标准化 beats 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized = sorted(set(_round_time(item) for item in timestamps))
    if len(normalized) < 2:
        normalized = [0.0, 0.1]
    return [
        {
            "time": _round_time(time_value),
            "type": "major" if index % 4 == 0 else "minor",
            "source": "adaptive",
        }
        for index, time_value in enumerate(normalized)
    ]


def _build_beats_from_segments(segments: list[dict[str, Any]], fallback_timestamps: list[float]) -> list[dict[str, Any]]:
    """
    功能说明：优先基于分段边界构建 beats，缺失时回退旧时戳集合。
    参数说明：
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    - fallback_timestamps: 当分段不可用时使用的备选时间戳列表。
    返回值：
    - list[dict[str, Any]]: 标准化 beats 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not segments:
        return _build_beats_from_timestamps(fallback_timestamps)

    timestamps = [float(item["start_time"]) for item in segments]
    timestamps.append(float(segments[-1]["end_time"]))
    return _build_beats_from_timestamps(timestamps)


def _build_fallback_big_segments(duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：构建规则化大段落。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 规则生成的大段落列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    segment_count = min(len(labels), max(1, int(duration_seconds // 30) + 1))
    step = duration_seconds / segment_count

    output: list[dict[str, Any]] = []
    current_start = 0.0
    for index in range(segment_count):
        end_time = duration_seconds if index == segment_count - 1 else current_start + step
        output.append(
            {
                "segment_id": f"big_{index + 1:03d}",
                "start_time": _round_time(current_start),
                "end_time": _round_time(end_time),
                "label": labels[index % len(labels)],
            }
        )
        current_start = end_time
    return output


def _build_fallback_energy_features(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：RMS 不可用时的规则化能量特征。
    参数说明：
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    返回值：
    - list[dict[str, Any]]: 规则生成的能量特征列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    patterns = [
        ("low", "up", 0.30),
        ("mid", "up", 0.55),
        ("high", "flat", 0.85),
        ("mid", "down", 0.50),
        ("low", "flat", 0.25),
    ]
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        energy_level, trend, rhythm_tension = patterns[index % len(patterns)]
        output.append(
            {
                "start_time": _round_time(float(segment["start_time"])),
                "end_time": _round_time(float(segment["end_time"])),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": rhythm_tension,
            }
        )
    return output

def _build_grid_timestamps(duration_seconds: float, interval_seconds: float) -> list[float]:
    """
    功能说明：生成规则网格时间戳。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - interval_seconds: 以秒为单位的时间参数。
    返回值：
    - list[float]: 归一化后的时间戳列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    safe_interval = interval_seconds if interval_seconds > 0 else 0.5
    points: list[float] = []
    cursor = 0.0
    while cursor < duration_seconds:
        points.append(_round_time(cursor))
        cursor += safe_interval
    points.append(_round_time(duration_seconds))
    return _normalize_timestamp_list(points, duration_seconds)


def _normalize_timestamp_list(timestamps: list[float], duration_seconds: float) -> list[float]:
    """
    功能说明：归一化时间戳（裁剪、去重、升序、最小间隔）。
    参数说明：
    - timestamps: 时间戳列表（秒）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[float]: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    clipped = sorted({_clamp_time(value, duration_seconds) for value in timestamps})
    if not clipped:
        return [0.0, _round_time(duration_seconds)]

    filtered: list[float] = [clipped[0]]
    for value in clipped[1:]:
        if value - filtered[-1] >= 0.1:
            filtered.append(value)

    if filtered[0] > 0.0:
        filtered.insert(0, 0.0)
    else:
        filtered[0] = 0.0

    last_time = _round_time(duration_seconds)
    if filtered[-1] < last_time:
        filtered.append(last_time)
    else:
        filtered[-1] = last_time

    dedup: list[float] = [filtered[0]]
    for value in filtered[1:]:
        if value - dedup[-1] >= 0.1:
            dedup.append(value)
    if dedup[-1] < last_time:
        dedup.append(last_time)

    return [_round_time(value) for value in dedup]


def _find_big_segment(time_value: float, big_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：按时间定位所属大段落。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - big_segments: 大段落列表，每项包含起止时间、标签与段落ID。
    返回值：
    - dict[str, Any]: 命中的大段落字典对象。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    for item in big_segments:
        if float(item["start_time"]) <= time_value <= float(item["end_time"]):
            return item
    return big_segments[-1]


def _rms_value_at(time_value: float, rms_times: list[float], rms_values: list[float]) -> float:
    """
    功能说明：按时间点读取最邻近 RMS 值。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    返回值：
    - float: 目标时间点的RMS能量值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not rms_times or not rms_values:
        return 0.0
    index = bisect.bisect_left(rms_times, time_value)
    if index <= 0:
        return float(rms_values[0])
    if index >= len(rms_times):
        return float(rms_values[-1])

    left_gap = abs(time_value - rms_times[index - 1])
    right_gap = abs(rms_times[index] - time_value)
    target_index = index - 1 if left_gap <= right_gap else index
    return float(rms_values[target_index])


def _rms_delta_at(
    time_value: float,
    rms_times: list[float],
    rms_values: list[float],
    window_ms: float = 100.0,
) -> float:
    """
    功能说明：计算目标时间点前后的能量正向落差（瞬态爆发强度）。
    参数说明：
    - time_value: 目标时间戳（秒）。
    - rms_times: RMS 时间序列。
    - rms_values: RMS 数值序列。
    - window_ms: 回看时间窗口（毫秒），默认 100ms。
    返回值：
    - float: 正向能量落差值，越大代表突变越明显。
    异常说明：无。
    边界条件：当 RMS 数据为空或出现能量下降时返回 0.0。
    """
    if not rms_times or not rms_values:
        return 0.0
    window_seconds = max(0.0, window_ms / 1000.0)
    current_rms = _rms_value_at(time_value, rms_times, rms_values)
    previous_rms = _rms_value_at(time_value - window_seconds, rms_times, rms_values)
    return max(0.0, current_rms - previous_rms)


def _slice_rms(start_time: float, end_time: float, rms_times: list[float], rms_values: list[float]) -> list[float]:
    """
    功能说明：提取时间区间内 RMS 子集。
    参数说明：
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    - rms_times: RMS能量时间轴（秒）。
    - rms_values: RMS能量值序列。
    返回值：
    - list[float]: 指定时间区间内的RMS子序列。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    output: list[float] = []
    for index, time_value in enumerate(rms_times):
        if start_time <= time_value <= end_time:
            output.append(float(rms_values[index]))
    return output


def _snap_to_nearest_beat(target_time: float, beat_pool: list[float], threshold_seconds: float) -> float:
    """
    功能说明：歌词时间戳吸附到最近节拍点。
    参数说明：
    - target_time: 待吸附的目标时间戳（秒）。
    - beat_pool: 归一化后的节拍候选池（秒）。
    - threshold_seconds: 节拍吸附阈值（秒）。
    返回值：
    - float: 吸附后的时间戳（阈值外返回原始时间）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not beat_pool:
        return target_time

    insert_index = bisect.bisect_left(beat_pool, target_time)
    candidates: list[float] = []
    if insert_index > 0:
        candidates.append(beat_pool[insert_index - 1])
    if insert_index < len(beat_pool):
        candidates.append(beat_pool[insert_index])
    if not candidates:
        return target_time

    nearest = min(candidates, key=lambda value: abs(value - target_time))
    if abs(nearest - target_time) <= threshold_seconds:
        return nearest
    return target_time


def _clamp_time(time_value: float, duration_seconds: float) -> float:
    """
    功能说明：将时间戳限制在 [0, duration]。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - float: 裁剪到合法范围后的时间戳。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    safe_duration = max(0.1, duration_seconds)
    return max(0.0, min(safe_duration, float(time_value)))


def _round_time(time_value: float) -> float:
    """
    功能说明：统一时间戳保留 3 位小数。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    返回值：
    - float: 保留三位小数后的时间戳。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return round(float(time_value), 3)
