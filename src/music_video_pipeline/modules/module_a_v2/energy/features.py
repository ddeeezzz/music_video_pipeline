"""
文件用途：提供模块A V2能量特征构建能力。
核心流程：按最终小段切片RMS并计算自适应能量等级、趋势与节奏紧张度。
输入输出：输入 segments 与声学序列，输出 energy_features 列表。
依赖说明：依赖标准库 bisect 与 v2 时间工具。
维护说明：本文件只承载能量构建逻辑，不混入编排流程。
"""

# 标准库：二分检索
import bisect
# 标准库：类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time

# 常量：能量归一化低锚点分位数（用于抑制极值）
ENERGY_LOW_ANCHOR_QUANTILE = 0.15
# 常量：能量归一化高锚点分位数（用于抑制极值）
ENERGY_HIGH_ANCHOR_QUANTILE = 0.85
# 常量：能量等级低档分位阈值（分位归一化后）
ENERGY_LEVEL_LOW_QUANTILE = 0.35
# 常量：能量等级高档分位阈值（分位归一化后）
ENERGY_LEVEL_HIGH_QUANTILE = 0.72
# 常量：能量分布过平时判定“全中能量”的最小跨度
ENERGY_LEVEL_MIN_SPREAD = 0.08
# 常量：趋势判定阈值最小值（相对变化率）
TREND_GATE_MIN = 0.06
# 常量：趋势判定阈值最大值（相对变化率）
TREND_GATE_MAX = 0.30
# 常量：趋势阈值取样分位数（绝对相对变化率）
TREND_GATE_QUANTILE = 0.65
# 常量：低能量段趋势分母下限占全局峰值比例，避免静音段夸大趋势
TREND_BASELINE_RATIO = 0.18


def build_energy_features(
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
    边界条件：当 RMS 不可用时回退规则特征。
    """
    if not segments:
        return []
    if not rms_times or not rms_values:
        return _build_fallback_energy_features(segments)

    safe_max = max(max(rms_values), 1e-6)
    segment_stats: list[dict[str, Any]] = []
    mean_energy_list: list[float] = []
    relative_trend_abs_list: list[float] = []
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        duration = max(0.1, end_time - start_time)

        value_list = _slice_rms(start_time, end_time, rms_times, rms_values)
        if not value_list:
            value_list = [_rms_value_at(start_time, rms_times, rms_values), _rms_value_at(end_time, rms_times, rms_values)]

        mean_energy = sum(value_list) / len(value_list)
        first_half = value_list[: max(1, len(value_list) // 2)]
        second_half = value_list[max(1, len(value_list) // 2) :]
        trend_delta = (sum(second_half) / max(1, len(second_half))) - (sum(first_half) / max(1, len(first_half)))
        trend_baseline = max(mean_energy, safe_max * TREND_BASELINE_RATIO, 1e-6)
        relative_trend = trend_delta / trend_baseline

        beat_count = sum(1 for beat in beat_candidates if start_time <= beat <= end_time)
        rhythm_tension = round(max(0.0, min(1.0, (beat_count / duration) / 4.0)), 3)

        segment_stats.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "mean_energy": mean_energy,
                "relative_trend": relative_trend,
                "rhythm_tension": rhythm_tension,
            }
        )
        mean_energy_list.append(mean_energy)
        relative_trend_abs_list.append(abs(relative_trend))

    normalized_mean_list = _normalize_by_quantile_anchors(mean_energy_list)
    raw_energy_spread = max(mean_energy_list) - min(mean_energy_list)
    raw_energy_baseline = max(max(mean_energy_list), 1e-6)
    raw_energy_spread_ratio = raw_energy_spread / raw_energy_baseline
    energy_low_gate = _percentile(normalized_mean_list, ENERGY_LEVEL_LOW_QUANTILE)
    energy_high_gate = _percentile(normalized_mean_list, ENERGY_LEVEL_HIGH_QUANTILE)
    if energy_high_gate <= energy_low_gate:
        energy_high_gate = min(1.0, energy_low_gate + 0.12)

    trend_gate = _clamp(_percentile(relative_trend_abs_list, TREND_GATE_QUANTILE), TREND_GATE_MIN, TREND_GATE_MAX)

    features: list[dict[str, Any]] = []
    for index, segment_stat in enumerate(segment_stats):
        start_time = float(segment_stat["start_time"])
        end_time = float(segment_stat["end_time"])
        normalized_mean = normalized_mean_list[index]
        relative_trend = float(segment_stat["relative_trend"])

        if raw_energy_spread_ratio < ENERGY_LEVEL_MIN_SPREAD:
            energy_level = "mid"
        elif normalized_mean <= energy_low_gate:
            energy_level = "low"
        elif normalized_mean >= energy_high_gate:
            energy_level = "high"
        else:
            energy_level = "mid"

        if relative_trend > trend_gate:
            trend = "up"
        elif relative_trend < (-trend_gate):
            trend = "down"
        else:
            trend = "flat"

        features.append(
            {
                "start_time": round_time(start_time),
                "end_time": round_time(end_time),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": float(segment_stat["rhythm_tension"]),
            }
        )
    return features


def _build_fallback_energy_features(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：RMS 不可用时的规则化能量特征。
    参数说明：
    - segments: 小段落列表，每项包含起止时间、标签与归属大段ID。
    返回值：
    - list[dict[str, Any]]: 规则生成的能量特征列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循固定模板循环生成。
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
                "start_time": round_time(float(segment["start_time"])),
                "end_time": round_time(float(segment["end_time"])),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": rhythm_tension,
            }
        )
    return output


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
    边界条件：RMS 为空时返回 0.0。
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
    边界条件：无命中样本时返回空列表。
    """
    output: list[float] = []
    for index, time_value in enumerate(rms_times):
        if start_time <= time_value <= end_time:
            output.append(float(rms_values[index]))
    return output


def _percentile(values: list[float], quantile: float) -> float:
    """
    功能说明：计算线性插值分位数。
    参数说明：
    - values: 输入数值序列。
    - quantile: 分位比例，取值范围 [0.0, 1.0]。
    返回值：
    - float: 对应分位数值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空数组返回 0.0，单值数组返回该值。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    safe_quantile = _clamp(quantile, 0.0, 1.0)
    position = safe_quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    blend = position - lower_index
    return (sorted_values[lower_index] * (1.0 - blend)) + (sorted_values[upper_index] * blend)


def _normalize_by_quantile_anchors(values: list[float]) -> list[float]:
    """
    功能说明：按分位锚点对序列做鲁棒归一化。
    参数说明：
    - values: 原始数值序列。
    返回值：
    - list[float]: 归一化后的序列，范围 [0.0, 1.0]。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：分布过平时回退到 0.5 中值序列。
    """
    if not values:
        return []

    low_anchor = _percentile(values, ENERGY_LOW_ANCHOR_QUANTILE)
    high_anchor = _percentile(values, ENERGY_HIGH_ANCHOR_QUANTILE)
    if high_anchor <= low_anchor:
        low_anchor = min(float(item) for item in values)
        high_anchor = max(float(item) for item in values)
    if high_anchor <= low_anchor:
        return [0.5 for _ in values]

    denominator = high_anchor - low_anchor
    return [_clamp((float(item) - low_anchor) / denominator, 0.0, 1.0) for item in values]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """
    功能说明：限制数值在指定区间内。
    参数说明：
    - value: 输入值。
    - min_value: 下界。
    - max_value: 上界。
    返回值：
    - float: 截断后的数值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当下界大于上界时按下界返回。
    """
    if min_value > max_value:
        return float(min_value)
    return float(max(min_value, min(max_value, value)))
