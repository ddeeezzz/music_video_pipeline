"""
文件用途：对窗口执行四分类（lyric/chant/inst/silence）。
核心流程：歌词窗口直标 lyric -> 其他窗口按双路RMS峰值判定。
输入输出：输入窗口与RMS序列，输出带 role 的窗口列表。
依赖说明：仅依赖标准库。
维护说明：本文件只负责分类，不负责并段与边界矫正。
"""

# 标准库：用于类型提示
from typing import Any


# 常量：浮点比较容差
EPSILON_SECONDS = 1e-6


# 常量：RMS静音下界
RMS_SILENCE_FLOOR = 0.003


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


def _rms_value_at(timestamp: float, rms_times: list[float], rms_values: list[float]) -> float:
    """
    功能说明：取最邻近采样点RMS值。
    参数说明：
    - timestamp: 目标时间。
    - rms_times/rms_values: RMS时间和值序列。
    返回值：
    - float: 对应RMS值。
    异常说明：无。
    边界条件：空序列返回0。
    """
    if not rms_times or not rms_values:
        return 0.0
    if len(rms_times) != len(rms_values):
        pair_count = min(len(rms_times), len(rms_values))
        rms_times = rms_times[:pair_count]
        rms_values = rms_values[:pair_count]
    nearest_index = min(
        range(len(rms_times)),
        key=lambda index: abs(_safe_float(rms_times[index], 0.0) - float(timestamp)),
    )
    return max(0.0, _safe_float(rms_values[nearest_index], 0.0))


def _peak_rms_in_window(
    window_start: float,
    window_end: float,
    rms_times: list[float],
    rms_values: list[float],
) -> float:
    """
    功能说明：计算窗口内RMS峰值。
    参数说明：
    - window_start/window_end: 窗口起止时间。
    - rms_times/rms_values: RMS序列。
    返回值：
    - float: 峰值RMS。
    异常说明：无。
    边界条件：无采样点时回退中点采样。
    """
    if window_end - window_start <= EPSILON_SECONDS:
        return 0.0
    samples: list[float] = []
    for index, time_item in enumerate(rms_times):
        time_value = _safe_float(time_item, window_start)
        if window_start <= time_value <= window_end and index < len(rms_values):
            samples.append(max(0.0, _safe_float(rms_values[index], 0.0)))
    if not samples:
        mid_time = (window_start + window_end) / 2.0
        samples = [_rms_value_at(mid_time, rms_times, rms_values)]
    return max(samples) if samples else 0.0


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


def classify_window_roles(
    windows: list[dict[str, Any]],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
) -> list[dict[str, Any]]:
    """
    功能说明：为窗口添加 role 分类。
    参数说明：
    - windows: 窗口列表。
    - vocal_rms_times/vocal_rms_values: 人声RMS序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏RMS序列。
    返回值：
    - list[dict[str, Any]]: 含 role 的窗口列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：歌词句窗口固定 role=lyric。
    """
    if not windows:
        return []

    vocal_threshold = _estimate_active_threshold(vocal_rms_values, quantile=0.35)
    accompaniment_threshold = _estimate_active_threshold(accompaniment_rms_values, quantile=0.20)

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
            output.append(rewritten)
            continue

        vocal_peak = _peak_rms_in_window(start_time, end_time, vocal_rms_times, vocal_rms_values)
        accompaniment_peak = _peak_rms_in_window(start_time, end_time, accompaniment_rms_times, accompaniment_rms_values)
        vocal_active = vocal_peak > vocal_threshold + EPSILON_SECONDS
        accompaniment_active = accompaniment_peak > accompaniment_threshold + EPSILON_SECONDS

        role = "silence"
        if vocal_active:
            role = "chant"
        elif accompaniment_active:
            role = "inst"

        rewritten["role"] = role
        rewritten["vocal_peak_rms"] = _round_time(vocal_peak)
        rewritten["accompaniment_peak_rms"] = _round_time(accompaniment_peak)
        rewritten["merge_action"] = "keep_original"
        output.append(rewritten)
    return output
