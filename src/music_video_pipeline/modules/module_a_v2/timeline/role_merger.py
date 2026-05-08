"""
文件用途：执行窗口细分与并段（长窗口按节拍细分 + tiny合并）。
核心流程：先按节拍切分长非歌词窗口，再按小节阈值合并短窗口。
输入输出：输入已分类窗口与节拍，输出切分后窗口或并段后窗口。
依赖说明：仅依赖标准库。
维护说明：本文件只负责窗口细分和并段，不负责边界矫正。
"""

# 标准库：用于数学计算
import math
# 标准库：用于统计中位数
import statistics
# 标准库：用于类型提示
from typing import Any


# 常量：浮点比较容差
EPSILON_SECONDS = 1e-6


# 常量：tiny并段默认阈值（小节）
DEFAULT_TINY_MERGE_BARS = 0.9
# 常量：major局部onset峰值能量窗口（小节）
MAJOR_ONSET_ENERGY_WINDOW_BARS = 0.5
# 常量：beat与最近onset的最远候选距离（小节），超过即剔除
BEAT_ONSET_MAX_DISTANCE_BARS = 0.25
# 常量：onset能量稳健归一化低分位
ONSET_ENERGY_P10 = 0.10
# 常量：onset能量稳健归一化高分位
ONSET_ENERGY_P90 = 0.90
# 常量：chant 人声主导判定阈值（vocal_rms >= accompaniment_rms * ratio）
CHANT_VOCAL_DOMINANCE_RATIO = 1.05
# 常量：inst/silence 角色分数权重
ROLE_SCORE_WEIGHTS_INST = {
    "chroma_delta": 0.55,
    "onset_delta": 0.25,
    "energy": 0.20,
    "f0_delta": 0.00,
}
# 常量：chant 角色分数权重
ROLE_SCORE_WEIGHTS_CHANT = {
    "chroma_delta": 0.10,
    "onset_delta": 0.25,
    "energy": 0.15,
    "f0_delta": 0.50,
}
# 常量：other 统一重拍切分权重（切分时不预设 chant/inst/silence 细角色）
ROLE_SCORE_WEIGHTS_OTHER = {
    "chroma_delta": 0.34,
    "onset_delta": 0.33,
    "energy": 0.33,
    "f0_delta": 0.00,
}

# 常量：tiny源窗口处理优先级（值越小越先处理）
TINY_SOURCE_ROLE_PRIORITY = {
    "lyric": 0,
    "chant": 1,
    "inst": 2,
    "silence": 3,
}
# 常量：tiny 相似度权重 - onset
TINY_SIMILARITY_WEIGHT_ONSET = 0.35
# 常量：tiny 相似度权重 - energy
TINY_SIMILARITY_WEIGHT_ENERGY = 0.20
# 常量：tiny 相似度权重 - chroma
TINY_SIMILARITY_WEIGHT_CHROMA = 0.25
# 常量：tiny 相似度权重 - voiced/f0
TINY_SIMILARITY_WEIGHT_VOICED_F0 = 0.20
# 常量：tiny onset 摘要窗口最小秒数，避免极短段密度失真
TINY_ONSET_SUMMARY_MIN_SECONDS = 0.2
# 常量：tiny onset 密度归一化上限（每秒）
TINY_ONSET_DENSITY_CAP = 12.0
# 常量：tiny energy 差值映射上限
TINY_ENERGY_DIFF_CAP = 1.0
# 常量：tiny voiced 比例差值映射上限
TINY_VOICED_RATIO_DIFF_CAP = 1.0
# 常量：tiny F0 半音差值映射上限
TINY_F0_MIDI_DIFF_CAP = 12.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：安全转换为浮点数。
    参数说明：
    - value: 输入对象。
    - default: 转换失败回退值。
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


def _window_duration(window_item: dict[str, Any]) -> float:
    """
    功能说明：计算窗口时长。
    参数说明：
    - window_item: 窗口对象。
    返回值：
    - float: 时长（秒）。
    异常说明：无。
    边界条件：负值按0处理。
    """
    start_time = _safe_float(window_item.get("start_time", 0.0), 0.0)
    end_time = _safe_float(window_item.get("end_time", start_time), start_time)
    return max(0.0, end_time - start_time)


def _clamp_unit_interval(value: float) -> float:
    """
    功能说明：将数值夹到 0~1 区间。
    参数说明：
    - value: 输入值。
    返回值：
    - float: 截断后的数值。
    异常说明：无。
    边界条件：NaN/inf 已由上游规避。
    """
    return min(1.0, max(0.0, float(value)))


def _quantile(values: list[float], quantile: float) -> float:
    """
    功能说明：计算线性插值分位数。
    参数说明：
    - values: 样本列表。
    - quantile: 分位点（0~1）。
    返回值：
    - float: 分位值。
    异常说明：无。
    边界条件：空样本返回0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    safe_q = max(0.0, min(1.0, float(quantile)))
    position = safe_q * (len(sorted_values) - 1)
    left_index = int(position)
    right_index = min(len(sorted_values) - 1, left_index + 1)
    if left_index == right_index:
        return sorted_values[left_index]
    weight = position - left_index
    return sorted_values[left_index] * (1.0 - weight) + sorted_values[right_index] * weight


def _collect_major_times(beats: list[dict[str, Any]]) -> list[float]:
    """
    功能说明：提取重拍（major）时间序列。
    参数说明：
    - beats: 结构化节拍列表。
    返回值：
    - list[float]: 升序重拍时间列表。
    异常说明：无。
    边界条件：缺失时返回空列表。
    """
    return sorted(
        {
            _safe_float(item.get("time", 0.0), 0.0)
            for item in beats
            if str(item.get("type", "")).lower().strip() == "major"
            and _safe_float(item.get("time", 0.0), 0.0) >= 0.0
        }
    )


def _collect_beat_rows(beats: list[dict[str, Any]]) -> list[tuple[float, str]]:
    """
    功能说明：提取节拍时间与类型（major/minor）。
    参数说明：
    - beats: 结构化节拍列表。
    返回值：
    - list[tuple[float, str]]: 升序节拍序列，元素为 (time, beat_type)。
    异常说明：无。
    边界条件：同时间多条记录优先保留 major。
    """
    merged_by_time: dict[float, str] = {}
    for item in beats:
        time_value = _round_time(_safe_float(item.get("time", 0.0), 0.0))
        if time_value < 0.0:
            continue
        beat_type = str(item.get("type", "")).lower().strip() or "unknown"
        previous_type = merged_by_time.get(time_value, "")
        if previous_type != "major" and beat_type == "major":
            merged_by_time[time_value] = "major"
        elif time_value not in merged_by_time:
            merged_by_time[time_value] = beat_type
    return [(time_item, merged_by_time[time_item]) for time_item in sorted(merged_by_time.keys())]


def _normalize_onset_points(onset_points: list[dict[str, Any]] | None) -> list[dict[str, float]]:
    """
    功能说明：归一化 onset 点列表为 time+energy_raw 结构。
    参数说明：
    - onset_points: 原始onset点列表。
    返回值：
    - list[dict[str, float]]: 归一化并按时间排序后的onset点。
    异常说明：无。
    边界条件：空输入返回空列表；同时间点保留更高能量。
    """
    if not onset_points:
        return []
    merged_by_time: dict[float, float] = {}
    for item in onset_points:
        if not isinstance(item, dict):
            continue
        time_value = _round_time(max(0.0, _safe_float(item.get("time", 0.0), 0.0)))
        energy_raw = max(0.0, _safe_float(item.get("energy_raw", 0.0), 0.0))
        previous_energy = merged_by_time.get(time_value, 0.0)
        if energy_raw > previous_energy:
            merged_by_time[time_value] = energy_raw
    return [
        {"time": round(time_value, 6), "energy_raw": round(energy_raw, 6)}
        for time_value, energy_raw in sorted(merged_by_time.items(), key=lambda pair: pair[0])
    ]


def _normalize_robust_scores(raw_values: list[float]) -> list[float]:
    """
    功能说明：对分数序列执行稳健分位归一化（p10/p90）。
    参数说明：
    - raw_values: 原始分数列表。
    返回值：
    - list[float]: 归一化后分数（0~1）。
    异常说明：无。
    边界条件：空列表返回空；p90≈p10 时退化为二值归一化。
    """
    if not raw_values:
        return []
    low_quantile = _quantile(raw_values, ONSET_ENERGY_P10)
    high_quantile = _quantile(raw_values, ONSET_ENERGY_P90)
    if high_quantile - low_quantile <= EPSILON_SECONDS:
        return [1.0 if item > EPSILON_SECONDS else 0.0 for item in raw_values]
    denominator = max(EPSILON_SECONDS, high_quantile - low_quantile)
    normalized_values: list[float] = []
    for item in raw_values:
        clipped_value = min(high_quantile, max(low_quantile, float(item)))
        normalized_values.append(min(1.0, max(0.0, (clipped_value - low_quantile) / denominator)))
    return normalized_values


def _normalize_chroma_points(chroma_points: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    功能说明：归一化 chroma 点列表为 time+12维向量。
    参数说明：
    - chroma_points: 原始 chroma 点列表。
    返回值：
    - list[dict[str, Any]]: 归一化并按时间排序结果。
    异常说明：无。
    边界条件：空输入返回空列表；同时间保留总能量更高的向量。
    """
    if not chroma_points:
        return []
    merged: dict[float, list[float]] = {}
    for item in chroma_points:
        if not isinstance(item, dict):
            continue
        raw_vector = item.get("chroma", [])
        if not isinstance(raw_vector, list) or len(raw_vector) < 12:
            continue
        time_value = _round_time(max(0.0, _safe_float(item.get("time", 0.0), 0.0)))
        chroma_vector = [max(0.0, _safe_float(raw_vector[index], 0.0)) for index in range(12)]
        previous_vector = merged.get(time_value, [])
        if sum(chroma_vector) >= sum(previous_vector):
            merged[time_value] = chroma_vector
    return [
        {"time": round(time_value, 6), "chroma": [round(float(value), 6) for value in vector]}
        for time_value, vector in sorted(merged.items(), key=lambda pair: pair[0])
    ]


def _normalize_f0_points(f0_points: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    功能说明：归一化 F0 点列表为 time+f0_hz+voiced+confidence。
    参数说明：
    - f0_points: 原始 F0 点列表。
    返回值：
    - list[dict[str, Any]]: 归一化并按时间排序结果。
    异常说明：无。
    边界条件：空输入返回空列表；同时间优先保留更可信且有声点。
    """
    if not f0_points:
        return []
    merged: dict[float, dict[str, Any]] = {}
    for item in f0_points:
        if not isinstance(item, dict):
            continue
        time_value = _round_time(max(0.0, _safe_float(item.get("time", 0.0), 0.0)))
        f0_hz = max(0.0, _safe_float(item.get("f0_hz", 0.0), 0.0))
        voiced = bool(item.get("voiced", False)) and f0_hz > EPSILON_SECONDS
        confidence = max(0.0, min(1.0, _safe_float(item.get("confidence", 0.0), 0.0)))
        current_item = {"time": round(time_value, 6), "f0_hz": round(f0_hz, 6), "voiced": voiced, "confidence": round(confidence, 6)}
        previous = merged.get(time_value)
        if previous is None:
            merged[time_value] = current_item
            continue
        previous_priority = (1 if bool(previous.get("voiced", False)) else 0, _safe_float(previous.get("confidence", 0.0), 0.0), _safe_float(previous.get("f0_hz", 0.0), 0.0))
        current_priority = (1 if voiced else 0, confidence, f0_hz)
        if current_priority >= previous_priority:
            merged[time_value] = current_item
    return [merged[time_key] for time_key in sorted(merged.keys())]


def _collect_points_in_window(
    points: list[dict[str, Any]],
    start_time: float,
    end_time: float,
) -> list[dict[str, Any]]:
    """
    功能说明：筛选窗口时间范围内的点序列。
    参数说明：
    - points: 点序列，要求存在 time 字段。
    - start_time/end_time: 窗口起止时间。
    返回值：
    - list[dict[str, Any]]: 落在窗口内的点。
    异常说明：无。
    边界条件：空输入返回空；采用闭区间容差。
    """
    if not points:
        return []
    lower_bound = float(start_time) - EPSILON_SECONDS
    upper_bound = float(end_time) + EPSILON_SECONDS
    output: list[dict[str, Any]] = []
    for item in points:
        time_value = _safe_float(item.get("time", 0.0), 0.0)
        if lower_bound <= time_value <= upper_bound:
            output.append(item)
    return output


def _collect_series_values_in_window(
    times: list[float],
    values: list[float],
    start_time: float,
    end_time: float,
) -> list[float]:
    """
    功能说明：筛选窗口范围内的时间序列数值。
    参数说明：
    - times/values: 时间和值序列。
    - start_time/end_time: 窗口起止时间。
    返回值：
    - list[float]: 落在窗口内的值列表。
    异常说明：无。
    边界条件：长度不齐时按较短长度截断。
    """
    pair_count = min(len(times), len(values))
    if pair_count <= 0:
        return []
    lower_bound = float(start_time) - EPSILON_SECONDS
    upper_bound = float(end_time) + EPSILON_SECONDS
    output: list[float] = []
    for index in range(pair_count):
        time_value = _safe_float(times[index], 0.0)
        if lower_bound <= time_value <= upper_bound:
            output.append(max(0.0, _safe_float(values[index], 0.0)))
    return output


def _median_vector(vectors: list[list[float]]) -> list[float]:
    """
    功能说明：计算 12 维向量序列的逐维中位数。
    参数说明：
    - vectors: 向量序列。
    返回值：
    - list[float]: 12 维中位向量。
    异常说明：无。
    边界条件：空输入返回空列表。
    """
    if not vectors:
        return []
    return [float(statistics.median([vector[index] for vector in vectors])) for index in range(12)]


def _cosine_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """
    功能说明：计算两个向量的余弦距离（1-cosine）。
    参数说明：
    - vector_a: 向量A。
    - vector_b: 向量B。
    返回值：
    - float: 距离值。
    异常说明：无。
    边界条件：向量范数过小时返回0。
    """
    if not vector_a or not vector_b:
        return 0.0
    dot_value = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a <= EPSILON_SECONDS or norm_b <= EPSILON_SECONDS:
        return 0.0
    cosine_sim = dot_value / max(EPSILON_SECONDS, norm_a * norm_b)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return max(0.0, 1.0 - cosine_sim)


def _hz_to_midi_value(f0_hz: float) -> float | None:
    """
    功能说明：将频率（Hz）转换为 MIDI 音高数值。
    参数说明：
    - f0_hz: 频率值。
    返回值：
    - float | None: MIDI 数值；无效频率返回 None。
    异常说明：无。
    边界条件：f0<=0 返回 None。
    """
    if f0_hz <= EPSILON_SECONDS:
        return None
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def _sample_local_rms(
    rms_times: list[float],
    rms_values: list[float],
    center_time: float,
    half_window_seconds: float,
) -> float:
    """
    功能说明：采样某个时间点附近的局部 RMS（优先窗口中位数，缺失时取最近值）。
    参数说明：
    - rms_times/rms_values: RMS 时间与数值序列。
    - center_time: 采样中心时间。
    - half_window_seconds: 局部窗口半径。
    返回值：
    - float: 局部 RMS 值。
    异常说明：无。
    边界条件：空序列返回0。
    """
    if not rms_times or not rms_values:
        return 0.0
    pair_count = min(len(rms_times), len(rms_values))
    if pair_count <= 0:
        return 0.0
    lower_bound = center_time - half_window_seconds
    upper_bound = center_time + half_window_seconds
    local_values = [
        max(0.0, _safe_float(rms_values[index], 0.0))
        for index in range(pair_count)
        if lower_bound - EPSILON_SECONDS <= _safe_float(rms_times[index], 0.0) <= upper_bound + EPSILON_SECONDS
    ]
    if local_values:
        return float(statistics.median(local_values))
    nearest_index = min(
        range(pair_count),
        key=lambda index: abs(_safe_float(rms_times[index], 0.0) - center_time),
    )
    return max(0.0, _safe_float(rms_values[nearest_index], 0.0))


def _compute_chroma_delta_raw(
    beat_time: float,
    chroma_points: list[dict[str, Any]],
    half_window_seconds: float,
) -> float:
    """
    功能说明：计算 beat 前后半窗的 chroma 差异（1-cosine）。
    参数说明：
    - beat_time: beat 时间。
    - chroma_points: chroma 点列表。
    - half_window_seconds: 半窗时长。
    返回值：
    - float: chroma 差异。
    异常说明：无。
    边界条件：任一侧无有效向量返回0。
    """
    if not chroma_points:
        return 0.0
    lower_bound = beat_time - half_window_seconds
    upper_bound = beat_time + half_window_seconds
    pre_vectors: list[list[float]] = []
    post_vectors: list[list[float]] = []
    for point in chroma_points:
        time_value = _safe_float(point.get("time", 0.0), 0.0)
        if time_value < lower_bound - EPSILON_SECONDS or time_value > upper_bound + EPSILON_SECONDS:
            continue
        vector = point.get("chroma", [])
        if not isinstance(vector, list) or len(vector) < 12:
            continue
        safe_vector = [max(0.0, _safe_float(vector[index], 0.0)) for index in range(12)]
        if time_value < beat_time - EPSILON_SECONDS:
            pre_vectors.append(safe_vector)
        else:
            post_vectors.append(safe_vector)
    pre_median_vector = _median_vector(pre_vectors)
    post_median_vector = _median_vector(post_vectors)
    if not pre_median_vector or not post_median_vector:
        return 0.0
    return _cosine_distance(pre_median_vector, post_median_vector)


def _compute_f0_delta_raw(
    beat_time: float,
    f0_points: list[dict[str, Any]],
    half_window_seconds: float,
) -> tuple[float, bool]:
    """
    功能说明：计算 beat 前后半窗的中位音高差（半音）。
    参数说明：
    - beat_time: beat 时间。
    - f0_points: F0 点列表。
    - half_window_seconds: 半窗时长。
    返回值：
    - tuple[float, bool]: (音高差, 是否存在有效F0)。
    异常说明：无。
    边界条件：任一侧无有效有声音高时返回 (0, False)。
    """
    if not f0_points:
        return 0.0, False
    lower_bound = beat_time - half_window_seconds
    upper_bound = beat_time + half_window_seconds
    pre_values: list[float] = []
    post_values: list[float] = []
    for point in f0_points:
        time_value = _safe_float(point.get("time", 0.0), 0.0)
        if time_value < lower_bound - EPSILON_SECONDS or time_value > upper_bound + EPSILON_SECONDS:
            continue
        if not bool(point.get("voiced", False)):
            continue
        midi_value = _hz_to_midi_value(_safe_float(point.get("f0_hz", 0.0), 0.0))
        if midi_value is None:
            continue
        if time_value < beat_time - EPSILON_SECONDS:
            pre_values.append(midi_value)
        else:
            post_values.append(midi_value)
    if not pre_values or not post_values:
        return 0.0, False
    return abs(float(statistics.median(post_values)) - float(statistics.median(pre_values))), True


def _build_tiny_window_summary(
    window_item: dict[str, Any],
    onset_points: list[dict[str, float]],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    chroma_points: list[dict[str, Any]],
    f0_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    功能说明：构建 tiny 相似度比较所需的窗口级摘要特征。
    参数说明：
    - window_item: 待摘要窗口。
    - onset_points: onset 点序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏 RMS 序列。
    - chroma_points: chroma 点序列。
    - f0_points: F0 点序列。
    返回值：
    - dict[str, Any]: onset/energy/chroma/voiced_f0 摘要结果。
    异常说明：无。
    边界条件：任一特征缺失时返回可退化字段，不抛异常。
    """
    start_time = _safe_float(window_item.get("start_time", 0.0), 0.0)
    end_time = _safe_float(window_item.get("end_time", start_time), start_time)
    duration = max(EPSILON_SECONDS, end_time - start_time)

    onset_rows = _collect_points_in_window(onset_points, start_time=start_time, end_time=end_time)
    onset_count = len(onset_rows)
    onset_density = min(
        TINY_ONSET_DENSITY_CAP,
        float(onset_count) / max(TINY_ONSET_SUMMARY_MIN_SECONDS, duration),
    )
    onset_energy_values = [max(0.0, _safe_float(item.get("energy_raw", 0.0), 0.0)) for item in onset_rows]
    onset_energy_median = float(statistics.median(onset_energy_values)) if onset_energy_values else 0.0

    energy_values = _collect_series_values_in_window(
        times=accompaniment_rms_times,
        values=accompaniment_rms_values,
        start_time=start_time,
        end_time=end_time,
    )
    energy_median = float(statistics.median(energy_values)) if energy_values else 0.0

    chroma_rows = _collect_points_in_window(chroma_points, start_time=start_time, end_time=end_time)
    chroma_vectors = [
        [max(0.0, _safe_float(vector[index], 0.0)) for index in range(12)]
        for vector in [item.get("chroma", []) for item in chroma_rows]
        if isinstance(vector, list) and len(vector) >= 12
    ]
    chroma_vector = _median_vector(chroma_vectors)

    f0_rows = _collect_points_in_window(f0_points, start_time=start_time, end_time=end_time)
    voiced_rows = [
        item for item in f0_rows
        if bool(item.get("voiced", False)) and _safe_float(item.get("f0_hz", 0.0), 0.0) > EPSILON_SECONDS
    ]
    voiced_ratio = (
        float(len(voiced_rows)) / max(1, len(f0_rows))
        if f0_rows
        else 0.0
    )
    voiced_midi_values = [
        midi_value
        for midi_value in [
            _hz_to_midi_value(_safe_float(item.get("f0_hz", 0.0), 0.0))
            for item in voiced_rows
        ]
        if midi_value is not None
    ]
    voiced_midi_median = float(statistics.median(voiced_midi_values)) if voiced_midi_values else None

    return {
        "duration": round(duration, 6),
        "onset_density": round(float(onset_density), 6),
        "onset_energy_median": round(float(onset_energy_median), 6),
        "energy_median": round(float(energy_median), 6),
        "chroma_vector": [round(float(value), 6) for value in chroma_vector] if chroma_vector else [],
        "voiced_ratio": round(float(voiced_ratio), 6),
        "voiced_midi_median": round(float(voiced_midi_median), 6) if voiced_midi_median is not None else None,
    }


def _difference_to_similarity(diff_value: float, diff_cap: float) -> float:
    """
    功能说明：将差值映射为 0~1 相似度。
    参数说明：
    - diff_value: 特征差值。
    - diff_cap: 差值上限，达到上限视为0相似度。
    返回值：
    - float: 相似度。
    异常说明：无。
    边界条件：上限非正时退化为完全相似。
    """
    safe_cap = max(EPSILON_SECONDS, float(diff_cap))
    return _clamp_unit_interval(1.0 - min(safe_cap, max(0.0, float(diff_value))) / safe_cap)


def _compute_tiny_similarity_components(
    source_summary: dict[str, Any],
    neighbor_summary: dict[str, Any],
) -> dict[str, float]:
    """
    功能说明：计算 tiny 源窗与邻窗的四项相似度。
    参数说明：
    - source_summary: tiny 源窗摘要。
    - neighbor_summary: 邻窗摘要。
    返回值：
    - dict[str, float]: 四项组件与总分。
    异常说明：无。
    边界条件：缺失特征时自动退化为保守比较。
    """
    onset_density_similarity = _difference_to_similarity(
        abs(_safe_float(source_summary.get("onset_density", 0.0), 0.0) - _safe_float(neighbor_summary.get("onset_density", 0.0), 0.0)),
        TINY_ONSET_DENSITY_CAP,
    )
    onset_energy_similarity = _difference_to_similarity(
        abs(_safe_float(source_summary.get("onset_energy_median", 0.0), 0.0) - _safe_float(neighbor_summary.get("onset_energy_median", 0.0), 0.0)),
        TINY_ENERGY_DIFF_CAP,
    )
    onset_similarity = _clamp_unit_interval((onset_density_similarity + onset_energy_similarity) / 2.0)

    energy_similarity = _difference_to_similarity(
        abs(_safe_float(source_summary.get("energy_median", 0.0), 0.0) - _safe_float(neighbor_summary.get("energy_median", 0.0), 0.0)),
        TINY_ENERGY_DIFF_CAP,
    )

    source_chroma_vector = source_summary.get("chroma_vector", [])
    neighbor_chroma_vector = neighbor_summary.get("chroma_vector", [])
    chroma_similarity = 0.0
    if isinstance(source_chroma_vector, list) and isinstance(neighbor_chroma_vector, list):
        chroma_similarity = _clamp_unit_interval(1.0 - _cosine_distance(source_chroma_vector, neighbor_chroma_vector))

    voiced_ratio_similarity = _difference_to_similarity(
        abs(_safe_float(source_summary.get("voiced_ratio", 0.0), 0.0) - _safe_float(neighbor_summary.get("voiced_ratio", 0.0), 0.0)),
        TINY_VOICED_RATIO_DIFF_CAP,
    )
    source_midi = source_summary.get("voiced_midi_median")
    neighbor_midi = neighbor_summary.get("voiced_midi_median")
    if source_midi is None or neighbor_midi is None:
        voiced_f0_similarity = voiced_ratio_similarity
    else:
        f0_similarity = _difference_to_similarity(
            abs(_safe_float(source_midi, 0.0) - _safe_float(neighbor_midi, 0.0)),
            TINY_F0_MIDI_DIFF_CAP,
        )
        voiced_f0_similarity = _clamp_unit_interval((voiced_ratio_similarity + f0_similarity) / 2.0)

    similarity_total = (
        TINY_SIMILARITY_WEIGHT_ONSET * onset_similarity
        + TINY_SIMILARITY_WEIGHT_ENERGY * energy_similarity
        + TINY_SIMILARITY_WEIGHT_CHROMA * chroma_similarity
        + TINY_SIMILARITY_WEIGHT_VOICED_F0 * voiced_f0_similarity
    )
    return {
        "onset_similarity": round(float(onset_similarity), 6),
        "energy_similarity": round(float(energy_similarity), 6),
        "chroma_similarity": round(float(chroma_similarity), 6),
        "voiced_f0_similarity": round(float(voiced_f0_similarity), 6),
        "total_similarity": round(float(_clamp_unit_interval(similarity_total)), 6),
    }


def _pick_pitch_source_for_chant(
    beat_time: float,
    half_window_seconds: float,
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    vocal_f0_points: list[dict[str, Any]],
    accompaniment_f0_points: list[dict[str, Any]],
) -> tuple[float, str]:
    """
    功能说明：为 chant 角色按局部能量主导选择 F0 源，并计算音高差。
    参数说明：见调用处。
    返回值：
    - tuple[float, str]: (f0_delta_raw, pitch_source)。
    异常说明：无。
    边界条件：两侧均无有效F0时返回 (0, "fallback")。
    """
    vocal_rms = _sample_local_rms(
        rms_times=vocal_rms_times,
        rms_values=vocal_rms_values,
        center_time=beat_time,
        half_window_seconds=half_window_seconds,
    )
    accompaniment_rms = _sample_local_rms(
        rms_times=accompaniment_rms_times,
        rms_values=accompaniment_rms_values,
        center_time=beat_time,
        half_window_seconds=half_window_seconds,
    )
    preferred_source = "vocals" if vocal_rms >= accompaniment_rms * CHANT_VOCAL_DOMINANCE_RATIO else "no_vocals"
    if preferred_source == "vocals":
        preferred_delta, preferred_valid = _compute_f0_delta_raw(beat_time, vocal_f0_points, half_window_seconds)
        if preferred_valid:
            return preferred_delta, "vocals"
        fallback_delta, fallback_valid = _compute_f0_delta_raw(beat_time, accompaniment_f0_points, half_window_seconds)
        if fallback_valid:
            return fallback_delta, "fallback"
        return 0.0, "fallback"

    preferred_delta, preferred_valid = _compute_f0_delta_raw(beat_time, accompaniment_f0_points, half_window_seconds)
    if preferred_valid:
        return preferred_delta, "no_vocals"
    fallback_delta, fallback_valid = _compute_f0_delta_raw(beat_time, vocal_f0_points, half_window_seconds)
    if fallback_valid:
        return fallback_delta, "fallback"
    return 0.0, "fallback"


def _resolve_role_weights(role: str) -> dict[str, float]:
    """
    功能说明：根据角色返回多特征融合权重。
    参数说明：
    - role: 角色名称。
    返回值：
    - dict[str, float]: 四项权重。
    异常说明：无。
    边界条件：未知角色按 inst/silence 权重。
    """
    if role == "chant":
        return ROLE_SCORE_WEIGHTS_CHANT
    if role == "other":
        return ROLE_SCORE_WEIGHTS_OTHER
    return ROLE_SCORE_WEIGHTS_INST


def _compute_beat_feature_scores(
    beat_rows: list[tuple[float, str]],
    onset_points: list[dict[str, float]],
    chroma_points: list[dict[str, Any]],
    vocal_f0_points: list[dict[str, Any]],
    accompaniment_f0_points: list[dict[str, Any]],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    bar_length_seconds: float,
    role: str,
) -> dict[float, dict[str, Any]]:
    """
    功能说明：计算每个 beat 的多特征分数（onset/chroma/f0 + 距离惩罚）。
    参数说明：见调用处。
    返回值：
    - dict[float, dict[str, Any]]: key=beat时间(6位小数)，value=打分元数据。
    异常说明：无。
    边界条件：无 beat 返回空字典。
    """
    if not beat_rows:
        return {}
    half_window_seconds = max(EPSILON_SECONDS, float(bar_length_seconds) * MAJOR_ONSET_ENERGY_WINDOW_BARS)
    max_onset_distance_seconds = max(EPSILON_SECONDS, float(bar_length_seconds) * BEAT_ONSET_MAX_DISTANCE_BARS)
    has_onset_reference = bool(onset_points)
    weights = _resolve_role_weights(role=role)

    time_keys: list[float] = []
    beat_types: list[str] = []
    energy_raw_scores: list[float] = []
    delta_raw_scores: list[float] = []
    nearest_distance_scores: list[float] = []
    distance_penalties: list[float] = []
    near_onset_flags: list[bool] = []
    chroma_raw_scores: list[float] = []
    f0_raw_scores: list[float] = []
    pitch_sources: list[str] = []

    for beat_time, beat_type in beat_rows:
        lower_bound = beat_time - half_window_seconds
        upper_bound = beat_time + half_window_seconds
        peak_energy = 0.0
        pre_peak_energy = 0.0
        post_peak_energy = 0.0
        nearest_onset_distance = float("inf")
        for onset_item in onset_points:
            onset_time = _safe_float(onset_item.get("time", 0.0), 0.0)
            nearest_onset_distance = min(nearest_onset_distance, abs(onset_time - beat_time))
            if onset_time < lower_bound - EPSILON_SECONDS or onset_time > upper_bound + EPSILON_SECONDS:
                continue
            energy_raw = max(0.0, _safe_float(onset_item.get("energy_raw", 0.0), 0.0))
            if energy_raw > peak_energy:
                peak_energy = energy_raw
            if onset_time < beat_time - EPSILON_SECONDS:
                pre_peak_energy = max(pre_peak_energy, energy_raw)
            else:
                post_peak_energy = max(post_peak_energy, energy_raw)

        if has_onset_reference:
            near_onset_candidate = nearest_onset_distance <= max_onset_distance_seconds + EPSILON_SECONDS
            distance_penalty = (
                max(0.0, 1.0 - min(1.0, nearest_onset_distance / max_onset_distance_seconds))
                if near_onset_candidate
                else 0.0
            )
        else:
            near_onset_candidate = True
            distance_penalty = 1.0
            nearest_onset_distance = 999.0

        chroma_delta_raw = _compute_chroma_delta_raw(
            beat_time=beat_time,
            chroma_points=chroma_points,
            half_window_seconds=half_window_seconds,
        )

        if role == "chant":
            f0_delta_raw, pitch_source = _pick_pitch_source_for_chant(
                beat_time=beat_time,
                half_window_seconds=half_window_seconds,
                vocal_rms_times=vocal_rms_times,
                vocal_rms_values=vocal_rms_values,
                accompaniment_rms_times=accompaniment_rms_times,
                accompaniment_rms_values=accompaniment_rms_values,
                vocal_f0_points=vocal_f0_points,
                accompaniment_f0_points=accompaniment_f0_points,
            )
        else:
            f0_delta_raw, has_f0 = _compute_f0_delta_raw(
                beat_time=beat_time,
                f0_points=accompaniment_f0_points,
                half_window_seconds=half_window_seconds,
            )
            pitch_source = "no_vocals" if has_f0 else "fallback"

        time_keys.append(_round_time(beat_time))
        beat_types.append(str(beat_type))
        energy_raw_scores.append(float(peak_energy))
        delta_raw_scores.append(float(abs(post_peak_energy - pre_peak_energy)))
        nearest_distance_scores.append(float(nearest_onset_distance))
        distance_penalties.append(float(distance_penalty))
        near_onset_flags.append(bool(near_onset_candidate))
        chroma_raw_scores.append(float(chroma_delta_raw))
        f0_raw_scores.append(float(max(0.0, f0_delta_raw)))
        pitch_sources.append(str(pitch_source))

    energy_norm_scores = _normalize_robust_scores(energy_raw_scores)
    delta_norm_scores = _normalize_robust_scores(delta_raw_scores)
    chroma_norm_scores = _normalize_robust_scores(chroma_raw_scores)
    f0_norm_scores = _normalize_robust_scores(f0_raw_scores)

    score_map: dict[float, dict[str, Any]] = {}
    for index, time_key in enumerate(time_keys):
        energy_norm = float(energy_norm_scores[index]) if index < len(energy_norm_scores) else 0.0
        onset_delta_norm = float(delta_norm_scores[index]) if index < len(delta_norm_scores) else 0.0
        chroma_norm = float(chroma_norm_scores[index]) if index < len(chroma_norm_scores) else 0.0
        f0_norm = float(f0_norm_scores[index]) if index < len(f0_norm_scores) else 0.0
        distance_penalty = float(distance_penalties[index])
        near_onset_candidate = bool(near_onset_flags[index])

        onset_delta_component = weights["onset_delta"] * onset_delta_norm
        energy_component = weights["energy"] * energy_norm
        chroma_component = weights["chroma_delta"] * chroma_norm
        f0_component = weights["f0_delta"] * f0_norm
        score_base = onset_delta_component + energy_component + chroma_component + f0_component
        score_total = score_base * distance_penalty if near_onset_candidate else 0.0

        score_map[time_key] = {
            "energy_raw": round(float(energy_raw_scores[index]), 6),
            "energy_norm": round(energy_norm, 6),
            "delta_raw": round(float(delta_raw_scores[index]), 6),
            "delta_norm": round(onset_delta_norm, 6),
            "chroma_delta_raw": round(float(chroma_raw_scores[index]), 6),
            "chroma_delta_norm": round(chroma_norm, 6),
            "f0_delta_raw": round(float(f0_raw_scores[index]), 6),
            "f0_delta_norm": round(f0_norm, 6),
            "pitch_source": pitch_sources[index],
            "score": round(score_total, 6),
            "score_base": round(score_base, 6),
            "score_components": {
                "onset_delta": round(onset_delta_component * distance_penalty, 6),
                "energy": round(energy_component * distance_penalty, 6),
                "chroma_delta": round(chroma_component * distance_penalty, 6),
                "f0_delta": round(f0_component * distance_penalty, 6),
            },
            "nearest_onset_distance": round(float(nearest_distance_scores[index]), 6),
            "distance_penalty": round(distance_penalty, 6),
            "near_onset_candidate": near_onset_candidate,
            "beat_type": beat_types[index],
        }
    return score_map


def _pick_beat_in_bucket_by_score(
    bucket_beat_rows: list[tuple[float, str]],
    beat_score_map: dict[float, dict[str, Any]],
) -> tuple[float, dict[str, Any], str]:
    """
    功能说明：在一个步长桶内挑选切分beat（优先综合分，平分看中心距离与时间）。
    参数说明：
    - bucket_beat_rows: 当前桶beat序列（按时间升序，含beat_type）。
    - beat_score_map: beat分数字典。
    返回值：
    - tuple[float, dict[str, Any], str]: (selected_beat_time, selected_meta, reason)。
    异常说明：无。
    边界条件：当桶内分数全零时回退桶尾beat。
    """
    default_meta = {
        "energy_raw": 0.0,
        "energy_norm": 0.0,
        "delta_raw": 0.0,
        "delta_norm": 0.0,
        "chroma_delta_raw": 0.0,
        "chroma_delta_norm": 0.0,
        "f0_delta_raw": 0.0,
        "f0_delta_norm": 0.0,
        "pitch_source": "fallback",
        "score": 0.0,
        "score_base": 0.0,
        "score_components": {"onset_delta": 0.0, "energy": 0.0, "chroma_delta": 0.0, "f0_delta": 0.0},
        "nearest_onset_distance": 999.0,
        "distance_penalty": 0.0,
        "near_onset_candidate": False,
        "beat_type": "unknown",
    }
    if not bucket_beat_rows:
        return 0.0, default_meta, "fallback_index"

    score_rows: list[tuple[float, float, dict[str, Any]]] = []
    for time_item, beat_type in bucket_beat_rows:
        time_key = _round_time(time_item)
        meta = dict(beat_score_map.get(time_key, default_meta))
        meta.setdefault("beat_type", beat_type)
        score_rows.append((float(time_item), float(_safe_float(meta.get("score", 0.0), 0.0)), meta))

    candidate_rows = [row for row in score_rows if bool(row[2].get("near_onset_candidate", False))]
    if not candidate_rows:
        return 0.0, default_meta, "fallback_index"

    if max(row[1] for row in candidate_rows) <= EPSILON_SECONDS:
        selected_time, _, selected_meta = max(candidate_rows, key=lambda row: row[0])
        return float(selected_time), dict(selected_meta), "fallback_index"

    bucket_center = (float(bucket_beat_rows[0][0]) + float(bucket_beat_rows[-1][0])) / 2.0
    selected_time, _, selected_meta = min(
        candidate_rows,
        key=lambda row: (
            -row[1],
            abs(row[0] - bucket_center),
            row[0],
        ),
    )
    return float(selected_time), dict(selected_meta), "energy_peak"

def split_long_other_windows_by_major(
    windows: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    bar_length_seconds: float,
    long_window_split_min_bars: float,
    major_split_step_bars: float,
    eligible_window_role_hints: set[str] | None = None,
    score_role: str = "other",
    onset_points: list[dict[str, Any]] | None = None,
    vocal_rms_times: list[float] | None = None,
    vocal_rms_values: list[float] | None = None,
    accompaniment_rms_times: list[float] | None = None,
    accompaniment_rms_values: list[float] | None = None,
    accompaniment_chroma_points: list[dict[str, Any]] | None = None,
    vocal_f0_points: list[dict[str, Any]] | None = None,
    accompaniment_f0_points: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：将长非歌词窗口按“固定步长时间窗 + role自适应多特征打分”滑动切分。
    参数说明：
    - windows: 已分类窗口列表。
    - beats: 节拍对象列表。
    - bar_length_seconds: 小节时长（秒）。
    - long_window_split_min_bars: 触发 downbeat 细分的时长阈值（小节）。
    - major_split_step_bars: downbeat 滑动桶步长（小节）。
    - eligible_window_role_hints: 允许参与重拍切分的 window_role_hint 集合；None 表示默认仅切 other。
    - score_role: 重拍打分使用的统一角色语义。
    - onset_points: onset强度点列表（time+energy_raw）。
    - vocal_rms_times/vocal_rms_values: 人声 RMS 序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏 RMS 序列。
    - accompaniment_chroma_points: 伴奏 chroma 点序列。
    - vocal_f0_points/accompaniment_f0_points: 两路 F0 点序列。
    返回值：
    - list[dict[str, Any]]: 切分后的窗口列表。
    异常说明：无。
    边界条件：仅处理指定 role_hint 且时长超过指定小节阈值的窗口。
    """
    safe_bar_seconds = max(0.2, float(bar_length_seconds))
    safe_long_window_split_min_bars = max(EPSILON_SECONDS, float(long_window_split_min_bars))
    safe_major_split_step_bars = max(EPSILON_SECONDS, float(major_split_step_bars))
    safe_score_role = str(score_role).lower().strip() or "other"
    safe_role_hints = {str(item).lower().strip() for item in (eligible_window_role_hints or {"other"}) if str(item).strip()}
    if not safe_role_hints:
        safe_role_hints = {"other"}
    beat_rows = _collect_beat_rows(beats=beats)
    if not beat_rows:
        return windows
    normalized_onset_points = _normalize_onset_points(onset_points=onset_points)
    normalized_chroma_points = _normalize_chroma_points(chroma_points=accompaniment_chroma_points)
    normalized_vocal_f0_points = _normalize_f0_points(f0_points=vocal_f0_points)
    normalized_accompaniment_f0_points = _normalize_f0_points(f0_points=accompaniment_f0_points)

    output: list[dict[str, Any]] = []
    for item in windows:
        role_hint = str(item.get("window_role_hint", "other")).lower().strip()
        if role_hint not in safe_role_hints:
            output.append(dict(item))
            continue
        role = safe_score_role

        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = _safe_float(item.get("end_time", start_time), start_time)
        duration = max(0.0, end_time - start_time)
        if duration <= safe_long_window_split_min_bars * safe_bar_seconds + EPSILON_SECONDS:
            output.append(dict(item))
            continue

        beat_inside_rows = [
            beat_row
            for beat_row in beat_rows
            if start_time + EPSILON_SECONDS < float(beat_row[0]) < end_time - EPSILON_SECONDS
            and str(beat_row[1]).lower().strip() == "major"
        ]
        if not beat_inside_rows:
            output.append(dict(item))
            continue

        step_bars = float(safe_major_split_step_bars)
        bucket_window_seconds = float(step_bars) * safe_bar_seconds
        if bucket_window_seconds <= EPSILON_SECONDS:
            output.append(dict(item))
            continue

        selected_boundaries: list[float] = []
        selected_boundary_meta: list[dict[str, Any]] = []
        beat_score_map = _compute_beat_feature_scores(
            beat_rows=beat_inside_rows,
            onset_points=normalized_onset_points,
            chroma_points=normalized_chroma_points,
            vocal_f0_points=normalized_vocal_f0_points,
            accompaniment_f0_points=normalized_accompaniment_f0_points,
            vocal_rms_times=list(vocal_rms_times or []),
            vocal_rms_values=list(vocal_rms_values or []),
            accompaniment_rms_times=list(accompaniment_rms_times or []),
            accompaniment_rms_values=list(accompaniment_rms_values or []),
            bar_length_seconds=safe_bar_seconds,
            role=role,
        )
        anchor_time = float(start_time)
        while anchor_time + bucket_window_seconds <= end_time - EPSILON_SECONDS:
            bucket_end_time = anchor_time + bucket_window_seconds
            bucket_beat_rows = [
                beat_row
                for beat_row in beat_inside_rows
                if anchor_time + EPSILON_SECONDS < float(beat_row[0]) <= bucket_end_time + EPSILON_SECONDS
            ]
            if not bucket_beat_rows:
                break
            selected_beat_time, selected_meta, pick_reason = _pick_beat_in_bucket_by_score(
                bucket_beat_rows=bucket_beat_rows,
                beat_score_map=beat_score_map,
            )
            if selected_beat_time <= anchor_time + EPSILON_SECONDS:
                break
            selected_beat_time = max(anchor_time + EPSILON_SECONDS, float(selected_beat_time))
            if selected_beat_time >= end_time - EPSILON_SECONDS:
                break
            selected_boundaries.append(_round_time(selected_beat_time))
            selected_energy_raw = _safe_float(selected_meta.get("energy_raw", 0.0), 0.0)
            selected_energy_norm = _safe_float(selected_meta.get("energy_norm", 0.0), 0.0)
            selected_delta_raw = _safe_float(selected_meta.get("delta_raw", 0.0), 0.0)
            selected_delta_norm = _safe_float(selected_meta.get("delta_norm", 0.0), 0.0)
            selected_chroma_delta_raw = _safe_float(selected_meta.get("chroma_delta_raw", 0.0), 0.0)
            selected_chroma_delta_norm = _safe_float(selected_meta.get("chroma_delta_norm", 0.0), 0.0)
            selected_f0_delta_raw = _safe_float(selected_meta.get("f0_delta_raw", 0.0), 0.0)
            selected_f0_delta_norm = _safe_float(selected_meta.get("f0_delta_norm", 0.0), 0.0)
            selected_pitch_source = str(selected_meta.get("pitch_source", "fallback"))
            selected_score = _safe_float(selected_meta.get("score", 0.0), 0.0)
            selected_score_base = _safe_float(selected_meta.get("score_base", 0.0), 0.0)
            selected_onset_distance = _safe_float(selected_meta.get("nearest_onset_distance", 999.0), 999.0)
            selected_distance_penalty = _safe_float(selected_meta.get("distance_penalty", 0.0), 0.0)
            selected_score_components = selected_meta.get("score_components", {})
            selected_beat_type = str(selected_meta.get("beat_type", "unknown"))
            selected_boundary_meta.append(
                {
                    "split_major_energy_raw": round(float(selected_energy_raw), 6),
                    "split_major_energy_norm": round(float(selected_energy_norm), 6),
                    "split_major_pick_reason": str(pick_reason),
                    "split_pick_beat_type": selected_beat_type,
                    "split_beat_energy_raw": round(float(selected_energy_raw), 6),
                    "split_beat_energy_norm": round(float(selected_energy_norm), 6),
                    "split_beat_delta_raw": round(float(selected_delta_raw), 6),
                    "split_beat_delta_norm": round(float(selected_delta_norm), 6),
                    "split_beat_chroma_delta_raw": round(float(selected_chroma_delta_raw), 6),
                    "split_beat_chroma_delta_norm": round(float(selected_chroma_delta_norm), 6),
                    "split_beat_f0_delta_raw": round(float(selected_f0_delta_raw), 6),
                    "split_beat_f0_delta_norm": round(float(selected_f0_delta_norm), 6),
                    "split_pitch_source": selected_pitch_source,
                    "split_beat_score_base": round(float(selected_score_base), 6),
                    "split_beat_score": round(float(selected_score), 6),
                    "split_beat_score_components": {
                        "onset_delta": round(_safe_float(selected_score_components.get("onset_delta", 0.0), 0.0), 6),
                        "energy": round(_safe_float(selected_score_components.get("energy", 0.0), 0.0), 6),
                        "chroma_delta": round(_safe_float(selected_score_components.get("chroma_delta", 0.0), 0.0), 6),
                        "f0_delta": round(_safe_float(selected_score_components.get("f0_delta", 0.0), 0.0), 6),
                    },
                    "split_beat_onset_distance": round(float(selected_onset_distance), 6),
                    "split_beat_distance_penalty": round(float(selected_distance_penalty), 6),
                }
            )
            anchor_time = float(selected_beat_time)

        if not selected_boundaries:
            output.append(dict(item))
            continue

        boundary_points = [start_time, *selected_boundaries, end_time]
        source_window_id = str(item.get("window_id", ""))
        for split_index in range(len(boundary_points) - 1):
            split_start = _round_time(boundary_points[split_index])
            split_end = _round_time(max(split_start, boundary_points[split_index + 1]))
            if split_end - split_start <= EPSILON_SECONDS:
                continue
            split_item = dict(item)
            split_item["window_id"] = f"{source_window_id}_sp{split_index + 1:02d}"
            split_item["start_time"] = split_start
            split_item["end_time"] = split_end
            split_item["duration"] = _round_time(split_end - split_start)
            split_item["split_source_window_id"] = source_window_id
            split_item["split_step_bars"] = round(float(step_bars), 3)
            split_item["split_basis"] = "major"
            split_item["window_type"] = f"{str(item.get('window_type', 'window'))}_major_split"
            split_item["merge_action"] = "split_by_major"
            split_item["source_window_ids"] = [source_window_id] if source_window_id else []
            if selected_boundary_meta:
                meta_index = min(split_index, len(selected_boundary_meta) - 1)
                split_item.update(selected_boundary_meta[meta_index])
            output.append(split_item)
    return output


def _pick_by_gap(
    left_index: int,
    right_index: int,
    left_gap_seconds: float,
    right_gap_seconds: float,
    left_reason: str,
    right_reason: str,
    tie_reason: str,
) -> tuple[int, str]:
    """
    功能说明：按源窗口到左右邻居的边界间隔选择并段目标。
    参数说明：
    - left_index/right_index: 左右邻居索引。
    - left_gap_seconds/right_gap_seconds: 源窗口到左右邻居的边界间隔（秒）。
    - left_reason/right_reason/tie_reason: 左选中/右选中/平局时原因标记。
    返回值：
    - tuple[int, str]: (目标索引, 决策原因)。
    异常说明：无。
    边界条件：平局时固定选左，保证可复现。
    """
    safe_left_gap = max(0.0, float(left_gap_seconds))
    safe_right_gap = max(0.0, float(right_gap_seconds))
    if safe_left_gap + EPSILON_SECONDS < safe_right_gap:
        return left_index, left_reason
    if safe_right_gap + EPSILON_SECONDS < safe_left_gap:
        return right_index, right_reason
    return left_index, tie_reason


def _get_tiny_source_role_priority(role: str) -> int:
    """
    功能说明：返回 tiny 源窗口处理优先级。
    参数说明：
    - role: 源窗口角色。
    返回值：
    - int: 角色优先级，数值越小越先处理。
    异常说明：无。
    边界条件：未知角色按最低优先级处理。
    """
    return int(TINY_SOURCE_ROLE_PRIORITY.get(str(role).lower().strip(), 99))


def _pick_merge_target_index(windows: list[dict[str, Any]], source_index: int) -> tuple[int | None, str]:
    """
    功能说明：根据旧规则选择被吸收段的目标邻居（保留未接线）。
    参数说明：
    - windows: 当前窗口列表。
    - source_index: 待吸收窗口索引。
    返回值：
    - tuple[int | None, str]: (目标索引, 决策原因)。
    异常说明：无。
    边界条件：两侧都不存在时返回 None。
    """
    left_index = source_index - 1 if source_index - 1 >= 0 else None
    right_index = source_index + 1 if source_index + 1 < len(windows) else None

    if left_index is None and right_index is None:
        return None, "no_neighbor"
    if left_index is None:
        return right_index, "edge_right_only"
    if right_index is None:
        return left_index, "edge_left_only"

    source_start = _safe_float(windows[source_index].get("start_time", 0.0), 0.0)
    source_end = _safe_float(windows[source_index].get("end_time", source_start), source_start)
    left_end = _safe_float(windows[left_index].get("end_time", source_start), source_start)
    right_start = _safe_float(windows[right_index].get("start_time", source_end), source_end)
    left_gap_seconds = max(0.0, source_start - left_end)
    right_gap_seconds = max(0.0, right_start - source_end)
    source_role = str(windows[source_index].get("role", "silence")).lower().strip()
    left_role = str(windows[left_index].get("role", "silence")).lower().strip()
    right_role = str(windows[right_index].get("role", "silence")).lower().strip()

    # 规则1：微静音段默认并左；只有“右静音且左非静音”时改并右。
    if source_role == "silence":
        if right_role == "silence" and left_role != "silence":
            return right_index, "tiny_silence_follow_right_silence"
        return left_index, "tiny_silence_default_left"

    # 规则2：微吟唱段默认并左，优先保持人声侧的听感连续。
    if source_role == "chant":
        return left_index, "tiny_chant_default_left"

    # 规则3：微器乐段默认并左；只有“右侧是 inst 且左侧是 lyric/chant”时并右。
    if source_role == "inst":
        if right_role == "inst" and left_role in {"lyric", "chant"}:
            return right_index, "tiny_inst_follow_right_inst"
        return left_index, "tiny_inst_default_left"

    # 规则4：微歌词段按歌词连续性处理；两侧都为歌词时按更短 gap 选择。
    if left_role == "lyric" and right_role == "lyric":
        return _pick_by_gap(
            left_index=left_index,
            right_index=right_index,
            left_gap_seconds=left_gap_seconds,
            right_gap_seconds=right_gap_seconds,
            left_reason="both_lyric_shorter_gap_left",
            right_reason="both_lyric_shorter_gap_right",
            tie_reason="both_lyric_equal_gap_left",
        )

    if left_role == "lyric":
        return left_index, "neighbor_lyric_left"
    if right_role == "lyric":
        return right_index, "neighbor_lyric_right"
    return left_index, "tiny_default_left"


def _pick_merge_target_index_by_similarity(
    windows: list[dict[str, Any]],
    source_index: int,
    onset_points: list[dict[str, float]],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    chroma_points: list[dict[str, Any]],
    f0_points: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any]]:
    """
    功能说明：按左右相似度选择 tiny 并段目标。
    参数说明：
    - windows: 当前窗口列表。
    - source_index: 待吸收窗口索引。
    - onset_points/accompaniment_rms_times/accompaniment_rms_values/chroma_points/f0_points: 相似度特征输入。
    返回值：
    - tuple[int | None, dict[str, Any]]: (目标索引, 决策元数据)。
    异常说明：无。
    边界条件：平分固定选左；边界窗直接选唯一邻居。
    """
    left_index = source_index - 1 if source_index - 1 >= 0 else None
    right_index = source_index + 1 if source_index + 1 < len(windows) else None

    if left_index is None and right_index is None:
        return None, {"reason": "no_neighbor", "decision_strategy": "similarity"}
    if left_index is None:
        return right_index, {
            "reason": "edge_right_only",
            "decision_strategy": "similarity",
            "selected_side": "right",
            "tie_break_applied": False,
        }
    if right_index is None:
        return left_index, {
            "reason": "edge_left_only",
            "decision_strategy": "similarity",
            "selected_side": "left",
            "tie_break_applied": False,
        }

    source_summary = _build_tiny_window_summary(
        window_item=windows[source_index],
        onset_points=onset_points,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
        chroma_points=chroma_points,
        f0_points=f0_points,
    )
    left_summary = _build_tiny_window_summary(
        window_item=windows[left_index],
        onset_points=onset_points,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
        chroma_points=chroma_points,
        f0_points=f0_points,
    )
    right_summary = _build_tiny_window_summary(
        window_item=windows[right_index],
        onset_points=onset_points,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
        chroma_points=chroma_points,
        f0_points=f0_points,
    )
    left_components = _compute_tiny_similarity_components(source_summary=source_summary, neighbor_summary=left_summary)
    right_components = _compute_tiny_similarity_components(source_summary=source_summary, neighbor_summary=right_summary)
    left_total = _safe_float(left_components.get("total_similarity", 0.0), 0.0)
    right_total = _safe_float(right_components.get("total_similarity", 0.0), 0.0)
    if left_total > right_total + EPSILON_SECONDS:
        return left_index, {
            "reason": "similarity_left_higher",
            "decision_strategy": "similarity",
            "selected_side": "left",
            "tie_break_applied": False,
            "source_summary": source_summary,
            "left_summary": left_summary,
            "right_summary": right_summary,
            "left_similarity_total": round(float(left_total), 6),
            "right_similarity_total": round(float(right_total), 6),
            "left_similarity_components": left_components,
            "right_similarity_components": right_components,
        }
    if right_total > left_total + EPSILON_SECONDS:
        return right_index, {
            "reason": "similarity_right_higher",
            "decision_strategy": "similarity",
            "selected_side": "right",
            "tie_break_applied": False,
            "source_summary": source_summary,
            "left_summary": left_summary,
            "right_summary": right_summary,
            "left_similarity_total": round(float(left_total), 6),
            "right_similarity_total": round(float(right_total), 6),
            "left_similarity_components": left_components,
            "right_similarity_components": right_components,
        }
    return left_index, {
        "reason": "similarity_tie_left",
        "decision_strategy": "similarity",
        "selected_side": "left",
        "tie_break_applied": True,
        "source_summary": source_summary,
        "left_summary": left_summary,
        "right_summary": right_summary,
        "left_similarity_total": round(float(left_total), 6),
        "right_similarity_total": round(float(right_total), 6),
        "left_similarity_components": left_components,
        "right_similarity_components": right_components,
    }


def _merge_one_window(
    windows: list[dict[str, Any]],
    source_index: int,
    target_index: int,
    reason: str,
    merge_kind: str,
    decision_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：执行单次窗口吸收操作。
    参数说明：
    - windows: 当前窗口列表。
    - source_index: 被吸收窗口索引。
    - target_index: 吸收目标窗口索引。
    - reason: 决策原因。
    - merge_kind: 并段类型（tiny/bar）。
    - decision_meta: 额外决策解释字段。
    返回值：
    - tuple[list[dict[str, Any]], dict[str, Any]]: (新窗口列表, 并段事件)。
    异常说明：无。
    边界条件：自动处理向左/向右并段边界。
    """
    source_item = dict(windows[source_index])
    target_item = dict(windows[target_index])

    source_start = _safe_float(source_item.get("start_time", 0.0), 0.0)
    source_end = _safe_float(source_item.get("end_time", source_start), source_start)
    target_start = _safe_float(target_item.get("start_time", 0.0), 0.0)
    target_end = _safe_float(target_item.get("end_time", target_start), target_start)

    direction = "to_left" if target_index < source_index else "to_right"
    if direction == "to_left":
        target_item["end_time"] = _round_time(max(target_end, source_end))
        target_item["merge_action"] = f"absorb_{merge_kind}_{reason}"
        target_item["source_window_ids"] = list(target_item.get("source_window_ids", [target_item.get("window_id", "")])) + list(
            source_item.get("source_window_ids", [source_item.get("window_id", "")])
        )
        windows[target_index] = target_item
        windows.pop(source_index)
    else:
        target_item["start_time"] = _round_time(min(target_start, source_start))
        target_item["merge_action"] = f"absorb_{merge_kind}_{reason}"
        target_item["source_window_ids"] = list(source_item.get("source_window_ids", [source_item.get("window_id", "")])) + list(
            target_item.get("source_window_ids", [target_item.get("window_id", "")])
        )
        windows[target_index] = target_item
        windows.pop(source_index)

    event = {
        "merge_kind": merge_kind,
        "reason": reason,
        "direction": direction,
        "source_window_id": str(source_item.get("window_id", "")),
        "source_role": str(source_item.get("role", "silence")),
        "source_start_time": _round_time(source_start),
        "source_end_time": _round_time(source_end),
        "target_window_id": str(target_item.get("window_id", "")),
        "target_role": str(target_item.get("role", "silence")),
    }
    if decision_meta:
        for key, value in decision_meta.items():
            event[key] = value
    return windows, event


def _normalize_windows_continuity(windows: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：修复并段后窗口连续性。
    参数说明：
    - windows: 窗口列表。
    - duration_seconds: 音频总时长。
    返回值：
    - list[dict[str, Any]]: 连续窗口列表。
    异常说明：无。
    边界条件：首段起点固定0，末段终点固定总时长。
    """
    sorted_windows = sorted(windows, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))
    if not sorted_windows:
        return []

    sorted_windows[0]["start_time"] = 0.0
    for index in range(1, len(sorted_windows)):
        sorted_windows[index]["start_time"] = _round_time(_safe_float(sorted_windows[index - 1].get("end_time", 0.0), 0.0))
        if _safe_float(sorted_windows[index].get("end_time", 0.0), 0.0) < _safe_float(sorted_windows[index].get("start_time", 0.0), 0.0):
            sorted_windows[index]["end_time"] = sorted_windows[index]["start_time"]
    sorted_windows[-1]["end_time"] = _round_time(max(0.0, float(duration_seconds)))

    for item in sorted_windows:
        item["duration"] = _round_time(_window_duration(item))
    return sorted_windows


def estimate_bar_length_seconds(beats: list[dict[str, Any]], beat_candidates: list[float]) -> float:
    """
    功能说明：按 beats 动态估计一小节时长。
    参数说明：
    - beats: 节拍对象列表。
    - beat_candidates: 节拍时间候选列表。
    返回值：
    - float: 小节时长（秒）。
    异常说明：无。
    边界条件：无 major 时回退4拍估计；极端情况下回退2秒。
    """
    major_times = sorted(
        {
            _safe_float(item.get("time", 0.0), 0.0)
            for item in beats
            if str(item.get("type", "")).lower().strip() == "major"
        }
    )
    if len(major_times) >= 2:
        major_diffs = [
            major_times[index + 1] - major_times[index]
            for index in range(len(major_times) - 1)
            if major_times[index + 1] - major_times[index] > EPSILON_SECONDS
        ]
        if major_diffs:
            return max(0.2, float(statistics.median(major_diffs)))

    beat_times = sorted({_safe_float(item, 0.0) for item in beat_candidates if _safe_float(item, 0.0) >= 0.0})
    if len(beat_times) < 2:
        beat_times = sorted({_safe_float(item.get("time", 0.0), 0.0) for item in beats if _safe_float(item.get("time", 0.0), 0.0) >= 0.0})

    if len(beat_times) >= 2:
        beat_diffs = [
            beat_times[index + 1] - beat_times[index]
            for index in range(len(beat_times) - 1)
            if beat_times[index + 1] - beat_times[index] > EPSILON_SECONDS
        ]
        if beat_diffs:
            beat_interval = float(statistics.median(beat_diffs))
            return max(0.2, beat_interval * 4.0)

    return 2.0


def merge_windows_by_rules(
    windows_classified: list[dict[str, Any]],
    tiny_merge_bars: float,
    bar_length_seconds: float,
    beats: list[dict[str, Any]],
    duration_seconds: float,
    long_window_split_min_bars: float = 1.0,
    major_split_step_bars: float = 2.5,
    onset_points: list[dict[str, Any]] | None = None,
    vocal_rms_times: list[float] | None = None,
    vocal_rms_values: list[float] | None = None,
    accompaniment_rms_times: list[float] | None = None,
    accompaniment_rms_values: list[float] | None = None,
    accompaniment_chroma_points: list[dict[str, Any]] | None = None,
    vocal_f0_points: list[dict[str, Any]] | None = None,
    accompaniment_f0_points: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    功能说明：执行按小节阈值的 tiny 并段。
    参数说明：
    - windows_classified: 已分类窗口。
    - tiny_merge_bars: tiny阈值（小节）。
    - bar_length_seconds: 一小节时长（秒）。
    - beats: 结构化节拍列表（当前仅为接口兼容保留）。
    - duration_seconds: 音频总时长（秒）。
    - long_window_split_min_bars/major_split_step_bars: 兼容旧接口保留，当前不参与逻辑。
    - onset_points/vocal_rms_times/vocal_rms_values: tiny 相似度所需的局部特征输入。
    - accompaniment_rms_times/accompaniment_rms_values: tiny 相似度的 energy 参考。
    - accompaniment_chroma_points/vocal_f0_points/accompaniment_f0_points: tiny 相似度的 chroma/F0 参考。
    返回值：
    - tuple[list[dict[str, Any]], list[dict[str, Any]]]: (并段后窗口, 并段事件)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：所有角色均可作为被吸收对象；长 other 切分需在本函数之前完成。
    """
    windows = [dict(item) for item in sorted(windows_classified, key=lambda row: _safe_float(row.get("start_time", 0.0), 0.0))]
    for item in windows:
        item.setdefault("source_window_ids", [str(item.get("window_id", ""))])
        item.setdefault("merge_action", "keep_original")

    merge_events: list[dict[str, Any]] = []
    safe_tiny_bars = _safe_float(tiny_merge_bars, DEFAULT_TINY_MERGE_BARS)
    if safe_tiny_bars <= 0.0:
        safe_tiny_bars = DEFAULT_TINY_MERGE_BARS
    safe_tiny_seconds = max(0.2, float(bar_length_seconds)) * safe_tiny_bars
    normalized_onset_points = _normalize_onset_points(onset_points=onset_points)
    normalized_chroma_points = _normalize_chroma_points(chroma_points=accompaniment_chroma_points)
    normalized_vocal_f0_points = _normalize_f0_points(f0_points=vocal_f0_points)
    normalized_accompaniment_f0_points = _normalize_f0_points(f0_points=accompaniment_f0_points)
    merged_f0_points = sorted(
        normalized_vocal_f0_points + normalized_accompaniment_f0_points,
        key=lambda item: _safe_float(item.get("time", 0.0), 0.0),
    )

    # 单阶段：按小节阈值执行tiny并段
    while True:
        tiny_candidates: list[tuple[int, int]] = []
        for index, item in enumerate(windows):
            if _window_duration(item) < safe_tiny_seconds:
                source_role = str(item.get("role", "silence")).lower().strip()
                tiny_candidates.append((_get_tiny_source_role_priority(source_role), index))
        tiny_index = None
        if tiny_candidates:
            tiny_candidates.sort(key=lambda item: (item[0], item[1]))
            tiny_index = int(tiny_candidates[0][1])
        if tiny_index is None:
            break

        target_index, decision_meta = _pick_merge_target_index_by_similarity(
            windows=windows,
            source_index=tiny_index,
            onset_points=normalized_onset_points,
            accompaniment_rms_times=list(accompaniment_rms_times or []),
            accompaniment_rms_values=list(accompaniment_rms_values or []),
            chroma_points=normalized_chroma_points,
            f0_points=merged_f0_points,
        )
        if target_index is None:
            break
        reason = str(decision_meta.get("reason", "similarity_tie_left"))
        windows, event = _merge_one_window(
            windows=windows,
            source_index=tiny_index,
            target_index=target_index,
            reason=reason,
            merge_kind="tiny",
            decision_meta=decision_meta,
        )
        merge_events.append(event)

    return _normalize_windows_continuity(windows=windows, duration_seconds=duration_seconds), merge_events
