"""
文件用途：执行窗口细分与并段（长窗口按节拍细分 + tiny合并）。
核心流程：先按节拍切分长非歌词窗口，再按小节阈值合并短窗口。
输入输出：输入已分类窗口与节拍，输出处理后窗口与并段事件。
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
# 常量：触发长窗口细分的最小时长（小节）
LONG_WINDOW_SPLIT_MIN_BARS = 2.0
# 常量：major切分滑动桶步长（小节）
MAJOR_SPLIT_STEP_BARS = 3
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
# 常量：角色优先级（数值越小优先级越高）
ROLE_PRIORITY = {
    "lyric": 1,
    "chant": 2,
    "inst": 3,
    "silence": 4,
}


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

def _split_long_other_windows_by_major(
    windows: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    bar_length_seconds: float,
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
    - onset_points: onset强度点列表（time+energy_raw）。
    - vocal_rms_times/vocal_rms_values: 人声 RMS 序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏 RMS 序列。
    - accompaniment_chroma_points: 伴奏 chroma 点序列。
    - vocal_f0_points/accompaniment_f0_points: 两路 F0 点序列。
    返回值：
    - list[dict[str, Any]]: 切分后的窗口列表。
    异常说明：无。
    边界条件：仅处理 role!=lyric 且时长超过2小节的窗口。
    """
    safe_bar_seconds = max(0.2, float(bar_length_seconds))
    beat_rows = _collect_beat_rows(beats=beats)
    if not beat_rows:
        return windows
    normalized_onset_points = _normalize_onset_points(onset_points=onset_points)
    normalized_chroma_points = _normalize_chroma_points(chroma_points=accompaniment_chroma_points)
    normalized_vocal_f0_points = _normalize_f0_points(f0_points=vocal_f0_points)
    normalized_accompaniment_f0_points = _normalize_f0_points(f0_points=accompaniment_f0_points)

    output: list[dict[str, Any]] = []
    for item in windows:
        role = str(item.get("role", "silence")).lower().strip()
        if role == "lyric":
            output.append(dict(item))
            continue

        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = _safe_float(item.get("end_time", start_time), start_time)
        duration = max(0.0, end_time - start_time)
        if duration <= LONG_WINDOW_SPLIT_MIN_BARS * safe_bar_seconds + EPSILON_SECONDS:
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

        step_bars = max(1, int(MAJOR_SPLIT_STEP_BARS))
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
            split_item["split_step_bars"] = step_bars
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


def _pick_merge_target_index(windows: list[dict[str, Any]], source_index: int) -> tuple[int | None, str]:
    """
    功能说明：根据并段层级选择被吸收段的目标邻居。
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
    left_role = str(windows[left_index].get("role", "silence")).lower().strip()
    right_role = str(windows[right_index].get("role", "silence")).lower().strip()

    # 规则1：左右都是 lyric 时，按“有效边界间隔更短”并入；平局固定并左。
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

    # 规则2：一侧 lyric 一侧其他，优先并入 lyric。
    if left_role == "lyric":
        return left_index, "neighbor_lyric_left"
    if right_role == "lyric":
        return right_index, "neighbor_lyric_right"

    # 规则3：无 lyric 且左右都是 chant，优先并右（偏向承接未识别尾音）。
    if left_role == "chant" and right_role == "chant":
        return right_index, "both_chant_prefer_right"

    # 规则4：无 lyric 且一侧 chant、一侧 inst/silence，优先并入 chant。
    if left_role == "chant" and right_role in {"inst", "silence"}:
        return left_index, "chant_vs_instsilence_left"
    if right_role == "chant" and left_role in {"inst", "silence"}:
        return right_index, "chant_vs_instsilence_right"

    # 规则5：无 lyric/chant 时优先并入 inst。
    if left_role == "inst" and right_role != "inst":
        return left_index, "inst_prefer_left"
    if right_role == "inst" and left_role != "inst":
        return right_index, "inst_prefer_right"

    if left_role == "inst" and right_role == "inst":
        return _pick_by_gap(
            left_index=left_index,
            right_index=right_index,
            left_gap_seconds=left_gap_seconds,
            right_gap_seconds=right_gap_seconds,
            left_reason="both_inst_shorter_gap_left",
            right_reason="both_inst_shorter_gap_right",
            tie_reason="both_inst_equal_gap_left",
        )
    if left_role == "silence" and right_role == "silence":
        return _pick_by_gap(
            left_index=left_index,
            right_index=right_index,
            left_gap_seconds=left_gap_seconds,
            right_gap_seconds=right_gap_seconds,
            left_reason="both_silence_shorter_gap_left",
            right_reason="both_silence_shorter_gap_right",
            tie_reason="both_silence_equal_gap_left",
        )

    if left_role == right_role:
        return left_index, "same_role_left"

    # 兜底：角色优先级 + 时长稳定决策，保证未知角色也可并段。
    left_priority = ROLE_PRIORITY.get(left_role, 99)
    right_priority = ROLE_PRIORITY.get(right_role, 99)
    if left_priority < right_priority:
        return left_index, "priority_left"
    if right_priority < left_priority:
        return right_index, "priority_right"

    left_duration = _window_duration(windows[left_index])
    right_duration = _window_duration(windows[right_index])
    if right_duration > left_duration + EPSILON_SECONDS:
        return right_index, "same_priority_longer_right"
    return left_index, "same_priority_longer_or_equal_left"


def _merge_one_window(
    windows: list[dict[str, Any]],
    source_index: int,
    target_index: int,
    reason: str,
    merge_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：执行单次窗口吸收操作。
    参数说明：
    - windows: 当前窗口列表。
    - source_index: 被吸收窗口索引。
    - target_index: 吸收目标窗口索引。
    - reason: 决策原因。
    - merge_kind: 并段类型（tiny/bar）。
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
    功能说明：执行“长窗口按重拍细分 + 按小节tiny并段”。
    参数说明：
    - windows_classified: 已分类窗口。
    - tiny_merge_bars: tiny阈值（小节）。
    - bar_length_seconds: 一小节时长（秒）。
    - beats: 结构化节拍列表（用于重拍切分）。
    - duration_seconds: 音频总时长（秒）。
    - onset_points: onset强度点列表（time+energy_raw）。
    - vocal_rms_times/vocal_rms_values: 人声 RMS 序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏 RMS 序列。
    - accompaniment_chroma_points: 伴奏 chroma 点序列。
    - vocal_f0_points/accompaniment_f0_points: 人声/伴奏 F0 点序列。
    返回值：
    - tuple[list[dict[str, Any]], list[dict[str, Any]]]: (并段后窗口, 并段事件)。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：所有角色均可作为被吸收对象，最终连续性由归一化步骤保证。
    """
    windows = [dict(item) for item in sorted(windows_classified, key=lambda row: _safe_float(row.get("start_time", 0.0), 0.0))]
    windows = _split_long_other_windows_by_major(
        windows=windows,
        beats=beats,
        bar_length_seconds=bar_length_seconds,
        onset_points=onset_points,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
        accompaniment_chroma_points=accompaniment_chroma_points,
        vocal_f0_points=vocal_f0_points,
        accompaniment_f0_points=accompaniment_f0_points,
    )
    for item in windows:
        item.setdefault("source_window_ids", [str(item.get("window_id", ""))])
        item.setdefault("merge_action", "keep_original")

    merge_events: list[dict[str, Any]] = []
    safe_tiny_bars = _safe_float(tiny_merge_bars, DEFAULT_TINY_MERGE_BARS)
    if safe_tiny_bars <= 0.0:
        safe_tiny_bars = DEFAULT_TINY_MERGE_BARS
    safe_tiny_seconds = max(0.2, float(bar_length_seconds)) * safe_tiny_bars

    # 单阶段：按小节阈值执行tiny并段
    while True:
        tiny_index = None
        for index, item in enumerate(windows):
            if _window_duration(item) <= safe_tiny_seconds + EPSILON_SECONDS:
                tiny_index = index
                break
        if tiny_index is None:
            break

        target_index, reason = _pick_merge_target_index(windows=windows, source_index=tiny_index)
        if target_index is None:
            break
        windows, event = _merge_one_window(
            windows=windows,
            source_index=tiny_index,
            target_index=target_index,
            reason=reason,
            merge_kind="tiny",
        )
        merge_events.append(event)

    return _normalize_windows_continuity(windows=windows, duration_seconds=duration_seconds), merge_events
