"""
文件用途：构建模块A V2时间窗口（歌词句窗口 + 其他窗口）。
核心流程：句级歌词归一化 -> 句间阈值判定 -> 生成覆盖全时长窗口。
输入输出：输入句级歌词与动态阈值，输出窗口列表。
依赖说明：仅依赖标准库与本文件基础工具函数。
维护说明：本文件只负责窗口构造，不承担角色分类与边界矫正。
"""

# 标准库：用于数学计算
import math
# 标准库：用于正则匹配
import re
# 标准库：用于类型提示
from typing import Any


# 常量：浮点比较容差
EPSILON_SECONDS = 1e-6


# 常量：句间分窗阈值默认值（秒），回退 funasr_lyrics 默认阈值
DEFAULT_DYNAMIC_GAP_SECONDS = 0.35
# 常量：长歌词窗口触发重切阈值（小节）
LONG_LYRIC_RESPLIT_MAX_BARS = 3.0
# 常量：动态阈值下限（秒），防止误切到极小间隔
MIN_DYNAMIC_GAP_SECONDS = 0.04
# 常量：MAD 离群阈值（Modified Z-Score）
GAP_OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
# 常量：MAD 数值稳定阈值
MAD_EPSILON = 1e-6
# 常量：单窗口重切最大迭代次数，避免极端数据死循环
MAX_LONG_LYRIC_RESPLIT_ITERATIONS = 64
# 常量：标点识别模式（中英文常见标点）
PUNCTUATION_PATTERN = re.compile(r"[，、；：。！？!?,.;:]")


def _is_effective_lyric_token_text(text: str) -> bool:
    """
    功能说明：判断 token 文本是否属于“有效发音内容”（非纯标点/空白）。
    参数说明：
    - text: token 文本。
    返回值：
    - bool: 有效发音返回 True。
    异常说明：无。
    边界条件：中日韩字符、字母数字均视为有效内容。
    """
    token_text = str(text).strip()
    if not token_text:
        return False
    for char in token_text:
        if char.isalnum():
            return True
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3040" <= char <= "\u30ff":
            return True
        if "\uac00" <= char <= "\ud7af":
            return True
    return False


def _extract_effective_token_bounds(token_units: list[dict[str, Any]]) -> tuple[float, float] | None:
    """
    功能说明：提取句内“有效发音 token”首尾时间边界。
    参数说明：
    - token_units: 规范化 token 列表。
    返回值：
    - tuple[float, float] | None: (有效起点, 有效终点)；无有效token返回 None。
    异常说明：无。
    边界条件：仅依据 token 文本有效性筛选，不改变 token 原序。
    """
    effective_tokens: list[dict[str, Any]] = []
    for item in token_units:
        if _is_effective_lyric_token_text(str(item.get("text", ""))):
            effective_tokens.append(item)
    if not effective_tokens:
        return None
    start_time = _safe_float(effective_tokens[0].get("start_time", 0.0), 0.0)
    end_time = _safe_float(effective_tokens[-1].get("end_time", start_time), start_time)
    return _round_time(start_time), _round_time(max(start_time, end_time))


def _extract_sentence_tail_end_time(token_units: list[dict[str, Any]]) -> float | None:
    """
    功能说明：提取句尾边界（优先句尾 token 的右边界，含句尾标点）。
    参数说明：
    - token_units: 规范化 token 列表。
    返回值：
    - float | None: 句尾右边界；无 token 返回 None。
    异常说明：无。
    边界条件：句尾标点与句尾歌词均可作为句尾锚点。
    """
    if not token_units:
        return None
    tail_end = _safe_float(token_units[-1].get("end_time", 0.0), 0.0)
    return _round_time(max(0.0, tail_end))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：将任意值安全转换为浮点数。
    参数说明：
    - value: 待转换对象。
    - default: 转换失败时的默认值。
    返回值：
    - float: 有效浮点数。
    异常说明：异常在函数内部吞并。
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
    功能说明：统一时间精度到毫秒级。
    参数说明：
    - value: 原始秒数。
    返回值：
    - float: 四舍五入后的秒数。
    异常说明：无。
    边界条件：负值由调用方裁剪。
    """
    return round(float(value), 6)


def _clamp_time(value: float, duration_seconds: float) -> float:
    """
    功能说明：将时间裁剪到 [0, duration_seconds] 区间。
    参数说明：
    - value: 原始时间。
    - duration_seconds: 总时长。
    返回值：
    - float: 裁剪后时间。
    异常说明：无。
    边界条件：duration_seconds 小于0时按0处理。
    """
    safe_duration = max(0.0, float(duration_seconds))
    return max(0.0, min(float(value), safe_duration))


def _normalize_token_units(token_units: Any) -> list[dict[str, Any]]:
    """
    功能说明：规范化 token 列表并按起点排序。
    参数说明：
    - token_units: 原始 token 列表。
    返回值：
    - list[dict[str, Any]]: 规范化 token。
    异常说明：无。
    边界条件：无效输入返回空列表。
    """
    if not isinstance(token_units, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in token_units:
        if not isinstance(item, dict):
            continue
        start_time = _safe_float(item.get("start_time", 0.0), 0.0)
        end_time = max(start_time, _safe_float(item.get("end_time", start_time), start_time))
        normalized.append(
            {
                "text": str(item.get("text", "")),
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
            }
        )
    return sorted(normalized, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))


def normalize_sentence_units(
    sentence_units: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：规范化句级歌词并消除重叠，确保时间单调。
    参数说明：
    - sentence_units: 原始句级歌词。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 规范化句级歌词。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：零时长句会被过滤。
    """
    safe_duration = max(0.0, float(duration_seconds))
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(list(sentence_units)):
        if not isinstance(item, dict):
            continue
        start_time = _clamp_time(_safe_float(item.get("start_time", 0.0), 0.0), safe_duration)
        end_time = _clamp_time(_safe_float(item.get("end_time", start_time), start_time), safe_duration)
        token_units = _normalize_token_units(item.get("token_units", []))
        effective_bounds = _extract_effective_token_bounds(token_units)
        if effective_bounds is not None:
            effective_start, _effective_end = effective_bounds
            start_time = max(start_time, _clamp_time(effective_start, safe_duration))
        sentence_tail_end = _extract_sentence_tail_end_time(token_units)
        if sentence_tail_end is not None:
            end_time = min(end_time, _clamp_time(sentence_tail_end, safe_duration))
        end_time = max(start_time, end_time)
        if end_time - start_time <= EPSILON_SECONDS:
            continue
        normalized.append(
            {
                "sentence_index": int(item.get("source_sentence_index", index)),
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "text": str(item.get("text", "")),
                "confidence": _safe_float(item.get("confidence", 0.0), 0.0),
                "token_units": token_units,
            }
        )

    normalized.sort(key=lambda unit: _safe_float(unit.get("start_time", 0.0), 0.0))
    if not normalized:
        return []

    deoverlap: list[dict[str, Any]] = []
    for item in normalized:
        if not deoverlap:
            deoverlap.append(item)
            continue
        previous_end = _safe_float(deoverlap[-1].get("end_time", 0.0), 0.0)
        current_start = max(previous_end, _safe_float(item.get("start_time", 0.0), 0.0))
        current_end = max(current_start, _safe_float(item.get("end_time", current_start), current_start))
        if current_end - current_start <= EPSILON_SECONDS:
            continue
        rewritten = dict(item)
        rewritten["start_time"] = _round_time(current_start)
        rewritten["end_time"] = _round_time(current_end)
        deoverlap.append(rewritten)
    return deoverlap


def _build_window_item(
    window_id: str,
    start_time: float,
    end_time: float,
    window_role_hint: str,
    window_type: str,
    source_sentence_index: int = -1,
) -> dict[str, Any]:
    """
    功能说明：构造统一窗口对象。
    参数说明：
    - window_id: 窗口ID。
    - start_time/end_time: 窗口起止秒。
    - window_role_hint: 窗口角色提示（lyric/other）。
    - window_type: 窗口类型（lyric_sentence/other_leading/...）。
    - source_sentence_index: 来源句索引。
    返回值：
    - dict[str, Any]: 标准窗口对象。
    异常说明：无。
    边界条件：duration 自动计算。
    """
    safe_start = _round_time(max(0.0, float(start_time)))
    safe_end = _round_time(max(safe_start, float(end_time)))
    return {
        "window_id": window_id,
        "start_time": safe_start,
        "end_time": safe_end,
        "duration": _round_time(max(0.0, safe_end - safe_start)),
        "window_role_hint": str(window_role_hint),
        "window_type": str(window_type),
        "source_sentence_index": int(source_sentence_index),
    }


def _normalize_windows_continuity(
    windows: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：修复窗口连续性并重排 window_id。
    参数说明：
    - windows: 窗口列表。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[dict[str, Any]]: 连续且重排后的窗口列表。
    异常说明：无。
    边界条件：首段起点固定0，末段终点固定duration。
    """
    if not windows:
        return []

    safe_duration = max(0.0, float(duration_seconds))
    sorted_windows = sorted(windows, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))

    for index, item in enumerate(sorted_windows):
        current = dict(item)
        if index == 0:
            current_start = 0.0
        else:
            current_start = _safe_float(sorted_windows[index - 1].get("end_time", 0.0), 0.0)
        current_end = _safe_float(current.get("end_time", current_start), current_start)
        current_end = max(current_start, current_end)
        current["window_id"] = f"win_{index + 1:04d}"
        current["start_time"] = _round_time(current_start)
        current["end_time"] = _round_time(current_end)
        current["duration"] = _round_time(max(0.0, current_end - current_start))
        sorted_windows[index] = current

    sorted_windows[-1]["end_time"] = _round_time(safe_duration)
    sorted_windows[-1]["duration"] = _round_time(
        max(0.0, safe_duration - _safe_float(sorted_windows[-1].get("start_time", 0.0), 0.0))
    )
    return sorted_windows


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


def _median(values: list[float]) -> float:
    """
    功能说明：计算中位数。
    参数说明：
    - values: 数值列表。
    返回值：
    - float: 中位数。
    异常说明：无。
    边界条件：空列表返回0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    count = len(sorted_values)
    middle = count // 2
    if count % 2 == 1:
        return sorted_values[middle]
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


def _filter_high_outliers_by_log_mad(samples: list[float]) -> tuple[list[float], list[float]]:
    """
    功能说明：使用 log1p + MAD 剔除超长间隔离群值。
    参数说明：
    - samples: 原始 gap 样本。
    返回值：
    - tuple[list[float], list[float]]: (保留样本, 离群样本)。
    异常说明：无。
    边界条件：样本过少或 MAD 接近0时不剔除。
    """
    safe_samples = [max(0.0, float(item)) for item in samples if float(item) > 0.0]
    if len(safe_samples) < 4:
        return safe_samples, []

    transformed = [math.log1p(item) for item in safe_samples]
    median_value = _median(transformed)
    mad_value = _median([abs(item - median_value) for item in transformed])
    if mad_value <= MAD_EPSILON:
        return safe_samples, []

    kept_samples: list[float] = []
    outlier_samples: list[float] = []
    for raw_sample, transformed_sample in zip(safe_samples, transformed):
        modified_z_score = 0.6745 * (transformed_sample - median_value) / mad_value
        if modified_z_score > GAP_OUTLIER_MODIFIED_Z_THRESHOLD:
            outlier_samples.append(raw_sample)
        else:
            kept_samples.append(raw_sample)
    return kept_samples, outlier_samples


def _is_punctuation_only_token_text(text: str) -> bool:
    """
    功能说明：判断 token 是否仅包含标点与空白。
    参数说明：
    - text: token 文本。
    返回值：
    - bool: 仅标点/空白返回 True。
    异常说明：无。
    边界条件：空文本返回 False。
    """
    token_text = str(text).strip()
    if not token_text:
        return False
    return bool(PUNCTUATION_PATTERN.search(token_text)) and not _is_effective_lyric_token_text(token_text)


def _find_neighbor_content_token_index(
    token_units: list[dict[str, Any]],
    center_index: int,
    direction: int,
    segment_start: float,
    segment_end: float,
) -> int | None:
    """
    功能说明：在窗口内向左/右查找最近有效内容 token。
    参数说明：
    - token_units: token 列表。
    - center_index: 中心索引。
    - direction: 查找方向（-1左/+1右）。
    - segment_start/segment_end: 目标窗口边界。
    返回值：
    - int | None: 命中索引。
    异常说明：无。
    边界条件：越界或无内容 token 时返回 None。
    """
    current_index = center_index + direction
    while 0 <= current_index < len(token_units):
        token_item = token_units[current_index]
        token_text = str(token_item.get("text", ""))
        if not _is_effective_lyric_token_text(token_text):
            current_index += direction
            continue
        token_start = _safe_float(token_item.get("start_time", 0.0), 0.0)
        token_end = _safe_float(token_item.get("end_time", token_start), token_start)
        if token_start < segment_start - EPSILON_SECONDS or token_end > segment_end + EPSILON_SECONDS:
            current_index += direction
            continue
        return current_index
    return None


def _compute_local_dynamic_gap_threshold(samples: list[float], fallback_value: float) -> tuple[float, dict[str, Any]]:
    """
    功能说明：根据窗口内样本计算局部动态 gap 阈值。
    参数说明：
    - samples: gap 样本列表。
    - fallback_value: 样本不足时回退阈值。
    返回值：
    - tuple[float, dict[str, Any]]: (阈值, 统计信息)。
    异常说明：无。
    边界条件：有效样本为空时回退 fallback。
    """
    safe_fallback = max(MIN_DYNAMIC_GAP_SECONDS, float(fallback_value))
    kept_samples, outlier_samples = _filter_high_outliers_by_log_mad(samples)
    effective_samples = kept_samples if kept_samples else [max(0.0, float(item)) for item in samples if float(item) > 0.0]
    if not effective_samples:
        return safe_fallback, {
            "sample_count_raw": len(samples),
            "sample_count_kept": 0,
            "sample_count_outlier": len(outlier_samples),
        }
    mean_gap = sum(effective_samples) / max(1, len(effective_samples))
    threshold = max(MIN_DYNAMIC_GAP_SECONDS, float(mean_gap))
    return threshold, {
        "sample_count_raw": len(samples),
        "sample_count_kept": len(effective_samples),
        "sample_count_outlier": len(outlier_samples),
    }


def _collect_punctuation_boundary_gap_candidates(
    token_units: list[dict[str, Any]],
    segment_start: float,
    segment_end: float,
) -> list[dict[str, float]]:
    """
    功能说明：收集窗口内“标点左右内容 token”边界 gap 候选。
    参数说明：
    - token_units: token 列表。
    - segment_start/segment_end: 目标窗口边界。
    返回值：
    - list[dict[str, float]]: 候选列表（boundary/gap）。
    异常说明：无。
    边界条件：仅保留窗口内且 gap>0 的候选。
    """
    gap_map: dict[float, float] = {}
    for token_index, token_item in enumerate(token_units):
        token_text = str(token_item.get("text", ""))
        if not _is_punctuation_only_token_text(token_text):
            continue
        token_start = _safe_float(token_item.get("start_time", 0.0), 0.0)
        token_end = _safe_float(token_item.get("end_time", token_start), token_start)
        if token_end <= segment_start + EPSILON_SECONDS or token_start >= segment_end - EPSILON_SECONDS:
            continue
        left_index = _find_neighbor_content_token_index(
            token_units=token_units,
            center_index=token_index,
            direction=-1,
            segment_start=segment_start,
            segment_end=segment_end,
        )
        right_index = _find_neighbor_content_token_index(
            token_units=token_units,
            center_index=token_index,
            direction=1,
            segment_start=segment_start,
            segment_end=segment_end,
        )
        if left_index is None or right_index is None:
            continue
        left_end = _safe_float(token_units[left_index].get("end_time", 0.0), 0.0)
        right_start = _safe_float(token_units[right_index].get("start_time", left_end), left_end)
        gap_value = max(0.0, right_start - left_end)
        boundary = _round_time(right_start)
        if gap_value <= EPSILON_SECONDS:
            continue
        if not (segment_start + EPSILON_SECONDS < boundary < segment_end - EPSILON_SECONDS):
            continue
        if gap_value > _safe_float(gap_map.get(boundary, 0.0), 0.0):
            gap_map[boundary] = float(gap_value)
    return [
        {"boundary": float(boundary), "gap": float(gap_map[boundary])}
        for boundary in sorted(gap_map.keys())
    ]


def _collect_ranked_token_gap_candidates(
    token_units: list[dict[str, Any]],
    segment_start: float,
    segment_end: float,
) -> list[dict[str, float]]:
    """
    功能说明：收集窗口内 token 间隔候选并按 gap 从大到小排序。
    参数说明：
    - token_units: token 列表。
    - segment_start/segment_end: 目标窗口边界。
    返回值：
    - list[dict[str, float]]: 候选列表（rank/boundary/gap）。
    异常说明：无。
    边界条件：去重同一 boundary，仅保留最大 gap。
    """
    gap_map: dict[float, float] = {}
    for left_index, left_item in enumerate(token_units[:-1]):
        left_end = _safe_float(left_item.get("end_time", 0.0), 0.0)
        if left_end <= segment_start + EPSILON_SECONDS or left_end >= segment_end - EPSILON_SECONDS:
            continue
        right_index = _find_neighbor_content_token_index(
            token_units=token_units,
            center_index=left_index,
            direction=1,
            segment_start=segment_start,
            segment_end=segment_end,
        )
        if right_index is None:
            continue
        right_start = _safe_float(token_units[right_index].get("start_time", left_end), left_end)
        gap_value = max(0.0, right_start - left_end)
        boundary = _round_time(right_start)
        if gap_value <= EPSILON_SECONDS:
            continue
        if not (segment_start + EPSILON_SECONDS < boundary < segment_end - EPSILON_SECONDS):
            continue
        if gap_value > _safe_float(gap_map.get(boundary, 0.0), 0.0):
            gap_map[boundary] = float(gap_value)
    ranked_boundaries = sorted(gap_map.items(), key=lambda item: (-float(item[1]), float(item[0])))
    return [
        {
            "rank": float(index + 1),
            "boundary": float(boundary),
            "gap": float(gap_value),
        }
        for index, (boundary, gap_value) in enumerate(ranked_boundaries)
    ]


def _split_window_with_boundaries(
    window_item: dict[str, Any],
    boundary_points: list[float],
    split_basis: str,
    split_rank_map: dict[float, int],
    split_source_window_id: str,
) -> list[dict[str, Any]]:
    """
    功能说明：按边界点切分单个窗口并写入重切追踪字段。
    参数说明：
    - window_item: 原窗口对象。
    - boundary_points: 分割点列表（含起止点）。
    - split_basis: 切分依据标记。
    - split_rank_map: 边界到 rank 的映射。
    - split_source_window_id: 原始窗口ID。
    返回值：
    - list[dict[str, Any]]: 切分后的子窗口。
    异常说明：无。
    边界条件：零时长子窗口会被过滤。
    """
    split_points = sorted({_round_time(float(point)) for point in boundary_points})
    if len(split_points) <= 2:
        return [dict(window_item)]

    output: list[dict[str, Any]] = []
    for index in range(len(split_points) - 1):
        part_start = split_points[index]
        part_end = split_points[index + 1]
        if part_end - part_start <= EPSILON_SECONDS:
            continue
        rewritten = dict(window_item)
        rewritten["start_time"] = _round_time(part_start)
        rewritten["end_time"] = _round_time(part_end)
        rewritten["duration"] = _round_time(max(0.0, part_end - part_start))
        rewritten["split_basis"] = str(split_basis)
        rewritten["split_source_window_id"] = str(split_source_window_id)
        left_rank = split_rank_map.get(_round_time(part_start))
        right_rank = split_rank_map.get(_round_time(part_end))
        rank_candidates = [int(item) for item in [left_rank, right_rank] if item is not None]
        if rank_candidates:
            rewritten["split_rank"] = int(min(rank_candidates))
        output.append(rewritten)
    return output


def _resplit_one_long_lyric_window(
    window_item: dict[str, Any],
    token_units: list[dict[str, Any]],
    max_lyric_window_seconds: float,
    dynamic_gap_threshold_seconds: float,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    功能说明：对单个超长 lyric 窗口执行“标点动态阈值 + token gap 排名”递进切分。
    参数说明：
    - window_item: 原 lyric 窗口。
    - token_units: 对应句级 token 列表。
    - max_lyric_window_seconds: 最大允许窗口时长（秒）。
    - dynamic_gap_threshold_seconds: 全局动态阈值（秒）。
    - events: 重切事件累计列表。
    返回值：
    - list[dict[str, Any]]: 切分后的窗口列表。
    异常说明：无。
    边界条件：无法继续切分时保留剩余超长窗口。
    """
    source_window_id = str(window_item.get("window_id", ""))
    working_windows = [dict(window_item)]
    safe_max_duration = max(EPSILON_SECONDS, float(max_lyric_window_seconds))
    safe_dynamic_gap_threshold = max(MIN_DYNAMIC_GAP_SECONDS, float(dynamic_gap_threshold_seconds))

    for _ in range(MAX_LONG_LYRIC_RESPLIT_ITERATIONS):
        overlong_exists = any(_window_duration(item) > safe_max_duration + EPSILON_SECONDS for item in working_windows)
        if not overlong_exists:
            break

        stage_a_changed = False
        after_stage_a: list[dict[str, Any]] = []
        for segment in working_windows:
            segment_duration = _window_duration(segment)
            if segment_duration <= safe_max_duration + EPSILON_SECONDS:
                after_stage_a.append(segment)
                continue
            segment_start = _safe_float(segment.get("start_time", 0.0), 0.0)
            segment_end = _safe_float(segment.get("end_time", segment_start), segment_start)
            punct_candidates = _collect_punctuation_boundary_gap_candidates(
                token_units=token_units,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            if not punct_candidates:
                after_stage_a.append(segment)
                continue
            punct_samples = [float(item.get("gap", 0.0)) for item in punct_candidates]
            local_threshold, local_stats = _compute_local_dynamic_gap_threshold(
                samples=punct_samples,
                fallback_value=safe_dynamic_gap_threshold,
            )
            selected_candidates = [
                item for item in punct_candidates if float(item.get("gap", 0.0)) >= local_threshold - EPSILON_SECONDS
            ]
            if not selected_candidates:
                after_stage_a.append(segment)
                continue
            ranked_candidates = sorted(
                selected_candidates,
                key=lambda item: (-_safe_float(item.get("gap", 0.0), 0.0), _safe_float(item.get("boundary", 0.0), 0.0)),
            )
            split_rank_map: dict[float, int] = {}
            for rank_index, candidate in enumerate(ranked_candidates):
                split_rank_map[_round_time(_safe_float(candidate.get("boundary", 0.0), 0.0))] = int(rank_index + 1)
            boundary_points = [segment_start, segment_end] + [
                _safe_float(item.get("boundary", 0.0), 0.0) for item in selected_candidates
            ]
            split_segments = _split_window_with_boundaries(
                window_item=segment,
                boundary_points=boundary_points,
                split_basis="punct_dynamic",
                split_rank_map=split_rank_map,
                split_source_window_id=source_window_id,
            )
            if len(split_segments) <= 1:
                after_stage_a.append(segment)
                continue
            stage_a_changed = True
            after_stage_a.extend(split_segments)
            for candidate in ranked_candidates:
                boundary_value = _round_time(_safe_float(candidate.get("boundary", 0.0), 0.0))
                events.append(
                    {
                        "source_window_id": source_window_id,
                        "split_basis": "punct_dynamic",
                        "split_rank": int(split_rank_map.get(boundary_value, 0)),
                        "boundary_time": boundary_value,
                        "gap_seconds": _round_time(_safe_float(candidate.get("gap", 0.0), 0.0)),
                        "segment_start_time": _round_time(segment_start),
                        "segment_end_time": _round_time(segment_end),
                        "local_threshold_seconds": _round_time(local_threshold),
                        "sample_count_raw": int(local_stats.get("sample_count_raw", 0)),
                        "sample_count_kept": int(local_stats.get("sample_count_kept", 0)),
                        "sample_count_outlier": int(local_stats.get("sample_count_outlier", 0)),
                    }
                )
        working_windows = after_stage_a

        overlong_exists = any(_window_duration(item) > safe_max_duration + EPSILON_SECONDS for item in working_windows)
        if not overlong_exists:
            break

        stage_b_changed = False
        after_stage_b: list[dict[str, Any]] = []
        for segment in working_windows:
            segment_duration = _window_duration(segment)
            if segment_duration <= safe_max_duration + EPSILON_SECONDS:
                after_stage_b.append(segment)
                continue
            segment_start = _safe_float(segment.get("start_time", 0.0), 0.0)
            segment_end = _safe_float(segment.get("end_time", segment_start), segment_start)
            ranked_candidates = _collect_ranked_token_gap_candidates(
                token_units=token_units,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            if not ranked_candidates:
                after_stage_b.append(segment)
                continue

            split_windows = [dict(segment)]
            while True:
                target_index = None
                for index, split_window in enumerate(split_windows):
                    if _window_duration(split_window) > safe_max_duration + EPSILON_SECONDS:
                        target_index = index
                        break
                if target_index is None:
                    break

                target_window = split_windows[target_index]
                target_start = _safe_float(target_window.get("start_time", 0.0), 0.0)
                target_end = _safe_float(target_window.get("end_time", target_start), target_start)
                chosen_candidate: dict[str, float] | None = None
                for candidate in ranked_candidates:
                    boundary = _safe_float(candidate.get("boundary", 0.0), 0.0)
                    if target_start + EPSILON_SECONDS < boundary < target_end - EPSILON_SECONDS:
                        chosen_candidate = candidate
                        break
                if chosen_candidate is None:
                    break

                boundary_time = _round_time(_safe_float(chosen_candidate.get("boundary", 0.0), 0.0))
                rank_value = int(_safe_float(chosen_candidate.get("rank", 0.0), 0.0))
                split_segments = _split_window_with_boundaries(
                    window_item=target_window,
                    boundary_points=[target_start, boundary_time, target_end],
                    split_basis="token_gap_rank",
                    split_rank_map={boundary_time: rank_value},
                    split_source_window_id=source_window_id,
                )
                if len(split_segments) <= 1:
                    break
                stage_b_changed = True
                split_windows = split_windows[:target_index] + split_segments + split_windows[target_index + 1 :]
                events.append(
                    {
                        "source_window_id": source_window_id,
                        "split_basis": "token_gap_rank",
                        "split_rank": rank_value,
                        "boundary_time": boundary_time,
                        "gap_seconds": _round_time(_safe_float(chosen_candidate.get("gap", 0.0), 0.0)),
                        "segment_start_time": _round_time(target_start),
                        "segment_end_time": _round_time(target_end),
                    }
                )
            after_stage_b.extend(split_windows)
        working_windows = after_stage_b

        if not stage_a_changed and not stage_b_changed:
            break

    return working_windows


def _pick_long_lyric_local_tiny_merge_target_index(
    split_windows: list[dict[str, Any]],
    source_index: int,
) -> tuple[int | None, str]:
    """
    功能说明：在“同一长句切分子窗”内选择 tiny 并段目标。
    参数说明：
    - split_windows: 同一长句重切后的子窗列表（仅句内窗口）。
    - source_index: 待并入的 tiny 子窗索引。
    返回值：
    - tuple[int | None, str]: (目标索引, 决策原因)。
    异常说明：无。
    边界条件：仅在句内选择左右邻居，不越界到其他窗口。
    """
    left_index = source_index - 1 if source_index - 1 >= 0 else None
    right_index = source_index + 1 if source_index + 1 < len(split_windows) else None
    if left_index is None and right_index is None:
        return None, "no_neighbor"
    if left_index is None:
        return right_index, "local_edge_right_only"
    if right_index is None:
        return left_index, "local_edge_left_only"

    source_start = _safe_float(split_windows[source_index].get("start_time", 0.0), 0.0)
    source_end = _safe_float(split_windows[source_index].get("end_time", source_start), source_start)
    left_end = _safe_float(split_windows[left_index].get("end_time", source_start), source_start)
    right_start = _safe_float(split_windows[right_index].get("start_time", source_end), source_end)
    left_gap_seconds = max(0.0, source_start - left_end)
    right_gap_seconds = max(0.0, right_start - source_end)
    if left_gap_seconds + EPSILON_SECONDS < right_gap_seconds:
        return left_index, "local_shorter_gap_left"
    if right_gap_seconds + EPSILON_SECONDS < left_gap_seconds:
        return right_index, "local_shorter_gap_right"
    return left_index, "local_equal_gap_left"


def _merge_one_long_lyric_local_tiny_window(
    split_windows: list[dict[str, Any]],
    source_index: int,
    target_index: int,
    reason: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：执行单次“长句内部 tiny 并段”。
    参数说明：
    - split_windows: 同一长句切分子窗列表。
    - source_index: 被并入的 tiny 子窗索引。
    - target_index: 目标子窗索引。
    - reason: 决策原因。
    返回值：
    - tuple[list[dict[str, Any]], dict[str, Any]]: (并段后子窗, 事件)。
    异常说明：无。
    边界条件：并段仅发生在同一长句子窗集合中。
    """
    source_item = dict(split_windows[source_index])
    target_item = dict(split_windows[target_index])
    source_start = _safe_float(source_item.get("start_time", 0.0), 0.0)
    source_end = _safe_float(source_item.get("end_time", source_start), source_start)
    target_start = _safe_float(target_item.get("start_time", source_start), source_start)
    target_end = _safe_float(target_item.get("end_time", target_start), target_start)

    direction = "to_left" if target_index < source_index else "to_right"
    if direction == "to_left":
        target_item["end_time"] = _round_time(max(target_end, source_end))
        target_item["merge_action"] = f"absorb_long_lyric_tiny_{reason}"
        split_windows[target_index] = target_item
        split_windows.pop(source_index)
    else:
        target_item["start_time"] = _round_time(min(target_start, source_start))
        target_item["merge_action"] = f"absorb_long_lyric_tiny_{reason}"
        split_windows[target_index] = target_item
        split_windows.pop(source_index)

    for item in split_windows:
        item["duration"] = _round_time(_window_duration(item))
    event = {
        "merge_kind": "long_lyric_inner_tiny",
        "reason": reason,
        "direction": direction,
        "source_window_id": str(source_item.get("window_id", "")),
        "target_window_id": str(target_item.get("window_id", "")),
        "source_start_time": _round_time(source_start),
        "source_end_time": _round_time(source_end),
        "split_source_window_id": str(source_item.get("split_source_window_id", source_item.get("window_id", ""))),
        "source_sentence_index": int(_safe_float(source_item.get("source_sentence_index", -1), -1)),
    }
    return split_windows, event


def _merge_tiny_within_long_lyric_window(
    split_windows: list[dict[str, Any]],
    tiny_merge_seconds: float,
    max_lyric_window_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    功能说明：在同一长句切分子窗集合内部执行 tiny 并段。
    参数说明：
    - split_windows: 同一长句重切后的子窗列表。
    - tiny_merge_seconds: tiny 阈值（秒）。
    - max_lyric_window_seconds: 长句子窗允许的最大时长（秒）。
    返回值：
    - tuple[list[dict[str, Any]], list[dict[str, Any]]]: (并段后子窗, 并段事件)。
    异常说明：无。
    边界条件：仅在句内并段，不影响外部时间窗。
    """
    working_windows = [dict(item) for item in sorted(split_windows, key=lambda row: _safe_float(row.get("start_time", 0.0), 0.0))]
    merge_events: list[dict[str, Any]] = []
    safe_tiny_seconds = max(EPSILON_SECONDS, float(tiny_merge_seconds))
    safe_max_lyric_seconds = max(EPSILON_SECONDS, float(max_lyric_window_seconds))
    while True:
        chosen_source_index: int | None = None
        chosen_target_index: int | None = None
        chosen_reason = ""
        for index, item in enumerate(working_windows):
            if _window_duration(item) > safe_tiny_seconds + EPSILON_SECONDS:
                continue
            preferred_target_index, preferred_reason = _pick_long_lyric_local_tiny_merge_target_index(
                split_windows=working_windows,
                source_index=index,
            )
            if preferred_target_index is None:
                continue
            left_index = index - 1 if index - 1 >= 0 else None
            right_index = index + 1 if index + 1 < len(working_windows) else None
            candidate_targets: list[tuple[int, str]] = [(preferred_target_index, preferred_reason)]
            if left_index is not None and left_index != preferred_target_index:
                candidate_targets.append((left_index, "local_alternate_left_for_max_window"))
            if right_index is not None and right_index != preferred_target_index:
                candidate_targets.append((right_index, "local_alternate_right_for_max_window"))

            source_start = _safe_float(working_windows[index].get("start_time", 0.0), 0.0)
            source_end = _safe_float(working_windows[index].get("end_time", source_start), source_start)
            for target_index, reason in candidate_targets:
                target_start = _safe_float(working_windows[target_index].get("start_time", source_start), source_start)
                target_end = _safe_float(working_windows[target_index].get("end_time", target_start), target_start)
                merged_duration = max(source_end, target_end) - min(source_start, target_start)
                if merged_duration <= safe_max_lyric_seconds + EPSILON_SECONDS:
                    chosen_source_index = index
                    chosen_target_index = target_index
                    chosen_reason = reason
                    break
            if chosen_source_index is not None:
                break

        if chosen_source_index is None or chosen_target_index is None:
            break
        working_windows, event = _merge_one_long_lyric_local_tiny_window(
            split_windows=working_windows,
            source_index=chosen_source_index,
            target_index=chosen_target_index,
            reason=chosen_reason,
        )
        merge_events.append(event)
    return working_windows, merge_events


def resplit_long_lyric_windows(
    windows_raw: list[dict[str, Any]],
    sentence_units: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    beat_candidates: list[float],
    duration_seconds: float,
    dynamic_gap_threshold_seconds: float,
    tiny_merge_bars: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：对超长 lyric 窗口执行递进重切（先标点动态阈值，再 token gap 排名兜底）。
    参数说明：
    - windows_raw: 初始窗口列表。
    - sentence_units: 句级歌词列表（含 token_units）。
    - beats/beat_candidates: 节拍对象与候选列表。
    - duration_seconds: 音频总时长（秒）。
    - dynamic_gap_threshold_seconds: 全局动态 gap 阈值（秒）。
    - tiny_merge_bars: tiny 并段阈值（小节），用于长句内部并段。
    返回值：
    - tuple[list[dict[str, Any]], dict[str, Any]]: (重切后窗口, 统计信息)。
    异常说明：无。
    边界条件：无可切分信息时保留原窗口并记录剩余超长数量。
    """
    # 项目内模块：复用小节时长估计能力
    from .role_merger import estimate_bar_length_seconds

    safe_duration = max(0.0, float(duration_seconds))
    normalized_windows = _normalize_windows_continuity(windows=windows_raw, duration_seconds=safe_duration)
    bar_length_seconds = max(0.2, float(estimate_bar_length_seconds(beats=beats, beat_candidates=beat_candidates)))
    max_lyric_window_seconds = float(bar_length_seconds) * float(LONG_LYRIC_RESPLIT_MAX_BARS)
    safe_tiny_merge_bars = _safe_float(tiny_merge_bars, 0.9)
    if safe_tiny_merge_bars <= 0.0:
        safe_tiny_merge_bars = 0.9
    long_lyric_inner_tiny_merge_seconds = float(bar_length_seconds) * safe_tiny_merge_bars
    safe_dynamic_gap_threshold = max(MIN_DYNAMIC_GAP_SECONDS, float(dynamic_gap_threshold_seconds))
    sentence_lookup = {
        int(item.get("sentence_index", index)): list(item.get("token_units", []))
        for index, item in enumerate(normalize_sentence_units(sentence_units=sentence_units, duration_seconds=safe_duration))
    }

    resplit_events: list[dict[str, Any]] = []
    long_lyric_inner_tiny_merge_events: list[dict[str, Any]] = []
    output_windows: list[dict[str, Any]] = []
    for window_item in normalized_windows:
        rewritten = dict(window_item)
        rewritten["duration"] = _round_time(_window_duration(rewritten))
        role_hint = str(rewritten.get("window_role_hint", "")).lower().strip()
        if role_hint != "lyric":
            output_windows.append(rewritten)
            continue
        if _window_duration(rewritten) <= max_lyric_window_seconds + EPSILON_SECONDS:
            output_windows.append(rewritten)
            continue
        sentence_index = int(_safe_float(rewritten.get("source_sentence_index", -1), -1))
        token_units = sentence_lookup.get(sentence_index, [])
        if not token_units:
            output_windows.append(rewritten)
            continue
        split_windows = _resplit_one_long_lyric_window(
            window_item=rewritten,
            token_units=token_units,
            max_lyric_window_seconds=max_lyric_window_seconds,
            dynamic_gap_threshold_seconds=safe_dynamic_gap_threshold,
            events=resplit_events,
        )
        split_result_windows = split_windows if split_windows else [rewritten]
        all_within_max_lyric_window = all(
            _window_duration(item) <= max_lyric_window_seconds + EPSILON_SECONDS
            for item in split_result_windows
        )
        if all_within_max_lyric_window:
            split_result_windows, local_merge_events = _merge_tiny_within_long_lyric_window(
                split_windows=split_result_windows,
                tiny_merge_seconds=long_lyric_inner_tiny_merge_seconds,
                max_lyric_window_seconds=max_lyric_window_seconds,
            )
            long_lyric_inner_tiny_merge_events.extend(local_merge_events)
        output_windows.extend(split_result_windows)

    normalized_output_windows = _normalize_windows_continuity(windows=output_windows, duration_seconds=safe_duration)
    remaining_over3_count = 0
    for item in normalized_output_windows:
        role_hint = str(item.get("window_role_hint", "")).lower().strip()
        if role_hint != "lyric":
            continue
        if _window_duration(item) > max_lyric_window_seconds + EPSILON_SECONDS:
            remaining_over3_count += 1

    stats = {
        "bar_length_seconds": _round_time(bar_length_seconds),
        "max_lyric_window_seconds": _round_time(max_lyric_window_seconds),
        "long_lyric_resplit_events": resplit_events,
        "long_lyric_inner_tiny_merge_seconds": _round_time(long_lyric_inner_tiny_merge_seconds),
        "long_lyric_inner_tiny_merge_events": long_lyric_inner_tiny_merge_events,
        "long_lyric_remaining_over3_count": int(remaining_over3_count),
    }
    return normalized_output_windows, stats


def build_windows_from_sentences(
    sentence_units: list[dict[str, Any]],
    duration_seconds: float,
    dynamic_gap_threshold_seconds: float,
) -> list[dict[str, Any]]:
    """
    功能说明：根据句级歌词构建“歌词句窗口 + 其他窗口”。
    参数说明：
    - sentence_units: 句级歌词单元列表。
    - duration_seconds: 音频总时长（秒）。
    - dynamic_gap_threshold_seconds: 句间成窗阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 覆盖全时长的窗口列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无歌词时返回单个其他窗口覆盖全曲。
    """
    safe_duration = max(0.0, float(duration_seconds))
    safe_gap_threshold = float(dynamic_gap_threshold_seconds)
    if safe_gap_threshold <= 0.0:
        safe_gap_threshold = DEFAULT_DYNAMIC_GAP_SECONDS

    normalized_sentences = normalize_sentence_units(sentence_units=sentence_units, duration_seconds=safe_duration)
    if not normalized_sentences:
        return [
            _build_window_item(
                window_id="win_0001",
                start_time=0.0,
                end_time=safe_duration,
                window_role_hint="other",
                window_type="other_full_track",
            )
        ]

    windows: list[dict[str, Any]] = []
    cursor = 0.0
    window_index = 1

    for sentence_index, sentence_item in enumerate(normalized_sentences):
        sentence_start = _safe_float(sentence_item.get("start_time", 0.0), 0.0)
        sentence_end = _safe_float(sentence_item.get("end_time", sentence_start), sentence_start)
        sentence_start = max(cursor, sentence_start)
        sentence_end = max(sentence_start, sentence_end)

        gap_seconds = sentence_start - cursor
        if gap_seconds > EPSILON_SECONDS:
            if gap_seconds >= safe_gap_threshold - EPSILON_SECONDS:
                window_type = "other_leading" if sentence_index == 0 else "other_between"
                windows.append(
                    _build_window_item(
                        window_id=f"win_{window_index:04d}",
                        start_time=cursor,
                        end_time=sentence_start,
                        window_role_hint="other",
                        window_type=window_type,
                    )
                )
                window_index += 1
            else:
                # 短gap左并：优先把短空隙扩到前一句歌词窗口末尾。
                if windows and str(windows[-1].get("window_role_hint", "other")).lower().strip() == "lyric":
                    windows[-1]["end_time"] = _round_time(sentence_start)
                    windows[-1]["duration"] = _round_time(
                        max(
                            0.0,
                            _safe_float(windows[-1].get("end_time", 0.0), 0.0)
                            - _safe_float(windows[-1].get("start_time", 0.0), 0.0),
                        )
                    )
                else:
                    # 无可用左邻歌词窗口时，保留为其他窗口，避免丢失时间覆盖。
                    window_type = "other_leading" if sentence_index == 0 else "other_between"
                    windows.append(
                        _build_window_item(
                            window_id=f"win_{window_index:04d}",
                            start_time=cursor,
                            end_time=sentence_start,
                            window_role_hint="other",
                            window_type=window_type,
                        )
                    )
                    window_index += 1

        windows.append(
            _build_window_item(
                window_id=f"win_{window_index:04d}",
                start_time=sentence_start,
                end_time=sentence_end,
                window_role_hint="lyric",
                window_type="lyric_sentence",
                source_sentence_index=int(sentence_item.get("sentence_index", sentence_index)),
            )
        )
        window_index += 1
        cursor = sentence_end

    if safe_duration - cursor > EPSILON_SECONDS:
        windows.append(
            _build_window_item(
                window_id=f"win_{window_index:04d}",
                start_time=cursor,
                end_time=safe_duration,
                window_role_hint="other",
                window_type="other_trailing",
            )
        )

    return _normalize_windows_continuity(windows=windows, duration_seconds=safe_duration)


def inject_boundary_points_into_windows(
    windows: list[dict[str, Any]],
    boundary_points: list[float],
    duration_seconds: float,
    target_window_role_hint: str | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：将外部边界点（如A0边界）注入窗口切分，得到更细粒度窗口。
    参数说明：
    - windows: 原始窗口列表。
    - boundary_points: 外部边界点列表（秒）。
    - duration_seconds: 音频总时长（秒）。
    - target_window_role_hint: 仅切分指定 role_hint 的窗口；None 表示全部窗口均可切分。
    返回值：
    - list[dict[str, Any]]: 注入边界后的窗口列表。
    异常说明：无。
    边界条件：仅在窗口内部的边界点才会触发切分；可按 role_hint 过滤切分目标。
    """
    if not windows:
        return []

    safe_duration = max(0.0, float(duration_seconds))
    normalized_points = sorted(
        {
            _round_time(max(0.0, min(safe_duration, _safe_float(point, 0.0))))
            for point in list(boundary_points)
            if EPSILON_SECONDS < _safe_float(point, 0.0) < safe_duration - EPSILON_SECONDS
        }
    )
    if not normalized_points:
        return _normalize_windows_continuity(windows=windows, duration_seconds=safe_duration)

    output: list[dict[str, Any]] = []
    safe_target_role_hint = None if target_window_role_hint is None else str(target_window_role_hint).strip().lower()
    for window_item in sorted(windows, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0)):
        current_role_hint = str(window_item.get("window_role_hint", "")).strip().lower()
        should_split_current = True
        if safe_target_role_hint is not None and current_role_hint != safe_target_role_hint:
            should_split_current = False

        start_time = _safe_float(window_item.get("start_time", 0.0), 0.0)
        end_time = max(start_time, _safe_float(window_item.get("end_time", start_time), start_time))
        if end_time - start_time <= EPSILON_SECONDS:
            continue

        split_points = [start_time, end_time]
        if should_split_current:
            split_points = [start_time]
            for point in normalized_points:
                if start_time + EPSILON_SECONDS < point < end_time - EPSILON_SECONDS:
                    split_points.append(point)
            split_points.append(end_time)
            split_points = sorted(split_points)

        has_inner_split = len(split_points) > 2
        for index in range(len(split_points) - 1):
            part_start = split_points[index]
            part_end = split_points[index + 1]
            if part_end - part_start <= EPSILON_SECONDS:
                continue
            rewritten = dict(window_item)
            rewritten["start_time"] = _round_time(part_start)
            rewritten["end_time"] = _round_time(part_end)
            rewritten["duration"] = _round_time(max(0.0, part_end - part_start))
            rewritten["source_window_id"] = str(window_item.get("window_id", ""))
            if should_split_current and has_inner_split:
                rewritten["window_type"] = f"{str(window_item.get('window_type', 'window'))}_a0_split"
            output.append(rewritten)

    return _normalize_windows_continuity(windows=output, duration_seconds=safe_duration)
