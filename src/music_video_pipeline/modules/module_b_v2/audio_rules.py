"""
文件用途：提供模块B v2 的音频离散化与节奏张力语义增强规则。
核心流程：读取模块A音频特征 -> 归一化 tension 语义 -> 输出按 segment_id 索引的音频语义画像。
输入输出：输入模块A输出与编排模板，输出按 segment_id 索引的增强规则结果。
依赖说明：依赖标准库 statistics 与项目内 v2 数据结构。
维护说明：本文件只做确定性规则，不调用任何 LLM；不负责运镜/转场 preset 的候选映射。
"""

# 标准库：用于中位数统计。
from statistics import median
from typing import Any

# 项目内模块：导入 v2 数据结构。
from music_video_pipeline.modules.module_b_v2.models import (
    SegmentAudioFeaturesV2,
    StoryboardTemplate,
)


def build_segment_audio_features_v2(
    module_a_output: dict[str, Any],
    storyboard_template: StoryboardTemplate,
) -> dict[str, SegmentAudioFeaturesV2]:
    """
    功能说明：构建按 segment_id 索引的增强音频规则结果。
    参数说明：
    - module_a_output: 模块A输出对象。
    - storyboard_template: 已编译的编排模板。
    返回值：
    - dict[str, SegmentAudioFeaturesV2]: segment_id 到增强特征的映射。
    异常说明：无。
    边界条件：缺失 rhythm_tension 时按 0.0 处理，缺失 next energy 时回退 low。
    """
    segments = module_a_output.get("segments", [])
    energy_features = module_a_output.get("energy_features", [])
    if not isinstance(segments, list):
        return {}

    del storyboard_template

    grouped_segment_indexes: dict[str, list[int]] = {}
    tension_values_by_big_segment: dict[str, list[float]] = {}
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        big_segment_id = str(segment.get("big_segment_id", "")).strip()
        grouped_segment_indexes.setdefault(big_segment_id, []).append(segment_index)
        tension_values_by_big_segment.setdefault(big_segment_id, []).append(
            _resolve_rhythm_tension(energy_features=energy_features, segment_index=segment_index)
        )

    result: dict[str, SegmentAudioFeaturesV2] = {}
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            continue
        big_segment_id = str(segment.get("big_segment_id", "")).strip()
        grouped_indexes = grouped_segment_indexes.get(big_segment_id, [segment_index])
        rank_index = grouped_indexes.index(segment_index) if segment_index in grouped_indexes else 0
        group_tensions = tension_values_by_big_segment.get(big_segment_id, [0.0])
        current_tension = _resolve_rhythm_tension(energy_features=energy_features, segment_index=segment_index)
        prev_tension = _resolve_rhythm_tension(energy_features=energy_features, segment_index=max(0, segment_index - 1))
        energy_item = _resolve_energy_item(energy_features=energy_features, segment_index=segment_index)
        energy_level = str(energy_item.get("energy_level", "mid")).strip() or "mid"
        trend = str(energy_item.get("trend", "flat")).strip() or "flat"
        tension_band = _classify_tension_band(current_tension=current_tension, group_tensions=group_tensions)
        tension_delta = _classify_tension_delta(previous_tension=prev_tension, current_tension=current_tension)
        is_local_peak = _compute_is_local_peak(
            energy_features=energy_features,
            segment_index=segment_index,
            current_tension=current_tension,
        )
        position_in_big_segment = _resolve_position_in_big_segment(
            rank_index=rank_index,
            total_count=len(grouped_indexes),
        )

        result[segment_id] = {
            "segment_id": segment_id,
            "big_segment_id": big_segment_id,
            "energy_level": energy_level,
            "trend": trend,
            "tension_band": tension_band,
            "tension_delta": tension_delta,
            "is_local_peak": is_local_peak,
            "position_in_big_segment": position_in_big_segment,
            "segment_rank_in_big_segment": rank_index + 1,
            "segment_count_in_big_segment": len(grouped_indexes),
            "beat_positions": list(energy_item.get("beat_positions", [])),
            "onset_positions": [
                {"offset": float(op.get("offset", 0.0)), "energy": float(op.get("energy", 0.0))}
                for op in energy_item.get("onset_positions", [])
                if isinstance(op, dict)
            ],
            "onset_density": float(energy_item.get("onset_density", 0.0)),
            "spectral_centroid_mean": float(energy_item.get("spectral_centroid_mean", 0.5)),
        }
    return result


def _resolve_energy_item(energy_features: list[Any], segment_index: int) -> dict[str, Any]:
    """
    功能说明：安全读取指定 segment 的能量条目。
    参数说明：
    - energy_features: 模块A energy_features 列表。
    - segment_index: 小段索引。
    返回值：
    - dict[str, Any]: 能量条目字典。
    异常说明：无。
    边界条件：缺失时回退为 mid/flat。
    """
    if not isinstance(energy_features, list) or not energy_features:
        return {"energy_level": "mid", "trend": "flat", "rhythm_tension": 0.0}
    safe_index = max(0, min(len(energy_features) - 1, int(segment_index)))
    item = energy_features[safe_index]
    if not isinstance(item, dict):
        return {"energy_level": "mid", "trend": "flat", "rhythm_tension": 0.0}
    return item


def _resolve_rhythm_tension(energy_features: list[Any], segment_index: int) -> float:
    """
    功能说明：读取并归一化 rhythm_tension。
    参数说明：
    - energy_features: 模块A energy_features 列表。
    - segment_index: 小段索引。
    返回值：
    - float: tension 数值。
    异常说明：无。
    边界条件：非法值回退为 0.0。
    """
    item = _resolve_energy_item(energy_features=energy_features, segment_index=segment_index)
    try:
        return float(item.get("rhythm_tension", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _classify_tension_band(current_tension: float, group_tensions: list[float]) -> str:
    """
    功能说明：根据大段内分布将 tension 离散为三档。
    参数说明：
    - current_tension: 当前段 tension。
    - group_tensions: 当前大段全部 tension 列表。
    返回值：
    - str: low/mid/high。
    异常说明：无。
    边界条件：样本数不足时采用简单阈值。
    """
    valid_values = [float(item) for item in group_tensions] if group_tensions else [0.0]
    sorted_values = sorted(valid_values)
    if len(sorted_values) < 3:
        if current_tension >= 0.66:
            return "high"
        if current_tension <= 0.33:
            return "low"
        return "mid"
    low_cut = sorted_values[max(0, len(sorted_values) // 3 - 1)]
    high_cut = sorted_values[min(len(sorted_values) - 1, (len(sorted_values) * 2) // 3)]
    if current_tension <= low_cut:
        return "low"
    if current_tension >= high_cut:
        return "high"
    return "mid"


def _classify_tension_delta(previous_tension: float, current_tension: float) -> str:
    """
    功能说明：将 tension 的相对变化离散为 down/flat/up。
    参数说明：
    - previous_tension: 前一段 tension。
    - current_tension: 当前段 tension。
    返回值：
    - str: down/flat/up。
    异常说明：无。
    边界条件：绝对差值小于 0.08 视为 flat。
    """
    delta = float(current_tension) - float(previous_tension)
    if delta >= 0.08:
        return "up"
    if delta <= -0.08:
        return "down"
    return "flat"


def _compute_is_local_peak(
    energy_features: list[Any],
    segment_index: int,
    current_tension: float,
) -> bool:
    """
    功能说明：判断当前段是否为局部张力峰值。
    参数说明：
    - energy_features: 模块A energy_features 列表。
    - segment_index: 当前小段索引。
    - current_tension: 当前 tension。
    返回值：
    - bool: 是否为局部峰值。
    异常说明：无。
    边界条件：边界位置采用存在的相邻值比较。
    """
    left_tension = _resolve_rhythm_tension(energy_features=energy_features, segment_index=max(0, segment_index - 1))
    right_tension = _resolve_rhythm_tension(
        energy_features=energy_features,
        segment_index=min(len(energy_features) - 1, segment_index + 1),
    )
    return current_tension >= left_tension and current_tension >= right_tension and current_tension >= 0.5


def _resolve_position_in_big_segment(rank_index: int, total_count: int) -> str:
    """
    功能说明：将小段在大段内的位置离散为四类。
    参数说明：
    - rank_index: 0 基序号。
    - total_count: 大段总小段数。
    返回值：
    - str: start/early_mid/late_mid/end。
    异常说明：无。
    边界条件：总数不足 2 时直接返回 start。
    """
    if total_count <= 1:
        return "start"
    if rank_index == 0:
        return "start"
    if rank_index == total_count - 1:
        return "end"
    half_index = max(1, total_count // 2)
    if rank_index < half_index:
        return "early_mid"
    return "late_mid"
