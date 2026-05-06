"""
文件用途：提供模块B v2 的音频离散化、运镜映射与转场映射规则。
核心流程：读取模块A音频特征 -> 归一化 tension 语义 -> 选择候选运镜与转场 preset。
输入输出：输入模块A输出与编排模板，输出按 segment_id 索引的增强规则结果。
依赖说明：依赖标准库 statistics 与项目内 v2 数据结构。
维护说明：本文件只做确定性规则，不调用任何 LLM。
"""

# 标准库：用于中位数统计。
from statistics import median
from typing import Any

# 项目内模块：导入 v2 数据结构。
from music_video_pipeline.modules.module_b_v2.models import (
    CameraPlan,
    SegmentAudioFeaturesV2,
    StoryboardTemplate,
    TransitionPlan,
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

    camera_presets = {
        str(item.get("preset_id", "")).strip(): _normalize_camera_plan(item)
        for item in storyboard_template.get("camera_plan_presets", [])
        if isinstance(item, dict)
    }
    transition_presets = {
        str(item.get("preset_id", "")).strip(): _normalize_transition_plan(item)
        for item in storyboard_template.get("transition_presets", [])
        if isinstance(item, dict)
    }
    camera_mapping = {
        (str(item.get("energy_level", "")).strip(), str(item.get("trend", "")).strip()): item
        for item in storyboard_template.get("camera_mapping", [])
        if isinstance(item, dict)
    }
    transition_mapping = {
        (
            str(item.get("current_energy_level", "")).strip(),
            str(item.get("next_energy_level", "")).strip(),
        ): item
        for item in storyboard_template.get("transition_mapping", [])
        if isinstance(item, dict)
    }

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
        next_energy_item = _resolve_energy_item(
            energy_features=energy_features,
            segment_index=min(len(segments) - 1, segment_index + 1),
        )
        energy_level = str(energy_item.get("energy_level", "mid")).strip() or "mid"
        trend = str(energy_item.get("trend", "flat")).strip() or "flat"
        next_energy_level = str(next_energy_item.get("energy_level", "low")).strip() or "low"
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

        camera_rule = camera_mapping.get((energy_level, trend), {})
        transition_rule = transition_mapping.get((energy_level, next_energy_level), {})
        camera_candidates = _resolve_camera_candidates(
            preset_ids=camera_rule.get("candidate_preset_ids", []),
            presets=camera_presets,
        )
        transition_candidates = _resolve_transition_candidates(
            preset_ids=transition_rule.get("candidate_preset_ids", []),
            presets=transition_presets,
        )
        default_camera_plan = camera_presets.get(
            str(camera_rule.get("default_preset_id", "none")).strip(),
            camera_presets.get("none", _build_none_camera_plan()),
        )
        default_transition_plan = transition_presets.get(
            str(transition_rule.get("default_preset_id", "none")).strip(),
            transition_presets.get("none", _build_none_transition_plan()),
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
            "camera_plan_candidates": camera_candidates,
            "transition_plan_candidates": transition_candidates,
            "default_camera_plan": default_camera_plan,
            "default_transition_plan": default_transition_plan,
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


def _resolve_camera_candidates(preset_ids: Any, presets: dict[str, CameraPlan]) -> list[CameraPlan]:
    """
    功能说明：根据 preset_id 列表解析运镜候选。
    参数说明：
    - preset_ids: 候选 preset_id 数组。
    - presets: preset 索引映射。
    返回值：
    - list[CameraPlan]: 合法候选数组。
    异常说明：无。
    边界条件：空结果时回退为 none。
    """
    results: list[CameraPlan] = []
    if isinstance(preset_ids, list):
        for preset_id in preset_ids:
            normalized_id = str(preset_id).strip()
            if normalized_id and normalized_id in presets:
                results.append(dict(presets[normalized_id]))
    if results:
        return results
    return [dict(presets.get("none", _build_none_camera_plan()))]


def _resolve_transition_candidates(preset_ids: Any, presets: dict[str, TransitionPlan]) -> list[TransitionPlan]:
    """
    功能说明：根据 preset_id 列表解析转场候选。
    参数说明：
    - preset_ids: 候选 preset_id 数组。
    - presets: preset 索引映射。
    返回值：
    - list[TransitionPlan]: 合法候选数组。
    异常说明：无。
    边界条件：空结果时回退为 none。
    """
    results: list[TransitionPlan] = []
    if isinstance(preset_ids, list):
        for preset_id in preset_ids:
            normalized_id = str(preset_id).strip()
            if normalized_id and normalized_id in presets:
                results.append(dict(presets[normalized_id]))
    if results:
        return results
    return [dict(presets.get("none", _build_none_transition_plan()))]


def _normalize_camera_plan(payload: dict[str, Any]) -> CameraPlan:
    """
    功能说明：将原始运镜 preset 载荷标准化为 CameraPlan。
    参数说明：
    - payload: 原始 preset 字典。
    返回值：
    - CameraPlan: 标准化后的对象。
    异常说明：无。
    边界条件：缺失字段时按 none 回退。
    """
    return {
        "preset_id": str(payload.get("preset_id", "none")).strip() or "none",
        "mode": str(payload.get("mode", "none")).strip() or "none",
        "direction": str(payload.get("direction", "center")).strip() or "center",
        "strength": str(payload.get("strength", "none")).strip() or "none",
        "easing": str(payload.get("easing", "linear")).strip() or "linear",
    }


def _normalize_transition_plan(payload: dict[str, Any]) -> TransitionPlan:
    """
    功能说明：将原始转场 preset 载荷标准化为 TransitionPlan。
    参数说明：
    - payload: 原始 preset 字典。
    返回值：
    - TransitionPlan: 标准化后的对象。
    异常说明：无。
    边界条件：缺失字段时按 none 回退。
    """
    try:
        duration_ms = int(payload.get("duration_ms", 0))
    except (TypeError, ValueError):
        duration_ms = 0
    return {
        "preset_id": str(payload.get("preset_id", "none")).strip() or "none",
        "kind": str(payload.get("kind", "none")).strip() or "none",
        "duration_ms": max(0, duration_ms),
        "easing": str(payload.get("easing", "linear")).strip() or "linear",
    }


def _build_none_camera_plan() -> CameraPlan:
    """
    功能说明：构建 none 运镜对象。
    参数说明：无。
    返回值：
    - CameraPlan: none 对象。
    异常说明：无。
    边界条件：用于缺省回退。
    """
    return {
        "preset_id": "none",
        "mode": "none",
        "direction": "center",
        "strength": "none",
        "easing": "linear",
    }


def _build_none_transition_plan() -> TransitionPlan:
    """
    功能说明：构建 none 转场对象。
    参数说明：无。
    返回值：
    - TransitionPlan: none 对象。
    异常说明：无。
    边界条件：用于缺省回退。
    """
    return {
        "preset_id": "none",
        "kind": "none",
        "duration_ms": 0,
        "easing": "linear",
    }
