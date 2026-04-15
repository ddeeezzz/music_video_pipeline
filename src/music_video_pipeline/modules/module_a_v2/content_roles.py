"""
文件用途：编排模块A V2“窗口 -> 角色 -> 并段 -> A1”的统一流程。
核心流程：窗口构造 -> 四分类 -> 并段 -> A1求解 -> 最终S段。
输入输出：输入A0/分句/RMS/节拍，输出最终segments与中间产物。
依赖说明：依赖 timeline 子模块，不直接承担底层算法细节。
维护说明：本文件应保持薄编排，不堆叠复杂规则实现。
"""

# 标准库：用于类型提示
from typing import Any

# 项目内模块：时间轴子模块入口
from music_video_pipeline.modules.module_a_v2.timeline import (
    build_windows_from_sentences,
    classify_window_roles,
    estimate_bar_length_seconds,
    merge_windows_by_rules,
    resplit_long_lyric_windows,
    resolve_big_timestamps_and_segments,
)


# 常量：tiny并段默认阈值（小节）
DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS = 0.9


# 常量：近锚点冲突默认阈值（秒）
DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS = 1.5


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：将输入安全转换为浮点数。
    参数说明：
    - value: 输入对象。
    - default: 回退值。
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


def _resolve_dynamic_gap_threshold(sentence_split_stats: dict[str, Any], fallback_value: float = 0.35) -> float:
    """
    功能说明：从分句统计中提取动态句间阈值。
    参数说明：
    - sentence_split_stats: 分句统计字典。
    - fallback_value: 缺失时回退值。
    返回值：
    - float: 有效阈值（秒）。
    异常说明：无。
    边界条件：非法值回退 fallback。
    """
    if not isinstance(sentence_split_stats, dict):
        return max(0.04, float(fallback_value))
    threshold = _safe_float(sentence_split_stats.get("dynamic_gap_threshold_seconds", fallback_value), fallback_value)
    if threshold <= 0.0:
        return max(0.04, float(fallback_value))
    return float(threshold)


def apply_content_role_pipeline(
    big_segments_stage1: list[dict[str, Any]],
    sentence_units: list[dict[str, Any]],
    sentence_split_stats: dict[str, Any],
    beat_candidates: list[float],
    beats: list[dict[str, Any]],
    vocal_rms_times: list[float],
    vocal_rms_values: list[float],
    accompaniment_rms_times: list[float],
    accompaniment_rms_values: list[float],
    tiny_merge_bars: float,
    lyric_head_offset_seconds: float,
    near_anchor_seconds: float,
    duration_seconds: float,
    onset_points: list[dict[str, Any]] | None = None,
    accompaniment_chroma_points: list[dict[str, Any]] | None = None,
    vocal_f0_points: list[dict[str, Any]] | None = None,
    accompaniment_f0_points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    功能说明：统一执行模块A V2窗口化与时间轴求解流程。
    参数说明：
    - big_segments_stage1: A0大段列表。
    - sentence_units: 句级歌词列表。
    - sentence_split_stats: 分句统计信息（含动态阈值）。
    - beat_candidates/beats: 节拍候选与结构化节拍。
    - vocal_rms_times/vocal_rms_values: 人声RMS序列。
    - accompaniment_rms_times/accompaniment_rms_values: 伴奏RMS序列。
    - onset_points: 伴奏onset强度点（time+energy_raw）。
    - accompaniment_chroma_points: 伴奏 chroma 点（time+12维向量）。
    - vocal_f0_points: 人声 F0 点（time+f0_hz+voiced）。
    - accompaniment_f0_points: 伴奏 F0 点（time+f0_hz+voiced）。
    - tiny_merge_bars: tiny并段阈值（小节）。
    - lyric_head_offset_seconds: 句首锚点后移秒数。
    - near_anchor_seconds: 近锚点冲突阈值。
    - duration_seconds: 音频总时长。
    返回值：
    - dict[str, Any]: 全量中间产物与最终产物。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无歌词时仍生成“其他窗口”并输出完整segments。
    """
    safe_tiny_merge_bars = _safe_float(tiny_merge_bars, DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS)
    if safe_tiny_merge_bars <= 0.0:
        safe_tiny_merge_bars = DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS

    safe_near_anchor_seconds = _safe_float(near_anchor_seconds, DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS)
    if safe_near_anchor_seconds <= 0.0:
        safe_near_anchor_seconds = DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS

    dynamic_gap_threshold_seconds = _resolve_dynamic_gap_threshold(sentence_split_stats=sentence_split_stats, fallback_value=0.35)

    windows_raw = build_windows_from_sentences(
        sentence_units=sentence_units,
        duration_seconds=duration_seconds,
        dynamic_gap_threshold_seconds=dynamic_gap_threshold_seconds,
    )
    windows_raw, long_lyric_resplit_stats = resplit_long_lyric_windows(
        windows_raw=windows_raw,
        sentence_units=sentence_units,
        beats=beats,
        beat_candidates=beat_candidates,
        duration_seconds=duration_seconds,
        dynamic_gap_threshold_seconds=dynamic_gap_threshold_seconds,
        tiny_merge_bars=safe_tiny_merge_bars,
    )
    bar_length_seconds = _safe_float(
        long_lyric_resplit_stats.get("bar_length_seconds", 0.0),
        estimate_bar_length_seconds(beats=beats, beat_candidates=beat_candidates),
    )
    if bar_length_seconds <= 0.0:
        bar_length_seconds = estimate_bar_length_seconds(beats=beats, beat_candidates=beat_candidates)
    windows_classified = classify_window_roles(
        windows=windows_raw,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
    )
    windows_merged, merge_events = merge_windows_by_rules(
        windows_classified=windows_classified,
        tiny_merge_bars=safe_tiny_merge_bars,
        bar_length_seconds=bar_length_seconds,
        beats=beats,
        duration_seconds=duration_seconds,
        onset_points=list(onset_points or []),
        vocal_rms_times=list(vocal_rms_times or []),
        vocal_rms_values=list(vocal_rms_values or []),
        accompaniment_rms_times=list(accompaniment_rms_times or []),
        accompaniment_rms_values=list(accompaniment_rms_values or []),
        accompaniment_chroma_points=list(accompaniment_chroma_points or []),
        vocal_f0_points=list(vocal_f0_points or []),
        accompaniment_f0_points=list(accompaniment_f0_points or []),
    )
    (
        big_segments_a1,
        small_timestamps,
        segments_final,
        boundary_conflict_resolved,
        big_boundary_moves,
    ) = resolve_big_timestamps_and_segments(
        big_segments_a0=big_segments_stage1,
        windows_merged=windows_merged,
        sentence_units=sentence_units,
        duration_seconds=duration_seconds,
        head_offset_seconds=lyric_head_offset_seconds,
        near_anchor_seconds=safe_near_anchor_seconds,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
    )

    return {
        "dynamic_gap_threshold_seconds": dynamic_gap_threshold_seconds,
        "bar_length_seconds": bar_length_seconds,
        "max_lyric_window_seconds": _safe_float(long_lyric_resplit_stats.get("max_lyric_window_seconds", 0.0), 0.0),
        "long_lyric_resplit_events": list(long_lyric_resplit_stats.get("long_lyric_resplit_events", [])),
        "long_lyric_inner_tiny_merge_seconds": _safe_float(
            long_lyric_resplit_stats.get("long_lyric_inner_tiny_merge_seconds", 0.0), 0.0
        ),
        "long_lyric_inner_tiny_merge_events": list(long_lyric_resplit_stats.get("long_lyric_inner_tiny_merge_events", [])),
        "long_lyric_remaining_over3_count": int(
            _safe_float(long_lyric_resplit_stats.get("long_lyric_remaining_over3_count", 0), 0)
        ),
        "windows_raw": windows_raw,
        "windows_classified": windows_classified,
        "windows_merged": windows_merged,
        "window_merge_events": merge_events,
        "big_segments_a1": big_segments_a1,
        "small_timestamps": small_timestamps,
        "segments_final": segments_final,
        "boundary_conflict_resolved": boundary_conflict_resolved,
        "big_boundary_moves": {
            **big_boundary_moves,
            "window_merge_events": merge_events,
            "dynamic_gap_threshold_seconds": dynamic_gap_threshold_seconds,
            "bar_length_seconds": bar_length_seconds,
            "max_lyric_window_seconds": _safe_float(long_lyric_resplit_stats.get("max_lyric_window_seconds", 0.0), 0.0),
            "long_lyric_resplit_events": list(long_lyric_resplit_stats.get("long_lyric_resplit_events", [])),
            "long_lyric_inner_tiny_merge_seconds": _safe_float(
                long_lyric_resplit_stats.get("long_lyric_inner_tiny_merge_seconds", 0.0), 0.0
            ),
            "long_lyric_inner_tiny_merge_events": list(
                long_lyric_resplit_stats.get("long_lyric_inner_tiny_merge_events", [])
            ),
            "long_lyric_remaining_over3_count": int(
                _safe_float(long_lyric_resplit_stats.get("long_lyric_remaining_over3_count", 0), 0)
            ),
        },
    }
