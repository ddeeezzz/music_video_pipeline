"""
文件用途：提供模块A包化后的兼容导出入口。
核心流程：聚合 orchestrator/backends/lyrics/segmentation/timing_energy 子模块符号。
输入输出：无输入，输出模块A公共与兼容私有函数。
依赖说明：依赖模块A子模块实现。
维护说明：为兼容历史测试与 monkeypatch，不要随意删除已导出私有符号。
"""

# 维护提示：此文件导出的私有符号属于兼容面，测试和外部脚本可能直接引用。

# 项目内模块：导出编排入口与流程函数
from music_video_pipeline.modules.module_a.orchestrator import (
    _probe_audio_duration,
    _run_fallback_pipeline,
    _run_real_pipeline,
    run_module_a,
)
# 项目内模块：导出外部后端函数
from music_video_pipeline.modules.module_a.backends import (
    _build_lyric_units_from_sentence_info,
    _build_lyric_units_from_timestamp,
    _build_sentence_unit_from_tokens,
    _build_token_units_from_timestamp,
    _detect_big_segments_with_allin1,
    _extract_acoustic_candidates_with_librosa,
    _extract_funasr_records,
    _import_allin1_backend,
    _infer_funasr_time_scale,
    _normalize_funasr_confidence,
    _normalize_funasr_language,
    _recognize_lyrics_with_funasr,
    _separate_with_demucs,
    _split_text_for_timestamp,
)
# 项目内模块：导出歌词处理函数
from music_video_pipeline.modules.module_a.lyrics import (
    _attach_lyrics_to_segments,
    _build_visual_lyric_units,
    _build_visual_unit_from_token_slice,
    _can_merge_visual_units,
    _clean_lyric_units,
    _is_obvious_noise_text,
    _is_placeholder_text,
    _is_probable_vocal_presence,
    _is_vocalise_text,
    _merge_adjacent_visual_lyric_units,
    _merge_two_visual_units,
    _normalize_lyric_segment_policy,
    _normalize_non_negative_threshold,
    _normalize_positive_threshold,
    _normalize_token_units,
    _select_best_overlap_segment,
    _split_sentence_unit_for_adaptive_policy,
)
# 项目内模块：导出分段函数
from music_video_pipeline.modules.module_a.segmentation import (
    _build_segments_with_lyric_priority,
    _build_small_segments,
    _merge_short_vocal_non_lyric_ranges,
    _normalize_segment_ranges,
    _select_small_timestamps,
    _split_instrumental_range_once_by_energy,
    _split_range_by_rhythm,
)
# 项目内模块：导出时间与能量工具函数
from music_video_pipeline.modules.module_a.timing_energy import (
    _build_beats_from_segments,
    _build_beats_from_timestamps,
    _build_energy_features,
    _build_fallback_big_segments,
    _build_fallback_energy_features,
    _build_grid_timestamps,
    _clamp_time,
    _find_big_segment,
    _normalize_timestamp_list,
    _rms_delta_at,
    _rms_value_at,
    _round_time,
    _slice_rms,
    _snap_to_nearest_beat,
)

__all__ = [
    "run_module_a",
    "_run_real_pipeline",
    "_run_fallback_pipeline",
    "_probe_audio_duration",
    "_separate_with_demucs",
    "_detect_big_segments_with_allin1",
    "_import_allin1_backend",
    "_extract_acoustic_candidates_with_librosa",
    "_recognize_lyrics_with_funasr",
    "_normalize_funasr_language",
    "_extract_funasr_records",
    "_infer_funasr_time_scale",
    "_build_lyric_units_from_sentence_info",
    "_build_lyric_units_from_timestamp",
    "_build_sentence_unit_from_tokens",
    "_build_token_units_from_timestamp",
    "_split_text_for_timestamp",
    "_normalize_funasr_confidence",
    "_select_small_timestamps",
    "_build_small_segments",
    "_build_segments_with_lyric_priority",
    "_split_instrumental_range_once_by_energy",
    "_split_range_by_rhythm",
    "_normalize_segment_ranges",
    "_merge_short_vocal_non_lyric_ranges",
    "_clean_lyric_units",
    "_build_visual_lyric_units",
    "_normalize_lyric_segment_policy",
    "_normalize_positive_threshold",
    "_normalize_non_negative_threshold",
    "_split_sentence_unit_for_adaptive_policy",
    "_build_visual_unit_from_token_slice",
    "_merge_adjacent_visual_lyric_units",
    "_can_merge_visual_units",
    "_merge_two_visual_units",
    "_normalize_token_units",
    "_is_placeholder_text",
    "_is_obvious_noise_text",
    "_is_probable_vocal_presence",
    "_is_vocalise_text",
    "_attach_lyrics_to_segments",
    "_select_best_overlap_segment",
    "_build_energy_features",
    "_build_beats_from_timestamps",
    "_build_beats_from_segments",
    "_build_fallback_big_segments",
    "_build_fallback_energy_features",
    "_build_grid_timestamps",
    "_normalize_timestamp_list",
    "_find_big_segment",
    "_rms_value_at",
    "_rms_delta_at",
    "_slice_rms",
    "_snap_to_nearest_beat",
    "_clamp_time",
    "_round_time",
]
