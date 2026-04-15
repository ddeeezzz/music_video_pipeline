"""
文件用途：实现模块A V2算法层（窗口化分段 + A1时间轴求解 + 最终挂载）。
核心流程：清洗分句 -> 窗口构造/分类/合并 -> A0->A1 -> 近锚点裁决 -> 输出最终S/BIG。
输入输出：输入 PerceptionBundle 与配置参数，输出 AlgorithmBundle。
依赖说明：依赖 v2 内容角色编排与歌词/能量处理能力。
维护说明：本文件仅负责编排，不堆叠底层规则细节。
"""

# 标准库：用于数据结构定义
from dataclasses import dataclass

# 项目内模块：V2内容角色与时间轴统一编排
from music_video_pipeline.modules.module_a_v2.content_roles import (
    DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS,
    DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS,
    apply_content_role_pipeline,
)
# 项目内模块：V2能量特征构建
from music_video_pipeline.modules.module_a_v2.energy.features import build_energy_features
# 项目内模块：V2歌词挂载
from music_video_pipeline.modules.module_a_v2.lyrics.attachment import attach_lyrics_to_segments
# 项目内模块：V2歌词清洗
from music_video_pipeline.modules.module_a_v2.lyrics.cleaner import clean_lyric_units
# 项目内模块：V2产物管理
from music_video_pipeline.modules.module_a_v2.artifacts import ModuleAV2Artifacts, dump_json_artifact
# 项目内模块：V2感知层产物
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle

# 常量：V2歌词挂载启用“跨段小残片归后段”策略
V2_PREFER_NEXT_SEGMENT_FOR_SMALL_BOUNDARY_TOKEN = True
# 常量：V2歌词挂载“前段小残片归后段”阈值（秒）
V2_SMALL_BOUNDARY_TOKEN_FRAGMENT_SECONDS = 0.021


@dataclass(frozen=True)
class AlgorithmBundle:
    """
    功能说明：封装算法层输出给编排层的统一结果。
    参数说明：各字段对应模块A最终输出与关键中间态。
    返回值：不适用。
    异常说明：不适用。
    边界条件：lyric_units 允许为空列表。
    """

    big_segments_stage1: list[dict]
    big_segments: list[dict]
    segments: list[dict]
    beats: list[dict]
    lyric_units: list[dict]
    energy_features: list[dict]


def run_algorithm_stage(
    perception: PerceptionBundle,
    duration_seconds: float,
    instrumental_labels: list[str],
    merge_gap_seconds: float,
    lyric_head_offset_seconds: float,
    lyric_boundary_near_anchor_seconds: float,
    content_role_tiny_merge_bars: float,
    artifacts: ModuleAV2Artifacts,
    logger,
) -> AlgorithmBundle:
    """
    功能说明：执行模块A V2算法层并写入算法阶段中间产物。
    参数说明：
    - perception: 感知层统一输出。
    - duration_seconds: 音频总时长（秒）。
    - instrumental_labels: 器乐标签集合（用于歌词清洗）。
    - merge_gap_seconds: 分句阈值缺失时的兼容兜底（秒）。
    - lyric_head_offset_seconds: 句首锚点后移量（秒）。
    - lyric_boundary_near_anchor_seconds: 近锚点冲突判定阈值（秒）。
    - content_role_tiny_merge_bars: 内容角色tiny并段阈值（小节）。
    - artifacts: V2产物路径对象。
    - logger: 日志记录器。
    返回值：
    - AlgorithmBundle: 算法层统一输出。
    异常说明：关键输出为空时抛错，由上层统一处理。
    边界条件：歌词可空时仍需返回可用 segments/energy_features。
    """
    big_segments_stage1 = list(perception.big_segments_stage1)
    if not big_segments_stage1:
        raise RuntimeError("模块A V2算法层失败：A0大段为空")

    sentence_units: list[dict] = []
    if perception.lyric_sentence_units:
        sentence_units = clean_lyric_units(
            lyric_units_raw=perception.lyric_sentence_units,
            big_segments=big_segments_stage1,
            instrumental_labels=instrumental_labels,
            logger=logger,
        )
    if sentence_units:
        logger.info("模块A V2算法层已接收分句结果，句数=%s", len(sentence_units))
    else:
        logger.info("模块A V2算法层分句为空，将仅生成其他窗口")

    safe_near_anchor_seconds = DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS
    try:
        safe_near_anchor_seconds = float(lyric_boundary_near_anchor_seconds)
    except Exception:  # noqa: BLE001
        logger.warning(
            "模块A V2-lyric_boundary_near_anchor_seconds 配置非法，已回退默认值=%s，原始值=%s",
            DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS,
            lyric_boundary_near_anchor_seconds,
        )
        safe_near_anchor_seconds = DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS
    if safe_near_anchor_seconds <= 0.0:
        logger.warning(
            "模块A V2-lyric_boundary_near_anchor_seconds 必须大于0，已回退默认值=%s，原始值=%s",
            DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS,
            lyric_boundary_near_anchor_seconds,
        )
        safe_near_anchor_seconds = DEFAULT_LYRIC_BOUNDARY_NEAR_ANCHOR_SECONDS

    safe_content_role_tiny_merge_bars = DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS
    try:
        safe_content_role_tiny_merge_bars = float(content_role_tiny_merge_bars)
    except Exception:  # noqa: BLE001
        logger.warning(
            "模块A V2-content_role_tiny_merge_bars 配置非法，已回退默认值=%s，原始值=%s",
            DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS,
            content_role_tiny_merge_bars,
        )
        safe_content_role_tiny_merge_bars = DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS
    if safe_content_role_tiny_merge_bars <= 0.0:
        logger.warning(
            "模块A V2-content_role_tiny_merge_bars 必须大于0，已回退默认值=%s，原始值=%s",
            DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS,
            content_role_tiny_merge_bars,
        )
        safe_content_role_tiny_merge_bars = DEFAULT_CONTENT_ROLE_TINY_MERGE_BARS

    # 当感知层分句统计缺失时，回退 merge_gap_seconds 作为兼容兜底。
    sentence_split_stats = dict(perception.sentence_split_stats)
    if not sentence_split_stats:
        sentence_split_stats = {"dynamic_gap_threshold_seconds": float(max(0.04, merge_gap_seconds))}

    pipeline_result = apply_content_role_pipeline(
        big_segments_stage1=big_segments_stage1,
        sentence_units=sentence_units,
        sentence_split_stats=sentence_split_stats,
        beat_candidates=perception.beat_candidates,
        beats=perception.beats,
        vocal_rms_times=perception.vocal_rms_times,
        vocal_rms_values=perception.vocal_rms_values,
        accompaniment_rms_times=perception.rms_times,
        accompaniment_rms_values=perception.rms_values,
        tiny_merge_bars=safe_content_role_tiny_merge_bars,
        lyric_head_offset_seconds=lyric_head_offset_seconds,
        near_anchor_seconds=safe_near_anchor_seconds,
        duration_seconds=duration_seconds,
        onset_points=perception.onset_points,
        accompaniment_chroma_points=perception.accompaniment_chroma_points,
        vocal_f0_points=perception.vocal_f0_points,
        accompaniment_f0_points=perception.accompaniment_f0_points,
    )

    windows_raw = list(pipeline_result.get("windows_raw", []))
    windows_classified = list(pipeline_result.get("windows_classified", []))
    windows_merged = list(pipeline_result.get("windows_merged", []))
    small_timestamps = list(pipeline_result.get("small_timestamps", []))
    big_segments = list(pipeline_result.get("big_segments_a1", []))
    segments = list(pipeline_result.get("segments_final", []))
    boundary_conflict_resolved = dict(pipeline_result.get("boundary_conflict_resolved", {}))
    big_boundary_moves = dict(pipeline_result.get("big_boundary_moves", {}))

    if not big_segments:
        raise RuntimeError("模块A V2算法层失败：A1大段为空")
    if not segments:
        raise RuntimeError("模块A V2算法层失败：最终S段为空")

    lyric_units = (
        attach_lyrics_to_segments(
            sentence_units,
            segments,
            prefer_next_segment_for_small_boundary_token=V2_PREFER_NEXT_SEGMENT_FOR_SMALL_BOUNDARY_TOKEN,
            small_boundary_token_fragment_seconds=V2_SMALL_BOUNDARY_TOKEN_FRAGMENT_SECONDS,
        )
        if sentence_units
        else []
    )
    energy_features = build_energy_features(segments, perception.rms_times, perception.rms_values, perception.beat_candidates)
    if not energy_features:
        raise RuntimeError("模块A V2算法层失败：能量特征为空")

    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_big_a0_path,
        payload=big_segments_stage1,
        logger=logger,
        artifact_name="stage_big_a0",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_big_a1_path,
        payload=big_segments,
        logger=logger,
        artifact_name="stage_big_a1",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_lyric_sentence_units_cleaned_path,
        payload=sentence_units,
        logger=logger,
        artifact_name="stage_lyric_sentence_units_cleaned",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_small_timestamps_path,
        payload=small_timestamps,
        logger=logger,
        artifact_name="stage_small_timestamps",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_boundary_conflict_resolved_path,
        payload=boundary_conflict_resolved,
        logger=logger,
        artifact_name="stage_boundary_conflict_resolved",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_timeline_stage_big_boundary_moves_path,
        payload=big_boundary_moves,
        logger=logger,
        artifact_name="stage_big_boundary_moves",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_window_stage_windows_raw_path,
        payload=windows_raw,
        logger=logger,
        artifact_name="stage_windows_raw",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_window_stage_windows_classified_path,
        payload=windows_classified,
        logger=logger,
        artifact_name="stage_windows_classified",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_window_stage_windows_merged_path,
        payload=windows_merged,
        logger=logger,
        artifact_name="stage_windows_merged",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_final_stage_segments_final_path,
        payload=segments,
        logger=logger,
        artifact_name="stage_segments_final",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_final_stage_lyric_attached_path,
        payload=lyric_units,
        logger=logger,
        artifact_name="stage_lyric_attached",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_final_stage_energy_path,
        payload=energy_features,
        logger=logger,
        artifact_name="stage_energy",
    )
    dump_json_artifact(
        output_path=artifacts.algorithm_final_analysis_data_path,
        payload={
            "big_segments_stage1": big_segments_stage1,
            "big_segments": big_segments,
            "segments": segments,
            "beats": perception.beats,
            "lyric_units": lyric_units,
            "energy_features": energy_features,
            "windows_raw": windows_raw,
            "windows_classified": windows_classified,
            "windows_merged": windows_merged,
            "boundary_conflict_resolved": boundary_conflict_resolved,
            "big_boundary_moves": big_boundary_moves,
            "sentence_split_stats": sentence_split_stats,
        },
        logger=logger,
        artifact_name="final_analysis_data",
    )

    return AlgorithmBundle(
        big_segments_stage1=big_segments_stage1,
        big_segments=big_segments,
        segments=segments,
        beats=perception.beats,
        lyric_units=lyric_units,
        energy_features=energy_features,
    )
