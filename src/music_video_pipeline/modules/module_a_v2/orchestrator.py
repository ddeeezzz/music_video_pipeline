"""
文件用途：实现模块A V2单一路径编排入口。
核心流程：按“感知层 -> 算法层 -> 契约校验 -> 输出落盘”执行。
输入输出：输入 RuntimeContext，输出 module_a_output.json 路径。
依赖说明：依赖 module_a_v2 子模块与公共上下文/IO能力。
维护说明：本文件仅负责调度与失败语义，不承载分段算法细节。
"""

# 标准库：用于 JSON 读取
import json
# 标准库：用于路径类型提示
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON落盘工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：类型契约校验
from music_video_pipeline.types import validate_module_a_output
# 项目内模块：V2别名映射
from music_video_pipeline.modules.module_a_v2.utils.alias_map import build_module_a_v2_alias_map
# 项目内模块：V2媒体时长探测
from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration
# 项目内模块：V2算法层
from music_video_pipeline.modules.module_a_v2.algorithm import AlgorithmBundle, run_algorithm_stage
# 项目内模块：V2产物路径管理
from music_video_pipeline.modules.module_a_v2.artifacts import ModuleAV2Artifacts, build_module_a_v2_artifacts
# 项目内模块：V2感知层
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle, run_perception_stage
# 项目内模块：V2歌词来源统一解析（轻量重跑使用）
from music_video_pipeline.modules.module_a_v2.lyrics_resolver import resolve_lyrics_with_priority
# 项目内模块：V2可视化聚合与渲染
from music_video_pipeline.modules.module_a_v2.visualization import collect_visualization_payload, render_visualization_html


# 常量：模块A V2自动可视化默认输出文件名模板（task_id 前缀）
AUTO_VISUALIZATION_FILE_TEMPLATE = "{task_id}_module_a_v2_visualization.html"
# 常量：模块A V2自动可视化默认音频处理模式
AUTO_VISUALIZATION_AUDIO_MODE = "copy"


def _run_perception_stage(
    audio_path: Path,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    context: RuntimeContext,
) -> PerceptionBundle:
    """
    功能说明：封装感知层调用，便于测试替换与分层调度。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: 产物路径对象。
    - context: 运行上下文。
    返回值：
    - PerceptionBundle: 感知层统一输出。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return run_perception_stage(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        artifacts_dir=context.artifacts_dir,
        device=context.config.module_a.device,
        demucs_model=context.config.module_a.demucs_model,
        funasr_model=context.config.module_a.funasr_model,
        funasr_language=context.config.module_a.funasr_language,
        skip_funasr_when_vocals_silent=context.config.module_a.skip_funasr_when_vocals_silent,
        vocal_skip_peak_rms_threshold=context.config.module_a.vocal_skip_peak_rms_threshold,
        vocal_skip_active_ratio_threshold=context.config.module_a.vocal_skip_active_ratio_threshold,
        fpcalc_bin=context.config.module_a.fpcalc_bin,
        acoustid_api_key_file=context.config.module_a.acoustid_api_key_file,
        lyrics_enable_fingerprint_lookup=context.config.module_a.lyrics_enable_fingerprint_lookup,
        logger=context.logger,
    )


def _run_algorithm_stage(
    perception: PerceptionBundle,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    context: RuntimeContext | None = None,
    instrumental_labels: list[str] | None = None,
    merge_gap_seconds: float = 0.25,
    vocal_energy_enter_quantile: float = 0.70,
    vocal_energy_exit_quantile: float = 0.45,
    mid_segment_min_duration_seconds: float = 0.8,
    short_vocal_non_lyric_merge_seconds: float = 1.2,
    visual_lead_seconds: float = 0.06,
    lyric_boundary_near_anchor_seconds: float = 3.0,
    content_role_tiny_merge_bars: float = 0.8,
    long_lyric_resplit_max_bars: float = 2.0,
    long_other_split_min_bars: float = 1.0,
    major_split_step_bars: float = 2.5,
    logger: Any | None = None,
    skip_lyrics_cleaner: bool = False,
) -> AlgorithmBundle:
    """
    功能说明：封装算法层调用，便于测试替换与分层调度。
    参数说明：
    - perception: 感知层产物。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: 产物路径对象。
    - context: 运行上下文。
    返回值：
    - AlgorithmBundle: 算法层统一输出。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return run_algorithm_stage(
        perception=perception,
        duration_seconds=duration_seconds,
        instrumental_labels=instrumental_labels or (context.config.module_a.instrumental_labels if context else []),
        merge_gap_seconds=merge_gap_seconds,
        vocal_energy_enter_quantile=vocal_energy_enter_quantile,
        vocal_energy_exit_quantile=vocal_energy_exit_quantile,
        mid_segment_min_duration_seconds=mid_segment_min_duration_seconds,
        short_vocal_non_lyric_merge_seconds=short_vocal_non_lyric_merge_seconds,
        visual_lead_seconds=visual_lead_seconds,
        lyric_boundary_near_anchor_seconds=lyric_boundary_near_anchor_seconds,
        content_role_tiny_merge_bars=content_role_tiny_merge_bars,
        long_lyric_resplit_max_bars=long_lyric_resplit_max_bars,
        long_other_split_min_bars=long_other_split_min_bars,
        major_split_step_bars=major_split_step_bars,
        artifacts=artifacts,
        skip_lyrics_cleaner=skip_lyrics_cleaner,
        logger=logger or (context.logger if context else __import__('logging').getLogger('run_algorithm_stage')),
    )


def _render_module_a_v2_visualization(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块A V2自动可视化渲染并返回HTML路径。
    参数说明：
    - context: 运行时上下文对象。
    返回值：
    - Path: 可视化HTML路径。
    异常说明：可视化聚合或渲染失败时向上抛出异常。
    边界条件：输出覆盖同名历史文件，实现“每次运行都重绘”。
    """
    output_name = AUTO_VISUALIZATION_FILE_TEMPLATE.format(task_id=context.task_id)
    output_html_path = context.task_dir / output_name
    payload = collect_visualization_payload(task_dir=context.task_dir)
    return render_visualization_html(
        payload=payload,
        output_html_path=output_html_path,
        audio_mode=AUTO_VISUALIZATION_AUDIO_MODE,
    )


def run_module_a_v2(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块A V2并产出标准JSON。
    参数说明：
    - context: 运行时上下文对象。
    返回值：
    - Path: 模块A输出JSON文件路径。
    异常说明：关键产物缺失时抛 RuntimeError。
    边界条件：lyric_units 允许为空，其他关键字段必须有效。
    """
    context.logger.info("模块A V2开始执行，task_id=%s，输入音频=%s", context.task_id, context.audio_path)
    duration_seconds = probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )
    artifacts = build_module_a_v2_artifacts(context.artifacts_dir / "module_a_work_v2")
    perception = _run_perception_stage(
        audio_path=context.audio_path,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        context=context,
    )
    analysis_bundle = _run_algorithm_stage(
        perception=perception,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        context=context,
    )

    if not analysis_bundle.big_segments:
        raise RuntimeError("模块A V2失败：big_segments 为空")
    if not analysis_bundle.segments:
        raise RuntimeError("模块A V2失败：segments 为空")
    if len(analysis_bundle.beats) < 2:
        raise RuntimeError("模块A V2失败：beats 少于2个")
    if not analysis_bundle.energy_features:
        raise RuntimeError("模块A V2失败：energy_features 为空")

    analysis_data = {
        "big_segments_stage1": analysis_bundle.big_segments_stage1,
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
    }
    alias_map = build_module_a_v2_alias_map(mode="v2_single", analysis_data=analysis_data)
    output_data = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
        "alias_map": alias_map,
    }
    validate_module_a_output(output_data)
    output_path = context.artifacts_dir / "module_a_output.json"
    write_json(output_path, output_data)
    try:
        visualization_path = _render_module_a_v2_visualization(context=context)
        context.logger.info("模块A V2自动可视化完成，task_id=%s，输出=%s", context.task_id, visualization_path)
    except Exception as error:  # noqa: BLE001
        context.logger.warning("模块A V2自动可视化失败，已忽略，task_id=%s，错误=%s", context.task_id, error)
    context.logger.info("模块A V2执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


def _safe_read_json(file_path: Path, logger, name: str) -> dict[str, Any]:
    """
    功能说明：安全读取 JSON 文件，文件缺失或解析失败时返回空字典。
    参数说明：
    - file_path: JSON 文件路径。
    - logger: 日志对象。
    - name: 缓存名称（用于日志可读性）。
    返回值：
    - dict[str, Any]: 解析后的字典；失败时返回空字典。
    """
    try:
        if file_path.exists() and file_path.is_file():
            return dict(json.loads(file_path.read_text(encoding="utf-8")))
        logger.warning("模块A V2-轻量重跑：%s 缓存文件不存在，path=%s", name, file_path)
        return {}
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-轻量重跑：%s 缓存读取失败，path=%s，错误=%s", name, file_path, error)
        return {}


def _load_cached_perception_data(artifacts: ModuleAV2Artifacts, logger) -> dict[str, Any]:
    """
    功能说明：从磁盘读取缓存的感知层数据（Allin1 大段/节拍 + Librosa 特征），用于轻量重跑。
    参数说明：
    - artifacts: 模块A V2产物路径对象。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: 缓存的感知数据字典。
    异常说明：
    - RuntimeError: Allin1 或 Librosa 缓存缺失时抛出，提示应先执行完整模块A。
    """
    allin1_data = _safe_read_json(artifacts.perception_model_allin1_raw_response_path, logger, "allin1_raw_response")
    raw_segments = list(allin1_data.get("segments", []))
    big_segments_stage1: list[dict[str, Any]] = []
    for seg in raw_segments:
        st = float(seg.get("start", 0.0))
        et = float(seg.get("end", 0.0))
        if et <= st:
            continue
        big_segments_stage1.append({
            "segment_id": f"big_{len(big_segments_stage1) + 1:03d}",
            "start_time": st,
            "end_time": et,
            "label": str(seg.get("label", "unknown")).strip() or "unknown",
        })
    beat_candidates = [float(t) for t in allin1_data.get("beat_times", [])]
    raw_beats = list(allin1_data.get("beats", []))
    raw_positions = list(allin1_data.get("beat_positions", []))
    beats = []
    for i, b in enumerate(raw_beats):
        if isinstance(b, (int, float)):
            beat_type = "major" if i % 4 == 0 else "minor"
            if i < len(raw_positions) and raw_positions[i] is not None:
                beat_type = "major" if int(raw_positions[i]) == 1 else "minor"
            beats.append({"time": round(float(b), 3), "type": beat_type, "source": "allin1"})
        else:
            beats.append(dict(b))
    if not big_segments_stage1 or not beats:
        raise RuntimeError(
            "模块A V2轻量重跑：Allin1 缓存缺失（big_segments 或 beats 为空），请先执行一次完整模块A。"
        )

    accomp_data = _safe_read_json(artifacts.perception_signal_librosa_accompaniment_path, logger, "accompaniment")
    onset_candidates = [float(t) for t in accomp_data.get("onset_candidates", [])]
    rms_times = [float(t) for t in accomp_data.get("rms_times", [])]
    rms_values = [float(v) for v in accomp_data.get("rms_values", [])]
    onset_points = list(accomp_data.get("onset_points", []))
    chroma_points = list(accomp_data.get("chroma_points", []))
    accomp_f0 = list(accomp_data.get("f0_points_no_vocals", []))
    if not rms_times or not rms_values:
        raise RuntimeError(
            "模块A V2轻量重跑：Librosa 伴奏缓存缺失（rms_times 或 rms_values 为空），请先执行一次完整模块A。"
        )

    vocal_data = _safe_read_json(artifacts.perception_signal_librosa_vocal_candidates_path, logger, "vocal_candidates")
    vocal_onset_candidates = [float(t) for t in vocal_data.get("onset_candidates", [])]
    vocal_rms_times = [float(t) for t in vocal_data.get("rms_times", [])]
    vocal_rms_values = [float(v) for v in vocal_data.get("rms_values", [])]
    vocal_f0 = list(vocal_data.get("f0_points_vocals", []))

    precheck_data = _safe_read_json(artifacts.perception_signal_librosa_vocal_precheck_path, logger, "vocal_precheck")
    funasr_skipped = bool(precheck_data.get("should_skip_funasr", False))

    return {
        "big_segments_stage1": big_segments_stage1,
        "beat_candidates": beat_candidates,
        "beats": beats,
        "onset_candidates": onset_candidates,
        "rms_times": rms_times,
        "rms_values": rms_values,
        "vocal_onset_candidates": vocal_onset_candidates,
        "vocal_rms_times": vocal_rms_times,
        "vocal_rms_values": vocal_rms_values,
        "funasr_skipped_for_silent_vocals": funasr_skipped,
        "onset_points": onset_points,
        "accompaniment_chroma_points": chroma_points,
        "vocal_f0_points": vocal_f0,
        "accompaniment_f0_points": accomp_f0,
    }


def run_module_a_v2_lyrics_only(context: RuntimeContext) -> Path:
    """
    功能说明：轻量重跑模块A V2——跳过信号处理（Demucs/Allin1/Librosa/FunASR），
              直接从缓存感知数据 + 新歌词结果执行算法层。
    参数说明：
    - context: 运行时上下文（需已启用联网歌词状态）。
    返回值：
    - Path: 模块A输出JSON文件路径。
    异常说明：
    - RuntimeError: 缓存的感知数据缺失或关键输出为空时抛出。
    边界条件：底层 resolve_lyrics_with_priority 会优先拾取已启用的联网歌词；
              感知缓存来源于上一次完整模块A执行产物。
    """
    context.logger.info("模块A V2轻量重跑开始，task_id=%s，跳过信号处理层", context.task_id)
    duration_seconds = probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )
    artifacts = build_module_a_v2_artifacts(context.artifacts_dir / "module_a_work_v2")
    cached = _load_cached_perception_data(artifacts=artifacts, logger=context.logger)

    lyric_result = resolve_lyrics_with_priority(
        audio_path=context.audio_path,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        artifacts_dir=context.artifacts_dir,
        logger=context.logger,
        fpcalc_bin=str(context.config.module_a.fpcalc_bin),
        acoustid_api_key_file=str(context.config.module_a.acoustid_api_key_file),
        enable_fingerprint_lookup=False,
        funasr_fallback_runner=lambda: None,
    )

    perception = PerceptionBundle(
        big_segments_stage1=cached["big_segments_stage1"],
        beat_candidates=cached["beat_candidates"],
        beats=cached["beats"],
        lyric_sentence_units=list(lyric_result.get("lyric_sentence_units", [])),
        sentence_split_stats=dict(lyric_result.get("sentence_split_stats", {})),
        vocals_path=Path(""),
        no_vocals_path=Path(""),
        demucs_stems={},
        onset_candidates=cached["onset_candidates"],
        rms_times=cached["rms_times"],
        rms_values=cached["rms_values"],
        vocal_onset_candidates=cached["vocal_onset_candidates"],
        vocal_rms_times=cached["vocal_rms_times"],
        vocal_rms_values=cached["vocal_rms_values"],
        funasr_skipped_for_silent_vocals=cached["funasr_skipped_for_silent_vocals"],
        onset_points=cached["onset_points"],
        accompaniment_chroma_points=cached["accompaniment_chroma_points"],
        vocal_f0_points=cached["vocal_f0_points"],
        accompaniment_f0_points=cached["accompaniment_f0_points"],
    )

    analysis_bundle = _run_algorithm_stage(
        perception=perception,
        duration_seconds=duration_seconds,
        instrumental_labels=context.config.module_a.instrumental_labels,
        merge_gap_seconds=context.config.module_a.merge_gap_seconds,
        vocal_energy_enter_quantile=context.config.module_a.vocal_energy_enter_quantile,
        vocal_energy_exit_quantile=context.config.module_a.vocal_energy_exit_quantile,
        mid_segment_min_duration_seconds=context.config.module_a.mid_segment_min_duration_seconds,
        short_vocal_non_lyric_merge_seconds=context.config.module_a.short_vocal_non_lyric_merge_seconds,
        visual_lead_seconds=context.config.module_a.visual_lead_seconds,
        lyric_boundary_near_anchor_seconds=context.config.module_a.lyric_boundary_near_anchor_seconds,
        content_role_tiny_merge_bars=context.config.module_a.content_role_tiny_merge_bars,
        long_lyric_resplit_max_bars=context.config.module_a.long_lyric_resplit_max_bars,
        long_other_split_min_bars=context.config.module_a.long_other_split_min_bars,
        major_split_step_bars=context.config.module_a.major_split_step_bars,
        artifacts=artifacts,
        skip_lyrics_cleaner=True,
        logger=context.logger,
    )

    if not analysis_bundle.big_segments:
        raise RuntimeError("模块A V2轻量重跑失败：big_segments 为空")
    if not analysis_bundle.segments:
        raise RuntimeError("模块A V2轻量重跑失败：segments 为空")
    if len(analysis_bundle.beats) < 2:
        raise RuntimeError("模块A V2轻量重跑失败：beats 少于2个")
    if not analysis_bundle.energy_features:
        raise RuntimeError("模块A V2轻量重跑失败：energy_features 为空")

    analysis_data = {
        "big_segments_stage1": analysis_bundle.big_segments_stage1,
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
    }
    alias_map = build_module_a_v2_alias_map(mode="v2_single", analysis_data=analysis_data)
    output_data = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
        "alias_map": alias_map,
    }
    validate_module_a_output(output_data)
    output_path = context.artifacts_dir / "module_a_output.json"
    write_json(output_path, output_data)
    try:
        visualization_path = _render_module_a_v2_visualization(context=context)
        context.logger.info("模块A V2轻量重跑可视化完成，task_id=%s，输出=%s", context.task_id, visualization_path)
    except Exception as error:  # noqa: BLE001
        context.logger.warning("模块A V2轻量重跑可视化失败，已忽略，task_id=%s，错误=%s", context.task_id, error)
    context.logger.info("模块A V2轻量重跑完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
