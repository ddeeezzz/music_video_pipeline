"""
文件用途：实现模块A V2感知层（模型感知 + 信号感知）。
核心流程：执行 Demucs/Allin1/FunASR/Librosa 并返回统一感知产物。
输入输出：输入音频与运行参数，输出 PerceptionBundle。
依赖说明：依赖 v2 感知后端子模块与 artifacts 落盘工具。
维护说明：本层只负责“感知与证据落盘”，不承担分段决策。
"""

# 标准库：用于数据结构定义
from dataclasses import dataclass, field
# 标准库：用于并行执行双轨Librosa提取
from concurrent.futures import ThreadPoolExecutor
# 标准库：用于路径类型
from pathlib import Path
# 标准库：用于耗时统计
from time import perf_counter, time_ns

# 项目内模块：V2 Allin1 后端
from music_video_pipeline.modules.module_a_v2.backends.allin1 import analyze_with_allin1
# 项目内模块：V2 Demucs 后端
from music_video_pipeline.modules.module_a_v2.backends.demucs import prepare_stems_with_allin1_demucs
# 项目内模块：V2 FunASR 歌词重建
from music_video_pipeline.modules.module_a_v2.funasr_lyrics import recognize_lyrics_with_funasr_v2
# 项目内模块：V2 Librosa 后端
from music_video_pipeline.modules.module_a_v2.backends.librosa import extract_acoustic_candidates_with_librosa
# 项目内模块：V2产物管理
from music_video_pipeline.modules.module_a_v2.artifacts import (
    ModuleAV2Artifacts,
    dump_json_artifact,
)


@dataclass(frozen=True)
class PerceptionBundle:
    """
    功能说明：封装感知层输出给算法层的统一输入。
    参数说明：各字段为标准化感知结果。
    返回值：不适用。
    异常说明：不适用。
    边界条件：lyric_sentence_units 允许为空列表。
    """

    big_segments_stage1: list[dict]
    beat_candidates: list[float]
    beats: list[dict]
    lyric_sentence_units: list[dict]
    sentence_split_stats: dict
    vocals_path: Path
    no_vocals_path: Path
    demucs_stems: dict[str, Path]
    onset_candidates: list[float]
    rms_times: list[float]
    rms_values: list[float]
    vocal_onset_candidates: list[float]
    vocal_rms_times: list[float]
    vocal_rms_values: list[float]
    funasr_skipped_for_silent_vocals: bool
    onset_points: list[dict] = field(default_factory=list)
    accompaniment_chroma_points: list[dict] = field(default_factory=list)
    vocal_f0_points: list[dict] = field(default_factory=list)
    accompaniment_f0_points: list[dict] = field(default_factory=list)


def _should_skip_funasr_by_vocal_energy(
    vocal_rms_values: list[float],
    peak_threshold: float,
    active_ratio_threshold: float,
) -> tuple[bool, float, float]:
    """
    功能说明：根据人声音轨 RMS 能量判断是否可跳过 FunASR。
    参数说明：
    - vocal_rms_values: 人声音轨 RMS 序列。
    - peak_threshold: 峰值阈值（RMS）。
    - active_ratio_threshold: 活跃帧占比阈值（0~1）。
    返回值：
    - tuple[bool, float, float]: `(should_skip, peak_rms, active_ratio)`。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当 RMS 序列为空时默认不跳过。
    """
    if not vocal_rms_values:
        return False, 0.0, 0.0

    safe_peak_threshold = max(1e-6, float(peak_threshold))
    safe_active_ratio_threshold = max(0.0, min(1.0, float(active_ratio_threshold)))
    safe_values = [max(0.0, float(item)) for item in vocal_rms_values]
    peak_rms = max(safe_values) if safe_values else 0.0
    active_gate = safe_peak_threshold * 0.5
    active_count = sum(1 for item in safe_values if item >= active_gate)
    active_ratio = active_count / max(1, len(safe_values))
    should_skip = peak_rms <= safe_peak_threshold and active_ratio <= safe_active_ratio_threshold
    return should_skip, peak_rms, active_ratio


def run_perception_stage(
    audio_path: Path,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    device: str,
    demucs_model: str,
    funasr_model: str,
    funasr_language: str,
    skip_funasr_when_vocals_silent: bool,
    vocal_skip_peak_rms_threshold: float,
    vocal_skip_active_ratio_threshold: float,
    logger,
) -> PerceptionBundle:
    """
    功能说明：执行模块A V2感知层并落盘中间证据产物。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: V2产物路径对象。
    - device: 推理设备标识。
    - demucs_model: Demucs 模型名称。
    - funasr_model: FunASR 模型名称。
    - funasr_language: FunASR 语言策略。
    - skip_funasr_when_vocals_silent: 是否启用低能量跳过策略。
    - vocal_skip_peak_rms_threshold: 峰值 RMS 跳过阈值。
    - vocal_skip_active_ratio_threshold: 活跃占比跳过阈值。
    - logger: 日志记录器。
    返回值：
    - PerceptionBundle: 感知层统一输出。
    异常说明：关键模型调用失败时抛错，由上层统一处理。
    边界条件：FunASR失败或跳过时 lyric_sentence_units 允许为空。
    """
    logger.info("模块A V2感知层开始，输入=%s", audio_path)

    vocals_path, no_vocals_path, stems_input = prepare_stems_with_allin1_demucs(
        audio_path=audio_path,
        output_dir=artifacts.perception_model_demucs_runtime_dir,
        device=device,
        model_name=demucs_model,
        logger=logger,
    )
    demucs_stems = {
        "vocals": vocals_path,
        "bass": Path(str(stems_input["bass"])),
        "drums": Path(str(stems_input["drums"])),
        "other": Path(str(stems_input["other"])),
        "no_vocals": no_vocals_path,
    }

    precheck_rms_times: list[float] = []
    precheck_rms_values: list[float] = []
    funasr_skipped_for_silent_vocals = False
    allin1_work_dir = artifacts.perception_model_allin1_runtime_dir / f"run_{time_ns()}"
    allin1_work_dir.mkdir(parents=True, exist_ok=True)
    stage_started_at = perf_counter()
    timing_marks: dict[str, float | None] = {
        "librosa_vocals_start": None,
        "librosa_vocals_end": None,
        "librosa_no_vocals_start": None,
        "librosa_no_vocals_end": None,
        "librosa_parallel_start": None,
        "librosa_parallel_end": None,
        "precheck_start": None,
        "precheck_end": None,
        "funasr_start": None,
        "funasr_end": None,
    }

    def _safe_duration(start_key: str, end_key: str) -> float:
        """
        功能说明：计算两个打点之间的耗时，缺失时返回0。
        参数说明：
        - start_key: 起点键名。
        - end_key: 终点键名。
        返回值：
        - float: 非负耗时（秒）。
        异常说明：无。
        边界条件：任一打点缺失时返回0。
        """
        start_at = timing_marks.get(start_key)
        end_at = timing_marks.get(end_key)
        if start_at is None or end_at is None:
            return 0.0
        return max(0.0, float(end_at) - float(start_at))

    def _mark_values(mark_keys: list[str]) -> list[float]:
        """
        功能说明：读取一组打点键对应的有效时间值列表。
        参数说明：
        - mark_keys: 打点键名列表。
        返回值：
        - list[float]: 有效打点时间（秒）数组。
        异常说明：无。
        边界条件：缺失键会被忽略。
        """
        return [float(mark_value) for mark_key in mark_keys if (mark_value := timing_marks.get(mark_key)) is not None]

    def _safe_span(start_keys: list[str], end_keys: list[str]) -> float:
        """
        功能说明：计算“最早开始到最晚结束”的墙钟跨度，缺失时返回0。
        参数说明：
        - start_keys: 起点键名数组。
        - end_keys: 终点键名数组。
        返回值：
        - float: 非负跨度（秒）。
        异常说明：无。
        边界条件：任一侧无有效打点时返回0。
        """
        start_values = _mark_values(start_keys)
        end_values = _mark_values(end_keys)
        if not start_values or not end_values:
            return 0.0
        return max(0.0, max(end_values) - min(start_values))

    def _run_librosa_track_with_timing(track_name: str, track_path: Path, **kwargs):
        """
        功能说明：封装单轨Librosa提取并记录起止时间。
        参数说明：
        - track_name: 轨道名（vocals/no_vocals）。
        - track_path: 轨道音频路径。
        - kwargs: 透传给Librosa后端的开关参数。
        返回值：
        - tuple: Librosa后端返回的候选结果。
        异常说明：异常向上抛出，由主流程处理回退或失败语义。
        边界条件：仅记录打点，不改变后端行为。
        """
        start_key = f"librosa_{track_name}_start"
        end_key = f"librosa_{track_name}_end"
        timing_marks[start_key] = perf_counter()
        try:
            return extract_acoustic_candidates_with_librosa(
                audio_path=track_path,
                duration_seconds=duration_seconds,
                logger=logger,
                **kwargs,
            )
        finally:
            timing_marks[end_key] = perf_counter()

    def _run_funasr_task_with_timing():
        """
        功能说明：封装FunASR任务并记录起止时间。
        参数说明：无。
        返回值：
        - tuple: FunASR原始结果、分句结果、分句统计。
        异常说明：异常向上抛出，由主流程降级处理。
        边界条件：不改变FunASR识别语义。
        """
        timing_marks["funasr_start"] = perf_counter()
        try:
            return recognize_lyrics_with_funasr_v2(
                audio_path=str(demucs_stems["vocals"]),
                model_name=funasr_model,
                device=device,
                funasr_language=funasr_language,
                logger=logger,
            )
        finally:
            timing_marks["funasr_end"] = perf_counter()

    logger.info("模块A V2-Librosa双轨并行提取开始，tracks=vocals/no_vocals")
    timing_marks["librosa_parallel_start"] = perf_counter()
    vocal_result = None
    vocal_error = None
    accompaniment_result = None
    funasr_future = None
    funasr_executor: ThreadPoolExecutor | None = None
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="librosa") as librosa_executor:
            vocal_future = librosa_executor.submit(
                _run_librosa_track_with_timing,
                "vocals",
                demucs_stems["vocals"],
                with_f0_points=True,
            )
            accompaniment_future = librosa_executor.submit(
                _run_librosa_track_with_timing,
                "no_vocals",
                demucs_stems["no_vocals"],
                with_onset_points=True,
                with_chroma_points=True,
                with_f0_points=True,
            )

            if bool(skip_funasr_when_vocals_silent):
                timing_marks["precheck_start"] = perf_counter()
                _, _, precheck_rms_times, precheck_rms_values = extract_acoustic_candidates_with_librosa(
                    audio_path=demucs_stems["vocals"],
                    duration_seconds=duration_seconds,
                    logger=logger,
                    extract_beat=False,
                    extract_onset=False,
                )
                timing_marks["precheck_end"] = perf_counter()
                should_skip, peak_rms, active_ratio = _should_skip_funasr_by_vocal_energy(
                    vocal_rms_values=precheck_rms_values,
                    peak_threshold=vocal_skip_peak_rms_threshold,
                    active_ratio_threshold=vocal_skip_active_ratio_threshold,
                )
                funasr_skipped_for_silent_vocals = bool(should_skip)
                dump_json_artifact(
                    output_path=artifacts.perception_signal_librosa_vocal_precheck_path,
                    payload={
                        "rms_times": precheck_rms_times,
                        "rms_values": precheck_rms_values,
                        "should_skip_funasr": bool(should_skip),
                        "peak_rms": float(peak_rms),
                        "active_ratio": float(active_ratio),
                        "peak_threshold": float(vocal_skip_peak_rms_threshold),
                        "active_ratio_threshold": float(vocal_skip_active_ratio_threshold),
                    },
                    logger=logger,
                    artifact_name="vocal_precheck_rms",
                )

            if not funasr_skipped_for_silent_vocals:
                funasr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="funasr")
                funasr_future = funasr_executor.submit(_run_funasr_task_with_timing)

            allin1_analysis = analyze_with_allin1(
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                logger=logger,
                raw_response_path=artifacts.perception_model_allin1_raw_response_path,
                stems_input=stems_input,
                work_dir=allin1_work_dir,
                runtime_device=device,
            )
            big_segments_stage1 = list(allin1_analysis.get("big_segments", []))
            beat_candidates = [float(item) for item in allin1_analysis.get("beat_times", [])]
            beats = list(allin1_analysis.get("beats", []))
            if not big_segments_stage1 or len(beats) < 2 or len(beat_candidates) < 2:
                raise RuntimeError("模块A V2感知层关键产物不足：Allin1大段或节拍不可用")

            try:
                vocal_result = vocal_future.result()
                logger.info(
                    "模块A V2-Librosa子任务完成，track=vocals，耗时=%.3fs",
                    _safe_duration("librosa_vocals_start", "librosa_vocals_end"),
                )
            except Exception as error:  # noqa: BLE001
                vocal_error = error
                logger.warning("模块A V2-人声候选提取失败，回退预检序列，错误=%s", error)

            try:
                accompaniment_result = accompaniment_future.result()
                logger.info(
                    "模块A V2-Librosa子任务完成，track=no_vocals，耗时=%.3fs",
                    _safe_duration("librosa_no_vocals_start", "librosa_no_vocals_end"),
                )
            finally:
                timing_marks["librosa_parallel_end"] = perf_counter()
                librosa_extract_wall_seconds = _safe_span(
                    ["librosa_vocals_start", "librosa_no_vocals_start"],
                    ["librosa_vocals_end", "librosa_no_vocals_end"],
                )
                librosa_schedule_seconds = _safe_duration("librosa_parallel_start", "librosa_parallel_end")
                logger.info(
                    "模块A V2-Librosa双轨并行提取结束，双轨墙钟=%.3fs，调度跨度=%.3fs（含precheck/allin1并发等待）",
                    librosa_extract_wall_seconds,
                    librosa_schedule_seconds,
                )
    finally:
        if funasr_executor is not None:
            funasr_executor.shutdown(wait=True)

    if vocal_error is not None:
        vocal_onset_candidates = []
        vocal_rms_times = precheck_rms_times.copy()
        vocal_rms_values = precheck_rms_values.copy()
        vocal_f0_points = []
    else:
        assert vocal_result is not None
        _, vocal_onset_candidates, vocal_rms_times, vocal_rms_values = vocal_result[:4]
        vocal_f0_points = list(vocal_result[4]) if len(vocal_result) >= 5 else []

    dump_json_artifact(
        output_path=artifacts.perception_signal_librosa_vocal_candidates_path,
        payload={
            "onset_candidates": vocal_onset_candidates,
            "rms_times": vocal_rms_times,
            "rms_values": vocal_rms_values,
            "f0_points_vocals": vocal_f0_points,
        },
        logger=logger,
        artifact_name="vocal_candidates",
    )

    lyric_sentence_units: list[dict] = []
    sentence_split_stats: dict = {}
    if funasr_skipped_for_silent_vocals:
        dump_json_artifact(
            output_path=artifacts.perception_model_funasr_raw_response_path,
            payload={"skipped": True, "reason": "silent_vocals_precheck"},
            logger=logger,
            artifact_name="funasr_raw_response",
        )
        sentence_split_stats = {
            "skipped": True,
            "reason": "silent_vocals_precheck",
            "dynamic_gap_threshold_seconds": 0.35,
            "sample_source": "none",
            "sample_count_raw": 0,
            "sample_count_kept": 0,
            "sample_count_outlier": 0,
            "outlier_samples": [],
        }
        dump_json_artifact(
            output_path=artifacts.perception_model_funasr_sentence_split_stats_path,
            payload=sentence_split_stats,
            logger=logger,
            artifact_name="funasr_sentence_split_stats",
        )
    else:
        try:
            funasr_raw_result, lyric_sentence_units, sentence_split_stats = funasr_future.result()
            dump_json_artifact(
                output_path=artifacts.perception_model_funasr_raw_response_path,
                payload=funasr_raw_result,
                logger=logger,
                artifact_name="funasr_raw_response",
            )
            dump_json_artifact(
                output_path=artifacts.perception_model_funasr_sentence_split_stats_path,
                payload=sentence_split_stats,
                logger=logger,
                artifact_name="funasr_sentence_split_stats",
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A V2-FunASR失败，按空歌词继续，错误=%s", error)
            dump_json_artifact(
                output_path=artifacts.perception_model_funasr_raw_response_path,
                payload={"error": str(error), "skipped": False, "result": []},
                logger=logger,
                artifact_name="funasr_raw_response",
            )
            sentence_split_stats = {
                "error": str(error),
                "skipped": False,
                "dynamic_gap_threshold_seconds": 0.35,
                "sample_source": "none",
                "sample_count_raw": 0,
                "sample_count_kept": 0,
                "sample_count_outlier": 0,
                "outlier_samples": [],
            }
            dump_json_artifact(
                output_path=artifacts.perception_model_funasr_sentence_split_stats_path,
                payload=sentence_split_stats,
                logger=logger,
                artifact_name="funasr_sentence_split_stats",
            )
            lyric_sentence_units = []

    relative_timing_payload = {
        mark_name: (round(float(mark_value) - stage_started_at, 3) if mark_value is not None else None)
        for mark_name, mark_value in timing_marks.items()
    }
    logger.info("模块A V2并发打点（相对秒）=%s", relative_timing_payload)

    librosa_parallel_seconds = _safe_span(
        ["librosa_vocals_start", "librosa_no_vocals_start"],
        ["librosa_vocals_end", "librosa_no_vocals_end"],
    )
    librosa_schedule_seconds = _safe_duration("librosa_parallel_start", "librosa_parallel_end")
    funasr_seconds = _safe_duration("funasr_start", "funasr_end")
    span_seconds = librosa_parallel_seconds
    overlap_seconds = 0.0
    benefit_seconds = 0.0
    if timing_marks["funasr_start"] is not None and timing_marks["funasr_end"] is not None:
        librosa_start_values = _mark_values(["librosa_vocals_start", "librosa_no_vocals_start"])
        librosa_end_values = _mark_values(["librosa_vocals_end", "librosa_no_vocals_end"])
        if librosa_start_values and librosa_end_values:
            librosa_started_at = min(librosa_start_values)
            librosa_ended_at = max(librosa_end_values)
        else:
            librosa_started_at = float(timing_marks["librosa_parallel_start"] or 0.0)
            librosa_ended_at = float(timing_marks["librosa_parallel_end"] or librosa_started_at)
        funasr_started_at = float(timing_marks["funasr_start"])
        funasr_ended_at = float(timing_marks["funasr_end"])
        span_seconds = max(librosa_ended_at, funasr_ended_at) - min(librosa_started_at, funasr_started_at)
        overlap_seconds = max(0.0, min(librosa_ended_at, funasr_ended_at) - max(librosa_started_at, funasr_started_at))
        benefit_seconds = max(0.0, librosa_parallel_seconds + funasr_seconds - span_seconds)
    logger.info(
        "模块A V2并发收益统计：librosa_parallel_seconds=%.3f，librosa_schedule_seconds=%.3f，funasr_seconds=%.3f，span_seconds=%.3f，overlap_seconds=%.3f，benefit_seconds=%.3f",
        librosa_parallel_seconds,
        librosa_schedule_seconds,
        funasr_seconds,
        span_seconds,
        overlap_seconds,
        benefit_seconds,
    )
    if timing_marks["funasr_start"] is not None and timing_marks["funasr_end"] is not None and benefit_seconds <= 1.0:
        logger.warning(
            "模块A V2并发收益不足：benefit_seconds=%.3f（<=1.0），可能存在CPU或IO竞争导致并发提速有限",
            benefit_seconds,
        )

    dump_json_artifact(
        output_path=artifacts.perception_model_funasr_lyric_sentence_units_path,
        payload=lyric_sentence_units,
        logger=logger,
        artifact_name="lyric_sentence_units",
    )

    assert accompaniment_result is not None
    _, onset_candidates, rms_times, rms_values = accompaniment_result[:4]
    onset_points = [{"time": float(item), "energy_raw": 0.0} for item in onset_candidates]
    accompaniment_chroma_points: list[dict] = []
    accompaniment_f0_points: list[dict] = []
    if len(accompaniment_result) >= 5:
        onset_points = list(accompaniment_result[4])
    if len(accompaniment_result) >= 6:
        accompaniment_chroma_points = list(accompaniment_result[5])
    if len(accompaniment_result) >= 7:
        accompaniment_f0_points = list(accompaniment_result[6])
    dump_json_artifact(
        output_path=artifacts.perception_signal_librosa_accompaniment_path,
        payload={
            "onset_candidates": onset_candidates,
            "onset_points": onset_points,
            "chroma_points": accompaniment_chroma_points,
            "f0_points_no_vocals": accompaniment_f0_points,
            "rms_times": rms_times,
            "rms_values": rms_values,
        },
        logger=logger,
        artifact_name="accompaniment_candidates",
    )

    return PerceptionBundle(
        big_segments_stage1=big_segments_stage1,
        beat_candidates=beat_candidates,
        beats=beats,
        lyric_sentence_units=lyric_sentence_units,
        sentence_split_stats=sentence_split_stats,
        vocals_path=demucs_stems["vocals"],
        no_vocals_path=demucs_stems["no_vocals"],
        demucs_stems=demucs_stems,
        onset_candidates=onset_candidates,
        rms_times=rms_times,
        rms_values=rms_values,
        vocal_onset_candidates=vocal_onset_candidates,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
        funasr_skipped_for_silent_vocals=funasr_skipped_for_silent_vocals,
        onset_points=onset_points,
        accompaniment_chroma_points=accompaniment_chroma_points,
        vocal_f0_points=vocal_f0_points,
        accompaniment_f0_points=accompaniment_f0_points,
    )
