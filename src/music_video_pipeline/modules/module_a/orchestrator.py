"""
文件用途：实现模块A编排入口与真实/降级流水线调度。
核心流程：读取音频时长，执行真实链路或规则链路并输出契约结果。
输入输出：输入 RuntimeContext，输出 module_a_output.json。
依赖说明：依赖 mutagen 与项目内 module_a 包导出的兼容函数。
维护说明：为兼容 monkeypatch，内部 helper 调用统一经 module_a 包命名空间分发。
"""

# 标准库：动态导入
import importlib
# 标准库：日志构建
import logging
# 标准库：多进程隔离执行
import multiprocessing
# 标准库：并发执行
from concurrent.futures import ThreadPoolExecutor
# 标准库：队列空异常
from queue import Empty
# 标准库：子进程调用
import subprocess
# 标准库：异常堆栈格式化
import traceback
# 标准库：路径处理
from pathlib import Path
# 标准库：类型提示
from typing import Any

# 第三方库：读取音频时长
from mutagen import File as MutagenFile

# 项目内模块：上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON写入
from music_video_pipeline.io_utils import write_json
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output


def _module_a_namespace():
    """
    功能说明：返回 module_a 包命名空间，用于兼容 monkeypatch 动态分发。
    参数说明：
    - 无。
    返回值：
    - 未显式标注（实际为模块对象）: 模块A包命名空间对象，用于动态分发可被 monkeypatch 的函数。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return importlib.import_module("music_video_pipeline.modules.module_a")


def _spawn_callable_entrypoint(callable_obj, kwargs: dict[str, Any], logger_name: str, result_queue) -> None:
    """
    功能说明：子进程执行入口，运行可调用对象并通过队列回传结果。
    参数说明：
    - callable_obj: 目标可调用对象（顶层函数）。
    - kwargs: 目标函数关键字参数。
    - logger_name: 日志器名称，用于子进程复用同名 logger。
    - result_queue: multiprocessing 队列，用于回传结果字典。
    返回值：无。
    异常说明：异常不会向外抛出，统一写入队列由父进程处理。
    边界条件：结果需可序列化；不可序列化时自动降级去除 raw_item。
    """
    logger = logging.getLogger(logger_name)
    try:
        runtime_kwargs = dict(kwargs)
        injected_logger = False
        if "logger" not in runtime_kwargs:
            runtime_kwargs["logger"] = logger
            injected_logger = True
        try:
            result = callable_obj(**runtime_kwargs)
        except TypeError as error:
            # 兼容不接收 logger 参数的通用可调用对象（用于基础隔离能力测试）。
            if injected_logger and "unexpected keyword argument 'logger'" in str(error):
                runtime_kwargs.pop("logger", None)
                result = callable_obj(**runtime_kwargs)
            else:
                raise
        if isinstance(result, dict) and "raw_item" in result:
            # 原始后端对象可能不可序列化，进程间回传时移除，主链路不依赖该字段。
            result = {key: value for key, value in result.items() if key != "raw_item"}
        result_queue.put({"ok": True, "result": result})
    except Exception as error:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "logger": logger.name,
            }
        )


def _run_callable_in_spawn_process(callable_obj, kwargs: dict[str, Any], logger_name: str, task_label: str) -> Any:
    """
    功能说明：在独立 spawn 进程中运行目标函数并返回结果。
    参数说明：
    - callable_obj: 目标可调用对象（顶层函数）。
    - kwargs: 目标函数关键字参数。
    - logger_name: 父进程 logger 名称，用于子进程日志归档。
    - task_label: 任务标识，仅用于异常消息。
    返回值：
    - Any: 子进程返回的函数结果。
    异常说明：子进程执行异常或异常退出时抛 RuntimeError。
    边界条件：采用 spawn 隔离，避免与父进程 torch 状态互相污染。
    """
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_spawn_callable_entrypoint,
        args=(callable_obj, kwargs, logger_name, result_queue),
        daemon=False,
    )
    process.start()
    process.join()

    try:
        payload = result_queue.get(timeout=0.2)
    except Empty:
        payload = None
    finally:
        result_queue.close()
        result_queue.join_thread()

    if payload is None:
        raise RuntimeError(f"模块A-{task_label}子进程未返回结果，exit_code={process.exitcode}")
    if not bool(payload.get("ok")):
        error_text = str(payload.get("error", "unknown error"))
        trace_text = str(payload.get("traceback", "")).strip()
        if trace_text:
            raise RuntimeError(f"模块A-{task_label}子进程失败: {error_text}\n{trace_text}")
        raise RuntimeError(f"模块A-{task_label}子进程失败: {error_text}")
    return payload.get("result")


def _is_spawn_parallel_eligible(module_a) -> bool:
    """
    功能说明：判断当前执行环境是否可安全使用 spawn 进程并行。
    参数说明：
    - module_a: module_a 包命名空间对象。
    返回值：
    - bool: `True` 表示可使用 spawn 隔离并行；`False` 表示回退线程并发兼容路径。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当关键函数被 monkeypatch 到非 backends 模块时返回 False。
    """
    analyze_callable = getattr(module_a, "_analyze_with_allin1", None)
    recognize_callable = getattr(module_a, "_recognize_lyrics_with_funasr", None)
    expected_module = "music_video_pipeline.modules.module_a.backends"
    if analyze_callable is None or recognize_callable is None:
        return False
    analyze_module_name = str(getattr(analyze_callable, "__module__", ""))
    recognize_module_name = str(getattr(recognize_callable, "__module__", ""))
    return analyze_module_name == expected_module and recognize_module_name == expected_module


def run_module_a(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块A并产出标准JSON。
    参数说明：
    - context: 运行时上下文对象，包含配置、日志器、任务ID与路径信息。
    返回值：
    - Path: 模块A输出JSON文件路径。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    context.logger.info("模块A开始执行，task_id=%s，输入音频=%s", context.task_id, context.audio_path)
    duration_seconds = _probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )

    mode = context.config.module_a.mode.lower().strip()
    fallback_enabled = bool(context.config.module_a.fallback_enabled)
    module_a = _module_a_namespace()

    try:
        if mode == "fallback_only":
            analysis_data = module_a._run_fallback_pipeline(
                duration_seconds=duration_seconds,
                beat_interval_seconds=context.config.mock.beat_interval_seconds,
                instrumental_labels=context.config.module_a.instrumental_labels,
                logger=context.logger,
            )
        else:
            segmentation_tuning = module_a._build_segmentation_tuning(
                vocal_energy_enter_quantile=context.config.module_a.vocal_energy_enter_quantile,
                vocal_energy_exit_quantile=context.config.module_a.vocal_energy_exit_quantile,
                mid_segment_min_duration_seconds=context.config.module_a.mid_segment_min_duration_seconds,
                short_vocal_non_lyric_merge_seconds=context.config.module_a.short_vocal_non_lyric_merge_seconds,
                instrumental_single_split_min_seconds=context.config.module_a.instrumental_single_split_min_seconds,
                accent_delta_trigger_ratio=context.config.module_a.accent_delta_trigger_ratio,
            )
            analysis_data = module_a._run_real_pipeline(
                audio_path=context.audio_path,
                duration_seconds=duration_seconds,
                work_dir=context.artifacts_dir / "module_a_work",
                snap_threshold_ms=context.config.module_a.lyric_beat_snap_threshold_ms,
                instrumental_labels=context.config.module_a.instrumental_labels,
                device=context.config.module_a.device,
                funasr_model=context.config.module_a.funasr_model,
                funasr_language=context.config.module_a.funasr_language,
                lyric_segment_policy=context.config.module_a.lyric_segment_policy,
                comma_pause_seconds=context.config.module_a.comma_pause_seconds,
                long_pause_seconds=context.config.module_a.long_pause_seconds,
                merge_gap_seconds=context.config.module_a.merge_gap_seconds,
                max_visual_unit_seconds=context.config.module_a.max_visual_unit_seconds,
                demucs_model=context.config.module_a.demucs_model,
                beat_interval_seconds=context.config.mock.beat_interval_seconds,
                strict_lyric_timestamps=(mode == "real_strict"),
                logger=context.logger,
                segmentation_tuning=segmentation_tuning,
                skip_funasr_when_vocals_silent=context.config.module_a.skip_funasr_when_vocals_silent,
                vocal_skip_peak_rms_threshold=context.config.module_a.vocal_skip_peak_rms_threshold,
                vocal_skip_active_ratio_threshold=context.config.module_a.vocal_skip_active_ratio_threshold,
            )
    except Exception as error:  # noqa: BLE001
        if mode == "real_strict" or not fallback_enabled:
            raise RuntimeError(f"模块A真实链路失败且不允许降级: {error}") from error
        context.logger.warning("模块A真实链路失败，已降级到规则链，错误=%s", error)
        analysis_data = module_a._run_fallback_pipeline(
            duration_seconds=duration_seconds,
            beat_interval_seconds=context.config.mock.beat_interval_seconds,
            instrumental_labels=context.config.module_a.instrumental_labels,
            logger=context.logger,
        )

    output_data = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "big_segments": analysis_data["big_segments"],
        "segments": analysis_data["segments"],
        "beats": analysis_data["beats"],
        "lyric_units": analysis_data["lyric_units"],
        "energy_features": analysis_data["energy_features"],
    }
    validate_module_a_output(output_data)
    output_path = context.artifacts_dir / "module_a_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块A执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


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


def _run_real_pipeline(
    audio_path: Path,
    duration_seconds: float,
    work_dir: Path,
    snap_threshold_ms: int,
    instrumental_labels: list[str],
    device: str,
    funasr_model: str,
    funasr_language: str,
    lyric_segment_policy: str,
    comma_pause_seconds: float,
    long_pause_seconds: float,
    merge_gap_seconds: float,
    max_visual_unit_seconds: float,
    demucs_model: str,
    beat_interval_seconds: float,
    strict_lyric_timestamps: bool,
    logger,
    *,
    segmentation_tuning: Any | None = None,
    skip_funasr_when_vocals_silent: bool = True,
    vocal_skip_peak_rms_threshold: float = 0.010,
    vocal_skip_active_ratio_threshold: float = 0.020,
) -> dict[str, Any]:
    """
    功能说明：真实模型优先链路。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - work_dir: 模块A工作目录，用于保存中间产物。
    - snap_threshold_ms: 歌词吸附到节拍的阈值（毫秒）。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - device: 推理设备标识（如 cpu/cuda/auto）。
    - funasr_model: FunASR识别模型名称。
    - funasr_language: FunASR语言策略配置。
    - lyric_segment_policy: 歌词视觉切分策略（如 sentence_strict/adaptive_phrase）。
    - comma_pause_seconds: 逗号停顿触发切分阈值（秒）。
    - long_pause_seconds: 长停顿触发切分阈值（秒）。
    - merge_gap_seconds: 相邻短单元允许合并的最大间隔（秒）。
    - max_visual_unit_seconds: 单个视觉歌词单元允许的最大时长（秒）。
    - demucs_model: Demucs分离模型名称。
    - segmentation_tuning: 分段调参对象（优先使用）。
    - beat_interval_seconds: 规则网格节拍间隔（秒）。
    - strict_lyric_timestamps: 是否强制要求歌词时间戳可用。
    - logger: 日志记录器，用于输出过程与异常信息。
    - skip_funasr_when_vocals_silent: 人声音轨能量极低时是否跳过 FunASR。
    - vocal_skip_peak_rms_threshold: 峰值 RMS 跳过阈值。
    - vocal_skip_active_ratio_threshold: 活跃帧占比跳过阈值。
    返回值：
    - dict[str, Any]: 包含 big_segments/segments/beats/lyric_units/energy_features 的分析结果字典。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = audio_path
    accompaniment_path = audio_path
    stems_input: dict[str, Any] | None = None
    allin1_analysis: dict[str, Any] | None = None
    allin1_failed = False
    beat_candidates: list[float] = []
    beats: list[dict[str, Any]] = []
    module_a = _module_a_namespace()
    big_segments = module_a._build_fallback_big_segments(duration_seconds)
    safe_segmentation_tuning = segmentation_tuning or module_a._build_segmentation_tuning()
    raw_response_path = work_dir / "allin1_raw_response.json"
    lyric_units_raw: list[dict[str, Any]] = []
    allin1_error: Exception | None = None
    funasr_error: Exception | None = None
    funasr_skipped_for_silent_vocals = False

    prechecked_vocal_onset_candidates: list[float] | None = None
    prechecked_vocal_rms_times: list[float] | None = None
    prechecked_vocal_rms_values: list[float] | None = None

    onset_candidates: list[float] = []
    rms_times: list[float] = []
    rms_values: list[float] = []
    vocal_onset_candidates: list[float] = []
    vocal_rms_times: list[float] = []
    vocal_rms_values: list[float] = []
    acoustic_ready = False

    try:
        vocals_path, accompaniment_path, stems_input = module_a._prepare_stems_with_allin1_demucs(
            audio_path,
            work_dir / "allin1_demucs",
            device,
            demucs_model,
            logger,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-allin1fix Demucs 失败，已回退独立 Demucs，错误=%s", error)
        try:
            vocals_path, accompaniment_path = module_a._separate_with_demucs(
                audio_path,
                work_dir / "demucs",
                device,
                demucs_model,
                logger,
            )
        except Exception as demucs_error:  # noqa: BLE001
            logger.warning("模块A-Demucs失败，已回退原始音频，错误=%s", demucs_error)

    def _apply_allin1_analysis(candidate: dict[str, Any] | None) -> None:
        nonlocal allin1_analysis
        nonlocal allin1_failed
        nonlocal big_segments
        nonlocal beat_candidates
        nonlocal beats
        if candidate is None:
            allin1_analysis = None
            allin1_failed = True
            return
        allin1_analysis = candidate
        big_segments = candidate.get("big_segments", big_segments)
        beat_candidates = [float(item) for item in candidate.get("beat_times", [])]
        beats = candidate.get("beats", [])
        if len(beat_candidates) < 2 or len(beats) < 2:
            allin1_failed = True
            logger.warning("模块A-Allin1节拍结果不足，后续将回退规则链")
            return
        allin1_failed = False

    def _prepare_acoustic_candidates() -> None:
        nonlocal acoustic_ready
        nonlocal onset_candidates
        nonlocal rms_times
        nonlocal rms_values
        nonlocal vocal_onset_candidates
        nonlocal vocal_rms_times
        nonlocal vocal_rms_values
        if acoustic_ready:
            return
        try:
            _, onset_candidates, rms_times, rms_values = module_a._extract_acoustic_candidates_with_librosa(
                accompaniment_path,
                duration_seconds,
                logger,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A-Librosa失败，已回退规则起音/能量候选池，错误=%s", error)
            onset_candidates = beat_candidates.copy()
            rms_times = beat_candidates.copy()
            rms_values = [1.0 for _ in rms_times]

        vocal_onset_candidates = onset_candidates.copy()
        vocal_rms_times = rms_times.copy()
        vocal_rms_values = rms_values.copy()
        if (
            prechecked_vocal_onset_candidates is not None
            and prechecked_vocal_rms_times is not None
            and prechecked_vocal_rms_values is not None
        ):
            vocal_onset_candidates = prechecked_vocal_onset_candidates
            vocal_rms_times = prechecked_vocal_rms_times
            vocal_rms_values = prechecked_vocal_rms_values
        else:
            try:
                _, vocal_onset_candidates, vocal_rms_times, vocal_rms_values = module_a._extract_acoustic_candidates_with_librosa(
                    vocals_path,
                    duration_seconds,
                    logger,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("模块A-人声音轨声学提取失败，已回退伴奏侧候选，错误=%s", error)
        acoustic_ready = True

    def _build_non_lyric_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        _prepare_acoustic_candidates()
        big_segments_v2_base = module_a._build_big_segments_v2_by_lyric_overlap(
            big_segments_stage1=big_segments,
            lyric_units=[],
            duration_seconds=duration_seconds,
        )
        if not big_segments_v2_base:
            big_segments_v2_base = big_segments
        module_a._select_small_timestamps(
            duration_seconds=duration_seconds,
            big_segments=big_segments_v2_base,
            beat_candidates=beat_candidates,
            onset_candidates=onset_candidates,
            rms_times=rms_times,
            rms_values=rms_values,
            lyric_sentence_starts=[],
            instrumental_labels=instrumental_labels,
            snap_threshold_ms=snap_threshold_ms,
        )
        segments_base = module_a._build_segments_with_lyric_priority(
            duration_seconds=duration_seconds,
            big_segments=big_segments_v2_base,
            beat_candidates=beat_candidates,
            onset_candidates=onset_candidates,
            lyric_units=[],
            instrumental_labels=instrumental_labels,
            rms_times=rms_times,
            rms_values=rms_values,
            vocal_onset_candidates=vocal_onset_candidates,
            vocal_rms_times=vocal_rms_times,
            vocal_rms_values=vocal_rms_values,
            tuning=safe_segmentation_tuning,
        )
        energy_base = module_a._build_energy_features(segments_base, rms_times, rms_values, beat_candidates)
        return big_segments_v2_base, segments_base, energy_base

    if bool(skip_funasr_when_vocals_silent):
        try:
            _, prechecked_vocal_onset_candidates, prechecked_vocal_rms_times, prechecked_vocal_rms_values = module_a._extract_acoustic_candidates_with_librosa(
                vocals_path,
                duration_seconds,
                logger,
            )
            should_skip, peak_rms, active_ratio = _should_skip_funasr_by_vocal_energy(
                vocal_rms_values=prechecked_vocal_rms_values,
                peak_threshold=vocal_skip_peak_rms_threshold,
                active_ratio_threshold=vocal_skip_active_ratio_threshold,
            )
            if should_skip:
                funasr_skipped_for_silent_vocals = True
                logger.info(
                    "模块A-人声音轨能量极低，已跳过FunASR，peak_rms=%.6f，active_ratio=%.4f，peak_threshold=%.6f，active_ratio_threshold=%.4f",
                    peak_rms,
                    active_ratio,
                    float(vocal_skip_peak_rms_threshold),
                    float(vocal_skip_active_ratio_threshold),
                )
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A-人声音轨能量预检失败，已继续执行FunASR，错误=%s", error)

    big_segments_v2 = big_segments
    segments_v2: list[dict[str, Any]] = []
    energy_features: list[dict[str, Any]] = []

    def _run_allin1_in_isolated_process() -> dict[str, Any]:
        return _run_callable_in_spawn_process(
            callable_obj=module_a._analyze_with_allin1,
            kwargs={
                "audio_path": audio_path,
                "duration_seconds": duration_seconds,
                "raw_response_path": raw_response_path,
                "stems_input": stems_input,
                "work_dir": work_dir,
            },
            logger_name=getattr(logger, "name", "music_video_pipeline"),
            task_label="Allin1",
        )

    def _run_funasr_in_isolated_process() -> tuple[list[float], list[dict[str, Any]]]:
        return _run_callable_in_spawn_process(
            callable_obj=module_a._recognize_lyrics_with_funasr,
            kwargs={
                "audio_path": vocals_path,
                "model_name": funasr_model,
                "device": device,
                "funasr_language": funasr_language,
            },
            logger_name=getattr(logger, "name", "music_video_pipeline"),
            task_label="FunASR",
        )

    if stems_input is not None:
        spawn_parallel_eligible = _is_spawn_parallel_eligible(module_a=module_a)
        if spawn_parallel_eligible:
            spawn_context = multiprocessing.get_context("spawn")
            allin1_queue = spawn_context.Queue(maxsize=1)
            allin1_process = spawn_context.Process(
                target=_spawn_callable_entrypoint,
                args=(
                    module_a._analyze_with_allin1,
                    {
                        "audio_path": audio_path,
                        "duration_seconds": duration_seconds,
                        "raw_response_path": raw_response_path,
                        "stems_input": stems_input,
                        "work_dir": work_dir,
                    },
                    getattr(logger, "name", "music_video_pipeline"),
                    allin1_queue,
                ),
                daemon=False,
            )
            allin1_process.start()

            funasr_queue = None
            funasr_process = None
            if not funasr_skipped_for_silent_vocals:
                funasr_queue = spawn_context.Queue(maxsize=1)
                funasr_process = spawn_context.Process(
                    target=_spawn_callable_entrypoint,
                    args=(
                        module_a._recognize_lyrics_with_funasr,
                        {
                            "audio_path": vocals_path,
                            "model_name": funasr_model,
                            "device": device,
                            "funasr_language": funasr_language,
                        },
                        getattr(logger, "name", "music_video_pipeline"),
                        funasr_queue,
                    ),
                    daemon=False,
                )
                funasr_process.start()

            try:
                allin1_process.join()
                try:
                    allin1_payload = allin1_queue.get(timeout=0.2)
                except Empty:
                    allin1_payload = None
                if allin1_payload is None:
                    raise RuntimeError(f"模块A-Allin1子进程未返回结果，exit_code={allin1_process.exitcode}")
                if not bool(allin1_payload.get("ok")):
                    trace_text = str(allin1_payload.get("traceback", "")).strip()
                    error_text = str(allin1_payload.get("error", "unknown error"))
                    if trace_text:
                        raise RuntimeError(f"{error_text}\n{trace_text}")
                    raise RuntimeError(error_text)
                allin1_candidate = allin1_payload.get("result")
                if isinstance(allin1_candidate, dict) and "raw_item" in allin1_candidate:
                    allin1_candidate = {key: value for key, value in allin1_candidate.items() if key != "raw_item"}
                _apply_allin1_analysis(allin1_candidate)
            except Exception as error:  # noqa: BLE001
                allin1_error = error
                _apply_allin1_analysis(None)
            finally:
                allin1_queue.close()
                allin1_queue.join_thread()

            big_segments_v2, segments_v2, energy_features = _build_non_lyric_outputs()
            if funasr_process is not None and funasr_queue is not None:
                try:
                    funasr_process.join()
                    try:
                        funasr_payload = funasr_queue.get(timeout=0.2)
                    except Empty:
                        funasr_payload = None
                    if funasr_payload is None:
                        raise RuntimeError(f"模块A-FunASR子进程未返回结果，exit_code={funasr_process.exitcode}")
                    if not bool(funasr_payload.get("ok")):
                        trace_text = str(funasr_payload.get("traceback", "")).strip()
                        error_text = str(funasr_payload.get("error", "unknown error"))
                        if trace_text:
                            raise RuntimeError(f"{error_text}\n{trace_text}")
                        raise RuntimeError(error_text)
                    funasr_result = funasr_payload.get("result")
                    if (
                        isinstance(funasr_result, tuple)
                        and len(funasr_result) == 2
                        and isinstance(funasr_result[1], list)
                    ):
                        _, lyric_units_raw = funasr_result
                    else:
                        raise RuntimeError("模块A-FunASR子进程返回值格式非法")
                except Exception as error:  # noqa: BLE001
                    funasr_error = error
                finally:
                    funasr_queue.close()
                    funasr_queue.join_thread()
        else:
            # 兼容路径：测试环境 monkeypatch 会替换后端函数，替换函数无法跨 spawn 进程传递。
            with ThreadPoolExecutor(max_workers=2) as executor:
                allin1_future = executor.submit(
                    module_a._analyze_with_allin1,
                    audio_path,
                    duration_seconds,
                    logger,
                    raw_response_path,
                    stems_input,
                    work_dir,
                )
                funasr_future = None
                if not funasr_skipped_for_silent_vocals:
                    funasr_future = executor.submit(
                        module_a._recognize_lyrics_with_funasr,
                        vocals_path,
                        funasr_model,
                        device,
                        funasr_language,
                        logger,
                    )
                try:
                    _apply_allin1_analysis(allin1_future.result())
                except Exception as error:  # noqa: BLE001
                    allin1_error = error
                    _apply_allin1_analysis(None)
                big_segments_v2, segments_v2, energy_features = _build_non_lyric_outputs()
                if funasr_future is not None:
                    try:
                        _, lyric_units_raw = funasr_future.result()
                    except Exception as error:  # noqa: BLE001
                        funasr_error = error
    else:
        try:
            _apply_allin1_analysis(
                module_a._analyze_with_allin1(
                    audio_path=audio_path,
                    duration_seconds=duration_seconds,
                    logger=logger,
                    raw_response_path=raw_response_path,
                    stems_input=stems_input,
                    work_dir=work_dir,
                )
            )
        except Exception as error:  # noqa: BLE001
            allin1_error = error
            _apply_allin1_analysis(None)
        big_segments_v2, segments_v2, energy_features = _build_non_lyric_outputs()
        if not funasr_skipped_for_silent_vocals:
            try:
                _, lyric_units_raw = module_a._recognize_lyrics_with_funasr(
                    vocals_path,
                    funasr_model,
                    device,
                    funasr_language,
                    logger,
                )
            except Exception as error:  # noqa: BLE001
                funasr_error = error

    if allin1_error is not None:
        logger.warning("模块A-Allin1并行任务失败，已仅重试Allin1并复用其他结果，错误=%s", allin1_error)
        try:
            if stems_input is not None and _is_spawn_parallel_eligible(module_a=module_a):
                _apply_allin1_analysis(_run_allin1_in_isolated_process())
            else:
                _apply_allin1_analysis(
                    module_a._analyze_with_allin1(
                        audio_path=audio_path,
                        duration_seconds=duration_seconds,
                        logger=logger,
                        raw_response_path=raw_response_path,
                        stems_input=stems_input,
                        work_dir=work_dir,
                    )
                )
            big_segments_v2, segments_v2, energy_features = _build_non_lyric_outputs()
        except Exception as error:  # noqa: BLE001
            allin1_failed = True
            logger.warning("模块A-Allin1重试失败，后续将回退规则链，错误=%s", error)

    if funasr_error is not None and not funasr_skipped_for_silent_vocals:
        logger.warning("模块A-FunASR并行任务失败，已仅重试FunASR并复用其他结果，错误=%s", funasr_error)
        try:
            if stems_input is not None and _is_spawn_parallel_eligible(module_a=module_a):
                _, lyric_units_raw = _run_funasr_in_isolated_process()
            else:
                _, lyric_units_raw = module_a._recognize_lyrics_with_funasr(
                    vocals_path,
                    funasr_model,
                    device,
                    funasr_language,
                    logger,
                )
        except Exception as error:  # noqa: BLE001
            if strict_lyric_timestamps:
                raise RuntimeError(f"模块A-FunASR失败且 strict 模式要求歌词时间戳可用: {error}") from error
            logger.warning("模块A-FunASR失败，歌词链降级为空，错误=%s", error)

    final_big_segments = big_segments_v2
    final_segments = segments_v2
    final_energy_features = energy_features
    final_lyric_units: list[dict[str, Any]] = []
    if lyric_units_raw:
        sentence_lyric_units = module_a._clean_lyric_units(
            lyric_units_raw=lyric_units_raw,
            big_segments=big_segments,
            instrumental_labels=instrumental_labels,
            logger=logger,
        )
        visual_lyric_units = module_a._build_visual_lyric_units(
            sentence_units=sentence_lyric_units,
            big_segments=big_segments,
            instrumental_labels=instrumental_labels,
            lyric_segment_policy=lyric_segment_policy,
            comma_pause_seconds=comma_pause_seconds,
            long_pause_seconds=long_pause_seconds,
            merge_gap_seconds=merge_gap_seconds,
            max_visual_unit_seconds=max_visual_unit_seconds,
            logger=logger,
        )
        segmentation_anchor_lyric_units = module_a._build_segmentation_anchor_lyric_units(
            sentence_units=sentence_lyric_units,
            logger=logger,
        )
        if not segmentation_anchor_lyric_units and visual_lyric_units:
            # 兜底：若锚点构建为空，回退到既有视觉歌词单元，保证链路可用。
            segmentation_anchor_lyric_units = visual_lyric_units
        if segmentation_anchor_lyric_units:
            lyric_big_segments_v2 = module_a._build_big_segments_v2_by_lyric_overlap(
                big_segments_stage1=big_segments,
                lyric_units=segmentation_anchor_lyric_units,
                duration_seconds=duration_seconds,
            )
            if not lyric_big_segments_v2:
                lyric_big_segments_v2 = big_segments
            lyric_sentence_starts = [float(item["start_time"]) for item in segmentation_anchor_lyric_units]
            module_a._select_small_timestamps(
                duration_seconds=duration_seconds,
                big_segments=lyric_big_segments_v2,
                beat_candidates=beat_candidates,
                onset_candidates=onset_candidates,
                rms_times=rms_times,
                rms_values=rms_values,
                lyric_sentence_starts=lyric_sentence_starts,
                instrumental_labels=instrumental_labels,
                snap_threshold_ms=snap_threshold_ms,
            )
            lyric_segments_v2 = module_a._build_segments_with_lyric_priority(
                duration_seconds=duration_seconds,
                big_segments=lyric_big_segments_v2,
                beat_candidates=beat_candidates,
                onset_candidates=onset_candidates,
                lyric_units=segmentation_anchor_lyric_units,
                instrumental_labels=instrumental_labels,
                rms_times=rms_times,
                rms_values=rms_values,
                vocal_onset_candidates=vocal_onset_candidates,
                vocal_rms_times=vocal_rms_times,
                vocal_rms_values=vocal_rms_values,
                tuning=safe_segmentation_tuning,
            )
            if lyric_segments_v2:
                final_big_segments = lyric_big_segments_v2
                final_segments = lyric_segments_v2
                final_energy_features = module_a._build_energy_features(final_segments, rms_times, rms_values, beat_candidates)
            final_lyric_units = module_a._attach_lyrics_to_segments(segmentation_anchor_lyric_units, final_segments)

    if allin1_failed or not final_big_segments or not final_segments or len(beats) < 2:
        logger.warning("模块A真实链路结果不完整，回退规则链")
        return module_a._run_fallback_pipeline(duration_seconds, beat_interval_seconds, instrumental_labels, logger)

    return {
        "big_segments": final_big_segments,
        "big_segments_stage1": big_segments,
        "segments": final_segments,
        "beats": beats,
        "lyric_units": final_lyric_units,
        "energy_features": final_energy_features,
    }


def _run_fallback_pipeline(duration_seconds: float, beat_interval_seconds: float, instrumental_labels: list[str], logger) -> dict[str, Any]:
    """
    功能说明：纯规则降级链路。
    参数说明：
    - duration_seconds: 音频总时长（秒）。
    - beat_interval_seconds: 规则网格节拍间隔（秒）。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - dict[str, Any]: 包含 big_segments/segments/beats/lyric_units/energy_features 的分析结果字典。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    logger.info("模块A进入规则降级链路")
    module_a = _module_a_namespace()
    big_segments = module_a._build_fallback_big_segments(duration_seconds)
    beat_candidates = module_a._build_grid_timestamps(duration_seconds, beat_interval_seconds)
    onset_candidates = beat_candidates.copy()
    rms_times = beat_candidates.copy()
    rms_values = [1.0 + (index % 5) * 0.1 for index in range(len(rms_times))]

    final_timestamps = module_a._select_small_timestamps(
        duration_seconds=duration_seconds,
        big_segments=big_segments,
        beat_candidates=beat_candidates,
        onset_candidates=onset_candidates,
        rms_times=rms_times,
        rms_values=rms_values,
        lyric_sentence_starts=[],
        instrumental_labels=instrumental_labels,
        snap_threshold_ms=200,
    )
    segments = module_a._build_small_segments(final_timestamps, big_segments, duration_seconds)
    beats = module_a._build_beats_from_timestamps(final_timestamps)
    energy_features = module_a._build_energy_features(segments, rms_times, rms_values, beat_candidates)

    return {
        "big_segments": big_segments,
        "big_segments_stage1": big_segments,
        "segments": segments,
        "beats": beats,
        "lyric_units": [],
        "energy_features": energy_features,
    }


def _probe_audio_duration(audio_path: Path, ffprobe_bin: str, logger) -> float:
    """
    功能说明：读取音频时长，优先 mutagen，失败时 ffprobe。
    参数说明：
    - audio_path: 输入音频文件路径。
    - ffprobe_bin: ffprobe可执行命令路径或命令名。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - float: 音频时长（秒）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    try:
        media_obj = MutagenFile(audio_path)
        if media_obj is not None and media_obj.info is not None and getattr(media_obj.info, "length", None):
            return max(0.1, float(media_obj.info.length))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A使用 mutagen 读取时长失败，已尝试 ffprobe，错误=%s", error)

    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return max(0.1, float(result.stdout.strip()))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A使用 ffprobe 读取时长失败，已降级默认20秒，错误=%s", error)
        return 20.0
