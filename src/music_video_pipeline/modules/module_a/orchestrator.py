"""
文件用途：实现模块A编排入口与真实/降级流水线调度。
核心流程：读取音频时长，执行真实链路或规则链路并输出契约结果。
输入输出：输入 RuntimeContext，输出 module_a_output.json。
依赖说明：依赖 mutagen 与项目内 module_a 包导出的兼容函数。
维护说明：为兼容 monkeypatch，内部 helper 调用统一经 module_a 包命名空间分发。
"""

# 标准库：动态导入
import importlib
# 标准库：子进程调用
import subprocess
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
    - beat_interval_seconds: 规则网格节拍间隔（秒）。
    - strict_lyric_timestamps: 是否强制要求歌词时间戳可用。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - dict[str, Any]: 包含 big_segments/segments/beats/lyric_units/energy_features 的分析结果字典。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = audio_path
    accompaniment_path = audio_path
    module_a = _module_a_namespace()

    try:
        vocals_path, accompaniment_path = module_a._separate_with_demucs(
            audio_path,
            work_dir / "demucs",
            device,
            demucs_model,
            logger,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Demucs失败，已回退原始音频，错误=%s", error)

    try:
        big_segments = module_a._detect_big_segments_with_allin1(audio_path, duration_seconds, logger)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Allin1失败，已回退规则大段落，错误=%s", error)
        big_segments = module_a._build_fallback_big_segments(duration_seconds)

    try:
        beat_candidates, onset_candidates, rms_times, rms_values = module_a._extract_acoustic_candidates_with_librosa(
            accompaniment_path,
            duration_seconds,
            logger,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Librosa失败，已回退规则候选池，错误=%s", error)
        beat_candidates = module_a._build_grid_timestamps(duration_seconds, beat_interval_seconds)
        onset_candidates = beat_candidates.copy()
        rms_times = beat_candidates.copy()
        rms_values = [1.0 for _ in rms_times]

    lyric_units_raw: list[dict[str, Any]] = []
    try:
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
    lyric_sentence_starts = [float(item["start_time"]) for item in visual_lyric_units]
    final_timestamps = module_a._select_small_timestamps(
        duration_seconds=duration_seconds,
        big_segments=big_segments,
        beat_candidates=beat_candidates,
        onset_candidates=onset_candidates,
        rms_times=rms_times,
        rms_values=rms_values,
        lyric_sentence_starts=lyric_sentence_starts,
        instrumental_labels=instrumental_labels,
        snap_threshold_ms=snap_threshold_ms,
    )
    segments = module_a._build_segments_with_lyric_priority(
        duration_seconds=duration_seconds,
        big_segments=big_segments,
        beat_candidates=beat_candidates,
        onset_candidates=onset_candidates,
        lyric_units=visual_lyric_units,
        instrumental_labels=instrumental_labels,
        rms_times=rms_times,
        rms_values=rms_values,
    )
    beats = module_a._build_beats_from_segments(segments=segments, fallback_timestamps=final_timestamps)
    lyric_units = module_a._attach_lyrics_to_segments(visual_lyric_units, segments)
    energy_features = module_a._build_energy_features(segments, rms_times, rms_values, beat_candidates)

    if not big_segments or not segments or len(beats) < 2:
        logger.warning("模块A真实链路结果不完整，回退规则链")
        return module_a._run_fallback_pipeline(duration_seconds, beat_interval_seconds, instrumental_labels, logger)

    return {
        "big_segments": big_segments,
        "segments": segments,
        "beats": beats,
        "lyric_units": lyric_units,
        "energy_features": energy_features,
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
