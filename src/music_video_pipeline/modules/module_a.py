"""
文件用途：实现模块A（音乐理解）的真实链路与降级链路。
核心流程：输出大时间戳与小时间戳，并将小段落作为最小视觉单元。
输入输出：输入 RuntimeContext，输出 ModuleAOutput JSON 文件。
依赖说明：mutagen + 可选 Demucs/Allin1/Librosa/Whisper。
维护说明：模型失败时必须可降级且不阻塞下游。
"""

# 标准库：二分检索
import bisect
# 标准库：动态导入模块
import importlib
# 标准库：正则校验
import re
# 标准库：子进程调用
import subprocess
# 标准库：路径处理
from pathlib import Path
# 标准库：命令探测
import shutil
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


def run_module_a(context: RuntimeContext) -> Path:
    """执行模块A并产出标准JSON。"""
    context.logger.info("模块A开始执行，task_id=%s，输入音频=%s", context.task_id, context.audio_path)
    duration_seconds = _probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )

    mode = context.config.module_a.mode.lower().strip()
    fallback_enabled = bool(context.config.module_a.fallback_enabled)

    try:
        if mode == "fallback_only":
            analysis_data = _run_fallback_pipeline(
                duration_seconds=duration_seconds,
                beat_interval_seconds=context.config.mock.beat_interval_seconds,
                instrumental_labels=context.config.module_a.instrumental_labels,
                logger=context.logger,
            )
        else:
            analysis_data = _run_real_pipeline(
                audio_path=context.audio_path,
                duration_seconds=duration_seconds,
                work_dir=context.artifacts_dir / "module_a_work",
                snap_threshold_ms=context.config.module_a.lyric_beat_snap_threshold_ms,
                instrumental_labels=context.config.module_a.instrumental_labels,
                device=context.config.module_a.device,
                whisper_model=context.config.module_a.whisper_model,
                whisper_language=context.config.module_a.whisper_language,
                demucs_model=context.config.module_a.demucs_model,
                beat_interval_seconds=context.config.mock.beat_interval_seconds,
                logger=context.logger,
            )
    except Exception as error:  # noqa: BLE001
        if mode == "real_strict" or not fallback_enabled:
            raise RuntimeError(f"模块A真实链路失败且不允许降级: {error}") from error
        context.logger.warning("模块A真实链路失败，已降级到规则链，错误=%s", error)
        analysis_data = _run_fallback_pipeline(
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
    whisper_model: str,
    whisper_language: str,
    demucs_model: str,
    beat_interval_seconds: float,
    logger,
) -> dict[str, Any]:
    """真实模型优先链路。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = audio_path
    accompaniment_path = audio_path

    try:
        vocals_path, accompaniment_path = _separate_with_demucs(audio_path, work_dir / "demucs", device, demucs_model, logger)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Demucs失败，已回退原始音频，错误=%s", error)

    try:
        big_segments = _detect_big_segments_with_allin1(audio_path, duration_seconds, logger)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Allin1失败，已回退规则大段落，错误=%s", error)
        big_segments = _build_fallback_big_segments(duration_seconds)

    try:
        beat_candidates, onset_candidates, rms_times, rms_values = _extract_acoustic_candidates_with_librosa(
            accompaniment_path, duration_seconds, logger
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Librosa失败，已回退规则候选池，错误=%s", error)
        beat_candidates = _build_grid_timestamps(duration_seconds, beat_interval_seconds)
        onset_candidates = beat_candidates.copy()
        rms_times = beat_candidates.copy()
        rms_values = [1.0 for _ in rms_times]

    lyric_sentence_starts: list[float] = []
    lyric_units_raw: list[dict[str, Any]] = []
    try:
        lyric_sentence_starts, lyric_units_raw = _recognize_lyrics_with_whisper(
            vocals_path,
            whisper_model,
            device,
            whisper_language,
            logger,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Whisper失败，歌词链降级为空，错误=%s", error)

    cleaned_lyric_units = _clean_whisper_lyric_units(
        lyric_units_raw=lyric_units_raw,
        big_segments=big_segments,
        instrumental_labels=instrumental_labels,
        logger=logger,
    )
    lyric_sentence_starts = [float(item["start_time"]) for item in cleaned_lyric_units]
    final_timestamps = _select_small_timestamps(
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
    segments = _build_segments_with_lyric_priority(
        duration_seconds=duration_seconds,
        big_segments=big_segments,
        beat_candidates=beat_candidates,
        onset_candidates=onset_candidates,
        lyric_units=cleaned_lyric_units,
        instrumental_labels=instrumental_labels,
    )
    beats = _build_beats_from_segments(segments=segments, fallback_timestamps=final_timestamps)
    lyric_units = _attach_lyrics_to_segments(cleaned_lyric_units, segments)
    energy_features = _build_energy_features(segments, rms_times, rms_values, beat_candidates)

    if not big_segments or not segments or len(beats) < 2:
        logger.warning("模块A真实链路结果不完整，回退规则链")
        return _run_fallback_pipeline(duration_seconds, beat_interval_seconds, instrumental_labels, logger)

    return {
        "big_segments": big_segments,
        "segments": segments,
        "beats": beats,
        "lyric_units": lyric_units,
        "energy_features": energy_features,
    }


def _run_fallback_pipeline(duration_seconds: float, beat_interval_seconds: float, instrumental_labels: list[str], logger) -> dict[str, Any]:
    """纯规则降级链路。"""
    logger.info("模块A进入规则降级链路")
    big_segments = _build_fallback_big_segments(duration_seconds)
    beat_candidates = _build_grid_timestamps(duration_seconds, beat_interval_seconds)
    onset_candidates = beat_candidates.copy()
    rms_times = beat_candidates.copy()
    rms_values = [1.0 + (index % 5) * 0.1 for index in range(len(rms_times))]

    final_timestamps = _select_small_timestamps(
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
    segments = _build_small_segments(final_timestamps, big_segments, duration_seconds)
    beats = _build_beats_from_timestamps(final_timestamps)
    energy_features = _build_energy_features(segments, rms_times, rms_values, beat_candidates)

    return {
        "big_segments": big_segments,
        "segments": segments,
        "beats": beats,
        "lyric_units": [],
        "energy_features": energy_features,
    }


def _probe_audio_duration(audio_path: Path, ffprobe_bin: str, logger) -> float:
    """读取音频时长，优先 mutagen，失败时 ffprobe。"""
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

def _separate_with_demucs(audio_path: Path, output_dir: Path, device: str, model_name: str, logger) -> tuple[Path, Path]:
    """调用 Demucs 分离人声与伴奏。"""
    demucs_bin = shutil.which("demucs")
    if demucs_bin is None:
        raise RuntimeError("未检测到 demucs 可执行命令")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        demucs_bin,
        "--two-stems",
        "vocals",
        "-n",
        model_name,
        "-o",
        str(output_dir),
        str(audio_path),
    ]
    if device in {"cpu", "cuda"}:
        command.extend(["--device", device])

    logger.info("模块A调用 Demucs，模型=%s，设备=%s", model_name, device)
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    stem_name = audio_path.stem
    vocals_candidates = list(output_dir.glob(f"**/{stem_name}/vocals.wav"))
    accomp_candidates = list(output_dir.glob(f"**/{stem_name}/no_vocals.wav"))
    if not vocals_candidates or not accomp_candidates:
        raise RuntimeError("Demucs 执行成功但未找到分离结果文件")

    return vocals_candidates[0], accomp_candidates[0]


def _detect_big_segments_with_allin1(audio_path: Path, duration_seconds: float, logger) -> list[dict[str, Any]]:
    """调用 Allin1/Allin1Fix 获取大段落，并标准化为连续区间。"""
    backend_name, backend_module = _import_allin1_backend()
    logger.info("模块A调用 %s 检测大时间戳", backend_name)

    if hasattr(backend_module, "analyze"):
        raw_result = backend_module.analyze(str(audio_path))
    elif hasattr(backend_module, "run"):
        raw_result = backend_module.run(str(audio_path))
    else:
        raise RuntimeError(f"{backend_name} 缺少可调用入口（analyze/run）")

    raw_segments: list[Any] = []
    if isinstance(raw_result, dict):
        for key in ["segments", "sections", "section_list"]:
            value = raw_result.get(key)
            if isinstance(value, list):
                raw_segments = value
                break
    elif hasattr(raw_result, "segments"):
        value = getattr(raw_result, "segments")
        if isinstance(value, list):
            raw_segments = value

    if not raw_segments:
        raise RuntimeError("allin1 未返回可用段落")

    parsed: list[dict[str, Any]] = []
    for item in raw_segments:
        if isinstance(item, dict):
            start_raw = item.get("start_time", item.get("start", 0.0))
            end_raw = item.get("end_time", item.get("end", 0.0))
            label_raw = item.get("label", item.get("name", "unknown"))
        else:
            start_raw = getattr(item, "start_time", getattr(item, "start", 0.0))
            end_raw = getattr(item, "end_time", getattr(item, "end", 0.0))
            label_raw = getattr(item, "label", getattr(item, "name", "unknown"))

        start_time = _clamp_time(float(start_raw), duration_seconds)
        end_time = _clamp_time(float(end_raw), duration_seconds)
        if end_time <= start_time:
            continue
        parsed.append({"start_time": start_time, "end_time": end_time, "label": str(label_raw).strip().lower() or "unknown"})

    if not parsed:
        raise RuntimeError("allin1 段落解析后为空")

    parsed.sort(key=lambda item: item["start_time"])
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    for item in parsed:
        start_time = max(cursor, item["start_time"])
        end_time = min(duration_seconds, max(start_time + 0.1, item["end_time"]))
        if end_time <= start_time:
            continue
        normalized.append(
            {
                "segment_id": f"big_{len(normalized) + 1:03d}",
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "label": item["label"],
            }
        )
        cursor = end_time

    if not normalized:
        raise RuntimeError("allin1 归一化后段落为空")

    if normalized[0]["start_time"] > 0.0:
        normalized.insert(
            0,
            {
                "segment_id": "big_000",
                "start_time": 0.0,
                "end_time": normalized[0]["start_time"],
                "label": normalized[0]["label"],
            },
        )
    if normalized[-1]["end_time"] < _round_time(duration_seconds):
        normalized.append(
            {
                "segment_id": f"big_{len(normalized) + 1:03d}",
                "start_time": normalized[-1]["end_time"],
                "end_time": _round_time(duration_seconds),
                "label": normalized[-1]["label"],
            }
        )

    for index, item in enumerate(normalized, start=1):
        item["segment_id"] = f"big_{index:03d}"
    return normalized


def _import_allin1_backend() -> tuple[str, Any]:
    """按优先级导入 allin1 后端，兼容 allin1fix 包名。"""
    import_errors: list[str] = []
    for module_name in ("allin1", "allin1fix"):
        try:
            module_obj = importlib.import_module(module_name)
            return module_name, module_obj
        except Exception as error:  # noqa: BLE001
            import_errors.append(f"{module_name}: {error}")
    raise RuntimeError(f"allin1 导入失败，已尝试 allin1/allin1fix，错误详情：{' | '.join(import_errors)}")


def _extract_acoustic_candidates_with_librosa(audio_path: Path, duration_seconds: float, logger) -> tuple[list[float], list[float], list[float], list[float]]:
    """调用 Librosa 提取 beat/onset/RMS 候选池。"""
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"librosa/numpy 导入失败: {error}") from error

    logger.info("模块A调用 Librosa 提取声学候选池")
    y, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()

    onset_frames = librosa.onset.onset_detect(y=y, sr=sample_rate, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate).tolist()

    hop_length = 512
    rms_values_np = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_values_np)), sr=sample_rate, hop_length=hop_length).tolist()
    rms_values = [float(value) for value in rms_values_np.tolist()]

    beat_times = _normalize_timestamp_list(beat_times + [0.0, duration_seconds], duration_seconds)
    onset_times = _normalize_timestamp_list(onset_times + [0.0, duration_seconds], duration_seconds)
    if not rms_times:
        rms_times = [0.0, _round_time(duration_seconds)]
        rms_values = [1.0, 1.0]

    return beat_times, onset_times, rms_times, rms_values


def _recognize_lyrics_with_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    whisper_language: str,
    logger,
) -> tuple[list[float], list[dict[str, Any]]]:
    """调用 Whisper 识别歌词并输出句首时间戳。"""
    try:
        import whisper  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"whisper 导入失败: {error}") from error

    normalized_language = _normalize_whisper_language(whisper_language=whisper_language, logger=logger)
    language_policy = "auto_detect" if normalized_language == "auto" else normalized_language
    logger.info("模块A调用 Whisper 识别歌词，模型=%s，设备=%s，语言策略=%s", model_name, device, language_policy)
    load_device = None if device == "auto" else device
    model = whisper.load_model(model_name, device=load_device)
    transcribe_kwargs: dict[str, Any] = {"word_timestamps": False}
    if normalized_language != "auto":
        transcribe_kwargs["language"] = normalized_language
    result = model.transcribe(str(audio_path), **transcribe_kwargs)

    sentence_starts: list[float] = []
    lyric_units_raw: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start_time = float(segment.get("start", 0.0))
        end_time = float(segment.get("end", start_time))
        avg_logprob = float(segment.get("avg_logprob", -1.0))
        no_speech_prob = float(segment.get("no_speech_prob", 1.0))
        confidence = max(0.0, min(1.0, 0.5 + avg_logprob / 5.0))
        sentence_starts.append(start_time)
        lyric_units_raw.append(
            {
                "start_time": _round_time(start_time),
                "end_time": _round_time(max(start_time, end_time)),
                "text": text,
                "confidence": round(confidence, 3),
                "avg_logprob": round(avg_logprob, 3),
                "no_speech_prob": round(max(0.0, min(1.0, no_speech_prob)), 3),
            }
        )
    return sentence_starts, lyric_units_raw


def _normalize_whisper_language(whisper_language: str, logger) -> str:
    """归一化 Whisper 语言配置，不合法时回退 auto。"""
    normalized = str(whisper_language).strip().lower().replace("_", "-")
    if not normalized or normalized == "auto":
        return "auto"
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", normalized):
        return normalized
    logger.warning("模块A-Whisper语言配置非法，已回退自动检测，原始值=%s", whisper_language)
    return "auto"

def _select_small_timestamps(
    duration_seconds: float,
    big_segments: list[dict[str, Any]],
    beat_candidates: list[float],
    onset_candidates: list[float],
    rms_times: list[float],
    rms_values: list[float],
    lyric_sentence_starts: list[float],
    instrumental_labels: list[str],
    snap_threshold_ms: int,
) -> list[float]:
    """按段落类型筛选最终小时间戳。"""
    snap_threshold_seconds = max(0.0, snap_threshold_ms / 1000.0)
    instrumental_set = {label.lower().strip() for label in instrumental_labels}

    beat_pool = _normalize_timestamp_list(beat_candidates + [0.0, duration_seconds], duration_seconds)
    onset_pool = _normalize_timestamp_list(onset_candidates + [0.0, duration_seconds], duration_seconds)
    timestamps: list[float] = [0.0, duration_seconds]

    for big_segment in big_segments:
        start_time = float(big_segment["start_time"])
        end_time = float(big_segment["end_time"])
        label = str(big_segment.get("label", "")).lower().strip()

        beat_in_segment = [value for value in beat_pool if start_time < value < end_time]
        onset_in_segment = [value for value in onset_pool if start_time < value < end_time]

        if label in instrumental_set:
            if onset_in_segment:
                peak_onset = max(onset_in_segment, key=lambda item: _rms_delta_at(item, rms_times, rms_values))
                peak_delta = _rms_delta_at(peak_onset, rms_times, rms_values)
                if peak_delta <= 1e-6:
                    peak_onset = max(onset_in_segment, key=lambda item: _rms_value_at(item, rms_times, rms_values))
                timestamps.append(peak_onset)
            elif beat_in_segment:
                timestamps.append(beat_in_segment[len(beat_in_segment) // 2])
            else:
                timestamps.append((start_time + end_time) / 2.0)
            timestamps.extend(beat_in_segment[::2] if len(beat_in_segment) > 4 else beat_in_segment)
            continue

        lyric_in_segment = [value for value in lyric_sentence_starts if start_time <= value <= end_time]
        if lyric_in_segment:
            for lyric_time in lyric_in_segment:
                timestamps.append(_snap_to_nearest_beat(lyric_time, beat_pool, snap_threshold_seconds))
        elif beat_in_segment:
            timestamps.extend(beat_in_segment[::4] if len(beat_in_segment) > 3 else beat_in_segment)
        elif onset_in_segment:
            timestamps.append(onset_in_segment[len(onset_in_segment) // 2])
        else:
            timestamps.append((start_time + end_time) / 2.0)

    return _normalize_timestamp_list(timestamps, duration_seconds)


def _build_small_segments(timestamps: list[float], big_segments: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """由相邻小时间戳构建最小视觉单元。"""
    normalized_times = _normalize_timestamp_list(timestamps, duration_seconds)
    if len(normalized_times) < 2:
        normalized_times = [0.0, _round_time(duration_seconds)]

    segments: list[dict[str, Any]] = []
    for index in range(len(normalized_times) - 1):
        start_time = normalized_times[index]
        end_time = normalized_times[index + 1]
        if end_time - start_time < 0.1:
            continue
        mid_time = (start_time + end_time) / 2.0
        big_segment = _find_big_segment(mid_time, big_segments)
        segments.append(
            {
                "segment_id": f"seg_{len(segments) + 1:04d}",
                "big_segment_id": str(big_segment["segment_id"]),
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "label": str(big_segment.get("label", "unknown")),
            }
        )

    if not segments:
        fallback_big_segment = big_segments[0]
        segments.append(
            {
                "segment_id": "seg_0001",
                "big_segment_id": str(fallback_big_segment["segment_id"]),
                "start_time": 0.0,
                "end_time": _round_time(duration_seconds),
                "label": str(fallback_big_segment.get("label", "unknown")),
            }
        )

    segments[0]["start_time"] = 0.0
    segments[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(segments)):
        segments[index]["start_time"] = segments[index - 1]["end_time"]
    return segments


def _build_segments_with_lyric_priority(
    duration_seconds: float,
    big_segments: list[dict[str, Any]],
    beat_candidates: list[float],
    onset_candidates: list[float],
    lyric_units: list[dict[str, Any]],
    instrumental_labels: list[str],
) -> list[dict[str, Any]]:
    """在保持节拍驱动的前提下，对人声段应用“歌词句优先”分段。"""
    if not big_segments:
        return _build_small_segments([0.0, duration_seconds], big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": duration_seconds, "label": "unknown"}], duration_seconds=duration_seconds)  # noqa: E501

    beat_pool = _normalize_timestamp_list(beat_candidates + [0.0, duration_seconds], duration_seconds)
    onset_pool = _normalize_timestamp_list(onset_candidates + [0.0, duration_seconds], duration_seconds)
    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    small_gap_merge_seconds = 0.35

    range_items: list[dict[str, Any]] = []
    spill_end = 0.0
    for big_segment in big_segments:
        start_time = max(float(big_segment["start_time"]), spill_end)
        end_time = float(big_segment["end_time"])
        label = str(big_segment.get("label", "unknown")).lower().strip()
        big_segment_id = str(big_segment["segment_id"])
        if end_time - start_time <= 1e-6:
            continue

        lyric_in_segment = [
            item
            for item in lyric_units
            if start_time <= float(item.get("start_time", 0.0)) < end_time
        ]
        lyric_in_segment.sort(key=lambda item: float(item.get("start_time", 0.0)))

        if label in instrumental_set or not lyric_in_segment:
            for left_time, right_time in _split_range_by_rhythm(
                start_time=start_time,
                end_time=end_time,
                beat_pool=beat_pool,
                onset_pool=onset_pool,
                dense_mode=label in instrumental_set,
            ):
                range_items.append(
                    {
                        "start_time": left_time,
                        "end_time": right_time,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": False,
                    }
                )
            continue

        cursor = start_time
        for lyric_item in lyric_in_segment:
            lyric_start = _clamp_time(float(lyric_item.get("start_time", cursor)), duration_seconds)
            lyric_end = _clamp_time(float(lyric_item.get("end_time", lyric_start)), duration_seconds)
            lyric_start = max(cursor, lyric_start)
            lyric_end = max(lyric_start, lyric_end)

            if lyric_start > cursor + 1e-6:
                # 句间极短空档不单独拆镜头，避免“歌词尚未唱完即跳图”的闪切观感。
                if lyric_start - cursor <= small_gap_merge_seconds:
                    lyric_start = cursor
                else:
                    for left_time, right_time in _split_range_by_rhythm(
                        start_time=cursor,
                        end_time=lyric_start,
                        beat_pool=beat_pool,
                        onset_pool=onset_pool,
                        dense_mode=False,
                    ):
                        range_items.append(
                            {
                                "start_time": left_time,
                                "end_time": right_time,
                                "big_segment_id": big_segment_id,
                                "label": label,
                                "lyric_anchor": False,
                            }
                        )

            if lyric_end > lyric_start + 1e-6:
                # 关键约束：一句歌词独占一个最小视觉单元，禁止句内再切段。
                range_items.append(
                    {
                        "start_time": lyric_start,
                        "end_time": lyric_end,
                        "big_segment_id": big_segment_id,
                        "label": label,
                        "lyric_anchor": True,
                    }
                )
            cursor = max(cursor, lyric_end)
            spill_end = max(spill_end, lyric_end)

        if cursor < end_time - 1e-6:
            if end_time - cursor <= small_gap_merge_seconds and range_items:
                range_items[-1]["end_time"] = end_time
            else:
                for left_time, right_time in _split_range_by_rhythm(
                    start_time=cursor,
                    end_time=end_time,
                    beat_pool=beat_pool,
                    onset_pool=onset_pool,
                    dense_mode=False,
                ):
                    range_items.append(
                        {
                            "start_time": left_time,
                            "end_time": right_time,
                            "big_segment_id": big_segment_id,
                            "label": label,
                            "lyric_anchor": False,
                        }
                    )

    normalized_ranges = _normalize_segment_ranges(range_items=range_items, duration_seconds=duration_seconds)
    normalized_ranges = _merge_short_vocal_non_lyric_ranges(
        range_items=normalized_ranges,
        duration_seconds=duration_seconds,
        instrumental_set=instrumental_set,
        min_duration_seconds=0.8,
    )
    if not normalized_ranges:
        return _build_small_segments(
            timestamps=[0.0, duration_seconds],
            big_segments=big_segments,
            duration_seconds=duration_seconds,
        )

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(normalized_ranges):
        segments.append(
            {
                "segment_id": f"seg_{index + 1:04d}",
                "big_segment_id": str(item["big_segment_id"]),
                "start_time": _round_time(float(item["start_time"])),
                "end_time": _round_time(float(item["end_time"])),
                "label": str(item["label"]),
            }
        )

    segments[0]["start_time"] = 0.0
    segments[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(segments)):
        segments[index]["start_time"] = segments[index - 1]["end_time"]
    return segments


def _split_range_by_rhythm(
    start_time: float,
    end_time: float,
    beat_pool: list[float],
    onset_pool: list[float],
    dense_mode: bool,
) -> list[tuple[float, float]]:
    """在无歌词区间按节拍/起音构建子区间。"""
    if end_time - start_time <= 0.12:
        return [(start_time, end_time)] if end_time > start_time else []

    beat_in_range = [value for value in beat_pool if start_time < value < end_time]
    onset_in_range = [value for value in onset_pool if start_time < value < end_time]

    if beat_in_range:
        step = 2 if dense_mode else 4
        if len(beat_in_range) <= step:
            selected = beat_in_range
        else:
            selected = beat_in_range[::step]
    elif onset_in_range:
        selected = [onset_in_range[len(onset_in_range) // 2]]
    else:
        selected = []

    boundaries = [start_time, *selected, end_time]
    boundaries = sorted(set(_round_time(item) for item in boundaries if start_time <= item <= end_time))
    if boundaries[0] > start_time:
        boundaries.insert(0, _round_time(start_time))
    if boundaries[-1] < end_time:
        boundaries.append(_round_time(end_time))

    ranges: list[tuple[float, float]] = []
    for index in range(len(boundaries) - 1):
        left_time = float(boundaries[index])
        right_time = float(boundaries[index + 1])
        if right_time - left_time < 0.08:
            continue
        ranges.append((left_time, right_time))

    if not ranges:
        return [(start_time, end_time)]
    return ranges


def _normalize_segment_ranges(range_items: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    """规范化区间列表，保证覆盖连续且无非法时长。"""
    if not range_items:
        return []

    sorted_items = sorted(range_items, key=lambda item: float(item.get("start_time", 0.0)))
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(sorted_items):
        original_end = _clamp_time(float(item.get("end_time", cursor)), duration_seconds)
        if index == len(sorted_items) - 1:
            end_time = _round_time(duration_seconds)
        else:
            end_time = _round_time(max(cursor, original_end))
        if end_time - cursor < 0.05:
            continue
        normalized.append(
            {
                "start_time": _round_time(cursor),
                "end_time": _round_time(end_time),
                "big_segment_id": str(item.get("big_segment_id", "")),
                "label": str(item.get("label", "unknown")),
                "lyric_anchor": bool(item.get("lyric_anchor", False)),
            }
        )
        cursor = end_time
        if cursor >= duration_seconds - 1e-6:
            break

    if not normalized:
        return []

    normalized[0]["start_time"] = 0.0
    normalized[-1]["end_time"] = _round_time(duration_seconds)
    for index in range(1, len(normalized)):
        normalized[index]["start_time"] = normalized[index - 1]["end_time"]
    return normalized


def _merge_short_vocal_non_lyric_ranges(
    range_items: list[dict[str, Any]],
    duration_seconds: float,
    instrumental_set: set[str],
    min_duration_seconds: float,
) -> list[dict[str, Any]]:
    """合并人声区过短的非歌词段，降低镜头闪切。"""
    if not range_items:
        return []

    merged_ranges = [dict(item) for item in range_items]
    index = 0
    while index < len(merged_ranges):
        item = merged_ranges[index]
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        duration = max(0.0, end_time - start_time)
        label = str(item.get("label", "unknown")).lower().strip()
        is_lyric = bool(item.get("lyric_anchor", False))

        should_merge = (
            label not in instrumental_set
            and not is_lyric
            and duration < min_duration_seconds
            and len(merged_ranges) > 1
        )
        if not should_merge:
            index += 1
            continue

        if index <= 0:
            merged_ranges[1]["start_time"] = start_time
            del merged_ranges[index]
            continue
        if index >= len(merged_ranges) - 1:
            merged_ranges[index - 1]["end_time"] = end_time
            del merged_ranges[index]
            index = max(0, index - 1)
            continue

        prev_item = merged_ranges[index - 1]
        next_item = merged_ranges[index + 1]
        prev_duration = float(prev_item.get("end_time", 0.0)) - float(prev_item.get("start_time", 0.0))
        next_duration = float(next_item.get("end_time", 0.0)) - float(next_item.get("start_time", 0.0))
        prev_same_label = str(prev_item.get("label", "")).lower().strip() == label
        next_same_label = str(next_item.get("label", "")).lower().strip() == label

        merge_to_prev = False
        if prev_same_label and not next_same_label:
            merge_to_prev = True
        elif next_same_label and not prev_same_label:
            merge_to_prev = False
        else:
            merge_to_prev = prev_duration >= next_duration

        if merge_to_prev:
            prev_item["end_time"] = end_time
            del merged_ranges[index]
            index = max(0, index - 1)
            continue

        next_item["start_time"] = start_time
        del merged_ranges[index]

    return _normalize_segment_ranges(range_items=merged_ranges, duration_seconds=duration_seconds)


def _clean_whisper_lyric_units(
    lyric_units_raw: list[dict[str, Any]],
    big_segments: list[dict[str, Any]],
    instrumental_labels: list[str],
    logger,
    min_confidence: float = 0.25,
) -> list[dict[str, Any]]:
    """清洗 Whisper 原始结果：输出可靠歌词/未识别歌词/吟唱三态。"""
    if not lyric_units_raw:
        return []

    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    cleaned_units: list[dict[str, Any]] = []
    unknown_text = "[未识别歌词]"
    for item in lyric_units_raw:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        start_time = _round_time(start_time)
        end_time = _round_time(max(start_time, end_time))
        text = str(item.get("text", "")).strip()
        confidence = round(float(item.get("confidence", 0.0)), 3)
        no_speech_prob = max(0.0, min(1.0, float(item.get("no_speech_prob", 1.0))))
        if not text:
            continue

        mid_time = (start_time + end_time) / 2.0
        big_segment = _find_big_segment(mid_time, big_segments)
        big_label = str(big_segment.get("label", "unknown")).lower().strip()
        if big_label in instrumental_set:
            continue

        if _is_vocalise_text(text):
            cleaned_units.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": "吟唱",
                    "confidence": max(0.5, confidence),
                }
            )
            continue

        if _is_obvious_noise_text(text=text, confidence=confidence, no_speech_prob=no_speech_prob):
            continue

        suspected_vocal = _is_probable_vocal_presence(
            confidence=confidence,
            no_speech_prob=no_speech_prob,
            start_time=start_time,
            end_time=end_time,
        )
        if _is_placeholder_text(text):
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                    }
                )
            continue

        if confidence < min_confidence:
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                    }
                )
            continue

        cleaned_units.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "confidence": confidence,
            }
        )

    cleaned_units.sort(key=lambda item: float(item["start_time"]))
    logger.info("模块A-Whisper清洗完成，原始=%s，保留=%s", len(lyric_units_raw), len(cleaned_units))
    return cleaned_units


def _is_placeholder_text(text: str) -> bool:
    """判断是否为歌词占位词（用于转写不可靠标记）。"""
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return False
    placeholder_set = {"lyrics", "lyric", "歌詞", "歌词", "歌詩", "lyrix", "music"}
    return normalized in placeholder_set


def _is_obvious_noise_text(text: str, confidence: float, no_speech_prob: float) -> bool:
    """判断明显噪声文本（用于直接过滤，避免幻觉词显示）。"""
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return True
    if len(normalized) <= 2 and re.fullmatch(r"[a-z]+", normalized):
        return True
    if len(normalized) <= 3 and re.fullmatch(r"[a-z]+", normalized) and confidence < 0.35:
        return True
    if no_speech_prob >= 0.85 and confidence < 0.4:
        return True
    return False


def _is_probable_vocal_presence(confidence: float, no_speech_prob: float, start_time: float, end_time: float) -> bool:
    """基于置信度+无声概率估计是否存在人声。"""
    duration = max(0.0, end_time - start_time)
    if no_speech_prob <= 0.45:
        return True
    if no_speech_prob <= 0.6 and (confidence >= 0.12 or duration >= 0.35):
        return True
    return False


def _is_vocalise_text(text: str) -> bool:
    """判断是否为无语义吟唱片段（lalala/dadada/啦啦啦 等）。"""
    raw_text = text.strip().lower()
    compact_text = re.sub(r"[\s\-\_\,\.\!\?]+", "", raw_text)
    if not compact_text:
        return False

    if re.fullmatch(r"(?:la|da|na|ra){3,}", compact_text):
        return True
    if re.fullmatch(r"[らラ啦らぁラァ]{3,}", compact_text):
        return True
    if re.fullmatch(r"(?:は|ハ|啊|あ){3,}", compact_text):
        return True
    if re.fullmatch(r"(?:吟唱)+", compact_text):
        return True
    return False


def _attach_lyrics_to_segments(lyric_units_raw: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将歌词单元按时间绑定到小段落。"""
    if not lyric_units_raw or not segments:
        return []

    output_items: list[dict[str, Any]] = []
    for item in lyric_units_raw:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        segment = _select_best_overlap_segment(start_time=start_time, end_time=end_time, segments=segments)
        output_items.append(
            {
                "segment_id": str(segment["segment_id"]),
                "start_time": _round_time(start_time),
                "end_time": _round_time(max(start_time, end_time)),
                "text": str(item.get("text", "")).strip(),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
            }
        )
    return output_items


def _select_best_overlap_segment(start_time: float, end_time: float, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """根据时间重叠度选择最匹配的小段落，避免边界歌词挂错片段。"""
    lyric_end = max(start_time, end_time)
    best_segment = segments[0]
    best_overlap = -1.0
    for segment in segments:
        seg_start = float(segment["start_time"])
        seg_end = float(segment["end_time"])
        overlap = max(0.0, min(lyric_end, seg_end) - max(start_time, seg_start))
        if overlap > best_overlap + 1e-6:
            best_segment = segment
            best_overlap = overlap
            continue
        if abs(overlap - best_overlap) <= 1e-6 and seg_start <= start_time <= seg_end:
            best_segment = segment
    return best_segment


def _build_energy_features(
    segments: list[dict[str, Any]],
    rms_times: list[float],
    rms_values: list[float],
    beat_candidates: list[float],
) -> list[dict[str, Any]]:
    """按小段落计算能量等级、趋势与节奏紧张度。"""
    if not segments:
        return []
    if not rms_times or not rms_values:
        return _build_fallback_energy_features(segments)

    safe_max = max(max(rms_values), 1e-6)
    features: list[dict[str, Any]] = []
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        duration = max(0.1, end_time - start_time)

        value_list = _slice_rms(start_time, end_time, rms_times, rms_values)
        if not value_list:
            value_list = [_rms_value_at(start_time, rms_times, rms_values), _rms_value_at(end_time, rms_times, rms_values)]

        mean_energy = sum(value_list) / len(value_list)
        normalized = max(0.0, min(1.0, mean_energy / safe_max))

        first_half = value_list[: max(1, len(value_list) // 2)]
        second_half = value_list[max(1, len(value_list) // 2) :]
        trend_delta = (sum(second_half) / max(1, len(second_half))) - (sum(first_half) / max(1, len(first_half)))

        if normalized < 0.33:
            energy_level = "low"
        elif normalized < 0.66:
            energy_level = "mid"
        else:
            energy_level = "high"

        if trend_delta > 0.02:
            trend = "up"
        elif trend_delta < -0.02:
            trend = "down"
        else:
            trend = "flat"

        beat_count = sum(1 for beat in beat_candidates if start_time <= beat <= end_time)
        rhythm_tension = round(max(0.0, min(1.0, (beat_count / duration) / 4.0)), 3)

        features.append(
            {
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": rhythm_tension,
            }
        )
    return features


def _build_beats_from_timestamps(timestamps: list[float]) -> list[dict[str, Any]]:
    """将最终小时戳映射为 beats 契约结构。"""
    normalized = sorted(set(_round_time(item) for item in timestamps))
    if len(normalized) < 2:
        normalized = [0.0, 0.1]
    return [
        {
            "time": _round_time(time_value),
            "type": "major" if index % 4 == 0 else "minor",
            "source": "adaptive",
        }
        for index, time_value in enumerate(normalized)
    ]


def _build_beats_from_segments(segments: list[dict[str, Any]], fallback_timestamps: list[float]) -> list[dict[str, Any]]:
    """优先基于分段边界构建 beats，缺失时回退旧时戳集合。"""
    if not segments:
        return _build_beats_from_timestamps(fallback_timestamps)

    timestamps = [float(item["start_time"]) for item in segments]
    timestamps.append(float(segments[-1]["end_time"]))
    return _build_beats_from_timestamps(timestamps)


def _build_fallback_big_segments(duration_seconds: float) -> list[dict[str, Any]]:
    """构建规则化大段落。"""
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    segment_count = min(len(labels), max(1, int(duration_seconds // 30) + 1))
    step = duration_seconds / segment_count

    output: list[dict[str, Any]] = []
    current_start = 0.0
    for index in range(segment_count):
        end_time = duration_seconds if index == segment_count - 1 else current_start + step
        output.append(
            {
                "segment_id": f"big_{index + 1:03d}",
                "start_time": _round_time(current_start),
                "end_time": _round_time(end_time),
                "label": labels[index % len(labels)],
            }
        )
        current_start = end_time
    return output


def _build_fallback_energy_features(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RMS 不可用时的规则化能量特征。"""
    patterns = [
        ("low", "up", 0.30),
        ("mid", "up", 0.55),
        ("high", "flat", 0.85),
        ("mid", "down", 0.50),
        ("low", "flat", 0.25),
    ]
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        energy_level, trend, rhythm_tension = patterns[index % len(patterns)]
        output.append(
            {
                "start_time": _round_time(float(segment["start_time"])),
                "end_time": _round_time(float(segment["end_time"])),
                "energy_level": energy_level,
                "trend": trend,
                "rhythm_tension": rhythm_tension,
            }
        )
    return output

def _build_grid_timestamps(duration_seconds: float, interval_seconds: float) -> list[float]:
    """生成规则网格时间戳。"""
    safe_interval = interval_seconds if interval_seconds > 0 else 0.5
    points: list[float] = []
    cursor = 0.0
    while cursor < duration_seconds:
        points.append(_round_time(cursor))
        cursor += safe_interval
    points.append(_round_time(duration_seconds))
    return _normalize_timestamp_list(points, duration_seconds)


def _normalize_timestamp_list(timestamps: list[float], duration_seconds: float) -> list[float]:
    """归一化时间戳（裁剪、去重、升序、最小间隔）。"""
    clipped = sorted({_clamp_time(value, duration_seconds) for value in timestamps})
    if not clipped:
        return [0.0, _round_time(duration_seconds)]

    filtered: list[float] = [clipped[0]]
    for value in clipped[1:]:
        if value - filtered[-1] >= 0.1:
            filtered.append(value)

    if filtered[0] > 0.0:
        filtered.insert(0, 0.0)
    else:
        filtered[0] = 0.0

    last_time = _round_time(duration_seconds)
    if filtered[-1] < last_time:
        filtered.append(last_time)
    else:
        filtered[-1] = last_time

    dedup: list[float] = [filtered[0]]
    for value in filtered[1:]:
        if value - dedup[-1] >= 0.1:
            dedup.append(value)
    if dedup[-1] < last_time:
        dedup.append(last_time)

    return [_round_time(value) for value in dedup]


def _find_big_segment(time_value: float, big_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """按时间定位所属大段落。"""
    for item in big_segments:
        if float(item["start_time"]) <= time_value <= float(item["end_time"]):
            return item
    return big_segments[-1]


def _rms_value_at(time_value: float, rms_times: list[float], rms_values: list[float]) -> float:
    """按时间点读取最邻近 RMS 值。"""
    if not rms_times or not rms_values:
        return 0.0
    index = bisect.bisect_left(rms_times, time_value)
    if index <= 0:
        return float(rms_values[0])
    if index >= len(rms_times):
        return float(rms_values[-1])

    left_gap = abs(time_value - rms_times[index - 1])
    right_gap = abs(rms_times[index] - time_value)
    target_index = index - 1 if left_gap <= right_gap else index
    return float(rms_values[target_index])


def _rms_delta_at(
    time_value: float,
    rms_times: list[float],
    rms_values: list[float],
    window_ms: float = 100.0,
) -> float:
    """
    功能说明：计算目标时间点前后的能量正向落差（瞬态爆发强度）。
    参数说明：
    - time_value: 目标时间戳（秒）。
    - rms_times: RMS 时间序列。
    - rms_values: RMS 数值序列。
    - window_ms: 回看时间窗口（毫秒），默认 100ms。
    返回值：
    - float: 正向能量落差值，越大代表突变越明显。
    异常说明：无。
    边界条件：当 RMS 数据为空或出现能量下降时返回 0.0。
    """
    if not rms_times or not rms_values:
        return 0.0
    window_seconds = max(0.0, window_ms / 1000.0)
    current_rms = _rms_value_at(time_value, rms_times, rms_values)
    previous_rms = _rms_value_at(time_value - window_seconds, rms_times, rms_values)
    return max(0.0, current_rms - previous_rms)


def _slice_rms(start_time: float, end_time: float, rms_times: list[float], rms_values: list[float]) -> list[float]:
    """提取时间区间内 RMS 子集。"""
    output: list[float] = []
    for index, time_value in enumerate(rms_times):
        if start_time <= time_value <= end_time:
            output.append(float(rms_values[index]))
    return output


def _snap_to_nearest_beat(target_time: float, beat_pool: list[float], threshold_seconds: float) -> float:
    """歌词时间戳吸附到最近节拍点。"""
    if not beat_pool:
        return target_time

    insert_index = bisect.bisect_left(beat_pool, target_time)
    candidates: list[float] = []
    if insert_index > 0:
        candidates.append(beat_pool[insert_index - 1])
    if insert_index < len(beat_pool):
        candidates.append(beat_pool[insert_index])
    if not candidates:
        return target_time

    nearest = min(candidates, key=lambda value: abs(value - target_time))
    if abs(nearest - target_time) > threshold_seconds:
        return nearest
    return target_time


def _clamp_time(time_value: float, duration_seconds: float) -> float:
    """将时间戳限制在 [0, duration]。"""
    safe_duration = max(0.1, duration_seconds)
    return max(0.0, min(safe_duration, float(time_value)))


def _round_time(time_value: float) -> float:
    """统一时间戳保留 3 位小数。"""
    return round(float(time_value), 3)
