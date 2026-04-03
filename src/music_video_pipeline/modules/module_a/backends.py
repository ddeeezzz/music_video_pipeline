"""
文件用途：提供模块A的外部后端调用实现。
核心流程：调用 Demucs/Allin1/Librosa/FunASR 并标准化输出。
输入输出：输入音频路径与配置，输出段落、候选池与歌词单元。
依赖说明：依赖标准库 subprocess/importlib/shutil 与第三方可选模型包。
维护说明：后端失败时由上层统一降级与状态记录。
"""

# 标准库：动态导入
import importlib
# 标准库：正则处理
import re
# 标准库：子进程调用
import subprocess
# 标准库：路径处理
from pathlib import Path
# 标准库：命令探测
import shutil
# 标准库：类型提示
from typing import Any

# 项目内模块：时间工具
from music_video_pipeline.modules.module_a.timing_energy import (
    _clamp_time,
    _normalize_timestamp_list,
    _round_time,
)


def _separate_with_demucs(audio_path: Path, output_dir: Path, device: str, model_name: str, logger) -> tuple[Path, Path]:
    """
    功能说明：调用 Demucs 分离人声与伴奏。
    参数说明：
    - audio_path: 输入音频文件路径。
    - output_dir: 当前步骤的输出目录。
    - device: 推理设备标识（如 cpu/cuda/auto）。
    - model_name: 模型名称或模型标识。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - tuple[Path, Path]: 人声与伴奏文件路径二元组 `(vocals_path, accompaniment_path)`。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
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
    """
    功能说明：调用 Allin1/Allin1Fix 获取大段落，并标准化为连续区间。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - list[dict[str, Any]]: 标准化后的大段落列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
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
    """
    功能说明：按优先级导入 allin1 后端，兼容 allin1fix 包名。
    参数说明：
    - 无。
    返回值：
    - tuple[str, Any]: 后端模块名称与模块对象二元组。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    import_errors: list[str] = []
    for module_name in ("allin1", "allin1fix"):
        try:
            module_obj = importlib.import_module(module_name)
            return module_name, module_obj
        except Exception as error:  # noqa: BLE001
            import_errors.append(f"{module_name}: {error}")
    raise RuntimeError(f"allin1 导入失败，已尝试 allin1/allin1fix，错误详情：{' | '.join(import_errors)}")


def _extract_acoustic_candidates_with_librosa(audio_path: Path, duration_seconds: float, logger) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    功能说明：调用 Librosa 提取 beat/onset/RMS 候选池。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - tuple[list[float], list[float], list[float], list[float]]: 节拍候选、起音候选、RMS时间轴、RMS值序列四元组。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    try:
        # 第三方库：音频分析与节拍、起音、能量特征提取
        import librosa  # type: ignore
        # 第三方库：向量计算与帧索引构建
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


def _recognize_lyrics_with_funasr(
    audio_path: Path,
    model_name: str,
    device: str,
    funasr_language: str,
    logger,
) -> tuple[list[float], list[dict[str, Any]]]:
    """
    功能说明：调用 FunASR 识别歌词并输出句级歌词单元。
    参数说明：
    - audio_path: 输入音频文件路径。
    - model_name: 模型名称或模型标识。
    - device: 推理设备标识（如 cpu/cuda/auto）。
    - funasr_language: FunASR语言策略配置。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - tuple[list[float], list[dict[str, Any]]]: 歌词句起点列表与句级歌词单元列表二元组。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    try:
        # 第三方库：FunASR 包版本信息读取与模型注册环境
        import funasr as funasr_pkg  # type: ignore
        # 第三方库：FunASR 统一模型入口
        from funasr import AutoModel  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"funasr 导入失败: {error}") from error

    normalized_language = _normalize_funasr_language(funasr_language=funasr_language, logger=logger)
    language_policy = "auto_detect" if normalized_language == "auto" else normalized_language
    logger.info("模块A调用 FunASR 识别歌词，模型=%s，设备=%s，语言策略=%s", model_name, device, language_policy)

    model_kwargs: dict[str, Any] = {
        "model": model_name,
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    }
    if str(device).strip().lower() != "auto":
        model_kwargs["device"] = device

    try:
        model = AutoModel(**model_kwargs)
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if "is not registered" in message or "FunASRNano" in message:
            funasr_version = str(getattr(funasr_pkg, "__version__", "unknown"))
            raise RuntimeError(
                f"FunASR 模型注册失败（FunASRNano 未注册），当前 funasr 版本={funasr_version}。"
                "请使用 README 对齐的 Git 锁定版 FunASR 依赖。"
            ) from error
        raise RuntimeError(f"FunASR 模型初始化失败: {error}") from error
    generate_kwargs: dict[str, Any] = {
        "input": [str(audio_path)],
        "cache": {},
        "batch_size_s": 0,
    }
    if normalized_language != "auto":
        generate_kwargs["language"] = normalized_language

    result = model.generate(**generate_kwargs)
    records = _extract_funasr_records(result=result)
    if not records:
        logger.warning("模块A-FunASR返回空结果，歌词链降级为空")
        return [], []

    time_scale = _infer_funasr_time_scale(records=records)
    lyric_units_raw: list[dict[str, Any]] = []
    for record in records:
        sentence_units = _build_lyric_units_from_sentence_info(record=record, time_scale=time_scale)
        if sentence_units:
            lyric_units_raw.extend(sentence_units)
            continue
        fallback_units = _build_lyric_units_from_timestamp(record=record, time_scale=time_scale)
        lyric_units_raw.extend(fallback_units)

    lyric_units_raw = [item for item in lyric_units_raw if float(item.get("end_time", 0.0)) >= float(item.get("start_time", 0.0))]
    lyric_units_raw.sort(key=lambda item: float(item.get("start_time", 0.0)))
    if not lyric_units_raw and any(str(item.get("text", "")).strip() for item in records):
        raise RuntimeError("FunASR 识别到文本但缺失可用时间戳")

    sentence_starts = [float(item["start_time"]) for item in lyric_units_raw]
    return sentence_starts, lyric_units_raw


def _normalize_funasr_language(funasr_language: str, logger) -> str:
    """
    功能说明：归一化 FunASR 语言配置，不合法时回退 auto。
    参数说明：
    - funasr_language: FunASR语言策略配置。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - str: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    normalized = str(funasr_language).strip().lower().replace("_", "-")
    if not normalized or normalized == "auto":
        return "auto"
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", normalized):
        return normalized
    logger.warning("模块A-FunASR语言配置非法，已回退自动检测，原始值=%s", funasr_language)
    return "auto"


def _extract_funasr_records(result: Any) -> list[dict[str, Any]]:
    """
    功能说明：将 FunASR 返回结果统一为记录列表。
    参数说明：
    - result: 后端模型原始返回结果。
    返回值：
    - list[dict[str, Any]]: 统一结构的FunASR记录列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        if isinstance(result.get("result"), list):
            return [item for item in result["result"] if isinstance(item, dict)]
        return [result]
    return []


def _infer_funasr_time_scale(records: list[dict[str, Any]]) -> float:
    """
    功能说明：推断 FunASR 时间单位倍率（毫秒->秒为 0.001，秒->秒为 1.0）。
    参数说明：
    - records: FunASR记录列表。
    返回值：
    - float: 时间单位缩放倍率（秒制通常为 1.0 或 0.001）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    values: list[float] = []
    for record in records:
        for sentence in record.get("sentence_info", []) if isinstance(record.get("sentence_info"), list) else []:
            if not isinstance(sentence, dict):
                continue
            for key in ["start", "end"]:
                value = sentence.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            for token in sentence.get("timestamp", []) if isinstance(sentence.get("timestamp"), list) else []:
                if isinstance(token, (list, tuple)) and len(token) >= 2:
                    left_value = token[0]
                    right_value = token[1]
                    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                        values.extend([float(left_value), float(right_value)])
                elif isinstance(token, dict):
                    for key in ["start", "end", "begin", "finish"]:
                        token_value = token.get(key)
                        if isinstance(token_value, (int, float)) and not isinstance(token_value, bool):
                            values.append(float(token_value))

        for token in record.get("timestamp", []) if isinstance(record.get("timestamp"), list) else []:
            if isinstance(token, (list, tuple)) and len(token) >= 2:
                left_value = token[0]
                right_value = token[1]
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                    values.extend([float(left_value), float(right_value)])
            elif isinstance(token, dict):
                for key in ["start", "end", "begin", "finish"]:
                    token_value = token.get(key)
                    if isinstance(token_value, (int, float)) and not isinstance(token_value, bool):
                        values.append(float(token_value))
        for token in record.get("timestamps", []) if isinstance(record.get("timestamps"), list) else []:
            if isinstance(token, (list, tuple)) and len(token) >= 2:
                left_value = token[0]
                right_value = token[1]
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                    values.extend([float(left_value), float(right_value)])
            elif isinstance(token, dict):
                for key in ["start", "end", "begin", "finish", "start_time", "end_time"]:
                    token_value = token.get(key)
                    if isinstance(token_value, (int, float)) and not isinstance(token_value, bool):
                        values.append(float(token_value))

    if not values:
        return 0.001

    max_value = max(values)
    has_fraction = any(abs(value - round(value)) > 1e-6 for value in values)
    if max_value >= 500.0:
        return 0.001
    if has_fraction and max_value <= 600.0:
        return 1.0
    if max_value <= 60.0:
        return 1.0
    return 0.001


def _build_lyric_units_from_sentence_info(record: dict[str, Any], time_scale: float) -> list[dict[str, Any]]:
    """
    功能说明：从 sentence_info 提取句级歌词单元。
    参数说明：
    - record: FunASR单条记录对象。
    - time_scale: FunASR时间单位缩放倍率。
    返回值：
    - list[dict[str, Any]]: 构建后的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    sentence_info = record.get("sentence_info", [])
    if not isinstance(sentence_info, list):
        return []

    sentence_units: list[dict[str, Any]] = []
    for sentence in sentence_info:
        if not isinstance(sentence, dict):
            continue
        text = str(sentence.get("text", "")).strip()
        if not text:
            continue

        start_value = sentence.get("start", 0.0)
        end_value = sentence.get("end", start_value)
        if not isinstance(start_value, (int, float)) or not isinstance(end_value, (int, float)):
            continue

        start_time = _round_time(float(start_value) * time_scale)
        end_time = _round_time(max(start_time, float(end_value) * time_scale))
        token_units = _build_token_units_from_timestamp(
            timestamp_items=sentence.get("timestamp", []),
            text=text,
            time_scale=time_scale,
        )
        confidence = _normalize_funasr_confidence(sentence.get("confidence", sentence.get("score", record.get("score"))))
        sentence_units.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "confidence": confidence,
                "no_speech_prob": 0.35,
                "token_units": token_units,
            }
        )
    return sentence_units


def _build_lyric_units_from_timestamp(record: dict[str, Any], time_scale: float) -> list[dict[str, Any]]:
    """
    功能说明：从 text + timestamp 回退构建句级歌词单元。
    参数说明：
    - record: FunASR单条记录对象。
    - time_scale: FunASR时间单位缩放倍率。
    返回值：
    - list[dict[str, Any]]: 构建后的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    text = str(record.get("text", "")).strip()
    timestamp_items = record.get("timestamp", record.get("timestamps", []))
    if not text or not isinstance(timestamp_items, list) or not timestamp_items:
        return []

    token_units = _build_token_units_from_timestamp(timestamp_items=timestamp_items, text=text, time_scale=time_scale)
    if not token_units:
        return []

    sentence_units: list[dict[str, Any]] = []
    current_tokens: list[dict[str, Any]] = []
    for token in token_units:
        current_tokens.append(token)
        token_text = str(token.get("text", ""))
        if re.search(r"[。！？!?；;]", token_text):
            sentence_units.append(_build_sentence_unit_from_tokens(tokens=current_tokens, score=record.get("score")))
            current_tokens = []

    if current_tokens:
        sentence_units.append(_build_sentence_unit_from_tokens(tokens=current_tokens, score=record.get("score")))
    return [item for item in sentence_units if item]


def _build_sentence_unit_from_tokens(tokens: list[dict[str, Any]], score: Any) -> dict[str, Any]:
    """
    功能说明：将 token 列表拼装为句级歌词单元。
    参数说明：
    - tokens: 业务处理所需输入参数。
    - score: 句级评分或置信度输入。
    返回值：
    - dict[str, Any]: 句级歌词单元字典。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not tokens:
        return {}
    start_time = _round_time(float(tokens[0]["start_time"]))
    end_time = _round_time(float(tokens[-1]["end_time"]))
    text_parts = [str(item.get("text", "")) for item in tokens]
    has_word_granularity = any(str(item.get("granularity", "")).strip().lower() == "word" for item in tokens)
    compact_text = " ".join([item for item in text_parts if item]).strip() if has_word_granularity else "".join(text_parts).strip()
    return {
        "start_time": start_time,
        "end_time": max(start_time, end_time),
        "text": compact_text,
        "confidence": _normalize_funasr_confidence(score),
        "no_speech_prob": 0.35,
        "token_units": tokens,
    }


def _build_token_units_from_timestamp(timestamp_items: Any, text: str, time_scale: float) -> list[dict[str, Any]]:
    """
    功能说明：将 FunASR timestamp 转换为统一 token_units 结构。
    参数说明：
    - timestamp_items: 原始时间戳条目集合。
    - text: 文本内容。
    - time_scale: FunASR时间单位缩放倍率。
    返回值：
    - list[dict[str, Any]]: token级时间单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not isinstance(timestamp_items, list) or not timestamp_items:
        return []

    if all(isinstance(item, dict) for item in timestamp_items):
        token_units: list[dict[str, Any]] = []
        for item in timestamp_items:
            token_text = str(item.get("text", item.get("token", ""))).strip()
            if not token_text:
                continue
            start_raw = item.get("start", item.get("begin", item.get("start_time", None)))
            end_raw = item.get("end", item.get("finish", item.get("end_time", None)))
            if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
                continue
            granularity_raw = str(item.get("granularity", item.get("type", "char"))).strip().lower()
            granularity = "word" if granularity_raw == "word" else "char"
            start_time = _round_time(float(start_raw) * time_scale)
            end_time = _round_time(max(start_time, float(end_raw) * time_scale))
            token_units.append(
                {
                    "text": token_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "granularity": granularity,
                }
            )
        return token_units

    if all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in timestamp_items):
        compact_text = str(text).strip()
        token_texts = _split_text_for_timestamp(compact_text)
        granularity = "word" if " " in compact_text.strip() else "char"
        token_units = []
        for index, item in enumerate(timestamp_items):
            start_raw = item[0]
            end_raw = item[1]
            if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
                continue
            token_text = token_texts[index] if index < len(token_texts) else ""
            start_time = _round_time(float(start_raw) * time_scale)
            end_time = _round_time(max(start_time, float(end_raw) * time_scale))
            token_units.append(
                {
                    "text": token_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "granularity": granularity,
                }
            )
        return token_units
    return []


def _split_text_for_timestamp(text: str) -> list[str]:
    """
    功能说明：按空格优先切词，否则逐字符切分，便于与 timestamp 对齐。
    参数说明：
    - text: 文本内容。
    返回值：
    - list[str]: 按时间戳粒度切分后的文本片段列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if not text:
        return []
    if " " in text:
        return [item for item in text.split(" ") if item]
    return [item for item in text if item.strip()]


def _normalize_funasr_confidence(raw_value: Any) -> float:
    """
    功能说明：将 FunASR 置信信号统一映射为 0~1。
    参数说明：
    - raw_value: 原始置信度值。
    返回值：
    - float: 归一化后的结果值。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    if raw_value is None or isinstance(raw_value, bool):
        return 0.65
    try:
        value = float(raw_value)
    except Exception:  # noqa: BLE001
        return 0.65

    if 0.0 <= value <= 1.0:
        return round(value, 3)
    if 1.0 < value <= 100.0:
        return round(max(0.0, min(1.0, value / 100.0)), 3)
    if -20.0 <= value < 0.0:
        return round(max(0.0, min(1.0, 1.0 + value / 20.0)), 3)
    return 0.65
