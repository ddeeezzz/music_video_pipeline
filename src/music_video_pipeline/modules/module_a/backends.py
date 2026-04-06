"""
文件用途：提供模块A的外部后端调用实现。
核心流程：调用 Demucs/Allin1/Librosa/FunASR 并标准化输出。
输入输出：输入音频路径与配置，输出段落、候选池与歌词单元。
依赖说明：依赖标准库 subprocess/importlib/shutil 与第三方可选模型包。
维护说明：后端失败时由上层统一降级与状态记录。
"""

# 标准库：动态导入
import importlib
# 标准库：JSON 序列化
import json
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


def _json_default_for_allin1_dump(value: Any) -> Any:
    """
    功能说明：为 Allin1 原始响应提供 JSON 序列化兜底转换。
    参数说明：
    - value: 待序列化对象。
    返回值：
    - Any: 可被 json.dump 处理的安全对象。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：复杂对象在无法结构化时回退字符串表示。
    """
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): item
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _save_allin1_raw_response(raw_result: Any, output_path: Path, logger) -> None:
    """
    功能说明：保存 Allin1 原始响应 JSON，用于结果追溯与标签证据核验。
    参数说明：
    - raw_result: Allin1 原始返回对象。
    - output_path: 原始响应 JSON 输出路径。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：无。
    异常说明：保存失败时不抛出，记录 warning 并继续主流程。
    边界条件：输出目录不存在时自动创建。
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                raw_result,
                file_obj,
                ensure_ascii=False,
                indent=2,
                default=_json_default_for_allin1_dump,
            )
        logger.info("模块A-Allin1原始响应已保存，路径=%s", output_path)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-Allin1原始响应保存失败，路径=%s，错误=%s", output_path, error)


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

    vocals_path = vocals_candidates[0]
    no_vocals_path = accomp_candidates[0]
    target_sample_rate = _probe_audio_sample_rate(audio_path=audio_path, logger=logger)
    _normalize_standard_stems_sample_rate(
        vocals_path=vocals_path,
        no_vocals_path=no_vocals_path,
        target_sample_rate=target_sample_rate,
        logger=logger,
    )
    return vocals_path, no_vocals_path


def _normalize_allin1_runtime_device(device: str) -> str:
    """
    功能说明：归一化 allin1fix 分离阶段设备参数，仅输出 cpu/cuda。
    参数说明：
    - device: 推理设备标识（如 cpu/cuda/auto）。
    返回值：
    - str: 归一化后的设备字符串，仅为 cpu 或 cuda。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当 torch 不可用或无 CUDA 时默认 cpu。
    """
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    try:
        # 第三方库：设备可用性探测
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _probe_audio_sample_rate(audio_path: Path, logger) -> int | None:
    """
    功能说明：通过 ffprobe 读取音频采样率（Hz）。
    参数说明：
    - audio_path: 输入音频路径。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - int | None: 采样率；失败时返回 None。
    异常说明：异常在函数内吞并并记录 warning，不中断主流程。
    边界条件：ffprobe 不可用时直接返回 None。
    """
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        logger.warning("模块A-未检测到 ffprobe，跳过采样率标准化探测")
        return None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        text = str(result.stdout).strip()
        if not text:
            return None
        return int(float(text.splitlines()[0]))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A-采样率探测失败，路径=%s，错误=%s", audio_path, error)
        return None


def _resample_audio_file_inplace(audio_path: Path, target_sample_rate: int, logger) -> None:
    """
    功能说明：使用 ffmpeg 原地重采样音频文件并强制双声道输出。
    参数说明：
    - audio_path: 待重采样音频路径。
    - target_sample_rate: 目标采样率（Hz）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：无。
    异常说明：重采样失败抛错，由上层统一处理。
    边界条件：ffmpeg 不可用时抛错，避免静默格式漂移。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("未检测到 ffmpeg，可用性不足以执行采样率标准化")

    temp_output_path = audio_path.with_name(f"{audio_path.stem}.resample_tmp.wav")
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ar",
        str(int(target_sample_rate)),
        "-ac",
        "2",
        str(temp_output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    temp_output_path.replace(audio_path)


def _mix_non_vocal_stems_to_no_vocals(
    bass_path: Path,
    drums_path: Path,
    other_path: Path,
    output_path: Path,
    logger,
) -> None:
    """
    功能说明：将 bass/drums/other 三轨等权求和并做峰值保护，输出 no_vocals.wav。
    参数说明：
    - bass_path: bass 轨路径。
    - drums_path: drums 轨路径。
    - other_path: other 轨路径。
    - output_path: 输出 no_vocals 路径。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：无。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：依赖 ffmpeg amix + alimiter，输出目录不存在时自动创建。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("未检测到 ffmpeg，可用性不足以合成 no_vocals")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(bass_path),
        "-i",
        str(drums_path),
        "-i",
        str(other_path),
        "-filter_complex",
        "[0:a][1:a][2:a]amix=inputs=3:normalize=0,alimiter=limit=0.98",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    logger.info("模块A-已合成 no_vocals，路径=%s", output_path)


def _normalize_standard_stems_sample_rate(
    vocals_path: Path,
    no_vocals_path: Path,
    target_sample_rate: int | None,
    logger,
) -> None:
    """
    功能说明：将标准二轨（vocals/no_vocals）统一重采样到指定采样率。
    参数说明：
    - vocals_path: vocals 轨路径。
    - no_vocals_path: no_vocals 轨路径。
    - target_sample_rate: 目标采样率（Hz）；为 None 时跳过。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：无。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅在目标采样率有效时执行重采样。
    """
    if target_sample_rate is None or target_sample_rate <= 0:
        return
    _resample_audio_file_inplace(audio_path=vocals_path, target_sample_rate=target_sample_rate, logger=logger)
    _resample_audio_file_inplace(audio_path=no_vocals_path, target_sample_rate=target_sample_rate, logger=logger)
    logger.info(
        "模块A-标准二轨采样率已统一，vocals=%s，no_vocals=%s，target_sr=%s",
        vocals_path,
        no_vocals_path,
        target_sample_rate,
    )


def _prepare_stems_with_allin1_demucs(
    audio_path: Path,
    output_dir: Path,
    device: str,
    model_name: str,
    logger,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    功能说明：复用 allin1fix 的 Demucs 分离能力，返回声轨路径与可供 allin1 分析的 stems_input。
    参数说明：
    - audio_path: 输入音频文件路径。
    - output_dir: allin1fix 分离结果输出目录。
    - device: 推理设备标识（如 cpu/cuda/auto）。
    - model_name: 分离模型名称（如 htdemucs）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - tuple[Path, Path, dict[str, Any]]: `(vocals_path, no_vocals_path, stems_input)`。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅在 allin1fix 后端可用时生效，不支持时抛错由上层降级。
    """
    backend_name, backend_module = _import_allin1_backend()
    if backend_name != "allin1fix" or not hasattr(backend_module, "get_stems"):
        raise RuntimeError(f"当前后端={backend_name}，不支持复用 allin1fix 的 Demucs 分离")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_device = _normalize_allin1_runtime_device(device=device)
    provider_obj = None
    provider_cls = getattr(backend_module, "DemucsProvider", None)
    if provider_cls is not None:
        try:
            provider_obj = provider_cls(model_name=model_name, device=runtime_device)
        except TypeError:
            provider_obj = provider_cls(device=runtime_device)

    logger.info("模块A调用 allin1fix-Demucs 分离，模型=%s，设备=%s", model_name, runtime_device)
    stem_dirs = backend_module.get_stems([audio_path], output_dir, provider_obj, runtime_device)
    if not stem_dirs:
        raise RuntimeError("allin1fix Demucs 未返回可用分离目录")

    stem_dir = Path(stem_dirs[0])
    stem_paths = {
        "bass": stem_dir / "bass.wav",
        "drums": stem_dir / "drums.wav",
        "other": stem_dir / "other.wav",
        "vocals": stem_dir / "vocals.wav",
    }
    missing_files = [name for name, path in stem_paths.items() if not path.exists()]
    if missing_files:
        raise RuntimeError(f"allin1fix Demucs 缺少分离文件: {missing_files}")

    stems_input = {
        "bass": stem_paths["bass"],
        "drums": stem_paths["drums"],
        "other": stem_paths["other"],
        "vocals": stem_paths["vocals"],
        "identifier": audio_path.stem,
    }
    no_vocals_path = stem_dir / "no_vocals.wav"
    _mix_non_vocal_stems_to_no_vocals(
        bass_path=stem_paths["bass"],
        drums_path=stem_paths["drums"],
        other_path=stem_paths["other"],
        output_path=no_vocals_path,
        logger=logger,
    )
    target_sample_rate = _probe_audio_sample_rate(audio_path=audio_path, logger=logger)
    _normalize_standard_stems_sample_rate(
        vocals_path=stem_paths["vocals"],
        no_vocals_path=no_vocals_path,
        target_sample_rate=target_sample_rate,
        logger=logger,
    )
    return stem_paths["vocals"], no_vocals_path, stems_input


def _extract_first_allin1_item(raw_result: Any) -> Any:
    """
    功能说明：统一提取 allin1 返回中的首个分析项（单曲场景）。
    参数说明：
    - raw_result: allin1 原始返回对象。
    返回值：
    - Any: 单曲分析对象（dict 或对象实例）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：列表返回为空时抛错。
    """
    if isinstance(raw_result, list):
        if not raw_result:
            raise RuntimeError("allin1 返回结果为空列表")
        return raw_result[0]
    return raw_result


def _extract_allin1_raw_segments(raw_item: Any) -> list[Any]:
    """
    功能说明：从 allin1 单曲结果中抽取原始段落数组。
    参数说明：
    - raw_item: allin1 单曲分析对象。
    返回值：
    - list[Any]: 原始段落数组。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当字段缺失时返回空数组由上层处理。
    """
    if isinstance(raw_item, dict):
        for key in ["segments", "sections", "section_list"]:
            value = raw_item.get(key)
            if isinstance(value, list):
                return value
        return []
    if hasattr(raw_item, "segments"):
        value = getattr(raw_item, "segments")
        if isinstance(value, list):
            return value
    return []


def _extract_allin1_beat_payload(raw_item: Any, duration_seconds: float) -> tuple[list[float], list[int | None]]:
    """
    功能说明：抽取并规范 allin1 beats 与 beat_positions，仅做裁剪/去重/升序。
    参数说明：
    - raw_item: allin1 单曲分析对象。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - tuple[list[float], list[int | None]]: `(beat_times, beat_positions)`。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：不注入额外节拍点，不补 0/duration。
    """
    raw_beats: list[Any] = []
    raw_positions: list[Any] = []
    if isinstance(raw_item, dict):
        beats_value = raw_item.get("beats")
        positions_value = raw_item.get("beat_positions")
        if isinstance(beats_value, list):
            raw_beats = beats_value
        if isinstance(positions_value, list):
            raw_positions = positions_value
    else:
        beats_value = getattr(raw_item, "beats", [])
        positions_value = getattr(raw_item, "beat_positions", [])
        if isinstance(beats_value, list):
            raw_beats = beats_value
        if isinstance(positions_value, list):
            raw_positions = positions_value

    parsed_pairs: list[tuple[float, int | None]] = []
    for index, beat_raw in enumerate(raw_beats):
        try:
            beat_time = _round_time(_clamp_time(float(beat_raw), duration_seconds))
        except Exception:  # noqa: BLE001
            continue
        beat_pos: int | None = None
        if index < len(raw_positions):
            try:
                beat_pos_value = int(float(raw_positions[index]))
                beat_pos = beat_pos_value if beat_pos_value > 0 else None
            except Exception:  # noqa: BLE001
                beat_pos = None
        parsed_pairs.append((beat_time, beat_pos))

    parsed_pairs.sort(key=lambda item: item[0])
    dedup_pairs: list[tuple[float, int | None]] = []
    for beat_time, beat_pos in parsed_pairs:
        if dedup_pairs and abs(beat_time - dedup_pairs[-1][0]) <= 1e-6:
            continue
        dedup_pairs.append((beat_time, beat_pos))

    beat_times = [item[0] for item in dedup_pairs]
    beat_positions = [item[1] for item in dedup_pairs]
    return beat_times, beat_positions


def _build_module_a_beats_from_allin1(beat_times: list[float], beat_positions: list[int | None]) -> list[dict[str, Any]]:
    """
    功能说明：将 allin1 beats 映射为 ModuleAOutput.beats 契约结构。
    参数说明：
    - beat_times: allin1 输出节拍时间列表（秒）。
    - beat_positions: allin1 输出拍位列表（1 表示小节首拍）。
    返回值：
    - list[dict[str, Any]]: 标准化 beats 列表（source 固定 allin1）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：beat_positions 缺失时退化为按索引 major/minor。
    """
    output: list[dict[str, Any]] = []
    for index, beat_time in enumerate(beat_times):
        beat_type = "major" if index % 4 == 0 else "minor"
        if index < len(beat_positions) and beat_positions[index] is not None:
            beat_type = "major" if int(beat_positions[index]) == 1 else "minor"
        output.append(
            {
                "time": _round_time(float(beat_time)),
                "type": beat_type,
                "source": "allin1",
            }
        )
    return output


def _analyze_with_allin1(
    audio_path: Path,
    duration_seconds: float,
    logger,
    raw_response_path: Path | None = None,
    stems_input: dict[str, Any] | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """
    功能说明：调用 allin1 并一次性返回大段落、节拍与原始响应解析结果。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器，用于输出过程与异常信息。
    - raw_response_path: allin1 原始响应 JSON 输出路径（可选）。
    - stems_input: allin1fix 直连分离声轨输入（可选）。
    - work_dir: allin1 运行工作目录（可选）。
    返回值：
    - dict[str, Any]: 包含 big_segments/beat_times/beat_positions/beats/raw_item。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当后端不支持 stems_input 时自动回退音频路径调用。
    """
    backend_name, backend_module = _import_allin1_backend()
    logger.info("模块A调用 %s 检测大时间戳", backend_name)

    raw_result: Any
    analyze_fn = getattr(backend_module, "analyze", None)
    run_fn = getattr(backend_module, "run", None)
    if stems_input is not None and backend_name == "allin1fix" and callable(analyze_fn):
        analyze_kwargs: dict[str, Any] = {
            "stems_input": stems_input,
            "multiprocess": False,
        }
        if work_dir is not None:
            analyze_kwargs["demix_dir"] = work_dir / "allin1_demix"
            analyze_kwargs["spec_dir"] = work_dir / "allin1_spec"
        try:
            raw_result = analyze_fn(**analyze_kwargs)
        except TypeError as error:
            logger.warning("模块A-allin1fix stems_input 调用失败，已回退音频路径调用，错误=%s", error)
            if callable(analyze_fn):
                raw_result = analyze_fn(str(audio_path))
            elif callable(run_fn):
                raw_result = run_fn(str(audio_path))
            else:
                raise RuntimeError(f"{backend_name} 缺少可调用入口（analyze/run）") from error
    elif callable(analyze_fn):
        raw_result = analyze_fn(str(audio_path))
    elif callable(run_fn):
        raw_result = run_fn(str(audio_path))
    else:
        raise RuntimeError(f"{backend_name} 缺少可调用入口（analyze/run）")

    raw_item = _extract_first_allin1_item(raw_result)
    if raw_response_path is not None:
        _save_allin1_raw_response(raw_result=raw_item, output_path=raw_response_path, logger=logger)

    raw_segments = _extract_allin1_raw_segments(raw_item)
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
    normalized_segments: list[dict[str, Any]] = []
    cursor = 0.0
    for item in parsed:
        start_time = max(cursor, item["start_time"])
        end_time = min(duration_seconds, max(start_time + 0.1, item["end_time"]))
        if end_time <= start_time:
            continue
        normalized_segments.append(
            {
                "segment_id": f"big_{len(normalized_segments) + 1:03d}",
                "start_time": _round_time(start_time),
                "end_time": _round_time(end_time),
                "label": item["label"],
            }
        )
        cursor = end_time

    if not normalized_segments:
        raise RuntimeError("allin1 归一化后段落为空")

    if normalized_segments[0]["start_time"] > 0.0:
        normalized_segments.insert(
            0,
            {
                "segment_id": "big_000",
                "start_time": 0.0,
                "end_time": normalized_segments[0]["start_time"],
                "label": normalized_segments[0]["label"],
            },
        )
    if normalized_segments[-1]["end_time"] < _round_time(duration_seconds):
        normalized_segments.append(
            {
                "segment_id": f"big_{len(normalized_segments) + 1:03d}",
                "start_time": normalized_segments[-1]["end_time"],
                "end_time": _round_time(duration_seconds),
                "label": normalized_segments[-1]["label"],
            }
        )
    for index, item in enumerate(normalized_segments, start=1):
        item["segment_id"] = f"big_{index:03d}"

    beat_times, beat_positions = _extract_allin1_beat_payload(raw_item=raw_item, duration_seconds=duration_seconds)
    beats = _build_module_a_beats_from_allin1(beat_times=beat_times, beat_positions=beat_positions)
    return {
        "big_segments": normalized_segments,
        "beat_times": beat_times,
        "beat_positions": beat_positions,
        "beats": beats,
        "raw_item": raw_item,
    }


def _detect_big_segments_with_allin1(
    audio_path: Path,
    duration_seconds: float,
    logger,
    raw_response_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：调用 Allin1/Allin1Fix 获取大段落，并标准化为连续区间。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器，用于输出过程与异常信息。
    - raw_response_path: Allin1 原始响应 JSON 输出路径（可选）。
    返回值：
    - list[dict[str, Any]]: 标准化后的大段落列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    analysis = _analyze_with_allin1(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        logger=logger,
        raw_response_path=raw_response_path,
    )
    return analysis["big_segments"]


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
        lyric_units_raw.extend(_build_lyric_units_from_record(record=record, time_scale=time_scale))

    lyric_units_raw = [item for item in lyric_units_raw if float(item.get("end_time", 0.0)) >= float(item.get("start_time", 0.0))]
    lyric_units_raw.sort(key=lambda item: float(item.get("start_time", 0.0)))
    if not lyric_units_raw and any(str(item.get("text", "")).strip() for item in records):
        raise RuntimeError("FunASR 识别到文本但缺失可用时间戳")

    sentence_starts = [float(item["start_time"]) for item in lyric_units_raw]
    return sentence_starts, lyric_units_raw


def _build_lyric_units_from_record(record: dict[str, Any], time_scale: float) -> list[dict[str, Any]]:
    """
    功能说明：统一从单条 FunASR record 构建歌词单元，优先 sentence_info，失败后回退 timestamp。
    参数说明：
    - record: FunASR单条记录对象。
    - time_scale: FunASR时间单位缩放倍率。
    返回值：
    - list[dict[str, Any]]: 构建后的句级歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：当 sentence_info 与 timestamp 均不可用时返回空列表。
    """
    sentence_units = _build_lyric_units_from_sentence_info(record=record, time_scale=time_scale)
    if sentence_units:
        return sentence_units
    return _build_lyric_units_from_timestamp(record=record, time_scale=time_scale)


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
    values = [value for value in _iter_funasr_record_time_values(records=records)]

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


def _iter_funasr_record_time_values(records: list[dict[str, Any]]) -> list[float]:
    """
    功能说明：统一提取 FunASR 记录中的数值时间戳样本，用于时间单位推断。
    参数说明：
    - records: FunASR记录列表。
    返回值：
    - list[float]: 收集到的全部数值时间戳样本。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无可用时间信息时返回空列表。
    """
    values: list[float] = []
    for record in records:
        sentence_items = record.get("sentence_info", [])
        if isinstance(sentence_items, list):
            for sentence in sentence_items:
                if not isinstance(sentence, dict):
                    continue
                for key in ["start", "end"]:
                    numeric_value = sentence.get(key)
                    if isinstance(numeric_value, (int, float)) and not isinstance(numeric_value, bool):
                        values.append(float(numeric_value))
                values.extend(_collect_numeric_time_values_from_timestamp_items(sentence.get("timestamp", [])))
        values.extend(_collect_numeric_time_values_from_timestamp_items(record.get("timestamp", [])))
        values.extend(_collect_numeric_time_values_from_timestamp_items(record.get("timestamps", [])))
    return values


def _collect_numeric_time_values_from_timestamp_items(timestamp_items: Any) -> list[float]:
    """
    功能说明：从 timestamp 列表中提取可用于时间尺度推断的数值样本。
    参数说明：
    - timestamp_items: FunASR timestamp 原始条目。
    返回值：
    - list[float]: 时间数值样本列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：输入非列表时返回空列表。
    """
    if not isinstance(timestamp_items, list):
        return []
    values: list[float] = []
    for item in timestamp_items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            left_value = item[0]
            right_value = item[1]
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                values.extend([float(left_value), float(right_value)])
            continue
        if not isinstance(item, dict):
            continue
        for key in ["start", "end", "begin", "finish", "start_time", "end_time"]:
            numeric_value = item.get(key)
            if isinstance(numeric_value, (int, float)) and not isinstance(numeric_value, bool):
                values.append(float(numeric_value))
    return values


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
