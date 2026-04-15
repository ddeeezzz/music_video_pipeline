"""
文件用途：提供模块A V2的 Allin1 分析与结果解析能力。
核心流程：调用 allin1/allin1fix，解析 big_segments 与 beats 契约结构。
输入输出：输入音频路径与运行参数，输出统一分析字典。
依赖说明：依赖 importlib/json 与 v2 时间工具。
维护说明：本文件只负责 allin1 解析，不承担编排与落盘编排逻辑。
"""

# 标准库：动态导入
import importlib
# 标准库：JSON 序列化
import json
# 标准库：路径处理
from pathlib import Path
# 标准库：类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time


def analyze_with_allin1(
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
    - logger: 日志记录器。
    - raw_response_path: allin1 原始响应 JSON 输出路径（可选）。
    - stems_input: allin1fix 直连分离声轨输入（可选）。
    - work_dir: allin1 运行工作目录（可选）。
    返回值：
    - dict[str, Any]: 包含 big_segments/beat_times/beat_positions/beats/raw_item。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：后端不支持 stems_input 时自动回退音频路径调用。
    """
    backend_name, backend_module = _import_allin1_backend()
    logger.info("模块A V2调用 %s 检测大时间戳", backend_name)

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
            logger.warning("模块A V2-allin1fix stems_input 调用失败，已回退音频路径调用，错误=%s", error)
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
                "start_time": round_time(start_time),
                "end_time": round_time(end_time),
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
    if normalized_segments[-1]["end_time"] < round_time(duration_seconds):
        normalized_segments.append(
            {
                "segment_id": f"big_{len(normalized_segments) + 1:03d}",
                "start_time": normalized_segments[-1]["end_time"],
                "end_time": round_time(duration_seconds),
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


def _json_default_for_allin1_dump(value: Any) -> Any:
    """
    功能说明：为 Allin1 原始响应提供 JSON 序列化兜底转换。
    参数说明：
    - value: 待序列化对象。
    返回值：
    - Any: 可被 json.dump 处理的安全对象。
    异常说明：无。
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
    - logger: 日志记录器。
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
        logger.info("模块A V2-Allin1原始响应已保存，路径=%s", output_path)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-Allin1原始响应保存失败，路径=%s，错误=%s", output_path, error)


def _extract_first_allin1_item(raw_result: Any) -> Any:
    """
    功能说明：统一提取 allin1 返回中的首个分析项（单曲场景）。
    参数说明：
    - raw_result: allin1 原始返回对象。
    返回值：
    - Any: 单曲分析对象（dict 或对象实例）。
    异常说明：空列表时抛错。
    边界条件：非列表输入直接返回。
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
    异常说明：无。
    边界条件：字段缺失时返回空数组。
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
    异常说明：无。
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
            beat_time = round_time(_clamp_time(float(beat_raw), duration_seconds))
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
    异常说明：无。
    边界条件：beat_positions 缺失时退化为按索引 major/minor。
    """
    output: list[dict[str, Any]] = []
    for index, beat_time in enumerate(beat_times):
        beat_type = "major" if index % 4 == 0 else "minor"
        if index < len(beat_positions) and beat_positions[index] is not None:
            beat_type = "major" if int(beat_positions[index]) == 1 else "minor"
        output.append(
            {
                "time": round_time(float(beat_time)),
                "type": beat_type,
                "source": "allin1",
            }
        )
    return output


def _import_allin1_backend() -> tuple[str, Any]:
    """
    功能说明：按优先级导入 allin1 后端，兼容 allin1fix 包名。
    参数说明：无。
    返回值：
    - tuple[str, Any]: 后端模块名称与模块对象二元组。
    异常说明：导入失败时抛 RuntimeError。
    边界条件：依次尝试 allin1 与 allin1fix。
    """
    import_errors: list[str] = []
    for module_name in ("allin1", "allin1fix"):
        try:
            module_obj = importlib.import_module(module_name)
            return module_name, module_obj
        except Exception as error:  # noqa: BLE001
            import_errors.append(f"{module_name}: {error}")
    raise RuntimeError(f"allin1 导入失败，已尝试 allin1/allin1fix，错误详情：{' | '.join(import_errors)}")


def _clamp_time(time_value: float, duration_seconds: float) -> float:
    """
    功能说明：将时间戳限制在 [0, duration]。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - float: 裁剪到合法范围后的时间戳。
    异常说明：无。
    边界条件：duration 取最小 0.1 秒。
    """
    safe_duration = max(0.1, duration_seconds)
    return max(0.0, min(safe_duration, float(time_value)))
