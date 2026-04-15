"""
文件用途：提供模块A V2内部处理与最终结果的可视化能力。
核心流程：读取任务产物JSON，组装可视化负载，并渲染为单文件HTML页面。
输入输出：输入任务目录与产物路径，输出可直接打开的可视化HTML文件。
依赖说明：依赖标准库 json/math/pathlib/shutil 与项目内 io_utils。
维护说明：本文件仅负责“读证据+可视化表达”，不参与算法推理与产物改写。
"""

# 标准库：用于JSON编码
import json
# 标准库：用于数学计算
import math
# 标准库：用于文本正则归一
import re
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于文件拷贝
import shutil
# 标准库：用于类型提示
from typing import Any

# 项目内模块：JSON读取工具
from music_video_pipeline.io_utils import read_json


# 常量：可视化页面默认文件名
DEFAULT_VISUALIZATION_FILE_NAME = "module_a_v2_visualization.html"
# 常量：可视化页面复制音频时的文件名后缀（前缀取 HTML stem）。
DEFAULT_AUDIO_COPY_SUFFIX = "_audio"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：将输入值安全转换为浮点数。
    参数说明：
    - value: 待转换值。
    - default: 转换失败时回退值。
    返回值：
    - float: 转换结果。
    异常说明：异常在函数内部吞并并回退 default。
    边界条件：NaN/无穷值回退 default。
    """
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if math.isnan(number) or math.isinf(number):
        return float(default)
    return number


def _read_required_json(path: Path) -> Any:
    """
    功能说明：读取必需JSON文件，缺失时抛出明确错误。
    参数说明：
    - path: 必需JSON路径。
    返回值：
    - Any: JSON反序列化对象。
    异常说明：
    - RuntimeError: 文件缺失。
    边界条件：保持原始JSON结构，不做字段修正。
    """
    if not path.exists():
        raise RuntimeError(f"模块A V2可视化失败：缺少必需产物文件 {path}")
    return read_json(path)


def _read_optional_json(path: Path, default_value: Any) -> Any:
    """
    功能说明：读取可选JSON文件，缺失时返回默认值。
    参数说明：
    - path: 可选JSON路径。
    - default_value: 缺失时返回的默认值。
    返回值：
    - Any: 文件内容或默认值。
    异常说明：JSON格式错误会向上抛出，避免静默错误掩盖证据问题。
    边界条件：仅在文件不存在时回退默认值。
    """
    if not path.exists():
        return default_value
    return read_json(path)


def _normalize_segment_item(item: dict[str, Any], layer_name: str) -> dict[str, Any]:
    """
    功能说明：归一化段落条目为前端统一结构。
    参数说明：
    - item: 原始段落字典。
    - layer_name: 图层名。
    返回值：
    - dict[str, Any]: 统一段落结构。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：缺失字段时给出保守默认值。
    """
    start_time = _safe_float(item.get("start_time", 0.0))
    end_time = _safe_float(item.get("end_time", start_time))
    if end_time < start_time:
        end_time = start_time
    segment_id = str(item.get("segment_id", "")).strip()
    if not segment_id:
        segment_id = str(item.get("window_id", "")).strip()
    if not segment_id:
        segment_id = str(item.get("id", "")).strip()
    role_text = str(item.get("role", ""))
    label_text = str(item.get("label", ""))
    display_text = role_text if layer_name == "ROLE" and role_text else label_text
    source_ids_raw = item.get("source_segment_ids", [])
    if not isinstance(source_ids_raw, list):
        source_ids_raw = item.get("source_window_ids", [])
    if not isinstance(source_ids_raw, list):
        source_ids_raw = []
    return {
        "id": segment_id,
        "segment_id": segment_id,
        "window_id": str(item.get("window_id", "")),
        "big_segment_id": str(item.get("big_segment_id", "")),
        "label": label_text,
        "role": role_text,
        "display_text": display_text,
        "merge_action": str(item.get("merge_action", "")),
        "source_segment_ids": list(source_ids_raw),
        "start_time": round(start_time, 6),
        "end_time": round(end_time, 6),
        "duration": round(max(0.0, end_time - start_time), 6),
        "layer": layer_name,
    }


def _normalize_lyric_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    """
    功能说明：归一化歌词条目为前端统一结构。
    参数说明：
    - item: 原始歌词字典。
    - index: 条目序号。
    返回值：
    - dict[str, Any]: 统一歌词结构。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：文本缺失时回退为空字符串。
    """
    start_time = _safe_float(item.get("start_time", 0.0))
    end_time = _safe_float(item.get("end_time", start_time))
    if end_time < start_time:
        end_time = start_time
    text = str(item.get("text", ""))
    display_text = _normalize_lyric_display_text(text)
    confidence = _safe_float(item.get("confidence", 0.0))
    return {
        "id": str(item.get("segment_id", "")) or f"lyric_{index + 1:04d}",
        "segment_id": str(item.get("segment_id", "")),
        "text": text,
        "display_text": display_text,
        "confidence": round(confidence, 4),
        "start_time": round(start_time, 6),
        "end_time": round(end_time, 6),
        "duration": round(max(0.0, end_time - start_time), 6),
    }


def _normalize_lyric_display_text(text: str) -> str:
    """
    功能说明：将歌词文本归一为可读展示文本（不改原始语义）。
    参数说明：
    - text: 原始歌词文本。
    返回值：
    - str: 归一后的展示文本。
    异常说明：无。
    边界条件：空文本返回空字符串。
    """
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return ""
    # CJK上下文中去掉“字与字/标点”之间的多余空格，避免出现“。 我”这类展示噪声。
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？、；：])", "", normalized)
    normalized = re.sub(r"(?<=[，。！？、；：])\s+(?=[\u4e00-\u9fff])", "", normalized)
    return normalized


def _normalize_energy_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    """
    功能说明：归一化能量特征条目为前端统一结构。
    参数说明：
    - item: 原始能量特征字典。
    - index: 条目序号。
    返回值：
    - dict[str, Any]: 统一能量结构。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：能量强度缺失时按0处理。
    """
    start_time = _safe_float(item.get("start_time", 0.0))
    end_time = _safe_float(item.get("end_time", start_time))
    if end_time < start_time:
        end_time = start_time
    return {
        "id": f"energy_{index + 1:04d}",
        "start_time": round(start_time, 6),
        "end_time": round(end_time, 6),
        "duration": round(max(0.0, end_time - start_time), 6),
        "energy_level": str(item.get("energy_level", "")),
        "trend": str(item.get("trend", "")),
        "rhythm_tension": round(_safe_float(item.get("rhythm_tension", 0.0)), 6),
    }


def _normalize_beats(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：归一化节拍列表，保留时间顺序与主要字段。
    参数说明：
    - beats: 原始节拍字典列表。
    返回值：
    - list[dict[str, Any]]: 统一节拍结构列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：非法时间条目会保底为0并参与排序。
    """
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(beats):
        time_value = _safe_float(item.get("time", 0.0))
        normalized.append(
            {
                "id": f"beat_{index + 1:04d}",
                "time": round(max(0.0, time_value), 6),
                "type": str(item.get("type", "")),
                "source": str(item.get("source", "")),
            }
        )
    return sorted(normalized, key=lambda entry: entry["time"])


def _quantile(values: list[float], quantile: float) -> float:
    """
    功能说明：计算线性插值分位数。
    参数说明：
    - values: 样本列表。
    - quantile: 分位点（0~1）。
    返回值：
    - float: 分位值。
    异常说明：无。
    边界条件：空样本回退0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    safe_q = max(0.0, min(1.0, float(quantile)))
    position = safe_q * (len(sorted_values) - 1)
    left_index = int(position)
    right_index = min(len(sorted_values) - 1, left_index + 1)
    if left_index == right_index:
        return sorted_values[left_index]
    weight = position - left_index
    return sorted_values[left_index] * (1.0 - weight) + sorted_values[right_index] * weight


def _normalize_onset_points(onset_points_raw: list[Any], onset_candidates_raw: list[Any]) -> list[dict[str, Any]]:
    """
    功能说明：归一化onset点并补充 energy_norm 供前端深浅渲染。
    参数说明：
    - onset_points_raw: 原始onset点（建议结构：time+energy_raw）。
    - onset_candidates_raw: 兼容旧结构的onset时间数组。
    返回值：
    - list[dict[str, Any]]: 统一onset点（time、energy_raw、energy_norm）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：缺失能量时按0；同时间点保留最高能量。
    """
    onset_energy_by_time: dict[float, float] = {}
    for item in list(onset_points_raw):
        if not isinstance(item, dict):
            continue
        time_value = round(max(0.0, _safe_float(item.get("time", 0.0), 0.0)), 6)
        energy_raw = max(0.0, _safe_float(item.get("energy_raw", 0.0), 0.0))
        previous_value = onset_energy_by_time.get(time_value, 0.0)
        if energy_raw > previous_value:
            onset_energy_by_time[time_value] = energy_raw
    for item in list(onset_candidates_raw):
        time_value = round(max(0.0, _safe_float(item, 0.0)), 6)
        onset_energy_by_time.setdefault(time_value, 0.0)

    normalized_times = sorted(onset_energy_by_time.keys())
    if not normalized_times:
        return []

    raw_values = [float(onset_energy_by_time[time_item]) for time_item in normalized_times]
    has_non_zero = max(raw_values) > 1e-9
    low_quantile = _quantile(raw_values, 0.10)
    high_quantile = _quantile(raw_values, 0.90)
    normalized_output: list[dict[str, Any]] = []
    for index, time_item in enumerate(normalized_times):
        energy_raw = float(onset_energy_by_time[time_item])
        if not has_non_zero:
            energy_norm = 0.0
        elif high_quantile - low_quantile <= 1e-9:
            energy_norm = 1.0 if energy_raw > 1e-9 else 0.0
        else:
            clipped_energy = min(high_quantile, max(low_quantile, energy_raw))
            energy_norm = (clipped_energy - low_quantile) / max(1e-9, high_quantile - low_quantile)
            energy_norm = min(1.0, max(0.0, energy_norm))
        normalized_output.append(
            {
                "id": f"onset_{index + 1:04d}",
                "time": round(time_item, 6),
                "energy_raw": round(energy_raw, 6),
                "energy_norm": round(float(energy_norm), 6),
            }
        )
    return normalized_output


def _downsample_series(times: list[Any], values: list[Any], max_points: int = 4000) -> tuple[list[float], list[float]]:
    """
    功能说明：对RMS序列做等步长下采样，降低页面渲染负担。
    参数说明：
    - times: 时间列表。
    - values: 数值列表。
    - max_points: 最大保留点数。
    返回值：
    - tuple[list[float], list[float]]: 下采样后的时间与数值列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空输入返回空列表。
    """
    paired: list[tuple[float, float]] = []
    pair_count = min(len(times), len(values))
    for index in range(pair_count):
        paired.append((_safe_float(times[index], 0.0), _safe_float(values[index], 0.0)))
    if not paired:
        return [], []
    if len(paired) <= max_points:
        return [item[0] for item in paired], [item[1] for item in paired]
    step = max(1, int(math.ceil(len(paired) / max_points)))
    sampled = [paired[index] for index in range(0, len(paired), step)]
    return [item[0] for item in sampled], [item[1] for item in sampled]


def _compute_boundary_shift_stats(a0_segments: list[dict[str, Any]], al_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：计算A0与A1边界偏移统计，用于页面摘要卡片。
    参数说明：
    - a0_segments: A0段列表。
    - al_segments: A1段列表。
    返回值：
    - dict[str, Any]: 边界偏移统计结果。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅比较可对齐索引上的“段结束边界”。
    """
    compared_count = min(len(a0_segments), len(al_segments))
    if compared_count == 0:
        return {
            "compared_count": 0,
            "adjusted_count": 0,
            "adjusted_ratio": 0.0,
            "average_abs_shift_seconds": 0.0,
            "max_abs_shift_seconds": 0.0,
        }
    shifts: list[float] = []
    for index in range(compared_count):
        a0_end = _safe_float(a0_segments[index].get("end_time", 0.0))
        al_end = _safe_float(al_segments[index].get("end_time", 0.0))
        shifts.append(abs(al_end - a0_end))
    threshold = 1e-6
    adjusted = [item for item in shifts if item > threshold]
    average_shift = sum(adjusted) / len(adjusted) if adjusted else 0.0
    max_shift = max(adjusted) if adjusted else 0.0
    return {
        "compared_count": compared_count,
        "adjusted_count": len(adjusted),
        "adjusted_ratio": round(len(adjusted) / max(1, compared_count), 6),
        "average_abs_shift_seconds": round(average_shift, 6),
        "max_abs_shift_seconds": round(max_shift, 6),
    }


def _compute_duration_seconds(payload: dict[str, Any]) -> float:
    """
    功能说明：从多图层数据中推断可视化总时长。
    参数说明：
    - payload: 可视化负载字典。
    返回值：
    - float: 推断出的总时长（秒）。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：无可用数据时返回0。
    """
    max_time = 0.0
    for layer_name in ["a0_segments", "al_segments", "b_segments", "s_segments", "lyric_units", "lyric_units_attached", "energy_features"]:
        for item in payload.get(layer_name, []):
            max_time = max(max_time, _safe_float(item.get("end_time", 0.0)))
    for beat in payload.get("beats", []):
        max_time = max(max_time, _safe_float(beat.get("time", 0.0)))
    for onset_item in payload.get("onset_points", []):
        if isinstance(onset_item, dict):
            max_time = max(max_time, _safe_float(onset_item.get("time", 0.0)))
    for onset_time in payload.get("onset_candidates", []):
        max_time = max(max_time, _safe_float(onset_time, 0.0))
    for rms_time in payload.get("vocal_precheck_rms", {}).get("times", []):
        max_time = max(max_time, _safe_float(rms_time, 0.0))
    for rms_time in payload.get("accompaniment_rms", {}).get("times", []):
        max_time = max(max_time, _safe_float(rms_time, 0.0))
    return round(max_time, 6)


def collect_visualization_payload(task_dir: Path) -> dict[str, Any]:
    """
    功能说明：从任务目录聚合模块A V2可视化所需数据。
    参数说明：
    - task_dir: 任务目录路径（如 runs/<task_id>）。
    返回值：
    - dict[str, Any]: 可直接渲染的可视化负载。
    异常说明：
    - RuntimeError: 必需文件缺失时抛错。
    边界条件：仅支持 module_a_v2 目录结构。
    """
    task_dir = task_dir.resolve()
    artifacts_dir = task_dir / "artifacts"
    work_dir = artifacts_dir / "module_a_work_v2"
    algorithm_dir = work_dir / "algorithm"
    window_dir = algorithm_dir / "window"
    timeline_dir = algorithm_dir / "timeline"
    final_dir = algorithm_dir / "final"
    perception_dir = work_dir / "perception"

    module_a_output_path = artifacts_dir / "module_a_output.json"
    stage_big_a0_path = timeline_dir / "stage_big_a0.json"
    stage_big_a1_path = timeline_dir / "stage_big_a1.json"
    stage_segments_final_path = final_dir / "stage_segments_final.json"
    stage_energy_path = final_dir / "stage_energy.json"
    stage_windows_classified_path = window_dir / "stage_windows_classified.json"
    stage_windows_merged_path = window_dir / "stage_windows_merged.json"

    output_data = _read_required_json(module_a_output_path)
    stage_big_a0 = _read_required_json(stage_big_a0_path)
    stage_big_a1 = _read_required_json(stage_big_a1_path)
    stage_segments_final = _read_required_json(stage_segments_final_path)
    stage_energy = _read_required_json(stage_energy_path)

    stage_lyric_attached = _read_optional_json(final_dir / "stage_lyric_attached.json", [])
    stage_lyric_sentence_units_cleaned = _read_optional_json(
        timeline_dir / "stage_lyric_sentence_units_cleaned.json",
        [],
    )
    stage_lyric_sentence_units_head_refined = _read_optional_json(
        timeline_dir / "stage_lyric_sentence_units_head_refined.json",
        [],
    )
    stage_windows_classified = _read_optional_json(stage_windows_classified_path, [])
    stage_windows_merged = _read_optional_json(stage_windows_merged_path, [])
    sentence_split_stats = _read_optional_json(
        perception_dir / "model" / "funasr" / "sentence_split_stats.json",
        {},
    )
    accompaniment_candidates = _read_optional_json(
        perception_dir / "signal" / "librosa" / "accompaniment_candidates.json",
        {},
    )
    vocal_precheck = _read_optional_json(
        perception_dir / "signal" / "librosa" / "vocal_precheck_rms.json",
        {},
    )
    funasr_raw = _read_optional_json(
        perception_dir / "model" / "funasr" / "funasr_raw_response.json",
        {},
    )

    a0_segments = [_normalize_segment_item(item, "A0") for item in list(stage_big_a0)]
    al_segments = [_normalize_segment_item(item, "A1") for item in list(stage_big_a1)]
    b_segments = [_normalize_segment_item(item, "B") for item in list(output_data.get("big_segments", []))]
    s_segments = [_normalize_segment_item(item, "S") for item in list(stage_segments_final)]

    funasr_lyric_sentence_units = _read_optional_json(
        perception_dir / "model" / "funasr" / "lyric_sentence_units.json",
        [],
    )
    lyric_full_source = (
        list(stage_lyric_sentence_units_cleaned)
        if stage_lyric_sentence_units_cleaned
        else (
            list(stage_lyric_sentence_units_head_refined)
            if stage_lyric_sentence_units_head_refined
            else (
                list(funasr_lyric_sentence_units)
                if funasr_lyric_sentence_units
                else (list(stage_lyric_attached) if stage_lyric_attached else list(output_data.get("lyric_units", [])))
            )
        )
    )
    lyric_attached_source = list(stage_lyric_attached) if stage_lyric_attached else list(output_data.get("lyric_units", []))
    lyric_units = [_normalize_lyric_item(item, index) for index, item in enumerate(lyric_full_source)]
    lyric_units_attached = [_normalize_lyric_item(item, index) for index, item in enumerate(lyric_attached_source)]
    energy_features = [_normalize_energy_item(item, index) for index, item in enumerate(list(stage_energy))]
    beats = _normalize_beats(list(output_data.get("beats", [])))

    onset_candidates = [_safe_float(item, 0.0) for item in list(accompaniment_candidates.get("onset_candidates", []))]
    onset_candidates = sorted({round(max(0.0, item), 6) for item in onset_candidates})
    onset_points = _normalize_onset_points(
        onset_points_raw=list(accompaniment_candidates.get("onset_points", [])),
        onset_candidates_raw=onset_candidates,
    )
    onset_candidates = [round(_safe_float(item.get("time", 0.0), 0.0), 6) for item in onset_points]
    accompaniment_rms_times, accompaniment_rms_values = _downsample_series(
        times=list(accompaniment_candidates.get("rms_times", [])),
        values=list(accompaniment_candidates.get("rms_values", [])),
        max_points=3000,
    )

    precheck_rms_times, precheck_rms_values = _downsample_series(
        times=list(vocal_precheck.get("rms_times", [])),
        values=list(vocal_precheck.get("rms_values", [])),
        max_points=3000,
    )
    should_skip_funasr = bool(vocal_precheck.get("should_skip_funasr", False))
    if "skipped" in funasr_raw:
        should_skip_funasr = bool(funasr_raw.get("skipped"))

    # 窗口角色横条优先展示“未并前”窗口，便于观察 tiny 段是否被并掉。
    content_roles_source = list(stage_windows_classified) if stage_windows_classified else list(stage_windows_merged)
    content_roles = [_normalize_segment_item(item, "ROLE") for item in content_roles_source]

    payload: dict[str, Any] = {
        "task_id": str(output_data.get("task_id", task_dir.name)),
        "task_dir": str(task_dir),
        "audio_path": str(output_data.get("audio_path", "")),
        "module_a_output_path": str(module_a_output_path),
        "a0_segments": a0_segments,
        "al_segments": al_segments,
        "b_segments": b_segments,
        "s_segments": s_segments,
        "content_roles": content_roles,
        "beats": beats,
        "lyric_units": lyric_units,
        "lyric_units_attached": lyric_units_attached,
        "energy_features": energy_features,
        "onset_candidates": onset_candidates,
        "onset_points": onset_points,
        "vocal_precheck_rms": {
            "times": precheck_rms_times,
            "values": precheck_rms_values,
            "should_skip_funasr": should_skip_funasr,
            "peak_rms": _safe_float(vocal_precheck.get("peak_rms", 0.0)),
            "active_ratio": _safe_float(vocal_precheck.get("active_ratio", 0.0)),
            "peak_threshold": _safe_float(vocal_precheck.get("peak_threshold", 0.0)),
            "active_ratio_threshold": _safe_float(vocal_precheck.get("active_ratio_threshold", 0.0)),
            "sample_source": str(sentence_split_stats.get("sample_source", "none")),
            "sample_count_raw": int(_safe_float(sentence_split_stats.get("sample_count_raw", 0), 0)),
            "sample_count_kept": int(_safe_float(sentence_split_stats.get("sample_count_kept", 0), 0)),
            "sample_count_outlier": int(_safe_float(sentence_split_stats.get("sample_count_outlier", 0), 0)),
            "dynamic_gap_threshold_seconds": _safe_float(
                sentence_split_stats.get("dynamic_gap_threshold_seconds", 0.35),
                0.35,
            ),
        },
        "accompaniment_rms": {
            "times": accompaniment_rms_times,
            "values": accompaniment_rms_values,
        },
        "summary": {
            "a0_count": len(a0_segments),
            "al_count": len(al_segments),
            "b_count": len(b_segments),
            "s_count": len(s_segments),
            "beat_count": len(beats),
            "lyric_count": len(lyric_units),
            "lyric_attached_count": len(lyric_units_attached),
            "energy_count": len(energy_features),
            "boundary_shift": _compute_boundary_shift_stats(a0_segments=a0_segments, al_segments=al_segments),
        },
    }
    payload["duration_seconds"] = _compute_duration_seconds(payload)
    return payload


def _resolve_audio_for_visualization(
    payload: dict[str, Any],
    output_html_path: Path,
    audio_mode: str,
) -> dict[str, Any]:
    """
    功能说明：根据音频处理策略决定可视化页面引用的音频资源。
    参数说明：
    - payload: 可视化负载。
    - output_html_path: 目标HTML路径。
    - audio_mode: 音频处理模式（copy/none）。
    返回值：
    - dict[str, Any]: 音频可用性与页面相对路径信息。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：音频不存在时返回 unavailable，不中断页面生成。
    """
    source_audio_path = Path(str(payload.get("audio_path", ""))) if payload.get("audio_path") else Path("")
    if audio_mode == "none":
        return {
            "available": False,
            "mode": "none",
            "message": "已禁用音频复制，页面仅展示时间轴。",
            "audio_rel_path": "",
            "audio_abs_path": "",
        }
    if not source_audio_path.exists():
        return {
            "available": False,
            "mode": "copy",
            "message": f"音频文件不存在：{source_audio_path}",
            "audio_rel_path": "",
            "audio_abs_path": str(source_audio_path),
        }
    target_dir = output_html_path.parent
    suffix = source_audio_path.suffix if source_audio_path.suffix else ".wav"
    target_audio_stem = f"{output_html_path.stem}{DEFAULT_AUDIO_COPY_SUFFIX}"
    target_audio_path = target_dir / f"{target_audio_stem}{suffix}"
    shutil.copy2(source_audio_path, target_audio_path)
    return {
        "available": True,
        "mode": "copy",
        "message": "音频已复制到页面同目录，可进行联动播放。",
        "audio_rel_path": target_audio_path.name,
        "audio_abs_path": str(target_audio_path),
    }


def render_visualization_html(
    payload: dict[str, Any],
    output_html_path: Path,
    audio_mode: str = "copy",
) -> Path:
    """
    功能说明：渲染并写出模块A V2可视化HTML页面。
    参数说明：
    - payload: collect_visualization_payload 返回的数据。
    - output_html_path: 目标HTML路径。
    - audio_mode: 音频处理模式（copy/none）。
    返回值：
    - Path: 输出HTML路径。
    异常说明：写入失败时抛 OSError。
    边界条件：页面不依赖外部网络资源，离线可打开。
    """
    output_html_path = output_html_path.resolve()
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    audio_binding = _resolve_audio_for_visualization(
        payload=payload,
        output_html_path=output_html_path,
        audio_mode=str(audio_mode).strip().lower(),
    )
    page_payload = {**payload, "audio_binding": audio_binding}
    payload_json = json.dumps(page_payload, ensure_ascii=False).replace("</", "<\\/")

    html_content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>模块A V2 可视化 - {{task_id}}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --card: #ffffff;
      --text: #1d2a33;
      --muted: #5d6b76;
      --line: #d3dbe2;
      --accent: #0f7b6c;
      --accent-soft: #d9f2ee;
      --danger: #b33f62;
      --lane-h: 36px;
      --label-w: 132px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% -10%, #ecfff8 0%, var(--bg) 42%);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 20px 20px 32px;
    }}
    .title {{
      margin: 0 0 8px 0;
      font-size: 26px;
      line-height: 1.2;
    }}
    .meta {{
      color: var(--muted);
      margin: 0 0 16px 0;
      word-break: break-all;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      min-height: 86px;
    }}
    .card .k {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .card .v {{
      font-size: 24px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .card .sub {{
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .controls {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 12px;
      display: grid;
      gap: 10px;
    }}
    .control-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .control-row label {{
      font-size: 13px;
      color: var(--muted);
    }}
    .layer-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      align-items: center;
    }}
    .layer-item {{
      font-size: 13px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fafcfe;
    }}
    .timeline-wrap {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
    }}
    .timeline-scroll {{
      position: relative;
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #fff 0%, #f9fbfd 100%);
    }}
    .timeline-inner {{
      position: relative;
      min-height: 340px;
      padding-bottom: 10px;
    }}
    .axis {{
      position: sticky;
      left: 0;
      top: 0;
      z-index: 4;
      background: rgba(255, 255, 255, 0.9);
      border-bottom: 1px solid var(--line);
    }}
    .axis-canvas {{
      position: relative;
      height: 30px;
    }}
    .tick {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: #e6edf2;
    }}
    .tick.major {{
      background: #ccd7df;
      width: 1.5px;
    }}
    .tick-label {{
      position: absolute;
      top: 2px;
      transform: translateX(3px);
      font-size: 11px;
      color: var(--muted);
      user-select: none;
      white-space: nowrap;
    }}
    .lane {{
      display: flex;
      align-items: stretch;
      min-height: var(--lane-h);
      border-bottom: 1px dashed #e6edf2;
    }}
    .lane-label {{
      width: var(--label-w);
      flex-shrink: 0;
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      border-right: 1px solid #edf2f6;
      background: #fdfefe;
      position: sticky;
      left: 0;
      z-index: 3;
    }}
    .lane-track {{
      position: relative;
      min-height: var(--lane-h);
      flex: 1;
    }}
    .seg {{
      position: absolute;
      top: 6px;
      height: calc(var(--lane-h) - 12px);
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid rgba(0,0,0,0.08);
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      padding: 0 6px;
      font-size: 11px;
      line-height: calc(var(--lane-h) - 12px);
      color: #102129;
    }}
    .beat-line, .onset-line {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      pointer-events: none;
    }}
    .beat-line {{ background: rgba(15, 123, 108, 0.44); }}
    .beat-line.major {{ background: rgba(15, 123, 108, 0.86); width: 2px; }}
    .onset-line {{ background: rgba(179, 63, 98, 0.15); }}
    .energy-low {{ background: #dbe7ff; }}
    .energy-mid {{ background: #b5f0e0; }}
    .energy-high {{ background: #ffe6a3; }}
    .label-inst {{ background: #f3f5f8; }}
    .label-verse {{ background: #d9f2ee; }}
    .label-chorus {{ background: #ffe8b5; }}
    .label-bridge {{ background: #e8ddff; }}
    .label-start {{ background: #cfe8ff; }}
    .label-end {{ background: #ffe1e1; }}
    .playhead {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #e63946;
      z-index: 10;
      pointer-events: none;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.5);
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      z-index: 99;
      max-width: 360px;
      padding: 8px 10px;
      border-radius: 8px;
      background: rgba(26, 37, 46, 0.96);
      color: #fff;
      font-size: 12px;
      opacity: 0;
      transition: opacity 120ms ease;
      white-space: pre-line;
    }}
    .audio-msg {{
      font-size: 12px;
      color: var(--muted);
    }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <div class="page">
    <h1 class="title">模块A V2 可视化（内部处理 + 最终结果）</h1>
    <p class="meta" id="metaText"></p>
    <div class="cards" id="summaryCards"></div>
    <div class="controls">
      <div class="control-row">
        <audio id="audioPlayer" controls preload="metadata"></audio>
        <span class="audio-msg" id="audioMsg"></span>
      </div>
      <div class="control-row">
        <label for="zoomRange">时间轴缩放（像素/秒）</label>
        <input id="zoomRange" type="range" min="20" max="320" step="10" value="90">
        <strong id="zoomValue">90</strong>
      </div>
      <div class="layer-grid" id="layerToggles"></div>
    </div>
    <div class="timeline-wrap">
      <div class="timeline-scroll" id="timelineScroll">
        <div class="timeline-inner" id="timelineInner">
          <div class="axis">
            <div class="axis-canvas" id="axisCanvas"></div>
          </div>
          <div id="lanesRoot"></div>
          <div class="playhead" id="playhead"></div>
        </div>
      </div>
    </div>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const PAYLOAD = {payload_json};

    function formatTime(sec) {{
      const v = Number(sec) || 0;
      const m = Math.floor(v / 60);
      const s = (v - m * 60).toFixed(3).padStart(6, "0");
      return `${{m}}:${{s}}`;
    }}

    function clip(v, min, max) {{
      return Math.max(min, Math.min(max, v));
    }}

    const layerState = {{
      a0: true,
      al: true,
      b: true,
      s: true,
      role: true,
      beats: true,
      lyrics: true,
      lyrics_attached: true,
      energy: true,
      onset: true,
      precheck: true,
      accompaniment_rms: true,
    }};

    const layerDefs = [
      {{ key: "a0", name: "A0段（Allin1直出）" }},
      {{ key: "al", name: "A1段（单次边界矫正）" }},
      {{ key: "b", name: "B段（最终大段）" }},
      {{ key: "s", name: "S段（最终小段）" }},
      {{ key: "role", name: "ContentRoles（lyric/chant/inst/silence）" }},
      {{ key: "beats", name: "Beats（最终节拍）" }},
      {{ key: "lyrics", name: "Lyrics（全量分句）" }},
      {{ key: "lyrics_attached", name: "Lyrics（挂载到lyric seg）" }},
      {{ key: "energy", name: "Energy（能量特征）" }},
      {{ key: "onset", name: "Onset（伴奏候选+能量）" }},
      {{ key: "precheck", name: "RMS预检（人声能量）" }},
      {{ key: "accompaniment_rms", name: "伴奏RMS（no_vocals）" }},
    ];

    const timelineScroll = document.getElementById("timelineScroll");
    const timelineInner = document.getElementById("timelineInner");
    const axisCanvas = document.getElementById("axisCanvas");
    const lanesRoot = document.getElementById("lanesRoot");
    const playhead = document.getElementById("playhead");
    const tooltip = document.getElementById("tooltip");
    const audioPlayer = document.getElementById("audioPlayer");
    const audioMsg = document.getElementById("audioMsg");
    const zoomRange = document.getElementById("zoomRange");
    const zoomValue = document.getElementById("zoomValue");
    const layerToggles = document.getElementById("layerToggles");

    let pxPerSec = Number(zoomRange.value);
    let duration = Number(PAYLOAD.duration_seconds || 0);
    duration = Math.max(duration, 1.0);
    let playheadRafHandle = null;

    function getLabelClass(label) {{
      const normalized = String(label || "").toLowerCase();
      if (normalized.includes("inst")) return "label-inst";
      if (normalized.includes("chorus")) return "label-chorus";
      if (normalized.includes("verse")) return "label-verse";
      if (normalized.includes("bridge")) return "label-bridge";
      if (normalized.includes("start")) return "label-start";
      if (normalized.includes("end")) return "label-end";
      return "label-inst";
    }}

    function attachTooltip(node, textBuilder) {{
      node.addEventListener("mousemove", (event) => {{
        tooltip.style.left = `${{event.clientX + 12}}px`;
        tooltip.style.top = `${{event.clientY + 12}}px`;
      }});
      node.addEventListener("mouseenter", (event) => {{
        tooltip.textContent = textBuilder(event);
        tooltip.style.opacity = "1";
      }});
      node.addEventListener("mouseleave", () => {{
        tooltip.style.opacity = "0";
      }});
    }}

    function buildCards() {{
      const summary = PAYLOAD.summary || {{}};
      const shift = summary.boundary_shift || {{}};
      const cards = [
        {{
          k: "任务ID",
          v: PAYLOAD.task_id || "-",
          sub: `总时长 ${{formatTime(PAYLOAD.duration_seconds || 0)}}`
        }},
        {{
          k: "段落统计",
          v: `A0 ${{summary.a0_count || 0}} / A1 ${{summary.al_count || 0}} / B ${{summary.b_count || 0}}`,
          sub: `S段 ${{summary.s_count || 0}}`
        }},
        {{
          k: "节拍与歌词",
          v: `Beat ${{summary.beat_count || 0}}`,
          sub: `歌词 ${{summary.lyric_count || 0}}，能量 ${{summary.energy_count || 0}}`
        }},
        {{
          k: "A0->A1 边界调整",
          v: `${{shift.adjusted_count || 0}} / ${{shift.compared_count || 0}}`,
          sub: `平均偏移 ${{Number(shift.average_abs_shift_seconds || 0).toFixed(3)}}s，最大 ${{Number(shift.max_abs_shift_seconds || 0).toFixed(3)}}s`
        }},
        {{
          k: "FunASR预检",
          v: (PAYLOAD.vocal_precheck_rms && PAYLOAD.vocal_precheck_rms.should_skip_funasr) ? "已跳过" : "未跳过",
          sub: `peak=${{Number(PAYLOAD.vocal_precheck_rms?.peak_rms || 0).toFixed(4)}} / th=${{Number(PAYLOAD.vocal_precheck_rms?.peak_threshold || 0).toFixed(4)}}`
        }},
      ];
      const root = document.getElementById("summaryCards");
      root.innerHTML = "";
      cards.forEach((item) => {{
        const node = document.createElement("div");
        node.className = "card";
        node.innerHTML = `<div class="k">${{item.k}}</div><div class="v">${{item.v}}</div><div class="sub">${{item.sub}}</div>`;
        root.appendChild(node);
      }});
    }}

    function buildMeta() {{
      const meta = document.getElementById("metaText");
      const outputPath = PAYLOAD.module_a_output_path || "";
      meta.textContent = `task_dir=${{PAYLOAD.task_dir}} | output=${{outputPath}} | audio=${{PAYLOAD.audio_path || "-"}}`;
    }}

    function buildAudio() {{
      const binding = PAYLOAD.audio_binding || {{}};
      if (binding.available && binding.audio_rel_path) {{
        audioPlayer.src = binding.audio_rel_path;
        audioMsg.textContent = binding.message || "音频已绑定，可联动播放。";
      }} else {{
        audioPlayer.classList.add("hidden");
        audioMsg.textContent = binding.message || "音频不可用，页面仅可查看时间轴。";
      }}
    }}

    function buildLayerToggles() {{
      layerToggles.innerHTML = "";
      layerDefs.forEach((layer) => {{
        const wrap = document.createElement("label");
        wrap.className = "layer-item";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!layerState[layer.key];
        input.addEventListener("change", () => {{
          layerState[layer.key] = input.checked;
          renderTimeline();
        }});
        wrap.appendChild(input);
        const text = document.createTextNode(` ${{layer.name}}`);
        wrap.appendChild(text);
        layerToggles.appendChild(wrap);
      }});
    }}

    function renderAxis(totalWidth) {{
      axisCanvas.innerHTML = "";
      axisCanvas.style.width = `${{totalWidth}}px`;
      const majorStep = duration <= 120 ? 5 : 10;
      const minorStep = majorStep / 2;
      for (let t = 0; t <= duration + 1e-6; t += minorStep) {{
        const x = t * pxPerSec;
        const tick = document.createElement("div");
        tick.className = "tick";
        tick.style.left = `${{x}}px`;
        const isMajor = Math.abs((t / majorStep) - Math.round(t / majorStep)) < 1e-6;
        if (isMajor) {{
          tick.classList.add("major");
          const label = document.createElement("div");
          label.className = "tick-label";
          label.style.left = `${{x}}px`;
          label.textContent = `${{Math.round(t)}}s`;
          axisCanvas.appendChild(label);
        }}
        axisCanvas.appendChild(tick);
      }}
    }}

    function makeLane(key, title) {{
      const lane = document.createElement("div");
      lane.className = "lane";
      lane.dataset.layer = key;
      const label = document.createElement("div");
      label.className = "lane-label";
      label.textContent = title;
      const track = document.createElement("div");
      track.className = "lane-track";
      lane.appendChild(label);
      lane.appendChild(track);
      return {{ lane, track }};
    }}

    function addSegmentBlocks(track, list, clickToSeek = true) {{
      list.forEach((item) => {{
        const left = Number(item.start_time || 0) * pxPerSec;
        const width = Math.max(1, (Number(item.end_time || item.start_time || 0) - Number(item.start_time || 0)) * pxPerSec);
        const node = document.createElement("div");
        node.className = `seg ${{getLabelClass(item.label)}}`;
        node.style.left = `${{left}}px`;
        node.style.width = `${{width}}px`;
        const segText = String(item.segment_id || item.id || "");
        const displayText = String(item.display_text || item.label || "");
        if (segText && displayText) {{
          node.textContent = `${{segText}} | ${{displayText}}`;
        }} else if (displayText) {{
          node.textContent = displayText;
        }} else {{
          node.textContent = segText;
        }}
        node.dataset.start = String(item.start_time || 0);
        attachTooltip(node, () => {{
          return [
            `图层: ${{item.layer || "-"}}`,
            `segment_id: ${{item.segment_id || item.id || "-"}}`,
            `window_id: ${{item.window_id || "-"}}`,
            `big_segment_id: ${{item.big_segment_id || "-"}}`,
            `label: ${{item.label || "-"}}`,
            `role: ${{item.role || "-"}}`,
            `merge_action: ${{item.merge_action || "-"}}`,
            `source_segment_ids: ${{(item.source_segment_ids || []).join(",") || "-"}}`,
            `start: ${{Number(item.start_time || 0).toFixed(3)}}s`,
            `end: ${{Number(item.end_time || 0).toFixed(3)}}s`,
            `duration: ${{Number(item.duration || 0).toFixed(3)}}s`,
          ].join("\\n");
        }});
        if (clickToSeek) {{
          node.addEventListener("click", () => {{
            const seek = Number(item.start_time || 0);
            if (audioPlayer && !audioPlayer.classList.contains("hidden")) {{
              audioPlayer.currentTime = clip(seek, 0, duration);
            }}
            updatePlayhead(seek);
          }});
        }}
        track.appendChild(node);
      }});
    }}

    function addBeatLines(track, beats) {{
      beats.forEach((beat) => {{
        const left = Number(beat.time || 0) * pxPerSec;
        const node = document.createElement("div");
        node.className = "beat-line";
        if (String(beat.type || "").toLowerCase() === "major") {{
          node.classList.add("major");
        }}
        node.style.left = `${{left}}px`;
        attachTooltip(node, () => {{
          return [
            "图层: Beats",
            `time: ${{Number(beat.time || 0).toFixed(3)}}s`,
            `type: ${{beat.type || "-"}}`,
            `source: ${{beat.source || "-"}}`,
          ].join("\\n");
        }});
        track.appendChild(node);
      }});
    }}

    function addOnsetLines(track, onsets) {{
      onsets.forEach((item, index) => {{
        const point = (typeof item === "number")
          ? {{ time: Number(item || 0), energy_raw: 0, energy_norm: 0 }}
          : {{
              time: Number((item && item.time) || 0),
              energy_raw: Number((item && item.energy_raw) || 0),
              energy_norm: Number((item && item.energy_norm) || 0),
            }};
        const left = Number(point.time || 0) * pxPerSec;
        const energyNorm = Math.max(0, Math.min(1, Number(point.energy_norm || 0)));
        const alpha = 0.15 + 0.80 * energyNorm;
        const node = document.createElement("div");
        node.className = "onset-line";
        node.style.left = `${{left}}px`;
        node.style.background = `rgba(179, 63, 98, ${{alpha.toFixed(3)}})`;
        attachTooltip(
          node,
          () => [
            "图层: Onset",
            `index: ${{index + 1}}`,
            `time: ${{Number(point.time || 0).toFixed(3)}}s`,
            `energy_raw: ${{Number(point.energy_raw || 0).toFixed(6)}}`,
            `energy_norm: ${{energyNorm.toFixed(3)}}`,
          ].join("\\n"),
        );
        track.appendChild(node);
      }});
    }}

    function addLyricBlocks(track, lyrics) {{
      lyrics.forEach((item) => {{
        const left = Number(item.start_time || 0) * pxPerSec;
        const width = Math.max(1, (Number(item.end_time || item.start_time || 0) - Number(item.start_time || 0)) * pxPerSec);
        const node = document.createElement("div");
        node.className = "seg label-verse";
        node.style.left = `${{left}}px`;
        node.style.width = `${{width}}px`;
        node.style.opacity = "0.92";
        node.textContent = String(item.display_text || item.text || "");
        attachTooltip(node, () => {{
          return [
            "图层: Lyrics",
            `segment_id: ${{item.segment_id || "-"}}`,
            `text_display: ${{item.display_text || "-"}}`,
            `text_raw: ${{item.text || "-"}}`,
            `confidence: ${{Number(item.confidence || 0).toFixed(3)}}`,
            `start: ${{Number(item.start_time || 0).toFixed(3)}}s`,
            `end: ${{Number(item.end_time || 0).toFixed(3)}}s`,
          ].join("\\n");
        }});
        node.addEventListener("click", () => {{
          const seek = Number(item.start_time || 0);
          if (audioPlayer && !audioPlayer.classList.contains("hidden")) {{
            audioPlayer.currentTime = clip(seek, 0, duration);
          }}
          updatePlayhead(seek);
        }});
        track.appendChild(node);
      }});
    }}

    function addEnergyBlocks(track, energies) {{
      energies.forEach((item) => {{
        const left = Number(item.start_time || 0) * pxPerSec;
        const width = Math.max(1, (Number(item.end_time || item.start_time || 0) - Number(item.start_time || 0)) * pxPerSec);
        const node = document.createElement("div");
        const level = String(item.energy_level || "low").toLowerCase();
        node.className = `seg energy-${{["low","mid","high"].includes(level) ? level : "low"}}`;
        node.style.left = `${{left}}px`;
        node.style.width = `${{width}}px`;
        node.style.opacity = "0.88";
        node.textContent = `${{item.energy_level || "-"}} | ${{item.trend || "-"}}`;
        attachTooltip(node, () => {{
          return [
            "图层: Energy",
            `energy_level: ${{item.energy_level || "-"}}`,
            `trend: ${{item.trend || "-"}}`,
            `rhythm_tension: ${{Number(item.rhythm_tension || 0).toFixed(3)}}`,
            `start: ${{Number(item.start_time || 0).toFixed(3)}}s`,
            `end: ${{Number(item.end_time || 0).toFixed(3)}}s`,
          ].join("\\n");
        }});
        track.appendChild(node);
      }});
    }}

    function addPrecheckLine(track, precheck) {{
      const times = precheck?.times || [];
      const values = precheck?.values || [];
      if (!times.length || !values.length) {{
        return;
      }}
      const maxVal = Math.max(
        ...values.map((v) => Number(v || 0)),
        Number(precheck.peak_threshold || 0),
        1e-6,
      );
      const laneHeight = 34;
      for (let i = 1; i < Math.min(times.length, values.length); i += 1) {{
        const x1 = Number(times[i - 1]) * pxPerSec;
        const x2 = Number(times[i]) * pxPerSec;
        const y1 = laneHeight - (Number(values[i - 1]) / maxVal) * laneHeight;
        const y2 = laneHeight - (Number(values[i]) / maxVal) * laneHeight;
        const dx = Math.max(1, x2 - x1);
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);
        const line = document.createElement("div");
        line.style.position = "absolute";
        line.style.left = `${{x1}}px`;
        line.style.top = `${{y1 + 1}}px`;
        line.style.width = `${{length}}px`;
        line.style.height = "1.5px";
        line.style.background = "rgba(179,63,98,0.86)";
        line.style.transformOrigin = "0 0";
        line.style.transform = `rotate(${{angle}}deg)`;
        track.appendChild(line);
      }}
      const thresholdY = laneHeight - (Number(precheck.peak_threshold || 0) / maxVal) * laneHeight;
      const threshold = document.createElement("div");
      threshold.style.position = "absolute";
      threshold.style.left = "0";
      threshold.style.right = "0";
      threshold.style.top = `${{thresholdY}}px`;
      threshold.style.height = "1px";
      threshold.style.background = "rgba(15,123,108,0.45)";
      track.appendChild(threshold);
      attachTooltip(threshold, () => {{
        return [
          "图层: 人声RMS预检",
          `should_skip_funasr: ${{precheck.should_skip_funasr ? "true" : "false"}}`,
          `peak_rms: ${{Number(precheck.peak_rms || 0).toFixed(5)}}`,
          `active_ratio: ${{Number(precheck.active_ratio || 0).toFixed(4)}}`,
          `peak_threshold: ${{Number(precheck.peak_threshold || 0).toFixed(5)}}`,
          `active_ratio_threshold: ${{Number(precheck.active_ratio_threshold || 0).toFixed(4)}}`,
          `sample_source: ${{String(precheck.sample_source || "none")}}`,
          `sample_count_raw: ${{Number(precheck.sample_count_raw || 0)}}`,
          `sample_count_kept: ${{Number(precheck.sample_count_kept || 0)}}`,
          `sample_count_outlier: ${{Number(precheck.sample_count_outlier || 0)}}`,
          `dynamic_gap_threshold: ${{Number(precheck.dynamic_gap_threshold_seconds || 0).toFixed(3)}}s`,
        ].join("\\n");
      }});
    }}

    function addAccompanimentRmsLine(track, rmsData) {{
      const times = rmsData?.times || [];
      const values = rmsData?.values || [];
      if (!times.length || !values.length) {{
        return;
      }}
      const maxVal = Math.max(...values.map((v) => Number(v || 0)), 1e-6);
      const laneHeight = 34;
      for (let i = 1; i < Math.min(times.length, values.length); i += 1) {{
        const x1 = Number(times[i - 1]) * pxPerSec;
        const x2 = Number(times[i]) * pxPerSec;
        const y1 = laneHeight - (Number(values[i - 1]) / maxVal) * laneHeight;
        const y2 = laneHeight - (Number(values[i]) / maxVal) * laneHeight;
        const dx = Math.max(1, x2 - x1);
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);
        const line = document.createElement("div");
        line.style.position = "absolute";
        line.style.left = `${{x1}}px`;
        line.style.top = `${{y1 + 1}}px`;
        line.style.width = `${{length}}px`;
        line.style.height = "1.5px";
        line.style.background = "rgba(42, 114, 198, 0.88)";
        line.style.transformOrigin = "0 0";
        line.style.transform = `rotate(${{angle}}deg)`;
        attachTooltip(line, () => {{
          return [
            "图层: 伴奏RMS",
            `time: ${{Number(times[i] || 0).toFixed(3)}}s`,
            `rms_value: ${{Number(values[i] || 0).toFixed(6)}}`,
          ].join("\\n");
        }});
        track.appendChild(line);
      }}
    }}

    function renderTimeline() {{
      const totalWidth = Math.max(1200, duration * pxPerSec + 32);
      timelineInner.style.width = `${{totalWidth + 140}}px`;
      lanesRoot.innerHTML = "";
      renderAxis(totalWidth);

      const definitions = [
        {{ key: "a0", title: "A0段（stage_big_a0）", type: "segments", data: PAYLOAD.a0_segments }},
        {{ key: "al", title: "A1段（timeline/stage_big_a1）", type: "segments", data: PAYLOAD.al_segments }},
        {{ key: "b", title: "B段（module_a_output.big_segments）", type: "segments", data: PAYLOAD.b_segments }},
        {{ key: "s", title: "S段（final/stage_segments_final）", type: "segments", data: PAYLOAD.s_segments }},
        {{ key: "role", title: "窗口角色（window/stage_windows_classified，含未并tiny）", type: "roles", data: PAYLOAD.content_roles }},
        {{ key: "beats", title: "Beats（module_a_output.beats）", type: "beats", data: PAYLOAD.beats }},
        {{ key: "lyrics", title: "Lyrics（全量分句：timeline_cleaned/funasr）", type: "lyrics", data: PAYLOAD.lyric_units }},
        {{ key: "lyrics_attached", title: "Lyrics（挂载结果：final/stage_lyric_attached）", type: "lyrics", data: PAYLOAD.lyric_units_attached }},
        {{ key: "energy", title: "Energy（stage_energy）", type: "energy", data: PAYLOAD.energy_features }},
        {{ key: "onset", title: "Onset（伴奏候选+能量）", type: "onset", data: PAYLOAD.onset_points }},
        {{ key: "precheck", title: "人声RMS预检（FunASR跳过判据）", type: "precheck", data: PAYLOAD.vocal_precheck_rms }},
        {{ key: "accompaniment_rms", title: "伴奏RMS（no_vocals）", type: "accompaniment_rms", data: PAYLOAD.accompaniment_rms }},
      ];

      definitions.forEach((def) => {{
        const {{ lane, track }} = makeLane(def.key, def.title);
        track.style.width = `${{totalWidth}}px`;
        if (!layerState[def.key]) {{
          lane.classList.add("hidden");
        }}
        if (def.type === "segments") {{
          addSegmentBlocks(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "roles") {{
          addSegmentBlocks(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "beats") {{
          addBeatLines(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "lyrics") {{
          addLyricBlocks(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "energy") {{
          addEnergyBlocks(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "onset") {{
          addOnsetLines(track, Array.isArray(def.data) ? def.data : []);
        }} else if (def.type === "precheck") {{
          addPrecheckLine(track, def.data || {{}});
        }} else if (def.type === "accompaniment_rms") {{
          addAccompanimentRmsLine(track, def.data || {{}});
        }}
        lanesRoot.appendChild(lane);
      }});
      updatePlayhead(audioPlayer.currentTime || 0);
    }}

    function updatePlayhead(timeSec) {{
      const safeTime = clip(Number(timeSec) || 0, 0, duration);
      playhead.style.left = `${{safeTime * pxPerSec + Number(getComputedStyle(document.documentElement).getPropertyValue("--label-w").replace("px","") || 132)}}px`;
    }}

    function stopPlayheadAnimation() {{
      if (playheadRafHandle !== null) {{
        cancelAnimationFrame(playheadRafHandle);
        playheadRafHandle = null;
      }}
    }}

    function animatePlayhead() {{
      updatePlayhead(audioPlayer.currentTime || 0);
      if (!audioPlayer.paused && !audioPlayer.ended) {{
        playheadRafHandle = requestAnimationFrame(animatePlayhead);
      }} else {{
        playheadRafHandle = null;
      }}
    }}

    function startPlayheadAnimation() {{
      stopPlayheadAnimation();
      playheadRafHandle = requestAnimationFrame(animatePlayhead);
    }}

    zoomRange.addEventListener("input", () => {{
      pxPerSec = Number(zoomRange.value);
      zoomValue.textContent = String(pxPerSec);
      renderTimeline();
    }});

    audioPlayer.addEventListener("timeupdate", () => {{
      if (audioPlayer.paused || audioPlayer.ended) {{
        updatePlayhead(audioPlayer.currentTime || 0);
      }}
    }});
    audioPlayer.addEventListener("seeked", () => {{
      updatePlayhead(audioPlayer.currentTime || 0);
      if (!audioPlayer.paused && !audioPlayer.ended) {{
        startPlayheadAnimation();
      }}
    }});
    audioPlayer.addEventListener("seeking", () => {{
      stopPlayheadAnimation();
    }});
    audioPlayer.addEventListener("play", () => {{
      updatePlayhead(audioPlayer.currentTime || 0);
      startPlayheadAnimation();
    }});
    audioPlayer.addEventListener("pause", () => {{
      stopPlayheadAnimation();
      updatePlayhead(audioPlayer.currentTime || 0);
    }});
    audioPlayer.addEventListener("ended", () => {{
      stopPlayheadAnimation();
      updatePlayhead(audioPlayer.currentTime || 0);
    }});
    audioPlayer.addEventListener("waiting", () => {{
      stopPlayheadAnimation();
      updatePlayhead(audioPlayer.currentTime || 0);
    }});
    timelineScroll.addEventListener("click", (event) => {{
      if (event.target !== timelineScroll) return;
      const rect = timelineScroll.getBoundingClientRect();
      const relativeX = event.clientX - rect.left + timelineScroll.scrollLeft - 132;
      const time = clip(relativeX / pxPerSec, 0, duration);
      if (audioPlayer && !audioPlayer.classList.contains("hidden")) {{
        audioPlayer.currentTime = time;
      }}
      updatePlayhead(time);
    }});

    buildMeta();
    buildCards();
    buildAudio();
    buildLayerToggles();
    renderTimeline();
  </script>
</body>
</html>
""".replace("{task_id}", str(payload.get("task_id", "-")))

    output_html_path.write_text(html_content, encoding="utf-8")
    return output_html_path
