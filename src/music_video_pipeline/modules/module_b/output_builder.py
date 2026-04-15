"""
文件用途：构建模块 B 输出数组并提供分镜增强逻辑。
核心流程：读取已完成单元分镜文件，按顺序聚合后补充段落与音频角色元信息。
输入输出：输入单元记录与模块A输出，输出模块B标准分镜数组。
依赖说明：依赖标准库 pathlib/typing 与项目内 JSON 工具。
维护说明：输出结构需保持与模块 C 消费逻辑兼容。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：JSON读取工具
from music_video_pipeline.io_utils import read_json


def build_module_b_output(
    done_unit_records: list[dict[str, Any]],
    module_a_output: dict[str, Any],
    instrumental_labels: list[str],
) -> list[dict[str, Any]]:
    """
    功能说明：聚合已完成单元分镜并输出模块 B 标准数组。
    参数说明：
    - done_unit_records: 模块B已完成单元记录数组（需含 unit_index/artifact_path）。
    - module_a_output: 模块A输出字典。
    - instrumental_labels: 器乐标签集合配置。
    返回值：
    - list[dict[str, Any]]: 模块 B 输出分镜数组。
    异常说明：
    - RuntimeError: 单元文件不存在或内容非法时抛出。
    边界条件：shot_id 按 unit_index 重新标准化，确保顺序稳定。
    """
    ordered_records = sorted(done_unit_records, key=lambda item: int(item.get("unit_index", 0)))
    shots: list[dict[str, Any]] = []
    for record in ordered_records:
        artifact_path_text = str(record.get("artifact_path", "")).strip()
        if not artifact_path_text:
            raise RuntimeError(f"模块B输出聚合失败：存在空 artifact_path，record={record}")
        artifact_path = Path(artifact_path_text)
        if not artifact_path.exists():
            raise RuntimeError(f"模块B输出聚合失败：单元分镜文件不存在，path={artifact_path}")

        shot_obj = read_json(artifact_path)
        if not isinstance(shot_obj, dict):
            raise RuntimeError(f"模块B输出聚合失败：单元分镜内容不是dict，path={artifact_path}")

        unit_index = int(record.get("unit_index", 0))
        normalized_shot = dict(shot_obj)
        normalized_shot["shot_id"] = f"shot_{unit_index + 1:03d}"
        if "start_time" not in normalized_shot:
            normalized_shot["start_time"] = float(record.get("start_time", 0.0))
        if "end_time" not in normalized_shot:
            normalized_shot["end_time"] = float(record.get("end_time", normalized_shot["start_time"]))
        shots.append(normalized_shot)

    return _enrich_shots_with_segment_meta(
        shots=shots,
        module_a_output=module_a_output,
        instrumental_labels=instrumental_labels,
    )


def _enrich_shots_with_segment_meta(
    shots: list[dict[str, Any]],
    module_a_output: dict[str, Any],
    instrumental_labels: list[str],
) -> list[dict[str, Any]]:
    """
    功能说明：为分镜补充大段落归属与器乐/人声标记。
    参数说明：
    - shots: 原始分镜数组。
    - module_a_output: 模块A输出字典。
    - instrumental_labels: 器乐标签集合配置。
    返回值：
    - list[dict[str, Any]]: 增强后的分镜数组。
    异常说明：无。
    边界条件：当无法匹配 segment 时写入空值，供下游展示为“<未知>”。
    """
    segments = module_a_output.get("segments", [])
    big_segments = module_a_output.get("big_segments", [])
    if not isinstance(segments, list):
        segments = []
    if not isinstance(big_segments, list):
        big_segments = []

    big_segment_map = {
        str(item.get("segment_id", "")).strip(): item
        for item in big_segments
        if isinstance(item, dict)
    }
    instrumental_set = {str(label).strip().lower() for label in instrumental_labels}
    instrumental_set.add("inst")
    vocal_role_set = {"lyric", "chant"}
    instrumental_role_set = {"inst", "silence"}

    enhanced_shots: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        segment = _resolve_segment_for_shot(shot=shot, shot_index=shot_index, segments=segments)

        big_segment_id = ""
        big_segment_label = ""
        segment_label = ""
        segment_role = ""
        audio_role = "vocal"
        if segment:
            segment_label = str(segment.get("label", "")).strip()
            normalized_label = segment_label.lower()
            segment_role = str(segment.get("role", "")).strip().lower()
            if segment_role in vocal_role_set:
                audio_role = "vocal"
            elif segment_role in instrumental_role_set:
                audio_role = "instrumental"
            elif normalized_label in instrumental_set:
                audio_role = "instrumental"
            else:
                audio_role = "vocal"

            big_segment_id = str(segment.get("big_segment_id", "")).strip()
            big_segment_obj = big_segment_map.get(big_segment_id, {})
            big_segment_label = str(big_segment_obj.get("label", "")).strip()

        enhanced_shot = dict(shot)
        enhanced_shot["big_segment_id"] = big_segment_id
        enhanced_shot["big_segment_label"] = big_segment_label
        enhanced_shot["segment_label"] = segment_label
        if segment_role:
            enhanced_shot["segment_role"] = segment_role
        enhanced_shot["audio_role"] = audio_role
        enhanced_shots.append(enhanced_shot)
    return enhanced_shots


def _resolve_segment_for_shot(
    shot: dict[str, Any],
    shot_index: int,
    segments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    功能说明：为单个分镜匹配最对应的小段落。
    参数说明：
    - shot: 当前分镜。
    - shot_index: 分镜索引。
    - segments: 模块A小段落数组。
    返回值：
    - dict[str, Any] | None: 匹配到的小段落，未匹配返回 None。
    异常说明：无。
    边界条件：优先按索引一一对应，失败时按时间重叠最大回退。
    """
    if shot_index < len(segments):
        candidate = segments[shot_index]
        if isinstance(candidate, dict):
            overlap = _calculate_time_overlap_seconds(
                left_start=shot.get("start_time", 0.0),
                left_end=shot.get("end_time", shot.get("start_time", 0.0)),
                right_start=candidate.get("start_time", 0.0),
                right_end=candidate.get("end_time", candidate.get("start_time", 0.0)),
            )
            if overlap > 1e-6:
                return candidate
    return _find_best_overlap_segment_for_shot(shot=shot, segments=segments)


def _find_best_overlap_segment_for_shot(
    shot: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    功能说明：按时间重叠度回退匹配小段落。
    参数说明：
    - shot: 当前分镜。
    - segments: 模块A小段落数组。
    返回值：
    - dict[str, Any] | None: 最佳重叠段落，若无法计算则返回 None。
    异常说明：无。
    边界条件：重叠相同则优先选择起点更接近的段落。
    """
    if not segments:
        return None
    try:
        shot_start = float(shot.get("start_time", 0.0))
        shot_end = float(shot.get("end_time", shot_start))
    except (TypeError, ValueError):
        return None

    shot_end = max(shot_start, shot_end)
    best_segment: dict[str, Any] | None = None
    best_overlap = -1.0
    best_start_gap = float("inf")
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            seg_start = float(segment.get("start_time", 0.0))
            seg_end = float(segment.get("end_time", seg_start))
        except (TypeError, ValueError):
            continue
        seg_end = max(seg_start, seg_end)
        overlap = _calculate_time_overlap_seconds(
            left_start=shot_start,
            left_end=shot_end,
            right_start=seg_start,
            right_end=seg_end,
        )
        start_gap = abs(seg_start - shot_start)
        if overlap > best_overlap + 1e-6:
            best_segment = segment
            best_overlap = overlap
            best_start_gap = start_gap
            continue
        if abs(overlap - best_overlap) <= 1e-6 and start_gap < best_start_gap:
            best_segment = segment
            best_start_gap = start_gap
    return best_segment


def _calculate_time_overlap_seconds(
    left_start: Any,
    left_end: Any,
    right_start: Any,
    right_end: Any,
) -> float:
    """
    功能说明：计算两个时间区间的重叠时长。
    参数说明：
    - left_start/left_end: 区间1起止时间。
    - right_start/right_end: 区间2起止时间。
    返回值：
    - float: 重叠秒数，异常或无重叠返回 0。
    异常说明：无（内部吞掉非法值）。
    边界条件：任一端点非法时返回 0。
    """
    try:
        left_start_val = float(left_start)
        left_end_val = max(left_start_val, float(left_end))
        right_start_val = float(right_start)
        right_end_val = max(right_start_val, float(right_end))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(left_end_val, right_end_val) - max(left_start_val, right_start_val))
