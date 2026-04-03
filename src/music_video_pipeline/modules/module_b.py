"""
文件用途：实现模块 B（视觉脚本生成）的 MVP 版本。
核心流程：读取模块 A 输出，调用分镜生成器并落盘模块 B JSON。
输入输出：输入 RuntimeContext，输出 ModuleBOutput JSON 路径。
依赖说明：依赖项目内脚本生成器工厂与 JSON 工具。
维护说明：接入真实 LLM 时只替换生成器，不改模块出口契约。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：分镜生成器工厂
from music_video_pipeline.generators import build_script_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def run_module_b(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 B 并输出分镜脚本 JSON。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 B 输出 JSON 路径。
    异常说明：输入文件缺失或契约不合法时抛异常。
    边界条件：生成器模式未知时自动降级为 mock。
    """
    context.logger.info("模块B开始执行，task_id=%s", context.task_id)
    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    generator = build_script_generator(mode=context.config.mode.script_generator, logger=context.logger)
    module_b_output = generator.generate(module_a_output=module_a_output)
    module_b_output = _enrich_shots_with_segment_meta(
        shots=module_b_output,
        module_a_output=module_a_output,
        instrumental_labels=context.config.module_a.instrumental_labels,
    )
    validate_module_b_output(module_b_output)

    output_path = context.artifacts_dir / "module_b_output.json"
    write_json(output_path, module_b_output)
    context.logger.info("模块B执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


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
    lyric_units_raw = module_a_output.get("lyric_units", None)
    if not isinstance(segments, list):
        segments = []
    if not isinstance(big_segments, list):
        big_segments = []
    lyric_units_available = isinstance(lyric_units_raw, list)
    lyric_segment_ids: set[str] = set()
    if lyric_units_available:
        for item in lyric_units_raw:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id", "")).strip()
            if segment_id:
                lyric_segment_ids.add(segment_id)

    big_segment_map = {
        str(item.get("segment_id", "")).strip(): item
        for item in big_segments
        if isinstance(item, dict)
    }
    instrumental_set = {str(label).strip().lower() for label in instrumental_labels}

    enhanced_shots: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        segment = _resolve_segment_for_shot(shot=shot, shot_index=shot_index, segments=segments)

        big_segment_id = ""
        big_segment_label = ""
        segment_label = ""
        audio_role = "vocal"
        if segment:
            segment_label = str(segment.get("label", "")).strip()
            normalized_label = segment_label.lower()
            segment_id = str(segment.get("segment_id", "")).strip()
            has_lyric = segment_id in lyric_segment_ids if lyric_units_available else True
            if normalized_label in instrumental_set:
                audio_role = "instrumental"
            elif not has_lyric:
                # 规则约束：人声标签段里无歌词单元时，按器乐段处理。
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
