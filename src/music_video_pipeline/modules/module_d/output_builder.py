"""
文件用途：构建模块 D 产物摘要对象。
核心流程：将单元状态聚合结果与终拼结果组装为可落盘 JSON。
输入输出：输入任务信息与终拼结果，输出模块 D 摘要字典。
依赖说明：依赖标准库 typing/pathlib。
维护说明：本摘要仅用于排障追踪，不作为下游模块输入契约。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any


def build_module_d_output(
    task_id: str,
    output_video_path: Path,
    done_unit_records: list[dict[str, Any]],
    concat_result: dict[str, Any],
    shot_payload_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    功能说明：组装模块 D 输出摘要对象。
    参数说明：
    - task_id: 任务唯一标识。
    - output_video_path: 最终视频路径。
    - done_unit_records: 已完成单元记录数组。
    - concat_result: 终拼执行信息。
    - shot_payload_map: 可选，shot_id到原始分镜/帧载荷的映射，用于附加回写提示词字段。
    返回值：
    - dict[str, Any]: 模块 D 输出摘要数据。
    异常说明：无。
    边界条件：done_unit_records 由上层保证按 unit_index 稳定排序。
    """
    safe_shot_payload_map = shot_payload_map or {}
    segment_items: list[dict[str, Any]] = []
    for item in done_unit_records:
        shot_id = str(item.get("unit_id", ""))
        segment_item = {
            "shot_id": shot_id,
            "segment_path": str(item.get("artifact_path", "")),
            "start_time": float(item.get("start_time", 0.0)),
            "end_time": float(item.get("end_time", 0.0)),
            "duration": float(item.get("duration", 0.0)),
            "unit_index": int(item.get("unit_index", 0)),
        }
        shot_payload = safe_shot_payload_map.get(shot_id, {})
        if isinstance(shot_payload, dict):
            for optional_key in [
                "scene_desc",
                "keyframe_prompt",
                "video_prompt",
                "keyframe_prompt_zh",
                "keyframe_prompt_en",
                "video_prompt_zh",
                "video_prompt_en",
            ]:
                optional_value = shot_payload.get(optional_key, "")
                if isinstance(optional_value, str) and optional_value.strip():
                    segment_item[optional_key] = optional_value.strip()
        segment_items.append(segment_item)
    return {
        "task_id": task_id,
        "output_video_path": str(output_video_path),
        "concat_mode": str(concat_result.get("mode", "unknown")),
        "copy_fallback_triggered": bool(concat_result.get("copy_fallback_triggered", False)),
        "segment_items": segment_items,
    }
