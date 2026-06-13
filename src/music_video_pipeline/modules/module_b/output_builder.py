"""
文件用途：提供模块 B 的输出构建函数。
核心流程：聚合 role3/role4 产物并整理为模块 B 标准输出结构。
输入输出：输入角色产物与模块 A 信息，输出 module_b_output.json 所需的分镜数组。
依赖说明：依赖 markdown_contracts 解析 role3/role4 产物。
维护说明：输出结构应与 validate_module_b_output 的 required_keys 保持一致。
"""

from pathlib import Path
from typing import Any
import re

from music_video_pipeline.modules.module_b.artifact_paths import (
    get_module_b_role_result_path,
    get_module_b_streaming_dir,
)
from music_video_pipeline.modules.module_b.markdown_contracts import (
    parse_shot_plans,
)
from music_video_pipeline.modules.module_b.orchestrator import (
    _parse_subject_descriptions,
    _build_shot_id,
)


def build_module_b_output(
    done_unit_records: list[dict[str, Any]],
    module_a_output: dict[str, Any],
    instrumental_labels: list[str],
    artifacts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """构建模块 B 最终输出数组。

    读取 role3/role4 的 markdown 产物，合并时间信息，输出满足
    validate_module_b_output 的分镜数组。
    """
    del instrumental_labels

    if artifacts_dir is None:
        return []

    # 1. 解析 role3 → {segment_id: {scene_desc_zh, big_segment_id, remotion_id}}
    role3_path = get_module_b_role_result_path(artifacts_dir, "role3")
    segment_meta: dict[str, dict[str, str]] = {}
    if role3_path.exists():
        try:
            role3_markdown = role3_path.read_text(encoding="utf-8")
            for sp in parse_shot_plans(role3_markdown):
                segment_meta[sp.segment_id] = {
                    "scene_desc_zh": sp.scene_desc_zh,
                    "big_segment_id": sp.big_segment_id,
                    "remotion_id": sp.remotion_id,
                }
        except Exception:
            pass

    # 2. 解析 role3 streaming 文件，构建有效 shot_id 集合 + shot_subject_kind 映射
    #    与 role4 选择器同源，确保只保留 role3 当前产出的 shot
    valid_shot_ids: set[str] = set()
    shot_subject_kind_map: dict[str, str] = {}  # shot_id → shot_subject_kind
    role3_streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role3")
    if role3_streaming_dir.exists():
        for stream_path in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            except Exception:
                continue
            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()
            for block in re.split(r"\n(?=### )", text):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n")
                heading = lines[0].strip()
                if heading.startswith("## "):
                    current_big = heading[3:].strip().split(" / ")[0].strip()
                    continue
                if not heading.startswith("### "):
                    continue
                seg_id = heading[4:].strip()
                if not seg_id:
                    continue
                scene_desc = ""
                remotion_id = ""
                seg_subject_kind = "human"
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith("- scene_desc_zh:"):
                        scene_desc = stripped[len("- scene_desc_zh:"):].strip()
                    elif stripped.startswith("- remotion_id:"):
                        remotion_id = stripped[len("- remotion_id:"):].strip()
                    elif stripped.startswith("- shot_subject_kind:"):
                        seg_subject_kind = stripped[len("- shot_subject_kind:"):].strip()
                subjects = _parse_subject_descriptions(scene_desc, remotion_id)
                for subj_idx, _ in enumerate(subjects, start=1):
                    shot_id = _build_shot_id(seg_id, subj_idx)
                    valid_shot_ids.add(shot_id)
                    shot_subject_kind_map[shot_id] = seg_subject_kind

    # 3. 解析 role4 per-shot streaming 文件 → {shot_id: {7 prompt 字段}}
    #    文件名: role4_prompt_output.streaming.{shot_id}.md
    #    内容格式: 纯 - field_name: value 行（无 ## 标题、无 ```md 围栏）
    role4_fields = [
        "subject_kind",
        "keyframe_prompt_start_zh", "keyframe_prompt_start_en",
        "keyframe_prompt_end_zh", "keyframe_prompt_end_en",
        "video_prompt_zh", "video_prompt_en",
    ]
    prompt_map: dict[str, dict[str, str]] = {}
    role4_streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role4")
    if role4_streaming_dir.exists():
        for streaming_path in sorted(role4_streaming_dir.glob("role4_prompt_output.streaming.*.md")):
            try:
                shot_id = streaming_path.stem.split(".streaming.", 1)[-1]
                if not shot_id:
                    continue
                # 跳过不在 role3 有效 shot 集合中的残留文件
                if valid_shot_ids and shot_id not in valid_shot_ids:
                    continue
                text = streaming_path.read_text(encoding="utf-8")
                fields: dict[str, str] = {}
                for f in role4_fields:
                    m = re.search(rf"^- {re.escape(f)}:\s*(.*)", text, re.MULTILINE)
                    fields[f] = m.group(1).strip() if m else ""
                prompt_map[shot_id] = fields
            except Exception:
                continue
    # 4. 构建 segment 时间映射（优先 Module A，回退 done_unit_records）
    seg_timing: dict[str, dict[str, float]] = {}
    for segment_item in (module_a_output.get("segments", []) if isinstance(module_a_output, dict) else []):
        if not isinstance(segment_item, dict):
            continue
        segment_id = str(segment_item.get("segment_id", "")).strip()
        if not segment_id:
            continue
        seg_timing[segment_id] = {
            "start_time": float(segment_item.get("start_time", 0) or 0),
            "end_time": float(segment_item.get("end_time", 0) or 0),
        }
    for rec in done_unit_records:
        seg_id = str(rec.get("unit_id", "")).strip()
        if seg_id and seg_id not in seg_timing:
            seg_timing[seg_id] = {
                "start_time": float(rec.get("start_time", 0) or 0),
                "end_time": float(rec.get("end_time", 0) or 0),
            }

    # 5. 合并：遍历 prompt_map（shot-level），从 shot_id 提取 seg_number 匹配 segment_meta
    output: list[dict[str, Any]] = []

    for shot_id in sorted(prompt_map):
        fields = prompt_map[shot_id]
        seg_number = ""
        shot_m = re.match(r"shot_(\d+)_\d+$", shot_id)
        if shot_m:
            seg_number = shot_m.group(1)
        seg_key = f"seg_{seg_number}" if seg_number else ""
        if not seg_key and shot_id.startswith("seg_"):
            seg_key = shot_id
        meta = segment_meta.get(seg_key, {})
        timing = seg_timing.get(seg_key, {})
        output.append({
            "shot_id": shot_id,
            "subject_kind": fields.get("subject_kind") or shot_subject_kind_map.get(shot_id, "human"),
            "big_segment_id": meta.get("big_segment_id", ""),
            "remotion_id": meta.get("remotion_id", ""),
            "start_time": timing.get("start_time", 0.0),
            "end_time": timing.get("end_time", 0.0),
            "scene_desc": meta.get("scene_desc_zh", ""),
            "keyframe_prompt_start_zh": fields.get("keyframe_prompt_start_zh", ""),
            "keyframe_prompt_start_en": fields.get("keyframe_prompt_start_en", ""),
            "keyframe_prompt_end_zh": fields.get("keyframe_prompt_end_zh", ""),
            "keyframe_prompt_end_en": fields.get("keyframe_prompt_end_en", ""),
            "video_prompt_zh": fields.get("video_prompt_zh", ""),
            "video_prompt_en": fields.get("video_prompt_en", ""),
        })

    return output




