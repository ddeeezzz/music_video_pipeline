"""
文件用途：实现模块 C（图像生成）的单元级断点重试编排入口。
核心流程：读取模块 B 分镜，按 shot 同步单元状态，并行执行待处理单元，汇总输出模块 C 清单。
输入输出：输入 RuntimeContext，输出模块 C 清单 JSON 路径。
依赖说明：依赖模块 C 子组件、生成器工厂与 JSON 工具。
维护说明：保持模块级状态机不变，单元级状态仅作为 C 内部恢复能力。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则解析流式文件
import re
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：关键帧生成器工厂
from music_video_pipeline.generators import build_keyframe_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块C单元执行器
from music_video_pipeline.modules.module_c.executor import execute_one_unit_with_retry, execute_units_with_retry
# 项目内模块：模块C输出对象构建器
from music_video_pipeline.modules.module_c.output_builder import build_module_c_output
# 项目内模块：模块C单元模型工具
from music_video_pipeline.modules.module_c.unit_models import build_module_c_units, build_unit_map, build_unit_sync_payload
# 项目内模块：模块 B shot 衍生工具
from music_video_pipeline.modules.module_b.orchestrator import _build_shot_id, _parse_subject_descriptions
# 项目内模块：模块 B 产物路径工具
from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_streaming_dir
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_b_output


def _overlay_role4_streaming_prompt(shot: dict[str, object], artifacts_dir: Path) -> dict[str, object]:
    """用 role4 流式文件的 prompt 字段覆盖 shot 中的旧数据，返回新的 shot dict。"""
    shot_id = str(shot.get("shot_id", "")).strip()
    if not shot_id:
        return shot
    streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role4")
    stream_path = streaming_dir / f"role4_prompt_output.streaming.{shot_id}.md"
    if not stream_path.exists():
        return shot
    try:
        text = stream_path.read_text(encoding="utf-8")
    except Exception:
        return shot
    fields = [
        "keyframe_prompt_start_zh",
        "keyframe_prompt_start_en",
        "keyframe_prompt_end_zh",
        "keyframe_prompt_end_en",
        "video_prompt_zh",
        "video_prompt_en",
    ]
    overlays: dict[str, str] = {}
    for field in fields:
        m = re.search(rf"^- {re.escape(field)}:\s*(.*)", text, re.MULTILINE)
        value = m.group(1).strip() if m else ""
        if value:
            overlays[field] = value
    if not overlays:
        return shot
    result = dict(shot)
    result.update(overlays)
    return result


def _merge_frame_items_with_shot_payloads(
    frame_items: list[dict[str, object]],
    units_by_id: dict[str, object],
) -> list[dict[str, object]]:
    """将模块 C 状态库反推的帧路径与模块 B shot 元数据合并。"""
    merged_items: list[dict[str, object]] = []
    for frame_item in frame_items:
        shot_id = str(frame_item.get("shot_id", "")).strip()
        unit_obj = units_by_id.get(shot_id)
        shot_payload = getattr(unit_obj, "shot", {}) if unit_obj is not None else {}
        merged_item = dict(frame_item)
        if isinstance(shot_payload, dict):
            for key, value in shot_payload.items():
                if key not in merged_item:
                    merged_item[key] = value
        merged_item["shot_id"] = shot_id
        merged_items.append(merged_item)
    return merged_items


def _load_seg_timing(artifacts_dir: Path) -> dict[str, dict[str, float]]:
    """从 module_a_output.json 读取 segment 时间映射。"""
    module_a_path = artifacts_dir / "module_a_output.json"
    if not module_a_path.exists():
        return {}
    try:
        module_a_output = read_json(module_a_path)
        seg_timing: dict[str, dict[str, float]] = {}
        for seg_item in module_a_output.get("segments", []) if isinstance(module_a_output, dict) else []:
            seg_id = str(seg_item.get("segment_id", "")).strip()
            if seg_id:
                seg_timing[seg_id] = {
                    "start_time": float(seg_item.get("start_time", 0) or 0),
                    "end_time": float(seg_item.get("end_time", 0) or 0),
                }
        return seg_timing
    except Exception:
        return {}


def _derive_shots_from_role3_streaming(artifacts_dir: Path) -> list[dict[str, object]]:
    """从 role3 streaming 文件衍生 shot 列表（module_b_output.json 的降级替代）。"""
    role3_streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role3")
    if not role3_streaming_dir.exists():
        raise RuntimeError(
            "模块C输入数据缺失：既没有 module_b_output.json，"
            "也没有 role3 streaming 产物。请先执行模块 B。"
        )
    stream_files = sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md"))
    if not stream_files:
        raise RuntimeError("模块C输入数据缺失：role3 streaming 目录为空。")

    seg_timing = _load_seg_timing(artifacts_dir)
    shots: list[dict[str, object]] = []

    for stream_path in stream_files:
        text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
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
            for line in lines[1:]:
                stripped = line.strip()
                if stripped.startswith("- scene_desc_zh:"):
                    scene_desc = stripped[len("- scene_desc_zh:"):].strip()
                elif stripped.startswith("- remotion_id:"):
                    remotion_id = stripped[len("- remotion_id:"):].strip()

            subjects = _parse_subject_descriptions(scene_desc, remotion_id)
            timing = seg_timing.get(seg_id, {})
            for subj_idx, _ in enumerate(subjects, start=1):
                shot_id = _build_shot_id(seg_id, subj_idx)
                shots.append({
                    "shot_id": shot_id,
                    "segment_id": seg_id,
                    "start_time": timing.get("start_time", 0.0),
                    "end_time": timing.get("end_time", 0.0),
                    "scene_desc": scene_desc,
                    "remotion_id": remotion_id,
                    "big_segment_id": current_big,
                })

    if not shots:
        raise RuntimeError("模块C输入数据缺失：role3 streaming 产物为空（无有效 shot）。")
    return shots


def _load_module_b_shots(artifacts_dir: Path, logger: Any) -> list[dict[str, object]]:
    """读取模块 B 输出，优先 module_b_output.json，降级到 role3 streaming 衍生。"""
    module_b_path = artifacts_dir / "module_b_output.json"
    if module_b_path.exists():
        try:
            module_b_output = read_json(module_b_path)
            validate_module_b_output(module_b_output)
            return module_b_output
        except Exception as exc:
            logger.warning("module_b_output.json 读取/校验失败（%s），降级到 role3 streaming。", exc)
    # Fallback: derive from role3 streaming files
    return _derive_shots_from_role3_streaming(artifacts_dir)


def run_module_c(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 C，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：输入产物缺失、单元重试耗尽或输出不完整时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块C开始执行，task_id=%s", context.task_id)

    module_b_shots = _load_module_b_shots(
        artifacts_dir=context.artifacts_dir,
        logger=context.logger,
    )
    units = build_module_c_units(shots=module_b_shots)
    # 用 role4 流式文件覆盖每个 unit 的 prompt 字段（支持 role4 重跑后最新内容）
    from dataclasses import replace
    units = [
        replace(unit, shot=_overlay_role4_streaming_prompt(unit.shot, context.artifacts_dir))
        for unit in units
    ]
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="C",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)

    pending_records = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["pending", "failed", "running"],
    )
    units_to_run = [
        units_by_id[str(record["unit_id"])]
        for record in pending_records
        if str(record["unit_id"]) in units_by_id
    ]
    context.logger.info(
        "模块C单元调度计划，task_id=%s，unit_total=%s，unit_to_run=%s",
        context.task_id,
        len(units),
        len(units_to_run),
    )

    frames_dir = context.artifacts_dir / "frames"
    task_seed = abs(hash(context.task_id)) % (2**32)
    generator = build_keyframe_generator(
        mode=context.config.module_c.render_backend,
        logger=context.logger,
        app_config=context.config,
        seed=task_seed,
    )
    context.logger.info("模块C开始执行 ComfyUI 探活/预热，task_id=%s", context.task_id)
    generator.prewarm()
    execute_units_with_retry(
        context=context,
        units_to_run=units_to_run,
        generator=generator,
        frames_dir=frames_dir,
    )

    frame_items = context.state_store.list_module_c_done_frame_items(task_id=context.task_id)
    if len(frame_items) != len(units):
        done_shot_ids = {str(item["shot_id"]) for item in frame_items}
        missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_shot_ids]
        raise RuntimeError(f"模块C执行失败：存在未完成单元，missing_unit_ids={missing_unit_ids}")
    frame_items = _merge_frame_items_with_shot_payloads(frame_items=frame_items, units_by_id=units_by_id)
    output_data = build_module_c_output(
        task_id=context.task_id,
        frames_dir=frames_dir,
        frame_items=frame_items,
    )
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块C执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


def run_module_c_shot(context: RuntimeContext, shot_id: str) -> Path:
    """
    功能说明：仅执行模块 C 的单个 shot 单元，并重建模块 C 输出清单。
    参数说明：
    - context: 运行上下文对象。
    - shot_id: 目标 shot 标识。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：输入脚本不存在、目标单元不存在或执行失败时抛出异常。
    边界条件：不要求模块 B 整体状态为 done，只要求目标 shot 已存在于 module_b_output.json。
    """
    normalized_shot_id = str(shot_id).strip()
    if not normalized_shot_id:
        raise ValueError("shot_id 不能为空。")

    context.logger.info("模块C shot 定向执行开始，task_id=%s，shot_id=%s", context.task_id, normalized_shot_id)

    module_b_shots = _load_module_b_shots(
        artifacts_dir=context.artifacts_dir,
        logger=context.logger,
    )
    units = build_module_c_units(shots=module_b_shots)
    # 用 role4 流式文件覆盖每个 unit 的 prompt 字段（支持 role4 重跑后最新内容）
    from dataclasses import replace
    units = [
        replace(unit, shot=_overlay_role4_streaming_prompt(unit.shot, context.artifacts_dir))
        for unit in units
    ]
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="C",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)
    target_unit = units_by_id.get(normalized_shot_id)
    if target_unit is None:
        raise RuntimeError(f"模块C shot 定向执行失败：找不到目标 shot，shot_id={normalized_shot_id}")

    frames_dir = context.artifacts_dir / "frames"
    task_seed = abs(hash(context.task_id)) % (2**32)
    generator = build_keyframe_generator(
        mode=context.config.module_c.render_backend,
        logger=context.logger,
        app_config=context.config,
        seed=task_seed,
    )
    context.logger.info("模块C shot 定向执行开始 ComfyUI 探活/预热，task_id=%s，shot_id=%s", context.task_id, normalized_shot_id)
    generator.prewarm()
    execute_one_unit_with_retry(
        context=context,
        unit=target_unit,
        generator=generator,
        frames_dir=frames_dir,
    )

    frame_items = context.state_store.list_module_c_done_frame_items(task_id=context.task_id)
    frame_items = _merge_frame_items_with_shot_payloads(frame_items=frame_items, units_by_id=units_by_id)
    output_data = build_module_c_output(
        task_id=context.task_id,
        frames_dir=frames_dir,
        frame_items=frame_items,
    )
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块C shot 定向执行完成，task_id=%s，shot_id=%s，输出=%s", context.task_id, normalized_shot_id, output_path)
    return output_path


def run_module_c_frame(context: RuntimeContext, shot_id: str, frame_type: str) -> Path:
    """
    功能说明：仅执行模块 C 的单个 shot 首帧或尾帧，并重建模块 C 输出清单。
    参数说明：
    - context: 运行上下文对象。
    - shot_id: 目标 shot 标识。
    - frame_type: "start" 或 "end"。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：目标单元不存在或单帧执行失败时抛出异常。
    边界条件：只更新目标帧路径，不要求另一帧同步重跑。
    """
    normalized_shot_id = str(shot_id).strip()
    normalized_frame_type = str(frame_type).strip().lower()
    if not normalized_shot_id:
        raise ValueError("shot_id 不能为空。")
    if normalized_frame_type not in {"start", "end"}:
        raise ValueError(f"frame_type 非法：{frame_type}")

    context.logger.info(
        "模块C 单帧定向执行开始，task_id=%s，shot_id=%s，frame_type=%s",
        context.task_id,
        normalized_shot_id,
        normalized_frame_type,
    )

    module_b_shots = _load_module_b_shots(
        artifacts_dir=context.artifacts_dir,
        logger=context.logger,
    )
    units = build_module_c_units(shots=module_b_shots)
    # 用 role4 流式文件覆盖每个 unit 的 prompt 字段（支持 role4 重跑后最新内容）
    from dataclasses import replace
    units = [
        replace(unit, shot=_overlay_role4_streaming_prompt(unit.shot, context.artifacts_dir))
        for unit in units
    ]
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="C",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)
    target_unit = units_by_id.get(normalized_shot_id)
    if target_unit is None:
        raise RuntimeError(
            f"模块C 单帧定向执行失败：找不到目标 shot，shot_id={normalized_shot_id}"
        )

    frames_dir = context.artifacts_dir / "frames"
    task_seed = abs(hash(context.task_id)) % (2**32)
    generator = build_keyframe_generator(
        mode=context.config.module_c.render_backend,
        logger=context.logger,
        app_config=context.config,
        seed=task_seed,
    )
    context.logger.info(
        "模块C 单帧定向执行开始 ComfyUI 探活/预热，task_id=%s，shot_id=%s，frame_type=%s",
        context.task_id,
        normalized_shot_id,
        normalized_frame_type,
    )
    generator.prewarm()
    from music_video_pipeline.modules.module_c.executor import _resolve_unit_dimensions
    width, height = _resolve_unit_dimensions(context=context, unit=target_unit)
    single_frame_item = generator.generate_one_frame(
        shot=target_unit.shot,
        output_dir=frames_dir,
        width=width,
        height=height,
        shot_index=target_unit.unit_index,
        frame_type=normalized_frame_type,
    )
    context.logger.info(
        "模块C 单帧生成结果已返回，task_id=%s，shot_id=%s，frame_type=%s，keys=%s",
        context.task_id,
        normalized_shot_id,
        normalized_frame_type,
        sorted(single_frame_item.keys()),
    )

    artifact_path = str(
        single_frame_item.get("frame_path_start" if normalized_frame_type == "start" else "frame_path_end", "")
    ).strip()
    context.state_store.set_module_unit_frame_status(
        task_id=context.task_id, module_name="C", unit_id=target_unit.unit_id,
        frame_type=normalized_frame_type, status="done",
    )
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="C",
        unit_id=target_unit.unit_id,
        status="done",
        artifact_path=artifact_path,
        error_message="",
    )
    context.logger.info(
        "模块C 单帧结果状态已写库，task_id=%s，shot_id=%s，frame_type=%s，artifact_path=%s",
        context.task_id,
        normalized_shot_id,
        normalized_frame_type,
        artifact_path,
    )

    frame_items = context.state_store.list_module_c_done_frame_items(task_id=context.task_id, frames_dir=frames_dir)
    frame_items = _merge_frame_items_with_shot_payloads(frame_items=frame_items, units_by_id=units_by_id)
    output_data = build_module_c_output(
        task_id=context.task_id,
        frames_dir=frames_dir,
        frame_items=frame_items,
    )
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, output_data)
    context.logger.info(
        "模块C 单帧定向执行完成，task_id=%s，shot_id=%s，frame_type=%s，输出=%s",
        context.task_id,
        normalized_shot_id,
        normalized_frame_type,
        output_path,
    )
    return output_path
