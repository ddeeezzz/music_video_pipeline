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
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_b_output
# 项目内模块：模块 B 产物路径工具
from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_streaming_dir


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


def run_module_c(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 C，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：输入脚本不存在、单元重试耗尽或输出不完整时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块C开始执行，task_id=%s", context.task_id)

    module_b_path = context.artifacts_dir / "module_b_output.json"
    module_b_output = read_json(module_b_path)
    try:
        validate_module_b_output(module_b_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "模块C输入契约校验失败：检测到旧版或不兼容的 module_b_output。"
            "请从模块B重跑，确保产物包含双关键帧字段与单视频轨字段（video_prompt_zh/video_prompt_en）。"
            f"原始错误：{error}"
        ) from error

    units = build_module_c_units(shots=module_b_output)
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

    module_b_path = context.artifacts_dir / "module_b_output.json"
    module_b_output = read_json(module_b_path)
    try:
        validate_module_b_output(module_b_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "模块C输入契约校验失败：检测到旧版或不兼容的 module_b_output。"
            "请从模块B重跑，确保产物包含双关键帧字段与单视频轨字段（video_prompt_zh/video_prompt_en）。"
            f"原始错误：{error}"
        ) from error

    units = build_module_c_units(shots=module_b_output)
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

    module_b_path = context.artifacts_dir / "module_b_output.json"
    module_b_output = read_json(module_b_path)
    try:
        validate_module_b_output(module_b_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "模块C输入契约校验失败：检测到旧版或不兼容的 module_b_output。"
            "请从模块B重跑，确保产物包含双关键帧字段与单视频轨字段（video_prompt_zh/video_prompt_en）。"
            f"原始错误：{error}"
        ) from error

    units = build_module_c_units(shots=module_b_output)
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
