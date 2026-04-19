"""
文件用途：封装跨模块调度中的任务执行与完成回收逻辑。
核心流程：执行 B/C/D 单元、回收 Future、写入失败阻断状态。
输入输出：输入运行上下文与单元对象，输出执行结果或失败记录。
依赖说明：依赖模块 B/C/D 执行器与状态库。
维护说明：本模块不负责调度策略与并发窗口调参。
"""

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.io_utils import read_json
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
from music_video_pipeline.modules.module_b.executor import execute_one_unit_with_retry as execute_one_b_unit
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
from music_video_pipeline.modules.module_c.executor import execute_one_unit_with_retry as execute_one_c_unit
from music_video_pipeline.modules.module_c.unit_models import ModuleCUnit
from music_video_pipeline.modules.module_d.executor import (
    execute_one_unit_with_retry as execute_one_d_unit,
    prewarm_animatediff_runtime,
)
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint, materialize_module_d_unit

# 常量：D 阶段 runtime 异步预热任务 stage 标识。
D_RUNTIME_PREWARM_STAGE = "D_PREWARM"


def _drain_finished_tasks(
    context: RuntimeContext,
    active_tasks: dict[Future, tuple[str, int, str | None]],
    failed_chain_indexes: set[int],
    failed_errors: dict[int, str],
    d_runtime_warmed_devices: set[str] | None = None,
) -> None:
    """
    功能说明：处理已完成 future 的结果，并写入链路失败隔离。
    参数说明：
    - context: 运行上下文对象。
    - active_tasks: 活跃任务映射。
    - failed_chain_indexes: 失败链路集合（会被原位更新）。
    - failed_errors: 失败错误映射（会被原位更新）。
    返回值：无。
    异常说明：无。
    边界条件：仅消费已完成任务，不阻塞等待。
    """
    finished_futures = [future for future in active_tasks if future.done()]
    for future in finished_futures:
        stage, unit_index, metadata = active_tasks.pop(future)
        if stage == D_RUNTIME_PREWARM_STAGE:
            prewarm_device = str(metadata or "").strip()
            try:
                result = future.result()
                if isinstance(result, str) and result.strip():
                    prewarm_device = result.strip()
                if d_runtime_warmed_devices is not None and prewarm_device:
                    d_runtime_warmed_devices.add(prewarm_device)
                logging.getLogger("D").info(
                    "模块D runtime 异步预热完成，task_id=%s，device=%s",
                    context.task_id,
                    prewarm_device,
                )
            except Exception as error:  # noqa: BLE001
                logging.getLogger("D").warning(
                    "模块D runtime 异步预热失败，已忽略并继续调度，task_id=%s，device=%s，错误=%s",
                    context.task_id,
                    prewarm_device,
                    error,
                )
            continue
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            failed_chain_indexes.add(unit_index)
            failed_errors[unit_index] = f"{stage}:{error}"
            if stage == "B":
                context.state_store.mark_bcd_downstream_blocked(
                    task_id=context.task_id,
                    unit_index=unit_index,
                    from_module="B",
                    reason=f"upstream_blocked:B:{error}",
                )
            elif stage == "C":
                context.state_store.mark_bcd_downstream_blocked(
                    task_id=context.task_id,
                    unit_index=unit_index,
                    from_module="C",
                    reason=f"upstream_blocked:C:{error}",
                )
            stage_logger = logging.getLogger(stage)
            stage_logger.error(
                "跨模块链路单元失败，task_id=%s，stage=%s，unit_index=%s，错误=%s",
                context.task_id,
                stage,
                unit_index,
                error,
            )


def _run_b_chain_unit(
    context: RuntimeContext,
    unit: ModuleBUnit,
    generator: Any,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
) -> str:
    """
    功能说明：执行单条链路的模块 B 单元。
    """
    shot_path = execute_one_b_unit(
        context=context,
        unit=unit,
        generator=generator,
        module_a_output=module_a_output,
        unit_outputs_dir=unit_outputs_dir,
    )
    return str(shot_path)


def _run_c_chain_unit(
    context: RuntimeContext,
    chain_unit: CrossChainUnit,
    c_row: dict[str, Any],
    generator: Any,
    frames_dir: Any,
) -> dict[str, Any]:
    """
    功能说明：执行单条链路的模块 C 单元。
    """
    b_row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="B", unit_id=chain_unit.segment_id)
    if not b_row:
        raise RuntimeError(f"跨模块调度失败：模块B单元不存在，segment_id={chain_unit.segment_id}")
    shot_path = str(b_row.get("artifact_path", "")).strip()
    if not shot_path:
        raise RuntimeError(f"跨模块调度失败：模块B单元产物缺失，segment_id={chain_unit.segment_id}")

    shot = read_json(Path(shot_path))
    if not isinstance(shot, dict):
        raise RuntimeError(f"跨模块调度失败：模块B单元产物非法，segment_id={chain_unit.segment_id}")
    shot_obj = dict(shot)
    shot_obj["shot_id"] = chain_unit.shot_id
    if "start_time" not in shot_obj:
        shot_obj["start_time"] = float(c_row.get("start_time", chain_unit.start_time))
    if "end_time" not in shot_obj:
        shot_obj["end_time"] = float(c_row.get("end_time", chain_unit.end_time))

    unit = ModuleCUnit(
        unit_id=chain_unit.shot_id,
        unit_index=chain_unit.unit_index,
        shot=shot_obj,
        start_time=float(c_row.get("start_time", chain_unit.start_time)),
        end_time=float(c_row.get("end_time", chain_unit.end_time)),
        duration=float(c_row.get("duration", chain_unit.duration)),
    )
    return execute_one_c_unit(
        context=context,
        unit=unit,
        generator=generator,
        frames_dir=frames_dir,
    )


def _run_d_chain_unit(
    context: RuntimeContext,
    blueprint: ModuleDUnitBlueprint,
    c_row: dict[str, Any],
    profile: dict[str, Any],
    device_override: str | None = None,
) -> str:
    """
    功能说明：执行单条链路的模块 D 单元。
    """
    frame_path = str(c_row.get("artifact_path", "")).strip()
    if not frame_path:
        raise RuntimeError(f"跨模块调度失败：模块C单元产物缺失，unit_index={blueprint.unit_index}")
    frame_item = {
        "shot_id": blueprint.unit_id,
        "frame_path": frame_path,
        "start_time": float(c_row.get("start_time", blueprint.start_time)),
        "end_time": float(c_row.get("end_time", blueprint.end_time)),
        "duration": float(c_row.get("duration", blueprint.duration)),
    }
    unit = materialize_module_d_unit(blueprint=blueprint, frame_item=frame_item)
    segment_path = execute_one_d_unit(
        context=context,
        unit=unit,
        profile=profile,
        device_override=device_override,
    )
    return str(segment_path)


def _run_d_runtime_prewarm(context: RuntimeContext, device_override: str) -> str:
    """
    功能说明：异步预热指定设备的 AnimateDiff runtime。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 目标设备字符串（如 cuda:0/cuda:1）。
    返回值：
    - str: 预热设备标识。
    异常说明：预热失败时抛 RuntimeError（由调度层记录 warning 后忽略）。
    边界条件：仅执行模型加载缓存，不触发推理。
    """
    prewarm_animatediff_runtime(context=context, device_override=device_override)
    return str(device_override)


def _submit_d_runtime_prewarm_tasks(
    context: RuntimeContext,
    executor: ThreadPoolExecutor,
    active_tasks: dict[Future, tuple[str, int, str | None]],
    d_context: RuntimeContext,
    d_device_pool: list[str],
    warmed_devices: set[str],
    prewarm_requested_devices: set[str],
) -> None:
    """
    功能说明：向线程池提交 D 阶段 runtime 预热任务（设备去重、异步不阻塞）。
    参数说明：
    - context: 主运行上下文（用于日志）。
    - executor: 调度线程池。
    - active_tasks: 活跃任务映射（会被原位追加）。
    - d_context: 模块 D 运行上下文（用于预热日志归属）。
    - d_device_pool: D 阶段目标设备池。
    - warmed_devices: 已确认热启动设备集合。
    - prewarm_requested_devices: 已提交过预热请求的设备集合。
    返回值：无。
    异常说明：无（失败由 _drain_finished_tasks 内统一记录 warning）。
    边界条件：同设备最多提交一次预热任务，BC 阶段已热设备直接跳过。
    """
    for device in d_device_pool:
        normalized_device = str(device).strip()
        if not normalized_device:
            continue
        if normalized_device in warmed_devices:
            continue
        if normalized_device in prewarm_requested_devices:
            continue
        future = executor.submit(_run_d_runtime_prewarm, d_context, normalized_device)
        active_tasks[future] = (D_RUNTIME_PREWARM_STAGE, -1, normalized_device)
        prewarm_requested_devices.add(normalized_device)
        context.logger.info(
            "模块D runtime 异步预热已提交，task_id=%s，device=%s",
            context.task_id,
            normalized_device,
        )


def _split_failed_stage_and_message(error_text: str) -> tuple[str, str]:
    """
    功能说明：解析失败文本中的阶段前缀与错误正文。
    """
    normalized = str(error_text)
    if ":" not in normalized:
        return "", normalized
    stage_name, message = normalized.split(":", 1)
    return str(stage_name).strip(), str(message).strip()


def _contains_cuda_oom(error_text: str) -> bool:
    """
    功能说明：判断错误文本是否包含 CUDA OOM 信号。
    """
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return ("out of memory" in normalized) or ("cuda out of memory" in normalized)
