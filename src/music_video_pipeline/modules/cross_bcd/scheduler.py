"""
文件用途：实现跨模块 B/C/D 的链路波前并行调度。
核心流程：按 unit_index 管理链路状态，执行 B->C->D 单元派发，失败仅阻断当前链路。
输入输出：输入运行上下文与链路单元，输出执行结果摘要。
依赖说明：依赖标准库并发工具与项目内模块 B/C/D 执行器。
维护说明：调度器只负责单元编排，不改写模块 A 时间轴。
"""

# 标准库：用于线程池并发
from concurrent.futures import Future, ThreadPoolExecutor
# 标准库：用于时间控制
import time
# 标准库：用于类型提示
from typing import Any
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：分镜生成器工厂
from music_video_pipeline.generators import build_script_generator, build_frame_generator
# 项目内模块：JSON读取工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：跨模块链路模型
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
# 项目内模块：模块 B 单元模型
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
# 项目内模块：模块 C 单元模型
from music_video_pipeline.modules.module_c.unit_models import ModuleCUnit
# 项目内模块：模块 D 单元蓝图与单元模型
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint, materialize_module_d_unit
# 项目内模块：模块 B 单元执行器
from music_video_pipeline.modules.module_b.executor import execute_one_unit_with_retry as execute_one_b_unit
# 项目内模块：模块 C 单元执行器
from music_video_pipeline.modules.module_c.executor import execute_one_unit_with_retry as execute_one_c_unit
# 项目内模块：模块 D 单元执行器
from music_video_pipeline.modules.module_d.executor import (
    execute_one_unit_with_retry as execute_one_d_unit,
    resolve_render_profile,
)


def execute_cross_bcd_wavefront(
    context: RuntimeContext,
    chain_units: list[CrossChainUnit],
    b_units_by_segment_id: dict[str, ModuleBUnit],
    d_blueprints_by_index: dict[int, ModuleDUnitBlueprint],
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
    frames_dir: Any,
    target_segment_id: str | None = None,
) -> dict[str, Any]:
    """
    功能说明：执行跨模块 B/C/D 波前并行调度。
    参数说明：
    - context: 运行上下文对象。
    - chain_units: 链路单元数组。
    - b_units_by_segment_id: segment_id 到模块 B 单元映射。
    - d_blueprints_by_index: unit_index 到模块 D 蓝图映射。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 模块 B 单元输出目录。
    - frames_dir: 模块 C 帧输出目录。
    - target_segment_id: 可选，指定仅执行目标链路。
    返回值：
    - dict[str, Any]: 调度摘要（失败链路/错误映射/是否执行D单元）。
    异常说明：无（失败在摘要中返回，由上层统一决定是否抛错）。
    边界条件：失败仅阻断当前链路，下游链路继续推进。
    """
    b_worker_limit = _normalize_b_worker_limit(context.config.module_b.script_workers)
    render_limit = _normalize_global_render_limit(context.config.cross_module.global_render_limit)
    tick_seconds = _normalize_scheduler_tick_seconds(context.config.cross_module.scheduler_tick_ms)

    if target_segment_id:
        selected_indexes = [item.unit_index for item in chain_units if item.segment_id == target_segment_id]
        if not selected_indexes:
            raise RuntimeError(f"跨模块调度失败：未找到目标链路，segment_id={target_segment_id}")
    else:
        selected_indexes = [item.unit_index for item in chain_units]

    selected_index_set = set(selected_indexes)
    chain_by_index = {item.unit_index: item for item in chain_units}
    script_generator = build_script_generator(
        mode=context.config.mode.script_generator,
        logger=context.logger,
        module_b_config=context.config.module_b,
    )
    frame_generator = build_frame_generator(mode=context.config.mode.frame_generator, logger=context.logger)
    d_profile = resolve_render_profile(context=context)

    active_tasks: dict[Future, tuple[str, int]] = {}
    failed_chain_indexes: set[int] = set()
    failed_errors: dict[int, str] = {}
    d_unit_executed = False

    max_workers = max(2, b_worker_limit + render_limit)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            _drain_finished_tasks(
                context=context,
                active_tasks=active_tasks,
                failed_chain_indexes=failed_chain_indexes,
                failed_errors=failed_errors,
            )

            b_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="B")
            c_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="C")
            d_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="D")
            b_by_index = {int(item["unit_index"]): item for item in b_rows}
            c_by_index = {int(item["unit_index"]): item for item in c_rows}
            d_by_index = {int(item["unit_index"]): item for item in d_rows}

            in_flight_b = {idx for stage, idx in active_tasks.values() if stage == "B"}
            in_flight_c = {idx for stage, idx in active_tasks.values() if stage == "C"}
            in_flight_d = {idx for stage, idx in active_tasks.values() if stage == "D"}
            active_b_count = len(in_flight_b)
            active_render_count = len(in_flight_c) + len(in_flight_d)

            dispatched_count = 0

            for unit_index in sorted(selected_index_set):
                if active_b_count >= b_worker_limit:
                    break
                if unit_index in failed_chain_indexes or unit_index in in_flight_b:
                    continue
                b_row = b_by_index.get(unit_index)
                if not b_row:
                    continue
                b_status = str(b_row.get("status", "pending"))
                if b_status not in {"pending", "running", "failed"}:
                    continue
                chain_unit = chain_by_index[unit_index]
                b_unit = b_units_by_segment_id.get(chain_unit.segment_id)
                if not b_unit:
                    continue
                future = executor.submit(
                    _run_b_chain_unit,
                    context,
                    b_unit,
                    script_generator,
                    module_a_output,
                    unit_outputs_dir,
                )
                active_tasks[future] = ("B", unit_index)
                active_b_count += 1
                dispatched_count += 1

            for unit_index in sorted(selected_index_set):
                if active_render_count >= render_limit:
                    break
                if unit_index in failed_chain_indexes or unit_index in in_flight_d:
                    continue
                b_row = b_by_index.get(unit_index)
                c_row = c_by_index.get(unit_index)
                d_row = d_by_index.get(unit_index)
                if not b_row or not c_row or not d_row:
                    continue
                if str(b_row.get("status", "pending")) != "done":
                    continue
                c_status = str(c_row.get("status", "pending"))
                d_status = str(d_row.get("status", "pending"))
                if c_status != "done" or d_status not in {"pending", "running", "failed"}:
                    continue
                blueprint = d_blueprints_by_index.get(unit_index)
                if not blueprint:
                    continue
                future = executor.submit(
                    _run_d_chain_unit,
                    context,
                    blueprint,
                    c_row,
                    d_profile,
                )
                active_tasks[future] = ("D", unit_index)
                active_render_count += 1
                dispatched_count += 1
                d_unit_executed = True

            for unit_index in sorted(selected_index_set):
                if active_render_count >= render_limit:
                    break
                if unit_index in failed_chain_indexes or unit_index in in_flight_c:
                    continue
                b_row = b_by_index.get(unit_index)
                c_row = c_by_index.get(unit_index)
                if not b_row or not c_row:
                    continue
                if str(b_row.get("status", "pending")) != "done":
                    continue
                c_status = str(c_row.get("status", "pending"))
                if c_status not in {"pending", "running", "failed"}:
                    continue
                chain_unit = chain_by_index[unit_index]
                future = executor.submit(
                    _run_c_chain_unit,
                    context,
                    chain_unit,
                    c_row,
                    frame_generator,
                    frames_dir,
                )
                active_tasks[future] = ("C", unit_index)
                active_render_count += 1
                dispatched_count += 1

            if not active_tasks and dispatched_count == 0:
                if not _has_runnable_or_unsettled_chain(
                    selected_indexes=selected_index_set,
                    failed_chain_indexes=failed_chain_indexes,
                    b_by_index=b_by_index,
                    c_by_index=c_by_index,
                    d_by_index=d_by_index,
                ):
                    break

            if active_tasks and dispatched_count == 0:
                time.sleep(tick_seconds)

    return {
        "failed_chain_indexes": sorted(failed_chain_indexes),
        "failed_errors": failed_errors,
        "d_unit_executed": d_unit_executed,
    }


def _drain_finished_tasks(
    context: RuntimeContext,
    active_tasks: dict[Future, tuple[str, int]],
    failed_chain_indexes: set[int],
    failed_errors: dict[int, str],
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
        stage, unit_index = active_tasks.pop(future)
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
            context.logger.error(
                "跨模块链路单元失败，task_id=%s，stage=%s，unit_index=%s，错误=%s",
                context.task_id,
                stage,
                unit_index,
                error,
            )


def _has_runnable_or_unsettled_chain(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    b_by_index: dict[int, dict[str, Any]],
    c_by_index: dict[int, dict[str, Any]],
    d_by_index: dict[int, dict[str, Any]],
) -> bool:
    """
    功能说明：判断是否仍有可调度或未收敛链路。
    参数说明：
    - selected_indexes: 本轮需处理的链路索引集合。
    - failed_chain_indexes: 已判定失败的链路索引集合。
    - b_by_index/c_by_index/d_by_index: 三模块单元状态映射。
    返回值：
    - bool: 若存在可继续推进链路则返回 True。
    异常说明：无。
    边界条件：失败链路会被视为已收敛，不再推进。
    """
    for unit_index in selected_indexes:
        if unit_index in failed_chain_indexes:
            continue
        b_status = str(b_by_index.get(unit_index, {}).get("status", "pending"))
        c_status = str(c_by_index.get(unit_index, {}).get("status", "pending"))
        d_status = str(d_by_index.get(unit_index, {}).get("status", "pending"))
        if d_status == "done":
            continue
        if b_status in {"pending", "running", "failed"}:
            return True
        if b_status == "done" and c_status in {"pending", "running", "failed"}:
            return True
        if c_status == "done" and d_status in {"pending", "running", "failed"}:
            return True
    return False


def _run_b_chain_unit(
    context: RuntimeContext,
    unit: ModuleBUnit,
    generator: Any,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
) -> str:
    """
    功能说明：执行单条链路的模块 B 单元。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 B 单元。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 模块 B 单元输出目录。
    返回值：
    - str: 单元分镜路径。
    异常说明：重试耗尽时抛 RuntimeError。
    边界条件：状态写入由模块 B 执行器内部处理。
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
    参数说明：
    - context: 运行上下文对象。
    - chain_unit: 跨模块链路单元。
    - c_row: 模块 C 单元状态行。
    - generator: 关键帧生成器实例。
    - frames_dir: 帧输出目录。
    返回值：
    - dict[str, Any]: 单元 frame_item。
    异常说明：上游分镜文件缺失或重试耗尽时抛 RuntimeError。
    边界条件：shot_id 固定按链路主键写入。
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
) -> str:
    """
    功能说明：执行单条链路的模块 D 单元。
    参数说明：
    - context: 运行上下文对象。
    - blueprint: 模块 D 单元蓝图。
    - c_row: 模块 C 单元状态行。
    - profile: 模块 D 编码 profile。
    返回值：
    - str: 单元片段路径。
    异常说明：帧路径缺失或重试耗尽时抛 RuntimeError。
    边界条件：exact_frames 固定来自蓝图预分配结果。
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
    )
    return str(segment_path)


def _normalize_b_worker_limit(script_workers: int) -> int:
    """
    功能说明：归一化模块 B 并发上限。
    参数说明：
    - script_workers: 原始配置值。
    返回值：
    - int: 合法 worker 数量。
    异常说明：无。
    边界条件：非法值回退为 3，范围限制为 1~8。
    """
    try:
        normalized = int(script_workers)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 8:
        return 8
    return normalized


def _normalize_global_render_limit(global_render_limit: int) -> int:
    """
    功能说明：归一化跨模块 C/D 共享并发上限。
    参数说明：
    - global_render_limit: 原始配置值。
    返回值：
    - int: 合法并发上限。
    异常说明：无。
    边界条件：非法值回退为 3，范围限制为 1~16。
    """
    try:
        normalized = int(global_render_limit)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 16:
        return 16
    return normalized


def _normalize_scheduler_tick_seconds(scheduler_tick_ms: int) -> float:
    """
    功能说明：归一化调度轮询间隔并转换为秒。
    参数说明：
    - scheduler_tick_ms: 原始毫秒值。
    返回值：
    - float: 秒级间隔。
    异常说明：无。
    边界条件：非法值回退为 0.05 秒，范围限制为 0.01~1.0 秒。
    """
    try:
        normalized = int(scheduler_tick_ms)
    except (TypeError, ValueError):
        return 0.05
    if normalized < 10:
        return 0.05
    if normalized > 1000:
        return 1.0
    return normalized / 1000.0
