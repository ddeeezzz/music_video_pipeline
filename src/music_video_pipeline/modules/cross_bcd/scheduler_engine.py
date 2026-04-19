"""
文件用途：实现跨模块 B/C/D 调度主循环与阶段状态机。
核心流程：刷新状态、应用自适应策略、派发 BC/D 任务、判定收敛退出。
输入输出：输入运行上下文与链路单元，输出执行摘要。
依赖说明：依赖 tasks/adaptive/allocators 子模块与 B/C/D 生成器工厂。
维护说明：本模块只负责调度编排，不承载自适应策略细节。
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import logging
import time
from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.generators import build_frame_generator, build_script_generator
from music_video_pipeline.modules.cross_bcd import scheduler_adaptive, scheduler_allocators, scheduler_tasks
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
from music_video_pipeline.modules.module_d.executor import resolve_render_profile
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint

# 常量：仅当 C 未完成单元数严格大于该阈值时才触发 D runtime 异步预热。
_D_RUNTIME_PREWARM_MIN_C_NOT_DONE = 10


@dataclass
class _LoopState:
    current_phase: str
    d_dispatch_enabled: bool
    d_device_pool: list[str]
    d_device_cursor: int
    last_d_device_inflight: dict[str, int]
    c_dynamic_limit: int
    d_dynamic_limit: int
    probe_interval_seconds: float
    next_probe_at: float
    c_last_adjust_done_count: int = -1
    d_last_adjust_done_count: int = -1
    handled_oom_failures: set[tuple[str, int]] = field(default_factory=set)
    c_window_direction_history: list[int] = field(default_factory=list)
    d_window_direction_history: list[int] = field(default_factory=list)
    probe_failure_count: int = 0
    d_runtime_warmed_devices: set[str] = field(default_factory=set)
    d_runtime_prewarm_requested_devices: set[str] = field(default_factory=set)
    d_unit_executed: bool = False
    single_gpu_mode: bool = False
    oom_fallback_locked_c_then_d: bool = False


@dataclass
class _SnapshotState:
    b_by_index: dict[int, dict[str, Any]]
    c_by_index: dict[int, dict[str, Any]]
    d_by_index: dict[int, dict[str, Any]]
    c_done_count: int
    d_done_count: int


@dataclass
class _DispatchState:
    in_flight_b: set[int]
    in_flight_c: set[int]
    in_flight_d: set[int]
    active_b_count: int
    active_c_count: int
    active_d_count: int
    active_render_count: int
    d_device_inflight: dict[str, int]
    c_generator_inflight: dict[int, int]


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
    """
    b_worker_limit = scheduler_adaptive._normalize_b_worker_limit(context.config.module_b.script_workers)
    render_limit = scheduler_adaptive._normalize_global_render_limit(context.config.cross_module.global_render_limit)
    tick_seconds = scheduler_adaptive._normalize_scheduler_tick_seconds(context.config.cross_module.scheduler_tick_ms)
    render_backend = scheduler_adaptive._normalize_module_d_render_backend(context.config.module_d.render_backend)
    adaptive_window_runtime = scheduler_adaptive._build_adaptive_window_runtime(
        context=context,
        global_render_limit=render_limit,
        render_backend=render_backend,
    )
    adaptive_enabled = bool(adaptive_window_runtime["enabled"])

    if target_segment_id:
        selected_indexes = [item.unit_index for item in chain_units if item.segment_id == target_segment_id]
        if not selected_indexes:
            raise RuntimeError(f"跨模块调度失败：未找到目标链路，segment_id={target_segment_id}")
    else:
        selected_indexes = [item.unit_index for item in chain_units]

    selected_index_set = set(selected_indexes)
    chain_by_index = {item.unit_index: item for item in chain_units}
    b_context = replace(context, logger=logging.getLogger("B"))
    c_context = replace(context, logger=logging.getLogger("C"))
    d_context = replace(context, logger=logging.getLogger("D"))

    script_generator = build_script_generator(
        mode=context.config.mode.script_generator,
        logger=b_context.logger,
        module_b_config=context.config.module_b,
    )
    c_generator_pool_size = scheduler_allocators._resolve_c_generator_pool_size(
        context=context,
        adaptive_window_runtime=adaptive_window_runtime,
    )
    c_generator_pool = [
        build_frame_generator(mode=context.config.mode.frame_generator, logger=c_context.logger)
        for _ in range(c_generator_pool_size)
    ]
    c_generator_cursor = 0
    d_profile = resolve_render_profile(context=d_context)

    active_tasks: dict[Future, tuple[str, int, str | None]] = {}
    failed_chain_indexes: set[int] = set()
    failed_errors: dict[int, str] = {}
    initial_snapshot_state = _refresh_state_snapshot(
        context=context,
        selected_indexes=selected_index_set,
        failed_chain_indexes=failed_chain_indexes,
    )
    c_not_done_before_d_phase = _count_selected_not_done_units(
        selected_indexes=selected_index_set,
        failed_chain_indexes=failed_chain_indexes,
        unit_by_index=initial_snapshot_state.c_by_index,
    )

    d_device_pool_all = scheduler_allocators._build_d_device_pool(
        c_gpu_index=int(adaptive_window_runtime["c_gpu_index"]),
        d_gpu_index=int(adaptive_window_runtime["d_gpu_index"]),
    )
    c_gpu_device = f"cuda:{int(adaptive_window_runtime['c_gpu_index'])}"
    d_gpu_device = f"cuda:{int(adaptive_window_runtime['d_gpu_index'])}"
    single_gpu_mode = bool(adaptive_window_runtime.get("single_gpu_mode", False))
    bc_allow_d_dispatch = not (render_backend == "animatediff" and c_gpu_device == d_gpu_device)
    if single_gpu_mode:
        bc_allow_d_dispatch = True
    loop_state = _LoopState(
        current_phase="bc",
        d_dispatch_enabled=bc_allow_d_dispatch,
        d_device_pool=[d_gpu_device],
        d_device_cursor=0,
        last_d_device_inflight={d_gpu_device: 0},
        c_dynamic_limit=int(adaptive_window_runtime["c_dynamic_limit"]),
        d_dynamic_limit=int(adaptive_window_runtime["d_dynamic_limit"]),
        probe_interval_seconds=float(adaptive_window_runtime["probe_interval_seconds"]),
        next_probe_at=0.0,
        single_gpu_mode=single_gpu_mode,
        oom_fallback_locked_c_then_d=bool(adaptive_window_runtime.get("oom_fallback_locked_c_then_d", False)),
    )
    if loop_state.single_gpu_mode:
        loop_state.c_dynamic_limit = 1
        loop_state.d_dynamic_limit = 1
        adaptive_window_runtime["c_dynamic_limit"] = 1
        adaptive_window_runtime["d_dynamic_limit"] = 1
        adaptive_window_runtime["oom_fallback_locked_c_then_d"] = bool(loop_state.oom_fallback_locked_c_then_d)
        context.logger.info(
            "跨模块进入单卡模式，task_id=%s，detected_gpu_count=%s，source=%s，C=%s，D=%s，BC阶段D并发=%s",
            context.task_id,
            adaptive_window_runtime.get("detected_gpu_count"),
            adaptive_window_runtime.get("gpu_count_source"),
            loop_state.c_dynamic_limit,
            loop_state.d_dynamic_limit,
            "on" if loop_state.d_dispatch_enabled else "off",
        )

    prewarm_worker_budget = 2 if render_backend == "animatediff" else 0
    max_workers = max(2, b_worker_limit + int(adaptive_window_runtime["max_render_workers"]) + prewarm_worker_budget)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            scheduler_tasks._drain_finished_tasks(
                context=context,
                active_tasks=active_tasks,
                failed_chain_indexes=failed_chain_indexes,
                failed_errors=failed_errors,
                d_runtime_warmed_devices=loop_state.d_runtime_warmed_devices,
            )

            snapshot_state = _refresh_state_snapshot(
                context=context,
                selected_indexes=selected_index_set,
                failed_chain_indexes=failed_chain_indexes,
            )

            _apply_oom_downscale(
                context=context,
                failed_errors=failed_errors,
                loop_state=loop_state,
                adaptive_window_runtime=adaptive_window_runtime,
                c_done_count=snapshot_state.c_done_count,
                d_done_count=snapshot_state.d_done_count,
            )

            if loop_state.current_phase == "bc" and _all_selected_c_done(
                selected_indexes=selected_index_set,
                failed_chain_indexes=failed_chain_indexes,
                c_by_index=snapshot_state.c_by_index,
            ):
                loop_state.current_phase = "d"
                loop_state.d_dispatch_enabled = True
                loop_state.d_device_pool = scheduler_allocators._resolve_d_device_pool_for_phase_d(
                    context=context,
                    adaptive_window_runtime=adaptive_window_runtime,
                    fallback_device_pool=d_device_pool_all,
                )
                loop_state.last_d_device_inflight = {device: 0 for device in loop_state.d_device_pool}
                c_done_count = _count_selected_c_done(
                    selected_indexes=selected_index_set,
                    failed_chain_indexes=failed_chain_indexes,
                    c_by_index=snapshot_state.c_by_index,
                )
                context.logger.info(
                    "跨模块阶段切换 BC->D，task_id=%s，c_done=%s/%s，d_device_pool=%s",
                    context.task_id,
                    c_done_count,
                    len(selected_index_set),
                    loop_state.d_device_pool,
                )
                if render_backend == "animatediff" and c_not_done_before_d_phase > _D_RUNTIME_PREWARM_MIN_C_NOT_DONE:
                    scheduler_tasks._submit_d_runtime_prewarm_tasks(
                        context=context,
                        executor=executor,
                        active_tasks=active_tasks,
                        d_context=d_context,
                        d_device_pool=loop_state.d_device_pool,
                        warmed_devices=loop_state.d_runtime_warmed_devices,
                        prewarm_requested_devices=loop_state.d_runtime_prewarm_requested_devices,
                    )
                elif render_backend == "animatediff":
                    context.logger.info(
                        "模块D runtime 异步预热已跳过，task_id=%s，c_not_done_before_d_phase=%s，阈值条件=>%s",
                        context.task_id,
                        c_not_done_before_d_phase,
                        _D_RUNTIME_PREWARM_MIN_C_NOT_DONE,
                    )

            _apply_adaptive_tick(
                context=context,
                active_tasks=active_tasks,
                adaptive_enabled=adaptive_enabled,
                adaptive_window_runtime=adaptive_window_runtime,
                loop_state=loop_state,
                render_backend=render_backend,
                c_done_count=snapshot_state.c_done_count,
                d_done_count=snapshot_state.d_done_count,
            )

            dispatch_state = _build_dispatch_state(
                active_tasks=active_tasks,
                d_device_pool=loop_state.d_device_pool,
                c_generator_pool_size=c_generator_pool_size,
            )
            loop_state.last_d_device_inflight = dict(dispatch_state.d_device_inflight)

            bc_dispatched_count, c_generator_cursor = _dispatch_bc_units(
                executor=executor,
                loop_state=loop_state,
                adaptive_enabled=adaptive_enabled,
                b_worker_limit=b_worker_limit,
                render_limit=render_limit,
                selected_indexes=selected_index_set,
                failed_chain_indexes=failed_chain_indexes,
                b_by_index=snapshot_state.b_by_index,
                c_by_index=snapshot_state.c_by_index,
                chain_by_index=chain_by_index,
                b_units_by_segment_id=b_units_by_segment_id,
                b_context=b_context,
                c_context=c_context,
                script_generator=script_generator,
                module_a_output=module_a_output,
                unit_outputs_dir=unit_outputs_dir,
                c_generator_pool=c_generator_pool,
                c_generator_cursor=c_generator_cursor,
                c_generator_pool_size=c_generator_pool_size,
                frames_dir=frames_dir,
                active_tasks=active_tasks,
                dispatch_state=dispatch_state,
            )

            d_dispatched_count = _dispatch_d_units(
                executor=executor,
                loop_state=loop_state,
                adaptive_enabled=adaptive_enabled,
                render_limit=render_limit,
                render_backend=render_backend,
                selected_indexes=selected_index_set,
                failed_chain_indexes=failed_chain_indexes,
                b_by_index=snapshot_state.b_by_index,
                c_by_index=snapshot_state.c_by_index,
                d_by_index=snapshot_state.d_by_index,
                d_blueprints_by_index=d_blueprints_by_index,
                d_context=d_context,
                d_profile=d_profile,
                active_tasks=active_tasks,
                dispatch_state=dispatch_state,
                d_done_count=snapshot_state.d_done_count,
            )
            dispatched_count = bc_dispatched_count + d_dispatched_count

            if _should_exit_loop(
                active_tasks=active_tasks,
                dispatched_count=dispatched_count,
                selected_indexes=selected_index_set,
                failed_chain_indexes=failed_chain_indexes,
                b_by_index=snapshot_state.b_by_index,
                c_by_index=snapshot_state.c_by_index,
                d_by_index=snapshot_state.d_by_index,
            ):
                break

            if active_tasks and dispatched_count == 0:
                time.sleep(tick_seconds)

    return {
        "failed_chain_indexes": sorted(failed_chain_indexes),
        "failed_errors": failed_errors,
        "d_unit_executed": loop_state.d_unit_executed,
        "adaptive_window_snapshot": scheduler_adaptive._build_adaptive_window_snapshot(
            adaptive_window_runtime=adaptive_window_runtime,
            render_limit=render_limit,
            render_backend=render_backend,
            current_phase=loop_state.current_phase,
            d_dispatch_enabled=loop_state.d_dispatch_enabled,
            d_device_pool=loop_state.d_device_pool,
            d_device_inflight=loop_state.last_d_device_inflight,
        ),
    }


def _refresh_state_snapshot(
    context: RuntimeContext,
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
) -> _SnapshotState:
    b_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="B")
    c_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="C")
    d_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="D")
    b_by_index = {int(item["unit_index"]): item for item in b_rows}
    c_by_index = {int(item["unit_index"]): item for item in c_rows}
    d_by_index = {int(item["unit_index"]): item for item in d_rows}
    c_done_count = _count_selected_done_units(
        selected_indexes=selected_indexes,
        failed_chain_indexes=failed_chain_indexes,
        unit_by_index=c_by_index,
    )
    d_done_count = _count_selected_done_units(
        selected_indexes=selected_indexes,
        failed_chain_indexes=failed_chain_indexes,
        unit_by_index=d_by_index,
    )
    return _SnapshotState(
        b_by_index=b_by_index,
        c_by_index=c_by_index,
        d_by_index=d_by_index,
        c_done_count=c_done_count,
        d_done_count=d_done_count,
    )


def _apply_adaptive_tick(
    context: RuntimeContext,
    active_tasks: dict[Future, tuple[str, int, str | None]],
    adaptive_enabled: bool,
    adaptive_window_runtime: dict[str, Any],
    loop_state: _LoopState,
    render_backend: str,
    c_done_count: int,
    d_done_count: int,
) -> None:
    if not adaptive_enabled and not loop_state.single_gpu_mode:
        return
    now_mono = time.monotonic()
    if now_mono < loop_state.next_probe_at:
        return
    if loop_state.single_gpu_mode:
        loop_state.c_dynamic_limit = 1
        loop_state.d_dynamic_limit = 1
        adaptive_window_runtime["c_dynamic_limit"] = 1
        adaptive_window_runtime["d_dynamic_limit"] = 1
        loop_state.next_probe_at = now_mono + max(loop_state.probe_interval_seconds, 0.5)
        return

    active_d_probe_count = sum(1 for stage, _, _ in active_tasks.values() if stage == "D")
    if loop_state.current_phase == "d" and render_backend == "animatediff" and active_d_probe_count > 0:
        loop_state.next_probe_at = now_mono + max(loop_state.probe_interval_seconds, 2.0)
        return

    probe_rows, probe_error = scheduler_adaptive._run_gpu_probe_script(
        context=context,
        timeout_seconds=4.0,
    )
    adaptive_window_runtime["last_probe_error"] = str(probe_error)
    adaptive_window_runtime["last_probe_rows"] = probe_rows
    if probe_error:
        loop_state.probe_failure_count += 1
        backoff_seconds = min(
            max(loop_state.probe_interval_seconds, 1.0) * (2 ** min(loop_state.probe_failure_count, 5)),
            30.0,
        )
        if loop_state.probe_failure_count == 1 or loop_state.probe_failure_count % 5 == 0:
            context.logger.warning(
                "跨模块自适应窗口采样失败，task_id=%s，错误=%s，failure_count=%s，backoff=%.1fs",
                context.task_id,
                probe_error,
                loop_state.probe_failure_count,
                backoff_seconds,
            )
        if loop_state.current_phase == "bc":
            old_c_limit = loop_state.c_dynamic_limit
            loop_state.c_dynamic_limit = int(adaptive_window_runtime["fallback_c_limit"])
            if old_c_limit != loop_state.c_dynamic_limit:
                context.logger.info(
                    "跨模块自适应窗口降级静态值（BC阶段），task_id=%s，C:%s->%s",
                    context.task_id,
                    old_c_limit,
                    loop_state.c_dynamic_limit,
                )
        else:
            old_d_limit = loop_state.d_dynamic_limit
            loop_state.d_dynamic_limit = int(adaptive_window_runtime["fallback_d_limit"])
            if old_d_limit != loop_state.d_dynamic_limit:
                context.logger.info(
                    "跨模块自适应窗口降级静态值（D阶段），task_id=%s，D:%s->%s",
                    context.task_id,
                    old_d_limit,
                    loop_state.d_dynamic_limit,
                )
        adaptive_window_runtime["c_dynamic_limit"] = loop_state.c_dynamic_limit
        adaptive_window_runtime["d_dynamic_limit"] = loop_state.d_dynamic_limit
        loop_state.next_probe_at = now_mono + backoff_seconds
        return

    loop_state.probe_failure_count = 0
    if loop_state.current_phase == "bc":
        old_c_limit = loop_state.c_dynamic_limit
        c_ratio = scheduler_adaptive._extract_gpu_used_ratio(
            probe_rows=probe_rows,
            gpu_index=int(adaptive_window_runtime["c_gpu_index"]),
        )
        can_adjust_c = c_done_count > loop_state.c_last_adjust_done_count
        if can_adjust_c:
            loop_state.c_dynamic_limit = scheduler_adaptive._adjust_dynamic_limit(
                current_limit=loop_state.c_dynamic_limit,
                used_ratio=c_ratio,
                low_watermark=float(adaptive_window_runtime["low_watermark"]),
                high_watermark=float(adaptive_window_runtime["high_watermark"]),
                limit_min=int(adaptive_window_runtime["c_limit_min"]),
                limit_max=int(adaptive_window_runtime["c_limit_max"]),
            )
        if old_c_limit != loop_state.c_dynamic_limit:
            loop_state.c_last_adjust_done_count = c_done_count
            loop_state.c_window_direction_history = scheduler_adaptive._append_window_direction(
                history=loop_state.c_window_direction_history,
                old_limit=old_c_limit,
                new_limit=loop_state.c_dynamic_limit,
            )
            if scheduler_adaptive._is_two_round_trip_flap(loop_state.c_window_direction_history):
                lowered_limit = max(int(adaptive_window_runtime["c_limit_min"]), loop_state.c_dynamic_limit - 1)
                if lowered_limit != loop_state.c_dynamic_limit:
                    context.logger.warning(
                        "跨模块自适应窗口反抖降一级（BC阶段），task_id=%s，C:%s->%s，history=%s",
                        context.task_id,
                        loop_state.c_dynamic_limit,
                        lowered_limit,
                        loop_state.c_window_direction_history,
                    )
                    loop_state.c_dynamic_limit = lowered_limit
                loop_state.c_window_direction_history = []
            context.logger.info(
                "跨模块自适应窗口调整（BC阶段），task_id=%s，C:%s->%s(gpu%s=%s)，next_adjust_after_done>%s",
                context.task_id,
                old_c_limit,
                loop_state.c_dynamic_limit,
                adaptive_window_runtime["c_gpu_index"],
                c_ratio,
                loop_state.c_last_adjust_done_count,
            )
    else:
        old_d_limit = loop_state.d_dynamic_limit
        d_ratio = scheduler_adaptive._extract_gpu_used_ratio(
            probe_rows=probe_rows,
            gpu_index=int(adaptive_window_runtime["d_gpu_index"]),
        )
        can_adjust_d = d_done_count > loop_state.d_last_adjust_done_count
        if can_adjust_d:
            loop_state.d_dynamic_limit = scheduler_adaptive._adjust_dynamic_limit(
                current_limit=loop_state.d_dynamic_limit,
                used_ratio=d_ratio,
                low_watermark=float(adaptive_window_runtime["low_watermark"]),
                high_watermark=float(adaptive_window_runtime["high_watermark"]),
                limit_min=int(adaptive_window_runtime["d_limit_min"]),
                limit_max=int(adaptive_window_runtime["d_limit_max"]),
            )
        if old_d_limit != loop_state.d_dynamic_limit:
            loop_state.d_last_adjust_done_count = d_done_count
            loop_state.d_window_direction_history = scheduler_adaptive._append_window_direction(
                history=loop_state.d_window_direction_history,
                old_limit=old_d_limit,
                new_limit=loop_state.d_dynamic_limit,
            )
            if scheduler_adaptive._is_two_round_trip_flap(loop_state.d_window_direction_history):
                lowered_limit = max(int(adaptive_window_runtime["d_limit_min"]), loop_state.d_dynamic_limit - 1)
                if lowered_limit != loop_state.d_dynamic_limit:
                    context.logger.warning(
                        "跨模块自适应窗口反抖降一级（D阶段），task_id=%s，D:%s->%s，history=%s",
                        context.task_id,
                        loop_state.d_dynamic_limit,
                        lowered_limit,
                        loop_state.d_window_direction_history,
                    )
                    loop_state.d_dynamic_limit = lowered_limit
                loop_state.d_window_direction_history = []
            context.logger.info(
                "跨模块自适应窗口调整（D阶段），task_id=%s，D:%s->%s(gpu%s=%s)，backend=%s，next_adjust_after_done>%s",
                context.task_id,
                old_d_limit,
                loop_state.d_dynamic_limit,
                adaptive_window_runtime["d_gpu_index"],
                d_ratio,
                render_backend,
                loop_state.d_last_adjust_done_count,
            )
    adaptive_window_runtime["c_dynamic_limit"] = loop_state.c_dynamic_limit
    adaptive_window_runtime["d_dynamic_limit"] = loop_state.d_dynamic_limit
    loop_state.next_probe_at = now_mono + loop_state.probe_interval_seconds


def _build_dispatch_state(
    active_tasks: dict[Future, tuple[str, int, str | None]],
    d_device_pool: list[str],
    c_generator_pool_size: int,
) -> _DispatchState:
    in_flight_b = {idx for stage, idx, _ in active_tasks.values() if stage == "B"}
    in_flight_c = {idx for stage, idx, _ in active_tasks.values() if stage == "C"}
    in_flight_d = {idx for stage, idx, _ in active_tasks.values() if stage == "D"}
    active_b_count = len(in_flight_b)
    active_c_count = len(in_flight_c)
    active_d_count = len(in_flight_d)
    active_render_count = active_c_count + active_d_count
    d_device_inflight = scheduler_allocators._build_d_device_inflight(active_tasks=active_tasks, d_device_pool=d_device_pool)
    c_generator_inflight = scheduler_allocators._build_c_generator_inflight(
        active_tasks=active_tasks,
        generator_pool_size=c_generator_pool_size,
    )
    return _DispatchState(
        in_flight_b=in_flight_b,
        in_flight_c=in_flight_c,
        in_flight_d=in_flight_d,
        active_b_count=active_b_count,
        active_c_count=active_c_count,
        active_d_count=active_d_count,
        active_render_count=active_render_count,
        d_device_inflight=d_device_inflight,
        c_generator_inflight=c_generator_inflight,
    )


def _dispatch_bc_units(
    executor: ThreadPoolExecutor,
    loop_state: _LoopState,
    adaptive_enabled: bool,
    b_worker_limit: int,
    render_limit: int,
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    b_by_index: dict[int, dict[str, Any]],
    c_by_index: dict[int, dict[str, Any]],
    chain_by_index: dict[int, CrossChainUnit],
    b_units_by_segment_id: dict[str, ModuleBUnit],
    b_context: RuntimeContext,
    c_context: RuntimeContext,
    script_generator: Any,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
    c_generator_pool: list[Any],
    c_generator_cursor: int,
    c_generator_pool_size: int,
    frames_dir: Any,
    active_tasks: dict[Future, tuple[str, int, str | None]],
    dispatch_state: _DispatchState,
) -> tuple[int, int]:
    if loop_state.current_phase != "bc":
        return 0, c_generator_cursor
    dispatched_count = 0

    for unit_index in sorted(selected_indexes):
        if dispatch_state.active_b_count >= b_worker_limit:
            break
        if unit_index in failed_chain_indexes or unit_index in dispatch_state.in_flight_b:
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
            scheduler_tasks._run_b_chain_unit,
            b_context,
            b_unit,
            script_generator,
            module_a_output,
            unit_outputs_dir,
        )
        active_tasks[future] = ("B", unit_index, None)
        dispatch_state.active_b_count += 1
        dispatch_state.in_flight_b.add(unit_index)
        dispatched_count += 1

    for unit_index in sorted(selected_indexes):
        if adaptive_enabled or loop_state.single_gpu_mode:
            if dispatch_state.active_c_count >= loop_state.c_dynamic_limit:
                break
        elif dispatch_state.active_render_count >= render_limit:
            break
        if unit_index in failed_chain_indexes or unit_index in dispatch_state.in_flight_c:
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
        generator_index, c_generator_cursor = scheduler_allocators._pick_next_available_c_generator(
            c_generator_inflight=dispatch_state.c_generator_inflight,
            generator_pool_size=c_generator_pool_size,
            start_cursor=c_generator_cursor,
        )
        if generator_index is None:
            break
        chain_unit = chain_by_index[unit_index]
        future = executor.submit(
            scheduler_tasks._run_c_chain_unit,
            c_context,
            chain_unit,
            c_row,
            c_generator_pool[generator_index],
            frames_dir,
        )
        active_tasks[future] = ("C", unit_index, f"cgen:{generator_index}")
        dispatch_state.active_c_count += 1
        dispatch_state.active_render_count += 1
        dispatch_state.in_flight_c.add(unit_index)
        dispatch_state.c_generator_inflight[generator_index] = int(dispatch_state.c_generator_inflight.get(generator_index, 0)) + 1
        dispatched_count += 1

    return dispatched_count, c_generator_cursor


def _dispatch_d_units(
    executor: ThreadPoolExecutor,
    loop_state: _LoopState,
    adaptive_enabled: bool,
    render_limit: int,
    render_backend: str,
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    b_by_index: dict[int, dict[str, Any]],
    c_by_index: dict[int, dict[str, Any]],
    d_by_index: dict[int, dict[str, Any]],
    d_blueprints_by_index: dict[int, ModuleDUnitBlueprint],
    d_context: RuntimeContext,
    d_profile: dict[str, Any],
    active_tasks: dict[Future, tuple[str, int, str | None]],
    dispatch_state: _DispatchState,
    d_done_count: int,
) -> int:
    if not loop_state.d_dispatch_enabled:
        return 0
    if loop_state.single_gpu_mode and loop_state.oom_fallback_locked_c_then_d and loop_state.current_phase == "bc":
        return 0
    dispatched_count = 0
    d_stage_limit = loop_state.d_dynamic_limit if (adaptive_enabled or loop_state.single_gpu_mode) else render_limit
    animatediff_per_device_limit = (
        max(1, int(loop_state.d_dynamic_limit))
        if (adaptive_enabled or loop_state.single_gpu_mode)
        else max(1, int(render_limit))
    )

    for unit_index in sorted(selected_indexes):
        if dispatch_state.active_d_count >= d_stage_limit:
            break
        if unit_index in failed_chain_indexes or unit_index in dispatch_state.in_flight_d:
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
        device_override: str | None = None
        if render_backend == "animatediff":
            device_override, loop_state.d_device_cursor = scheduler_allocators._pick_next_available_d_device(
                d_device_pool=loop_state.d_device_pool,
                d_device_inflight=dispatch_state.d_device_inflight,
                start_cursor=loop_state.d_device_cursor,
                per_device_limit=animatediff_per_device_limit,
            )
            if device_override is None:
                break
        future = executor.submit(
            scheduler_tasks._run_d_chain_unit,
            d_context,
            blueprint,
            c_row,
            d_profile,
            device_override,
        )
        active_tasks[future] = ("D", unit_index, device_override)
        dispatch_state.active_d_count += 1
        dispatch_state.active_render_count += 1
        dispatch_state.in_flight_d.add(unit_index)
        dispatched_count += 1
        loop_state.d_unit_executed = True
        if device_override:
            dispatch_state.d_device_inflight[device_override] = int(dispatch_state.d_device_inflight.get(device_override, 0)) + 1
            if loop_state.current_phase == "bc" and render_backend == "animatediff":
                loop_state.d_runtime_warmed_devices.add(device_override)
    loop_state.last_d_device_inflight = dict(dispatch_state.d_device_inflight)
    return dispatched_count


def _apply_oom_downscale(
    context: RuntimeContext,
    failed_errors: dict[int, str],
    loop_state: _LoopState,
    adaptive_window_runtime: dict[str, Any],
    c_done_count: int,
    d_done_count: int,
) -> None:
    for unit_index, error_text in list(failed_errors.items()):
        stage_name, stage_error = scheduler_tasks._split_failed_stage_and_message(error_text)
        failure_key = (stage_name, int(unit_index))
        if failure_key in loop_state.handled_oom_failures:
            continue
        if not scheduler_tasks._contains_cuda_oom(stage_error):
            continue
        loop_state.handled_oom_failures.add(failure_key)
        if loop_state.single_gpu_mode and not loop_state.oom_fallback_locked_c_then_d:
            loop_state.oom_fallback_locked_c_then_d = True
            loop_state.current_phase = "bc"
            loop_state.d_dispatch_enabled = False
            adaptive_window_runtime["oom_fallback_locked_c_then_d"] = True
            context.logger.warning(
                "单卡模式触发OOM降级锁定，task_id=%s，unit_index=%s，stage=%s，后续切换为先C后D",
                context.task_id,
                unit_index,
                stage_name,
            )
        if stage_name == "C":
            old_c_limit = loop_state.c_dynamic_limit
            loop_state.c_dynamic_limit = max(int(adaptive_window_runtime["c_limit_min"]), loop_state.c_dynamic_limit - 1)
            adaptive_window_runtime["c_dynamic_limit"] = loop_state.c_dynamic_limit
            loop_state.c_last_adjust_done_count = c_done_count
            if loop_state.c_dynamic_limit != old_c_limit:
                context.logger.warning(
                    "跨模块自适应窗口OOM降档（BC阶段），task_id=%s，C:%s->%s，unit_index=%s，错误=%s",
                    context.task_id,
                    old_c_limit,
                    loop_state.c_dynamic_limit,
                    unit_index,
                    stage_error,
                )
        if stage_name == "D":
            old_d_limit = loop_state.d_dynamic_limit
            loop_state.d_dynamic_limit = max(int(adaptive_window_runtime["d_limit_min"]), loop_state.d_dynamic_limit - 1)
            adaptive_window_runtime["d_dynamic_limit"] = loop_state.d_dynamic_limit
            loop_state.d_last_adjust_done_count = d_done_count
            if loop_state.d_dynamic_limit != old_d_limit:
                context.logger.warning(
                    "跨模块自适应窗口OOM降档（D阶段），task_id=%s，D:%s->%s，unit_index=%s，错误=%s",
                    context.task_id,
                    old_d_limit,
                    loop_state.d_dynamic_limit,
                    unit_index,
                    stage_error,
                )


def _should_exit_loop(
    active_tasks: dict[Future, tuple[str, int, str | None]],
    dispatched_count: int,
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    b_by_index: dict[int, dict[str, Any]],
    c_by_index: dict[int, dict[str, Any]],
    d_by_index: dict[int, dict[str, Any]],
) -> bool:
    if active_tasks or dispatched_count != 0:
        return False
    return not _has_runnable_or_unsettled_chain(
        selected_indexes=selected_indexes,
        failed_chain_indexes=failed_chain_indexes,
        b_by_index=b_by_index,
        c_by_index=c_by_index,
        d_by_index=d_by_index,
    )


def _has_runnable_or_unsettled_chain(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    b_by_index: dict[int, dict[str, Any]],
    c_by_index: dict[int, dict[str, Any]],
    d_by_index: dict[int, dict[str, Any]],
) -> bool:
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


def _all_selected_c_done(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    c_by_index: dict[int, dict[str, Any]],
) -> bool:
    for unit_index in selected_indexes:
        if unit_index in failed_chain_indexes:
            continue
        c_status = str(c_by_index.get(unit_index, {}).get("status", "pending"))
        if c_status != "done":
            return False
    return True


def _count_selected_c_done(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    c_by_index: dict[int, dict[str, Any]],
) -> int:
    done_count = 0
    for unit_index in selected_indexes:
        if unit_index in failed_chain_indexes:
            continue
        c_status = str(c_by_index.get(unit_index, {}).get("status", "pending"))
        if c_status == "done":
            done_count += 1
    return done_count


def _count_selected_done_units(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    unit_by_index: dict[int, dict[str, Any]],
) -> int:
    done_count = 0
    for unit_index in selected_indexes:
        if unit_index in failed_chain_indexes:
            continue
        status = str(unit_by_index.get(unit_index, {}).get("status", "pending"))
        if status == "done":
            done_count += 1
    return done_count


def _count_selected_not_done_units(
    selected_indexes: set[int],
    failed_chain_indexes: set[int],
    unit_by_index: dict[int, dict[str, Any]],
) -> int:
    not_done_count = 0
    for unit_index in selected_indexes:
        if unit_index in failed_chain_indexes:
            continue
        status = str(unit_by_index.get(unit_index, {}).get("status", "pending"))
        if status != "done":
            not_done_count += 1
    return not_done_count
