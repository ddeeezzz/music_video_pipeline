"""
文件用途：验证跨模块 B/C/D 波前并行调度核心语义。
核心流程：构造链路单元与状态库，打桩单元执行函数，断言调度顺序、失败隔离与并发门控。
输入输出：输入临时任务环境，输出调度结果断言。
依赖说明：依赖 pytest 与项目内 cross_bcd.scheduler。
维护说明：跨模块调度策略变更时需同步更新本测试。
"""

# 标准库：用于日志构建
import logging
# 标准库：用于线程同步
import threading
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于短暂等待，模拟任务执行时间
import time

# 项目内模块：配置数据类
from music_video_pipeline.config import (
    AppConfig,
    CrossModuleAdaptiveWindowConfig,
    CrossModuleConfig,
    FfmpegConfig,
    LoggingConfig,
    MockConfig,
    ModeConfig,
    ModuleAConfig,
    ModuleBConfig,
    ModuleCConfig,
    ModuleDConfig,
    PathsConfig,
)
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：跨模块链路模型
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
# 项目内模块：跨模块调度器
from music_video_pipeline.modules.cross_bcd import scheduler
from music_video_pipeline.modules.cross_bcd import scheduler_adaptive
from music_video_pipeline.modules.cross_bcd import scheduler_engine
from music_video_pipeline.modules.cross_bcd import scheduler_tasks
# 项目内模块：模块 B 单元模型
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
# 项目内模块：模块 D 单元蓝图
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint
# 项目内模块：状态库
from music_video_pipeline.state_store import StateStore


def test_cross_scheduler_should_run_wavefront_order_and_finish_all_units(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨模块调度能按 B->C->D 波前推进并完成全部链路。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用打桩执行器避免真实模型依赖。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(tmp_path=tmp_path, task_id="chain_ok")
    events: list[tuple[str, int]] = []

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        events.append(("B", int(unit.unit_index)))
        time.sleep(0.01)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        events.append(("C", int(chain_unit.unit_index)))
        time.sleep(0.01)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = device_override
        events.append(("D", int(blueprint.unit_index)))
        time.sleep(0.01)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    summary_d = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")
    assert summary_d["status_counts"]["done"] == len(chain_units)

    stage_orders: dict[int, list[str]] = {}
    for stage, unit_index in events:
        stage_orders.setdefault(unit_index, []).append(stage)
    for unit_index in sorted(stage_orders):
        assert stage_orders[unit_index].index("B") < stage_orders[unit_index].index("C")
        assert stage_orders[unit_index].index("C") < stage_orders[unit_index].index("D")


def test_cross_scheduler_should_stop_only_failed_chain(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证某链路失败后仅阻断该链路，下游其余链路继续执行。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：失败链路采用模块B失败场景，触发下游阻断标记。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(tmp_path=tmp_path, task_id="chain_fail")

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        if int(unit.unit_index) == 1:
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="B",
                unit_id=unit.unit_id,
                status="failed",
                error_message="mock b fail",
            )
            raise RuntimeError("mock b fail")
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = device_override
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == [1]
    b2 = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="B", unit_id="seg_0002")
    c2 = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="C", unit_id="shot_002")
    d2 = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id="shot_002")
    assert b2 is not None and b2["status"] == "failed"
    assert c2 is not None and c2["status"] == "failed"
    assert "upstream_blocked:B" in str(c2["error_message"])
    assert d2 is not None and d2["status"] == "failed"
    assert "upstream_blocked:B" in str(d2["error_message"])

    d1 = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id="shot_001")
    d3 = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id="shot_003")
    assert d1 is not None and d1["status"] == "done"
    assert d3 is not None and d3["status"] == "done"


def test_cross_scheduler_should_respect_global_render_limit(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨模块 C/D 共享并发上限生效。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过并发计数器记录 C/D 峰值并发。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_limit",
        global_render_limit=2,
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    render_lock = threading.Lock()
    active_render = 0
    max_render = 0

    def _mark_render_enter() -> None:
        nonlocal active_render, max_render
        with render_lock:
            active_render += 1
            if active_render > max_render:
                max_render = active_render

    def _mark_render_leave() -> None:
        nonlocal active_render
        with render_lock:
            active_render -= 1

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        _mark_render_enter()
        try:
            time.sleep(0.02)
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="C",
                unit_id=chain_unit.shot_id,
                status="done",
                artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
            )
            return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}
        finally:
            _mark_render_leave()

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = device_override
        _mark_render_enter()
        try:
            time.sleep(0.02)
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="D",
                unit_id=blueprint.unit_id,
                status="done",
                artifact_path=str(blueprint.segment_path),
            )
            return str(blueprint.segment_path)
        finally:
            _mark_render_leave()

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert max_render <= 2
    assert result["adaptive_window_snapshot"]["enabled"] is False


def test_cross_scheduler_should_allow_d_dispatch_in_bc_when_secondary_gpu_available(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 BC 阶段在 C/D 分卡时允许 D 派发（避免次卡空闲）。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过状态计数检测 D 在 C 全量完成前是否已开始执行。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_phase_gate",
        global_render_limit=1,
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    d_started_before_all_c_done = {"value": False}

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        time.sleep(0.02)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (profile, device_override)
        c_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="C")
        if int(c_summary["status_counts"].get("done", 0)) < len(chain_units):
            d_started_before_all_c_done["value"] = True
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert d_started_before_all_c_done["value"] is True
    assert result["adaptive_window_snapshot"]["current_phase"] == "d"
    assert result["adaptive_window_snapshot"]["d_dispatch_enabled"] is True


def test_cross_scheduler_should_dispatch_second_animatediff_d_before_first_done(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 animatediff 调度在首个 D 未 done 前即可派发第二个 D。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过慢速 D 打桩构造窗口，不依赖真实 GPU。
    """
    adaptive_cfg = CrossModuleAdaptiveWindowConfig(
        enabled=True,
        probe_interval_ms=200,
        c_gpu_index=0,
        d_gpu_index=1,
        c_limit_min=1,
        c_limit_max=4,
        d_limit_min=1,
        d_limit_max=2,
    )
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_d_cold_start_gate_removed",
        chain_count=4,
        global_render_limit=2,
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=adaptive_cfg,
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})
    monkeypatch.setattr(
        scheduler_adaptive,
        "_run_gpu_probe_script",
        lambda context, timeout_seconds: (
            [
                {"index": 0, "total_mb": 15000, "used_mb": 1300, "used_ratio": 0.09},
                {"index": 1, "total_mb": 15000, "used_mb": 1500, "used_ratio": 0.10},
            ],
            "",
        ),
    )

    d_times_lock = threading.Lock()
    d_start_times: dict[str, float] = {}
    d_end_times: dict[str, float] = {}

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        _ = (c_row, generator)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile, device_override)
        start_time = time.perf_counter()
        with d_times_lock:
            d_start_times[blueprint.unit_id] = float(start_time)
        time.sleep(0.12)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        with d_times_lock:
            d_end_times[blueprint.unit_id] = float(time.perf_counter())
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert len(d_start_times) >= 2
    ordered_by_start = sorted(d_start_times.items(), key=lambda item: item[1])
    first_unit_id = ordered_by_start[0][0]
    second_unit_id = ordered_by_start[1][0]
    assert float(d_start_times[second_unit_id]) < float(d_end_times[first_unit_id])


def test_cross_scheduler_should_assign_animatediff_d_jobs_to_two_devices_in_d_phase(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 D 阶段 animatediff 任务可在双设备池分配，且同卡并发不超过1。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩设备分配与in-flight计数模拟，不依赖真实GPU。
    """
    adaptive_cfg = CrossModuleAdaptiveWindowConfig(
        enabled=True,
        probe_interval_ms=200,
        c_gpu_index=0,
        d_gpu_index=1,
        c_limit_min=1,
        c_limit_max=6,
        d_limit_min=1,
        d_limit_max=2,
    )
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_d_dual_device",
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=adaptive_cfg,
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})
    monkeypatch.setattr(
        scheduler_tasks,
        "prewarm_animatediff_runtime",
        lambda context, device_override=None: {"device": str(device_override), "cache_key": "mock-cache"},
    )
    monkeypatch.setattr(
        scheduler_adaptive,
        "_run_gpu_probe_script",
        lambda context, timeout_seconds: (
            [
                {"index": 0, "total_mb": 15000, "used_mb": 1200, "used_ratio": 0.08},
                {"index": 1, "total_mb": 15000, "used_mb": 1400, "used_ratio": 0.09},
            ],
            "",
        ),
    )

    device_events: list[str] = []
    inflight_by_device: dict[str, int] = {}
    max_inflight_by_device: dict[str, int] = {}
    state_lock = threading.Lock()

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile)
        active_device = str(device_override)
        with state_lock:
            device_events.append(active_device)
            inflight_by_device[active_device] = int(inflight_by_device.get(active_device, 0)) + 1
            current_active = inflight_by_device[active_device]
            previous_peak = int(max_inflight_by_device.get(active_device, 0))
            if current_active > previous_peak:
                max_inflight_by_device[active_device] = current_active
        try:
            time.sleep(0.03)
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="D",
                unit_id=blueprint.unit_id,
                status="done",
                artifact_path=str(blueprint.segment_path),
            )
            return str(blueprint.segment_path)
        finally:
            with state_lock:
                inflight_by_device[active_device] = int(inflight_by_device.get(active_device, 1)) - 1

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert "cuda:0" in device_events
    assert "cuda:1" in device_events
    assert max_inflight_by_device.get("cuda:0", 0) <= 2
    assert max_inflight_by_device.get("cuda:1", 0) <= 2
    snapshot = result["adaptive_window_snapshot"]
    assert snapshot["current_phase"] == "d"
    assert snapshot["d_dispatch_enabled"] is True
    assert snapshot["d_device_pool"] == ["cuda:0", "cuda:1"]


def test_cross_scheduler_should_use_single_gpu_mode_with_c_and_d_one_inflight_each(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单卡模式下 C/D 可并行但各模块内部并发固定为 1。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩模拟单卡环境，不依赖真实 GPU。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_single_gpu_mode",
        chain_count=6,
        global_render_limit=3,
        adaptive_enabled=True,
        render_backend="animatediff",
    )
    monkeypatch.setattr(scheduler_adaptive, "_detect_available_gpu_count", lambda context: (1, "test"))
    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})

    state_lock = threading.Lock()
    active_c = 0
    active_d = 0
    max_active_c = 0
    max_active_d = 0
    saw_c_and_d_overlap = False
    seen_d_devices: set[str] = set()

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        nonlocal active_c, max_active_c, saw_c_and_d_overlap
        _ = (c_row, generator)
        with state_lock:
            active_c += 1
            max_active_c = max(max_active_c, active_c)
            if active_d > 0:
                saw_c_and_d_overlap = True
        try:
            time.sleep(0.03)
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="C",
                unit_id=chain_unit.shot_id,
                status="done",
                artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
            )
            return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}
        finally:
            with state_lock:
                active_c -= 1

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        nonlocal active_d, max_active_d, saw_c_and_d_overlap
        _ = (c_row, profile)
        with state_lock:
            active_d += 1
            max_active_d = max(max_active_d, active_d)
            seen_d_devices.add(str(device_override))
            if active_c > 0:
                saw_c_and_d_overlap = True
        try:
            time.sleep(0.03)
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="D",
                unit_id=blueprint.unit_id,
                status="done",
                artifact_path=str(blueprint.segment_path),
            )
            return str(blueprint.segment_path)
        finally:
            with state_lock:
                active_d -= 1

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert max_active_c <= 1
    assert max_active_d <= 1
    assert saw_c_and_d_overlap is True
    assert seen_d_devices == {"cuda:0"}
    snapshot = result["adaptive_window_snapshot"]
    assert snapshot["single_gpu_mode"] is True
    assert snapshot["oom_fallback_locked_c_then_d"] is False
    assert snapshot["c_dynamic_limit"] == 1
    assert snapshot["d_dynamic_limit"] == 1
    assert snapshot["d_device_pool"] == ["cuda:0"]


def test_cross_scheduler_single_gpu_should_lock_c_then_d_after_oom(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单卡模式任一 CUDA OOM 后会锁定为先 C 后 D，不再回到 BC 并发 D。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证调度锁定语义，不依赖真实模型推理。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_single_gpu_oom_lock",
        chain_count=6,
        global_render_limit=3,
        adaptive_enabled=True,
        render_backend="animatediff",
    )
    monkeypatch.setattr(scheduler_adaptive, "_detect_available_gpu_count", lambda context: (1, "test"))
    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})

    first_oom_raised = {"value": False}
    c_done_before_non_oom_d: list[int] = []

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        _ = (c_row, generator)
        time.sleep(0.02)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile, device_override)
        if not first_oom_raised["value"]:
            first_oom_raised["value"] = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        c_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="C")
        c_done_before_non_oom_d.append(int(c_summary["status_counts"].get("done", 0)))
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert first_oom_raised["value"] is True
    assert result["failed_chain_indexes"] == [0]
    assert c_done_before_non_oom_d, "OOM 后至少应继续调度剩余 D 单元"
    assert min(c_done_before_non_oom_d) >= (len(chain_units) - 1)
    snapshot = result["adaptive_window_snapshot"]
    assert snapshot["single_gpu_mode"] is True
    assert snapshot["oom_fallback_locked_c_then_d"] is True


def test_cross_scheduler_adaptive_should_adjust_limit_by_gpu_ratio() -> None:
    """
    功能说明：验证自适应窗口可按显存占用比例放量/收缩，并遵守边界。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：低于低水位上调，高于高水位下调，未知占用保持不变。
    """
    up_limit = scheduler_adaptive._adjust_dynamic_limit(
        current_limit=2,
        used_ratio=0.20,
        low_watermark=0.65,
        high_watermark=0.90,
        limit_min=1,
        limit_max=3,
    )
    assert up_limit == 3

    down_limit = scheduler_adaptive._adjust_dynamic_limit(
        current_limit=3,
        used_ratio=0.95,
        low_watermark=0.65,
        high_watermark=0.90,
        limit_min=1,
        limit_max=3,
    )
    assert down_limit == 2

    stable_limit = scheduler_adaptive._adjust_dynamic_limit(
        current_limit=2,
        used_ratio=None,
        low_watermark=0.65,
        high_watermark=0.90,
        limit_min=1,
        limit_max=3,
    )
    assert stable_limit == 2


def test_cross_scheduler_adaptive_should_down_when_ratio_reach_096() -> None:
    """
    功能说明：验证高水位设为 0.96 时，显存占用达到阈值会触发窗口下调。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证调节函数行为，不依赖真实调度循环。
    """
    down_limit = scheduler_adaptive._adjust_dynamic_limit(
        current_limit=3,
        used_ratio=0.965,
        low_watermark=0.65,
        high_watermark=0.96,
        limit_min=1,
        limit_max=3,
    )
    assert down_limit == 2


def test_cross_scheduler_should_detect_two_round_trip_flap() -> None:
    """
    功能说明：验证窗口方向出现两轮来回抖动时可被识别。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅当最近4次变化为 +-+- 或 -+-+ 才命中。
    """
    assert scheduler_adaptive._is_two_round_trip_flap([1, -1, 1, -1]) is True
    assert scheduler_adaptive._is_two_round_trip_flap([-1, 1, -1, 1]) is True
    assert scheduler_adaptive._is_two_round_trip_flap([1, 1, -1, -1]) is False


def test_cross_scheduler_adaptive_should_fallback_to_static_when_probe_failed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 GPU 采样失败时，自适应窗口降级到静态窗口且不中断流程。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：采样失败不应抛异常，快照应返回 fallback 窗口。
    """
    context, _, _, _ = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_probe_fallback",
        global_render_limit=3,
        adaptive_enabled=True,
    )
    monkeypatch.setattr(scheduler_adaptive, "_run_gpu_probe_script", lambda context, timeout_seconds: ([], "mock probe failed"))

    snapshot = scheduler.collect_adaptive_window_status_snapshot(context=context)
    assert snapshot["enabled"] is True
    assert snapshot["last_probe_ok"] is False
    assert snapshot["last_probe_error"] == "mock probe failed"
    assert snapshot["c_dynamic_limit"] == snapshot["fallback_c_limit"]
    assert snapshot["d_dynamic_limit"] == snapshot["fallback_d_limit"]


def test_cross_scheduler_adaptive_should_follow_adaptive_range_when_backend_is_animatediff(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 animatediff 后端下，D 窗口由 adaptive_window 范围直接控制。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当 d_limit_max=2 时应保留 2，不再硬编码压到 1。
    """
    context, _, _, _ = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_adaptive_animatediff",
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=CrossModuleAdaptiveWindowConfig(
            enabled=True,
            d_limit_min=1,
            d_limit_max=2,
        ),
    )
    monkeypatch.setattr(scheduler_adaptive, "_detect_available_gpu_count", lambda context: (2, "test"))

    runtime = scheduler_adaptive._build_adaptive_window_runtime(
        context=context,
        global_render_limit=3,
        render_backend="animatediff",
    )
    assert runtime["d_limit_max"] == 2
    assert runtime["d_dynamic_limit"] == 2


def test_cross_scheduler_should_prewarm_d_phase_devices_once_and_skip_bc_warmed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 BC->D 切换时会异步预热 D 设备池，且 BC 已热设备会被跳过。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩预热函数与设备分配，不依赖真实 GPU/runtime。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_prewarm_skip_bc_warmed",
        chain_count=12,
        global_render_limit=2,
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=CrossModuleAdaptiveWindowConfig(
            enabled=True,
            probe_interval_ms=200,
            c_gpu_index=0,
            d_gpu_index=1,
            c_limit_min=1,
            c_limit_max=6,
            d_limit_min=1,
            d_limit_max=2,
        ),
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})
    monkeypatch.setattr(
        scheduler_adaptive,
        "_run_gpu_probe_script",
        lambda context, timeout_seconds: (
            [
                {"index": 0, "total_mb": 15000, "used_mb": 1200, "used_ratio": 0.08},
                {"index": 1, "total_mb": 15000, "used_mb": 1400, "used_ratio": 0.09},
            ],
            "",
        ),
    )

    prewarm_calls: list[str] = []
    bc_phase_d_devices: list[str] = []

    def _fake_prewarm(context, device_override=None):  # noqa: ANN001
        _ = context
        prewarm_calls.append(str(device_override))
        return {"device": str(device_override), "cache_key": f"cache-{device_override}"}

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        time.sleep(0.05)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile)
        c_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="C")
        if int(c_summary["status_counts"].get("done", 0)) < len(chain_units):
            bc_phase_d_devices.append(str(device_override))
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "prewarm_animatediff_runtime", _fake_prewarm)
    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert "cuda:1" in bc_phase_d_devices
    assert prewarm_calls.count("cuda:0") == 1
    assert "cuda:1" not in prewarm_calls


def test_cross_scheduler_should_skip_prewarm_when_c_not_done_is_not_greater_than_10(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 animatediff 下当 C 未完成数量 <=10 时，不提交 D runtime 异步预热。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使不预热，D 调度仍应完整执行并全部完成。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_prewarm_skip_threshold",
        chain_count=10,
        global_render_limit=2,
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=CrossModuleAdaptiveWindowConfig(
            enabled=True,
            probe_interval_ms=200,
            c_gpu_index=0,
            d_gpu_index=1,
            c_limit_min=1,
            c_limit_max=6,
            d_limit_min=1,
            d_limit_max=2,
        ),
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})
    monkeypatch.setattr(
        scheduler_adaptive,
        "_run_gpu_probe_script",
        lambda context, timeout_seconds: (
            [
                {"index": 0, "total_mb": 15000, "used_mb": 1200, "used_ratio": 0.08},
                {"index": 1, "total_mb": 15000, "used_mb": 1400, "used_ratio": 0.09},
            ],
            "",
        ),
    )

    prewarm_submit_calls = {"count": 0}

    def _fake_submit_prewarm(*args, **kwargs):  # noqa: ANN001
        _ = (args, kwargs)
        prewarm_submit_calls["count"] += 1

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        time.sleep(0.01)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile, device_override)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_submit_d_runtime_prewarm_tasks", _fake_submit_prewarm)
    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output={},
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == []
    assert prewarm_submit_calls["count"] == 0
    summary_d = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")
    assert int(summary_d["status_counts"].get("done", 0)) == len(chain_units)


def test_cross_scheduler_should_ignore_prewarm_failure_and_continue_d_dispatch(tmp_path: Path, monkeypatch, caplog) -> None:
    """
    功能说明：验证 runtime 预热失败仅记录 warning，不会阻断 D 单元调度。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    - caplog: pytest 日志捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：预热失败后 D 仍应全部完成且链路不记为 failed。
    """
    context, chain_units, b_units_map, d_blueprints_map = _build_cross_fixture(
        tmp_path=tmp_path,
        task_id="chain_prewarm_fail_non_blocking",
        chain_count=12,
        global_render_limit=2,
        adaptive_enabled=True,
        render_backend="animatediff",
        adaptive_window=CrossModuleAdaptiveWindowConfig(
            enabled=True,
            probe_interval_ms=200,
            c_gpu_index=0,
            d_gpu_index=1,
            c_limit_min=1,
            c_limit_max=6,
            d_limit_min=1,
            d_limit_max=2,
        ),
    )

    monkeypatch.setattr(scheduler_engine, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "animatediff"})
    monkeypatch.setattr(
        scheduler_adaptive,
        "_run_gpu_probe_script",
        lambda context, timeout_seconds: (
            [
                {"index": 0, "total_mb": 15000, "used_mb": 1200, "used_ratio": 0.08},
                {"index": 1, "total_mb": 15000, "used_mb": 1400, "used_ratio": 0.09},
            ],
            "",
        ),
    )

    def _fake_prewarm(context, device_override=None):  # noqa: ANN001
        _ = context
        if str(device_override) == "cuda:0":
            raise RuntimeError("mock prewarm failed")
        return {"device": str(device_override), "cache_key": f"cache-{device_override}"}

    def _fake_run_b(context, unit, generator, module_a_output, unit_outputs_dir):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(unit_outputs_dir / f"{unit.unit_id}.json"),
        )
        return "ok"

    def _fake_run_c(context, chain_unit, c_row, generator, frames_dir):
        time.sleep(0.05)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=chain_unit.shot_id,
            status="done",
            artifact_path=str(frames_dir / f"{chain_unit.shot_id}.png"),
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(context, blueprint, c_row, profile, device_override=None):
        _ = (c_row, profile, device_override)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "prewarm_animatediff_runtime", _fake_prewarm)
    monkeypatch.setattr(scheduler_tasks, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    with caplog.at_level(logging.WARNING):
        result = scheduler.execute_cross_bcd_wavefront(
            context=context,
            chain_units=chain_units,
            b_units_by_segment_id=b_units_map,
            d_blueprints_by_index=d_blueprints_map,
            module_a_output={},
            unit_outputs_dir=context.artifacts_dir / "module_b_units",
            frames_dir=context.artifacts_dir / "frames",
            target_segment_id=None,
        )

    assert result["failed_chain_indexes"] == []
    summary_d = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")
    assert int(summary_d["status_counts"].get("done", 0)) == len(chain_units)
    assert "模块D runtime 异步预热失败" in caplog.text


def _build_cross_fixture(
    tmp_path: Path,
    task_id: str,
    chain_count: int = 3,
    global_render_limit: int = 3,
    adaptive_enabled: bool = False,
    render_backend: str = "ffmpeg",
    adaptive_window: CrossModuleAdaptiveWindowConfig | None = None,
) -> tuple[RuntimeContext, list[CrossChainUnit], dict[str, ModuleBUnit], dict[int, ModuleDUnitBlueprint]]:
    """
    功能说明：构造跨模块调度测试所需上下文与单元蓝图。
    参数说明：
    - tmp_path: pytest 临时目录。
    - task_id: 任务标识。
    - global_render_limit: C/D 共享并发上限。
    返回值：
    - tuple: (RuntimeContext, 链路单元, B单元映射, D蓝图映射)。
    异常说明：无。
    边界条件：默认构建3条链路，可通过 chain_count 调整规模。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    task_dir = workspace_root / task_id
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")

    config = AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir="runs", default_audio_path="resources/demo.mp3"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        mock=MockConfig(beat_interval_seconds=0.5, video_width=960, video_height=540),
        module_b=ModuleBConfig(script_workers=3, unit_retry_times=1),
        module_c=ModuleCConfig(render_workers=3, unit_retry_times=1),
        module_d=ModuleDConfig(
            render_backend=render_backend,
            segment_workers=3,
            unit_retry_times=1,
        ),
        cross_module=CrossModuleConfig(
            global_render_limit=global_render_limit,
            scheduler_tick_ms=20,
            adaptive_window=adaptive_window
            if adaptive_window is not None
            else CrossModuleAdaptiveWindowConfig(enabled=adaptive_enabled, probe_interval_ms=200),
        ),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
    logger = logging.getLogger(f"cross_scheduler_test_{task_id}")
    logger.setLevel(logging.INFO)
    state_store = StateStore(db_path=workspace_root / "state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path="config.json")

    chain_units = [
        CrossChainUnit(
            unit_index=idx,
            segment_id=f"seg_{idx + 1:04d}",
            shot_id=f"shot_{idx + 1:03d}",
            start_time=float(idx),
            end_time=float(idx + 1),
            duration=1.0,
        )
        for idx in range(chain_count)
    ]
    b_units = {
        item.segment_id: ModuleBUnit(
            unit_id=item.segment_id,
            unit_index=item.unit_index,
            segment={"segment_id": item.segment_id},
            start_time=item.start_time,
            end_time=item.end_time,
            duration=item.duration,
        )
        for item in chain_units
    }
    d_blueprints = {
        item.unit_index: ModuleDUnitBlueprint(
            unit_id=item.shot_id,
            unit_index=item.unit_index,
            start_time=item.start_time,
            end_time=item.end_time,
            duration=item.duration,
            exact_frames=24,
            segment_path=artifacts_dir / "segments" / f"segment_{item.unit_index + 1:03d}.mp4",
            temp_segment_path=artifacts_dir / "segments" / f"segment_{item.unit_index + 1:03d}.tmp.mp4",
        )
        for item in chain_units
    }

    state_store.sync_module_units(
        task_id=task_id,
        module_name="B",
        units=[
            {
                "unit_id": item.segment_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in chain_units
        ],
    )
    state_store.sync_module_units(
        task_id=task_id,
        module_name="C",
        units=[
            {
                "unit_id": item.shot_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in chain_units
        ],
    )
    state_store.sync_module_units(
        task_id=task_id,
        module_name="D",
        units=[
            {
                "unit_id": item.shot_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in chain_units
        ],
    )

    context = RuntimeContext(
        task_id=task_id,
        audio_path=audio_path,
        task_dir=task_dir,
        artifacts_dir=artifacts_dir,
        config=config,
        logger=logger,
        state_store=state_store,
    )
    return context, chain_units, b_units, d_blueprints
