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

    monkeypatch.setattr(scheduler, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "resolve_render_profile", lambda context: {"name": "cpu"})

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

    def _fake_run_d(context, blueprint, c_row, profile):
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

    monkeypatch.setattr(scheduler, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler, "_run_d_chain_unit", _fake_run_d)

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

    monkeypatch.setattr(scheduler, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "resolve_render_profile", lambda context: {"name": "cpu"})

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

    def _fake_run_d(context, blueprint, c_row, profile):
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=blueprint.unit_id,
            status="done",
            artifact_path=str(blueprint.segment_path),
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler, "_run_d_chain_unit", _fake_run_d)

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

    monkeypatch.setattr(scheduler, "build_script_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "build_frame_generator", lambda mode, logger, module_b_config=None: object())
    monkeypatch.setattr(scheduler, "resolve_render_profile", lambda context: {"name": "cpu"})

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

    def _fake_run_d(context, blueprint, c_row, profile):
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

    monkeypatch.setattr(scheduler, "_run_b_chain_unit", _fake_run_b)
    monkeypatch.setattr(scheduler, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler, "_run_d_chain_unit", _fake_run_d)

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


def _build_cross_fixture(
    tmp_path: Path,
    task_id: str,
    global_render_limit: int = 3,
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
    边界条件：固定构建3条链路，便于验证并发行为。
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
        module_d=ModuleDConfig(segment_workers=3, unit_retry_times=1),
        cross_module=CrossModuleConfig(global_render_limit=global_render_limit, scheduler_tick_ms=20),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
    logger = logging.getLogger(f"cross_scheduler_test_{task_id}")
    logger.setLevel(logging.INFO)
    state_store = StateStore(db_path=workspace_root / "state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path="config.json")

    chain_units = [
        CrossChainUnit(unit_index=0, segment_id="seg_0001", shot_id="shot_001", start_time=0.0, end_time=1.0, duration=1.0),
        CrossChainUnit(unit_index=1, segment_id="seg_0002", shot_id="shot_002", start_time=1.0, end_time=2.0, duration=1.0),
        CrossChainUnit(unit_index=2, segment_id="seg_0003", shot_id="shot_003", start_time=2.0, end_time=3.0, duration=1.0),
    ]
    b_units = {
        "seg_0001": ModuleBUnit(unit_id="seg_0001", unit_index=0, segment={"segment_id": "seg_0001"}, start_time=0.0, end_time=1.0, duration=1.0),
        "seg_0002": ModuleBUnit(unit_id="seg_0002", unit_index=1, segment={"segment_id": "seg_0002"}, start_time=1.0, end_time=2.0, duration=1.0),
        "seg_0003": ModuleBUnit(unit_id="seg_0003", unit_index=2, segment={"segment_id": "seg_0003"}, start_time=2.0, end_time=3.0, duration=1.0),
    }
    d_blueprints = {
        0: ModuleDUnitBlueprint(
            unit_id="shot_001",
            unit_index=0,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            exact_frames=24,
            segment_path=artifacts_dir / "segments" / "segment_001.mp4",
            temp_segment_path=artifacts_dir / "segments" / "segment_001.tmp.mp4",
        ),
        1: ModuleDUnitBlueprint(
            unit_id="shot_002",
            unit_index=1,
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            exact_frames=24,
            segment_path=artifacts_dir / "segments" / "segment_002.mp4",
            temp_segment_path=artifacts_dir / "segments" / "segment_002.tmp.mp4",
        ),
        2: ModuleDUnitBlueprint(
            unit_id="shot_003",
            unit_index=2,
            start_time=2.0,
            end_time=3.0,
            duration=1.0,
            exact_frames=24,
            segment_path=artifacts_dir / "segments" / "segment_003.mp4",
            temp_segment_path=artifacts_dir / "segments" / "segment_003.tmp.mp4",
        ),
    }

    state_store.sync_module_units(
        task_id=task_id,
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "seg_0002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "seg_0003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    state_store.sync_module_units(
        task_id=task_id,
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    state_store.sync_module_units(
        task_id=task_id,
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
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
