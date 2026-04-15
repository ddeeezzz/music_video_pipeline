"""
文件用途：验证任务监督快照构建逻辑。
核心流程：构造不同任务阶段状态并断言快照字段。
输入输出：输入状态库记录，输出快照断言结果。
依赖说明：依赖 pytest 与项目内 state_store/monitoring。
维护说明：快照字段变更时需同步更新本测试。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：监督快照构建函数
from music_video_pipeline.monitoring.snapshot import build_task_monitor_snapshot
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def test_build_task_monitor_snapshot_should_return_not_found_when_task_missing(tmp_path: Path) -> None:
    """
    功能说明：验证任务不存在时快照返回 not_found。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模块与链路统计应返回空值。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    snapshot = build_task_monitor_snapshot(state_store=state_store, task_id="task_missing")
    assert snapshot["task_id"] == "task_missing"
    assert snapshot["task_status"] == "not_found"
    assert snapshot["bcd_chains"] == []
    assert snapshot["chain_counts"] == {"pending": 0, "running": 0, "done": 0, "failed": 0}
    assert snapshot["module_overview"]["A"]["status"] == "not_found"


def test_build_task_monitor_snapshot_should_handle_task_initializing(tmp_path: Path) -> None:
    """
    功能说明：验证任务初始化阶段快照字段。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：B/C/D 尚未同步单元时进度应为0。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_pending"
    audio_path = tmp_path / "pending.mp3"
    config_path = tmp_path / "config.json"
    audio_path.write_bytes(b"fake")
    config_path.write_text("{}", encoding="utf-8")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))

    snapshot = build_task_monitor_snapshot(state_store=state_store, task_id=task_id)
    assert snapshot["task_status"] == "pending"
    assert snapshot["module_overview"]["A"]["total"] == 1
    assert snapshot["module_overview"]["A"]["done"] == 0
    assert snapshot["module_overview"]["B"]["total"] == 0
    assert snapshot["module_overview"]["C"]["total"] == 0
    assert snapshot["module_overview"]["D"]["total"] == 0


def test_build_task_monitor_snapshot_should_aggregate_running_progress(tmp_path: Path) -> None:
    """
    功能说明：验证运行中快照可正确聚合模块进度与链路计数。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：链路状态应遵循 B/C/D 当前状态合并规则。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_running"
    _seed_task_and_units(state_store=state_store, task_id=task_id, workspace=tmp_path)

    state_store.update_task_status(task_id=task_id, status="running")
    state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    state_store.set_module_status(task_id=task_id, module_name="B", status="running")
    state_store.set_module_status(task_id=task_id, module_name="C", status="running")
    state_store.set_module_status(task_id=task_id, module_name="D", status="running")

    state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0001", status="done")
    state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0002", status="done")
    state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="b failed")
    state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_001", status="done")
    state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_002", status="running")
    state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id="shot_001", status="done")

    snapshot = build_task_monitor_snapshot(state_store=state_store, task_id=task_id)
    assert snapshot["task_status"] == "running"
    assert snapshot["module_overview"]["A"]["progress"] == 100.0
    assert snapshot["module_overview"]["B"]["progress"] == 66.67
    assert snapshot["module_overview"]["C"]["progress"] == 33.33
    assert snapshot["module_overview"]["D"]["progress"] == 33.33
    assert snapshot["chain_counts"]["done"] == 1
    assert snapshot["chain_counts"]["running"] == 1
    assert snapshot["chain_counts"]["failed"] == 1


def test_build_task_monitor_snapshot_should_return_failed_task_status(tmp_path: Path) -> None:
    """
    功能说明：验证失败任务可在快照中返回 failed 状态。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：失败状态下模块进度仍保留当前值。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_failed"
    _seed_task_and_units(state_store=state_store, task_id=task_id, workspace=tmp_path)
    state_store.update_task_status(task_id=task_id, status="failed", error_message="mock failed")
    snapshot = build_task_monitor_snapshot(state_store=state_store, task_id=task_id)
    assert snapshot["task_status"] == "failed"
    assert snapshot["task_id"] == task_id


def test_build_task_monitor_snapshot_should_return_done_progress(tmp_path: Path) -> None:
    """
    功能说明：验证完成任务可返回100%进度与全done链路计数。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：B/C/D单元全done时链路应全done。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_done"
    _seed_task_and_units(state_store=state_store, task_id=task_id, workspace=tmp_path)

    state_store.set_module_status(task_id=task_id, module_name="A", status="done")
    state_store.set_module_status(task_id=task_id, module_name="B", status="done")
    state_store.set_module_status(task_id=task_id, module_name="C", status="done")
    state_store.set_module_status(task_id=task_id, module_name="D", status="done")
    for unit_id in ["seg_0001", "seg_0002", "seg_0003"]:
        state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id=unit_id, status="done")
    for unit_id in ["shot_001", "shot_002", "shot_003"]:
        state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id=unit_id, status="done")
        state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id=unit_id, status="done")
    state_store.update_task_status(task_id=task_id, status="done", output_video_path="final.mp4")

    snapshot = build_task_monitor_snapshot(state_store=state_store, task_id=task_id)
    assert snapshot["task_status"] == "done"
    assert snapshot["module_overview"]["A"]["progress"] == 100.0
    assert snapshot["module_overview"]["B"]["progress"] == 100.0
    assert snapshot["module_overview"]["C"]["progress"] == 100.0
    assert snapshot["module_overview"]["D"]["progress"] == 100.0
    assert snapshot["chain_counts"]["done"] == 3
    assert snapshot["chain_counts"]["failed"] == 0


def _seed_task_and_units(state_store: StateStore, task_id: str, workspace: Path) -> None:
    """
    功能说明：初始化任务基础记录并同步三条B/C/D单元。
    参数说明：
    - state_store: 状态库对象。
    - task_id: 任务标识。
    - workspace: 临时目录。
    返回值：无。
    异常说明：无。
    边界条件：unit_index 与 shot_id 按固定映射写入。
    """
    audio_path = workspace / f"{task_id}.mp3"
    config_path = workspace / f"{task_id}.json"
    audio_path.write_bytes(b"fake")
    config_path.write_text("{}", encoding="utf-8")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
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
