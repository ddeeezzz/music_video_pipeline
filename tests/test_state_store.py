"""
文件用途：验证状态库的核心状态流转与恢复规则。
核心流程：创建临时数据库，执行任务初始化、状态更新与恢复点查询。
输入输出：输入测试用 task_id，输出断言结果。
依赖说明：依赖 pytest 与项目内 StateStore。
维护说明：状态机规则调整时需同步修改本测试。
"""

# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于测试断言
import pytest

# 项目内模块：状态存储实现
from music_video_pipeline.state_store import StateStore


def test_state_store_should_follow_basic_flow(tmp_path: Path) -> None:
    """
    功能说明：验证模块状态从 pending 到 done 的基本流转。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：无。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_001", audio_path="a.mp3", config_path="c.json")

    status_map = store.get_module_status_map(task_id="task_001")
    assert status_map == {"A": "pending", "B": "pending", "C": "pending", "D": "pending"}
    assert store.first_non_done_module(task_id="task_001") == "A"

    store.set_module_status(task_id="task_001", module_name="A", status="done", artifact_path="a.json")
    store.set_module_status(task_id="task_001", module_name="B", status="done", artifact_path="b.json")
    store.set_module_status(task_id="task_001", module_name="C", status="done", artifact_path="c.json")
    store.set_module_status(task_id="task_001", module_name="D", status="done", artifact_path="d.mp4")
    store.mark_task_done_if_possible(task_id="task_001", output_video_path="d.mp4")

    task_record = store.get_task(task_id="task_001")
    assert task_record is not None
    assert task_record["status"] == "done"
    assert store.first_non_done_module(task_id="task_001") is None


def test_state_store_should_list_tasks_in_updated_order(tmp_path: Path) -> None:
    """
    功能说明：验证任务列表会按 updated_at 倒序返回关键字段。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同秒场景下允许 task_id 次级排序。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_old", audio_path="old.mp3", config_path="old.json")
    store.init_task(task_id="task_new", audio_path="new.mp3", config_path="new.json")
    store.update_task_status(task_id="task_old", status="running")
    store.update_task_status(task_id="task_new", status="failed")

    with store._connect() as connection:  # type: ignore[attr-defined]
        connection.execute("UPDATE tasks SET updated_at = ? WHERE task_id = ?", ("2026-04-17T12:00:00+08:00", "task_old"))
        connection.execute("UPDATE tasks SET updated_at = ? WHERE task_id = ?", ("2026-04-17T12:10:00+08:00", "task_new"))
        connection.commit()

    rows = store.list_tasks()
    assert [item["task_id"] for item in rows] == ["task_new", "task_old"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["config_path"] == "new.json"
    assert rows[0]["audio_path"] == "new.mp3"
    assert rows[0]["updated_at"] == "2026-04-17T12:10:00+08:00"


def test_state_store_should_list_task_module_status_map(tmp_path: Path) -> None:
    """
    功能说明：验证可批量返回任务 A/B/C/D 模块状态映射。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：缺失模块状态默认 pending。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_map_001", audio_path="a.mp3", config_path="a.json")
    store.init_task(task_id="task_map_002", audio_path="b.mp3", config_path="b.json")
    store.set_module_status(task_id="task_map_001", module_name="A", status="done", artifact_path="a_out.json")
    store.set_module_status(task_id="task_map_001", module_name="B", status="failed", error_message="mock")

    summary = store.list_task_module_status_map(task_ids=["task_map_001", "task_map_002"])
    assert summary["task_map_001"]["A"] == "done"
    assert summary["task_map_001"]["B"] == "failed"
    assert summary["task_map_001"]["C"] == "pending"
    assert summary["task_map_001"]["D"] == "pending"
    assert summary["task_map_002"] == {"A": "pending", "B": "pending", "C": "pending", "D": "pending"}


def test_state_store_should_block_downstream_when_upstream_not_done(tmp_path: Path) -> None:
    """
    功能说明：验证上游未完成时下游不可执行。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：无。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_002", audio_path="a.mp3", config_path="c.json")

    can_run_b, reason_b = store.can_run_module(task_id="task_002", module_name="B")
    assert can_run_b is False
    assert "上游模块 A 未完成" in reason_b

    store.set_module_status(task_id="task_002", module_name="A", status="done", artifact_path="a.json")
    can_run_b_after, _ = store.can_run_module(task_id="task_002", module_name="B")
    assert can_run_b_after is True


def test_state_store_should_reset_from_module(tmp_path: Path) -> None:
    """
    功能说明：验证 reset_from_module 会重置目标模块及其下游。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：上游模块状态应保持不变。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_003", audio_path="a.mp3", config_path="c.json")
    for module in ["A", "B", "C", "D"]:
        store.set_module_status(task_id="task_003", module_name=module, status="done", artifact_path=f"{module}.json")

    store.reset_from_module(task_id="task_003", module_name="C")
    status_map = store.get_module_status_map(task_id="task_003")
    assert status_map["A"] == "done"
    assert status_map["B"] == "done"
    assert status_map["C"] == "pending"
    assert status_map["D"] == "pending"


def test_state_store_should_reject_invalid_status(tmp_path: Path) -> None:
    """
    功能说明：验证非法状态值会被拒绝。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：无。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_004", audio_path="a.mp3", config_path="c.json")
    with pytest.raises(ValueError):
        store.set_module_status(task_id="task_004", module_name="A", status="unknown")


def test_reconcile_bcd_module_statuses_by_units_should_heal_stale_running_state(tmp_path: Path) -> None:
    """
    功能说明：验证当 B/C 单元已全部 done 时，模块级 running 状态会被自动自愈为 done。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：D 未完成时仅修正 B/C，不提前标记任务 done。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_heal_001", audio_path="a.mp3", config_path="c.json")
    store.update_task_status(task_id="task_heal_001", status="running")
    store.set_module_status(task_id="task_heal_001", module_name="A", status="done", artifact_path="a.json")
    store.set_module_status(task_id="task_heal_001", module_name="B", status="running")
    store.set_module_status(task_id="task_heal_001", module_name="C", status="running")
    store.set_module_status(task_id="task_heal_001", module_name="D", status="running", artifact_path="final.mp4")

    store.sync_module_units(
        task_id="task_heal_001",
        module_name="B",
        units=[{"unit_id": "seg_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0}],
    )
    store.sync_module_units(
        task_id="task_heal_001",
        module_name="C",
        units=[{"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0}],
    )
    store.sync_module_units(
        task_id="task_heal_001",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )

    store.set_module_unit_status(task_id="task_heal_001", module_name="B", unit_id="seg_001", status="done")
    store.set_module_unit_status(task_id="task_heal_001", module_name="C", unit_id="shot_001", status="done")
    store.set_module_unit_status(task_id="task_heal_001", module_name="D", unit_id="shot_001", status="done")
    store.set_module_unit_status(task_id="task_heal_001", module_name="D", unit_id="shot_002", status="running")

    healed_status_map = store.reconcile_bcd_module_statuses_by_units(task_id="task_heal_001")
    assert healed_status_map["A"] == "done"
    assert healed_status_map["B"] == "done"
    assert healed_status_map["C"] == "done"
    assert healed_status_map["D"] == "running"
    assert store.get_task(task_id="task_heal_001")["status"] == "running"

    store.set_module_unit_status(task_id="task_heal_001", module_name="D", unit_id="shot_002", status="done")
    healed_all_done = store.reconcile_bcd_module_statuses_by_units(task_id="task_heal_001")
    assert healed_all_done["D"] == "done"
    assert store.get_task(task_id="task_heal_001")["status"] == "done"


def test_state_store_should_sync_and_update_module_units(tmp_path: Path) -> None:
    """
    功能说明：验证模块单元状态可同步、更新并按状态筛选查询。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模块C单元以 unit_id 作为主键。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_001", audio_path="a.mp3", config_path="c.json")

    store.sync_module_units(
        task_id="task_units_001",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    pending_units = store.list_module_units_by_status(
        task_id="task_units_001",
        module_name="C",
        statuses=["pending"],
    )
    assert [item["unit_id"] for item in pending_units] == ["shot_001", "shot_002"]

    store.set_module_unit_status(
        task_id="task_units_001",
        module_name="C",
        unit_id="shot_001",
        status="done",
        artifact_path="/tmp/frame_001.png",
        error_message="",
    )
    store.set_module_unit_status(
        task_id="task_units_001",
        module_name="C",
        unit_id="shot_002",
        status="failed",
        artifact_path="",
        error_message="mock failed",
    )

    done_frame_items = store.list_module_c_done_frame_items(task_id="task_units_001")
    assert len(done_frame_items) == 1
    assert done_frame_items[0]["shot_id"] == "shot_001"
    assert done_frame_items[0]["frame_path"] == "/tmp/frame_001.png"

    failed_units = store.list_module_units_by_status(
        task_id="task_units_001",
        module_name="C",
        statuses=["failed"],
    )
    assert len(failed_units) == 1
    assert failed_units[0]["unit_id"] == "shot_002"
    assert failed_units[0]["error_message"] == "mock failed"


def test_state_store_should_clear_module_c_units_when_reset_from_upstream(tmp_path: Path) -> None:
    """
    功能说明：验证从 A/B/C 重置时会清理模块 C 单元状态，避免旧单元污染恢复流程。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：从 D 重置不应清理模块 C 单元状态。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_002", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_002",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
        ],
    )

    store.reset_from_module(task_id="task_units_002", module_name="D")
    units_after_reset_d = store.list_module_units_by_status(
        task_id="task_units_002",
        module_name="C",
        statuses=["pending"],
    )
    assert len(units_after_reset_d) == 1

    store.reset_from_module(task_id="task_units_002", module_name="C")
    units_after_reset_c = store.list_module_units_by_status(
        task_id="task_units_002",
        module_name="C",
        statuses=["pending", "running", "done", "failed"],
    )
    assert units_after_reset_c == []


def test_state_store_should_build_module_unit_status_summary(tmp_path: Path) -> None:
    """
    功能说明：验证模块单元状态摘要能正确统计各状态计数与问题单元列表。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例覆盖 pending/running/done/failed 四种状态。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_003", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_003",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
            {"unit_id": "shot_004", "unit_index": 3, "start_time": 3.0, "end_time": 4.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(
        task_id="task_units_003",
        module_name="C",
        unit_id="shot_001",
        status="done",
        artifact_path="/tmp/frame_001.png",
    )
    store.set_module_unit_status(
        task_id="task_units_003",
        module_name="C",
        unit_id="shot_002",
        status="failed",
        error_message="mock failed",
    )
    store.set_module_unit_status(
        task_id="task_units_003",
        module_name="C",
        unit_id="shot_003",
        status="running",
    )

    summary = store.get_module_unit_status_summary(task_id="task_units_003", module_name="C")
    assert summary["total_units"] == 4
    assert summary["status_counts"]["done"] == 1
    assert summary["status_counts"]["failed"] == 1
    assert summary["status_counts"]["running"] == 1
    assert summary["status_counts"]["pending"] == 1
    assert summary["done_unit_ids"] == ["shot_001"]
    assert summary["failed_unit_ids"] == ["shot_002"]
    assert summary["running_unit_ids"] == ["shot_003"]
    assert summary["pending_unit_ids"] == ["shot_004"]
    assert summary["problem_unit_ids"] == ["shot_002", "shot_003", "shot_004"]


def test_state_store_should_reset_single_module_unit(tmp_path: Path) -> None:
    """
    功能说明：验证重置单个模块单元时仅影响目标单元并清理相关字段。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不存在单元时应抛 RuntimeError。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_004", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_004",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(
        task_id="task_units_004",
        module_name="C",
        unit_id="shot_001",
        status="done",
        artifact_path="/tmp/frame_001.png",
    )
    store.set_module_unit_status(
        task_id="task_units_004",
        module_name="C",
        unit_id="shot_002",
        status="failed",
        artifact_path="",
        error_message="mock failed",
    )

    store.reset_module_unit(task_id="task_units_004", module_name="C", unit_id="shot_002")
    reset_record = store.get_module_unit_record(task_id="task_units_004", module_name="C", unit_id="shot_002")
    assert reset_record is not None
    assert reset_record["status"] == "pending"
    assert reset_record["artifact_path"] == ""
    assert reset_record["error_message"] == ""
    assert reset_record["started_at"] == ""
    assert reset_record["finished_at"] == ""

    untouched_record = store.get_module_unit_record(task_id="task_units_004", module_name="C", unit_id="shot_001")
    assert untouched_record is not None
    assert untouched_record["status"] == "done"
    assert untouched_record["artifact_path"] == "/tmp/frame_001.png"

    with pytest.raises(RuntimeError):
        store.reset_module_unit(task_id="task_units_004", module_name="C", unit_id="shot_404")


def test_state_store_should_list_module_b_done_shot_items(tmp_path: Path) -> None:
    """
    功能说明：验证模块B已完成单元读取接口可按顺序返回聚合所需字段。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅返回 status=done 且有 artifact_path 的记录。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_005", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_005",
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "seg_0002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(
        task_id="task_units_005",
        module_name="B",
        unit_id="seg_0001",
        status="done",
        artifact_path="/tmp/seg_0001.json",
        error_message="",
    )
    store.set_module_unit_status(
        task_id="task_units_005",
        module_name="B",
        unit_id="seg_0002",
        status="failed",
        artifact_path="",
        error_message="mock failed",
    )

    done_items = store.list_module_b_done_shot_items(task_id="task_units_005")
    assert len(done_items) == 1
    assert done_items[0]["unit_id"] == "seg_0001"
    assert done_items[0]["unit_index"] == 0
    assert done_items[0]["artifact_path"] == "/tmp/seg_0001.json"


def test_state_store_should_clear_module_b_units_when_reset_from_a_or_b(tmp_path: Path) -> None:
    """
    功能说明：验证从A/B重置时会清理模块B单元状态，避免旧单元污染恢复流程。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：从C重置不应清理模块B单元状态。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_006", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_006",
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
        ],
    )

    store.reset_from_module(task_id="task_units_006", module_name="C")
    units_after_reset_c = store.list_module_units_by_status(
        task_id="task_units_006",
        module_name="B",
        statuses=["pending"],
    )
    assert len(units_after_reset_c) == 1

    store.reset_from_module(task_id="task_units_006", module_name="B")
    units_after_reset_b = store.list_module_units_by_status(
        task_id="task_units_006",
        module_name="B",
        statuses=["pending", "running", "done", "failed"],
    )
    assert units_after_reset_b == []


def test_state_store_should_list_module_d_done_segment_items(tmp_path: Path) -> None:
    """
    功能说明：验证模块D已完成单元读取接口可按顺序返回终拼所需字段。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅返回 status=done 且有 artifact_path 的记录。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_007", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_007",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(
        task_id="task_units_007",
        module_name="D",
        unit_id="shot_001",
        status="done",
        artifact_path="/tmp/segment_001.mp4",
        error_message="",
    )
    store.set_module_unit_status(
        task_id="task_units_007",
        module_name="D",
        unit_id="shot_002",
        status="failed",
        artifact_path="",
        error_message="mock failed",
    )

    done_items = store.list_module_d_done_segment_items(task_id="task_units_007")
    assert len(done_items) == 1
    assert done_items[0]["unit_id"] == "shot_001"
    assert done_items[0]["unit_index"] == 0
    assert done_items[0]["artifact_path"] == "/tmp/segment_001.mp4"


def test_state_store_should_clear_module_d_units_when_reset_from_upstream_or_d(tmp_path: Path) -> None:
    """
    功能说明：验证从A/B/C/D重置时会清理模块D单元状态，避免旧单元污染恢复流程。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模块D单元清理策略应覆盖自身及全部上游触发场景。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_units_008", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_units_008",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
        ],
    )

    store.reset_from_module(task_id="task_units_008", module_name="D")
    units_after_reset_d = store.list_module_units_by_status(
        task_id="task_units_008",
        module_name="D",
        statuses=["pending", "running", "done", "failed"],
    )
    assert units_after_reset_d == []

    store.sync_module_units(
        task_id="task_units_008",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
        ],
    )
    store.reset_from_module(task_id="task_units_008", module_name="C")
    units_after_reset_c = store.list_module_units_by_status(
        task_id="task_units_008",
        module_name="D",
        statuses=["pending", "running", "done", "failed"],
    )
    assert units_after_reset_c == []


def test_state_store_should_build_bcd_chain_status(tmp_path: Path) -> None:
    """
    功能说明：验证 B/C/D 链路状态摘要可按 unit_index 聚合输出。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例覆盖 done/running/failed 三种链路状态。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_chain_001", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_chain_001",
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "seg_0002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "seg_0003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    store.sync_module_units(
        task_id="task_chain_001",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    store.sync_module_units(
        task_id="task_chain_001",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )

    store.set_module_unit_status(task_id="task_chain_001", module_name="B", unit_id="seg_0001", status="done", artifact_path="/tmp/b1.json")
    store.set_module_unit_status(task_id="task_chain_001", module_name="C", unit_id="shot_001", status="done", artifact_path="/tmp/c1.png")
    store.set_module_unit_status(task_id="task_chain_001", module_name="D", unit_id="shot_001", status="done", artifact_path="/tmp/d1.mp4")

    store.set_module_unit_status(task_id="task_chain_001", module_name="B", unit_id="seg_0002", status="done", artifact_path="/tmp/b2.json")
    store.set_module_unit_status(task_id="task_chain_001", module_name="C", unit_id="shot_002", status="running")

    store.set_module_unit_status(task_id="task_chain_001", module_name="B", unit_id="seg_0003", status="failed", error_message="mock failed")

    chain_rows = store.list_bcd_chain_status(task_id="task_chain_001")
    assert len(chain_rows) == 3
    assert chain_rows[0]["chain_status"] == "done"
    assert chain_rows[1]["chain_status"] == "running"
    assert chain_rows[2]["chain_status"] == "failed"
    assert chain_rows[0]["segment_id"] == "seg_0001"
    assert chain_rows[0]["shot_id"] == "shot_001"


def test_state_store_should_reset_bcd_chain_units_by_segment_id(tmp_path: Path) -> None:
    """
    功能说明：验证按 segment_id 重置链路时仅影响目标链路单元。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：重置后 B/C/D 目标链路状态均应为 pending。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_chain_002", audio_path="a.mp3", config_path="c.json")
    for module_name in ["A", "B", "C", "D"]:
        store.set_module_status(task_id="task_chain_002", module_name=module_name, status="done", artifact_path=f"{module_name}.json")
    store.sync_module_units(
        task_id="task_chain_002",
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "seg_0002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.sync_module_units(
        task_id="task_chain_002",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.sync_module_units(
        task_id="task_chain_002",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(task_id="task_chain_002", module_name="B", unit_id="seg_0001", status="done", artifact_path="/tmp/b1.json")
    store.set_module_unit_status(task_id="task_chain_002", module_name="B", unit_id="seg_0002", status="done", artifact_path="/tmp/b2.json")
    store.set_module_unit_status(task_id="task_chain_002", module_name="C", unit_id="shot_001", status="done", artifact_path="/tmp/c1.png")
    store.set_module_unit_status(task_id="task_chain_002", module_name="C", unit_id="shot_002", status="done", artifact_path="/tmp/c2.png")
    store.set_module_unit_status(task_id="task_chain_002", module_name="D", unit_id="shot_001", status="done", artifact_path="/tmp/d1.mp4")
    store.set_module_unit_status(task_id="task_chain_002", module_name="D", unit_id="shot_002", status="done", artifact_path="/tmp/d2.mp4")

    reset_result = store.reset_bcd_chain_units(task_id="task_chain_002", segment_id="seg_0002")
    assert reset_result["unit_index"] == 1
    assert reset_result["shot_id"] == "shot_002"

    b_1 = store.get_module_unit_record(task_id="task_chain_002", module_name="B", unit_id="seg_0001")
    b_2 = store.get_module_unit_record(task_id="task_chain_002", module_name="B", unit_id="seg_0002")
    c_1 = store.get_module_unit_record(task_id="task_chain_002", module_name="C", unit_id="shot_001")
    c_2 = store.get_module_unit_record(task_id="task_chain_002", module_name="C", unit_id="shot_002")
    d_1 = store.get_module_unit_record(task_id="task_chain_002", module_name="D", unit_id="shot_001")
    d_2 = store.get_module_unit_record(task_id="task_chain_002", module_name="D", unit_id="shot_002")
    assert b_1 is not None and b_1["status"] == "done"
    assert b_2 is not None and b_2["status"] == "pending"
    assert c_1 is not None and c_1["status"] == "done"
    assert c_2 is not None and c_2["status"] == "pending"
    assert d_1 is not None and d_1["status"] == "done"
    assert d_2 is not None and d_2["status"] == "pending"


def test_state_store_should_mark_bcd_downstream_blocked(tmp_path: Path) -> None:
    """
    功能说明：验证上游失败时可将目标链路下游单元标记为 failed。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不会覆盖已 done 的下游单元。
    """
    store = StateStore(db_path=tmp_path / "state.sqlite3")
    store.init_task(task_id="task_chain_003", audio_path="a.mp3", config_path="c.json")
    store.sync_module_units(
        task_id="task_chain_003",
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.sync_module_units(
        task_id="task_chain_003",
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    store.set_module_unit_status(task_id="task_chain_003", module_name="C", unit_id="shot_001", status="pending")
    store.set_module_unit_status(task_id="task_chain_003", module_name="D", unit_id="shot_001", status="done", artifact_path="/tmp/d1.mp4")

    store.mark_bcd_downstream_blocked(
        task_id="task_chain_003",
        unit_index=0,
        from_module="B",
        reason="upstream_blocked:B:mock failed",
    )

    c_1 = store.get_module_unit_record(task_id="task_chain_003", module_name="C", unit_id="shot_001")
    d_1 = store.get_module_unit_record(task_id="task_chain_003", module_name="D", unit_id="shot_001")
    assert c_1 is not None and c_1["status"] == "failed"
    assert c_1["error_message"] == "upstream_blocked:B:mock failed"
    assert d_1 is not None and d_1["status"] == "done"
