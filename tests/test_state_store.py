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
