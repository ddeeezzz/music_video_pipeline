"""
文件用途：验证任务监督服务的HTTP与WebSocket行为。
核心流程：启动服务、访问页面、接收快照并验证终态自动停止。
输入输出：输入任务状态记录，输出监控服务行为断言。
依赖说明：依赖 urllib/asyncio 与项目内 monitoring/state_store。
维护说明：服务URL与推送协议变更时需同步更新本测试。
"""

# 标准库：用于异步运行WebSocket客户端
import asyncio
# 标准库：用于JSON解析
import json
# 标准库：用于日志对象
import logging
# 标准库：用于线程执行异步客户端
import threading
# 标准库：用于时间轮询
import time
# 标准库：用于HTTP请求
from urllib.request import urlopen
# 标准库：用于URL解析
from urllib.parse import urlparse
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：测试框架
import pytest

# 项目内模块：任务监督服务
from music_video_pipeline.monitoring.server import TaskMonitorService
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def test_task_monitor_service_should_serve_page_and_push_snapshot(tmp_path: Path) -> None:
    """
    功能说明：验证监督服务可返回HTML并推送WebSocket快照。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：任务未结束时服务保持运行。
    """
    websockets = pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_001"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_serve_page_and_push_snapshot")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(state_store=state_store, task_id=task_id, logger=logger, tick_seconds=0.2)
    service.start()
    try:
        html_text = urlopen(service.monitor_url, timeout=3).read().decode("utf-8")
        assert "任务监督面板" in html_text
        monitor_parsed = urlparse(service.monitor_url)
        snapshot_url = f"http://{monitor_parsed.netloc}/snapshot?task_id={task_id}"
        snapshot_payload = json.loads(urlopen(snapshot_url, timeout=3).read().decode("utf-8"))
        assert snapshot_payload["task_id"] == task_id
        assert snapshot_payload["task_status"] == "running"

        async def _recv_one_snapshot() -> dict:
            async with websockets.connect(service.websocket_url_for(task_id=task_id)) as websocket:
                payload = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                return json.loads(str(payload))

        snapshot = asyncio.run(_recv_one_snapshot())
        assert snapshot["task_id"] == task_id
        assert snapshot["task_status"] == "running"
        assert "module_overview" in snapshot
        assert "bcd_chains" in snapshot
    finally:
        service.stop()


def test_task_monitor_service_should_stop_when_task_finished(tmp_path: Path) -> None:
    """
    功能说明：验证任务进入终态后监督服务会自动停止。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：自动停止后再次 stop 应保持幂等。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_002"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_stop_when_task_finished")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(state_store=state_store, task_id=task_id, logger=logger, tick_seconds=0.2)
    service.start()
    state_store.update_task_status(task_id=task_id, status="done", output_video_path="final.mp4")

    deadline = time.time() + 5.0
    while service.is_running and time.time() < deadline:
        time.sleep(0.1)

    assert service.is_running is False
    service.stop()


def test_task_monitor_service_should_wait_for_browser_close_after_terminal(tmp_path: Path) -> None:
    """
    功能说明：验证任务终态后若仍有WS连接，服务会等待连接关闭再停止。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证单连接场景。
    """
    websockets = pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_003"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_wait_for_browser_close_after_terminal")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(state_store=state_store, task_id=task_id, logger=logger, tick_seconds=0.2)
    service.start()

    ws_opened = threading.Event()
    ws_release = threading.Event()

    def _hold_websocket() -> None:
        async def _run() -> None:
            async with websockets.connect(service.websocket_url_for(task_id=task_id)) as websocket:
                _ = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                ws_opened.set()
                while not ws_release.is_set():
                    await asyncio.sleep(0.1)

        asyncio.run(_run())

    ws_thread = threading.Thread(target=_hold_websocket, daemon=True)
    ws_thread.start()
    assert ws_opened.wait(timeout=3.0)

    state_store.update_task_status(task_id=task_id, status="done", output_video_path="final.mp4")
    time.sleep(0.8)
    assert service.is_running is True

    ws_release.set()
    ws_thread.join(timeout=3.0)

    deadline = time.time() + 5.0
    while service.is_running and time.time() < deadline:
        time.sleep(0.1)
    assert service.is_running is False


def _seed_task(state_store: StateStore, task_id: str, workspace: Path) -> None:
    """
    功能说明：写入测试任务初始化记录。
    参数说明：
    - state_store: 状态库对象。
    - task_id: 任务标识。
    - workspace: 临时目录路径。
    返回值：无。
    异常说明：无。
    边界条件：仅用于监督服务测试，不要求完整模块产物。
    """
    audio_path = workspace / f"{task_id}.mp3"
    config_path = workspace / f"{task_id}.json"
    audio_path.write_bytes(b"fake")
    config_path.write_text("{}", encoding="utf-8")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
