"""
文件用途：验证CLI手动监督命令（monitor）的解析、分发与入口页写入行为。
核心流程：调用 parser/dispatch/monitor helper，断言服务启动参数与任务目录产物。
输入输出：输入命令参数与FakeRunner，输出监控启动摘要断言结果。
依赖说明：依赖 argparse/logging 与项目内 cli/state_store。
维护说明：monitor 命令参数或入口页策略变更时需同步更新本测试。
"""

# 标准库：用于命令行命名空间
import argparse
# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：CLI实现
from music_video_pipeline import cli
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


class _FakeRunner:
    """
    功能说明：测试用调度器桩，仅提供 monitor 命令所需属性。
    参数说明：
    - state_store: 状态库对象。
    - runs_dir: 任务输出根目录。
    返回值：不适用。
    异常说明：不适用。
    边界条件：本测试不触发真实模块执行。
    """

    def __init__(self, state_store: StateStore, runs_dir: Path) -> None:
        self.state_store = state_store
        self.runs_dir = runs_dir


def test_build_parser_should_accept_monitor_command(tmp_path: Path) -> None:
    """
    功能说明：验证CLI解析器已注册 monitor 命令并支持 task_id 参数。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证参数解析，不触发实际服务。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    parser = cli._build_parser(workspace_root=workspace_root)

    args = parser.parse_args(["monitor", "--task-id", "task_cli_monitor_001"])
    assert args.command == "monitor"
    assert args.task_id == "task_cli_monitor_001"
    assert str(args.config).endswith("configs/default.json")


def test_dispatch_command_should_route_to_monitor_runner(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证CLI分发能正确路由到 monitor 命令执行函数。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用假实现避免阻塞等待。
    """
    workspace_root = tmp_path / "workspace_dispatch"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    runner = _FakeRunner(state_store=state_store, runs_dir=runs_dir)
    logger = logging.getLogger("test_dispatch_command_should_route_to_monitor_runner")
    logger.setLevel(logging.INFO)

    called: list[str] = []

    def _fake_run_task_monitor_command(args, runner, logger):  # noqa: ANN001
        _ = (args, runner, logger)
        called.append("monitor")
        return {"task_id": "task_cli_monitor_001", "kind": "monitor"}

    monkeypatch.setattr(cli, "_run_task_monitor_command", _fake_run_task_monitor_command)

    result = cli._dispatch_command(
        args=argparse.Namespace(command="monitor", task_id="task_cli_monitor_001"),
        runner=runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=None,  # type: ignore[arg-type]
        config_path=(workspace_root / "configs" / "default.json").resolve(),
        logger=logger,
    )
    assert result["kind"] == "monitor"
    assert called == ["monitor"]


def test_write_task_monitor_launch_page_should_write_redirect_file(tmp_path: Path) -> None:
    """
    功能说明：验证任务目录入口页会写入 runs/<task_id>/task_monitor.html 并包含跳转地址。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：入口页可重复覆盖写入。
    """
    task_dir = tmp_path / "runs" / "task_cli_monitor_002"
    launch_path = cli._write_task_monitor_launch_page(
        task_dir=task_dir,
        task_id="task_cli_monitor_002",
        monitor_url="http://127.0.0.1:45678/task-monitor?task_id=task_cli_monitor_002",
    )
    assert launch_path == task_dir / "task_monitor.html"
    assert launch_path.exists()
    html_text = launch_path.read_text(encoding="utf-8")
    assert "任务监督入口" in html_text
    assert "http://127.0.0.1:45678/task-monitor?task_id=task_cli_monitor_002" in html_text


def test_run_task_monitor_command_should_require_existing_task(tmp_path: Path) -> None:
    """
    功能说明：验证 monitor 命令在 task_id 不存在时会报错。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验不存在任务场景。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    runner = _FakeRunner(state_store=state_store, runs_dir=tmp_path / "runs")
    logger = logging.getLogger("test_run_task_monitor_command_should_require_existing_task")
    logger.setLevel(logging.INFO)

    try:
        cli._run_task_monitor_command(
            args=argparse.Namespace(task_id="task_not_found"),
            runner=runner,  # type: ignore[arg-type]
            logger=logger,
        )
        assert False, "预期应抛出 RuntimeError"
    except RuntimeError as error:
        assert "任务不存在" in str(error)


def test_run_task_monitor_command_should_start_service_and_write_launch_page(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 monitor 命令会以手动模式启动服务并写入任务目录入口页。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用FakeMonitor保证测试不阻塞。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runner = _FakeRunner(state_store=state_store, runs_dir=runs_dir)
    logger = logging.getLogger("test_run_task_monitor_command_should_start_service_and_write_launch_page")
    logger.setLevel(logging.INFO)

    task_id = "task_cli_monitor_003"
    task_dir = runs_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_path / f"{task_id}.mp3"
    config_path = tmp_path / f"{task_id}.json"
    audio_path.write_bytes(b"fake")
    config_path.write_text("{}", encoding="utf-8")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))

    calls: list[tuple[str, str, bool]] = []

    class _FakeMonitorService:
        def __init__(
            self,
            state_store,  # noqa: ANN001
            task_id,  # noqa: ANN001
            logger,  # noqa: ANN001
            host="127.0.0.1",  # noqa: ANN001
            port=0,  # noqa: ANN001
            tick_seconds=1.0,  # noqa: ANN001
            auto_stop_on_terminal=True,  # noqa: ANN001
        ) -> None:
            _ = (state_store, logger, host, port, tick_seconds)
            self.task_id = str(task_id)
            self._is_running = False
            self.monitor_url = f"http://127.0.0.1:9999/task-monitor?task_id={self.task_id}"
            self._auto_stop_on_terminal = bool(auto_stop_on_terminal)

        @property
        def is_running(self) -> bool:
            return self._is_running

        def start(self) -> None:
            self._is_running = True
            calls.append(("start", self.task_id, self._auto_stop_on_terminal))

        def wait_until_stopped(self, timeout_seconds=None) -> bool:  # noqa: ANN001
            _ = timeout_seconds
            calls.append(("wait", self.task_id, self._auto_stop_on_terminal))
            self._is_running = False
            return True

        def stop(self) -> None:
            calls.append(("stop", self.task_id, self._auto_stop_on_terminal))
            self._is_running = False

    monkeypatch.setattr(cli, "TaskMonitorService", _FakeMonitorService)

    summary = cli._run_task_monitor_command(
        args=argparse.Namespace(task_id=task_id),
        runner=runner,  # type: ignore[arg-type]
        logger=logger,
    )

    assert summary["task_id"] == task_id
    assert summary["interrupted_by_user"] is False
    assert "task-monitor?task_id=" in summary["monitor_url"]
    assert Path(summary["launch_page_path"]).exists()
    assert calls[0] == ("start", task_id, False)
    assert calls[1][0] == "wait"
    assert calls[2][0] == "stop"
    launch_text = Path(summary["launch_page_path"]).read_text(encoding="utf-8")
    assert "task_monitor.html" in str(summary["launch_page_path"])
    assert summary["monitor_url"] in launch_text
