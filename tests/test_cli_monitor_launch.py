"""
文件用途：验证CLI手动 Web 命令（web）的解析、分发与入口页写入行为。
核心流程：调用 parser/dispatch/web helper，断言服务启动参数与任务目录产物。
输入输出：输入命令参数与FakeRunner，输出监控启动摘要断言结果。
依赖说明：依赖 argparse/logging 与项目内 cli/state_store。
维护说明：web 命令参数或入口页策略变更时需同步更新本测试。
"""

# 标准库：用于命令行命名空间
import argparse
# 标准库：用于 JSON 配置写入
import json
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
    功能说明：测试用调度器桩，仅提供 web 命令所需属性。
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


def test_build_parser_should_accept_web_command(tmp_path: Path) -> None:
    """
    功能说明：验证CLI解析器已注册 web 命令，并允许省略 task_id。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证参数解析，不触发实际服务。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    parser = cli._build_parser(workspace_root=workspace_root)

    args = parser.parse_args(["web", "--task-id", "task_cli_monitor_001"])
    assert args.command == "web"
    assert args.task_id == "task_cli_monitor_001"
    assert str(args.runs_dir).replace("\\", "/") == "runs"
    assert args.host == "127.0.0.1"
    assert args.port == 45705

    args_without_task_id = parser.parse_args(["web"])
    assert args_without_task_id.command == "web"
    assert args_without_task_id.task_id is None
    assert str(args_without_task_id.runs_dir).replace("\\", "/") == "runs"
    assert args_without_task_id.host == "127.0.0.1"
    assert args_without_task_id.port == 45705


def test_dispatch_command_should_route_to_web_runner(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证CLI分发能正确路由到 web 命令执行函数。
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
    logger = logging.getLogger("test_dispatch_command_should_route_to_web_runner")
    logger.setLevel(logging.INFO)

    called: list[str] = []

    def _fake_run_task_monitor_command(args, runner, logger):  # noqa: ANN001
        _ = (args, runner, logger)
        called.append("web")
        return {"task_id": "task_cli_monitor_001", "kind": "web"}

    monkeypatch.setattr(cli, "_run_task_monitor_command", _fake_run_task_monitor_command)

    result = cli._dispatch_command(
        args=argparse.Namespace(
            command="web",
            task_id="task_cli_monitor_001",
            runs_dir="runs",
            host="127.0.0.1",
            port=45705,
        ),
        runner=runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=None,  # type: ignore[arg-type]
        config_path=(workspace_root / "configs" / "music_yby" / "default.json").resolve(),
        logger=logger,
    )
    assert result["kind"] == "web"
    assert called == ["web"]


def test_execute_request_with_loaded_runtime_should_build_lightweight_runtime_for_web(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 web 命令执行时不会构建完整 PipelineRunner，而是走轻量运行时。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验运行时装配，不触发真实监督服务。
    """
    workspace_root = tmp_path / "workspace_runtime"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = cli.CommandRequest(
        command="web",
        task_id="task_cli_monitor_001",
        runs_dir=Path("runs"),
        monitor_host="127.0.0.1",
        monitor_port=45705,
        config_path=(workspace_root / "configs" / "music_yby" / "default.json").resolve(),
    )
    fake_runner = object()
    calls: list[str] = []

    monkeypatch.setattr(cli, "setup_logging", lambda level: "logger_obj")

    def _fail_build_pipeline_runner(*, workspace_root, config, logger):  # noqa: ANN001
        _ = (workspace_root, config, logger)
        raise AssertionError("web 命令不应构建完整 PipelineRunner")

    def _fake_build_web_command_runtime(*, workspace_root, request, logger):  # noqa: ANN001
        _ = (workspace_root, logger)
        assert request.command == "web"
        assert request.runs_dir == Path("runs")
        calls.append("web_runtime")
        return fake_runner

    def _fake_execute_request(*, request, runner, workspace_root, config, logger):  # noqa: ANN001
        _ = (request, workspace_root, config, logger)
        assert runner is fake_runner
        return {"kind": "web", "task_id": "task_cli_monitor_001"}

    monkeypatch.setattr(cli, "_build_pipeline_runner", _fail_build_pipeline_runner)
    monkeypatch.setattr(cli, "_build_web_command_runtime", _fake_build_web_command_runtime)
    monkeypatch.setattr(cli, "_execute_request", _fake_execute_request)

    result = cli._execute_request_with_loaded_runtime(
        workspace_root=workspace_root,
        request=request,
    )

    assert result["kind"] == "web"
    assert calls == ["web_runtime"]


def test_build_web_command_runtime_should_use_runs_dir_and_host_port(tmp_path: Path) -> None:
    """
    功能说明：验证 web 轻量运行时直接基于 runs_dir 与 host/port 构建状态库访问。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不依赖配置文件中的 runs_dir。
    """
    workspace_root = tmp_path / "workspace_web_runtime"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = cli.CommandRequest(
        command="web",
        task_id="task_cli_monitor_001",
        runs_dir=Path("custom_runs"),
        monitor_host="0.0.0.0",
        monitor_port=19090,
        config_path=(workspace_root / "configs" / "music_yby" / "default.json").resolve(),
    )

    runner = cli._build_web_command_runtime(
        workspace_root=workspace_root,
        request=request,
        logger="logger_obj",
    )

    assert runner.runs_dir == (workspace_root / "custom_runs").resolve()
    assert runner.state_store.db_path == (workspace_root / "custom_runs" / "pipeline_state.sqlite3").resolve()
    assert runner.config.monitoring.host == "0.0.0.0"
    assert runner.config.monitoring.port == 19090


def test_write_task_web_entry_page_should_write_redirect_file(tmp_path: Path) -> None:
    """
    功能说明：验证任务目录入口页会写入 runs/<task_id>/task_web.html 并包含跳转地址。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：入口页可重复覆盖写入。
    """
    task_dir = tmp_path / "runs" / "task_cli_monitor_002"
    launch_path = cli._write_task_web_entry_page(
        task_dir=task_dir,
        task_id="task_cli_monitor_002",
        monitor_url="http://127.0.0.1:45678/tasks/task_cli_monitor_002/monitor",
    )
    assert launch_path == task_dir / "task_web.html"
    assert launch_path.exists()
    html_text = launch_path.read_text(encoding="utf-8")
    assert "任务 Web 入口" in html_text
    assert "http://127.0.0.1:45678/tasks/task_cli_monitor_002/monitor" in html_text


def test_build_web_rerun_handler_should_remap_foreign_audio_path_before_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能说明：验证 Web 页“生成”按钮会先把旧外机音频路径回映射到当前工作区，再启动子进程。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证命令组装与状态库自愈，不启动真实子进程。
    """
    workspace_root = tmp_path / "workspace_web_rerun"
    resources_dir = workspace_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    local_audio_path = (resources_dir / "jieranduhuo01.mp3").resolve()
    local_audio_path.write_bytes(b"fake-audio")

    config_path = workspace_root / "configs" / "music_windows_4060" / "jieranduhuo.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"paths": {"default_audio_path": "resources/jieranduhuo01.mp3"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    task_id = "jieranduhuo01"
    state_store.init_task(
        task_id=task_id,
        audio_path="\\root\\data\\t1\\resources\\jieranduhuo.mp3",
        config_path=str(config_path),
    )

    logger = logging.getLogger("test_build_web_rerun_handler_should_remap_foreign_audio_path_before_subprocess")
    logger.setLevel(logging.INFO)
    captured: dict[str, object] = {}

    def _fake_run(command, cwd, check):  # noqa: ANN001
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["check"] = check
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    handler = cli._build_web_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )

    result = handler(task_id)

    assert result["task_id"] == task_id
    assert captured["cwd"] == str(workspace_root)
    command = captured["command"]
    assert isinstance(command, list)
    audio_flag_index = command.index("--audio-path")
    assert command[audio_flag_index + 1] == str(local_audio_path)
    task_record = state_store.get_task(task_id=task_id)
    assert task_record is not None
    assert task_record["audio_path"] == str(local_audio_path)


def test_run_task_monitor_command_should_reject_unknown_explicit_task(tmp_path: Path) -> None:
    """
    功能说明：验证 web 命令在显式 task_id 不存在时会报错。
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


def test_run_task_monitor_command_should_start_service_without_default_task_id(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 web 命令在未传 task_id 时可直接进入任务列表页。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：状态库为空时也应允许服务启动。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runner = _FakeRunner(state_store=state_store, runs_dir=runs_dir)
    logger = logging.getLogger("test_run_task_monitor_command_should_start_service_without_default_task_id")
    logger.setLevel(logging.INFO)

    calls: list[tuple[str, str, bool]] = []

    class _FakeMonitorService:
        def __init__(
            self,
            state_store,  # noqa: ANN001
            task_id,  # noqa: ANN001
            logger,  # noqa: ANN001
            rerun_handler=None,  # noqa: ANN001
            host="127.0.0.1",  # noqa: ANN001
            port=0,  # noqa: ANN001
            tick_seconds=1.0,  # noqa: ANN001
            auto_stop_on_terminal=True,  # noqa: ANN001
        ) -> None:
            _ = (state_store, logger, rerun_handler, host, port, tick_seconds)
            self.task_id = str(task_id)
            self._is_running = False
            self.monitor_url = "http://127.0.0.1:9999/tasks" if not self.task_id else f"http://127.0.0.1:9999/tasks/{self.task_id}/monitor"
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
        args=argparse.Namespace(task_id=None),
        runner=runner,  # type: ignore[arg-type]
        logger=logger,
    )

    assert summary["task_id"] == ""
    assert summary["monitor_url"] == "http://127.0.0.1:9999/tasks"
    assert summary["launch_page_path"] == ""
    assert summary["interrupted_by_user"] is False
    assert calls[0] == ("start", "", False)
    assert calls[1][0] == "wait"
    assert calls[2][0] == "stop"


def test_run_task_monitor_command_should_start_service_and_write_launch_page(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 web 命令会以手动模式启动服务并写入任务目录入口页。
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
            rerun_handler=None,  # noqa: ANN001
            host="127.0.0.1",  # noqa: ANN001
            port=0,  # noqa: ANN001
            tick_seconds=1.0,  # noqa: ANN001
            auto_stop_on_terminal=True,  # noqa: ANN001
        ) -> None:
            _ = (state_store, logger, rerun_handler, host, port, tick_seconds)
            self.task_id = str(task_id)
            self._is_running = False
            self.monitor_url = f"http://127.0.0.1:9999/tasks/{self.task_id}/monitor"
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
    assert f"/tasks/{task_id}/monitor" in summary["monitor_url"]
    assert Path(summary["launch_page_path"]).exists()
    assert calls[0] == ("start", task_id, False)
    assert calls[1][0] == "wait"
    assert calls[2][0] == "stop"
    launch_text = Path(summary["launch_page_path"]).read_text(encoding="utf-8")
    assert "task_web.html" in str(summary["launch_page_path"])
    assert summary["monitor_url"] in launch_text


def test_run_task_monitor_command_should_not_preload_latest_task_when_task_id_missing(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 web 命令在未传 task_id 时不会自动选择最新任务。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使状态库中已有任务，也应先落在总览页。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runner = _FakeRunner(state_store=state_store, runs_dir=runs_dir)
    logger = logging.getLogger("test_run_task_monitor_command_should_not_preload_latest_task_when_task_id_missing")
    logger.setLevel(logging.INFO)

    old_task_id = "task_cli_monitor_old"
    new_task_id = "task_cli_monitor_new"
    for task_id in (old_task_id, new_task_id):
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
            rerun_handler=None,  # noqa: ANN001
            host="127.0.0.1",  # noqa: ANN001
            port=0,  # noqa: ANN001
            tick_seconds=1.0,  # noqa: ANN001
            auto_stop_on_terminal=True,  # noqa: ANN001
        ) -> None:
            _ = (state_store, logger, rerun_handler, host, port, tick_seconds)
            self.task_id = str(task_id)
            self._is_running = False
            self.monitor_url = "http://127.0.0.1:9999/tasks" if not self.task_id else f"http://127.0.0.1:9999/tasks/{self.task_id}/monitor"
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
        args=argparse.Namespace(task_id=None),
        runner=runner,  # type: ignore[arg-type]
        logger=logger,
    )

    assert summary["task_id"] == ""
    assert summary["monitor_url"] == "http://127.0.0.1:9999/tasks"
    assert summary["launch_page_path"] == ""
    assert calls[0] == ("start", "", False)
    assert calls[1][0] == "wait"
    assert calls[2][0] == "stop"
