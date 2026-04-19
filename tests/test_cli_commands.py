"""
文件用途：验证CLI命令解析与分发逻辑（含模块C排障命令）。
核心流程：构造解析器输入参数，调用分发函数并断言runner调用行为。
输入输出：输入命令参数数组，输出断言结果。
依赖说明：依赖 argparse/pathlib 与项目内 cli/config 类型。
维护说明：新增子命令时需同步补充解析与分发测试。
"""

# 标准库：用于命令行命名空间构造
import argparse
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：CLI实现
from music_video_pipeline import cli


class _FakeRunner:
    """
    功能说明：测试用调度器桩对象，记录被调用的方法与参数。
    参数说明：无。
    返回值：不适用。
    异常说明：不适用。
    边界条件：仅实现本测试覆盖到的接口。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_module_c_status_summary(self, task_id: str, config_path: Path) -> dict:
        """
        功能说明：记录模块C状态查询调用并返回占位结果。
        参数说明：
        - task_id: 任务标识。
        - config_path: 配置路径。
        返回值：
        - dict: 占位摘要。
        异常说明：无。
        边界条件：无。
        """
        self.calls.append(
            (
                "get_module_c_status_summary",
                {"task_id": task_id, "config_path": str(config_path)},
            )
        )
        return {"task_id": task_id, "kind": "c-status"}

    def retry_module_c_shot(self, task_id: str, shot_id: str, config_path: Path) -> dict:
        """
        功能说明：记录模块C定向重试调用并返回占位结果。
        参数说明：
        - task_id: 任务标识。
        - shot_id: 分镜单元标识。
        - config_path: 配置路径。
        返回值：
        - dict: 占位摘要。
        异常说明：无。
        边界条件：无。
        """
        self.calls.append(
            (
                "retry_module_c_shot",
                {"task_id": task_id, "shot_id": shot_id, "config_path": str(config_path)},
            )
        )
        return {"task_id": task_id, "shot_id": shot_id, "kind": "c-retry"}


def test_build_parser_should_accept_module_c_status_and_retry_commands(tmp_path: Path) -> None:
    """
    功能说明：验证CLI解析器已注册模块C状态查询与shot重试命令。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：命令仅验证参数解析，不触发实际执行。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    parser = cli._build_parser(workspace_root=workspace_root)

    status_args = parser.parse_args(["c-task-status", "--task-id", "task_cli_001"])
    assert status_args.command == "c-task-status"
    assert status_args.task_id == "task_cli_001"
    assert str(status_args.config).endswith("configs/default.json")

    retry_args = parser.parse_args(["c-retry-shot", "--task-id", "task_cli_001", "--shot-id", "shot_007"])
    assert retry_args.command == "c-retry-shot"
    assert retry_args.task_id == "task_cli_001"
    assert retry_args.shot_id == "shot_007"
    assert str(retry_args.config).endswith("configs/default.json")


def test_dispatch_command_should_route_to_module_c_status_and_retry_methods(tmp_path: Path) -> None:
    """
    功能说明：验证CLI分发能正确调用模块C状态查询与shot重试接口。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用FakeRunner隔离真实流水线执行。
    """
    workspace_root = tmp_path / "workspace_dispatch"
    workspace_root.mkdir(parents=True, exist_ok=True)
    fake_runner = _FakeRunner()
    config_path = (workspace_root / "configs" / "wuli_v2.json").resolve()
    app_config = _build_test_config(runs_dir=str(tmp_path / "runs_dispatch"))

    status_namespace = argparse.Namespace(command="c-task-status", task_id="task_dispatch_001")
    status_result = cli._dispatch_command(
        args=status_namespace,
        runner=fake_runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=app_config,
        config_path=config_path,
    )
    assert status_result["kind"] == "c-status"
    assert fake_runner.calls[0][0] == "get_module_c_status_summary"
    assert fake_runner.calls[0][1]["task_id"] == "task_dispatch_001"
    assert fake_runner.calls[0][1]["config_path"] == str(config_path)

    retry_namespace = argparse.Namespace(command="c-retry-shot", task_id="task_dispatch_001", shot_id="shot_003")
    retry_result = cli._dispatch_command(
        args=retry_namespace,
        runner=fake_runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=app_config,
        config_path=config_path,
    )
    assert retry_result["kind"] == "c-retry"
    assert retry_result["shot_id"] == "shot_003"
    assert fake_runner.calls[1][0] == "retry_module_c_shot"
    assert fake_runner.calls[1][1]["task_id"] == "task_dispatch_001"
    assert fake_runner.calls[1][1]["shot_id"] == "shot_003"
    assert fake_runner.calls[1][1]["config_path"] == str(config_path)


def test_build_parser_should_allow_no_subcommand_for_interactive_mode(tmp_path: Path) -> None:
    """
    功能说明：验证解析器允许无子命令输入，以支持交互模式入口。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：无参数时 command 应为空。
    """
    workspace_root = tmp_path / "workspace_interactive"
    workspace_root.mkdir(parents=True, exist_ok=True)
    parser = cli._build_parser(workspace_root=workspace_root)
    args = parser.parse_args([])
    assert getattr(args, "command", None) is None
    assert cli._should_enter_interactive_mode(args=args) is True


def test_build_parser_should_reject_removed_upload_worker_command(tmp_path: Path) -> None:
    """
    功能说明：验证 upload-worker 旧命令入口已下线。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：
    - SystemExit: argparse 解析未知子命令时会退出。
    边界条件：仅验证命令入口是否移除，不涉及执行流程。
    """
    workspace_root = tmp_path / "workspace_removed_upload_worker"
    workspace_root.mkdir(parents=True, exist_ok=True)
    parser = cli._build_parser(workspace_root=workspace_root)
    with pytest.raises(SystemExit):
        parser.parse_args(["upload-worker", "--once"])


def test_main_without_subcommand_should_enter_interactive_mode(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 mvpl 无子命令时会进入交互模式。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：交互模式在本测试中使用假实现防止阻塞。
    """
    called: list[Path] = []

    def _fake_run_interactive_cli(*, workspace_root: Path, default_config_path: Path, execute_request):  # noqa: ANN001
        _ = execute_request
        called.append(workspace_root)
        assert str(default_config_path).endswith("configs/default.json")
        return 0

    monkeypatch.setattr(cli, "run_interactive_cli", _fake_run_interactive_cli)
    monkeypatch.setattr(cli.sys, "argv", ["mvpl"])

    # 直接调用 main，不应抛异常。
    cli.main()
    assert called, "预期应进入交互模式"


def test_apply_user_custom_prompt_override_should_patch_runtime_config(tmp_path: Path) -> None:
    """
    功能说明：验证命令请求携带 user_custom_prompt 覆盖值时会注入运行时配置。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖值为空字符串时同样视为有效覆盖。
    """
    config = _build_test_config(runs_dir=str(tmp_path / "runs_cli_override"))
    request = cli.CommandRequest(
        command="run",
        task_id="task_override_001",
        config_path=(tmp_path / "config.json"),
        user_custom_prompt_override="赛博朋克女孩",
    )
    patched = cli._apply_user_custom_prompt_override(config=config, request=request)
    assert patched.module_b.llm.user_custom_prompt == "赛博朋克女孩"

    empty_override_request = cli.CommandRequest(
        command="run",
        task_id="task_override_001",
        config_path=(tmp_path / "config.json"),
        user_custom_prompt_override="",
    )
    empty_patched = cli._apply_user_custom_prompt_override(config=config, request=empty_override_request)
    assert empty_patched.module_b.llm.user_custom_prompt == ""


def _build_test_config(runs_dir: str) -> AppConfig:
    """
    功能说明：构造CLI分发测试用最小配置对象。
    参数说明：
    - runs_dir: 运行目录路径。
    返回值：
    - AppConfig: 配置对象。
    异常说明：无。
    边界条件：module_a.funasr_language 必填，固定为 auto。
    """
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=runs_dir, default_audio_path="resources/demo.mp3"),
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
        module_a=ModuleAConfig(funasr_language="auto"),
    )
