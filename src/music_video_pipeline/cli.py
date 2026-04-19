"""
文件用途：提供 MVP 流水线的命令行接口。
核心流程：解析 CLI 子命令，调用 PipelineRunner 或手动启动任务监督服务。
输入输出：输入命令行参数，输出执行日志与摘要。
依赖说明：依赖标准库 argparse/pathlib 与项目内 pipeline/config。
维护说明：新增命令时需同步更新 docs 使用说明。
"""

# 标准库：用于命令行参数解析
import argparse
# 标准库：用于HTML转义
from html import escape
# 标准库：用于 dataclass 局部替换
from dataclasses import replace
# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于系统退出码
import sys
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行期噪声过滤器（需最早安装，避免导入期噪声刷屏）
from music_video_pipeline.log_filters import install_runtime_noise_filters

install_runtime_noise_filters()

# 项目内模块：命令服务层
from music_video_pipeline.command_service import CommandRequest, MvplCommandService
# 项目内模块：配置加载
from music_video_pipeline.config import AppConfig, load_config
# 项目内模块：交互式 CLI
from music_video_pipeline.interactive_cli import run_interactive_cli
# 项目内模块：日志初始化
from music_video_pipeline.logging_utils import setup_logging

# 任务监督服务类采用延迟导入，避免交互菜单启动时加载重依赖。
TaskMonitorService: Any | None = None


def main() -> None:
    """
    功能说明：CLI 主入口，解析参数并分发子命令。
    参数说明：无（读取命令行参数）。
    返回值：无。
    异常说明：发生异常时输出中文错误并以非零码退出。
    边界条件：默认配置文件为 t1/configs/default.json。
    """
    workspace_root = Path(__file__).resolve().parents[2]
    parser = _build_parser(workspace_root=workspace_root)
    args = parser.parse_args()

    if _should_enter_interactive_mode(args=args):
        interactive_exit_code = run_interactive_cli(
            workspace_root=workspace_root,
            default_config_path=(workspace_root / "configs" / "default.json"),
            execute_request=lambda request: _execute_request_with_loaded_runtime(
                workspace_root=workspace_root,
                request=request,
            ),
        )
        if interactive_exit_code != 0:
            sys.exit(interactive_exit_code)
        return

    config_path = _resolve_path(workspace_root=workspace_root, input_path=Path(args.config))
    config = load_config(config_path=config_path)
    logger = setup_logging(level=config.logging.level)

    runner = _build_pipeline_runner(
        workspace_root=workspace_root,
        config=config,
        logger=logger,
    )
    command_failed = False
    try:
        request = _build_command_request(
            args=args,
            config_path=config_path,
        )
        config = _apply_user_custom_prompt_override(config=config, request=request)
        summary = _execute_request(
            request=request,
            runner=runner,
            workspace_root=workspace_root,
            config=config,
            logger=logger,
        )
        logger.info("任务执行摘要：%s", summary)
    except KeyboardInterrupt:
        command_failed = True
        logger.warning("命令已被用户中断。")
    except Exception as error:  # noqa: BLE001
        command_failed = True
        logger.error("命令执行失败：%s", error)
    if command_failed:
        sys.exit(1)


def _should_enter_interactive_mode(args: argparse.Namespace) -> bool:
    """
    功能说明：判断当前是否应进入交互模式。
    参数说明：
    - args: 解析后的命令行参数。
    返回值：
    - bool: True 表示进入交互模式。
    异常说明：无。
    边界条件：无子命令或显式 --interactive 均进入交互。
    """
    interactive_enabled = bool(getattr(args, "interactive", False))
    command = str(getattr(args, "command", "") or "").strip()
    if interactive_enabled:
        return True
    return not command


def _build_parser(workspace_root: Path) -> argparse.ArgumentParser:
    """
    功能说明：构建并返回命令行参数解析器。
    参数说明：
    - workspace_root: 项目根目录路径。
    返回值：
    - argparse.ArgumentParser: 配置完成的解析器。
    异常说明：无。
    边界条件：子命令必须二选一或三选一，不允许缺失。
    """
    default_config_path = workspace_root / "configs" / "default.json"
    parser = argparse.ArgumentParser(description="MVP 音画同步流水线 CLI")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式（无子命令时默认进入）",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="执行全链路运行")
    run_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    run_parser.add_argument("--audio-path", required=False, help="输入音频路径（默认读取配置）")
    run_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")
    run_parser.add_argument("--force-module", choices=["A", "B", "C", "D"], help="从指定模块强制重跑")

    resume_parser = subparsers.add_parser("resume", help="从断点恢复运行")
    resume_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    resume_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")
    resume_parser.add_argument("--force-module", choices=["A", "B", "C", "D"], help="从指定模块强制恢复")

    module_parser = subparsers.add_parser("run-module", help="执行单模块调试")
    module_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    module_parser.add_argument("--module", required=True, choices=["A", "B", "C", "D"], help="模块名")
    module_parser.add_argument("--audio-path", required=False, help="输入音频路径（首次任务初始化时需要）")
    module_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")
    module_parser.add_argument("--force", action="store_true", help="重置当前模块及其下游后再执行")

    c_status_parser = subparsers.add_parser("c-task-status", help="查看模块C单元状态摘要")
    c_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    c_status_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    c_retry_parser = subparsers.add_parser("c-retry-shot", help="按shot_id重试模块C单元，并在成功后重建视频")
    c_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    c_retry_parser.add_argument("--shot-id", required=True, help="模块C单元标识（等价shot_id）")
    c_retry_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    b_status_parser = subparsers.add_parser("b-task-status", help="查看模块B单元状态摘要")
    b_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_status_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    b_retry_parser = subparsers.add_parser("b-retry-segment", help="按segment_id重试模块B单元（不自动重建C/D）")
    b_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_retry_parser.add_argument("--segment-id", required=True, help="模块B单元标识（等价segment_id）")
    b_retry_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    d_status_parser = subparsers.add_parser("d-task-status", help="查看模块D单元状态摘要")
    d_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    d_status_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    d_retry_parser = subparsers.add_parser("d-retry-shot", help="按shot_id重试模块D单元，并在D内重建最终视频")
    d_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    d_retry_parser.add_argument("--shot-id", required=True, help="模块D单元标识（等价shot_id）")
    d_retry_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    bcd_status_parser = subparsers.add_parser("bcd-task-status", help="查看跨模块B/C/D链路状态摘要")
    bcd_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    bcd_status_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    bcd_retry_parser = subparsers.add_parser("bcd-retry-segment", help="按segment_id重试跨模块B/C/D链路")
    bcd_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    bcd_retry_parser.add_argument("--segment-id", required=True, help="目标链路segment_id")
    bcd_retry_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    monitor_parser = subparsers.add_parser("monitor", help="手动启动任务监督服务（按需）")
    monitor_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    monitor_parser.add_argument("--config", default=str(default_config_path), help="配置文件路径")

    return parser


def _build_command_request(
    args: argparse.Namespace,
    config_path: Path,
) -> CommandRequest:
    """
    功能说明：根据子命令构建结构化命令请求。
    参数说明：
    - args: 命令行解析结果。
    - config_path: 已解析的配置路径。
    返回值：
    - CommandRequest: 结构化命令请求。
    异常说明：参数缺失或执行失败时抛 RuntimeError。
    边界条件：run 命令若未给音频路径，使用配置默认音频。
    """
    if args.command == "run":
        audio_path = Path(args.audio_path) if args.audio_path else None
        return CommandRequest(
            command="run",
            task_id=args.task_id,
            audio_path=audio_path,
            config_path=config_path,
            force_module=args.force_module,
        )

    if args.command == "resume":
        return CommandRequest(
            command="resume",
            task_id=args.task_id,
            config_path=config_path,
            force_module=args.force_module,
        )

    if args.command == "run-module":
        audio_path = Path(args.audio_path) if args.audio_path else None
        return CommandRequest(
            command="run-module",
            task_id=args.task_id,
            module=args.module,
            audio_path=audio_path,
            force=args.force,
            config_path=config_path,
        )

    if args.command == "c-task-status":
        return CommandRequest(
            command="c-task-status",
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "c-retry-shot":
        return CommandRequest(
            command="c-retry-shot",
            task_id=args.task_id,
            shot_id=args.shot_id,
            config_path=config_path,
        )

    if args.command == "b-task-status":
        return CommandRequest(
            command="b-task-status",
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "b-retry-segment":
        return CommandRequest(
            command="b-retry-segment",
            task_id=args.task_id,
            segment_id=args.segment_id,
            config_path=config_path,
        )

    if args.command == "d-task-status":
        return CommandRequest(
            command="d-task-status",
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "d-retry-shot":
        return CommandRequest(
            command="d-retry-shot",
            task_id=args.task_id,
            shot_id=args.shot_id,
            config_path=config_path,
        )

    if args.command == "bcd-task-status":
        return CommandRequest(
            command="bcd-task-status",
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "bcd-retry-segment":
        return CommandRequest(
            command="bcd-retry-segment",
            task_id=args.task_id,
            segment_id=args.segment_id,
            config_path=config_path,
        )

    if args.command == "monitor":
        return CommandRequest(
            command="monitor",
            task_id=args.task_id,
            config_path=config_path,
        )

    raise RuntimeError(f"未知命令: {args.command}")


def _execute_request(
    *,
    request: CommandRequest,
    runner: Any,
    workspace_root: Path,
    config: AppConfig,
    logger: Any | None = None,
) -> dict:
    """
    功能说明：执行命令请求并返回摘要。
    参数说明：
    - request: 命令请求对象。
    - runner: 流水线调度器（提供状态库与 runs_dir）。
    - workspace_root: 项目根目录。
    - config: 应用配置。
    - logger: 日志对象。
    返回值：
    - dict: 执行摘要。
    异常说明：下游执行失败时透传异常。
    边界条件：monitor 命令会走专用 handler。
    """
    service_logger = logger if logger is not None else logging.getLogger("SYS")
    service = MvplCommandService(
        runner=runner,
        workspace_root=workspace_root,
        config=config,
        logger=service_logger,
        monitor_handler=_monitor_handler_for_service,
    )
    return service.execute(request)


def _monitor_handler_for_service(task_id: str, runner: Any, logger: Any) -> dict:
    """
    功能说明：为命令服务层提供 monitor 兼容桥接。
    参数说明：
    - task_id: 任务标识。
    - runner: 流水线调度器。
    - logger: 日志对象。
    返回值：
    - dict: monitor 执行摘要。
    异常说明：透传 monitor 执行异常。
    边界条件：通过旧签名函数调用，兼容 monkeypatch 钩子。
    """
    return _run_task_monitor_command(
        args=argparse.Namespace(task_id=task_id),
        runner=runner,
        logger=logger,
    )


def _execute_request_with_loaded_runtime(*, workspace_root: Path, request: CommandRequest) -> dict:
    """
    功能说明：按请求中的配置路径初始化运行时并执行命令。
    参数说明：
    - workspace_root: 项目根目录。
    - request: 命令请求对象。
    返回值：
    - dict: 执行摘要。
    异常说明：配置加载或执行失败时抛出异常。
    边界条件：每次执行按请求配置独立初始化 logger/runner。
    """
    config = load_config(config_path=request.config_path)
    config = _apply_user_custom_prompt_override(config=config, request=request)
    logger = setup_logging(level=config.logging.level)
    runner = _build_pipeline_runner(
        workspace_root=workspace_root,
        config=config,
        logger=logger,
    )
    return _execute_request(
        request=request,
        runner=runner,
        workspace_root=workspace_root,
        config=config,
        logger=logger,
    )


def _apply_user_custom_prompt_override(*, config: AppConfig, request: CommandRequest) -> AppConfig:
    """
    功能说明：将命令请求中的 user_custom_prompt 覆盖值注入到运行时配置。
    参数说明：
    - config: 已加载配置对象。
    - request: 命令请求对象。
    返回值：
    - AppConfig: 注入后的配置对象（若无覆盖值则返回原对象）。
    异常说明：无。
    边界条件：覆盖值为 None 时不改写配置；空字符串属于有效覆盖值。
    """
    if request.user_custom_prompt_override is None:
        return config
    patched_llm = replace(config.module_b.llm, user_custom_prompt=request.user_custom_prompt_override)
    patched_module_b = replace(config.module_b, llm=patched_llm)
    return replace(config, module_b=patched_module_b)


def _dispatch_command(
    args: argparse.Namespace,
    runner: Any,
    workspace_root: Path,
    config: AppConfig,
    config_path: Path,
    logger: Any | None = None,
) -> dict:
    """
    功能说明：兼容旧测试与调用方的分发函数。
    参数说明：
    - args: 命令行解析结果。
    - runner: 流水线调度器。
    - workspace_root: 项目根目录。
    - config: 应用配置对象。
    - config_path: 配置路径。
    - logger: 日志对象。
    返回值：
    - dict: 执行摘要。
    异常说明：参数缺失或执行失败时抛 RuntimeError。
    边界条件：内部委托给 command service。
    """
    request = _build_command_request(args=args, config_path=config_path)
    return _execute_request(
        request=request,
        runner=runner,
        workspace_root=workspace_root,
        config=config,
        logger=logger,
    )


def _run_task_monitor_command(
    args: argparse.Namespace,
    runner: Any,
    logger: Any,
) -> dict:
    """
    功能说明：手动启动任务监督服务（兼容旧签名）。
    参数说明：
    - args: 命令行参数对象。
    - runner: 流水线调度器（提供状态库与 runs_dir）。
    - logger: 日志对象。
    返回值：
    - dict: 监督服务摘要信息。
    异常说明：
    - RuntimeError: task_id 不存在或服务启动失败时抛出。
    边界条件：服务默认持续运行，直到用户中断（Ctrl+C）或显式停止。
    """
    task_id = str(getattr(args, "task_id", "")).strip()
    return _run_task_monitor_command_by_task(
        task_id=task_id,
        runner=runner,
        logger=logger,
    )


def _run_task_monitor_command_by_task(
    task_id: str,
    runner: Any,
    logger: Any,
) -> dict:
    """
    功能说明：按 task_id 手动启动任务监督服务，并生成任务目录入口页。
    参数说明：
    - task_id: 任务标识。
    - runner: 流水线调度器（提供状态库与 runs_dir）。
    - logger: 日志对象。
    返回值：
    - dict: 监督服务摘要信息。
    异常说明：
    - RuntimeError: task_id 不存在或服务启动失败时抛出。
    边界条件：服务默认持续运行，直到用户中断（Ctrl+C）或显式停止。
    """
    if not task_id:
        raise RuntimeError("monitor 命令缺少 task_id。")
    if not runner.state_store.get_task(task_id=task_id):
        raise RuntimeError(f"任务不存在，无法启动监督服务：task_id={task_id}")
    monitor_host, monitor_port = _resolve_monitor_host_port(runner=runner)

    monitor_service_class = _get_task_monitor_service_class()
    monitor_service = monitor_service_class(
        state_store=runner.state_store,
        task_id=task_id,
        logger=logger,
        host=monitor_host,
        port=monitor_port,
        auto_stop_on_terminal=False,
    )
    monitor_service.start()
    launch_page_path = _write_task_monitor_launch_page(
        task_dir=runner.runs_dir / task_id,
        task_id=task_id,
        monitor_url=monitor_service.monitor_url,
    )
    logger.info("任务监督服务已开启（手动模式），task_id=%s，地址=%s", task_id, monitor_service.monitor_url)
    logger.info("监督入口页已写入：%s", launch_page_path)
    logger.info("请在浏览器打开任务目录下页面：%s", launch_page_path)
    logger.info("停止监督服务请按 Ctrl+C")

    interrupted_by_user = False
    try:
        while monitor_service.is_running:
            stopped = monitor_service.wait_until_stopped(timeout_seconds=1.0)
            if stopped:
                break
    except KeyboardInterrupt:
        interrupted_by_user = True
        logger.info("收到中断信号，正在停止任务监督服务，task_id=%s", task_id)
    finally:
        monitor_service.stop()

    return {
        "task_id": task_id,
        "monitor_url": monitor_service.monitor_url,
        "launch_page_path": str(launch_page_path),
        "interrupted_by_user": interrupted_by_user,
    }


def _resolve_monitor_host_port(*, runner: Any) -> tuple[str, int]:
    """
    功能说明：解析任务监督服务监听 host/port（优先读取运行配置，缺失时回退默认值）。
    参数说明：
    - runner: 流水线调度器对象。
    返回值：
    - tuple[str, int]: (host, port)。
    异常说明：无。
    边界条件：非法端口值回退到 45705。
    """
    default_host = "127.0.0.1"
    default_port = 45705
    config = getattr(runner, "config", None)
    monitoring = getattr(config, "monitoring", None) if config is not None else None
    host = str(getattr(monitoring, "host", default_host) or default_host).strip() or default_host
    try:
        port = int(getattr(monitoring, "port", default_port))
    except (TypeError, ValueError):
        port = default_port
    return host, port


def _build_pipeline_runner(*, workspace_root: Path, config: AppConfig, logger: Any) -> Any:
    """
    功能说明：延迟导入并构建 PipelineRunner，降低交互菜单首屏启动耗时。
    参数说明：
    - workspace_root: 项目根目录。
    - config: 运行时配置。
    - logger: 日志对象。
    返回值：
    - Any: PipelineRunner 实例。
    异常说明：透传导入或构造异常。
    边界条件：仅在实际执行命令时触发导入。
    """
    from music_video_pipeline.pipeline import PipelineRunner

    return PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)


def _get_task_monitor_service_class() -> Any:
    """
    功能说明：按需加载 TaskMonitorService，兼容测试中的 monkeypatch。
    参数说明：无。
    返回值：
    - Any: TaskMonitorService 类对象。
    异常说明：导入失败时透传异常。
    边界条件：若模块级 TaskMonitorService 已被替换（测试桩），直接复用。
    """
    global TaskMonitorService
    if TaskMonitorService is None:
        from music_video_pipeline.monitoring import TaskMonitorService as _TaskMonitorService

        TaskMonitorService = _TaskMonitorService
    return TaskMonitorService


def _write_task_monitor_launch_page(task_dir: Path, task_id: str, monitor_url: str) -> Path:
    """
    功能说明：在任务根目录写入监督入口页，打开后自动跳转到本地监督服务URL。
    参数说明：
    - task_dir: 任务目录路径（runs/<task_id>）。
    - task_id: 任务唯一标识。
    - monitor_url: 本次监督服务URL。
    返回值：
    - Path: 写入后的入口页路径。
    异常说明：无。
    边界条件：每次 monitor 启动都会覆盖写入，确保URL端口与当前服务一致。
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    launch_page_path = task_dir / "task_monitor.html"
    raw_monitor_url = str(monitor_url)
    safe_task_id = escape(str(task_id), quote=True)
    safe_monitor_url = escape(raw_monitor_url, quote=True)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>任务监督入口 - {safe_task_id}</title>
  <meta http-equiv="refresh" content="0;url={safe_monitor_url}">
</head>
<body>
  <p>任务监督服务正在跳转中：<a href="{safe_monitor_url}">{safe_monitor_url}</a></p>
  <script>
    (function () {{
      var targetUrl = {raw_monitor_url!r};
      if (window.location.href !== targetUrl) {{
        window.location.replace(targetUrl);
      }}
    }})();
  </script>
</body>
</html>
"""
    launch_page_path.write_text(html_text, encoding="utf-8")
    return launch_page_path


def _resolve_path(workspace_root: Path, input_path: Path) -> Path:
    """
    功能说明：将相对路径解析为绝对路径。
    参数说明：
    - workspace_root: 项目根目录。
    - input_path: 输入路径（可相对可绝对）。
    返回值：
    - Path: 解析后的绝对路径。
    异常说明：无。
    边界条件：不会主动检查路径是否存在。
    """
    if input_path.is_absolute():
        return input_path.resolve()
    return (workspace_root / input_path).resolve()
