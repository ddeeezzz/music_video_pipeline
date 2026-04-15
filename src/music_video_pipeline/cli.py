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
# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于系统退出码
import sys
# 标准库：用于类型提示
from typing import Any

# 项目内模块：配置加载
from music_video_pipeline.config import AppConfig, load_config
# 项目内模块：日志初始化
from music_video_pipeline.logging_utils import setup_logging
# 项目内模块：任务监督服务
from music_video_pipeline.monitoring import TaskMonitorService
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


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

    config_path = _resolve_path(workspace_root=workspace_root, input_path=Path(args.config))
    config = load_config(config_path=config_path)
    logger = setup_logging(level=config.logging.level)

    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)
    command_failed = False
    try:
        summary = _dispatch_command(
            args=args,
            runner=runner,
            workspace_root=workspace_root,
            config=config,
            config_path=config_path,
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
    subparsers = parser.add_subparsers(dest="command", required=True)

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


def _dispatch_command(
    args: argparse.Namespace,
    runner: PipelineRunner,
    workspace_root: Path,
    config: AppConfig,
    config_path: Path,
    logger: Any | None = None,
) -> dict:
    """
    功能说明：根据子命令调用对应执行逻辑。
    参数说明：
    - args: 命令行解析结果。
    - runner: 流水线调度器。
    - workspace_root: 项目根目录。
    - config: 应用配置对象。
    - config_path: 已解析的配置路径。
    返回值：
    - dict: 执行摘要。
    异常说明：参数缺失或执行失败时抛 RuntimeError。
    边界条件：run 命令若未给音频路径，使用配置默认音频。
    """
    if args.command == "run":
        audio_path_text = args.audio_path if args.audio_path else config.paths.default_audio_path
        audio_path = _resolve_path(workspace_root=workspace_root, input_path=Path(audio_path_text))
        return runner.run(
            task_id=args.task_id,
            audio_path=audio_path,
            config_path=config_path,
            force_module=args.force_module,
        )

    if args.command == "resume":
        return runner.resume(
            task_id=args.task_id,
            config_path=config_path,
            force_module=args.force_module,
        )

    if args.command == "run-module":
        audio_path = _resolve_path(workspace_root=workspace_root, input_path=Path(args.audio_path)) if args.audio_path else None
        return runner.run_single_module(
            task_id=args.task_id,
            module_name=args.module,
            audio_path=audio_path,
            force=args.force,
            config_path=config_path,
        )

    if args.command == "c-task-status":
        return runner.get_module_c_status_summary(
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "c-retry-shot":
        return runner.retry_module_c_shot(
            task_id=args.task_id,
            shot_id=args.shot_id,
            config_path=config_path,
        )

    if args.command == "b-task-status":
        return runner.get_module_b_status_summary(
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "b-retry-segment":
        return runner.retry_module_b_segment(
            task_id=args.task_id,
            segment_id=args.segment_id,
            config_path=config_path,
        )

    if args.command == "d-task-status":
        return runner.get_module_d_status_summary(
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "d-retry-shot":
        return runner.retry_module_d_shot(
            task_id=args.task_id,
            shot_id=args.shot_id,
            config_path=config_path,
        )

    if args.command == "bcd-task-status":
        return runner.get_bcd_status_summary(
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command == "bcd-retry-segment":
        return runner.retry_bcd_segment(
            task_id=args.task_id,
            segment_id=args.segment_id,
            config_path=config_path,
        )

    if args.command == "monitor":
        dispatch_logger = logger if logger is not None else logging.getLogger("music_video_pipeline.cli")
        return _run_task_monitor_command(
            args=args,
            runner=runner,
            logger=dispatch_logger,
        )

    raise RuntimeError(f"未知命令: {args.command}")


def _run_task_monitor_command(
    args: argparse.Namespace,
    runner: PipelineRunner,
    logger: Any,
) -> dict:
    """
    功能说明：手动启动任务监督服务，并生成任务目录入口页。
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
    if not task_id:
        raise RuntimeError("monitor 命令缺少 task_id。")
    if not runner.state_store.get_task(task_id=task_id):
        raise RuntimeError(f"任务不存在，无法启动监督服务：task_id={task_id}")

    monitor_service = TaskMonitorService(
        state_store=runner.state_store,
        task_id=task_id,
        logger=logger,
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
