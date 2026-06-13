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
# 标准库：用于JSON序列化
import json
# 标准库：用于 dataclass 局部替换
from dataclasses import replace
# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于子进程执行
import subprocess
# 标准库：用于系统退出码
import sys
# 标准库：用于时间戳
import time
# 标准库：用于轻量命名空间对象
from types import SimpleNamespace
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行期噪声过滤器（需最早安装，避免导入期噪声刷屏）
from music_video_pipeline.log_filters import install_runtime_noise_filters

install_runtime_noise_filters()

# 项目内模块：命令服务层
from music_video_pipeline.command_service import CommandRequest, MvplCommandService
# 项目内模块：配置加载
from music_video_pipeline.config import AppConfig, load_config
# 项目内模块：常量定义
from music_video_pipeline.constants import TASK_WEB_ENTRY_PAGE_FILE_NAME
# 项目内模块：日志配置
from music_video_pipeline.logging_utils import setup_logging
# 项目内模块：任务音频路径回映射
from music_video_pipeline.task_audio_path import remap_windows_absolute_path, resolve_task_audio_path, resolve_workspace_path
# 任务监督服务类采用延迟导入，避免交互菜单启动时加载重依赖。
TaskMonitorService: Any | None = None

# 常量：Web 触发的模块 B 重跑子进程状态文件名。
ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME = "active_module_b_rerun_process.json"


def _build_active_module_b_rerun_process_path(*, runs_dir: Path, task_id: str) -> Path:
    """
    功能说明：构建模块 B 活跃重跑子进程状态文件路径。
    参数说明：
    - runs_dir: runs 根目录。
    - task_id: 任务唯一标识。
    返回值：
    - Path: 状态文件绝对路径。
    异常说明：无。
    边界条件：文件固定放在 runs/<task_id>/ 目录下。
    """
    return (runs_dir / str(task_id).strip() / ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME).resolve()


def _persist_active_module_b_rerun_process(
    *,
    runs_dir: Path,
    task_id: str,
    mode: str,
    role_name: str,
    pid: int,
    shot_id: str = "",
) -> Path:
    """
    功能说明：持久化 Web 触发的模块 B 活跃重跑子进程元信息。
    参数说明：
    - runs_dir: runs 根目录。
    - task_id: 任务唯一标识。
    - mode: 重跑模式（role/segment）。
    - role_name: 角色名。
    - pid: 子进程 PID。
    - shot_id: 可选 shot_id。
    返回值：
    - Path: 写入后的状态文件路径。
    异常说明：目录不可写时透传异常。
    边界条件：每次新进程启动都会覆盖旧文件。
    """
    process_file_path = _build_active_module_b_rerun_process_path(runs_dir=runs_dir, task_id=task_id)
    process_file_path.parent.mkdir(parents=True, exist_ok=True)
    submitted_at_ms = int(time.time() * 1000)
    payload = {
        "task_id": str(task_id).strip(),
        "mode": str(mode).strip(),
        "role_name": str(role_name).strip(),
        "shot_id": str(shot_id).strip(),
        "pid": int(pid),
        "submitted_at_ms": submitted_at_ms,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
    }
    process_file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return process_file_path


def _clear_active_module_b_rerun_process(*, runs_dir: Path, task_id: str, pid: int) -> None:
    """
    功能说明：按 PID 清理模块 B 活跃重跑子进程状态文件。
    参数说明：
    - runs_dir: runs 根目录。
    - task_id: 任务唯一标识。
    - pid: 当前结束的子进程 PID。
    返回值：无。
    异常说明：无；清理失败时静默忽略。
    边界条件：若文件中的 PID 已被新进程覆盖，则不删除。
    """
    process_file_path = _build_active_module_b_rerun_process_path(runs_dir=runs_dir, task_id=task_id)
    if not process_file_path.exists():
        return
    try:
        payload = json.loads(process_file_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        payload = {}
    if int(payload.get("pid", 0) or 0) != int(pid):
        return
    try:
        process_file_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        return


def main() -> None:
    """
    功能说明：CLI 主入口，解析参数并分发子命令。
    参数说明：无（读取命令行参数）。
    返回值：无。
    异常说明：发生异常时输出中文错误并以非零码退出。
    边界条件：默认配置文件为 t1/configs/music_yby/default.json。
    """
    workspace_root = Path(__file__).resolve().parents[2]
    default_config_path = workspace_root / "configs" / "music_yby" / "default.json"
    parser = _build_parser(workspace_root=workspace_root, default_config_path=default_config_path)
    args = parser.parse_args()

    if _should_enter_interactive_mode(args=args):
        from music_video_pipeline.interactive_cli import run_interactive_cli

        interactive_exit_code = run_interactive_cli(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            execute_request=lambda request: _execute_request_with_loaded_runtime(
                workspace_root=workspace_root,
                request=request,
            ),
        )
        if interactive_exit_code != 0:
            sys.exit(interactive_exit_code)
        return

    command_failed = False
    try:
        config_path = _resolve_request_config_path(
            args=args,
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
        request = _build_command_request(
            args=args,
            config_path=config_path,
        )
        summary = _execute_request_with_loaded_runtime(
            workspace_root=workspace_root,
            request=request,
        )
        logging.getLogger("SYS").info("任务执行摘要：%s", summary)
    except KeyboardInterrupt:
        command_failed = True
        logging.getLogger("SYS").warning("命令已被用户中断。")
    except Exception as error:  # noqa: BLE001
        command_failed = True
        logging.getLogger("SYS").error("命令执行失败：%s", error)
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


def _build_parser(workspace_root: Path, default_config_path: Path | None = None) -> argparse.ArgumentParser:
    """
    功能说明：构建并返回命令行参数解析器。
    参数说明：
    - workspace_root: 项目根目录路径。
    返回值：
    - argparse.ArgumentParser: 配置完成的解析器。
    异常说明：无。
    边界条件：默认配置固定为 configs/music_yby/default.json。
    """
    resolved_default_config_path = (
        default_config_path
        if default_config_path is not None
        else workspace_root / "configs" / "music_yby" / "default.json"
    )
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
    run_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")
    run_parser.add_argument("--force-module", choices=["A", "B", "C", "D"], help="从指定模块强制重跑")
    run_parser.add_argument("--lyrics-only", action="store_true", help="仅更新歌词→算法层，跳过信号处理")

    resume_parser = subparsers.add_parser("resume", help="从断点恢复运行")
    resume_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    resume_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")
    resume_parser.add_argument("--force-module", choices=["A", "B", "C", "D"], help="从指定模块强制恢复")

    module_parser = subparsers.add_parser("run-module", help="执行单模块调试")
    module_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    module_parser.add_argument("--module", required=True, choices=["A", "B", "C", "D"], help="模块名")
    module_parser.add_argument("--audio-path", required=False, help="输入音频路径（首次任务初始化时需要）")
    module_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")
    module_parser.add_argument("--force", action="store_true", help="重置当前模块及其下游后再执行")

    c_status_parser = subparsers.add_parser("c-task-status", help="查看模块C单元状态摘要")
    c_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    c_status_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    c_retry_parser = subparsers.add_parser("c-retry-shot", help="按shot_id重试模块C单元，并在成功后重建视频")
    c_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    c_retry_parser.add_argument("--shot-id", required=True, help="模块C单元标识（等价shot_id）")
    c_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    c_retry_frame_parser = subparsers.add_parser("c-retry-frame", help="按shot_id+frame_type重试模块C单帧，并在成功后重建视频")
    c_retry_frame_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    c_retry_frame_parser.add_argument("--shot-id", required=True, help="模块C单元标识（等价shot_id）")
    c_retry_frame_parser.add_argument("--frame-type", required=True, choices=["start", "end"], help="目标帧类型")
    c_retry_frame_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    b_status_parser = subparsers.add_parser("b-task-status", help="查看模块B单元状态摘要")
    b_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_status_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    b_retry_parser = subparsers.add_parser("b-retry-segment", help="按segment_id重试模块B单元（不自动重建C/D）")
    b_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_retry_parser.add_argument("--segment-id", required=True, help="模块B单元标识（等价segment_id）")
    b_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    b_role_retry_parser = subparsers.add_parser("b-retry-role", help="按 role 起点重试模块B（不自动重建C/D）")
    b_role_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_role_retry_parser.add_argument(
        "--role-name",
        required=True,
        choices=["role1", "role2", "role3", "role4"],
        help="模块B 角色名",
    )
    b_role_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    b_role_shot_retry_parser = subparsers.add_parser(
        "b-retry-role-shot",
        help="按 role 内 shot 重试模块B（不自动重建C/D）",
    )
    b_role_shot_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    b_role_shot_retry_parser.add_argument(
        "--role-name",
        required=True,
        choices=["role3", "role4"],
        help="模块B 角色名（仅支持 shot 级角色）",
    )
    b_role_shot_retry_parser.add_argument("--shot-id", required=True, help="目标 shot_id")
    b_role_shot_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    d_status_parser = subparsers.add_parser("d-task-status", help="查看模块D单元状态摘要")
    d_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    d_status_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    d_retry_parser = subparsers.add_parser("d-retry-shot", help="按shot_id重试模块D单元，并在D内重建最终视频")
    d_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    d_retry_parser.add_argument("--shot-id", required=True, help="模块D单元标识（等价shot_id）")
    d_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    bcd_status_parser = subparsers.add_parser("bcd-task-status", help="查看跨模块B/C/D链路状态摘要")
    bcd_status_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    bcd_status_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    bcd_retry_parser = subparsers.add_parser("bcd-retry-segment", help="按segment_id重试跨模块B/C/D链路")
    bcd_retry_parser.add_argument("--task-id", required=True, help="任务唯一标识")
    bcd_retry_parser.add_argument("--segment-id", required=True, help="目标链路segment_id")
    bcd_retry_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")

    web_parser = subparsers.add_parser(
        "web",
        help="启动任务 Web 服务（可省略 --task-id 直接进入任务列表）",
    )
    web_parser.add_argument("--task-id", help="任务唯一标识（可选；传入时直接打开对应任务）")
    web_parser.add_argument("--runs-dir", default="runs", help="任务产物根目录（默认: runs）")
    web_parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址（默认: 127.0.0.1）")
    web_parser.add_argument("--port", type=int, default=45705, help="Web 服务起始监听端口（默认: 45705；占用时自动顺延）")

    template_render_parser = subparsers.add_parser(
        "template-render",
        aliases=["tr"],
        help="生成正式模板请求并渲染模板片段",
    )
    template_render_parser.add_argument("--config", default=str(resolved_default_config_path), help="配置文件路径")
    template_render_parser.add_argument(
        "--template",
        default="center",
        choices=["center", "grid", "scroll"],
        help="模板标识",
    )
    template_render_parser.add_argument("--run-name", default="", help="输出 runs 子目录名；留空时按模板自动决定")
    template_render_parser.add_argument(
        "--symbol-src",
        default="/fixtures/center-symbol.svg",
        help="CenterTemplate 符号资源路径，可写 public 目录路径或绝对文件路径",
    )
    template_render_parser.add_argument(
        "--background-kind",
        default="solid",
        choices=["none", "solid", "image", "video"],
        help="背景类型，默认 solid",
    )
    template_render_parser.add_argument("--background-color", default="#FFFFFF", help="纯色背景颜色")
    template_render_parser.add_argument("--background-src", default="", help="图片或视频背景资源路径")
    template_render_parser.add_argument(
        "--symbol-src-list",
        nargs="*",
        default=[],
        help="GridTemplate 使用的三个符号资源路径；留空时使用内置 fixtures",
    )
    template_render_parser.add_argument(
        "--grid-direction",
        default="left_to_right",
        choices=["left_to_right", "right_to_left"],
        help="GridTemplate 槽位进入方向",
    )
    template_render_parser.add_argument(
        "--scroll-direction",
        default="right_to_left",
        choices=["left_to_right", "right_to_left"],
        help="ScrollTemplate 横向滚动方向",
    )

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
            lyrics_only=bool(getattr(args, "lyrics_only", False)),
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

    if args.command == "c-retry-frame":
        return CommandRequest(
            command="c-retry-frame",
            task_id=args.task_id,
            shot_id=args.shot_id,
            frame_type=args.frame_type,
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

    if args.command == "b-retry-role":
        return CommandRequest(
            command="b-retry-role",
            task_id=args.task_id,
            role_name=args.role_name,
            config_path=config_path,
        )

    if args.command == "b-retry-role-shot":
        return CommandRequest(
            command="b-retry-role-shot",
            task_id=args.task_id,
            role_name=args.role_name,
            shot_id=args.shot_id,
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

    if args.command == "web":
        return CommandRequest(
            command="web",
            runs_dir=Path(args.runs_dir),
            monitor_host=args.host,
            monitor_port=args.port,
            task_id=args.task_id,
            config_path=config_path,
        )

    if args.command in {"template-render", "tr"}:
        return CommandRequest(
            command="template-render",
            config_path=config_path,
            run_name=args.run_name,
            template_name=args.template,
            symbol_src=args.symbol_src,
            symbol_src_list=args.symbol_src_list,
            background_kind=args.background_kind,
            background_color=args.background_color,
            background_src=args.background_src,
            grid_direction=args.grid_direction,
            scroll_direction=args.scroll_direction,
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
    边界条件：web 命令会走专用 handler。
    """
    service_logger = logger if logger is not None else logging.getLogger("SYS")
    if str(request.command).strip() == "web":
        return _run_task_monitor_command(
            args=argparse.Namespace(task_id=request.task_id),
            runner=runner,
            logger=service_logger,
        )
    service = MvplCommandService(
        runner=runner,
        workspace_root=workspace_root,
        config=config,
        logger=service_logger,
        monitor_handler=_monitor_handler_for_service,
    )
    return service.execute(request)


def _monitor_handler_for_service(task_id: str | None, runner: Any, logger: Any) -> dict:
    """
    功能说明：为命令服务层提供 web 命令桥接。
    参数说明：
    - task_id: 任务标识；为空时启动通用任务列表页。
    - runner: 流水线调度器。
    - logger: 日志对象。
    返回值：
    - dict: web 执行摘要。
    异常说明：透传 web 执行异常。
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
    config, logger, runner = _build_runtime_for_request(
        workspace_root=workspace_root,
        request=request,
    )
    return _execute_request(
        request=request,
        runner=runner,
        workspace_root=workspace_root,
        config=config,
        logger=logger,
    )


def _build_runtime_for_request(*, workspace_root: Path, request: CommandRequest) -> tuple[Any, Any, Any]:
    """
    功能说明：按命令类型加载运行时对象，供 CLI 与交互式入口复用。
    参数说明：
    - workspace_root: 项目根目录。
    - request: 结构化命令请求。
    返回值：
    - tuple[Any, Any, Any]: (config_like, logger, runner_like)。
    异常说明：配置加载或运行时构建失败时透传异常。
    边界条件：web 命令走轻量运行时，其余命令走完整 PipelineRunner。
    """
    if str(request.command).strip() == "web":
        logger = setup_logging(level="INFO")
        runner = _build_web_command_runtime(
            workspace_root=workspace_root,
            request=request,
            logger=logger,
        )
        config = getattr(runner, "config", SimpleNamespace())
    else:
        config = load_config(config_path=request.config_path)
        config = _apply_storyboard_template_override(config=config, request=request)
        logger = setup_logging(level=config.logging.level)
        runner = _build_pipeline_runner(
            workspace_root=workspace_root,
            config=config,
            logger=logger,
        )
    return config, logger, runner


def _apply_storyboard_template_override(*, config: AppConfig, request: CommandRequest) -> AppConfig:
    """
    功能说明：将命令请求中的 storyboard_template_file 覆盖值注入到运行时配置。
    参数说明：
    - config: 已加载配置对象。
    - request: 命令请求对象。
    返回值：
    - AppConfig: 注入后的配置对象；若无覆盖值则返回原对象。
    异常说明：无。
    边界条件：覆盖值为空或空白时视为无效覆盖。
    """
    override_path = str(request.storyboard_template_file_override or "").strip()
    if not override_path:
        return config
    patched_module_b = replace(config.module_b, storyboard_template_file=override_path)
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
    功能说明：手动启动任务 Web 服务（兼容旧签名）。
    参数说明：
    - args: 命令行参数对象。
    - runner: 流水线调度器（提供状态库与 runs_dir）。
    - logger: 日志对象。
    返回值：
    - dict: 监督服务摘要信息。
    异常说明：
    - RuntimeError: 显式 task_id 不存在或服务启动失败时抛出。
    边界条件：未传 task_id 时进入任务列表，不预选具体任务。
    """
    task_id = _resolve_monitor_target_task_id(args=args, runner=runner, logger=logger)
    return _run_task_monitor_command_by_task(
        task_id=task_id,
        runner=runner,
        logger=logger,
    )


def _resolve_monitor_target_task_id(
    args: argparse.Namespace,
    runner: Any,
    logger: Any,
) -> str:
    """
    功能说明：解析 web 命令实际应预选的任务ID。
    参数说明：
    - args: 命令行参数对象。
    - runner: 流水线调度器（提供状态库访问）。
    - logger: 日志对象。
    返回值：
    - str: 最终用于启动服务的任务ID；空字符串表示进入任务列表。
    异常说明：无。
    边界条件：未传 task_id 时不自动选择最新任务。
    """
    raw_task_id = getattr(args, "task_id", "")
    _ = (runner, logger)
    return str(raw_task_id).strip() if raw_task_id is not None else ""


def _run_task_monitor_command_by_task(
    task_id: str,
    runner: Any,
    logger: Any,
) -> dict:
    """
    功能说明：按 task_id 手动启动任务 Web 服务；缺省时进入任务列表页。
    参数说明：
    - task_id: 任务标识；空字符串表示不预选任务。
    - runner: 流水线调度器（提供状态库与 runs_dir）。
    - logger: 日志对象。
    返回值：
    - dict: 监督服务摘要信息。
    异常说明：
    - RuntimeError: 显式 task_id 不存在或服务启动失败时抛出。
    边界条件：仅在传入 task_id 时写任务目录入口页。
    """
    normalized_task_id = str(task_id).strip()
    if normalized_task_id and not runner.state_store.get_task(task_id=normalized_task_id):
        raise RuntimeError(f"任务不存在，无法启动监督服务：task_id={normalized_task_id}")
    monitor_host, monitor_port = _resolve_monitor_host_port(runner=runner)

    monitor_service_class = _get_task_monitor_service_class()
    monitor_service = monitor_service_class(
        state_store=runner.state_store,
        task_id=normalized_task_id,
        logger=logger,
        rerun_handler=getattr(runner, "_rerun_task_from_module_a_for_monitor", None),
        module_b_role_rerun_handler=getattr(runner, "_rerun_module_b_role_for_monitor", None),
        module_b_role_segment_rerun_handler=getattr(runner, "_rerun_module_b_role_segment_for_monitor", None),
        module_c_shot_rerun_handler=getattr(runner, "_rerun_module_c_shot_for_monitor", None),
        module_c_frame_rerun_handler=getattr(runner, "_rerun_module_c_frame_for_monitor", None),
        lyrics_only_rerun_handler=getattr(runner, "_rerun_task_from_module_a_lyrics_only_for_monitor", None),
        app_config=getattr(runner, "config", None),
        host=monitor_host,
        port=monitor_port,
        auto_stop_on_terminal=False,
    )
    monitor_service.start()
    launch_page_path: Path | None = None
    if normalized_task_id:
        launch_page_path = _write_task_web_entry_page(
            task_dir=runner.runs_dir / normalized_task_id,
            task_id=normalized_task_id,
            monitor_url=monitor_service.monitor_url,
        )
        logger.info("任务监督服务已开启（手动模式），task_id=%s，地址=%s", normalized_task_id, monitor_service.monitor_url)
        logger.info("任务 Web 入口页已写入：%s", launch_page_path)
        logger.info("请在浏览器打开任务目录下页面：%s", launch_page_path)
    else:
        logger.info("任务 Web 服务已开启（手动模式，无默认任务），地址=%s", monitor_service.monitor_url)
        logger.info("未指定 task_id，当前打开的是任务列表页。")
    logger.info("停止监督服务请按 Ctrl+C")

    interrupted_by_user = False
    try:
        while monitor_service.is_running:
            stopped = monitor_service.wait_until_stopped(timeout_seconds=1.0)
            if stopped:
                break
    except KeyboardInterrupt:
        interrupted_by_user = True
        logger.info(
            "收到中断信号，正在停止任务监督服务，task_id=%s",
            normalized_task_id if normalized_task_id else "<none>",
        )
    finally:
        monitor_service.stop()

    return {
        "task_id": normalized_task_id,
        "monitor_url": monitor_service.monitor_url,
        "launch_page_path": str(launch_page_path) if launch_page_path is not None else "",
        "interrupted_by_user": interrupted_by_user,
    }


def _resolve_monitor_host_port(*, runner: Any) -> tuple[str, int]:
    """
    功能说明：解析任务监督服务监听 host/起始 port（优先读取运行配置，缺失时回退默认值）。
    参数说明：
    - runner: 流水线调度器对象。
    返回值：
    - tuple[str, int]: (host, start_port)。
    异常说明：无。
    边界条件：非法端口值回退到 45705，实际绑定时若被占用会自动顺延。
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


def _build_web_command_runtime(*, workspace_root: Path, request: CommandRequest, logger: Any) -> Any:
    """
    功能说明：为 web 命令构建轻量运行时，避免启动时导入完整流水线。
    参数说明：
    - workspace_root: 项目根目录。
    - request: web 命令请求对象。
    - logger: 日志对象。
    返回值：
    - Any: 具备 state_store / runs_dir / config / rerun_handler 的轻量对象。
    异常说明：状态库初始化或回调构建失败时透传异常。
    边界条件：仅满足任务 Web 服务所需能力。
    """
    from music_video_pipeline.state_store import StateStore

    raw_runs_dir = request.runs_dir if request.runs_dir is not None else Path("runs")
    runs_dir = _resolve_path(workspace_root=workspace_root, input_path=Path(raw_runs_dir))
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    config = load_config(config_path=request.config_path)
    monitor_host = str(request.monitor_host or "127.0.0.1").strip() or "127.0.0.1"
    monitor_port = int(request.monitor_port if request.monitor_port is not None else 45705)
    rerun_handler = _build_web_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    module_b_role_rerun_handler = _build_web_module_b_role_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    module_b_role_segment_rerun_handler = _build_web_module_b_role_segment_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    module_c_shot_rerun_handler = _build_web_module_c_shot_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    module_c_frame_rerun_handler = _build_web_module_c_frame_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    lyrics_only_rerun_handler = _build_web_lyrics_only_rerun_handler(
        workspace_root=workspace_root,
        state_store=state_store,
        logger=logger,
    )
    return SimpleNamespace(
        config=replace(
            config,
            monitoring=replace(
                config.monitoring,
                host=monitor_host,
                port=monitor_port,
            ),
        ),
        runs_dir=runs_dir,
        state_store=state_store,
        _rerun_task_from_module_a_for_monitor=rerun_handler,
        _rerun_module_b_role_for_monitor=module_b_role_rerun_handler,
        _rerun_module_b_role_segment_for_monitor=module_b_role_segment_rerun_handler,
        _rerun_module_c_shot_for_monitor=module_c_shot_rerun_handler,
        _rerun_module_c_frame_for_monitor=module_c_frame_rerun_handler,
        _rerun_task_from_module_a_lyrics_only_for_monitor=lyrics_only_rerun_handler,
    )


def _build_web_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """
    功能说明：为 web 服务中的“生成”按钮构建后台重跑回调。
    参数说明：
    - workspace_root: 项目根目录。
    - state_store: 状态库对象。
    - logger: 日志对象。
    返回值：
    - Any: 可被监督服务调用的 task_id -> dict 回调。
    异常说明：任务缺少必要字段或子进程失败时抛出 RuntimeError。
    边界条件：实际生成通过独立 CLI 子进程执行，避免阻塞当前服务启动导入链。
    """

    def _rerun_task_from_module_a_for_web(task_id: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行强制重跑：task_id={task_id}")
        audio_path_text = str(task_record.get("audio_path", "")).strip()
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not audio_path_text or not config_path_text:
            raise RuntimeError(f"任务缺少 audio_path 或 config_path，无法执行强制重跑：task_id={task_id}")
        resolved_audio_path = resolve_task_audio_path(
            raw_audio_path=audio_path_text,
            config_path=config_path_text,
            workspace_roots=[workspace_root],
        )
        if str(resolved_audio_path) != audio_path_text:
            state_store.init_task(task_id=task_id, audio_path=str(resolved_audio_path), config_path=config_path_text)
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "run",
            "--task-id",
            task_id,
            "--audio-path",
            str(resolved_audio_path),
            "--config",
            config_path_text,
            "--force-module",
            "A",
        ]
        logger.info("Web 服务触发任务强制重跑，task_id=%s，from_module=A，command=%s", task_id, command)
        completed = subprocess.run(
            command,
            cwd=str(workspace_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"任务强制重跑子进程执行失败，task_id={task_id}，exit_code={completed.returncode}")
        return {
            "task_id": task_id,
            "exit_code": completed.returncode,
            "force_module": "A",
        }

    return _rerun_task_from_module_a_for_web


def _build_web_lyrics_only_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """为 web 服务中的"仅更新歌词"按钮构建后台重跑回调（轻量模式，跳过信号处理）。"""

    def _rerun_lyrics_only_for_web(task_id: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行轻量重跑：task_id={task_id}")
        audio_path_text = str(task_record.get("audio_path", "")).strip()
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not audio_path_text or not config_path_text:
            raise RuntimeError(f"任务缺少 audio_path 或 config_path，无法执行轻量重跑：task_id={task_id}")
        resolved_audio_path = resolve_task_audio_path(
            raw_audio_path=audio_path_text,
            config_path=config_path_text,
            workspace_roots=[workspace_root],
        )
        if str(resolved_audio_path) != audio_path_text:
            state_store.init_task(task_id=task_id, audio_path=str(resolved_audio_path), config_path=config_path_text)
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "run",
            "--task-id",
            task_id,
            "--audio-path",
            str(resolved_audio_path),
            "--config",
            config_path_text,
            "--lyrics-only",
        ]
        logger.info("Web 服务触发轻量重跑（歌词→算法层，跳过信号），task_id=%s，command=%s", task_id, command)
        completed = subprocess.run(
            command,
            cwd=str(workspace_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            error_output = str(completed.stderr or "").strip() or str(completed.stdout or "").strip()
            raise RuntimeError(f"轻量重跑子进程执行失败，task_id={task_id}，exit_code={completed.returncode}，原因={error_output[:500]}")
        return {"task_id": task_id, "exit_code": completed.returncode}

    return _rerun_lyrics_only_for_web


def _build_web_module_b_role_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """
    功能说明：为 web 服务中的模块 B role 重跑构建后台回调。
    参数说明：
    - workspace_root: 项目根目录。
    - state_store: 状态库对象。
    - logger: 日志对象。
    返回值：
    - Any: 可被监督服务调用的 (task_id, role_name) -> dict 回调。
    异常说明：任务缺少必要字段或子进程失败时抛出 RuntimeError。
    边界条件：实际执行通过独立 CLI 子进程完成。
    """

    def _rerun_module_b_role_for_web(task_id: str, role_name: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行模块B role 重跑：task_id={task_id}")
        runs_dir = state_store.db_path.parent.resolve()
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not config_path_text:
            raise RuntimeError(f"任务缺少 config_path，无法执行模块B role 重跑：task_id={task_id}")
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "b-retry-role",
            "--task-id",
            task_id,
            "--role-name",
            str(role_name).strip(),
            "--config",
            config_path_text,
        ]
        logger.info("Web 服务触发模块B role 重跑，task_id=%s，role_name=%s，command=%s", task_id, role_name, command)
        process = subprocess.Popen(
            command,
            cwd=str(workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _persist_active_module_b_rerun_process(
            runs_dir=runs_dir,
            task_id=task_id,
            mode="role",
            role_name=str(role_name).strip(),
            pid=int(process.pid),
        )
        stdout_text, stderr_text = process.communicate()
        returncode = process.returncode
        _clear_active_module_b_rerun_process(runs_dir=runs_dir, task_id=task_id, pid=int(process.pid))
        if returncode != 0:
            detail_parts: list[str] = [
                f"模块B role 重跑子进程执行失败，task_id={task_id}，role_name={role_name}，exit_code={returncode}"
            ]
            if stderr_text.strip():
                detail_parts.append(f"stderr:\n{stderr_text.strip()}")
            if stdout_text.strip():
                detail_parts.append(f"stdout:\n{stdout_text.strip()}")
            raise RuntimeError("\n".join(detail_parts))
        return {
            "task_id": task_id,
            "role_name": str(role_name).strip(),
            "exit_code": returncode,
            "pid": int(process.pid),
        }

    return _rerun_module_b_role_for_web


def _build_web_module_b_role_segment_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """
    功能说明：为 web 服务中的模块 B role 内 shot 重跑构建后台回调。
    参数说明：
    - workspace_root: 项目根目录。
    - state_store: 状态库对象。
    - logger: 日志对象。
    返回值：
    - Any: 可被监督服务调用的 (task_id, role_name, shot_id) -> dict 回调。
    异常说明：任务缺少必要字段或子进程失败时抛出 RuntimeError。
    边界条件：实际执行通过独立 CLI 子进程完成。
    """

    def _rerun_module_b_role_segment_for_web(task_id: str, role_name: str, shot_id: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行模块B shot 重跑：task_id={task_id}")
        runs_dir = state_store.db_path.parent.resolve()
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not config_path_text:
            raise RuntimeError(f"任务缺少 config_path，无法执行模块B shot 重跑：task_id={task_id}")
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "b-retry-role-shot",
            "--task-id",
            task_id,
            "--role-name",
            str(role_name).strip(),
            "--shot-id",
            str(shot_id).strip(),
            "--config",
            config_path_text,
        ]
        logger.info(
            "Web 服务触发模块B shot 重跑，task_id=%s，role_name=%s，shot_id=%s，command=%s",
            task_id,
            role_name,
            shot_id,
            command,
        )
        process = subprocess.Popen(
            command,
            cwd=str(workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _persist_active_module_b_rerun_process(
            runs_dir=runs_dir,
            task_id=task_id,
            mode="segment",
            role_name=str(role_name).strip(),
            shot_id=str(shot_id).strip(),
            pid=int(process.pid),
        )
        stdout_text, stderr_text = process.communicate()
        returncode = process.returncode
        _clear_active_module_b_rerun_process(runs_dir=runs_dir, task_id=task_id, pid=int(process.pid))
        if returncode != 0:
            detail_parts: list[str] = [
                "模块B shot 重跑子进程执行失败，"
                f"task_id={task_id}，role_name={role_name}，shot_id={shot_id}，exit_code={returncode}"
            ]
            if stderr_text.strip():
                detail_parts.append(f"stderr:\n{stderr_text.strip()}")
            if stdout_text.strip():
                detail_parts.append(f"stdout:\n{stdout_text.strip()}")
            raise RuntimeError("\n".join(detail_parts))
        return {
            "task_id": task_id,
            "role_name": str(role_name).strip(),
            "shot_id": str(shot_id).strip(),
            "exit_code": returncode,
            "pid": int(process.pid),
        }

    return _rerun_module_b_role_segment_for_web


def _build_web_module_c_shot_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """
    功能说明：为 web 服务中的模块 C shot 重跑构建后台回调。
    参数说明：
    - workspace_root: 项目根目录。
    - state_store: 状态库对象。
    - logger: 日志对象。
    返回值：
    - Any: 可被监督服务调用的 (task_id, shot_id) -> dict 回调。
    异常说明：任务缺少必要字段或子进程失败时抛出 RuntimeError。
    边界条件：实际执行通过独立 CLI 子进程完成。
    """

    def _rerun_module_c_shot_for_web(task_id: str, shot_id: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行模块C shot 重跑：task_id={task_id}")
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not config_path_text:
            raise RuntimeError(f"任务缺少 config_path，无法执行模块C shot 重跑：task_id={task_id}")
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "c-retry-shot",
            "--task-id",
            task_id,
            "--shot-id",
            str(shot_id).strip(),
            "--config",
            config_path_text,
        ]
        logger.info("Web 服务触发模块C shot 重跑，task_id=%s，shot_id=%s，command=%s", task_id, shot_id, command)
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"模块C shot 重跑子进程执行超时，task_id={task_id}，shot_id={shot_id}，timeout_seconds=600"
            ) from error
        if completed.returncode != 0:
            error_excerpt = (
                str(completed.stderr or "").strip()
                or str(completed.stdout or "").strip()
            )
            if error_excerpt:
                error_excerpt = error_excerpt.splitlines()[-1].strip()
            raise RuntimeError(
                "模块C shot 重跑子进程执行失败，"
                f"task_id={task_id}，shot_id={shot_id}，exit_code={completed.returncode}"
                + (f"，原因={error_excerpt}" if error_excerpt else "")
            )
        return {
            "task_id": task_id,
            "shot_id": str(shot_id).strip(),
            "exit_code": completed.returncode,
        }

    return _rerun_module_c_shot_for_web


def _build_web_module_c_frame_rerun_handler(*, workspace_root: Path, state_store: Any, logger: Any) -> Any:
    """
    功能说明：为 web 服务中的模块 C 单帧重跑构建后台回调。
    参数说明：
    - workspace_root: 项目根目录。
    - state_store: 状态库对象。
    - logger: 日志对象。
    返回值：
    - Any: 可被监督服务调用的 (task_id, shot_id, frame_type) -> dict 回调。
    异常说明：任务缺少必要字段或子进程失败时抛出 RuntimeError。
    边界条件：实际执行通过独立 CLI 子进程完成。
    """

    def _rerun_module_c_frame_for_web(task_id: str, shot_id: str, frame_type: str) -> dict:
        task_record = state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法执行模块C单帧重跑：task_id={task_id}")
        config_path_text = str(task_record.get("config_path", "")).strip()
        if not config_path_text:
            raise RuntimeError(f"任务缺少 config_path，无法执行模块C单帧重跑：task_id={task_id}")
        command = [
            sys.executable,
            "-m",
            "music_video_pipeline.cli",
            "c-retry-frame",
            "--task-id",
            task_id,
            "--shot-id",
            str(shot_id).strip(),
            "--frame-type",
            str(frame_type).strip(),
            "--config",
            config_path_text,
        ]
        logger.info(
            "Web 服务触发模块C单帧重跑，task_id=%s，shot_id=%s，frame_type=%s，command=%s",
            task_id,
            shot_id,
            frame_type,
            command,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"模块C单帧重跑子进程执行超时，task_id={task_id}，shot_id={shot_id}，frame_type={frame_type}，timeout_seconds=600"
            ) from error
        if completed.returncode != 0:
            error_output = (
                str(completed.stderr or "").strip()
                or str(completed.stdout or "").strip()
            )
            error_excerpt = ""
            if error_output:
                error_lines = [line.strip() for line in error_output.splitlines() if line.strip()]
                error_excerpt = "；".join(error_lines[-5:]) or error_output[:500]
            logger.error(
                "模块C单帧重跑子进程执行失败，task_id=%s，shot_id=%s，frame_type=%s，exit_code=%s，完整错误=%s",
                task_id,
                shot_id,
                frame_type,
                completed.returncode,
                error_output[:2000],
            )
            raise RuntimeError(
                "模块C单帧重跑子进程执行失败，"
                f"task_id={task_id}，shot_id={shot_id}，frame_type={frame_type}，exit_code={completed.returncode}"
                + (f"，原因={error_excerpt}" if error_excerpt else "")
            )
        logger.info(
            "Web 服务触发模块C单帧重跑执行成功，task_id=%s，shot_id=%s，frame_type=%s，exit_code=%s",
            task_id,
            shot_id,
            frame_type,
            completed.returncode,
        )
        return {
            "task_id": task_id,
            "shot_id": str(shot_id).strip(),
            "frame_type": str(frame_type).strip(),
            "exit_code": completed.returncode,
        }

    return _rerun_module_c_frame_for_web


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


def _write_task_web_entry_page(task_dir: Path, task_id: str, monitor_url: str) -> Path:
    """
    功能说明：在任务根目录写入任务 Web 入口页，打开后自动跳转到本地监督服务URL。
    参数说明：
    - task_dir: 任务目录路径（runs/<task_id>）。
    - task_id: 任务唯一标识。
    - monitor_url: 本次监督服务URL。
    返回值：
    - Path: 写入后的入口页路径。
    异常说明：无。
    边界条件：每次 web 启动都会覆盖写入，确保URL端口与当前服务一致。
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    launch_page_path = task_dir / TASK_WEB_ENTRY_PAGE_FILE_NAME
    raw_monitor_url = str(monitor_url)
    safe_task_id = escape(str(task_id), quote=True)
    safe_monitor_url = escape(raw_monitor_url, quote=True)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>任务 Web 入口 - {safe_task_id}</title>
  <meta http-equiv="refresh" content="0;url={safe_monitor_url}">
</head>
<body>
  <p>任务 Web 页面正在跳转中：<a href="{safe_monitor_url}">{safe_monitor_url}</a></p>
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
    边界条件：不会主动检查路径是否存在；若输入为 Windows 盘符绝对路径
              （如 ``M:\\foo\\bar\\configs\\...``），会自动回映射到当前工作区。
              若输入为 Linux 风格绝对路径（如 ``/root/data/...``），在 Windows 上
              会尝试按已知项目目录标记（configs/resources/runs）回映射到工作区。
    """
    path_text = str(input_path)
    remapped = remap_windows_absolute_path(workspace_root=workspace_root, path_text=path_text)
    if remapped is not None:
        return remapped
    if input_path.is_absolute():
        return input_path.resolve()
    return resolve_workspace_path(workspace_root=workspace_root, path_text=path_text)


def _resolve_request_config_path(*, args: argparse.Namespace, workspace_root: Path, default_config_path: Path) -> Path:
    """
    功能说明：解析命令请求实际使用的配置路径。
    参数说明：
    - args: 命令行解析结果。
    - workspace_root: 项目根目录。
    - default_config_path: 默认配置文件路径。
    返回值：
    - Path: 已解析的配置文件绝对路径。
    异常说明：无。
    边界条件：web 命令不依赖配置文件，返回默认配置路径占位即可。
    """
    if str(getattr(args, "command", "")).strip() == "web":
        return default_config_path.resolve()
    config_text = str(getattr(args, "config", "")).strip()
    if not config_text:
        return default_config_path.resolve()
    return _resolve_path(workspace_root=workspace_root, input_path=Path(config_text))


if __name__ == "__main__":
    main()
