"""
文件用途：提供 MVP 流水线的命令行接口。
核心流程：解析 run/resume/run-module 子命令并调用 PipelineRunner。
输入输出：输入命令行参数，输出执行日志与摘要。
依赖说明：依赖标准库 argparse/pathlib 与项目内 pipeline/config。
维护说明：新增命令时需同步更新 docs 使用说明。
"""

# 标准库：用于命令行参数解析
import argparse
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于系统退出码
import sys

# 项目内模块：配置加载
from music_video_pipeline.config import AppConfig, load_config
# 项目内模块：日志初始化
from music_video_pipeline.logging_utils import setup_logging
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
    try:
        summary = _dispatch_command(args=args, runner=runner, workspace_root=workspace_root, config=config, config_path=config_path)
        logger.info("任务执行摘要：%s", summary)
    except Exception as error:  # noqa: BLE001
        logger.error("命令执行失败：%s", error)
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

    return parser


def _dispatch_command(
    args: argparse.Namespace,
    runner: PipelineRunner,
    workspace_root: Path,
    config: AppConfig,
    config_path: Path,
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

    raise RuntimeError(f"未知命令: {args.command}")


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
