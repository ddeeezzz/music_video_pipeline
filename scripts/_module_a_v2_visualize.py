"""
文件用途：生成模块A V2内部处理与最终结果的可视化HTML页面。
核心流程：解析任务目录 -> 聚合可视化负载 -> 渲染单文件HTML。
输入输出：输入 task_id/task_dir，输出 module_a_v2_visualization.html。
依赖说明：依赖标准库 argparse/pathlib 与项目内 config/visualization。
维护说明：本脚本为独立工具，不影响 run/resume/run-module 主流程。
"""

# 标准库：用于命令行参数解析
import argparse
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：配置加载（用于解析 runs 根目录）
from music_video_pipeline.config import load_config
# 项目内模块：模块A V2可视化能力
from music_video_pipeline.modules.module_a_v2.visualization import (
    collect_visualization_payload,
    render_visualization_html,
)


def parse_args() -> argparse.Namespace:
    """
    功能说明：解析命令行参数。
    参数说明：无。
    返回值：
    - argparse.Namespace: 参数对象。
    异常说明：参数非法时由 argparse 自动退出。
    边界条件：task_id 与 task_dir 至少提供一个。
    """
    parser = argparse.ArgumentParser(description="模块A V2可视化生成脚本（单HTML+原生JS）")
    parser.add_argument("--task-id", default="", help="任务ID（当未提供 --task-dir 时必填）")
    parser.add_argument("--task-dir", default="", help="任务目录绝对路径（提供时优先于 --task-id）")
    parser.add_argument("--config", default="configs/default.json", help="配置路径（用于解析 runs 根目录）")
    parser.add_argument(
        "--output",
        default="",
        help="输出HTML路径（默认 task_dir/<task_id>_module_a_v2_visualization.html）",
    )
    parser.add_argument(
        "--audio-mode",
        default="copy",
        choices=["copy", "none"],
        help="音频处理模式：copy=复制音频到页面目录并联动播放，none=不绑定音频",
    )
    return parser.parse_args()


def _resolve_task_dir(args: argparse.Namespace, workspace_root: Path) -> Path:
    """
    功能说明：根据参数解析目标任务目录。
    参数说明：
    - args: 命令行参数对象。
    - workspace_root: 项目根目录。
    返回值：
    - Path: 解析后的任务目录路径。
    异常说明：
    - RuntimeError: 参数不足或任务目录不存在时抛错。
    边界条件：--task-dir 优先于 --task-id。
    """
    task_dir_text = str(args.task_dir).strip()
    if task_dir_text:
        task_dir = Path(task_dir_text)
        if not task_dir.is_absolute():
            task_dir = (workspace_root / task_dir).resolve()
        if not task_dir.exists():
            raise RuntimeError(f"任务目录不存在：{task_dir}")
        return task_dir

    task_id = str(args.task_id).strip()
    if not task_id:
        raise RuntimeError("请提供 --task-id 或 --task-dir。")

    config_path = Path(str(args.config))
    if not config_path.is_absolute():
        config_path = (workspace_root / config_path).resolve()
    config = load_config(config_path=config_path)
    runs_dir = Path(config.paths.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (workspace_root / runs_dir).resolve()
    task_dir = (runs_dir / task_id).resolve()
    if not task_dir.exists():
        raise RuntimeError(f"任务目录不存在：{task_dir}")
    return task_dir


def _resolve_output_path(args: argparse.Namespace, task_dir: Path) -> Path:
    """
    功能说明：解析HTML输出路径。
    参数说明：
    - args: 命令行参数对象。
    - task_dir: 任务目录。
    返回值：
    - Path: 目标HTML路径。
    异常说明：无。
    边界条件：默认输出到任务目录根路径。
    """
    output_text = str(args.output).strip()
    if output_text:
        output_path = Path(output_text)
        if not output_path.is_absolute():
            output_path = output_path.resolve()
        return output_path
    # 默认命名采用 task_id 前缀，便于同目录多次产物区分与追溯。
    default_name = f"{task_dir.name}_module_a_v2_visualization.html"
    return (task_dir / default_name).resolve()


def main() -> None:
    """
    功能说明：脚本主入口，执行可视化生成。
    参数说明：无（读取命令行参数）。
    返回值：无。
    异常说明：异常向上抛出，由调用方看到明确失败原因。
    边界条件：仅支持 module_a_v2 产物目录结构。
    """
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    task_dir = _resolve_task_dir(args=args, workspace_root=workspace_root)
    output_path = _resolve_output_path(args=args, task_dir=task_dir)
    payload = collect_visualization_payload(task_dir=task_dir)
    result_path = render_visualization_html(
        payload=payload,
        output_html_path=output_path,
        audio_mode=str(args.audio_mode).strip().lower(),
    )
    print("模块A V2可视化页面生成完成")
    print(f"task_dir: {task_dir}")
    print(f"output_html: {result_path}")


if __name__ == "__main__":
    main()
