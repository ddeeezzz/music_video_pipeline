#!/usr/bin/env python3
"""
文件用途：手动触发“白名单清单 vs 百度网盘远端目录”核对。
核心流程：读取配置 -> 调用 upload.compare 核对函数 -> 输出报告路径与摘要。
输入输出：输入 task_id/config，输出核对摘要与报告文件路径。
依赖说明：依赖项目内配置加载与上传核对函数。
维护说明：脚本为人工排障入口，worker 会在上传成功后自动执行同一核对逻辑。
"""

# 标准库：命令行参数解析
import argparse
# 标准库：日志
import logging
# 标准库：路径
from pathlib import Path
# 标准库：模块路径注入
import sys


def _bootstrap_python_path(workspace_root: Path) -> None:
    """
    功能说明：将项目 src 路径注入模块搜索路径。
    参数说明：
    - workspace_root: 项目根目录路径。
    返回值：无。
    异常说明：无。
    边界条件：重复注入时跳过。
    """
    src_path = str((workspace_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    """
    功能说明：脚本主入口。
    参数说明：无（由 argparse 解析）。
    返回值：
    - int: 0 成功；非0 失败。
    异常说明：异常外抛由调用方观察。
    边界条件：task_dir 可选，不传时按 config.runs_dir 推导。
    """
    parser = argparse.ArgumentParser(description="核对白名单上传清单与百度网盘远端目录一致性")
    parser.add_argument("--task-id", required=True, help="任务ID，例如 wuli_01")
    parser.add_argument(
        "--config",
        default="/root/data/t1/configs/music_yby/wuli_v2.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--workspace-root",
        default="/root/data/t1",
        help="项目根目录路径",
    )
    parser.add_argument(
        "--task-dir",
        default="",
        help="可选：显式指定任务目录",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    _bootstrap_python_path(workspace_root=workspace_root)

    # 项目内模块：配置加载
    from music_video_pipeline.config import load_config
    # 项目内模块：上传白名单策略常量
    from music_video_pipeline.upload.staging import UPLOAD_SELECTION_PROFILE_WHITELIST_V1
    # 项目内模块：上传核对实现
    from music_video_pipeline.upload.compare import run_whitelist_remote_compare

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("upload_check")

    config_path = Path(args.config).resolve()
    config = load_config(config_path=config_path)
    task_id = str(args.task_id).strip()

    if args.task_dir:
        task_dir = Path(args.task_dir).resolve()
    else:
        runs_dir = Path(config.paths.runs_dir).expanduser()
        if not runs_dir.is_absolute():
            runs_dir = (workspace_root / runs_dir).resolve()
        task_dir = runs_dir / task_id

    report = run_whitelist_remote_compare(
        task_dir=task_dir,
        task_id=task_id,
        upload_config=config.bypy_upload,
        selection_profile=str(config.bypy_upload.selection_profile or UPLOAD_SELECTION_PROFILE_WHITELIST_V1),
        logger=logger,
    )
    summary = dict(report.get("summary", {}))
    print(f"JSON报告: {report.get('report_json_path', '')}")
    print(f"TXT报告: {report.get('report_txt_path', '')}")
    print(
        "核对摘要: "
        f"same={summary.get('same', 0)}, "
        f"different={summary.get('different', 0)}, "
        f"local_only={summary.get('local_only', 0)}, "
        f"remote_only={summary.get('remote_only', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
