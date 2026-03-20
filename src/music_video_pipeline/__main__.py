"""
文件用途：支持 `python -m music_video_pipeline` 启动命令行入口。
核心流程：转发至 CLI 主函数。
输入输出：读取命令行参数并执行流水线子命令。
依赖说明：依赖项目内 cli 模块。
维护说明：该文件应保持轻量，仅负责入口转发。
"""

# 项目内模块：提供命令行主入口
from music_video_pipeline.cli import main


def run() -> None:
    """
    功能说明：模块化入口函数，便于测试或外部调用。
    参数说明：无。
    返回值：无。
    异常说明：由 CLI 主函数统一处理退出码。
    边界条件：无。
    """
    main()


if __name__ == "__main__":
    run()
