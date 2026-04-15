"""
文件用途：聚合跨模块 B/C/D 并行调度入口。
核心流程：从 orchestrator 导出运行函数供 pipeline 调度层调用。
输入输出：输入 RuntimeContext，输出跨模块执行摘要。
依赖说明：依赖同包下 orchestrator 实现。
维护说明：对外仅暴露 run_cross_module_bcd，避免上层感知内部调度细节。
"""

# 项目内模块：跨模块调度入口
from music_video_pipeline.modules.cross_bcd.orchestrator import run_cross_module_bcd

__all__ = ["run_cross_module_bcd"]
