"""
文件用途：提供模块A V2公开入口。
核心流程：导出 run_module_a_v2 供路由层调用。
输入输出：无输入，输出模块A V2公共函数符号。
依赖说明：依赖 module_a_v2.orchestrator 实现。
维护说明：V2阶段仅暴露稳定入口，避免扩张私有兼容面。
"""

# 项目内模块：导出模块A V2执行入口
from music_video_pipeline.modules.module_a_v2.orchestrator import run_module_a_v2

__all__ = ["run_module_a_v2"]
