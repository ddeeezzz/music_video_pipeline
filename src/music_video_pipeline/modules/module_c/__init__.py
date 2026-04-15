"""
文件用途：导出模块 C（图像生成）的主入口函数。
核心流程：统一从 orchestrator 暴露 run_module_c，供 pipeline 调用。
输入输出：无输入，输出模块 C 运行函数符号。
依赖说明：依赖同包下 orchestrator 实现。
维护说明：模块 C 拆分后，对外入口保持 run_module_c 不变。
"""

# 项目内模块：导出模块C编排入口
from music_video_pipeline.modules.module_c.orchestrator import run_module_c

__all__ = ["run_module_c"]
