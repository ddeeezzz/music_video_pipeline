"""
文件用途：导出模块 B（视觉脚本）的主入口函数与兼容增强工具。
核心流程：统一从 orchestrator 暴露 run_module_b，并导出分镜增强函数供测试复用。
输入输出：无输入，输出模块 B 运行函数符号。
依赖说明：依赖同包下 orchestrator/output_builder 实现。
维护说明：模块 B 拆分后，对外入口保持 run_module_b 不变。
"""

# 项目内模块：导出模块B编排入口
from music_video_pipeline.modules.module_b.orchestrator import run_module_b
# 项目内模块：导出模块B分镜增强函数（测试兼容）
from music_video_pipeline.modules.module_b.output_builder import _enrich_shots_with_segment_meta

__all__ = ["run_module_b", "_enrich_shots_with_segment_meta"]
