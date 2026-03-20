"""
文件用途：聚合导出 A/B/C/D 模块执行函数。
核心流程：统一提供模块调用入口映射。
输入输出：无输入，输出模块函数符号。
依赖说明：依赖项目内 modules 子模块。
维护说明：新增模块时需同步更新导出列表与 pipeline 映射。
"""

# 项目内模块：导出模块 A 执行函数
from music_video_pipeline.modules.module_a import run_module_a
# 项目内模块：导出模块 B 执行函数
from music_video_pipeline.modules.module_b import run_module_b
# 项目内模块：导出模块 C 执行函数
from music_video_pipeline.modules.module_c import run_module_c
# 项目内模块：导出模块 D 执行函数
from music_video_pipeline.modules.module_d import run_module_d

__all__ = ["run_module_a", "run_module_b", "run_module_c", "run_module_d"]
