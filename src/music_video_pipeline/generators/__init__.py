"""
文件用途：导出生成器工厂与抽象定义。
核心流程：聚合分镜与关键帧生成器接口。
输入输出：无输入，输出可导入符号。
依赖说明：依赖项目内 generators 子模块。
维护说明：新增生成器类型时需在此处补充导出。
"""

# 项目内模块：导出关键帧生成器工厂
from music_video_pipeline.generators.frame_generator import FrameGenerator, build_frame_generator
# 项目内模块：导出分镜生成器工厂
from music_video_pipeline.generators.script_generator import ScriptGenerator, build_script_generator

__all__ = ["FrameGenerator", "ScriptGenerator", "build_frame_generator", "build_script_generator"]
