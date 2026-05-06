"""
文件用途：模块 D 渲染后端导出入口。
核心流程：统一暴露 ComfyUI 视频渲染函数，供 executor 与跨模块调度调用。
输入输出：无输入，输出函数符号。
依赖说明：依赖同目录下 comfyui_renderer。
维护说明：模块 D 仅保留 ComfyUI 视频渲染入口，供执行器与调度器直接复用。
"""

# 项目内模块：导出模块 D 的 ComfyUI 渲染函数。
from music_video_pipeline.modules.module_d.backends.comfyui_renderer import (
    prewarm_comfyui_runtime,
    render_one_unit_comfyui,
)

__all__ = [
    "prewarm_comfyui_runtime",
    "render_one_unit_comfyui",
]
