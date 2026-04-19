"""
文件用途：模块 D 渲染后端导出入口。
核心流程：统一暴露 AnimateDiff 渲染函数，供 executor 分发调用。
输入输出：无输入，输出函数符号。
依赖说明：依赖同目录下 animatediff_renderer。
维护说明：新增后端时可在此集中导出。
"""

from music_video_pipeline.modules.module_d.backends.animatediff_renderer import (
    generate_mv_clip,
    prewarm_animatediff_runtime,
    run_one_unit_animatediff_denoise_stage,
    run_one_unit_animatediff_post_stage,
    render_one_unit_animatediff,
)

__all__ = [
    "generate_mv_clip",
    "prewarm_animatediff_runtime",
    "run_one_unit_animatediff_denoise_stage",
    "run_one_unit_animatediff_post_stage",
    "render_one_unit_animatediff",
]
