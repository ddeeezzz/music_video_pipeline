"""
文件用途：导出模块 D（视频合成）的主入口函数与关键工具符号。
核心流程：统一从 orchestrator 暴露 run_module_d，并对外提供测试复用工具函数。
输入输出：无输入，输出模块 D 运行函数符号。
依赖说明：依赖同包下 orchestrator/unit_models/executor/finalizer 实现。
维护说明：模块 D 拆分后，对外入口保持 run_module_d 不变。
"""

# 项目内模块：导出模块D编排入口
from music_video_pipeline.modules.module_d.orchestrator import run_module_d
# 项目内模块：导出模块D单元模型工具（测试复用）
from music_video_pipeline.modules.module_d.unit_models import _allocate_segment_frames_by_timeline
# 项目内模块：导出模块D执行器工具（测试复用）
from music_video_pipeline.modules.module_d.executor import _build_single_segment_command, _render_single_segment_worker
# 项目内模块：导出模块D终拼工具（测试复用）
from music_video_pipeline.modules.module_d.finalizer import _concat_segment_videos, _probe_ffmpeg_encoder_capabilities, _run_ffmpeg_command

__all__ = [
    "run_module_d",
    "_allocate_segment_frames_by_timeline",
    "_build_single_segment_command",
    "_render_single_segment_worker",
    "_concat_segment_videos",
    "_probe_ffmpeg_encoder_capabilities",
    "_run_ffmpeg_command",
]
