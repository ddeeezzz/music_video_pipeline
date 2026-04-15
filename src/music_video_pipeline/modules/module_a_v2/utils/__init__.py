"""
文件用途：聚合模块A V2通用工具能力。
核心流程：暴露时间处理、媒体探测与别名映射构建函数。
输入输出：作为包导出入口，不直接处理业务数据。
依赖说明：依赖同目录子模块。
维护说明：仅放“低耦合公共工具”，避免承载流程编排。
"""

# 项目内模块：V2别名映射构建
from music_video_pipeline.modules.module_a_v2.utils.alias_map import build_module_a_v2_alias_map
# 项目内模块：V2音频时长探测
from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration
# 项目内模块：V2时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time

__all__ = [
    "build_module_a_v2_alias_map",
    "probe_audio_duration",
    "round_time",
]
