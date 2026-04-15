"""
文件用途：聚合模块A V2歌词处理能力。
核心流程：暴露歌词清洗与歌词挂载函数。
输入输出：作为包导出入口，不直接处理业务数据。
依赖说明：依赖同目录子模块。
维护说明：歌词相关逻辑统一收敛到本目录维护。
"""

# 项目内模块：歌词挂载
from music_video_pipeline.modules.module_a_v2.lyrics.attachment import attach_lyrics_to_segments
# 项目内模块：歌词清洗
from music_video_pipeline.modules.module_a_v2.lyrics.cleaner import clean_lyric_units

__all__ = [
    "attach_lyrics_to_segments",
    "clean_lyric_units",
]
