"""
文件用途：聚合模块A V2感知后端能力。
核心流程：暴露 Demucs、Allin1、Librosa 三类后端函数。
输入输出：作为包导出入口，不直接处理业务编排。
依赖说明：依赖同目录子模块。
维护说明：后端细节集中维护，编排层保持轻量。
"""

# 项目内模块：Allin1分析
from music_video_pipeline.modules.module_a_v2.backends.allin1 import analyze_with_allin1
# 项目内模块：Demucs分离准备
from music_video_pipeline.modules.module_a_v2.backends.demucs import prepare_stems_with_allin1_demucs
# 项目内模块：Librosa候选提取
from music_video_pipeline.modules.module_a_v2.backends.librosa import extract_acoustic_candidates_with_librosa

__all__ = [
    "analyze_with_allin1",
    "prepare_stems_with_allin1_demucs",
    "extract_acoustic_candidates_with_librosa",
]
