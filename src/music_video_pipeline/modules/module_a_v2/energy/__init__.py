"""
文件用途：聚合模块A V2能量分析能力。
核心流程：暴露能量特征构建函数。
输入输出：作为包导出入口，不直接处理业务编排。
依赖说明：依赖同目录子模块。
维护说明：能量逻辑集中维护，避免散落到流程文件。
"""

# 项目内模块：能量特征构建
from music_video_pipeline.modules.module_a_v2.energy.features import build_energy_features

__all__ = ["build_energy_features"]
