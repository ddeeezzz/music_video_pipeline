"""
文件用途：导出模块A V2时间轴重构子模块公共入口。
核心流程：聚合窗口构造、角色分类、并段与A1求解能力。
输入输出：供 content_roles 编排层导入使用。
依赖说明：依赖 timeline 子模块。
维护说明：新增时间轴能力时在此补充导出。
"""

# 项目内模块：窗口构造
from .window_builder import (
    build_windows_from_sentences,
    inject_boundary_points_into_windows,
    resplit_long_lyric_windows,
)
# 项目内模块：窗口四分类
from .role_classifier import classify_window_roles
# 项目内模块：窗口并段与小节估计
from .role_merger import estimate_bar_length_seconds, merge_windows_by_rules
# 项目内模块：A1与最终S段求解
from .big_timestamp_resolver import resolve_big_timestamps_and_segments

__all__ = [
    "build_windows_from_sentences",
    "inject_boundary_points_into_windows",
    "resplit_long_lyric_windows",
    "classify_window_roles",
    "estimate_bar_length_seconds",
    "merge_windows_by_rules",
    "resolve_big_timestamps_and_segments",
]
