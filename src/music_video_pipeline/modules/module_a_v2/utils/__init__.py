"""
文件用途：聚合模块A V2通用工具能力。
核心流程：按需惰性暴露时间处理、媒体探测与别名映射构建函数。
输入输出：作为包导出入口，不直接处理业务数据。
依赖说明：依赖同目录子模块。
维护说明：仅放低耦合公共工具，避免包初始化时加载额外依赖。
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "build_module_a_v2_alias_map",
    "probe_audio_duration",
    "round_time",
]


def __getattr__(name: str) -> Any:
    """
    功能说明：按需惰性导出 utils 公共函数。
    参数说明：
    - name: 访问的属性名。
    返回值：
    - Any: 对应公共函数。
    异常说明：
    - AttributeError: 请求未知符号时抛出。
    边界条件：仅支持 __all__ 中列出的工具函数。
    """
    if name == "build_module_a_v2_alias_map":
        from music_video_pipeline.modules.module_a_v2.utils.alias_map import build_module_a_v2_alias_map

        return build_module_a_v2_alias_map
    if name == "probe_audio_duration":
        from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration

        return probe_audio_duration
    if name == "round_time":
        from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time

        return round_time
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
