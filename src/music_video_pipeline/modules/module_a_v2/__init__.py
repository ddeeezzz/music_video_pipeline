"""
文件用途：提供模块A V2公开入口。
核心流程：按需惰性导出 run_module_a_v2，避免包初始化时加载整条执行链。
输入输出：无输入，输出模块A V2公共函数符号。
依赖说明：依赖 module_a_v2.orchestrator 实现。
维护说明：仅暴露稳定入口，避免在 __init__ 中引入重依赖副作用。
"""

from __future__ import annotations

from typing import Any


__all__ = ["run_module_a_v2"]


def __getattr__(name: str) -> Any:
    """
    功能说明：按需惰性导出模块A V2主入口。
    参数说明：
    - name: 访问的属性名。
    返回值：
    - Any: 对应导出的公共符号。
    异常说明：
    - AttributeError: 请求未知符号时抛出。
    边界条件：仅支持 run_module_a_v2。
    """
    if name == "run_module_a_v2":
        from music_video_pipeline.modules.module_a_v2.orchestrator import run_module_a_v2

        return run_module_a_v2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
