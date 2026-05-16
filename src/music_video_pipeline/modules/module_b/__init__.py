"""
文件用途：导出模块 B 的对外入口。
核心流程：按需加载编排入口与输出辅助函数。
输入输出：无直接输入，输出模块 B 相关导出符号。
依赖说明：依赖同包内编排与输出模块。
维护说明：对外导出名应与上层调用保持一致。
"""

__all__ = [
    "MultiRoleScriptGenerator",
    "run_module_b",
    "_enrich_shots_with_segment_meta",
]


def __getattr__(name: str):
    """
    功能说明：按需导出模块 B 入口与辅助函数。
    参数说明：
    - name: 属性名。
    返回值：
    - object: 对应导出对象。
    异常说明：
    - AttributeError: 属性不存在时抛出。
    边界条件：仅支持 __all__ 中声明的导出符号。
    """
    if name in {"MultiRoleScriptGenerator", "run_module_b"}:
        from music_video_pipeline.modules.module_b.orchestrator import MultiRoleScriptGenerator, run_module_b

        exports = {
            "MultiRoleScriptGenerator": MultiRoleScriptGenerator,
            "run_module_b": run_module_b,
        }
        return exports[name]
    if name == "_enrich_shots_with_segment_meta":
        from music_video_pipeline.modules.module_b.output_builder import _enrich_shots_with_segment_meta

        return _enrich_shots_with_segment_meta
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")




