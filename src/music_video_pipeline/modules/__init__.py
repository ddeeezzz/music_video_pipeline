"""
文件用途：聚合导出 A/B/C/D 模块执行函数。
核心流程：统一提供模块调用入口映射。
输入输出：无输入，输出模块函数符号。
依赖说明：依赖项目内 modules 子模块。
维护说明：新增模块时需同步更新导出列表与 pipeline 映射。
"""

__all__ = ["run_module_a_v2", "run_module_b", "run_module_c", "run_module_d"]


def __getattr__(name: str):
    """
    功能说明：按需导出各模块运行入口，避免包导入阶段触发不必要的重型依赖。
    参数说明：
    - name: 属性名。
    返回值：
    - object: 对应导出对象。
    异常说明：
    - AttributeError: 属性不存在时抛出。
    边界条件：仅支持 __all__ 中声明的导出符号。
    """
    if name == "run_module_a_v2":
        from music_video_pipeline.modules.module_a_v2 import run_module_a_v2

        return run_module_a_v2
    if name == "run_module_b":
        from music_video_pipeline.modules.module_b import run_module_b

        return run_module_b
    if name == "run_module_c":
        from music_video_pipeline.modules.module_c import run_module_c

        return run_module_c
    if name == "run_module_d":
        from music_video_pipeline.modules.module_d import run_module_d

        return run_module_d
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
