"""
文件用途：导出模块B v2 多角色编排入口。
核心流程：聚合模板解析、4角色调度与最终分镜输出构建能力。
输入输出：无直接输入，输出可供上层导入的生成器与执行函数。
依赖说明：依赖项目内 module_b_v2 子模块。
维护说明：本目录只承载 v2 新链路，不与旧 module_b 逻辑混写。
"""

__all__ = [
    "MultiRoleScriptGeneratorV2",
    "run_module_b_v2",
    "run_module_b_v2_incremental",
    "invalidate_module_b_v2_role_outputs",
    "invalidate_module_b_v2_role_shot_outputs",
]


def __getattr__(name: str):
    """
    功能说明：按需惰性导出 module_b_v2 主入口，避免 package 初始化时引入循环依赖。
    参数说明：
    - name: 属性名。
    返回值：
    - object: 对应导出对象。
    异常说明：
    - AttributeError: 属性不存在时抛出。
    边界条件：仅支持 __all__ 中声明的两个入口。
    """
    if name in {
        "MultiRoleScriptGeneratorV2",
        "run_module_b_v2",
        "run_module_b_v2_incremental",
        "invalidate_module_b_v2_role_outputs",
        "invalidate_module_b_v2_role_shot_outputs",
    }:
        from music_video_pipeline.modules.module_b_v2.orchestrator import (
            MultiRoleScriptGeneratorV2,
            invalidate_module_b_v2_role_outputs,
            invalidate_module_b_v2_role_shot_outputs,
            run_module_b_v2,
            run_module_b_v2_incremental,
        )

        exports = {
            "MultiRoleScriptGeneratorV2": MultiRoleScriptGeneratorV2,
            "run_module_b_v2": run_module_b_v2,
            "run_module_b_v2_incremental": run_module_b_v2_incremental,
            "invalidate_module_b_v2_role_outputs": invalidate_module_b_v2_role_outputs,
            "invalidate_module_b_v2_role_shot_outputs": invalidate_module_b_v2_role_shot_outputs,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
