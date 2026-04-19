"""
文件用途：聚合跨模块 B/C/D 调度公共入口。
核心流程：对外暴露波前调度与自适应窗口状态快照函数。
输入输出：输入 RuntimeContext 与链路参数，输出调度摘要或窗口快照。
依赖说明：依赖 scheduler_engine 与 scheduler_adaptive 子模块。
维护说明：本文件仅保留公共接口，私有实现应放在子模块。
"""

from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
from music_video_pipeline.modules.cross_bcd.scheduler_adaptive import (
    collect_adaptive_window_status_snapshot as _collect_adaptive_window_status_snapshot,
)
from music_video_pipeline.modules.cross_bcd.scheduler_engine import (
    execute_cross_bcd_wavefront as _execute_cross_bcd_wavefront,
)
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint

__all__ = [
    "execute_cross_bcd_wavefront",
    "collect_adaptive_window_status_snapshot",
]


# 为保持类型提示与文档可见性，保留同名符号签名注释（运行时由导入实现提供）。
def execute_cross_bcd_wavefront(
    context: RuntimeContext,
    chain_units: list[CrossChainUnit],
    b_units_by_segment_id: dict[str, ModuleBUnit],
    d_blueprints_by_index: dict[int, ModuleDUnitBlueprint],
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
    frames_dir: Any,
    target_segment_id: str | None = None,
) -> dict[str, Any]:
    return _execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_by_segment_id,
        d_blueprints_by_index=d_blueprints_by_index,
        module_a_output=module_a_output,
        unit_outputs_dir=unit_outputs_dir,
        frames_dir=frames_dir,
        target_segment_id=target_segment_id,
    )


def collect_adaptive_window_status_snapshot(context: RuntimeContext) -> dict[str, Any]:
    return _collect_adaptive_window_status_snapshot(context=context)
