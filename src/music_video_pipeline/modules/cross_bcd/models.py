"""
文件用途：定义跨模块 B/C/D 链路并行调度的数据模型与构建函数。
核心流程：基于模块 A segments 构建稳定链路单元，统一 segment_id/shot_id 与时间轴索引。
输入输出：输入模块 A 输出，输出链路单元数组。
依赖说明：依赖标准库 dataclasses/typing。
维护说明：跨模块关联主键固定为 unit_index + 派生 shot_id，不引入额外稳定键。
"""

# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于类型提示
from typing import Any

# 项目内模块：模块 B 单元构建工具
from music_video_pipeline.modules.module_b.unit_models import build_module_b_units


@dataclass(frozen=True)
class CrossChainUnit:
    """
    功能说明：表示跨模块 B/C/D 的单条链路单元。
    参数说明：
    - unit_index: 链路顺序索引（0 基）。
    - segment_id: 模块 B 单元标识。
    - shot_id: 模块 C/D 单元标识（按索引派生）。
    - start_time: 起始时间（秒）。
    - end_time: 结束时间（秒）。
    - duration: 时长（秒）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：duration 最小值固定为 0.5 秒。
    """

    unit_index: int
    segment_id: str
    shot_id: str
    start_time: float
    end_time: float
    duration: float


def build_cross_chain_units(module_a_output: dict[str, Any]) -> list[CrossChainUnit]:
    """
    功能说明：基于模块 A 输出构建跨模块链路单元数组。
    参数说明：
    - module_a_output: 模块 A 输出字典。
    返回值：
    - list[CrossChainUnit]: 按 unit_index 升序的链路单元数组。
    异常说明：
    - ValueError: segment_id 缺失或重复时抛出。
    边界条件：shot_id 固定按 shot_{index+1:03d} 派生。
    """
    b_units = build_module_b_units(module_a_output=module_a_output)
    chain_units: list[CrossChainUnit] = []
    for b_unit in b_units:
        chain_units.append(
            CrossChainUnit(
                unit_index=int(b_unit.unit_index),
                segment_id=str(b_unit.unit_id),
                shot_id=f"shot_{int(b_unit.unit_index) + 1:03d}",
                start_time=float(b_unit.start_time),
                end_time=float(b_unit.end_time),
                duration=float(b_unit.duration),
            )
        )
    return chain_units
