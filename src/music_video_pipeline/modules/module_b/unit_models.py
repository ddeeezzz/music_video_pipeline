"""
文件用途：提供模块 B 的执行单元与辅助构建函数。
核心流程：根据模块 A 输出构建执行单元、状态同步载荷和索引映射。
输入输出：输入模块 A 输出或单元数组，输出模块 B 单元相关结构。
依赖说明：依赖标准库 dataclasses、typing。
维护说明：单元粒度应与编排、重试和状态记录策略保持一致。
"""

# 标准库：用于定义轻量不可变数据类。
from dataclasses import dataclass
# 标准库：用于类型标注。
from typing import Any


@dataclass(frozen=True)
class ModuleBUnit:
    """
    功能说明：表示模块 B 的最小执行单元。
    参数说明：
    - unit_id: 单元唯一标识。
    - unit_index: 单元在线路中的顺序编号。
    - segment: 与该单元对应的模块 A 段落信息。
    - start_time: 单元起始时间（秒）。
    - end_time: 单元结束时间（秒）。
    - duration: 单元时长（秒）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：segment 内部结构由上游决定，这里不做额外约束。
    """

    unit_id: str
    unit_index: int
    segment: dict[str, Any]
    start_time: float
    end_time: float
    duration: float


def build_module_b_units(module_a_output: dict[str, Any]) -> list[ModuleBUnit]:
    """
    功能说明：构建模块 B 的执行单元数组。
    参数说明：
    - module_a_output: 模块 A 输出对象。
    返回值：
    - list[ModuleBUnit]: 模块 B 执行单元数组。
    异常说明：按具体实现定义。
    边界条件：单元顺序应与模块 A 时间线保持一致。
    """
    big_segments: list[dict[str, Any]] = module_a_output.get("big_segments", []) or []
    if not isinstance(big_segments, list):
        big_segments = []
    if not big_segments:
        if module_a_output.get("segments"):
            big_segments = module_a_output["segments"]

    units: list[ModuleBUnit] = []
    for idx, seg in enumerate(big_segments):
        if not isinstance(seg, dict):
            continue
        seg_id = str(seg.get("segment_id", "")).strip() or f"seg_{idx:04d}"
        start_time = float(seg.get("start_time", 0) or 0)
        end_time = float(seg.get("end_time", start_time) or start_time)
        duration = float(seg.get("duration", 0) or max(0.5, end_time - start_time))
        units.append(
            ModuleBUnit(
                unit_id=seg_id,
                unit_index=idx,
                segment=seg,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
        )
    return units


def build_unit_sync_payload(units: list[ModuleBUnit]) -> list[dict[str, Any]]:
    """
    功能说明：构建模块 B 单元状态同步载荷。
    参数说明：
    - units: 模块 B 执行单元数组。
    返回值：
    - list[dict[str, Any]]: 状态同步载荷。
    异常说明：按具体实现定义。
    边界条件：输出字段应满足状态库写入要求。
    """
    return [
        {
            "unit_id": unit.unit_id,
            "unit_index": unit.unit_index,
            "start_time": unit.start_time,
            "end_time": unit.end_time,
            "duration": unit.duration,
        }
        for unit in units
    ]


def build_unit_map(units: list[ModuleBUnit]) -> dict[str, ModuleBUnit]:
    """
    功能说明：构建模块 B 单元索引映射。
    参数说明：
    - units: 模块 B 执行单元数组。
    返回值：
    - dict[str, ModuleBUnit]: 单元索引映射。
    异常说明：按具体实现定义。
    边界条件：映射键应与单元唯一标识保持一致。
    """
    return {unit.unit_id: unit for unit in units}
