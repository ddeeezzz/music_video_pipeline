"""
文件用途：定义模块 C 最小视觉单元的数据模型与构建函数。
核心流程：将模块 B 的 shot 列表转换为模块 C 单元结构，提供稳定索引与映射。
输入输出：输入模块 B 分镜数组，输出模块 C 单元对象与状态同步载荷。
依赖说明：依赖标准库 dataclasses/typing。
维护说明：最小视觉单元固定为 shot，unit_id 默认映射 shot_id。
"""

# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于类型提示
from typing import Any


@dataclass(frozen=True)
class ModuleCUnit:
    """
    功能说明：表示模块 C 的最小执行单元（一个 shot）。
    参数说明：
    - unit_id: 单元唯一标识（等价 shot_id）。
    - unit_index: 单元顺序索引（0 基）。
    - shot: 原始分镜数据。
    - start_time: 分镜起始时间（秒）。
    - end_time: 分镜结束时间（秒）。
    - duration: 分镜时长（秒）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：duration 最小值固定为 0.5 秒。
    """

    unit_id: str
    unit_index: int
    shot: dict[str, Any]
    start_time: float
    end_time: float
    duration: float


def build_module_c_units(shots: list[dict[str, Any]]) -> list[ModuleCUnit]:
    """
    功能说明：将模块 B 分镜数组转换为模块 C 单元数组。
    参数说明：
    - shots: 模块 B 输出分镜数组。
    返回值：
    - list[ModuleCUnit]: 模块 C 单元数组（按原始顺序）。
    异常说明：
    - ValueError: 缺失 shot_id 或存在重复 shot_id 时抛出。
    边界条件：duration <= 0 时统一修正为 0.5 秒。
    """
    units: list[ModuleCUnit] = []
    seen_unit_ids: set[str] = set()
    for shot_index, shot in enumerate(shots):
        unit_id = str(shot.get("shot_id", "")).strip()
        if not unit_id:
            raise ValueError(f"模块C单元构建失败：shot[{shot_index}] 缺失 shot_id")
        if unit_id in seen_unit_ids:
            raise ValueError(f"模块C单元构建失败：shot_id 重复，shot_id={unit_id}")
        seen_unit_ids.add(unit_id)

        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)
        units.append(
            ModuleCUnit(
                unit_id=unit_id,
                unit_index=shot_index,
                shot=dict(shot),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
        )
    return units


def build_unit_sync_payload(units: list[ModuleCUnit]) -> list[dict[str, Any]]:
    """
    功能说明：构建写入状态库的单元元信息载荷。
    参数说明：
    - units: 模块 C 单元数组。
    返回值：
    - list[dict[str, Any]]: 可直接传入状态库同步接口的字典数组。
    异常说明：无。
    边界条件：输出顺序保持与输入一致。
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


def build_unit_map(units: list[ModuleCUnit]) -> dict[str, ModuleCUnit]:
    """
    功能说明：将模块 C 单元数组转换为 unit_id 索引映射。
    参数说明：
    - units: 模块 C 单元数组。
    返回值：
    - dict[str, ModuleCUnit]: unit_id 到单元对象的映射。
    异常说明：无。
    边界条件：假设 unit_id 在输入中已唯一。
    """
    return {unit.unit_id: unit for unit in units}
