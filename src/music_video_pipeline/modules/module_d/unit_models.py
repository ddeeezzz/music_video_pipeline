"""
文件用途：定义模块 D 最小执行单元的数据模型与构建函数。
核心流程：将模块 C 的 frame_items 转换为模块 D 单元结构，并按时间轴分配精确帧数。
输入输出：输入模块 C 帧清单与音频时长，输出模块 D 单元对象与状态同步载荷。
依赖说明：依赖标准库 dataclasses/pathlib/typing。
维护说明：最小单元固定为 shot，unit_id 默认映射 shot_id。
"""

# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any


@dataclass(frozen=True)
class ModuleDUnit:
    """
    功能说明：表示模块 D 的最小执行单元（一个 shot 对应一个视频片段）。
    参数说明：
    - unit_id: 单元唯一标识（等价 shot_id）。
    - unit_index: 单元顺序索引（0 基）。
    - shot: 原始 frame_item 数据。
    - start_time: 分镜起始时间（秒）。
    - end_time: 分镜结束时间（秒）。
    - duration: 分镜时长（秒）。
    - exact_frames: 该单元应渲染的精确帧数。
    - segment_path: 单元最终片段路径。
    - temp_segment_path: 单元临时片段路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：exact_frames 至少为 1。
    """

    unit_id: str
    unit_index: int
    shot: dict[str, Any]
    start_time: float
    end_time: float
    duration: float
    exact_frames: int
    segment_path: Path
    temp_segment_path: Path


@dataclass(frozen=True)
class ModuleDUnitBlueprint:
    """
    功能说明：表示模块 D 单元的预分配蓝图（不包含 frame_path）。
    参数说明：
    - unit_id: 单元唯一标识（等价 shot_id）。
    - unit_index: 单元顺序索引（0 基）。
    - start_time: 分镜起始时间（秒）。
    - end_time: 分镜结束时间（秒）。
    - duration: 分镜时长（秒）。
    - exact_frames: 该单元应渲染的精确帧数。
    - segment_path: 单元最终片段路径。
    - temp_segment_path: 单元临时片段路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：exact_frames 至少为 1。
    """

    unit_id: str
    unit_index: int
    start_time: float
    end_time: float
    duration: float
    exact_frames: int
    segment_path: Path
    temp_segment_path: Path


def build_module_d_units(
    frame_items: list[dict[str, Any]],
    audio_duration: float,
    fps: int,
    segments_dir: Path,
) -> list[ModuleDUnit]:
    """
    功能说明：将模块 C 帧清单转换为模块 D 单元数组。
    参数说明：
    - frame_items: 模块 C 输出的 frame_items。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    - segments_dir: 单元片段输出目录。
    返回值：
    - list[ModuleDUnit]: 模块 D 单元数组（按原始顺序）。
    异常说明：
    - ValueError: 缺失 shot_id 或 shot_id 重复时抛出。
    - RuntimeError: 帧分配失败时抛出。
    边界条件：duration <= 0 时统一修正为 0.1 秒。
    """
    allocated_frames = _allocate_segment_frames_by_timeline(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=fps,
    )

    units: list[ModuleDUnit] = []
    seen_unit_ids: set[str] = set()
    for shot_index, shot in enumerate(frame_items):
        unit_id = str(shot.get("shot_id", "")).strip()
        if not unit_id:
            raise ValueError(f"模块D单元构建失败：frame_items[{shot_index}] 缺失 shot_id")
        if unit_id in seen_unit_ids:
            raise ValueError(f"模块D单元构建失败：shot_id 重复，shot_id={unit_id}")
        seen_unit_ids.add(unit_id)

        start_time = float(shot.get("start_time", 0.0))
        end_time = max(start_time, float(shot.get("end_time", start_time)))
        duration = round(max(0.1, end_time - start_time), 3)
        segment_index = shot_index + 1
        units.append(
            ModuleDUnit(
                unit_id=unit_id,
                unit_index=shot_index,
                shot=dict(shot),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                exact_frames=int(allocated_frames[shot_index]),
                segment_path=segments_dir / f"segment_{segment_index:03d}.mp4",
                temp_segment_path=segments_dir / f"segment_{segment_index:03d}.tmp.mp4",
            )
        )
    return units


def build_module_d_unit_blueprints(
    shots: list[dict[str, Any]],
    audio_duration: float,
    fps: int,
    segments_dir: Path,
) -> list[ModuleDUnitBlueprint]:
    """
    功能说明：基于全量时间轴预分配模块 D 单元蓝图（用于跨模块链路模式）。
    参数说明：
    - shots: 以 shot 为粒度的时间清单（至少含 shot_id/start_time/end_time）。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    - segments_dir: 单元片段输出目录。
    返回值：
    - list[ModuleDUnitBlueprint]: 模块 D 单元蓝图数组（按原始顺序）。
    异常说明：
    - ValueError: 缺失 shot_id 或 shot_id 重复时抛出。
    - RuntimeError: 帧分配失败时抛出。
    边界条件：duration <= 0 时统一修正为 0.1 秒。
    """
    frame_like_items = [
        {
            "shot_id": str(item.get("shot_id", "")),
            "start_time": float(item.get("start_time", 0.0)),
            "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
            "duration": float(item.get("duration", 0.0)),
        }
        for item in shots
    ]
    allocated_frames = _allocate_segment_frames_by_timeline(
        frame_items=frame_like_items,
        audio_duration=audio_duration,
        fps=fps,
    )
    blueprints: list[ModuleDUnitBlueprint] = []
    seen_unit_ids: set[str] = set()
    for shot_index, shot in enumerate(frame_like_items):
        unit_id = str(shot.get("shot_id", "")).strip()
        if not unit_id:
            raise ValueError(f"模块D蓝图构建失败：shots[{shot_index}] 缺失 shot_id")
        if unit_id in seen_unit_ids:
            raise ValueError(f"模块D蓝图构建失败：shot_id 重复，shot_id={unit_id}")
        seen_unit_ids.add(unit_id)
        start_time = float(shot.get("start_time", 0.0))
        end_time = max(start_time, float(shot.get("end_time", start_time)))
        duration = round(max(0.1, end_time - start_time), 3)
        segment_index = shot_index + 1
        blueprints.append(
            ModuleDUnitBlueprint(
                unit_id=unit_id,
                unit_index=shot_index,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                exact_frames=int(allocated_frames[shot_index]),
                segment_path=segments_dir / f"segment_{segment_index:03d}.mp4",
                temp_segment_path=segments_dir / f"segment_{segment_index:03d}.tmp.mp4",
            )
        )
    return blueprints


def materialize_module_d_unit(blueprint: ModuleDUnitBlueprint, frame_item: dict[str, Any]) -> ModuleDUnit:
    """
    功能说明：将模块 D 单元蓝图与 frame_item 合并为可执行单元对象。
    参数说明：
    - blueprint: 预分配蓝图。
    - frame_item: 模块 C 输出的单帧单元信息。
    返回值：
    - ModuleDUnit: 可直接交给执行器的模块 D 单元对象。
    异常说明：无。
    边界条件：当 frame_item 缺失时间字段时回退使用蓝图时间。
    """
    shot_obj = dict(frame_item)
    start_time = float(shot_obj.get("start_time", blueprint.start_time))
    end_time = max(start_time, float(shot_obj.get("end_time", blueprint.end_time)))
    duration = round(max(0.1, end_time - start_time), 3)
    return ModuleDUnit(
        unit_id=blueprint.unit_id,
        unit_index=blueprint.unit_index,
        shot=shot_obj,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        exact_frames=blueprint.exact_frames,
        segment_path=blueprint.segment_path,
        temp_segment_path=blueprint.temp_segment_path,
    )


def build_unit_sync_payload(units: list[ModuleDUnit]) -> list[dict[str, Any]]:
    """
    功能说明：构建写入状态库的单元元信息载荷。
    参数说明：
    - units: 模块 D 单元数组。
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


def build_unit_map(units: list[ModuleDUnit]) -> dict[str, ModuleDUnit]:
    """
    功能说明：将模块 D 单元数组转换为 unit_id 索引映射。
    参数说明：
    - units: 模块 D 单元数组。
    返回值：
    - dict[str, ModuleDUnit]: unit_id 到单元对象的映射。
    异常说明：无。
    边界条件：假设 unit_id 在输入中已唯一。
    """
    return {unit.unit_id: unit for unit in units}


def _allocate_segment_frames_by_timeline(
    frame_items: list[dict[str, Any]],
    audio_duration: float,
    fps: int,
) -> list[int]:
    """
    功能说明：根据全局时间轴为每个片段分配绝对帧数，消除累积舍入误差。
    参数说明：
    - frame_items: 模块 C 产出的帧清单，需包含 start_time/end_time。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    返回值：
    - list[int]: 与 frame_items 一一对应的片段帧数。
    异常说明：输入非法或无法满足最小帧分配时抛 RuntimeError。
    边界条件：每个片段至少分配 1 帧，总帧数严格等于 round(audio_duration * fps)。
    """
    if not frame_items:
        raise RuntimeError("模块D帧分配失败：frame_items 为空。")
    if fps <= 0:
        raise RuntimeError(f"模块D帧分配失败：fps 非法，fps={fps}")

    safe_audio_duration = max(0.1, float(audio_duration))
    target_total_frames = max(1, round(safe_audio_duration * fps))
    segment_count = len(frame_items)
    if segment_count > target_total_frames:
        raise RuntimeError(
            f"模块D帧分配失败：片段数大于目标总帧数，segment_count={segment_count}, target_total_frames={target_total_frames}"
        )

    normalized_end_frames: list[int] = []
    last_start_time = 0.0
    for index, item in enumerate(frame_items, start=1):
        start_time = float(item.get("start_time", 0.0))
        if "end_time" in item:
            end_time = float(item["end_time"])
        else:
            end_time = start_time + max(0.1, float(item.get("duration", 0.1)))

        if end_time < start_time:
            raise RuntimeError(f"模块D帧分配失败：片段时间区间非法，index={index}, start={start_time}, end={end_time}")
        if start_time < last_start_time:
            raise RuntimeError(
                f"模块D帧分配失败：片段开始时间未按升序，index={index}, previous_start={last_start_time}, start={start_time}"
            )
        last_start_time = start_time

        clamped_end = max(0.0, min(safe_audio_duration, end_time))
        normalized_end_frames.append(round(clamped_end * fps))

    allocated_frames: list[int] = []
    previous_end_frame = 0
    for index, raw_end_frame in enumerate(normalized_end_frames, start=1):
        remaining_segments = segment_count - index
        min_end_frame = previous_end_frame + 1
        max_end_frame = target_total_frames - remaining_segments
        clamped_end_frame = min(max(raw_end_frame, min_end_frame), max_end_frame)
        current_frames = clamped_end_frame - previous_end_frame
        allocated_frames.append(current_frames)
        previous_end_frame = clamped_end_frame

    if sum(allocated_frames) != target_total_frames:
        raise RuntimeError(
            f"模块D帧分配失败：总帧数不一致，allocated={sum(allocated_frames)}, target={target_total_frames}"
        )
    if any(frame_count <= 0 for frame_count in allocated_frames):
        raise RuntimeError("模块D帧分配失败：存在非正帧片段。")
    return allocated_frames


def _build_frame_allocation_detail_lines(frame_items: list[dict[str, Any]], allocated_frames: list[int], fps: int) -> list[str]:
    """
    功能说明：构建逐段帧分配明细文本，用于失败排障。
    参数说明：
    - frame_items: 帧清单数组。
    - allocated_frames: 已分配帧数组。
    - fps: 输出帧率。
    返回值：
    - list[str]: 可直接拼接输出的明细行。
    异常说明：无。
    边界条件：若字段缺失则使用默认值，不中断明细输出。
    """
    lines: list[str] = []
    cumulative_frames = 0
    for index, (item, frame_count) in enumerate(zip(frame_items, allocated_frames, strict=True), start=1):
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        cumulative_frames += frame_count
        lines.append(
            (
                f"- segment_{index:03d}: start={start_time:.3f}s, end={end_time:.3f}s, "
                f"frames={frame_count}, cumulative_frames={cumulative_frames}, fps={fps}"
            )
        )
    return lines
