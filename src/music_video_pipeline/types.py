"""
文件用途：定义模块间 JSON 契约类型与基础校验函数。
核心流程：使用 TypedDict 描述 A/B 输出结构，提供最低字段检查。
输入输出：输入 Python 字典，输出校验结果（通过/抛错）。
依赖说明：依赖标准库 typing。
维护说明：字段新增需保持向后兼容，不得删除最低必填字段。
"""

# 标准库：用于声明结构化类型
from typing import Literal, TypedDict


TaskState = Literal["pending", "running", "done", "failed"]


class SegmentItem(TypedDict):
    """
    功能说明：描述音乐宏观段落信息。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：时间字段统一秒（float）。
    """

    segment_id: str
    start_time: float
    end_time: float
    label: str


class BeatItem(TypedDict):
    """
    功能说明：描述节拍或卡点时间信息。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：time 字段必须为非负秒值。
    """

    time: float
    type: str
    source: str


class LyricUnitItem(TypedDict):
    """
    功能说明：描述歌词最小语义单元。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：当歌词不可用时可返回空数组。
    """

    start_time: float
    end_time: float
    text: str
    confidence: float


class EnergyFeatureItem(TypedDict):
    """
    功能说明：描述片段能量特征。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：trend 建议仅使用 up/down/flat。
    """

    start_time: float
    end_time: float
    energy_level: str
    trend: str
    rhythm_tension: float


class ModuleAOutput(TypedDict):
    """
    功能说明：模块 A 输出契约。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：最低字段必须完整。
    """

    task_id: str
    audio_path: str
    segments: list[SegmentItem]
    beats: list[BeatItem]
    lyric_units: list[LyricUnitItem]
    energy_features: list[EnergyFeatureItem]


class ModuleBOutputItem(TypedDict):
    """
    功能说明：模块 B 的单个分镜输出项。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：时间区间必须满足 end_time > start_time；camera_motion 允许 none。
    """

    shot_id: str
    start_time: float
    end_time: float
    scene_desc: str
    image_prompt: str
    camera_motion: str
    transition: str
    constraints: dict[str, bool]


ModuleBOutput = list[ModuleBOutputItem]


def validate_module_a_output(data: dict) -> None:
    """
    功能说明：校验模块 A 输出是否满足最低契约字段。
    参数说明：
    - data: 待校验的模块 A 输出字典。
    返回值：无。
    异常说明：
    - KeyError: 缺失必填字段时抛出。
    - TypeError: 字段类型明显不符合时抛出。
    边界条件：仅做最低字段校验，不覆盖所有语义约束。
    """
    required_keys = {"task_id", "audio_path", "segments", "beats", "lyric_units", "energy_features"}
    missing_keys = required_keys.difference(data.keys())
    if missing_keys:
        raise KeyError(f"ModuleAOutput 缺失字段: {sorted(missing_keys)}")
    if not isinstance(data["segments"], list):
        raise TypeError("ModuleAOutput.segments 必须是 list")
    if not isinstance(data["beats"], list):
        raise TypeError("ModuleAOutput.beats 必须是 list")
    if not isinstance(data["lyric_units"], list):
        raise TypeError("ModuleAOutput.lyric_units 必须是 list")
    if not isinstance(data["energy_features"], list):
        raise TypeError("ModuleAOutput.energy_features 必须是 list")


def validate_module_b_output(data: list[dict]) -> None:
    """
    功能说明：校验模块 B 输出是否满足最低契约字段。
    参数说明：
    - data: 待校验的分镜数组。
    返回值：无。
    异常说明：
    - ValueError: 输出不是列表或列表为空时抛出。
    - KeyError: 任一分镜缺失必填字段时抛出。
    边界条件：仅校验最低字段，不校验语义质量。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("ModuleBOutput 必须是非空 list")
    required_keys = {"shot_id", "start_time", "end_time", "scene_desc", "image_prompt", "camera_motion", "transition"}
    valid_camera_motion = {"slow_pan", "zoom_in", "shake", "push_pull", "none"}
    for index, item in enumerate(data):
        missing_keys = required_keys.difference(item.keys())
        if missing_keys:
            raise KeyError(f"ModuleBOutput[{index}] 缺失字段: {sorted(missing_keys)}")
        camera_motion = str(item["camera_motion"])
        if camera_motion not in valid_camera_motion:
            raise ValueError(
                f"ModuleBOutput[{index}].camera_motion 非法: {camera_motion}，"
                f"合法值={sorted(valid_camera_motion)}"
            )
