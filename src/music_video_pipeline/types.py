"""
文件用途：定义模块间 JSON 契约类型与基础校验函数。
核心流程：使用 TypedDict 描述 A/B 输出结构，提供最低字段与时间轴一致性检查。
输入输出：输入 Python 字典，输出校验结果（通过/抛错）。
依赖说明：依赖标准库 typing。
维护说明：字段新增需保持向后兼容，不得删除最小必填字段。
"""

# 标准库：用于声明结构化类型
from typing import Literal, NotRequired, TypedDict


TaskState = Literal["pending", "running", "done", "failed"]


class BigSegmentItem(TypedDict):
    """
    功能说明：描述宏观段落（大时间戳区间）。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：时间字段统一秒（float）。
    """

    segment_id: str
    start_time: float
    end_time: float
    label: str


class SegmentItem(TypedDict):
    """
    功能说明：描述最小视觉单元（小段落）。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：需关联所属大段落 big_segment_id。
    """

    segment_id: str
    big_segment_id: str
    start_time: float
    end_time: float
    label: str


class BeatItem(TypedDict):
    """
    功能说明：描述最终小时间戳点。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：time 必须非负且升序。
    """

    time: float
    type: str
    source: str


class LyricUnitItem(TypedDict):
    """
    功能说明：描述歌词语义单元并对齐到小段落。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：segment_id 必须能在 segments 中找到。
    """

    segment_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float
    token_units: NotRequired[list[dict]]
    source_sentence_index: NotRequired[int]
    unit_transform: NotRequired[str]


class EnergyFeatureItem(TypedDict):
    """
    功能说明：描述小段落能量特征。
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
    边界条件：最小字段必须完整。
    """

    task_id: str
    audio_path: str
    big_segments: list[BigSegmentItem]
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
    边界条件：end_time 必须大于 start_time。
    """

    shot_id: str
    start_time: float
    end_time: float
    scene_desc: str
    image_prompt: str
    camera_motion: str
    transition: str
    constraints: dict[str, bool]
    lyric_text: NotRequired[str]
    lyric_units: NotRequired[list[dict]]
    big_segment_id: NotRequired[str]
    big_segment_label: NotRequired[str]
    segment_label: NotRequired[str]
    audio_role: NotRequired[str]


ModuleBOutput = list[ModuleBOutputItem]


def _safe_float(value: object, field_name: str) -> float:
    """
    功能说明：将任意数值转换为 float 并做类型保护。
    参数说明：
    - value: 待转换值。
    - field_name: 字段名（用于报错定位）。
    返回值：
    - float: 转换后的浮点值。
    异常说明：
    - TypeError/ValueError: 转换失败时抛出。
    边界条件：布尔值视为非法数值。
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 不能是 bool")
    return float(value)


def validate_module_a_output(data: dict) -> None:
    """
    功能说明：校验模块 A 输出是否满足最小契约与时间轴规则。
    参数说明：
    - data: 待校验的模块 A 输出字典。
    返回值：无。
    异常说明：
    - KeyError: 缺失必填字段。
    - TypeError/ValueError: 字段类型或时间轴非法。
    边界条件：时间连续性容差为 0.02 秒。
    """
    required_keys = {"task_id", "audio_path", "big_segments", "segments", "beats", "lyric_units", "energy_features"}
    missing_keys = required_keys.difference(data.keys())
    if missing_keys:
        raise KeyError(f"ModuleAOutput 缺失字段: {sorted(missing_keys)}")

    for key in ["big_segments", "segments", "beats", "lyric_units", "energy_features"]:
        if not isinstance(data[key], list):
            raise TypeError(f"ModuleAOutput.{key} 必须是 list")

    big_segments = data["big_segments"]
    segments = data["segments"]
    beats = data["beats"]
    lyric_units = data["lyric_units"]

    if not big_segments:
        raise ValueError("ModuleAOutput.big_segments 不能为空")
    if not segments:
        raise ValueError("ModuleAOutput.segments 不能为空")
    if len(beats) < 2:
        raise ValueError("ModuleAOutput.beats 至少包含起止两个时间戳")

    tol = 0.02

    # 大段落校验
    last_end = None
    for index, item in enumerate(big_segments):
        for key in ["segment_id", "start_time", "end_time", "label"]:
            if key not in item:
                raise KeyError(f"ModuleAOutput.big_segments[{index}] 缺失字段: {key}")
        start_time = _safe_float(item["start_time"], f"big_segments[{index}].start_time")
        end_time = _safe_float(item["end_time"], f"big_segments[{index}].end_time")
        if end_time <= start_time:
            raise ValueError(f"ModuleAOutput.big_segments[{index}] 时间区间非法")
        if last_end is not None and start_time < last_end - tol:
            raise ValueError("ModuleAOutput.big_segments 存在重叠")
        last_end = end_time

    # 小段落校验
    segment_ids = set()
    big_segment_ids = {str(item["segment_id"]) for item in big_segments}
    last_end = None
    for index, item in enumerate(segments):
        for key in ["segment_id", "big_segment_id", "start_time", "end_time", "label"]:
            if key not in item:
                raise KeyError(f"ModuleAOutput.segments[{index}] 缺失字段: {key}")
        segment_id = str(item["segment_id"])
        if segment_id in segment_ids:
            raise ValueError(f"ModuleAOutput.segments.segment_id 重复: {segment_id}")
        segment_ids.add(segment_id)

        big_segment_id = str(item["big_segment_id"])
        if big_segment_id not in big_segment_ids:
            raise ValueError(f"ModuleAOutput.segments[{index}] 引用了不存在的 big_segment_id: {big_segment_id}")

        start_time = _safe_float(item["start_time"], f"segments[{index}].start_time")
        end_time = _safe_float(item["end_time"], f"segments[{index}].end_time")
        if end_time <= start_time:
            raise ValueError(f"ModuleAOutput.segments[{index}] 时间区间非法")
        if last_end is not None and abs(start_time - last_end) > tol:
            raise ValueError("ModuleAOutput.segments 时间轴不连续")
        last_end = end_time

    # 节拍校验（最终小时间戳）
    last_time = None
    for index, item in enumerate(beats):
        for key in ["time", "type", "source"]:
            if key not in item:
                raise KeyError(f"ModuleAOutput.beats[{index}] 缺失字段: {key}")
        time_value = _safe_float(item["time"], f"beats[{index}].time")
        if time_value < 0:
            raise ValueError("ModuleAOutput.beats.time 不能为负")
        if last_time is not None and time_value < last_time - 1e-6:
            raise ValueError("ModuleAOutput.beats 必须升序")
        last_time = time_value

    # 歌词校验
    for index, item in enumerate(lyric_units):
        for key in ["segment_id", "start_time", "end_time", "text", "confidence"]:
            if key not in item:
                raise KeyError(f"ModuleAOutput.lyric_units[{index}] 缺失字段: {key}")
        segment_id = str(item["segment_id"])
        if segment_id not in segment_ids:
            raise ValueError(f"ModuleAOutput.lyric_units[{index}] 引用了不存在的 segment_id: {segment_id}")
        start_time = _safe_float(item["start_time"], f"lyric_units[{index}].start_time")
        end_time = _safe_float(item["end_time"], f"lyric_units[{index}].end_time")
        if end_time < start_time:
            raise ValueError(f"ModuleAOutput.lyric_units[{index}] 时间区间非法")
        if not isinstance(item["text"], str):
            raise TypeError(f"ModuleAOutput.lyric_units[{index}].text 必须是 str")
        _safe_float(item["confidence"], f"lyric_units[{index}].confidence")
        if "token_units" in item:
            token_units = item["token_units"]
            if not isinstance(token_units, list):
                raise TypeError(f"ModuleAOutput.lyric_units[{index}].token_units 必须是 list")
            for token_index, token_item in enumerate(token_units):
                if not isinstance(token_item, dict):
                    raise TypeError(
                        f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}] 必须是 dict"
                    )
                for token_key in ["text", "start_time", "end_time", "granularity"]:
                    if token_key not in token_item:
                        raise KeyError(
                            f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}] 缺失字段: {token_key}"
                        )
                if not isinstance(token_item["text"], str):
                    raise TypeError(
                        f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}].text 必须是 str"
                    )
                token_start = _safe_float(
                    token_item["start_time"],
                    f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}].start_time",
                )
                token_end = _safe_float(
                    token_item["end_time"],
                    f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}].end_time",
                )
                if token_end < token_start:
                    raise ValueError(
                        f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}] 时间区间非法"
                    )
                granularity = str(token_item["granularity"]).strip()
                if granularity not in {"word", "char"}:
                    raise ValueError(
                        f"ModuleAOutput.lyric_units[{index}].token_units[{token_index}].granularity 非法: {granularity}"
                    )
        if "source_sentence_index" in item:
            source_sentence_index = item["source_sentence_index"]
            if not isinstance(source_sentence_index, int) or isinstance(source_sentence_index, bool) or source_sentence_index < 0:
                raise TypeError(f"ModuleAOutput.lyric_units[{index}].source_sentence_index 必须是非负 int")
        if "unit_transform" in item:
            unit_transform = str(item["unit_transform"]).strip()
            if unit_transform not in {"original", "split", "merged"}:
                raise ValueError(
                    f"ModuleAOutput.lyric_units[{index}].unit_transform 非法: {unit_transform}"
                )


def validate_module_b_output(data: list[dict]) -> None:
    """
    功能说明：校验模块 B 输出是否满足最小契约字段。
    参数说明：
    - data: 待校验的分镜数组。
    返回值：无。
    异常说明：
    - ValueError: 输出不是列表或列表为空。
    - KeyError: 任一分镜缺失必填字段。
    边界条件：兼容旧产物，歌词扩展字段允许缺失。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("ModuleBOutput 必须是非空 list")

    required_keys = {"shot_id", "start_time", "end_time", "scene_desc", "image_prompt", "camera_motion", "transition"}
    valid_camera_motion = {"slow_pan", "zoom_in", "shake", "push_pull", "none"}
    valid_audio_role = {"instrumental", "vocal"}

    for index, item in enumerate(data):
        missing_keys = required_keys.difference(item.keys())
        if missing_keys:
            raise KeyError(f"ModuleBOutput[{index}] 缺失字段: {sorted(missing_keys)}")

        camera_motion = str(item["camera_motion"])
        if camera_motion not in valid_camera_motion:
            raise ValueError(
                f"ModuleBOutput[{index}].camera_motion 非法: {camera_motion}，"
                f"合法值: {sorted(valid_camera_motion)}"
            )

        if "lyric_text" in item and not isinstance(item["lyric_text"], str):
            raise TypeError(f"ModuleBOutput[{index}].lyric_text 必须是 str")

        for optional_key in ["big_segment_id", "big_segment_label", "segment_label"]:
            if optional_key in item and not isinstance(item[optional_key], str):
                raise TypeError(f"ModuleBOutput[{index}].{optional_key} 必须是 str")

        if "audio_role" in item:
            if not isinstance(item["audio_role"], str):
                raise TypeError(f"ModuleBOutput[{index}].audio_role 必须是 str")
            audio_role = str(item["audio_role"]).strip()
            if audio_role not in valid_audio_role:
                raise ValueError(
                    f"ModuleBOutput[{index}].audio_role 非法: {audio_role}，"
                    f"合法值: {sorted(valid_audio_role)}"
                )

        if "lyric_units" in item:
            lyric_units = item["lyric_units"]
            if not isinstance(lyric_units, list):
                raise TypeError(f"ModuleBOutput[{index}].lyric_units 必须是 list")

            for lyric_index, lyric_item in enumerate(lyric_units):
                if not isinstance(lyric_item, dict):
                    raise TypeError(f"ModuleBOutput[{index}].lyric_units[{lyric_index}] 必须是 dict")

                for key in ["start_time", "end_time", "text", "confidence"]:
                    if key not in lyric_item:
                        raise KeyError(
                            f"ModuleBOutput[{index}].lyric_units[{lyric_index}] 缺失字段: {key}"
                        )

                start_time = _safe_float(
                    lyric_item["start_time"],
                    f"ModuleBOutput[{index}].lyric_units[{lyric_index}].start_time",
                )
                end_time = _safe_float(
                    lyric_item["end_time"],
                    f"ModuleBOutput[{index}].lyric_units[{lyric_index}].end_time",
                )
                if end_time < start_time:
                    raise ValueError(
                        f"ModuleBOutput[{index}].lyric_units[{lyric_index}] 时间区间非法"
                    )

                if not isinstance(lyric_item["text"], str):
                    raise TypeError(
                        f"ModuleBOutput[{index}].lyric_units[{lyric_index}].text 必须是 str"
                    )
                _safe_float(
                    lyric_item["confidence"],
                    f"ModuleBOutput[{index}].lyric_units[{lyric_index}].confidence",
                )
