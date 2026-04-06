"""
文件用途：提供模块 B 的分镜生成器抽象与实现。
核心流程：根据配置选择 mock/llm 生成器，输出统一分镜 JSON 列表。
输入输出：输入 ModuleAOutput 字典，输出 ModuleBOutput 列表。
依赖说明：依赖标准库 abc/logging。
维护说明：真实 LLM 接入时仅替换 LlmScriptGenerator 内部逻辑。
"""

# 标准库：定义抽象基类
from abc import ABC, abstractmethod
# 标准库：日志输出
import logging
# 标准库：正则处理
import re
# 标准库：类型提示
from typing import Any

UNKNOWN_LYRIC_TEXT = "[未识别歌词]"
CHANT_LYRIC_TEXT = "吟唱"
# 常量：用于清理歌词文本句首标点（含中英文常见符号）的正则。
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+")
# 常量：用于识别“纯标点文本”，避免将其作为可展示歌词。
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+$")


class ScriptGenerator(ABC):
    """分镜生成器接口定义。"""

    @abstractmethod
    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """根据模块A输出生成分镜数组。"""
        raise NotImplementedError


class MockScriptGenerator(ScriptGenerator):
    """规则化分镜生成器（MVP 默认实现）。"""

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：按小段落生成分镜，并执行歌词挂载（segment_id优先+时间重叠兜底）。
        参数说明：
        - module_a_output: 模块A输出字典，至少包含 segments/energy_features。
        返回值：
        - list[dict[str, Any]]: 模块B分镜数组。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：歌词挂载统一采用“segment_id 优先，时间重叠兜底”。
        """
        segments = module_a_output["segments"]
        big_segments = module_a_output.get("big_segments", [])
        energy_features = module_a_output["energy_features"]
        lyric_units = module_a_output.get("lyric_units", [])

        big_segment_label_map = {
            str(item.get("segment_id", "")): str(item.get("label", "unknown")) for item in big_segments
        }
        lyric_units_map, lyric_units_sorted = self._build_lyric_indexes(lyric_units=lyric_units)

        shots: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            energy = energy_features[min(index, len(energy_features) - 1)] if energy_features else {"energy_level": "mid", "trend": "flat"}
            energy_level = str(energy.get("energy_level", "mid"))
            trend = str(energy.get("trend", "flat"))
            camera_motion = self._choose_camera_motion(energy_level=energy_level, trend=trend)
            transition = "hard_cut" if energy_level == "high" else "crossfade"

            big_segment_id = str(segment.get("big_segment_id", ""))
            big_label = big_segment_label_map.get(big_segment_id, str(segment.get("label", "unknown")))
            small_label = str(segment.get("label", "unknown"))
            segment_id = str(segment.get("segment_id", ""))
            segment_start = float(segment.get("start_time", 0.0))
            segment_end = max(segment_start, float(segment.get("end_time", segment_start)))
            shot_lyric_units = [dict(item) for item in lyric_units_map.get(segment_id, [])]
            if not shot_lyric_units:
                shot_lyric_units = self._collect_lyric_units_by_overlap(
                    lyric_units_sorted=lyric_units_sorted,
                    start_time=segment_start,
                    end_time=segment_end,
                )
            shot_lyric_units.sort(key=lambda unit: float(unit.get("start_time", 0.0)))
            lyric_text = self._build_lyric_text(shot_lyric_units)

            shots.append(
                {
                    "shot_id": f"shot_{index + 1:03d}",
                    "start_time": segment_start,
                    "end_time": segment_end,
                    "scene_desc": f"大段落{big_label}中的小段落{small_label}，强调节奏同步",
                    "image_prompt": f"Cinematic scene, {big_label}, {small_label}, rhythm aligned, high detail",
                    "camera_motion": camera_motion,
                    "transition": transition,
                    "constraints": {"must_keep_style": True, "must_align_to_beat": True},
                    "lyric_text": lyric_text,
                    "lyric_units": shot_lyric_units,
                }
            )

        if not shots:
            shots.append(
                {
                    "shot_id": "shot_001",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "scene_desc": "默认场景占位",
                    "image_prompt": "Cinematic placeholder scene",
                    "camera_motion": "slow_pan",
                    "transition": "crossfade",
                    "constraints": {"must_keep_style": True, "must_align_to_beat": True},
                    "lyric_text": "",
                    "lyric_units": [],
                }
            )
        return shots

    def _build_lyric_indexes(self, lyric_units: Any) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        """
        功能说明：构建歌词的 segment_id 索引与时间排序索引。
        参数说明：
        - lyric_units: 模块A输出中的歌词单元列表。
        返回值：
        - tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
          第1项为按 segment_id 聚合索引，第2项为按 start_time 升序列表。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：输入非法时返回空索引。
        """
        lyric_units_sorted: list[dict[str, Any]] = []
        lyric_units_map = self._build_lyric_index_by_segment(lyric_units=lyric_units)
        if not isinstance(lyric_units, list):
            return lyric_units_map, lyric_units_sorted
        for item in lyric_units:
            if not isinstance(item, dict):
                continue
            normalized_item = self._normalize_lyric_unit(item=item)
            if normalized_item is None:
                continue
            lyric_units_sorted.append(normalized_item)
        lyric_units_sorted.sort(key=lambda unit: float(unit.get("start_time", 0.0)))
        return lyric_units_map, lyric_units_sorted

    def _collect_lyric_units_by_overlap(
        self,
        lyric_units_sorted: list[dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> list[dict[str, Any]]:
        """
        功能说明：按时间重叠收集歌词单元，作为 segment_id 缺失时的兜底挂载策略。
        参数说明：
        - lyric_units_sorted: 按 start_time 升序的歌词单元列表。
        - start_time/end_time: 当前分镜时间窗。
        返回值：
        - list[dict[str, Any]]: 与分镜有重叠的歌词单元列表。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：无重叠时返回空列表。
        """
        if not lyric_units_sorted or end_time <= start_time:
            return []
        overlap_units: list[dict[str, Any]] = []
        for item in lyric_units_sorted:
            lyric_start = float(item.get("start_time", 0.0))
            lyric_end = max(lyric_start, float(item.get("end_time", lyric_start)))
            if lyric_end <= start_time:
                continue
            if lyric_start >= end_time:
                break
            if max(0.0, min(end_time, lyric_end) - max(start_time, lyric_start)) <= 1e-6:
                continue
            overlap_units.append(dict(item))
        return overlap_units

    def _choose_camera_motion(self, energy_level: str, trend: str) -> str:
        """根据能量等级与变化趋势选择运镜策略。"""
        if energy_level == "low" and trend in {"flat", "down"}:
            return "none"
        if energy_level == "high" and trend == "up":
            return "push_pull"
        if energy_level == "high":
            return "shake"
        if energy_level == "mid":
            return "zoom_in"
        return "slow_pan"

    def _build_lyric_index_by_segment(self, lyric_units: Any) -> dict[str, list[dict[str, Any]]]:
        """
        功能说明：按 segment_id 聚合歌词单元，并按 start_time 升序排列。
        参数说明：
        - lyric_units: 模块A输出中的歌词单元列表。
        返回值：
        - dict[str, list[dict[str, Any]]]: segment_id 到歌词单元数组的映射。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：文本为空或仅标点的歌词单元会被过滤。
        """
        if not isinstance(lyric_units, list):
            return {}

        lyric_units_map: dict[str, list[dict[str, Any]]] = {}
        for item in lyric_units:
            if not isinstance(item, dict):
                continue
            normalized_item = self._normalize_lyric_unit(item=item)
            if normalized_item is None:
                continue
            segment_id = str(normalized_item.get("segment_id", "")).strip()
            if not segment_id:
                continue
            lyric_units_map.setdefault(segment_id, []).append(normalized_item)

        for segment_id in lyric_units_map:
            lyric_units_map[segment_id].sort(key=lambda unit: float(unit.get("start_time", 0.0)))
        return lyric_units_map

    def _build_lyric_text(self, lyric_units: list[dict[str, Any]]) -> str:
        """
        功能说明：将歌词单元文本按优先级聚合为分镜展示文案。
        参数说明：
        - lyric_units: 当前分镜挂载的歌词单元列表。
        返回值：
        - str: 聚合后的展示歌词文本。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：纯标点文本会被过滤。
        """
        reliable_text_items: list[str] = []
        has_unknown = False
        has_chant = False
        for item in lyric_units:
            text = self._clean_lyric_text(str(item.get("text", "")).strip())
            if not text:
                continue
            if text == UNKNOWN_LYRIC_TEXT:
                has_unknown = True
                continue
            if text == CHANT_LYRIC_TEXT:
                has_chant = True
                continue
            reliable_text_items.append(text)

        if reliable_text_items:
            return " ".join(reliable_text_items)
        if has_unknown:
            return UNKNOWN_LYRIC_TEXT
        if has_chant:
            return CHANT_LYRIC_TEXT
        return ""

    def _normalize_lyric_unit(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """
        功能说明：归一化单条歌词单元，统一时间与文本字段结构。
        参数说明：
        - item: 原始歌词单元字典。
        返回值：
        - dict[str, Any] | None: 归一化后的歌词单元；非法或纯标点文本返回 None。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：token_units 缺失时保留句级文本。
        """
        start_time = float(item.get("start_time", 0.0))
        end_time = max(start_time, float(item.get("end_time", item.get("start_time", 0.0))))
        text = self._clean_lyric_text(str(item.get("text", "")).strip())
        if not text:
            return None
        normalized_item = {
            "segment_id": str(item.get("segment_id", "")).strip(),
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
            "confidence": float(item.get("confidence", 0.0)),
        }
        raw_token_units = item.get("token_units", [])
        if isinstance(raw_token_units, list):
            token_units: list[dict[str, Any]] = []
            for token_item in raw_token_units:
                if not isinstance(token_item, dict):
                    continue
                token_text = self._clean_lyric_text(str(token_item.get("text", "")).strip())
                if not token_text:
                    continue
                token_start = float(token_item.get("start_time", 0.0))
                token_end = max(token_start, float(token_item.get("end_time", token_item.get("start_time", 0.0))))
                token_units.append(
                    {
                        "text": token_text,
                        "start_time": token_start,
                        "end_time": token_end,
                        "granularity": "word"
                        if str(token_item.get("granularity", "")).strip().lower() == "word"
                        else "char",
                    }
                )
            if token_units:
                normalized_item["token_units"] = token_units
        return normalized_item

    def _clean_lyric_text(self, text: str) -> str:
        """
        功能说明：清洗歌词文本，去除句首标点并过滤纯标点内容。
        参数说明：
        - text: 原始歌词文本。
        返回值：
        - str: 清洗后的歌词文本，若仅标点则返回空字符串。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：仅处理句首标点，句中与句尾标点保留。
        """
        cleaned_text = EDGE_PUNCTUATION_PATTERN.sub("", str(text).strip()).strip()
        if not cleaned_text:
            return ""
        if PUNCTUATION_ONLY_PATTERN.fullmatch(cleaned_text):
            return ""
        return cleaned_text


class LlmScriptGenerator(ScriptGenerator):
    """LLM 分镜生成占位实现（当前自动降级到 Mock）。"""

    def __init__(self, logger: logging.Logger) -> None:
        """初始化 LLM 占位生成器。"""
        self.logger = logger
        self._fallback = MockScriptGenerator()

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """当前阶段输出降级提示并使用 Mock 结果。"""
        self.logger.warning("LLM 分镜生成器尚未接入真实模型，已自动降级为 Mock 规则生成。")
        return self._fallback.generate(module_a_output=module_a_output)


def build_script_generator(mode: str, logger: logging.Logger) -> ScriptGenerator:
    """根据模式构建分镜生成器实例。"""
    mode_text = mode.lower().strip()
    if mode_text == "llm":
        return LlmScriptGenerator(logger=logger)
    if mode_text != "mock":
        logger.warning("未知分镜生成模式: %s，已降级为 mock。", mode)
    return MockScriptGenerator()
