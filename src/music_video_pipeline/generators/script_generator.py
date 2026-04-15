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
# 标准库：用于路径处理
from pathlib import Path
# 标准库：正则处理
import re
# 标准库：类型提示
from typing import Any

# 项目内模块：模块B配置类型
from music_video_pipeline.config import ModuleBConfig

UNKNOWN_LYRIC_TEXT = "[未识别歌词]"
CHANT_LYRIC_TEXT = "吟唱"
# 常量：用于清理歌词文本句首标点（含中英文常见符号）的正则。
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+")
# 常量：用于识别“纯标点文本”，避免将其作为可展示歌词。
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+$")


def generate_module_b_prompts(
    logger: logging.Logger,
    llm_config: Any,
    llm_input_payload: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    """
    功能说明：调用模块B真实LLM三字段生成（延迟导入，规避循环依赖）。
    参数说明：
    - logger: 日志对象。
    - llm_config: 模块B LLM配置对象。
    - llm_input_payload: 单段分镜输入上下文字典。
    - project_root: 项目根目录，用于解析密钥相对路径。
    返回值：
    - dict[str, str]: scene_desc/keyframe_prompt/video_prompt。
    异常说明：由模块B llm_generator 统一抛出。
    边界条件：本函数仅做转发，不做业务字段校验。
    """
    from music_video_pipeline.modules.module_b.llm_generator import (
        generate_module_b_prompts as _generate_module_b_prompts,
    )

    return _generate_module_b_prompts(
        logger=logger,
        llm_config=llm_config,
        llm_input_payload=llm_input_payload,
        project_root=project_root,
    )


class ScriptGenerator(ABC):
    """分镜生成器接口定义。"""

    @abstractmethod
    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """根据模块A输出生成分镜数组。"""
        raise NotImplementedError

    @abstractmethod
    def generate_one(
        self,
        module_a_output: dict[str, Any],
        segment: dict[str, Any],
        segment_index: int,
    ) -> dict[str, Any]:
        """根据单个segment生成单个分镜。"""
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
        shots: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            shots.append(
                self.generate_one(
                    module_a_output=module_a_output,
                    segment=segment,
                    segment_index=index,
                )
            )

        if not shots:
            shots.append(
                {
                    "shot_id": "shot_001",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "scene_desc": "默认场景占位",
                    "keyframe_prompt_zh": "电影感关键帧占位场景，中性光线，居中构图",
                    "keyframe_prompt": "Cinematic keyframe placeholder scene, neutral lighting, centered composition",
                    "keyframe_prompt_en": "Cinematic keyframe placeholder scene, neutral lighting, centered composition",
                    "video_prompt_zh": "电影感视频占位场景，轻微运动，稳定镜头，中性氛围",
                    "video_prompt": "Cinematic video scene, subtle motion, stable camera movement, neutral atmosphere",
                    "video_prompt_en": "Cinematic video scene, subtle motion, stable camera movement, neutral atmosphere",
                    "camera_motion": "slow_pan",
                    "transition": "crossfade",
                    "constraints": {"must_keep_style": True, "must_align_to_beat": True},
                    "lyric_text": "",
                    "lyric_units": [],
                }
            )
        return shots

    def generate_one(
        self,
        module_a_output: dict[str, Any],
        segment: dict[str, Any],
        segment_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：按单个segment生成一个分镜条目。
        参数说明：
        - module_a_output: 模块A输出字典。
        - segment: 当前segment对象。
        - segment_index: 当前segment顺序索引（0基）。
        返回值：
        - dict[str, Any]: 单个分镜字典。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：歌词挂载策略保持 segment_id 优先+时间重叠兜底。
        """
        big_segments = module_a_output.get("big_segments", [])
        energy_features = module_a_output.get("energy_features", [])
        lyric_units = module_a_output.get("lyric_units", [])

        big_segment_label_map = {
            str(item.get("segment_id", "")): str(item.get("label", "unknown")) for item in big_segments if isinstance(item, dict)
        }
        lyric_units_map, lyric_units_sorted = self._build_lyric_indexes(lyric_units=lyric_units)

        energy = (
            energy_features[min(segment_index, len(energy_features) - 1)]
            if isinstance(energy_features, list) and energy_features
            else {"energy_level": "mid", "trend": "flat"}
        )
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

        return {
            "shot_id": f"shot_{segment_index + 1:03d}",
            "start_time": segment_start,
            "end_time": segment_end,
            "scene_desc": f"大段落{big_label}中的小段落{small_label}，强调节奏同步",
            "keyframe_prompt_zh": f"电影感关键帧，{big_label}，{small_label}，节奏对齐，细节清晰，主体明确",
            "keyframe_prompt": f"Cinematic keyframe, {big_label}, {small_label}, rhythm aligned, high detail, sharp focus",
            "keyframe_prompt_en": f"Cinematic keyframe, {big_label}, {small_label}, rhythm aligned, high detail, sharp focus",
            "video_prompt_zh": f"电影感视频提示词，{big_label}，{small_label}，节奏运动，镜头平滑，氛围鲜明",
            "video_prompt": f"Cinematic video prompt, {big_label}, {small_label}, rhythmic movement, smooth camera motion, vivid atmosphere",
            "video_prompt_en": f"Cinematic video prompt, {big_label}, {small_label}, rhythmic movement, smooth camera motion, vivid atmosphere",
            "camera_motion": camera_motion,
            "transition": transition,
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "lyric_text": lyric_text,
            "lyric_units": shot_lyric_units,
        }

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


class LlmScriptGenerator(MockScriptGenerator):
    """LLM 分镜生成真实实现（单次调用输出 scene_desc 与 keyframe/video 中英文提示词）。"""

    def __init__(self, logger: logging.Logger, module_b_config: ModuleBConfig | None = None) -> None:
        """
        功能说明：初始化 LLM 分镜生成器。
        参数说明：
        - logger: 日志对象。
        - module_b_config: 模块B配置对象，含 llm 子配置。
        返回值：无。
        异常说明：无。
        边界条件：未传配置时使用 ModuleBConfig 默认值。
        """
        super().__init__()
        self.logger = logger
        self.module_b_config = module_b_config or ModuleBConfig()
        # 常量：项目根目录，用于解析 llm.api_key_file 相对路径。
        self.project_root = Path(__file__).resolve().parents[3]

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：批量生成分镜数组（沿用父类迭代流程）。
        参数说明：
        - module_a_output: 模块A输出字典。
        返回值：
        - list[dict[str, Any]]: 模块B分镜数组。
        异常说明：LLM请求或解析失败时抛 RuntimeError，由上层重试。
        边界条件：单元执行失败将由模块B执行器记录为 failed。
        """
        return super().generate(module_a_output=module_a_output)

    def generate_one(
        self,
        module_a_output: dict[str, Any],
        segment: dict[str, Any],
        segment_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：执行单段LLM分镜生成，并回填到标准shot结构。
        参数说明：
        - module_a_output: 模块A输出字典。
        - segment: 当前segment对象。
        - segment_index: 当前segment顺序索引（0基）。
        返回值：
        - dict[str, Any]: 单个分镜字典。
        异常说明：LLM请求失败或返回不合法时抛 RuntimeError。
        边界条件：camera_motion/transition 等节奏字段保持规则生成，不由LLM回填。
        """
        baseline_shot = super().generate_one(
            module_a_output=module_a_output,
            segment=segment,
            segment_index=segment_index,
        )

        big_segments = module_a_output.get("big_segments", [])
        big_segment_label_map = {
            str(item.get("segment_id", "")): str(item.get("label", "unknown"))
            for item in big_segments
            if isinstance(item, dict)
        }
        energy_features = module_a_output.get("energy_features", [])
        energy = (
            energy_features[min(segment_index, len(energy_features) - 1)]
            if isinstance(energy_features, list) and energy_features
            else {"energy_level": "mid", "trend": "flat"}
        )
        segment_label = str(segment.get("label", "unknown"))
        big_segment_label = big_segment_label_map.get(str(segment.get("big_segment_id", "")), segment_label)
        llm_input_payload = {
            "segment_id": str(segment.get("segment_id", "")),
            "start_time": float(segment.get("start_time", 0.0)),
            "end_time": float(segment.get("end_time", segment.get("start_time", 0.0))),
            "segment_label": segment_label,
            "big_segment_label": big_segment_label,
            "energy_level": str(energy.get("energy_level", "mid")),
            "trend": str(energy.get("trend", "flat")),
            "camera_motion_rule": str(baseline_shot.get("camera_motion", "none")),
            "transition_rule": str(baseline_shot.get("transition", "crossfade")),
            "lyric_text": str(baseline_shot.get("lyric_text", "")),
            "lyric_units": baseline_shot.get("lyric_units", []),
        }
        try:
            # 延迟导入：避免 generators <-> modules 包初始化阶段的循环导入。
            from music_video_pipeline.modules.module_b.llm_generator import ModuleBLlmGenerationError

            llm_result = generate_module_b_prompts(
                logger=self.logger,
                llm_config=self.module_b_config.llm,
                llm_input_payload=llm_input_payload,
                project_root=self.project_root,
            )
        except ModuleBLlmGenerationError as error:
            raise RuntimeError(f"模块B LLM单元生成失败，segment_id={llm_input_payload['segment_id']}，错误={error}") from error

        baseline_shot["scene_desc"] = llm_result["scene_desc"]
        baseline_shot["keyframe_prompt_zh"] = llm_result["keyframe_prompt_zh"]
        baseline_shot["keyframe_prompt_en"] = llm_result["keyframe_prompt_en"]
        baseline_shot["video_prompt_zh"] = llm_result["video_prompt_zh"]
        baseline_shot["video_prompt_en"] = llm_result["video_prompt_en"]
        # 兼容字段：下游当前消费 keyframe_prompt/video_prompt，默认写英文版本。
        baseline_shot["keyframe_prompt"] = llm_result["keyframe_prompt"]
        baseline_shot["video_prompt"] = llm_result["video_prompt"]
        return baseline_shot


def build_script_generator(mode: str, logger: logging.Logger, module_b_config: ModuleBConfig | None = None) -> ScriptGenerator:
    """根据模式构建分镜生成器实例。"""
    mode_text = mode.lower().strip()
    if mode_text == "llm":
        return LlmScriptGenerator(logger=logger, module_b_config=module_b_config)
    if mode_text != "mock":
        logger.warning("未知分镜生成模式: %s，已降级为 mock。", mode)
    return MockScriptGenerator()
