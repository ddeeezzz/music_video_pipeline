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
# 标准库：类型提示
from typing import Any


class ScriptGenerator(ABC):
    """分镜生成器接口定义。"""

    @abstractmethod
    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """根据模块A输出生成分镜数组。"""
        raise NotImplementedError


class MockScriptGenerator(ScriptGenerator):
    """规则化分镜生成器（MVP 默认实现）。"""

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """按小段落生成分镜，并通过 big_segment_id 关联大段落语义。"""
        segments = module_a_output["segments"]
        big_segments = module_a_output.get("big_segments", [])
        energy_features = module_a_output["energy_features"]

        big_segment_label_map = {
            str(item.get("segment_id", "")): str(item.get("label", "unknown")) for item in big_segments
        }

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

            shots.append(
                {
                    "shot_id": f"shot_{index + 1:03d}",
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "scene_desc": f"大段落{big_label}中的小段落{small_label}，强调节奏同步",
                    "image_prompt": f"Cinematic scene, {big_label}, {small_label}, rhythm aligned, high detail",
                    "camera_motion": camera_motion,
                    "transition": transition,
                    "constraints": {"must_keep_style": True, "must_align_to_beat": True},
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
                }
            )
        return shots

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
