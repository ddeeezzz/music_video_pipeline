"""
文件用途：提供模块 B 的分镜生成器抽象与实现。
核心流程：根据配置选择 mock/llm 生成器，输出统一的分镜 JSON 列表。
输入输出：输入 ModuleAOutput 字典，输出 ModuleBOutput 列表。
依赖说明：依赖标准库 abc/logging。
维护说明：真实 LLM 接入时仅替换 LlmScriptGenerator 内部逻辑。
"""

# 标准库：定义抽象基类
from abc import ABC, abstractmethod
# 标准库：用于日志输出
import logging
# 标准库：用于类型提示
from typing import Any


class ScriptGenerator(ABC):
    """
    功能说明：分镜生成器接口定义。
    参数说明：无。
    返回值：不适用。
    异常说明：子类应自行抛出实现相关异常。
    边界条件：输出必须满足 ModuleBOutput 最低契约。
    """

    @abstractmethod
    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：根据模块 A 输出生成分镜数组。
        参数说明：
        - module_a_output: 模块 A 输出字典。
        返回值：
        - list[dict[str, Any]]: 分镜数组。
        异常说明：由子类实现决定。
        边界条件：时间区间需满足 end_time > start_time。
        """
        raise NotImplementedError


class MockScriptGenerator(ScriptGenerator):
    """
    功能说明：规则化分镜生成器（MVP 默认实现）。
    参数说明：无。
    返回值：不适用。
    异常说明：输入缺失核心字段时抛 KeyError。
    边界条件：当能量数组长度不足时使用默认值填充。
    """

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：使用规则模板生成可用分镜。
        参数说明：
        - module_a_output: 模块 A 输出字典。
        返回值：
        - list[dict[str, Any]]: 规则生成的分镜数组。
        异常说明：输入字段缺失时抛 KeyError。
        边界条件：最低保证返回至少一条分镜。
        """
        segments = module_a_output["segments"]
        energy_features = module_a_output["energy_features"]

        shots: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            energy = energy_features[min(index, len(energy_features) - 1)] if energy_features else {"energy_level": "mid", "trend": "flat"}
            energy_level = str(energy.get("energy_level", "mid"))
            trend = str(energy.get("trend", "flat"))
            camera_motion = self._choose_camera_motion(energy_level=energy_level, trend=trend)
            transition = "hard_cut" if energy_level == "high" else "crossfade"
            shots.append(
                {
                    "shot_id": f"shot_{index + 1:03d}",
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "scene_desc": f"段落 {segment['label']} 的视觉表达，强调节奏同步",
                    "image_prompt": f"Cinematic scene, {segment['label']} mood, rhythm aligned, high detail",
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
        """
        功能说明：根据能量等级与变化趋势选取运镜策略。
        参数说明：
        - energy_level: 能量等级（low/mid/high）。
        - trend: 变化趋势（up/down/flat）。
        返回值：
        - str: 运镜类型（slow_pan/zoom_in/shake/push_pull/none）。
        异常说明：无。
        边界条件：低能量且平稳/下降段落返回 none，未知输入时回退为 slow_pan。
        """
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
    """
    功能说明：LLM 分镜生成占位实现（后续替换点）。
    参数说明：
    - logger: 日志记录器。
    返回值：不适用。
    异常说明：无。
    边界条件：当前版本会自动降级到 Mock。
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        功能说明：初始化 LLM 占位生成器。
        参数说明：
        - logger: 日志对象，用于输出降级提示。
        返回值：无。
        异常说明：无。
        边界条件：无。
        """
        self.logger = logger
        self._fallback = MockScriptGenerator()

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：当前阶段输出降级提示并使用 Mock 结果。
        参数说明：
        - module_a_output: 模块 A 输出字典。
        返回值：
        - list[dict[str, Any]]: 分镜数组。
        异常说明：由 Mock 生成器决定。
        边界条件：后续接入真模型时保持输出契约不变。
        """
        self.logger.warning("LLM 分镜生成器尚未接入真实模型，已自动降级为 Mock 规则生成。")
        return self._fallback.generate(module_a_output=module_a_output)


def build_script_generator(mode: str, logger: logging.Logger) -> ScriptGenerator:
    """
    功能说明：根据模式构建分镜生成器实例。
    参数说明：
    - mode: 生成模式（mock/llm）。
    - logger: 日志对象。
    返回值：
    - ScriptGenerator: 对应的生成器实例。
    异常说明：无。
    边界条件：未知模式将回退到 Mock。
    """
    mode_text = mode.lower().strip()
    if mode_text == "llm":
        return LlmScriptGenerator(logger=logger)
    if mode_text != "mock":
        logger.warning("未知分镜生成模式: %s，已降级为 mock。", mode)
    return MockScriptGenerator()
