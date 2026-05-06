"""
文件用途：提供模块 C 的关键帧生成器抽象与 ComfyUI 唯一工厂入口。
核心流程：定义统一生成接口 -> 校验后端为 comfyui -> 构建 ComfyUIFrameGenerator。
输入输出：输入生成参数，输出符合模块 D 双锚点契约的 frame_item。
依赖说明：依赖标准库 abc/logging/pathlib/typing 与项目内 ComfyUI 关键帧生成器。
维护说明：模块 C 已彻底收口为 ComfyUI 常驻服务路径；不再保留 mock 或本地 diffusers 生成实现。
"""

# 标准库：定义抽象基类。
from abc import ABC, abstractmethod
# 标准库：用于日志输出。
import logging
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any


class FrameGenerator(ABC):
    """
    功能说明：定义模块 C 单镜头关键帧生成器抽象接口。
    参数说明：不适用。
    返回值：不适用。
    异常说明：不适用。
    边界条件：实现类必须返回符合双关键帧契约的 frame_item。
    """

    @abstractmethod
    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：生成单个 shot 的双关键帧结果。
        参数说明：
        - shot: 模块 B 单元产物字典。
        - output_dir: 输出目录。
        - width: 输出宽度。
        - height: 输出高度。
        - shot_index: 镜头索引（0 基）。
        返回值：
        - dict[str, Any]: 模块 C 单元 frame_item。
        异常说明：由具体实现抛出运行异常。
        边界条件：必须显式返回 frame_path_start/frame_path_end/control_frame_paths。
        """

    def prewarm(self) -> None:
        """
        功能说明：执行批量生成前的显式预热或探活检查。
        参数说明：无。
        返回值：无。
        异常说明：由具体实现按需抛出运行异常。
        边界条件：默认实现为空操作，便于调用方统一在执行前触发。
        """
        return None


def build_keyframe_generator(mode: str, logger: logging.Logger, app_config: Any | None = None) -> FrameGenerator:
    """
    功能说明：构建模块 C 的关键帧生成器实例。
    参数说明：
    - mode: 渲染后端名称；当前仅允许 comfyui。
    - logger: 日志对象。
    - app_config: 应用配置对象；ComfyUI 模式必填。
    返回值：
    - FrameGenerator: ComfyUI 关键帧生成器实例。
    异常说明：
    - RuntimeError: mode 非 comfyui 或 app_config 缺失时抛出。
    边界条件：模块 C 不再提供任何本地模型后端回退。
    """
    mode_text = str(mode).strip().lower()
    if mode_text != "comfyui":
        raise RuntimeError(
            "模块C关键帧生成器构建失败：当前仅支持 comfyui，"
            f"收到 mode={mode_text or '<empty>'}。"
        )
    if app_config is None:
        raise RuntimeError("模块C ComfyUI 生成器构建失败：缺少 app_config。")

    from music_video_pipeline.generators.comfyui_frame_generator import ComfyUIFrameGenerator

    logger.info("模块C关键帧生成器已切换为 ComfyUI 唯一路径。")
    return ComfyUIFrameGenerator(app_config=app_config, logger=logger)
