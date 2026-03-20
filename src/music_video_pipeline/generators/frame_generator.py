"""
文件用途：提供模块 C 的关键帧生成器抽象与实现。
核心流程：根据分镜列表生成占位关键帧，预留 diffusion 替换接口。
输入输出：输入分镜数组与输出目录，输出帧清单。
依赖说明：依赖标准库 abc/logging/pathlib，以及第三方 Pillow。
维护说明：真实扩散模型接入时仅替换 DiffusionFrameGenerator。
"""

# 标准库：定义抽象基类
from abc import ABC, abstractmethod
# 标准库：用于日志输出
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于创建占位图像与加载字体
from PIL import Image, ImageDraw, ImageFont


class FrameGenerator(ABC):
    """
    功能说明：关键帧生成器接口定义。
    参数说明：无。
    返回值：不适用。
    异常说明：子类可抛出实现相关异常。
    边界条件：输出帧清单需要包含路径和时长。
    """

    @abstractmethod
    def generate(self, shots: list[dict[str, Any]], output_dir: Path, width: int, height: int) -> list[dict[str, Any]]:
        """
        功能说明：根据分镜列表生成关键帧。
        参数说明：
        - shots: 模块 B 输出的分镜数组。
        - output_dir: 图像输出目录。
        - width: 输出宽度。
        - height: 输出高度。
        返回值：
        - list[dict[str, Any]]: 帧清单。
        异常说明：由子类决定。
        边界条件：width/height 建议为偶数。
        """
        raise NotImplementedError


class MockFrameGenerator(FrameGenerator):
    """
    功能说明：生成占位关键帧（MVP 默认实现）。
    参数说明：无。
    返回值：不适用。
    异常说明：磁盘写入失败时抛出 OSError。
    边界条件：若分镜无效时自动使用最小时长 0.5 秒。
    """

    def generate(self, shots: list[dict[str, Any]], output_dir: Path, width: int, height: int) -> list[dict[str, Any]]:
        """
        功能说明：为每个分镜生成一张带文字的占位图。
        参数说明：
        - shots: 分镜数组。
        - output_dir: 帧输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        返回值：
        - list[dict[str, Any]]: 每个分镜对应的帧信息。
        异常说明：目录不可写时抛 OSError。
        边界条件：时长 <= 0 时自动修正为 0.5 秒。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        font_obj = _load_chinese_font(size=36)
        frame_items: list[dict[str, Any]] = []
        for index, shot in enumerate(shots):
            start_time = float(shot["start_time"])
            end_time = float(shot["end_time"])
            duration = round(max(0.5, end_time - start_time), 3)

            image_path = output_dir / f"frame_{index + 1:03d}.png"
            self._build_placeholder_image(image_path=image_path, width=width, height=height, shot=shot, font_obj=font_obj)

            frame_items.append(
                {
                    "shot_id": str(shot["shot_id"]),
                    "frame_path": str(image_path),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                }
            )
        return frame_items

    def _build_placeholder_image(
        self,
        image_path: Path,
        width: int,
        height: int,
        shot: dict[str, Any],
        font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> None:
        """
        功能说明：生成单张占位图并写入分镜信息文字。
        参数说明：
        - image_path: 输出图像路径。
        - width: 图像宽度。
        - height: 图像高度。
        - shot: 当前分镜字典。
        - font_obj: 预先加载的字体对象（中文优先）。
        返回值：无。
        异常说明：文件写入失败时抛 OSError。
        边界条件：文本较长时会自然截断显示，不影响流程。
        """
        background_color = (26, 51, 77)
        image = Image.new(mode="RGB", size=(width, height), color=background_color)
        drawer = ImageDraw.Draw(image)
        drawer.text((40, 40), f"shot_id: {shot['shot_id']}", fill=(255, 255, 255), font=font_obj)
        drawer.text((40, 100), f"time: {shot['start_time']:.2f}-{shot['end_time']:.2f}s", fill=(255, 255, 255), font=font_obj)
        drawer.text((40, 160), f"scene: {shot['scene_desc']}", fill=(255, 255, 255), font=font_obj)
        drawer.text((40, 220), f"motion: {shot['camera_motion']}", fill=(255, 255, 255), font=font_obj)
        drawer.text((40, 280), f"transition: {shot['transition']}", fill=(255, 255, 255), font=font_obj)
        image.save(image_path)


class DiffusionFrameGenerator(FrameGenerator):
    """
    功能说明：扩散模型关键帧生成占位实现（后续替换点）。
    参数说明：
    - logger: 日志对象。
    返回值：不适用。
    异常说明：无。
    边界条件：当前版本降级到 Mock，保证链路可跑通。
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        功能说明：初始化扩散生成器占位实现。
        参数说明：
        - logger: 日志对象。
        返回值：无。
        异常说明：无。
        边界条件：无。
        """
        self.logger = logger
        self._fallback = MockFrameGenerator()

    def generate(self, shots: list[dict[str, Any]], output_dir: Path, width: int, height: int) -> list[dict[str, Any]]:
        """
        功能说明：输出降级提示并调用 Mock 生成。
        参数说明：
        - shots: 分镜数组。
        - output_dir: 输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        返回值：
        - list[dict[str, Any]]: 帧清单。
        异常说明：由 Mock 生成器决定。
        边界条件：后续真实模型接入时保持输出清单结构不变。
        """
        self.logger.warning("Diffusion 关键帧生成器尚未接入真实模型，已自动降级为 Mock 占位图生成。")
        return self._fallback.generate(shots=shots, output_dir=output_dir, width=width, height=height)


def build_frame_generator(mode: str, logger: logging.Logger) -> FrameGenerator:
    """
    功能说明：根据模式构建关键帧生成器实例。
    参数说明：
    - mode: 生成模式（mock/diffusion）。
    - logger: 日志对象。
    返回值：
    - FrameGenerator: 对应生成器实例。
    异常说明：无。
    边界条件：未知模式将降级到 Mock。
    """
    mode_text = mode.lower().strip()
    if mode_text == "diffusion":
        return DiffusionFrameGenerator(logger=logger)
    if mode_text != "mock":
        logger.warning("未知关键帧生成模式: %s，已降级为 mock。", mode)
    return MockFrameGenerator()


def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    功能说明：加载可显示中文的字体，若不可用则回退到默认字体。
    参数说明：
    - size: 字号大小。
    返回值：
    - ImageFont.FreeTypeFont | ImageFont.ImageFont: 可用字体对象。
    异常说明：内部捕获字体加载异常并继续尝试下一个候选。
    边界条件：所有候选失败时返回 PIL 默认字体（可能不支持中文）。
    """
    font_candidates = [
        Path("resources/fonts/msyh.ttc"),
        Path("resources/fonts/simhei.ttf"),
        Path("resources/fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_candidates:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()
