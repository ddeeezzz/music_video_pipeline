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
# 标准库：用于正则处理
import re
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于创建占位图像与加载字体
from PIL import Image, ImageDraw, ImageFont

UNKNOWN_LYRIC_TEXT = "[未识别歌词]"
CHANT_LYRIC_TEXT = "吟唱"
INSTRUMENTAL_LYRIC_NOTE = "（说明：根据音源分离后的能量检测，此处为器乐段，但 Fun-ASR 识别到了歌词）"
# 常量：用于清理歌词文本句首标点（含中英文常见符号）的正则。
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+")
# 常量：用于识别“纯标点文本”，避免将其作为可展示歌词。
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+$")


class FrameGenerator(ABC):
    """
    功能说明：关键帧生成器接口定义。
    参数说明：无。
    返回值：不适用。
    异常说明：子类可抛出实现相关异常。
    边界条件：输出单元结构需包含 frame_path 与时间字段。
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
        功能说明：根据单个分镜生成关键帧。
        参数说明：
        - shot: 模块 B 输出的单个分镜对象。
        - output_dir: 图像输出目录。
        - width: 输出宽度。
        - height: 输出高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个 frame_item。
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

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        功能说明：初始化占位关键帧生成器。
        参数说明：
        - logger: 可选日志对象，用于输出字体降级告警。
        返回值：无。
        异常说明：无。
        边界条件：logger 为空时仅跳过告警输出。
        """
        self.logger = logger
        # 标记：避免中文字体缺失告警被重复刷屏。
        self._font_warning_emitted = False

    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：为单个分镜生成一张带文字的占位图。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 帧输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个分镜对应的 frame_item。
        异常说明：目录不可写时抛 OSError。
        边界条件：时长 <= 0 时自动修正为 0.5 秒。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        title_font_size = max(28, int(width * 0.035))
        body_font_size = max(24, int(width * 0.028))
        title_font_obj, title_font_source = _load_chinese_font(size=title_font_size)
        body_font_obj, body_font_source = _load_chinese_font(size=body_font_size)
        if self.logger and ("pil_default" in {title_font_source, body_font_source}) and (not self._font_warning_emitted):
            self.logger.warning("模块C未加载到可用中文字体，当前使用默认字体，中文可能无法正常显示。")
            self._font_warning_emitted = True

        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)

        image_path = output_dir / f"frame_{shot_index + 1:03d}.png"
        self._build_placeholder_image(
            image_path=image_path,
            width=width,
            height=height,
            shot=shot,
            title_font_obj=title_font_obj,
            body_font_obj=body_font_obj,
            title_font_size=title_font_size,
            body_font_size=body_font_size,
        )
        return {
            "shot_id": str(shot["shot_id"]),
            "frame_path": str(image_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "scene_desc": str(shot.get("scene_desc", "")),
            "keyframe_prompt_zh": str(shot.get("keyframe_prompt_zh", "")),
            "keyframe_prompt_en": str(shot.get("keyframe_prompt_en", "")),
            "keyframe_prompt": str(shot.get("keyframe_prompt", "")),
            "video_prompt_zh": str(shot.get("video_prompt_zh", "")),
            "video_prompt_en": str(shot.get("video_prompt_en", "")),
            "video_prompt": str(shot.get("video_prompt", "")),
        }

    def _build_placeholder_image(
        self,
        image_path: Path,
        width: int,
        height: int,
        shot: dict[str, Any],
        title_font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        body_font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        title_font_size: int,
        body_font_size: int,
    ) -> None:
        """
        功能说明：生成单张占位图并写入分镜信息文字。
        参数说明：
        - image_path: 输出图像路径。
        - width: 图像宽度。
        - height: 图像高度。
        - shot: 当前分镜字典。
        - title_font_obj: 标题字体对象。
        - body_font_obj: 正文字体对象。
        - title_font_size: 标题字号（像素）。
        - body_font_size: 正文字号（像素）。
        返回值：无。
        异常说明：文件写入失败时抛 OSError。
        边界条件：段落/scene/lyrics 文本按像素宽度自动换行，超限时追加省略号。
        """
        background_color = (26, 51, 77)
        image = Image.new(mode="RGB", size=(width, height), color=background_color)
        drawer = ImageDraw.Draw(image)
        text_color = (255, 255, 255)
        margin = max(24, int(width * 0.04))
        line_gap = max(8, int(body_font_size * 0.45))
        line_height = body_font_size + line_gap
        max_text_width = width - margin * 2
        cursor_y = margin

        drawer.text((margin, cursor_y), f"镜头ID：{shot['shot_id']}", fill=text_color, font=title_font_obj)
        cursor_y += title_font_size + line_gap

        drawer.text(
            (margin, cursor_y),
            f"时间：{float(shot['start_time']):.2f}-{float(shot['end_time']):.2f}s",
            fill=text_color,
            font=body_font_obj,
        )
        cursor_y += line_height

        big_segment_text = _extract_big_segment_display_for_shot(shot=shot)
        big_segment_line = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=f"大段落：{big_segment_text}",
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=1,
        )
        drawer.text((margin, cursor_y), big_segment_line[0], fill=text_color, font=body_font_obj)
        cursor_y += line_height

        role_text = _extract_audio_role_display_for_shot(shot=shot)
        role_line = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=f"段落类型：{role_text}",
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=1,
        )
        drawer.text((margin, cursor_y), role_line[0], fill=text_color, font=body_font_obj)
        cursor_y += line_height

        footer_rows = 2
        footer_reserved_height = footer_rows * body_font_size + (footer_rows - 1) * line_gap
        lyric_min_reserved_height = line_height
        scene_available_height = max(
            line_height,
            height - margin - footer_reserved_height - lyric_min_reserved_height - cursor_y,
        )
        max_scene_lines = max(1, min(3, scene_available_height // line_height))

        scene_text = f"场景：{str(shot['scene_desc'])}"
        scene_lines = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=scene_text,
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=max_scene_lines,
        )
        for line in scene_lines:
            drawer.text((margin, cursor_y), line, fill=text_color, font=body_font_obj)
            cursor_y += line_height

        lyric_text = _extract_lyric_text_for_shot(shot=shot)
        lyric_render_text = f"歌词：{lyric_text}" if lyric_text else "歌词：<无>"
        footer_y = height - margin - footer_reserved_height
        lyric_available_height = max(
            line_height,
            footer_y - cursor_y - line_gap,
        )
        max_lyric_lines = max(1, min(4, lyric_available_height // line_height))
        lyric_lines = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=lyric_render_text,
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=max_lyric_lines,
        )
        for line in lyric_lines:
            drawer.text((margin, cursor_y), line, fill=text_color, font=body_font_obj)
            cursor_y += line_height

        drawer.text((margin, footer_y), f"运镜：{shot['camera_motion']}", fill=text_color, font=body_font_obj)
        drawer.text((margin, footer_y + line_height), f"转场：{shot['transition']}", fill=text_color, font=body_font_obj)
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
        self._fallback = MockFrameGenerator(logger=logger)

    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：输出降级提示并调用 Mock 生成。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个 frame_item。
        异常说明：由 Mock 生成器决定。
        边界条件：后续真实模型接入时保持输出清单结构不变。
        """
        self.logger.warning("Diffusion 关键帧生成器尚未接入真实模型，已自动降级为 Mock 占位图生成。")
        return self._fallback.generate_one(
            shot=shot,
            output_dir=output_dir,
            width=width,
            height=height,
            shot_index=shot_index,
        )


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
    return MockFrameGenerator(logger=logger)


def _load_chinese_font(size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    """
    功能说明：加载可显示中文的字体，若不可用则回退到默认字体。
    参数说明：
    - size: 字号大小。
    返回值：
    - tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]: (字体对象, 字体来源标识)。
    异常说明：内部捕获字体加载异常并继续尝试下一个候选。
    边界条件：所有候选失败时返回 PIL 默认字体（可能不支持中文）。
    """
    font_candidates = _resolve_chinese_font_candidates()
    for font_path in font_candidates:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size), str(font_path)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default(), "pil_default"


def _resolve_chinese_font_candidates() -> list[Path]:
    """
    功能说明：返回中文字体候选列表（按优先级排序）。
    参数说明：无。
    返回值：
    - list[Path]: 字体路径候选。
    异常说明：无。
    边界条件：优先仓库内置字体，其次 Linux/Windows 系统字体。
    """
    project_root = Path(__file__).resolve().parents[3]
    return [
        project_root / "resources" / "fonts" / "NotoSansCJKsc-Regular.otf",
        project_root / "resources" / "fonts" / "msyh.ttc",
        project_root / "resources" / "fonts" / "simhei.ttf",
        project_root / "resources" / "fonts" / "simsun.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]


def _measure_text_pixel_width(
    drawer: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    """
    功能说明：测量文本在指定字体下的像素宽度。
    参数说明：
    - drawer: Pillow 绘制对象。
    - text: 待测量文本。
    - font_obj: 字体对象。
    返回值：
    - int: 文本像素宽度。
    异常说明：无。
    边界条件：空文本宽度为 0。
    """
    if not text:
        return 0
    left, _, right, _ = drawer.textbbox((0, 0), text, font=font_obj)
    return max(0, right - left)


def _wrap_text_by_pixel_width(
    drawer: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """
    功能说明：按像素宽度逐字换行，并限制最大行数。
    参数说明：
    - drawer: Pillow 绘制对象。
    - text: 待换行文本。
    - font_obj: 字体对象。
    - max_width: 允许的最大像素宽度。
    - max_lines: 最大行数。
    返回值：
    - list[str]: 换行结果。
    异常说明：无。
    边界条件：超出最大行数时最后一行自动追加省略号。
    """
    if max_lines <= 0:
        return []
    if not text:
        return [""]

    lines: list[str] = []
    current_line = ""
    for char_text in text:
        trial_line = f"{current_line}{char_text}"
        if _measure_text_pixel_width(drawer, trial_line, font_obj) <= max_width:
            current_line = trial_line
            continue
        if current_line:
            lines.append(current_line)
            current_line = char_text
        else:
            lines.append(char_text)
            current_line = ""
    if current_line:
        lines.append(current_line)

    if len(lines) <= max_lines:
        return lines

    clipped_lines = lines[:max_lines]
    clipped_lines[-1] = _append_ellipsis_to_line(
        drawer=drawer,
        line_text=clipped_lines[-1],
        font_obj=font_obj,
        max_width=max_width,
    )
    return clipped_lines


def _append_ellipsis_to_line(
    drawer: ImageDraw.ImageDraw,
    line_text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    """
    功能说明：在行尾追加省略号，必要时裁剪文本保证宽度合法。
    参数说明：
    - drawer: Pillow 绘制对象。
    - line_text: 原始行文本。
    - font_obj: 字体对象。
    - max_width: 最大像素宽度。
    返回值：
    - str: 处理后的行文本。
    异常说明：无。
    边界条件：若宽度极小，至少返回 "..."。
    """
    suffix = "..."
    if _measure_text_pixel_width(drawer, f"{line_text}{suffix}", font_obj) <= max_width:
        return f"{line_text}{suffix}"

    trimmed_text = line_text
    while trimmed_text and _measure_text_pixel_width(drawer, f"{trimmed_text}{suffix}", font_obj) > max_width:
        trimmed_text = trimmed_text[:-1]
    return f"{trimmed_text}{suffix}" if trimmed_text else suffix


def _extract_lyric_text_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：从分镜中提取歌词展示文本，兼容新旧字段结构。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 可渲染的歌词文本；无歌词时返回空字符串。
    异常说明：无。
    边界条件：优先使用 lyric_text，缺失时尝试从 lyric_units 聚合。
    """
    lyric_text = _clean_lyric_render_text(str(shot.get("lyric_text", "")).strip())
    if lyric_text:
        return _append_instrumental_lyric_note_if_needed(shot=shot, lyric_text=lyric_text)

    lyric_units = shot.get("lyric_units", [])
    if not isinstance(lyric_units, list):
        return ""

    reliable_text_items: list[str] = []
    has_unknown = False
    has_chant = False
    for item in lyric_units:
        if not isinstance(item, dict):
            continue
        text = _clean_lyric_render_text(str(item.get("text", "")).strip())
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
        lyric_text_joined = " ".join(reliable_text_items)
        return _append_instrumental_lyric_note_if_needed(shot=shot, lyric_text=lyric_text_joined)
    if has_unknown:
        return UNKNOWN_LYRIC_TEXT
    if has_chant:
        return CHANT_LYRIC_TEXT
    return ""


def _append_instrumental_lyric_note_if_needed(shot: dict[str, Any], lyric_text: str) -> str:
    """
    功能说明：在“器乐段但有有效歌词”场景追加固定说明文案，避免误判导致吞字。
    参数说明：
    - shot: 分镜字典，读取 audio_role 字段。
    - lyric_text: 已提取且清洗后的歌词文本。
    返回值：
    - str: 可能追加说明文案后的歌词文本。
    异常说明：无。
    边界条件：未识别标记/吟唱标记不追加说明文案。
    """
    cleaned_text = str(lyric_text).strip()
    if not cleaned_text:
        return ""
    if cleaned_text in {UNKNOWN_LYRIC_TEXT, CHANT_LYRIC_TEXT}:
        return cleaned_text
    audio_role = str(shot.get("audio_role", "")).strip().lower()
    if audio_role != "instrumental":
        return cleaned_text
    if INSTRUMENTAL_LYRIC_NOTE in cleaned_text:
        return cleaned_text
    return f"{cleaned_text}{INSTRUMENTAL_LYRIC_NOTE}"


def _clean_lyric_render_text(text: str) -> str:
    """
    功能说明：清洗占位图歌词文本，避免句首标点或纯标点上屏。
    参数说明：
    - text: 原始歌词文本。
    返回值：
    - str: 清洗后的可渲染文本；若仅标点返回空字符串。
    异常说明：无。
    边界条件：仅移除句首标点，句中与句尾标点保留。
    """
    cleaned_text = EDGE_PUNCTUATION_PATTERN.sub("", str(text).strip()).strip()
    if not cleaned_text:
        return ""
    if PUNCTUATION_ONLY_PATTERN.fullmatch(cleaned_text):
        return ""
    return cleaned_text


def _extract_big_segment_display_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：提取分镜对应的大段落展示文本（标签+ID）。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 大段落展示字符串。
    异常说明：无。
    边界条件：缺失字段时返回“<未知>”。
    """
    big_label = str(shot.get("big_segment_label", "")).strip()
    big_id = str(shot.get("big_segment_id", "")).strip()
    if big_label and big_id:
        return f"{big_label} ({big_id})"
    if big_label:
        return big_label
    if big_id:
        return big_id
    return "<未知>"


def _extract_audio_role_display_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：提取分镜对应的段落类型展示文本。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 段落类型（器乐段/人声段/<未知>）。
    异常说明：无。
    边界条件：字段缺失或非法值时返回“<未知>”。
    """
    audio_role = str(shot.get("audio_role", "")).strip().lower()
    if audio_role == "instrumental":
        return "器乐段"
    if audio_role == "vocal":
        return "人声段"
    return "<未知>"
