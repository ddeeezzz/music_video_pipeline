"""
文件用途：验证模块C占位图的中文字体与自适应排版行为。
核心流程：构造中文分镜数据，检查图片生成、字体优先加载、场景文本换行截断。
输入输出：输入临时目录与伪造分镜，输出断言结果。
依赖说明：依赖 pytest、Pillow 与项目内 frame_generator。
维护说明：若占位图布局策略调整，需要同步更新断言。
"""

# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于图像绘制对象构建
from PIL import Image, ImageDraw

# 项目内模块：占位图生成器实现与内部排版工具
from music_video_pipeline.generators.frame_generator import (
    MockFrameGenerator,
    _load_chinese_font,
    _measure_text_pixel_width,
    _wrap_text_by_pixel_width,
)


def test_mock_frame_generator_should_render_chinese_scene_text(tmp_path: Path) -> None:
    """
    功能说明：验证中文场景文案可生成占位图且不抛异常。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证生成成功与图片尺寸，不做像素级 OCR 检查。
    """
    generator = MockFrameGenerator()
    shots = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 2.5,
            "scene_desc": "夜色城市街景，霓虹灯与雨幕交织，人物缓慢前行",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
        }
    ]

    frame_items = generator.generate(shots=shots, output_dir=tmp_path / "frames", width=960, height=540)
    assert len(frame_items) == 1

    frame_path = Path(frame_items[0]["frame_path"])
    assert frame_path.exists()

    image = Image.open(frame_path)
    assert image.size == (960, 540)


def test_load_chinese_font_should_prioritize_repo_bundled_font() -> None:
    """
    功能说明：验证字体加载优先使用仓库内置字体文件。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当仓库字体文件存在时必须命中该路径。
    """
    _, font_source = _load_chinese_font(size=28)
    assert font_source.endswith("resources/fonts/NotoSansCJKsc-Regular.otf")


def test_wrap_text_by_pixel_width_should_clip_to_max_lines_with_ellipsis() -> None:
    """
    功能说明：验证超长中文文本会按像素宽度换行并在超限时追加省略号。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：逐字换行需兼容中文无空格文本。
    """
    font_obj, _ = _load_chinese_font(size=24)
    image = Image.new(mode="RGB", size=(500, 300), color=(0, 0, 0))
    drawer = ImageDraw.Draw(image)
    long_scene_text = "场景：" + "这是一个用于测试中文自动换行与截断行为的超长描述文本" * 6

    max_width = 260
    lines = _wrap_text_by_pixel_width(
        drawer=drawer,
        text=long_scene_text,
        font_obj=font_obj,
        max_width=max_width,
        max_lines=3,
    )

    assert len(lines) == 3
    assert lines[-1].endswith("...")
    assert all(_measure_text_pixel_width(drawer, line, font_obj) <= max_width for line in lines)

