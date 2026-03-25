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
    _extract_audio_role_display_for_shot,
    _extract_big_segment_display_for_shot,
    _extract_lyric_text_for_shot,
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


def test_mock_frame_generator_should_render_lyrics_text_with_new_fields(tmp_path: Path) -> None:
    """
    功能说明：验证分镜包含歌词扩展字段时，占位图可正常生成。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：歌词超长时应通过自动换行与截断处理。
    """
    generator = MockFrameGenerator()
    shots = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 3.0,
            "scene_desc": "雨夜下的城市街口，镜头缓慢推进",
            "camera_motion": "zoom_in",
            "transition": "crossfade",
            "lyric_text": "这是很长的一句歌词用于测试自动换行与布局保护 这是第二句歌词继续测试",
            "lyric_units": [
                {"start_time": 0.1, "end_time": 1.2, "text": "这是很长的一句歌词用于测试自动换行与布局保护", "confidence": 0.9},
                {"start_time": 1.3, "end_time": 2.7, "text": "这是第二句歌词继续测试", "confidence": 0.85},
            ],
            "big_segment_id": "big_003",
            "big_segment_label": "chorus",
            "segment_label": "chorus",
            "audio_role": "vocal",
        }
    ]

    frame_items = generator.generate(shots=shots, output_dir=tmp_path / "frames", width=960, height=540)
    assert len(frame_items) == 1
    assert Path(frame_items[0]["frame_path"]).exists()


def test_extract_lyric_text_for_shot_should_fallback_to_lyric_units() -> None:
    """
    功能说明：验证当 lyric_text 为空时可从 lyric_units 聚合展示文本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：lyric_units 中空文本会被忽略。
    """
    shot = {
        "lyric_text": "",
        "lyric_units": [
            {"text": "第一句"},
            {"text": "  "},
            {"text": "第二句"},
        ],
    }
    assert _extract_lyric_text_for_shot(shot=shot) == "第一句 第二句"


def test_extract_lyric_text_for_shot_should_return_unknown_marker_when_no_reliable_text() -> None:
    """
    功能说明：验证 lyric_units 仅包含未识别标记时，返回“未识别歌词”。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：兼容 lyric_text 缺失的旧分镜结构。
    """
    shot = {
        "lyric_units": [
            {"text": "吟唱"},
            {"text": "[未识别歌词]"},
        ],
    }
    assert _extract_lyric_text_for_shot(shot=shot) == "[未识别歌词]"


def test_extract_big_segment_display_for_shot_should_format_label_and_id() -> None:
    """
    功能说明：验证大段落展示文本按“标签+ID”格式输出。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：字段缺失时应返回“<未知>”。
    """
    assert _extract_big_segment_display_for_shot(
        shot={"big_segment_label": "chorus", "big_segment_id": "big_003"}
    ) == "chorus (big_003)"
    assert _extract_big_segment_display_for_shot(shot={}) == "<未知>"


def test_extract_audio_role_display_for_shot_should_map_role_text() -> None:
    """
    功能说明：验证 audio_role 到中文段落类型文案的映射。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：非法值应降级为“<未知>”。
    """
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "instrumental"}) == "器乐段"
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "vocal"}) == "人声段"
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "other"}) == "<未知>"


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
