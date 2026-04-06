"""
文件用途：验证模块B分镜生成器的歌词透传行为与排序规则。
核心流程：构造模块A输出样本，检查 shot 中 lyric_text/lyric_units 的映射结果。
输入输出：输入伪造模块A数据，输出分镜数组断言结果。
依赖说明：依赖 pytest 与项目内 MockScriptGenerator。
维护说明：若模块B歌词聚合策略变更，需同步更新本测试。
"""

# 项目内模块：模块B规则分镜生成器
from music_video_pipeline.generators.script_generator import MockScriptGenerator


def test_mock_script_generator_should_attach_lyrics_by_segment_and_sort() -> None:
    """
    功能说明：验证歌词按 segment_id 透传到分镜且按时间升序。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入歌词为乱序，输出应自动按 start_time 排序。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "verse"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"},
            {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 1.0, "end_time": 2.0, "label": "verse"},
        ],
        "energy_features": [
            {"energy_level": "mid", "trend": "flat"},
            {"energy_level": "high", "trend": "up"},
        ],
        "lyric_units": [
            {"segment_id": "seg_0001", "start_time": 0.8, "end_time": 0.95, "text": "第二句", "confidence": 0.9},
            {"segment_id": "seg_0002", "start_time": 1.2, "end_time": 1.6, "text": "第三句", "confidence": 0.8},
            {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 0.6, "text": "第一句", "confidence": 0.95},
        ],
    }

    shots = generator.generate(module_a_output=module_a_output)
    assert len(shots) == 2

    first_shot = shots[0]
    assert first_shot["lyric_text"] == "第一句 第二句"
    assert len(first_shot["lyric_units"]) == 2
    assert [item["text"] for item in first_shot["lyric_units"]] == ["第一句", "第二句"]
    assert [item["start_time"] for item in first_shot["lyric_units"]] == [0.2, 0.8]

    second_shot = shots[1]
    assert second_shot["lyric_text"] == "第三句"
    assert len(second_shot["lyric_units"]) == 1
    assert second_shot["lyric_units"][0]["segment_id"] == "seg_0002"


def test_mock_script_generator_should_keep_empty_lyrics_fields_when_no_lyrics() -> None:
    """
    功能说明：验证无歌词时也会输出 lyric_text/lyric_units 字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：module_a_output 缺失 lyric_units 字段时应安全降级为空。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "intro"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "intro"}
        ],
        "energy_features": [{"energy_level": "low", "trend": "flat"}],
    }

    shots = generator.generate(module_a_output=module_a_output)
    assert len(shots) == 1
    assert shots[0]["lyric_text"] == ""
    assert shots[0]["lyric_units"] == []


def test_mock_script_generator_should_prioritize_reliable_text_over_unknown_marker() -> None:
    """
    功能说明：验证同一镜头中存在正常歌词时，优先展示正常文本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同镜头中可同时存在“未识别歌词”标记与吟唱标记。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "verse"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_0001", "start_time": 0.1, "end_time": 0.5, "text": "[未识别歌词]", "confidence": 0.2},
            {"segment_id": "seg_0001", "start_time": 0.6, "end_time": 1.2, "text": "吟唱", "confidence": 0.6},
            {"segment_id": "seg_0001", "start_time": 1.3, "end_time": 1.9, "text": "真实歌词", "confidence": 0.9},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "真实歌词"


def test_mock_script_generator_should_fallback_to_unknown_marker_when_no_reliable_text() -> None:
    """
    功能说明：验证无正常歌词时，优先展示“未识别歌词”标记。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当仅有吟唱与未识别共存时优先未识别。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "verse"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 0.8, "text": "吟唱", "confidence": 0.6},
            {"segment_id": "seg_0001", "start_time": 0.9, "end_time": 1.7, "text": "[未识别歌词]", "confidence": 0.2},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "[未识别歌词]"


def test_mock_script_generator_should_fallback_to_time_overlap_when_segment_id_not_matched() -> None:
    """
    功能说明：验证当歌词缺失 segment_id 映射时，可按时间重叠兜底挂载到分镜。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：对所有分镜统一启用时间重叠兜底。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "verse"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.5, "label": "verse"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_xxx", "start_time": 0.3, "end_time": 1.2, "text": "重叠歌词", "confidence": 0.8},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "重叠歌词"
    assert len(shots[0]["lyric_units"]) == 1


def test_mock_script_generator_should_fallback_to_time_overlap_for_instrumental_segment() -> None:
    """
    功能说明：验证 inst 分镜在 segment_id 不匹配时仍可通过时间重叠挂载歌词。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该行为仅影响歌词透传，不改变 inst 标签本身。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "chorus"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 10.0, "end_time": 12.0, "label": "inst"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_xxx", "start_time": 10.2, "end_time": 11.6, "text": "器乐段命中歌词", "confidence": 0.8},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "器乐段命中歌词"
    assert len(shots[0]["lyric_units"]) == 1


def test_mock_script_generator_should_keep_lyrics_for_instrumental_segment() -> None:
    """
    功能说明：验证 inst 分镜在存在歌词映射时仍保留歌词字段（不再强制清空）。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅调整歌词挂载，不影响段落标签语义。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "chorus"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "inst"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.7, "text": "不应展示", "confidence": 0.8},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "不应展示"
    assert len(shots[0]["lyric_units"]) == 1


def test_mock_script_generator_should_strip_leading_punctuation_and_keep_trailing_punctuation() -> None:
    """
    功能说明：验证歌词文本仅去除句首标点，句尾标点应保留。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：纯标点文本仍应被过滤，不做整句去标点。
    """
    generator = MockScriptGenerator()
    module_a_output = {
        "big_segments": [{"segment_id": "big_001", "label": "verse"}],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
        ],
        "energy_features": [{"energy_level": "mid", "trend": "flat"}],
        "lyric_units": [
            {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.7, "text": "，你好。", "confidence": 0.9},
        ],
    }
    shots = generator.generate(module_a_output=module_a_output)
    assert shots[0]["lyric_text"] == "你好。"
