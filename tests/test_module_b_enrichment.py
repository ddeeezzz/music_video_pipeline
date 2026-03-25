"""
文件用途：验证模块B分镜增强阶段的大段落与段落类型补全逻辑。
核心流程：构造 shots 与模块A输出，检查索引匹配与重叠回退行为。
输入输出：输入伪造分镜和模块A数据，输出断言结果。
依赖说明：依赖 pytest 与模块B内部增强函数。
维护说明：若分镜补全策略调整，需同步更新本测试。
"""

# 项目内模块：模块B分镜增强函数
from music_video_pipeline.modules.module_b import _enrich_shots_with_segment_meta


def test_enrich_shots_should_fill_meta_by_index() -> None:
    """
    功能说明：验证分镜与segment同序时按索引补全元信息。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅按标签集合判定器乐/人声，不依赖歌词字段。
    """
    shots = [
        {"shot_id": "shot_001", "start_time": 0.0, "end_time": 1.0},
        {"shot_id": "shot_002", "start_time": 1.0, "end_time": 2.0},
    ]
    module_a_output = {
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "intro"},
            {"segment_id": "seg_0002", "big_segment_id": "big_002", "start_time": 1.0, "end_time": 2.0, "label": "verse"},
        ],
        "big_segments": [
            {"segment_id": "big_001", "label": "intro"},
            {"segment_id": "big_002", "label": "verse"},
        ],
    }
    enriched = _enrich_shots_with_segment_meta(
        shots=shots,
        module_a_output=module_a_output,
        instrumental_labels=["intro", "inst", "outro"],
    )
    assert enriched[0]["big_segment_id"] == "big_001"
    assert enriched[0]["big_segment_label"] == "intro"
    assert enriched[0]["segment_label"] == "intro"
    assert enriched[0]["audio_role"] == "instrumental"

    assert enriched[1]["big_segment_id"] == "big_002"
    assert enriched[1]["big_segment_label"] == "verse"
    assert enriched[1]["segment_label"] == "verse"
    assert enriched[1]["audio_role"] == "vocal"


def test_enrich_shots_should_fallback_to_overlap_when_index_mismatch() -> None:
    """
    功能说明：验证索引无法命中时回退到时间重叠最大匹配。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：shots 数量超过 segments 时，后续分镜也应可补全。
    """
    shots = [
        {"shot_id": "shot_001", "start_time": 0.0, "end_time": 1.0},
        {"shot_id": "shot_002", "start_time": 0.1, "end_time": 0.9},
    ]
    module_a_output = {
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_010", "start_time": 0.0, "end_time": 1.0, "label": "chorus"},
        ],
        "big_segments": [
            {"segment_id": "big_010", "label": "chorus"},
        ],
    }
    enriched = _enrich_shots_with_segment_meta(
        shots=shots,
        module_a_output=module_a_output,
        instrumental_labels=["intro", "inst", "outro"],
    )
    assert enriched[1]["big_segment_id"] == "big_010"
    assert enriched[1]["big_segment_label"] == "chorus"
    assert enriched[1]["segment_label"] == "chorus"
    assert enriched[1]["audio_role"] == "vocal"


def test_enrich_shots_should_fallback_when_index_order_is_misaligned() -> None:
    """
    功能说明：验证同长度但顺序错位时，会按时间重叠回退匹配。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：索引命中但无重叠时必须切换到重叠匹配。
    """
    shots = [
        {"shot_id": "shot_001", "start_time": 0.0, "end_time": 1.0},
        {"shot_id": "shot_002", "start_time": 1.0, "end_time": 2.0},
    ]
    module_a_output = {
        "segments": [
            {"segment_id": "seg_0100", "big_segment_id": "big_100", "start_time": 1.0, "end_time": 2.0, "label": "verse"},
            {"segment_id": "seg_0101", "big_segment_id": "big_101", "start_time": 0.0, "end_time": 1.0, "label": "intro"},
        ],
        "big_segments": [
            {"segment_id": "big_100", "label": "verse"},
            {"segment_id": "big_101", "label": "intro"},
        ],
    }
    enriched = _enrich_shots_with_segment_meta(
        shots=shots,
        module_a_output=module_a_output,
        instrumental_labels=["intro", "inst", "outro"],
    )
    assert enriched[0]["big_segment_id"] == "big_101"
    assert enriched[0]["audio_role"] == "instrumental"
    assert enriched[1]["big_segment_id"] == "big_100"
    assert enriched[1]["audio_role"] == "vocal"
