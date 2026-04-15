"""
文件用途：验证模块A V2内容角色统一编排流程。
核心流程：构造分句/节拍/RMS样本并断言窗口、A1与最终S段行为。
输入输出：输入伪造A0与感知序列，输出断言结果。
依赖说明：依赖 module_a_v2.content_roles 入口函数。
维护说明：当窗口或A1规则调整时需同步更新本测试。
"""

# 项目内模块：内容角色统一入口
from music_video_pipeline.modules.module_a_v2.content_roles import apply_content_role_pipeline
# 项目内模块：窗口并段规则
from music_video_pipeline.modules.module_a_v2.timeline.role_merger import merge_windows_by_rules


def test_content_role_pipeline_should_build_lyric_and_other_windows() -> None:
    """
    功能说明：验证可构造歌词句窗口与其他窗口，并输出最终S段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：句间gap大于动态阈值时应产生 other_between 窗口。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "chorus"},
        ],
        sentence_units=[
            {"start_time": 1.0, "end_time": 2.0, "text": "第一句", "token_units": [{"text": "第", "start_time": 1.0, "end_time": 1.1}]},
            {"start_time": 6.0, "end_time": 7.0, "text": "第二句", "token_units": [{"text": "二", "start_time": 6.0, "end_time": 6.1}]},
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 1.5},
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 20.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        vocal_rms_values=[0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
        accompaniment_rms_times=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        accompaniment_rms_values=[0.2, 0.15, 0.3, 0.25, 0.18, 0.2],
        tiny_merge_bars=0.01,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=20.0,
    )

    windows_raw = result["windows_raw"]
    windows_classified = result["windows_classified"]
    segments_final = result["segments_final"]

    assert windows_raw
    assert any(item.get("window_type") == "lyric_sentence" for item in windows_raw)
    assert any(item.get("window_type") in {"other_between", "other_leading", "other_trailing"} for item in windows_raw)
    assert any(str(item.get("role", "")) == "lyric" for item in windows_classified)
    assert segments_final
    assert all(str(item.get("role", "")) in {"lyric", "chant", "inst", "silence"} for item in segments_final)
    assert len(segments_final) <= len(result["windows_merged"])


def test_content_role_pipeline_should_move_boundary_by_cross_window_ratio_without_a0_injection() -> None:
    """
    功能说明：验证不做A0注入时，big边界可由“跨big窗口占比”后置动态调整。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：设置 near_anchor 极小值，避免近锚点裁决干扰本断言。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "chorus"},
        ],
        sentence_units=[
            {"start_time": 4.0, "end_time": 5.0, "text": "前句", "token_units": [{"text": "前", "start_time": 4.0, "end_time": 4.1}]},
            {"start_time": 15.0, "end_time": 16.0, "text": "后句", "token_units": [{"text": "后", "start_time": 15.0, "end_time": 15.1}]},
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 1.0},
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 20.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        vocal_rms_values=[0.01, 0.005, 0.004, 0.005, 0.01],
        accompaniment_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        accompaniment_rms_values=[0.3, 0.28, 0.32, 0.25, 0.2],
        tiny_merge_bars=0.01,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=0.01,
        duration_seconds=20.0,
    )

    big_a1 = result["big_segments_a1"]
    moves = result["big_boundary_moves"]

    assert len(big_a1) == 2
    assert abs(float(big_a1[0]["end_time"]) - 4.1) <= 1e-6
    assert float(big_a1[1]["start_time"]) == float(big_a1[0]["end_time"])
    other_window_ratio_moves = moves.get("other_window_ratio_moves", [])
    assert isinstance(other_window_ratio_moves, list)
    assert len(other_window_ratio_moves) == 1
    assert str(other_window_ratio_moves[0].get("action", "")) == "move_next_left_to_window_start"


def test_content_role_pipeline_should_left_merge_short_gap_between_lyrics() -> None:
    """
    功能说明：验证歌词句间短gap会左并到前一句，而不是右并到后一句。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅覆盖短gap(<阈值)场景。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
        ],
        sentence_units=[
            {"start_time": 1.0, "end_time": 2.0, "text": "第一句", "token_units": [{"text": "第", "start_time": 1.0, "end_time": 2.0}]},
            {"start_time": 2.1, "end_time": 3.0, "text": "第二句", "token_units": [{"text": "二", "start_time": 2.1, "end_time": 3.0}]},
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 10.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 5.0, 10.0],
        vocal_rms_values=[0.02, 0.02, 0.02],
        accompaniment_rms_times=[0.0, 5.0, 10.0],
        accompaniment_rms_values=[0.2, 0.2, 0.2],
        tiny_merge_bars=1.3,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=10.0,
    )

    windows_raw = result["windows_raw"]
    lyric_windows = [item for item in windows_raw if str(item.get("window_role_hint", "")).lower() == "lyric"]
    assert len(lyric_windows) >= 2
    assert abs(float(lyric_windows[0]["end_time"]) - float(lyric_windows[1]["start_time"])) <= 1e-6
    assert all(str(item.get("window_type", "")) != "other_between" for item in windows_raw)


def test_role_merger_should_merge_tiny_into_lyric_when_neighbor_has_lyric() -> None:
    """
    功能说明：验证 lyric|tiny-chant|chant 场景下，tiny 优先并入 lyric 邻段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：tiny段时长小于阈值，且两侧角色不一致。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0001",
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 0,
                "role": "lyric",
            },
            {
                "window_id": "win_0002",
                "start_time": 1.0,
                "end_time": 1.3,
                "duration": 0.3,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
            {
                "window_id": "win_0003",
                "start_time": 1.3,
                "end_time": 2.8,
                "duration": 1.5,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
        ],
        tiny_merge_bars=0.8,
        bar_length_seconds=1.2,
        beats=[],
        duration_seconds=2.8,
    )

    assert len(merged) == 2
    assert str(merged[0].get("role", "")) == "lyric"
    assert str(merged[1].get("role", "")) == "chant"
    assert abs(float(merged[0].get("end_time", 0.0)) - 1.3) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 1.3) <= 1e-6
    assert events
    assert any(str(item.get("reason", "")) == "neighbor_lyric_left" for item in events)


def test_role_merger_should_merge_tiny_lyric_to_shorter_gap_side_when_both_neighbors_are_lyric() -> None:
    """
    功能说明：验证 tiny lyric 被两侧 lyric 夹住时，按更短边界间隔并段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：单阶段tiny按小节阈值执行并段。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0001",
                "start_time": 0.0,
                "end_time": 1.4,
                "duration": 1.4,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 0,
                "role": "lyric",
            },
            {
                "window_id": "win_0002",
                "start_time": 1.7,
                "end_time": 1.9,
                "duration": 0.2,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 1,
                "role": "lyric",
            },
            {
                "window_id": "win_0003",
                "start_time": 2.0,
                "end_time": 3.6,
                "duration": 1.6,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 2,
                "role": "lyric",
            },
        ],
        tiny_merge_bars=1.0,
        bar_length_seconds=1.0,
        beats=[],
        duration_seconds=3.6,
    )

    assert len(merged) == 2
    assert str(merged[0].get("role", "")) == "lyric"
    assert str(merged[1].get("role", "")) == "lyric"
    assert abs(float(merged[0].get("end_time", 0.0)) - 1.4) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 1.4) <= 1e-6
    assert events
    assert any(str(item.get("reason", "")) == "both_lyric_shorter_gap_right" for item in events)


def test_role_merger_should_pick_right_when_both_neighbors_are_chant_without_lyric() -> None:
    """
    功能说明：验证无 lyric 且左右都是 chant 时，tiny 优先并入右侧。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅覆盖左右同为chant的规则分支。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0101",
                "start_time": 0.0,
                "end_time": 1.4,
                "duration": 1.4,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
            {
                "window_id": "win_0102",
                "start_time": 1.4,
                "end_time": 1.8,
                "duration": 0.4,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "inst",
            },
            {
                "window_id": "win_0103",
                "start_time": 1.8,
                "end_time": 3.4,
                "duration": 1.6,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
        ],
        tiny_merge_bars=1.0,
        bar_length_seconds=1.0,
        beats=[],
        duration_seconds=3.4,
    )

    assert len(merged) == 2
    assert events
    assert str(events[0].get("reason", "")) == "both_chant_prefer_right"
    assert str(events[0].get("direction", "")) == "to_right"
    assert str(events[0].get("target_role", "")) == "chant"


def test_role_merger_should_pick_chant_when_one_neighbor_is_chant_and_other_is_non_chant() -> None:
    """
    功能说明：验证无 lyric 时，chant 与 inst/silence 对冲下优先并入 chant。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖“1个chant + 1个silence”场景。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0201",
                "start_time": 0.0,
                "end_time": 1.6,
                "duration": 1.6,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "silence",
            },
            {
                "window_id": "win_0202",
                "start_time": 1.6,
                "end_time": 1.9,
                "duration": 0.3,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 1,
                "role": "lyric",
            },
            {
                "window_id": "win_0203",
                "start_time": 1.9,
                "end_time": 3.4,
                "duration": 1.5,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
        ],
        tiny_merge_bars=1.0,
        bar_length_seconds=1.0,
        beats=[],
        duration_seconds=3.4,
    )

    assert len(merged) == 2
    assert events
    assert str(events[0].get("source_role", "")) == "lyric"
    assert str(events[0].get("reason", "")) == "chant_vs_instsilence_right"
    assert str(events[0].get("target_role", "")) == "chant"


def test_role_merger_should_pick_inst_when_no_lyric_and_no_chant_neighbors() -> None:
    """
    功能说明：验证无 lyric/chant 时优先并入 inst 邻段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖“silence + tiny + inst”场景。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0301",
                "start_time": 0.0,
                "end_time": 1.4,
                "duration": 1.4,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "silence",
            },
            {
                "window_id": "win_0302",
                "start_time": 1.4,
                "end_time": 1.8,
                "duration": 0.4,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "silence",
            },
            {
                "window_id": "win_0303",
                "start_time": 1.8,
                "end_time": 3.3,
                "duration": 1.5,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "inst",
            },
        ],
        tiny_merge_bars=1.0,
        bar_length_seconds=1.0,
        beats=[],
        duration_seconds=3.3,
    )

    assert len(merged) == 2
    assert events
    assert str(events[0].get("reason", "")) == "inst_prefer_right"
    assert str(events[0].get("target_role", "")) == "inst"


def test_role_merger_should_merge_when_shorter_than_1_3_bar() -> None:
    """
    功能说明：验证“短于1.3小节”会触发tiny并段（按小节阈值）。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：2.4s 在 bar=2.0 时属于 1.2 小节，应被并段。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0001",
                "start_time": 0.0,
                "end_time": 3.0,
                "duration": 3.0,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 0,
                "role": "lyric",
            },
            {
                "window_id": "win_0002",
                "start_time": 3.0,
                "end_time": 5.4,
                "duration": 2.4,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
            {
                "window_id": "win_0003",
                "start_time": 5.4,
                "end_time": 8.6,
                "duration": 3.2,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 1,
                "role": "lyric",
            },
        ],
        tiny_merge_bars=1.3,
        bar_length_seconds=2.0,
        beats=[],
        duration_seconds=8.6,
    )

    assert len(merged) == 2
    assert str(merged[0].get("role", "")) == "lyric"
    assert abs(float(merged[0].get("end_time", 0.0)) - 5.4) <= 1e-6
    assert events
    assert any(str(item.get("merge_kind", "")) == "tiny" for item in events)


def test_role_merger_should_split_long_inst_window_by_major_before_tiny_merge() -> None:
    """
    功能说明：验证长器乐窗口会先按major切分，再进入tiny合并阶段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前固定3小节步长下，24秒窗口应切为 12s + 12s 两个子窗。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0001",
                "start_time": 0.0,
                "end_time": 24.0,
                "duration": 24.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "inst",
                "vocal_peak_rms": 0.01,
                "accompaniment_peak_rms": 0.8,
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 5.0, "type": "minor", "source": "allin1"},
            {"time": 6.0, "type": "minor", "source": "allin1"},
            {"time": 7.0, "type": "minor", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 9.0, "type": "minor", "source": "allin1"},
            {"time": 10.0, "type": "minor", "source": "allin1"},
            {"time": 11.0, "type": "minor", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 13.0, "type": "minor", "source": "allin1"},
            {"time": 14.0, "type": "minor", "source": "allin1"},
            {"time": 15.0, "type": "minor", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 17.0, "type": "minor", "source": "allin1"},
            {"time": 18.0, "type": "minor", "source": "allin1"},
            {"time": 19.0, "type": "minor", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
            {"time": 21.0, "type": "minor", "source": "allin1"},
            {"time": 22.0, "type": "minor", "source": "allin1"},
            {"time": 23.0, "type": "minor", "source": "allin1"},
        ],
        duration_seconds=24.0,
    )

    assert len(merged) == 2
    assert [round(float(item.get("duration", 0.0)), 6) for item in merged] == [12.0, 12.0]
    assert all(int(item.get("split_step_bars", 0)) == 3 for item in merged)
    assert all(str(item.get("split_basis", "")) == "major" for item in merged)
    assert events == []


def test_role_merger_should_pick_major_by_onset_energy_for_long_silence_window() -> None:
    """
    功能说明：验证长静音窗口会在每个步长桶内优先选择onset能量更高的major。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：silence 固定低能量，步长=4时一个桶内4个major。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0100",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "silence",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 5.0, "type": "minor", "source": "allin1"},
            {"time": 6.0, "type": "minor", "source": "allin1"},
            {"time": 7.0, "type": "minor", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 9.0, "type": "minor", "source": "allin1"},
            {"time": 10.0, "type": "minor", "source": "allin1"},
            {"time": 11.0, "type": "minor", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 13.0, "type": "minor", "source": "allin1"},
            {"time": 14.0, "type": "minor", "source": "allin1"},
            {"time": 15.0, "type": "minor", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 17.0, "type": "minor", "source": "allin1"},
            {"time": 18.0, "type": "minor", "source": "allin1"},
            {"time": 19.0, "type": "minor", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[
            {"time": 4.0, "energy_raw": 0.2},
            {"time": 8.0, "energy_raw": 0.9},
            {"time": 12.0, "energy_raw": 0.4},
        ],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 8.0) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 8.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "energy_peak"
    assert float(merged[0].get("split_major_energy_raw", 0.0)) >= 0.9 - 1e-6
    assert events == []


def test_role_merger_should_pick_major_by_onset_energy_for_long_chant_window() -> None:
    """
    功能说明：验证长吟唱窗口也会在每个步长桶内优先选择onset能量更高的major。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：chant 固定步长=4时一个桶内4个major。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0102",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "chant",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 5.0, "type": "minor", "source": "allin1"},
            {"time": 6.0, "type": "minor", "source": "allin1"},
            {"time": 7.0, "type": "minor", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 9.0, "type": "minor", "source": "allin1"},
            {"time": 10.0, "type": "minor", "source": "allin1"},
            {"time": 11.0, "type": "minor", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 13.0, "type": "minor", "source": "allin1"},
            {"time": 14.0, "type": "minor", "source": "allin1"},
            {"time": 15.0, "type": "minor", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 17.0, "type": "minor", "source": "allin1"},
            {"time": 18.0, "type": "minor", "source": "allin1"},
            {"time": 19.0, "type": "minor", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[
            {"time": 4.0, "energy_raw": 0.1},
            {"time": 8.0, "energy_raw": 1.1},
            {"time": 12.0, "energy_raw": 0.3},
        ],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 8.0) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 8.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "energy_peak"
    assert float(merged[0].get("split_major_energy_raw", 0.0)) >= 1.1 - 1e-6
    assert events == []


def test_role_merger_should_fallback_to_index_when_major_onset_energy_all_zero() -> None:
    """
    功能说明：验证major候选能量全为0时回退到桶尾major切分策略。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：silence 场景下固定步长=3，应回退取桶尾beat=12.0。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0101",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "silence",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 12.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "fallback_index"
    assert float(merged[0].get("split_major_energy_raw", 1.0)) == 0.0
    assert events == []


def test_role_merger_should_use_sliding_major_bucket_after_selected_boundary() -> None:
    """
    功能说明：验证major细分使用“滑动桶”，下一桶从上次选中major之后继续。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：固定步长=3时，滑动桶应在同一长窗内切出两刀。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0200",
                "start_time": 0.0,
                "end_time": 24.0,
                "duration": 24.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "inst",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
            {"time": 24.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=24.0,
        onset_points=[
            {"time": 4.0, "energy_raw": 10.0},
            {"time": 8.0, "energy_raw": 1.0},
            {"time": 12.0, "energy_raw": 1.0},
            {"time": 16.0, "energy_raw": 1.0},
            {"time": 20.0, "energy_raw": 9.0},
        ],
    )

    assert len(merged) == 3
    assert abs(float(merged[0].get("end_time", 0.0)) - 4.0) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 4.0) <= 1e-6
    assert abs(float(merged[1].get("end_time", 0.0)) - 16.0) <= 1e-6
    assert abs(float(merged[2].get("start_time", 0.0)) - 16.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "energy_peak"
    assert str(merged[1].get("split_major_pick_reason", "")) == "fallback_index"
    assert all(int(item.get("split_step_bars", 0)) == 3 for item in merged)
    assert events == []


def test_role_merger_should_use_chroma_delta_priority_for_inst_role() -> None:
    """
    功能说明：验证 inst 角色切分会优先跟随 chroma 变化。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：onset/f0 均无差异时，chroma 差异应主导选点。
    """
    merged, _ = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0300",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "inst",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[],
        accompaniment_chroma_points=[
            {"time": 7.0, "chroma": [1.0] + [0.0] * 11},
            {"time": 9.0, "chroma": [1.0] + [0.0] * 11},
            {"time": 11.0, "chroma": [1.0] + [0.0] * 11},
            {"time": 13.0, "chroma": [0.0, 1.0] + [0.0] * 10},
        ],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 12.0) <= 1e-6
    assert float(merged[0].get("split_beat_chroma_delta_raw", 0.0)) > 0.1
    assert float(merged[0].get("split_beat_f0_delta_raw", 1.0)) == 0.0


def test_role_merger_should_use_f0_priority_and_pitch_source_for_chant_role() -> None:
    """
    功能说明：验证 chant 角色切分会优先跟随 F0 变化，并写出 pitch_source。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：vocal RMS 主导时应优先取 vocals F0。
    """
    merged, _ = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0301",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "chant",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[],
        vocal_rms_times=[7.0, 8.0, 9.0, 12.0],
        vocal_rms_values=[1.0, 1.0, 1.0, 0.2],
        accompaniment_rms_times=[7.0, 8.0, 9.0, 12.0],
        accompaniment_rms_values=[0.4, 0.4, 0.4, 0.4],
        vocal_f0_points=[
            {"time": 7.0, "f0_hz": 220.0, "voiced": True, "confidence": 0.9},
            {"time": 9.0, "f0_hz": 440.0, "voiced": True, "confidence": 0.9},
        ],
        accompaniment_f0_points=[
            {"time": 7.0, "f0_hz": 330.0, "voiced": True, "confidence": 0.9},
            {"time": 9.0, "f0_hz": 330.0, "voiced": True, "confidence": 0.9},
        ],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 8.0) <= 1e-6
    assert str(merged[0].get("split_pitch_source", "")) == "vocals"
    assert float(merged[0].get("split_beat_f0_delta_raw", 0.0)) > 1.0
    assert isinstance(merged[0].get("split_beat_score_components", {}), dict)


def test_role_merger_should_skip_split_when_no_major_inside_window() -> None:
    """
    功能说明：验证切分候选改回仅major后，窗口内无major时不进行切分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：窗口内部仅minor且边界major被排除时，应保持原窗口不切分。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0201",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "inst",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "minor", "source": "allin1"},
            {"time": 8.0, "type": "minor", "source": "allin1"},
            {"time": 12.0, "type": "minor", "source": "allin1"},
            {"time": 16.0, "type": "minor", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[
            {"time": 12.0, "energy_raw": 9.0},
            {"time": 13.0, "energy_raw": 6.0},
        ],
    )

    assert len(merged) == 1
    assert abs(float(merged[0].get("start_time", 0.0)) - 0.0) <= 1e-6
    assert abs(float(merged[0].get("end_time", 0.0)) - 20.0) <= 1e-6
    assert "split_pick_beat_type" not in merged[0]
    assert events == []


def test_role_merger_should_reject_far_beat_from_onset_candidates() -> None:
    """
    功能说明：验证距离最近onset过远的beat不会进入候选。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：远beat能量更高时，仍应优先选择近onset beat。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_0202",
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "window_role_hint": "other",
                "window_type": "other_full_track",
                "source_sentence_index": -1,
                "role": "inst",
            },
        ],
        tiny_merge_bars=0.9,
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        onset_points=[
            {"time": 2.6, "energy_raw": 10.0},
            {"time": 8.05, "energy_raw": 6.0},
            {"time": 8.25, "energy_raw": 4.0},
        ],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 8.0) <= 1e-6
    assert float(merged[0].get("split_beat_onset_distance", 99.0)) <= 0.5
    assert float(merged[0].get("split_beat_distance_penalty", -1.0)) > 0.0
    assert str(merged[0].get("split_major_pick_reason", "")) == "energy_peak"
    assert events == []


def test_content_role_pipeline_should_bias_to_next_big_when_spans_are_close() -> None:
    """
    功能说明：验证跨big句子前后占比接近时，边界略优先分给后一个big。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：left=0.60s、right=0.58s 时应触发后段优先。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "chorus"},
        ],
        sentence_units=[
            {
                "start_time": 9.4,
                "end_time": 10.58,
                "text": "跨界句",
                "token_units": [
                    {"text": "跨", "start_time": 9.4, "end_time": 9.46},
                    {"text": "界", "start_time": 10.52, "end_time": 10.58},
                ],
            },
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.5},
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 20.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        vocal_rms_values=[0.02, 0.02, 0.02, 0.02, 0.02],
        accompaniment_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        accompaniment_rms_values=[0.2, 0.2, 0.2, 0.2, 0.2],
        tiny_merge_bars=0.01,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=0.01,
        duration_seconds=20.0,
    )

    big_a1 = result["big_segments_a1"]
    assert len(big_a1) == 2
    assert abs(float(big_a1[0]["end_time"]) - 9.4) <= 1e-6
    assert abs(float(big_a1[1]["start_time"]) - 9.4) <= 1e-6


def test_content_role_pipeline_should_bias_to_next_big_with_aggressive_ratio() -> None:
    """
    功能说明：验证激进阈值下，右侧占比达到前段45%即可优先后一个big。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：left=1.34s、right=0.64s（right/left≈0.477）应判定后段优先。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 150.16, "label": "chorus"},
            {"segment_id": "big_002", "start_time": 150.16, "end_time": 170.0, "label": "chorus"},
        ],
        sentence_units=[
            {
                "start_time": 148.82,
                "end_time": 150.8,
                "text": "you sounds like a papa.",
                "token_units": [
                    {"text": "you", "start_time": 148.82, "end_time": 148.88},
                    {"text": " sounds", "start_time": 149.3, "end_time": 149.36},
                    {"text": " like", "start_time": 149.78, "end_time": 149.84},
                    {"text": " a", "start_time": 150.2, "end_time": 150.26},
                    {"text": " papa", "start_time": 150.44, "end_time": 150.5},
                    {"text": ".", "start_time": 150.74, "end_time": 150.8},
                ],
            },
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.5},
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 170.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 60.0, 120.0, 150.0, 170.0],
        vocal_rms_values=[0.02, 0.02, 0.02, 0.02, 0.02],
        accompaniment_rms_times=[0.0, 60.0, 120.0, 150.0, 170.0],
        accompaniment_rms_values=[0.2, 0.2, 0.2, 0.2, 0.2],
        tiny_merge_bars=0.01,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=0.01,
        duration_seconds=170.0,
    )

    big_a1 = result["big_segments_a1"]
    assert len(big_a1) == 2
    assert abs(float(big_a1[0]["end_time"]) - 148.82) <= 1e-6
    assert abs(float(big_a1[1]["start_time"]) - 148.82) <= 1e-6


def test_content_role_pipeline_should_resplit_long_lyric_window_by_punctuation_dynamic_gap() -> None:
    """
    功能说明：验证超长 lyric 窗口可先按“标点局部动态阈值”切分到3小节以内。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：重切后写入 split_basis/split_rank/split_source_window_id 字段。
    """
    beats = [{"time": float(index), "type": "major", "source": "allin1"} for index in range(0, 9)]
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "verse"},
        ],
        sentence_units=[
            {
                "start_time": 0.0,
                "end_time": 6.6,
                "text": "A，B；C。D？E",
                "token_units": [
                    {"text": "A", "start_time": 0.0, "end_time": 0.2},
                    {"text": "，", "start_time": 0.2, "end_time": 0.25},
                    {"text": "B", "start_time": 1.5, "end_time": 1.7},
                    {"text": "；", "start_time": 1.7, "end_time": 1.75},
                    {"text": "C", "start_time": 3.2, "end_time": 3.4},
                    {"text": "。", "start_time": 3.4, "end_time": 3.45},
                    {"text": "D", "start_time": 4.8, "end_time": 5.0},
                    {"text": "？", "start_time": 5.0, "end_time": 5.05},
                    {"text": "E", "start_time": 6.4, "end_time": 6.6},
                ],
            }
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[float(index) for index in range(0, 9)],
        beats=beats,
        vocal_rms_times=[0.0, 2.0, 4.0, 6.0, 8.0],
        vocal_rms_values=[0.08, 0.09, 0.08, 0.09, 0.08],
        accompaniment_rms_times=[0.0, 2.0, 4.0, 6.0, 8.0],
        accompaniment_rms_values=[0.1, 0.1, 0.1, 0.1, 0.1],
        tiny_merge_bars=0.9,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=8.0,
    )

    assert int(result["long_lyric_remaining_over3_count"]) == 0
    assert any(str(item.get("split_basis", "")) == "punct_dynamic" for item in result["windows_raw"])
    assert any("split_rank" in item for item in result["windows_raw"] if str(item.get("split_basis", "")) == "punct_dynamic")
    assert any(
        str(item.get("split_source_window_id", "")).strip()
        for item in result["windows_raw"]
        if str(item.get("split_basis", "")) == "punct_dynamic"
    )


def test_content_role_pipeline_should_resplit_long_lyric_window_by_ranked_token_gap_when_no_punctuation() -> None:
    """
    功能说明：验证无标点时，超长 lyric 窗口会按 token gap 降序兜底切分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：切分后 remaining_over3_count 应为0。
    """
    beats = [{"time": float(index), "type": "major", "source": "allin1"} for index in range(0, 10)]
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 9.0, "label": "verse"},
        ],
        sentence_units=[
            {
                "start_time": 0.0,
                "end_time": 7.4,
                "text": "abcdef",
                "token_units": [
                    {"text": "a", "start_time": 0.0, "end_time": 0.2},
                    {"text": "b", "start_time": 1.9, "end_time": 2.1},
                    {"text": "c", "start_time": 3.4, "end_time": 3.6},
                    {"text": "d", "start_time": 4.0, "end_time": 4.2},
                    {"text": "e", "start_time": 5.9, "end_time": 6.1},
                    {"text": "f", "start_time": 7.2, "end_time": 7.4},
                ],
            }
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[float(index) for index in range(0, 10)],
        beats=beats,
        vocal_rms_times=[0.0, 3.0, 6.0, 9.0],
        vocal_rms_values=[0.1, 0.1, 0.1, 0.1],
        accompaniment_rms_times=[0.0, 3.0, 6.0, 9.0],
        accompaniment_rms_values=[0.08, 0.08, 0.08, 0.08],
        tiny_merge_bars=0.9,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=9.0,
    )

    assert int(result["long_lyric_remaining_over3_count"]) == 0
    assert any(str(item.get("split_basis", "")) == "token_gap_rank" for item in result["windows_raw"])
    assert any(
        int(float(item.get("split_rank", 0))) >= 1
        for item in result["windows_raw"]
        if str(item.get("split_basis", "")) == "token_gap_rank"
    )


def test_content_role_pipeline_should_resplit_multiple_long_lyric_windows_until_all_within_three_bars() -> None:
    """
    功能说明：验证多个超长 lyric 窗口会循环重切，直到均不超过3小节。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：一个窗口走标点切分，另一个窗口走 token gap 兜底。
    """
    beats = [{"time": float(index), "type": "major", "source": "allin1"} for index in range(0, 16)]
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 15.0, "label": "verse"},
        ],
        sentence_units=[
            {
                "start_time": 0.0,
                "end_time": 6.9,
                "text": "A，B。C？D",
                "token_units": [
                    {"text": "A", "start_time": 0.0, "end_time": 0.2},
                    {"text": "，", "start_time": 0.2, "end_time": 0.25},
                    {"text": "B", "start_time": 1.8, "end_time": 2.0},
                    {"text": "。", "start_time": 2.0, "end_time": 2.05},
                    {"text": "C", "start_time": 4.2, "end_time": 4.4},
                    {"text": "？", "start_time": 4.4, "end_time": 4.45},
                    {"text": "D", "start_time": 6.7, "end_time": 6.9},
                ],
            },
            {
                "start_time": 8.0,
                "end_time": 14.4,
                "text": "mnopq",
                "token_units": [
                    {"text": "m", "start_time": 8.0, "end_time": 8.2},
                    {"text": "n", "start_time": 9.7, "end_time": 9.9},
                    {"text": "o", "start_time": 11.2, "end_time": 11.4},
                    {"text": "p", "start_time": 12.6, "end_time": 12.8},
                    {"text": "q", "start_time": 14.2, "end_time": 14.4},
                ],
            },
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[float(index) for index in range(0, 16)],
        beats=beats,
        vocal_rms_times=[0.0, 5.0, 10.0, 15.0],
        vocal_rms_values=[0.09, 0.1, 0.09, 0.08],
        accompaniment_rms_times=[0.0, 5.0, 10.0, 15.0],
        accompaniment_rms_values=[0.08, 0.08, 0.08, 0.08],
        tiny_merge_bars=0.9,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=15.0,
    )

    max_lyric_window_seconds = float(result.get("max_lyric_window_seconds", 0.0))
    lyric_windows = [item for item in result["windows_raw"] if str(item.get("window_role_hint", "")).lower() == "lyric"]
    assert int(result["long_lyric_remaining_over3_count"]) == 0
    assert max_lyric_window_seconds > 0.0
    assert all(float(item.get("duration", 0.0)) <= max_lyric_window_seconds + 1e-6 for item in lyric_windows)
    assert len(result.get("long_lyric_resplit_events", [])) >= 2


def test_content_role_pipeline_should_merge_tiny_within_long_lyric_before_global_window_flow() -> None:
    """
    功能说明：验证长句重切完成后，tiny 子窗会先在长句内部并段，再进入全局窗口流程。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：句内 tiny 并段不应依赖后续全局 tiny 并段触发。
    """
    beats = [{"time": float(index), "type": "major", "source": "allin1"} for index in range(0, 9)]
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 9.0, "label": "verse"},
        ],
        sentence_units=[
            {
                "start_time": 0.0,
                "end_time": 8.8,
                "text": "ABCDEF",
                "token_units": [
                    {"text": "A", "start_time": 0.0, "end_time": 0.2},
                    {"text": "B", "start_time": 2.2, "end_time": 2.4},
                    {"text": "C", "start_time": 4.8, "end_time": 5.0},
                    {"text": "D", "start_time": 5.2, "end_time": 5.4},
                    {"text": "E", "start_time": 8.0, "end_time": 8.2},
                    {"text": "F", "start_time": 8.6, "end_time": 8.8},
                ],
            },
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[float(index) for index in range(0, 10)],
        beats=[{"time": float(index), "type": "major", "source": "allin1"} for index in range(0, 10)],
        vocal_rms_times=[0.0, 2.0, 4.0, 6.0, 8.0, 9.0],
        vocal_rms_values=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
        accompaniment_rms_times=[0.0, 2.0, 4.0, 6.0, 8.0, 9.0],
        accompaniment_rms_values=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        tiny_merge_bars=0.9,
        lyric_head_offset_seconds=0.03,
        near_anchor_seconds=1.5,
        duration_seconds=9.0,
    )

    windows_raw_lyric_count = sum(
        1 for item in result["windows_raw"] if str(item.get("window_role_hint", "")).lower() == "lyric"
    )
    windows_merged_lyric_count = sum(1 for item in result["windows_merged"] if str(item.get("role", "")).lower() == "lyric")
    long_lyric_resplit_events = list(result.get("long_lyric_resplit_events", []))
    long_lyric_inner_tiny_merge_events = list(result.get("long_lyric_inner_tiny_merge_events", []))
    merge_events = list(result.get("window_merge_events", []))
    assert windows_raw_lyric_count >= 3
    assert windows_merged_lyric_count <= windows_raw_lyric_count
    assert any(str(item.get("split_basis", "")) in {"punct_dynamic", "token_gap_rank"} for item in long_lyric_resplit_events)
    assert any(str(item.get("merge_kind", "")) == "long_lyric_inner_tiny" for item in long_lyric_inner_tiny_merge_events)
    assert all(str(item.get("merge_kind", "")) == "tiny" for item in merge_events if str(item.get("merge_kind", "")))
