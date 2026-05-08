"""
文件用途：验证模块A V2内容角色统一编排流程。
核心流程：构造分句/节拍/RMS样本并断言窗口、A1与最终S段行为。
输入输出：输入伪造A0与感知序列，输出断言结果。
依赖说明：依赖 module_a_v2.content_roles 入口函数。
维护说明：当窗口或A1规则调整时需同步更新本测试。
"""

# 项目内模块：内容角色统一入口
from music_video_pipeline.modules.module_a_v2.content_roles import apply_content_role_pipeline
# 项目内模块：窗口并段规则与长other切分规则
from music_video_pipeline.modules.module_a_v2.timeline.role_merger import (
    merge_windows_by_rules,
    split_long_other_windows_by_major,
)


def _split_long_other_windows_for_test(
    windows: list[dict],
    *,
    beats: list[dict],
    bar_length_seconds: float,
    duration_seconds: float,
    major_split_step_bars: float = 2.5,
    onset_points: list[dict] | None = None,
    vocal_rms_times: list[float] | None = None,
    vocal_rms_values: list[float] | None = None,
    accompaniment_rms_times: list[float] | None = None,
    accompaniment_rms_values: list[float] | None = None,
    accompaniment_chroma_points: list[dict] | None = None,
    vocal_f0_points: list[dict] | None = None,
    accompaniment_f0_points: list[dict] | None = None,
) -> list[dict]:
    """
    功能说明：为测试提供“前移长 other 切分”便捷入口。
    参数说明：透传长 other 切分所需的节拍与特征输入。
    返回值：
    - list[dict]: 切分后的窗口列表。
    异常说明：断言失败时由调用侧处理。
    边界条件：仅覆盖测试场景，不承担生产编排职责。
    """
    return split_long_other_windows_by_major(
        windows=windows,
        beats=beats,
        bar_length_seconds=bar_length_seconds,
        long_window_split_min_bars=1.0,
        major_split_step_bars=major_split_step_bars,
        onset_points=onset_points,
        vocal_rms_times=vocal_rms_times,
        vocal_rms_values=vocal_rms_values,
        accompaniment_rms_times=accompaniment_rms_times,
        accompaniment_rms_values=accompaniment_rms_values,
        accompaniment_chroma_points=accompaniment_chroma_points,
        vocal_f0_points=vocal_f0_points,
        accompaniment_f0_points=accompaniment_f0_points,
    )


def _build_tiny_similarity_features_for_test() -> dict[str, list]:
    """
    功能说明：构造 tiny 相似度测试所需的简化特征输入。
    参数说明：无。
    返回值：
    - dict[str, list]: onset/RMS/chroma/F0 测试特征。
    异常说明：无。
    边界条件：特征按时间分布，便于在左右窗口间制造明显相似度差异。
    """
    return {
        "onset_points": [
            {"time": 0.2, "energy_raw": 0.92},
            {"time": 0.6, "energy_raw": 0.88},
            {"time": 1.05, "energy_raw": 0.90},
            {"time": 1.2, "energy_raw": 0.86},
            {"time": 2.1, "energy_raw": 0.08},
        ],
        "accompaniment_rms_times": [0.2, 0.6, 1.05, 1.2, 2.1],
        "accompaniment_rms_values": [0.82, 0.80, 0.81, 0.79, 0.12],
        "accompaniment_chroma_points": [
            {"time": 0.3, "chroma": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"time": 1.1, "chroma": [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"time": 2.1, "chroma": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        ],
        "vocal_f0_points": [
            {"time": 0.35, "f0_hz": 220.0, "voiced": True, "confidence": 0.9},
            {"time": 1.1, "f0_hz": 221.0, "voiced": True, "confidence": 0.9},
        ],
        "accompaniment_f0_points": [
            {"time": 2.1, "f0_hz": 440.0, "voiced": True, "confidence": 0.9},
        ],
    }


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
        visual_lead_seconds=0.0,
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


def test_content_role_pipeline_should_split_other_windows_by_activity_boundaries_before_big_ratio_move() -> None:
    """
    功能说明：验证新四分类主链会先按活动窗切开 other 窗口，再进入 big 边界阶段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前 case 中活动窗已在 10.0 处切开，后续不再触发 other-window ratio move。
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
        visual_lead_seconds=0.0,
        near_anchor_seconds=0.01,
        duration_seconds=20.0,
    )

    big_a1 = result["big_segments_a1"]
    moves = result["big_boundary_moves"]
    windows_classified = result["windows_classified"]
    activity_windows = result["activity_windows"]

    assert len(big_a1) == 2
    assert abs(float(big_a1[0]["end_time"]) - 10.0) <= 1e-6
    assert float(big_a1[1]["start_time"]) == float(big_a1[0]["end_time"])
    assert [str(item.get("role", "")) for item in windows_classified] == [
        "chant",
        "lyric",
        "chant",
        "inst",
        "lyric",
        "inst",
    ]
    assert activity_windows["vocal"]["intervals"] == [
        {"start_time": 0.0, "end_time": 10.0, "duration": 10.0}
    ]
    other_window_ratio_moves = moves.get("other_window_ratio_moves", [])
    assert isinstance(other_window_ratio_moves, list)
    assert other_window_ratio_moves == []


def test_content_role_pipeline_should_pre_split_long_other_before_activity_boundary_inject() -> None:
    """
    功能说明：验证长 other 窗会先执行预分类与 major 切分，再进入活动边界注入。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：预分类仅服务前移切分，正式分类结果仍以 inject 后为准。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 24.0, "label": "bridge"},
        ],
        sentence_units=[
            {"start_time": 0.8, "end_time": 1.2, "text": "前句", "token_units": [{"text": "前", "start_time": 0.8, "end_time": 1.2}]},
            {"start_time": 22.2, "end_time": 22.8, "text": "后句", "token_units": [{"text": "后", "start_time": 22.2, "end_time": 22.8}]},
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.5},
        beat_candidates=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
            {"time": 24.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 24.0],
        vocal_rms_values=[0.0, 0.0, 0.08, 0.08, 0.0, 0.0, 0.06, 0.0],
        accompaniment_rms_times=[0.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 24.0],
        accompaniment_rms_values=[0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
        tiny_merge_bars=0.01,
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=24.0,
        onset_points=[
            {"time": 8.0, "energy_raw": 0.9},
            {"time": 20.0, "energy_raw": 0.8},
        ],
    )

    windows_pre_split_classified = result["windows_pre_split_classified"]
    windows_pre_boundary_other_split = result["windows_pre_boundary_other_split"]
    windows_classified = result["windows_classified"]

    assert any("pre_split_role" in item for item in windows_pre_split_classified)
    assert {
        str(item.get("pre_split_role", ""))
        for item in windows_pre_split_classified
    } <= {"lyric", "other"}
    assert any(str(item.get("split_basis", "")) == "major" for item in windows_pre_boundary_other_split)
    assert any(str(item.get("window_type", "")).endswith("_major_split") for item in windows_pre_boundary_other_split)
    assert any("final_role" in item for item in windows_classified)
    assert all(str(item.get("final_role", "")) == str(item.get("role", "")) for item in windows_classified)
    assert not any(
        str(item.get("window_type", "")).endswith("_a0_split")
        for item in windows_classified
        if str(item.get("source_window_id", "")).startswith("win_")
        and str(item.get("split_basis", "")).lower().strip() == "major"
    )


def test_content_role_pipeline_should_keep_pre_split_role_when_final_classification_overwrites_role() -> None:
    """
    功能说明：验证正式分类覆盖 role 时，会保留 pre_split_role 作为前移切分追踪字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同一窗口经 inject 后可重新分类，但 pre_split_role 不应丢失。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "bridge"},
        ],
        sentence_units=[],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[0.0, 4.0, 8.0, 12.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 3.0, 6.0, 9.0, 12.0],
        vocal_rms_values=[0.0, 0.0, 0.08, 0.0, 0.0],
        accompaniment_rms_times=[0.0, 3.0, 6.0, 9.0, 12.0],
        accompaniment_rms_values=[0.2, 0.2, 0.2, 0.2, 0.2],
        tiny_merge_bars=0.01,
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=12.0,
        onset_points=[{"time": 8.0, "energy_raw": 0.9}],
    )

    windows_classified = result["windows_classified"]
    assert windows_classified
    assert all("pre_split_role" in item for item in windows_classified)
    assert all("final_role" in item for item in windows_classified)


def test_content_role_pipeline_should_derive_chant_inst_and_silence_from_activity_windows() -> None:
    """
    功能说明：验证无歌词场景下会先抽活动窗，再得到 chant/inst/silence 三类窗口。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：活动边界直接决定窗口切分，避免退回旧版“整窗看峰值”策略。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "inst"},
        ],
        sentence_units=[],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 6.0, "type": "minor", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vocal_rms_values=[0.0, 0.0, 0.12, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        accompaniment_rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        accompaniment_rms_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0],
        vocal_energy_enter_quantile=0.70,
        vocal_energy_exit_quantile=0.45,
        vocal_mid_segment_min_duration_seconds=0.5,
        short_vocal_non_lyric_merge_seconds=0.2,
        tiny_merge_bars=0.01,
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=10.0,
    )

    windows_classified = result["windows_classified"]
    activity_windows = result["activity_windows"]

    assert [str(item.get("role", "")) for item in windows_classified] == [
        "silence",
        "chant",
        "silence",
        "inst",
        "silence",
    ]
    assert [(float(item["start_time"]), float(item["end_time"])) for item in activity_windows["vocal"]["intervals"]] == [
        (2.0, 4.0)
    ]
    assert [
        (float(item["start_time"]), float(item["end_time"])) for item in activity_windows["accompaniment"]["intervals"]
    ] == [
        (6.0, 8.0)
    ]


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
        visual_lead_seconds=0.0,
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
    features = _build_tiny_similarity_features_for_test()
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
        onset_points=features["onset_points"],
        accompaniment_rms_times=features["accompaniment_rms_times"],
        accompaniment_rms_values=features["accompaniment_rms_values"],
        accompaniment_chroma_points=features["accompaniment_chroma_points"],
        vocal_f0_points=features["vocal_f0_points"],
        accompaniment_f0_points=features["accompaniment_f0_points"],
    )

    assert len(merged) == 2
    assert str(merged[0].get("role", "")) == "lyric"
    assert str(merged[1].get("role", "")) == "chant"
    assert abs(float(merged[0].get("end_time", 0.0)) - 1.3) <= 1e-6
    assert abs(float(merged[1].get("start_time", 0.0)) - 1.3) <= 1e-6
    assert events
    assert any(str(item.get("reason", "")) == "similarity_left_higher" for item in events)
    assert any(float(item.get("left_similarity_total", 0.0)) > float(item.get("right_similarity_total", 0.0)) for item in events)


def test_role_merger_should_merge_tiny_to_right_when_right_similarity_is_higher() -> None:
    """
    功能说明：验证 tiny 窗口在右侧相似度更高时，会并向右侧。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：两侧邻居同时存在，且右侧特征与 tiny 更接近。
    """
    features = {
        "onset_points": [
            {"time": 0.2, "energy_raw": 0.12},
            {"time": 1.05, "energy_raw": 0.92},
            {"time": 1.25, "energy_raw": 0.88},
            {"time": 2.15, "energy_raw": 0.90},
            {"time": 2.45, "energy_raw": 0.85},
        ],
        "accompaniment_rms_times": [0.2, 1.05, 1.25, 2.15, 2.45],
        "accompaniment_rms_values": [0.10, 0.78, 0.76, 0.77, 0.75],
        "accompaniment_chroma_points": [
            {"time": 0.4, "chroma": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"time": 1.1, "chroma": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"time": 2.2, "chroma": [0.0, 0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        ],
        "vocal_f0_points": [
            {"time": 1.1, "f0_hz": 247.0, "voiced": True, "confidence": 0.9},
            {"time": 2.2, "f0_hz": 247.5, "voiced": True, "confidence": 0.9},
        ],
        "accompaniment_f0_points": [],
    }
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
        onset_points=features["onset_points"],
        accompaniment_rms_times=features["accompaniment_rms_times"],
        accompaniment_rms_values=features["accompaniment_rms_values"],
        accompaniment_chroma_points=features["accompaniment_chroma_points"],
        vocal_f0_points=features["vocal_f0_points"],
        accompaniment_f0_points=features["accompaniment_f0_points"],
    )

    assert len(merged) == 2
    assert str(merged[0].get("role", "")) == "lyric"
    assert str(merged[1].get("role", "")) == "lyric"
    assert events
    assert str(events[0].get("reason", "")) == "similarity_right_higher"
    assert str(events[0].get("direction", "")) == "to_right"
    assert float(events[0].get("right_similarity_total", 0.0)) > float(events[0].get("left_similarity_total", 0.0))


def test_role_merger_should_merge_tiny_inst_to_higher_similarity_side_between_chant_windows() -> None:
    """
    功能说明：验证微器乐段夹在两个 chant 之间时，会并向相似度更高的一侧。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖“left=chant, source=inst, right=chant”场景。
    """
    features = _build_tiny_similarity_features_for_test()
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
        onset_points=features["onset_points"],
        accompaniment_rms_times=features["accompaniment_rms_times"],
        accompaniment_rms_values=features["accompaniment_rms_values"],
        accompaniment_chroma_points=features["accompaniment_chroma_points"],
        vocal_f0_points=features["vocal_f0_points"],
        accompaniment_f0_points=features["accompaniment_f0_points"],
    )

    assert len(merged) == 2
    assert events
    assert str(events[0].get("reason", "")) == "similarity_right_higher"
    assert str(events[0].get("direction", "")) == "to_right"
    assert str(events[0].get("target_role", "")) == "chant"


def test_role_merger_should_tie_break_to_left_when_similarity_equal() -> None:
    """
    功能说明：验证左右相似度完全相等时，会稳定按左侧并段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不传任何特征时，两侧得分一致，应走平分规则。
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
    assert str(events[0].get("reason", "")) == "similarity_tie_left"
    assert bool(events[0].get("tie_break_applied", False)) is True
    assert str(events[0].get("target_role", "")) == "silence"


def test_role_merger_should_keep_similarity_explanation_fields_on_event() -> None:
    """
    功能说明：验证 tiny merge event 会写出左右相似度明细字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只要存在双侧邻居，就应输出左右总分与四项组件。
    """
    features = _build_tiny_similarity_features_for_test()
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
        onset_points=features["onset_points"],
        accompaniment_rms_times=features["accompaniment_rms_times"],
        accompaniment_rms_values=features["accompaniment_rms_values"],
        accompaniment_chroma_points=features["accompaniment_chroma_points"],
        vocal_f0_points=features["vocal_f0_points"],
        accompaniment_f0_points=features["accompaniment_f0_points"],
    )

    assert len(merged) == 2
    assert events
    assert str(events[0].get("decision_strategy", "")) == "similarity"
    assert "left_similarity_total" in events[0]
    assert "right_similarity_total" in events[0]
    assert "left_similarity_components" in events[0]
    assert "right_similarity_components" in events[0]
    assert sorted(events[0]["left_similarity_components"].keys()) == [
        "chroma_similarity",
        "energy_similarity",
        "onset_similarity",
        "total_similarity",
        "voiced_f0_similarity",
    ]


def test_role_merger_should_process_tiny_windows_by_source_role_priority() -> None:
    """
    功能说明：验证 tiny 并段会先处理 lyric，再处理 chant/inst/silence。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同一轮存在多个 tiny 候选时，先后顺序应由源角色优先级决定。
    """
    merged, events = merge_windows_by_rules(
        windows_classified=[
            {
                "window_id": "win_1001",
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": -1,
                "role": "lyric",
            },
            {
                "window_id": "win_1002",
                "start_time": 1.0,
                "end_time": 1.3,
                "duration": 0.3,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 0,
                "role": "lyric",
            },
            {
                "window_id": "win_1003",
                "start_time": 1.3,
                "end_time": 1.6,
                "duration": 0.3,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "chant",
            },
            {
                "window_id": "win_1004",
                "start_time": 1.6,
                "end_time": 1.9,
                "duration": 0.3,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "inst",
            },
            {
                "window_id": "win_1005",
                "start_time": 1.9,
                "end_time": 2.2,
                "duration": 0.3,
                "window_role_hint": "other",
                "window_type": "other_between",
                "source_sentence_index": -1,
                "role": "silence",
            },
            {
                "window_id": "win_1006",
                "start_time": 2.2,
                "end_time": 3.7,
                "duration": 1.5,
                "window_role_hint": "lyric",
                "window_type": "lyric_sentence",
                "source_sentence_index": 1,
                "role": "lyric",
            },
        ],
        tiny_merge_bars=0.5,
        bar_length_seconds=1.0,
        beats=[],
        duration_seconds=3.7,
    )

    assert len(merged) >= 1
    assert [str(item.get("source_role", "")) for item in events[:4]] == ["lyric", "chant", "inst", "silence"]


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
    边界条件：当前默认 2.5 小节步长下，24 秒窗口应切为 8s + 8s + 8s 三个子窗。
    """
    merged = _split_long_other_windows_for_test(
        [
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

    assert len(merged) == 3
    assert [round(float(item.get("duration", 0.0)), 6) for item in merged] == [8.0, 8.0, 8.0]
    assert all(abs(float(item.get("split_step_bars", 0.0)) - 2.5) <= 1e-6 for item in merged)
    assert all(str(item.get("split_basis", "")) == "major" for item in merged)


def test_role_merger_should_pick_major_by_onset_energy_for_long_silence_window() -> None:
    """
    功能说明：验证长静音窗口会在每个步长桶内优先选择onset能量更高的major。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：silence 固定低能量，步长=4时一个桶内4个major。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        major_split_step_bars=3.0,
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


def test_role_merger_should_pick_major_by_onset_energy_for_long_chant_window() -> None:
    """
    功能说明：验证长吟唱窗口也会在每个步长桶内优先选择onset能量更高的major。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：chant 固定步长=4时一个桶内4个major。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        major_split_step_bars=3.0,
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


def test_role_merger_should_fallback_to_index_when_major_onset_energy_all_zero() -> None:
    """
    功能说明：验证major候选能量全为0时回退到桶尾major切分策略。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：silence 场景下固定步长=3，应回退取桶尾beat=12.0。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        bar_length_seconds=4.0,
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
        ],
        duration_seconds=20.0,
        major_split_step_bars=3.0,
        onset_points=[],
    )

    assert len(merged) == 2
    assert abs(float(merged[0].get("end_time", 0.0)) - 12.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "fallback_index"
    assert float(merged[0].get("split_major_energy_raw", 1.0)) == 0.0


def test_role_merger_should_use_sliding_major_bucket_after_selected_boundary() -> None:
    """
    功能说明：验证major细分使用“滑动桶”，下一桶从上次选中major之后继续。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：固定步长=3时，滑动桶应在同一长窗内切出两刀。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        major_split_step_bars=3.0,
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


def test_role_merger_should_use_chroma_delta_priority_for_inst_role() -> None:
    """
    功能说明：验证 inst 角色切分会优先跟随 chroma 变化。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：onset/f0 均无差异时，chroma 差异应主导选点。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        major_split_step_bars=3.0,
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


def test_role_merger_should_split_other_by_major_without_chant_specific_f0_priority() -> None:
    """
    功能说明：验证 other 统一重拍切分不再依赖 chant 专属 F0 优先级。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当桶内无 onset 参照时，应按统一重拍规则回退到桶尾 major。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        major_split_step_bars=3.0,
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
    assert abs(float(merged[0].get("end_time", 0.0)) - 12.0) <= 1e-6
    assert str(merged[0].get("split_major_pick_reason", "")) == "fallback_index"
    assert isinstance(merged[0].get("split_beat_score_components", {}), dict)


def test_role_merger_should_skip_split_when_no_major_inside_window() -> None:
    """
    功能说明：验证切分候选改回仅major后，窗口内无major时不进行切分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：窗口内部仅minor且边界major被排除时，应保持原窗口不切分。
    """
    merged = _split_long_other_windows_for_test(
        [
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


def test_role_merger_should_reject_far_beat_from_onset_candidates() -> None:
    """
    功能说明：验证距离最近onset过远的beat不会进入候选。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：远beat能量更高时，仍应优先选择近onset beat。
    """
    merged = _split_long_other_windows_for_test(
        [
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
        visual_lead_seconds=0.0,
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
        visual_lead_seconds=0.0,
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
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=8.0,
    )

    assert int(result["long_lyric_remaining_over_limit_count"]) == 0
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
    边界条件：切分后 remaining_over_limit_count 应为0。
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
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=9.0,
    )

    assert int(result["long_lyric_remaining_over_limit_count"]) == 0
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
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=15.0,
    )

    max_lyric_window_seconds = float(result.get("max_lyric_window_seconds", 0.0))
    lyric_windows = [item for item in result["windows_raw"] if str(item.get("window_role_hint", "")).lower() == "lyric"]
    assert int(result["long_lyric_remaining_over_limit_count"]) == 0
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
        visual_lead_seconds=0.0,
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
    assert all(str(item.get("decision_strategy", "")) == "similarity" for item in merge_events if str(item.get("merge_kind", "")) == "tiny")


def test_content_role_pipeline_should_fallback_split_long_lyric_by_major_when_gap_rules_fail() -> None:
    """
    功能说明：验证超长 lyric 在标点与 token gap 均无法有效切开时，会按重拍执行兜底切分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：兜底切分仅在 lyric window 内部生效，不跨出原始歌词边界。
    """
    result = apply_content_role_pipeline(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 20.0, "label": "chorus"},
        ],
        sentence_units=[
            {
                "start_time": 1.0,
                "end_time": 12.2,
                "text": "みんなドア遊ぼう。",
                "token_units": [
                    {"text": "み", "start_time": 1.0, "end_time": 1.06},
                    {"text": "ん", "start_time": 1.12, "end_time": 1.18},
                    {"text": "な", "start_time": 1.24, "end_time": 1.30},
                    {"text": "ド", "start_time": 1.36, "end_time": 1.42},
                    {"text": "ア", "start_time": 1.48, "end_time": 1.54},
                    {"text": "遊", "start_time": 1.60, "end_time": 1.66},
                    {"text": "ぼ", "start_time": 1.72, "end_time": 1.78},
                    {"text": "う", "start_time": 1.84, "end_time": 1.90},
                    {"text": "。", "start_time": 12.14, "end_time": 12.20},
                ],
            }
        ],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35},
        beat_candidates=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 2.0, "type": "major", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
            {"time": 6.0, "type": "major", "source": "allin1"},
            {"time": 8.0, "type": "major", "source": "allin1"},
            {"time": 10.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "major", "source": "allin1"},
            {"time": 14.0, "type": "major", "source": "allin1"},
            {"time": 16.0, "type": "major", "source": "allin1"},
            {"time": 18.0, "type": "major", "source": "allin1"},
            {"time": 20.0, "type": "major", "source": "allin1"},
        ],
        vocal_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        vocal_rms_values=[0.08, 0.08, 0.08, 0.08, 0.08],
        accompaniment_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        accompaniment_rms_values=[0.08, 0.08, 0.08, 0.08, 0.08],
        tiny_merge_bars=0.8,
        visual_lead_seconds=0.0,
        near_anchor_seconds=1.5,
        duration_seconds=20.0,
        long_lyric_resplit_max_bars=2.0,
    )

    lyric_windows = [
        item
        for item in result["windows_raw"]
        if str(item.get("window_role_hint", "")).lower() == "lyric"
    ]
    assert int(result["long_lyric_remaining_over_limit_count"]) == 0
    assert any(str(item.get("split_basis", "")) == "major" for item in lyric_windows)
    assert any(str(item.get("split_basis", "")) == "major_fallback" for item in result["long_lyric_resplit_events"])
    assert all(1.0 - 1e-6 <= float(item.get("start_time", 0.0)) for item in lyric_windows)
    assert all(float(item.get("end_time", 0.0)) <= 12.2 + 1e-6 for item in lyric_windows)
