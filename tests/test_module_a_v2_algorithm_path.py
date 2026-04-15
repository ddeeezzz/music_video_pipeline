"""
文件用途：验证模块A V2算法层在新路径下的产物落盘与主流程行为。
核心流程：构造最小感知输入执行 run_algorithm_stage，检查新产物路径与结果。
输入输出：输入伪造感知层数据，输出断言结果。
依赖说明：依赖 module_a_v2.algorithm 与 artifacts。
维护说明：若产物目录结构调整，需同步更新本测试。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于路径类型
from pathlib import Path

# 项目内模块：JSON读取工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：V2算法层入口
from music_video_pipeline.modules.module_a_v2.algorithm import run_algorithm_stage
# 项目内模块：V2产物路径
from music_video_pipeline.modules.module_a_v2.artifacts import build_module_a_v2_artifacts
# 项目内模块：V2感知层数据结构
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle


def _build_perception_bundle(tmp_path: Path, lyric_sentence_units: list[dict], split_stats: dict) -> PerceptionBundle:
    """
    功能说明：构造算法层测试所需最小感知层输入对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    - lyric_sentence_units: 分句结果。
    - split_stats: 分句统计信息。
    返回值：
    - PerceptionBundle: 伪造感知层数据。
    异常说明：无。
    边界条件：beats 至少包含2个点以满足上层约束。
    """
    vocals_path = tmp_path / "vocals.wav"
    no_vocals_path = tmp_path / "no_vocals.wav"
    vocals_path.write_bytes(b"fake")
    no_vocals_path.write_bytes(b"fake")
    return PerceptionBundle(
        big_segments_stage1=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "chorus"},
        ],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 20.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 1.0, "type": "minor", "source": "allin1"},
            {"time": 2.0, "type": "minor", "source": "allin1"},
            {"time": 3.0, "type": "minor", "source": "allin1"},
            {"time": 4.0, "type": "major", "source": "allin1"},
        ],
        lyric_sentence_units=lyric_sentence_units,
        sentence_split_stats=split_stats,
        vocals_path=vocals_path,
        no_vocals_path=no_vocals_path,
        demucs_stems={},
        onset_candidates=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        rms_times=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        rms_values=[0.2, 0.25, 0.3, 0.2, 0.22, 0.2],
        vocal_onset_candidates=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        vocal_rms_times=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
        vocal_rms_values=[0.05, 0.02, 0.03, 0.05, 0.02, 0.03],
        funasr_skipped_for_silent_vocals=False,
    )


def test_run_algorithm_stage_should_write_new_artifact_layout(tmp_path: Path) -> None:
    """
    功能说明：验证算法层执行后按新分层目录写出关键产物。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用最小歌词输入覆盖“歌词句窗口 + 其他窗口”路径。
    """
    perception = _build_perception_bundle(
        tmp_path=tmp_path,
        lyric_sentence_units=[
            {
                "start_time": 1.0,
                "end_time": 2.0,
                "text": "第一句",
                "confidence": 0.9,
                "token_units": [{"text": "第", "start_time": 1.0, "end_time": 1.1}],
            },
            {
                "start_time": 6.0,
                "end_time": 7.0,
                "text": "第二句",
                "confidence": 0.9,
                "token_units": [{"text": "二", "start_time": 6.0, "end_time": 6.1}],
            },
        ],
        split_stats={"dynamic_gap_threshold_seconds": 1.0},
    )
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_algorithm_path")
    result = run_algorithm_stage(
        perception=perception,
        duration_seconds=20.0,
        instrumental_labels=["intro", "outro", "inst"],
        merge_gap_seconds=0.25,
        lyric_head_offset_seconds=0.03,
        lyric_boundary_near_anchor_seconds=1.5,
        content_role_tiny_merge_bars=1.3,
        artifacts=artifacts,
        logger=logger,
    )

    assert result.segments
    assert artifacts.algorithm_window_stage_windows_raw_path.exists()
    assert artifacts.algorithm_window_stage_windows_classified_path.exists()
    assert artifacts.algorithm_window_stage_windows_merged_path.exists()
    assert artifacts.algorithm_timeline_stage_big_a0_path.exists()
    assert artifacts.algorithm_timeline_stage_big_a1_path.exists()
    assert artifacts.algorithm_timeline_stage_lyric_sentence_units_cleaned_path.exists()
    assert artifacts.algorithm_timeline_stage_small_timestamps_path.exists()
    assert artifacts.algorithm_timeline_stage_boundary_conflict_resolved_path.exists()
    assert artifacts.algorithm_timeline_stage_big_boundary_moves_path.exists()
    assert artifacts.algorithm_final_stage_segments_final_path.exists()
    assert artifacts.algorithm_final_stage_lyric_attached_path.exists()
    assert artifacts.algorithm_final_stage_energy_path.exists()
    assert artifacts.algorithm_final_analysis_data_path.exists()


def test_run_algorithm_stage_should_build_other_window_when_no_lyrics(tmp_path: Path) -> None:
    """
    功能说明：验证无歌词时可生成覆盖全曲的“其他窗口”，并完成最终输出。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：lyric_units 为空时仍要求segments与energy有效，且窗口不再按A0边界注入切分。
    """
    perception = _build_perception_bundle(
        tmp_path=tmp_path,
        lyric_sentence_units=[],
        split_stats={"dynamic_gap_threshold_seconds": 0.35},
    )
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_algorithm_path")
    result = run_algorithm_stage(
        perception=perception,
        duration_seconds=20.0,
        instrumental_labels=["intro", "outro", "inst"],
        merge_gap_seconds=0.25,
        lyric_head_offset_seconds=0.03,
        lyric_boundary_near_anchor_seconds=1.5,
        content_role_tiny_merge_bars=1.3,
        artifacts=artifacts,
        logger=logger,
    )

    windows_raw = read_json(artifacts.algorithm_window_stage_windows_raw_path)
    assert len(windows_raw) == 1
    assert all(str(item.get("window_role_hint", "")) == "other" for item in windows_raw)
    assert all(str(item.get("window_type", "")) == "other_full_track" for item in windows_raw)
    assert abs(sum(float(item.get("duration", 0.0)) for item in windows_raw) - 20.0) <= 1e-6
    assert result.lyric_units == []
    assert result.segments
    assert result.energy_features


def test_run_algorithm_stage_should_enable_small_boundary_fragment_policy_for_v2(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证V2调用歌词挂载时会启用“跨段小残片归后段”策略及0.021s阈值。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证调用参数，不依赖内容角色规则细节。
    """
    perception = _build_perception_bundle(
        tmp_path=tmp_path,
        lyric_sentence_units=[
            {
                "start_time": 0.9,
                "end_time": 1.2,
                "text": "啊",
                "confidence": 0.9,
                "token_units": [{"text": "啊", "start_time": 0.979, "end_time": 1.2}],
            },
        ],
        split_stats={"dynamic_gap_threshold_seconds": 0.35},
    )
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_algorithm_path")
    captured_attach_kwargs: dict = {}

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.algorithm.apply_content_role_pipeline",
        lambda **_kwargs: {
            "windows_raw": [],
            "windows_classified": [],
            "windows_merged": [],
            "small_timestamps": [0.0, 20.0],
            "big_segments_a1": [
                {"segment_id": "big_001", "start_time": 0.0, "end_time": 20.0, "label": "verse"},
            ],
            "segments_final": [
                {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"},
                {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 1.0, "end_time": 20.0, "label": "verse"},
            ],
            "boundary_conflict_resolved": {},
            "big_boundary_moves": {},
        },
    )

    def _fake_attach(lyric_units_raw, segments, **kwargs):
        captured_attach_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.algorithm.attach_lyrics_to_segments", _fake_attach)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.algorithm.build_energy_features",
        lambda *_args, **_kwargs: [{"start_time": 0.0, "end_time": 20.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    result = run_algorithm_stage(
        perception=perception,
        duration_seconds=20.0,
        instrumental_labels=["intro", "outro", "inst"],
        merge_gap_seconds=0.25,
        lyric_head_offset_seconds=0.03,
        lyric_boundary_near_anchor_seconds=1.5,
        content_role_tiny_merge_bars=1.3,
        artifacts=artifacts,
        logger=logger,
    )

    assert result.segments
    assert captured_attach_kwargs["prefer_next_segment_for_small_boundary_token"] is True
    assert abs(float(captured_attach_kwargs["small_boundary_token_fragment_seconds"]) - 0.021) <= 1e-9
