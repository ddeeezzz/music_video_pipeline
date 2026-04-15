"""
文件用途：验证模块A V2中间产物目录与关键文件落盘。
核心流程：打桩感知层外部依赖并执行 run_perception_stage，检查产物文件。
输入输出：输入临时目录与打桩后端，输出断言结果。
依赖说明：依赖 pytest 与模块A V2感知层实现。
维护说明：若中间产物路径规范调整，需同步更新本测试断言。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：JSON读写工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：V2产物路径
from music_video_pipeline.modules.module_a_v2.artifacts import build_module_a_v2_artifacts
# 项目内模块：V2感知层
from music_video_pipeline.modules.module_a_v2.perception import run_perception_stage


def test_module_a_v2_perception_should_write_required_artifacts(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证感知层会落盘核心JSON产物且不再物化 Demucs stems 目录。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：全部外部后端能力通过打桩模拟。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_artifacts")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")

    source_stems_dir = tmp_path / "source_stems"
    source_stems_dir.mkdir(parents=True, exist_ok=True)
    vocals_source = source_stems_dir / "vocals.wav"
    bass_source = source_stems_dir / "bass.wav"
    drums_source = source_stems_dir / "drums.wav"
    other_source = source_stems_dir / "other.wav"
    no_vocals_source = source_stems_dir / "no_vocals.wav"
    for path in [vocals_source, bass_source, drums_source, other_source, no_vocals_source]:
        path.write_bytes(b"fake-wav")

    def _fake_prepare_stems_with_allin1_demucs(*_args, **_kwargs):
        stems_input = {
            "vocals": vocals_source,
            "bass": bass_source,
            "drums": drums_source,
            "other": other_source,
            "identifier": "demo",
        }
        return vocals_source, no_vocals_source, stems_input

    def _fake_analyze_with_allin1(*_args, **kwargs):
        raw_response_path = kwargs.get("raw_response_path")
        if raw_response_path is not None:
            write_json(Path(raw_response_path), {"segments": [{"start": 0.0, "end": 10.0, "label": "verse"}], "beats": [0.0, 10.0]})
        return {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
            "beat_times": [0.0, 10.0],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 10.0, "type": "minor", "source": "allin1"},
            ],
        }

    def _fake_recognize_lyrics_with_funasr_v2(*_args, **_kwargs):
        return (
            [{"text": "hello", "timestamp": [[0, 1000]]}],
            [{"start_time": 0.0, "end_time": 1.0, "text": "hello", "confidence": 0.9}],
            {"dynamic_gap_threshold_seconds": 0.2, "sample_count_raw": 1},
        )

    def _fake_extract_acoustic_candidates_with_librosa(*_args, **_kwargs):
        return [0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.2, 0.3, 0.1]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.perception.prepare_stems_with_allin1_demucs",
        _fake_prepare_stems_with_allin1_demucs,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.perception.analyze_with_allin1",
        _fake_analyze_with_allin1,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.perception.recognize_lyrics_with_funasr_v2",
        _fake_recognize_lyrics_with_funasr_v2,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.perception.extract_acoustic_candidates_with_librosa",
        _fake_extract_acoustic_candidates_with_librosa,
    )

    run_perception_stage(
        audio_path=audio_path,
        duration_seconds=10.0,
        artifacts=artifacts,
        device="cpu",
        demucs_model="htdemucs",
        funasr_model="fake-model",
        funasr_language="auto",
        skip_funasr_when_vocals_silent=False,
        vocal_skip_peak_rms_threshold=0.01,
        vocal_skip_active_ratio_threshold=0.02,
        logger=logger,
    )

    assert not (artifacts.work_dir / "perception" / "model" / "demucs" / "stems").exists()
    assert artifacts.perception_model_allin1_raw_response_path.exists()
    assert artifacts.perception_model_funasr_raw_response_path.exists()
    assert artifacts.perception_model_funasr_lyric_sentence_units_path.exists()
    assert artifacts.perception_model_funasr_sentence_split_stats_path.exists()
    accompaniment_payload = read_json(artifacts.perception_signal_librosa_accompaniment_path)
    assert "onset_points" in accompaniment_payload
    assert len(list(accompaniment_payload.get("onset_points", []))) == len(list(accompaniment_payload.get("onset_candidates", [])))
    assert "chroma_points" in accompaniment_payload
    assert "f0_points_no_vocals" in accompaniment_payload
    vocal_payload = read_json(artifacts.perception_signal_librosa_vocal_candidates_path)
    assert "f0_points_vocals" in vocal_payload
