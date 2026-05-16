"""
文件用途：验证模块A V2感知层的线程编排与按需 FunASR 兜底行为。
核心流程：打桩感知依赖并记录时间窗，断言双轨Librosa并行与 FunASR 按需执行语义。
输入输出：输入临时目录与伪造依赖，输出断言结果。
依赖说明：依赖 pytest 与模块A V2感知层入口。
维护说明：若歌词主链优先级或并发模型调整，需同步更新本测试断言。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于线程名采集
import threading
# 标准库：用于时间统计与模拟耗时
import time
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：JSON读写
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：V2产物路径
from music_video_pipeline.modules.module_a_v2.artifacts import build_module_a_v2_artifacts
# 项目内模块：V2感知层入口
from music_video_pipeline.modules.module_a_v2.perception import run_perception_stage


# 常量：并行测试中Librosa任务的模拟耗时（秒）
PARALLEL_SLEEP_SECONDS = 0.30
# 常量：测试中 FunASR 任务的模拟耗时（秒）
FUNASR_SLEEP_SECONDS = 0.30


def _build_fake_stems(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """
    功能说明：创建测试所需的五轨 stem 假文件。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - tuple[Path, ...]: vocals/bass/drums/other/no_vocals 五个文件路径。
    异常说明：无。
    边界条件：仅写入最小二进制占位数据。
    """
    source_stems_dir = tmp_path / "source_stems"
    source_stems_dir.mkdir(parents=True, exist_ok=True)
    vocals_source = source_stems_dir / "vocals.wav"
    bass_source = source_stems_dir / "bass.wav"
    drums_source = source_stems_dir / "drums.wav"
    other_source = source_stems_dir / "other.wav"
    no_vocals_source = source_stems_dir / "no_vocals.wav"
    for path in [vocals_source, bass_source, drums_source, other_source, no_vocals_source]:
        path.write_bytes(b"fake-wav")
    return vocals_source, bass_source, drums_source, other_source, no_vocals_source


def test_run_perception_stage_should_run_funasr_only_after_main_chain_not_selected(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证非静音跳过场景下，双轨Librosa保持并行，且 FunASR 仅在主链未选中时按需执行。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：主链因无元数据而无法命中 LRCLIB，从而触发 FunASR 兜底。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_perception_parallel")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    vocals_source, bass_source, drums_source, other_source, no_vocals_source = _build_fake_stems(tmp_path)

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
            write_json(
                Path(raw_response_path),
                {"segments": [{"start": 0.0, "end": 10.0, "label": "verse"}], "beats": [0.0, 10.0]},
            )
        return {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
            "beat_times": [0.0, 10.0],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 10.0, "type": "minor", "source": "allin1"},
            ],
        }

    funasr_records: list[dict] = []

    def _fake_recognize_lyrics_with_funasr_v2(*_args, **_kwargs):
        start_at = time.perf_counter()
        thread_name = threading.current_thread().name
        time.sleep(FUNASR_SLEEP_SECONDS)
        end_at = time.perf_counter()
        funasr_records.append({"thread_name": thread_name, "start_at": start_at, "end_at": end_at})
        return (
            [{"text": "hello", "timestamp": [[0, 1000]]}],
            [{"start_time": 0.0, "end_time": 1.0, "text": "hello", "confidence": 0.9}],
            {"dynamic_gap_threshold_seconds": 0.2, "sample_count_raw": 1},
        )

    extract_records: list[dict] = []

    def _fake_extract_acoustic_candidates_with_librosa(audio_path, duration_seconds, logger, **kwargs):
        del duration_seconds, logger
        start_at = time.perf_counter()
        thread_name = threading.current_thread().name
        time.sleep(PARALLEL_SLEEP_SECONDS)
        end_at = time.perf_counter()
        track_name = Path(audio_path).name
        extract_records.append(
            {
                "track_name": track_name,
                "kwargs": dict(kwargs),
                "thread_name": thread_name,
                "start_at": start_at,
                "end_at": end_at,
            }
        )

        if track_name == "vocals.wav":
            return (
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 10.0],
                [0.2, 0.3, 0.1],
                [{"time": 0.0, "f0_hz": 220.0, "voiced": True, "confidence": 0.9}],
            )

        return (
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.1, 0.2, 0.1],
            [
                {"time": 0.0, "energy_raw": 0.0},
                {"time": 5.0, "energy_raw": 0.3},
                {"time": 10.0, "energy_raw": 0.0},
            ],
            [{"time": 0.0, "chroma": [0.0] * 12}],
            [{"time": 0.0, "f0_hz": 110.0, "voiced": True, "confidence": 0.8}],
        )

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
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        lyrics_enable_fingerprint_lookup=False,
        logger=logger,
    )

    dual_track_records = [item for item in extract_records if item["track_name"] in {"vocals.wav", "no_vocals.wav"}]
    assert len(dual_track_records) == 2
    vocals_record = next(item for item in dual_track_records if item["track_name"] == "vocals.wav")
    accompaniment_record = next(item for item in dual_track_records if item["track_name"] == "no_vocals.wav")
    librosa_overlap_start = max(float(vocals_record["start_at"]), float(accompaniment_record["start_at"]))
    librosa_overlap_end = min(float(vocals_record["end_at"]), float(accompaniment_record["end_at"]))
    assert librosa_overlap_start < librosa_overlap_end

    assert str(vocals_record["thread_name"]).startswith("librosa")
    assert str(accompaniment_record["thread_name"]).startswith("librosa")
    assert bool(vocals_record["kwargs"].get("with_f0_points", False))
    assert not bool(vocals_record["kwargs"].get("with_onset_points", False))
    assert not bool(vocals_record["kwargs"].get("with_chroma_points", False))
    assert bool(accompaniment_record["kwargs"].get("with_onset_points", False))
    assert bool(accompaniment_record["kwargs"].get("with_chroma_points", False))
    assert bool(accompaniment_record["kwargs"].get("with_f0_points", False))

    assert len(funasr_records) == 1
    funasr_record = funasr_records[0]
    assert str(funasr_record["thread_name"]) == "MainThread"
    librosa_parallel_end = max(float(vocals_record["end_at"]), float(accompaniment_record["end_at"]))
    assert float(funasr_record["start_at"]) >= librosa_parallel_end


def test_run_perception_stage_should_skip_funasr_when_precheck_silent(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证启用静音跳过时，precheck命中后不触发FunASR并写入skip产物。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：保留双轨Librosa提取，FunASR必须完全不被调用。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_perception_parallel_skip")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    vocals_source, bass_source, drums_source, other_source, no_vocals_source = _build_fake_stems(tmp_path)

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
            write_json(
                Path(raw_response_path),
                {"segments": [{"start": 0.0, "end": 10.0, "label": "verse"}], "beats": [0.0, 10.0]},
            )
        return {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
            "beat_times": [0.0, 10.0],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 10.0, "type": "minor", "source": "allin1"},
            ],
        }

    funasr_called = {"count": 0}

    def _fake_recognize_lyrics_with_funasr_v2(*_args, **_kwargs):
        funasr_called["count"] += 1
        return (
            [{"text": "unexpected", "timestamp": [[0, 1000]]}],
            [{"start_time": 0.0, "end_time": 1.0, "text": "unexpected", "confidence": 0.9}],
            {"dynamic_gap_threshold_seconds": 0.2, "sample_count_raw": 1},
        )

    def _fake_extract_acoustic_candidates_with_librosa(audio_path, duration_seconds, logger, **kwargs):
        del duration_seconds, logger
        track_name = Path(audio_path).name
        if track_name == "vocals.wav":
            on_rms_ready = kwargs.get("on_rms_ready")
            if callable(on_rms_ready):
                on_rms_ready([0.0, 0.5, 1.0], [0.0001, 0.0001, 0.0001])
            return (
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 10.0],
                [0.2, 0.3, 0.1],
                [{"time": 0.0, "f0_hz": 220.0, "voiced": True, "confidence": 0.9}],
            )
        return (
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.1, 0.2, 0.1],
            [
                {"time": 0.0, "energy_raw": 0.0},
                {"time": 5.0, "energy_raw": 0.3},
                {"time": 10.0, "energy_raw": 0.0},
            ],
            [{"time": 0.0, "chroma": [0.0] * 12}],
            [{"time": 0.0, "f0_hz": 110.0, "voiced": True, "confidence": 0.8}],
        )

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
        skip_funasr_when_vocals_silent=True,
        vocal_skip_peak_rms_threshold=0.01,
        vocal_skip_active_ratio_threshold=0.02,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        lyrics_enable_fingerprint_lookup=False,
        logger=logger,
    )

    assert funasr_called["count"] == 0
    funasr_raw_payload = read_json(artifacts.perception_model_funasr_raw_response_path)
    assert bool(funasr_raw_payload.get("skipped", False))
    assert str(funasr_raw_payload.get("reason", "")) == "silent_vocals_precheck"
    precheck_payload = read_json(artifacts.perception_signal_librosa_vocal_precheck_path)
    assert bool(precheck_payload.get("should_skip_funasr", False))
