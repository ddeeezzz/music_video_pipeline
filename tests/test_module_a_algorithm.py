"""
文件用途：验证模块A双时间戳与小段落契约的关键行为。
核心流程：在 fallback_only 模式执行模块A，检查大段落/小段落/小时戳的一致性。
输入输出：输入临时音频与运行上下文，输出断言结果。
依赖说明：依赖 pytest 与项目内 run_module_a 实现。
维护说明：当模块A契约变更时需同步更新本测试断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于模块注入
import sys

# 项目内模块：配置对象
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块A实现
from music_video_pipeline.modules.module_a import (
    _attach_lyrics_to_segments,
    _build_segments_with_lyric_priority,
    _clean_whisper_lyric_units,
    _recognize_lyrics_with_whisper,
    _rms_delta_at,
    _select_small_timestamps,
    run_module_a,
)
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def test_module_a_fallback_should_output_big_and_small_timestamps(tmp_path: Path) -> None:
    """
    功能说明：验证模块A在规则链下仍能输出大段落、小段落与小时戳。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入音频可为占位字节，测试目标是时间轴结构完整性。
    """
    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio-content")

    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("module_a_algorithm_test")
    logger.setLevel(logging.INFO)

    context = RuntimeContext(
        task_id="algo_task",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "algo_task",
        artifacts_dir=tmp_path / "runs" / "algo_task" / "artifacts",
        config=config,
        logger=logger,
        state_store=StateStore(db_path=tmp_path / "state.sqlite3"),
    )
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_module_a(context)
    output_data = read_json(output_path)

    big_segments = output_data["big_segments"]
    segments = output_data["segments"]
    beats = output_data["beats"]
    lyric_units = output_data["lyric_units"]

    assert len(big_segments) > 0
    assert len(segments) > 0
    assert len(beats) >= 2
    assert all("big_segment_id" in item for item in segments)
    assert all("segment_id" in item for item in lyric_units) or not lyric_units

    # 小段落必须连续覆盖，不应出现倒序与重叠
    for index in range(1, len(segments)):
        assert float(segments[index]["start_time"]) == float(segments[index - 1]["end_time"])

    # beats 语义为最终小时戳，必须升序
    beat_times = [float(item["time"]) for item in beats]
    assert beat_times == sorted(beat_times)


def _build_test_config(tmp_path: Path) -> AppConfig:
    """
    功能说明：构建模块A算法测试专用配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - AppConfig: 配置对象。
    异常说明：无。
    边界条件：module_a 使用 fallback_only，确保不依赖外部模型环境。
    """
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path="demo.mp3"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        mock=MockConfig(beat_interval_seconds=0.5, video_width=640, video_height=360),
        module_a=ModuleAConfig(whisper_language="auto", mode="fallback_only"),
    )


def test_module_a_instrumental_should_prefer_rms_delta_peak() -> None:
    """
    功能说明：验证器乐段在可用 onset 候选下优先选择能量突变最大的时间点。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过清空 beat 候选避免附加节拍干扰断言。
    """
    timestamps = _select_small_timestamps(
        duration_seconds=4.0,
        big_segments=[
            {
                "segment_id": "big_001",
                "start_time": 0.0,
                "end_time": 4.0,
                "label": "intro",
            }
        ],
        beat_candidates=[],
        onset_candidates=[1.0, 2.0, 3.0],
        rms_times=[0.9, 1.0, 1.9, 2.0, 2.9, 3.0],
        rms_values=[0.1, 0.3, 0.2, 1.2, 1.7, 1.8],
        lyric_sentence_starts=[],
        instrumental_labels=["intro"],
        snap_threshold_ms=200,
    )
    assert 2.0 in timestamps


def test_module_a_instrumental_should_fallback_to_rms_peak_when_delta_is_zero() -> None:
    """
    功能说明：验证器乐段在无明显突变时会回退到绝对能量峰值策略。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：delta 近似 0 时通过峰值差异断言回退行为。
    """
    timestamps = _select_small_timestamps(
        duration_seconds=3.0,
        big_segments=[
            {
                "segment_id": "big_001",
                "start_time": 0.0,
                "end_time": 3.0,
                "label": "inst",
            }
        ],
        beat_candidates=[],
        onset_candidates=[1.0, 2.0],
        rms_times=[1.0, 2.0],
        rms_values=[0.2, 0.9],
        lyric_sentence_starts=[],
        instrumental_labels=["inst"],
        snap_threshold_ms=200,
    )
    assert 2.0 in timestamps
    assert 1.0 not in [item for item in timestamps if item not in {0.0, 3.0}]


def test_rms_delta_should_only_keep_positive_change() -> None:
    """
    功能说明：验证 RMS 落差函数只返回正向增量，不返回负值。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：窗口前后值相等或下降时结果应为 0。
    """
    rms_times = [0.0, 0.1, 0.2, 0.3]
    rms_values = [0.2, 0.2, 0.5, 0.1]
    assert _rms_delta_at(0.2, rms_times, rms_values, window_ms=100.0) > 0.0
    assert _rms_delta_at(0.3, rms_times, rms_values, window_ms=100.0) == 0.0


def test_recognize_lyrics_with_whisper_should_not_pass_language_when_auto(monkeypatch) -> None:
    """
    功能说明：验证 whisper_language=auto 时调用 transcribe 不传 language 参数。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过假 whisper 模块隔离外部依赖与模型加载。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def transcribe(self, _audio_path: str, **kwargs):
            captured_kwargs.update(kwargs)
            return {"segments": [{"start": 0.0, "end": 0.5, "text": "测试歌词", "avg_logprob": -0.2}]}

    class _FakeWhisperModule:
        @staticmethod
        def load_model(_model_name: str, device=None):
            captured_kwargs["device"] = device
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "whisper", _FakeWhisperModule())
    sentence_starts, lyric_units = _recognize_lyrics_with_whisper(
        audio_path=Path("dummy.wav"),
        model_name="base",
        device="auto",
        whisper_language="auto",
        logger=logging.getLogger("whisper_auto_test"),
    )

    assert len(sentence_starts) == 1
    assert len(lyric_units) == 1
    assert "language" not in captured_kwargs
    assert captured_kwargs["word_timestamps"] is False
    assert captured_kwargs["device"] is None


def test_recognize_lyrics_with_whisper_should_pass_language_when_forced(monkeypatch) -> None:
    """
    功能说明：验证 whisper_language 指定时会透传 language 参数到 transcribe。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：语言值会归一化为小写。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def transcribe(self, _audio_path: str, **kwargs):
            captured_kwargs.update(kwargs)
            return {"segments": [{"start": 0.0, "end": 0.5, "text": "lyrics", "avg_logprob": -0.2}]}

    class _FakeWhisperModule:
        @staticmethod
        def load_model(_model_name: str, device=None):
            captured_kwargs["device"] = device
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "whisper", _FakeWhisperModule())
    _recognize_lyrics_with_whisper(
        audio_path=Path("dummy.wav"),
        model_name="base",
        device="cpu",
        whisper_language="JA",
        logger=logging.getLogger("whisper_forced_test"),
    )

    assert captured_kwargs["language"] == "ja"
    assert captured_kwargs["device"] == "cpu"


def test_recognize_lyrics_with_whisper_should_fallback_auto_when_language_invalid(monkeypatch, caplog) -> None:
    """
    功能说明：验证非法 whisper_language 会告警并回退为自动检测。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - caplog: pytest 日志捕获夹具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：非法值不应中断识别流程。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def transcribe(self, _audio_path: str, **kwargs):
            captured_kwargs.update(kwargs)
            return {"segments": [{"start": 0.0, "end": 0.5, "text": "lyrics", "avg_logprob": -0.2}]}

    class _FakeWhisperModule:
        @staticmethod
        def load_model(_model_name: str, device=None):
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "whisper", _FakeWhisperModule())
    caplog.set_level(logging.WARNING)

    _recognize_lyrics_with_whisper(
        audio_path=Path("dummy.wav"),
        model_name="base",
        device="auto",
        whisper_language="bad lang!",
        logger=logging.getLogger("whisper_invalid_test"),
    )

    assert "模块A-Whisper语言配置非法，已回退自动检测" in caplog.text
    assert "language" not in captured_kwargs


def test_clean_whisper_lyric_units_should_filter_noise_and_normalize_vocalise() -> None:
    """
    功能说明：验证歌词清洗会输出三态：未识别歌词/吟唱/正常歌词。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：器乐段误识别文本与明显噪声应被剔除。
    """
    logger = logging.getLogger("lyric_clean_test")
    big_segments = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 5.0, "label": "intro"},
        {"segment_id": "big_002", "start_time": 5.0, "end_time": 15.0, "label": "verse"},
    ]
    raw_units = [
        {"start_time": 1.0, "end_time": 2.0, "text": "era", "confidence": 0.0, "no_speech_prob": 0.95},
        {"start_time": 6.0, "end_time": 7.0, "text": "歌詞", "confidence": 0.7, "no_speech_prob": 0.1},
        {"start_time": 8.0, "end_time": 9.0, "text": "lalalala", "confidence": 0.1, "no_speech_prob": 0.2},
        {"start_time": 10.0, "end_time": 11.0, "text": "寂しいな", "confidence": 0.4, "no_speech_prob": 0.2},
    ]

    cleaned = _clean_whisper_lyric_units(
        lyric_units_raw=raw_units,
        big_segments=big_segments,
        instrumental_labels=["intro", "inst", "outro"],
        logger=logger,
    )
    assert len(cleaned) == 3
    assert cleaned[0]["text"] == "[未识别歌词]"
    assert cleaned[1]["text"] == "吟唱"
    assert cleaned[2]["text"] == "寂しいな"


def test_clean_whisper_lyric_units_should_mark_unknown_when_low_confidence_but_vocal() -> None:
    """
    功能说明：验证低置信但疑似有人声时会标记为“未识别歌词”。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：no_speech_prob 低时应优先保留人声存在信号。
    """
    cleaned = _clean_whisper_lyric_units(
        lyric_units_raw=[
            {"start_time": 3.0, "end_time": 3.8, "text": "？？", "confidence": 0.12, "no_speech_prob": 0.2},
        ],
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        logger=logging.getLogger("lyric_unknown_test"),
    )
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "[未识别歌词]"


def test_build_segments_with_lyric_priority_should_not_split_one_sentence() -> None:
    """
    功能说明：验证歌词句优先分段时，不会在一句歌词内部再切分片段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：歌词前后空档仍可按节拍切分。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=12.0,
        big_segments=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "verse"},
        ],
        beat_candidates=[0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 8.0, 12.0],
        onset_candidates=[1.8, 3.6, 9.2],
        lyric_units=[
            {"start_time": 2.0, "end_time": 4.0, "text": "一句歌词", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )

    target_segments = [
        item for item in segments if float(item["start_time"]) <= 2.0 and float(item["end_time"]) >= 4.0
    ]
    assert len(target_segments) == 1
    inner_boundaries = [
        float(item["start_time"])
        for item in segments
        if 2.0 < float(item["start_time"]) < 4.0
    ]
    assert inner_boundaries == []


def test_build_segments_with_lyric_priority_should_allow_sentence_cross_big_boundary() -> None:
    """
    功能说明：验证歌词跨大段落边界时仍保持一句一段，不在边界处强制拆分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：跨段歌词的 segment 可归属到起始大段落。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=8.0,
        big_segments=[
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 3.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 3.0, "end_time": 8.0, "label": "chorus"},
        ],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        onset_candidates=[1.5, 3.5, 6.8],
        lyric_units=[
            {"start_time": 2.5, "end_time": 4.5, "text": "跨段歌词", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )

    covering = [item for item in segments if float(item["start_time"]) <= 2.5 and float(item["end_time"]) >= 4.5]
    assert len(covering) == 1


def test_build_segments_with_lyric_priority_should_merge_micro_gap_between_sentences() -> None:
    """
    功能说明：验证歌词句之间微小空档不会单独生成镜头。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：句间 0.35 秒以内应被吸收。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        onset_candidates=[0.5, 2.5, 4.5],
        lyric_units=[
            {"start_time": 1.0, "end_time": 2.0, "text": "第一句", "confidence": 0.9},
            {"start_time": 2.2, "end_time": 3.0, "text": "第二句", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )
    boundaries = [float(item["start_time"]) for item in segments]
    assert 2.2 not in boundaries
    assert not any(
        abs(float(item["start_time"]) - 2.0) < 1e-6 and abs(float(item["end_time"]) - 2.2) < 1e-6
        for item in segments
    )


def test_build_segments_with_lyric_priority_should_merge_short_vocal_non_lyric_segments() -> None:
    """
    功能说明：验证人声段中小于 0.8 秒的非歌词碎片会与邻段合并。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：器乐标签不参与该合并策略。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        beat_candidates=[0.0, 2.0, 2.7, 4.0, 5.0, 6.0],
        onset_candidates=[],
        lyric_units=[
            {"start_time": 2.8, "end_time": 4.0, "text": "一句", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )
    # 2.0~2.7 的 0.7 秒碎片不应作为独立片段保留。
    assert not any(
        abs(float(item["start_time"]) - 2.0) < 1e-6 and abs(float(item["end_time"]) - 2.7) < 1e-6
        for item in segments
    )


def test_attach_lyrics_to_segments_should_choose_max_overlap_segment() -> None:
    """
    功能说明：验证边界歌词会绑定到重叠时间最长的片段，而非仅按起点绑定。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：歌词起点落在短片段内，但主体时长覆盖后续长片段。
    """
    segments = [
        {"segment_id": "seg_0001", "start_time": 21.931, "end_time": 22.250, "label": "intro"},
        {"segment_id": "seg_0002", "start_time": 22.250, "end_time": 26.000, "label": "verse"},
    ]
    lyric_units = [
        {"start_time": 22.000, "end_time": 26.000, "text": "跨边界歌词", "confidence": 0.9},
    ]
    attached = _attach_lyrics_to_segments(lyric_units_raw=lyric_units, segments=segments)
    assert attached[0]["segment_id"] == "seg_0002"
