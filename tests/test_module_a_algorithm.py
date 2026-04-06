"""
文件用途：验证模块A双时间戳与小段落契约的关键行为。
核心流程：在 fallback_only 模式执行模块A，检查大段落/小段落/小时戳的一致性。
输入输出：输入临时音频与运行上下文，输出断言结果。
依赖说明：依赖 pytest 与项目内 run_module_a 实现。
维护说明：当模块A契约变更时需同步更新本测试断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于进程ID校验
import os
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于模块注入
import sys
# 标准库：用于并发时序验证
import threading

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置对象
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块A命名空间（用于打桩）
from music_video_pipeline.modules import module_a as module_a_impl
# 项目内模块：模块A实现
from music_video_pipeline.modules.module_a import (
    _attach_lyrics_to_segments,
    _merge_short_vocal_non_lyric_ranges,
    _build_segmentation_anchor_lyric_units,
    _build_segmentation_tuning,
    _build_segments_with_lyric_priority,
    _build_visual_lyric_units,
    _clean_lyric_units,
    _recognize_lyrics_with_funasr,
    _run_real_pipeline,
    _rms_delta_at,
    _snap_to_nearest_beat,
    _select_small_timestamps,
    _split_instrumental_range_once_by_energy,
    run_module_a,
)
# 项目内模块：模块A分段内部函数（用于能量边界算法回归）
from music_video_pipeline.modules.module_a.segmentation import (
    _build_big_segments_v2_by_lyric_overlap,
    _build_mid_segments_by_vocal_energy,
    _merge_short_inst_gaps_between_vocal_ranges,
)
# 项目内模块：子进程隔离执行工具（用于并行隔离行为验证）
from music_video_pipeline.modules.module_a.orchestrator import _run_callable_in_spawn_process
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def _return_process_id_for_spawn_test() -> int:
    """
    功能说明：返回当前进程 PID，用于验证 spawn 隔离行为。
    参数说明：无。
    返回值：
    - int: 当前进程ID。
    异常说明：无。
    边界条件：仅用于测试，不参与业务链路。
    """
    return int(os.getpid())


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
        module_a=ModuleAConfig(funasr_language="auto", mode="fallback_only"),
    )


def test_snap_to_nearest_beat_should_snap_when_diff_within_threshold() -> None:
    """
    功能说明：验证目标时间与最近节拍差值在阈值内时会发生吸附。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：阈值为正时按闭区间语义处理。
    """
    snapped_time = _snap_to_nearest_beat(
        target_time=1.05,
        beat_pool=[1.0, 2.0],
        threshold_seconds=0.2,
    )
    assert snapped_time == 1.0


def test_snap_to_nearest_beat_should_keep_target_when_diff_over_threshold() -> None:
    """
    功能说明：验证目标时间与最近节拍差值超过阈值时保持原始时间。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验超阈值不吸附语义，不涉及上游分段逻辑。
    """
    snapped_time = _snap_to_nearest_beat(
        target_time=1.35,
        beat_pool=[1.0, 2.0],
        threshold_seconds=0.2,
    )
    assert snapped_time == 1.35


def test_snap_to_nearest_beat_should_snap_when_diff_equals_threshold() -> None:
    """
    功能说明：验证差值等于阈值时按闭区间语义执行吸附。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：阈值边界值必须稳定，防止后续比较符号回归。
    """
    snapped_time = _snap_to_nearest_beat(
        target_time=1.2,
        beat_pool=[1.0, 2.0],
        threshold_seconds=0.2,
    )
    assert snapped_time == 1.0


def test_build_segmentation_tuning_should_be_single_normalization_entry() -> None:
    """
    功能说明：验证分段阈值只在统一入口归一化，原始参数路径与 tuning 对象路径行为一致。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入包含越界值时，需稳定裁剪到安全区间。
    """
    normalized_tuning = _build_segmentation_tuning(
        vocal_energy_enter_quantile=1.8,
        vocal_energy_exit_quantile=-0.3,
        mid_segment_min_duration_seconds=-2.0,
        short_vocal_non_lyric_merge_seconds=-0.1,
        instrumental_single_split_min_seconds=0.0,
        accent_delta_trigger_ratio=2.5,
        lyric_sentence_gap_merge_seconds=-1.0,
    )
    assert normalized_tuning.vocal_energy_enter_quantile == 1.0
    assert normalized_tuning.vocal_energy_exit_quantile == 0.0
    assert normalized_tuning.mid_segment_min_duration_seconds == 0.1
    assert normalized_tuning.short_vocal_non_lyric_merge_seconds == 0.1
    assert normalized_tuning.instrumental_single_split_min_seconds == 0.1
    assert normalized_tuning.accent_delta_trigger_ratio == 1.0
    assert normalized_tuning.lyric_sentence_gap_merge_seconds == 0.0

    common_kwargs = dict(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        onset_candidates=[1.0, 2.0, 4.0],
        lyric_units=[],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        rms_values=[0.2, 0.35, 0.45, 0.4, 0.5, 0.42],
        vocal_onset_candidates=[1.0, 2.0, 4.0],
        vocal_rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        vocal_rms_values=[0.6, 0.8, 0.85, 0.82, 0.79, 0.7],
    )
    segments_by_raw = _build_segments_with_lyric_priority(
        **common_kwargs,
        vocal_energy_enter_quantile=1.8,
        vocal_energy_exit_quantile=-0.3,
        mid_segment_min_duration_seconds=-2.0,
        short_vocal_non_lyric_merge_seconds=-0.1,
        instrumental_single_split_min_seconds=0.0,
        accent_delta_trigger_ratio=2.5,
    )
    segments_by_tuning = _build_segments_with_lyric_priority(
        **common_kwargs,
        tuning=normalized_tuning,
    )
    assert segments_by_raw == segments_by_tuning


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


def test_build_segments_with_lyric_priority_should_split_long_instrumental_once_by_max_delta() -> None:
    """
    功能说明：验证器乐长段仅切一次，切点选择能量落差最大的候选点。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使存在多个 beat/onset 候选，也不得按节拍连续切分。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=10.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "intro"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        onset_candidates=[2.0, 6.0, 8.0],
        lyric_units=[],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[1.9, 2.0, 5.9, 6.0, 7.9, 8.0],
        rms_values=[0.1, 0.2, 0.2, 1.2, 1.2, 1.25],
    )
    assert len(segments) == 2
    assert float(segments[0]["start_time"]) == 0.0
    assert float(segments[0]["end_time"]) == 6.0
    assert float(segments[1]["start_time"]) == 6.0
    assert float(segments[1]["end_time"]) == 10.0


def test_build_segments_with_lyric_priority_should_not_split_short_instrumental_segment() -> None:
    """
    功能说明：验证器乐短段（<4秒）保持单段，不做单次能量切分。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：候选点充足时也应保持单段，避免过度切分。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=3.5,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 3.5, "label": "intro"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.5],
        onset_candidates=[1.0, 2.0],
        lyric_units=[],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.9, 1.0, 1.9, 2.0],
        rms_values=[0.2, 0.8, 0.3, 0.9],
    )
    assert len(segments) == 1
    assert float(segments[0]["start_time"]) == 0.0
    assert float(segments[0]["end_time"]) == 3.5


def test_split_instrumental_range_once_by_energy_should_not_split_when_side_shorter_than_1_2s() -> None:
    """
    功能说明：验证器乐单次切分在任一侧小于 1.2 秒时会回退为单段，避免生成短 inst 边。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当区间长度不足以满足两侧最小时长时，即使触发长段逻辑也不得切分。
    """
    ranges = _split_instrumental_range_once_by_energy(
        start_time=0.0,
        end_time=2.2,
        beat_pool=[],
        onset_pool=[],
        rms_times=[],
        rms_values=[],
        long_segment_threshold_seconds=2.0,
    )
    assert ranges == [(0.0, 2.2)]


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


def test_recognize_lyrics_with_funasr_should_not_pass_language_when_auto(monkeypatch) -> None:
    """
    功能说明：验证 funasr_language=auto 时调用 generate 不传 language 参数。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过假 funasr 模块隔离外部依赖与模型加载。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            captured_kwargs["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                {
                    "text": "テスト歌詞",
                    "sentence_info": [
                        {
                            "text": "テスト歌詞",
                            "start": 0,
                            "end": 500,
                            "timestamp": [[0, 240], [240, 500]],
                            "score": 0.92,
                        }
                    ],
                }
            ]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    sentence_starts, lyric_units = _recognize_lyrics_with_funasr(
        audio_path=Path("dummy.wav"),
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="auto",
        funasr_language="auto",
        logger=logging.getLogger("funasr_auto_test"),
    )

    assert len(sentence_starts) == 1
    assert len(lyric_units) == 1
    assert "language" not in captured_kwargs
    assert "device" not in captured_kwargs["model_kwargs"]
    assert set(captured_kwargs["model_kwargs"].keys()) == {"model", "vad_model", "vad_kwargs"}
    assert captured_kwargs["model_kwargs"]["vad_model"] == "fsmn-vad"
    assert lyric_units[0]["token_units"][0]["granularity"] == "char"


def test_recognize_lyrics_with_funasr_should_pass_language_when_forced(monkeypatch) -> None:
    """
    功能说明：验证 funasr_language 指定时会透传 language 参数到 generate。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：语言值会归一化为小写。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            captured_kwargs["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                {
                    "text": "lyrics",
                    "sentence_info": [
                        {"text": "lyrics", "start": 0, "end": 500, "timestamp": [[0, 500]], "score": 0.6}
                    ],
                }
            ]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    _recognize_lyrics_with_funasr(
        audio_path=Path("dummy.wav"),
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="cpu",
        funasr_language="JA",
        logger=logging.getLogger("funasr_forced_test"),
    )

    assert captured_kwargs["language"] == "ja"
    assert captured_kwargs["model_kwargs"]["device"] == "cpu"


def test_recognize_lyrics_with_funasr_should_fallback_auto_when_language_invalid(monkeypatch, caplog) -> None:
    """
    功能说明：验证非法 funasr_language 会告警并回退为自动检测。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - caplog: pytest 日志捕获夹具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：非法值不应中断识别流程。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                {
                    "text": "lyrics",
                    "sentence_info": [
                        {"text": "lyrics", "start": 0, "end": 500, "timestamp": [[0, 500]], "score": 0.6}
                    ],
                }
            ]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    caplog.set_level(logging.WARNING)

    _recognize_lyrics_with_funasr(
        audio_path=Path("dummy.wav"),
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="auto",
        funasr_language="bad lang!",
        logger=logging.getLogger("funasr_invalid_test"),
    )

    assert "模块A-FunASR语言配置非法，已回退自动检测" in caplog.text
    assert "language" not in captured_kwargs


def test_recognize_lyrics_with_funasr_should_build_units_from_timestamp_when_sentence_info_missing(monkeypatch) -> None:
    """
    功能说明：验证当 sentence_info 缺失时，可由 text + timestamp 回退生成句级歌词单元。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：timestamp 采用二维数组时应生成 token_units。
    """
    class _FakeModel:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **_kwargs):
            return [{"text": "あいしてる。", "timestamp": [[0, 180], [180, 340], [340, 520], [520, 700], [700, 860], [860, 1000]]}]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    _, lyric_units = _recognize_lyrics_with_funasr(
        audio_path=Path("dummy.wav"),
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="auto",
        funasr_language="ja",
        logger=logging.getLogger("funasr_timestamp_fallback_test"),
    )
    assert lyric_units
    assert lyric_units[0]["text"] == "あいしてる。"
    assert lyric_units[0]["token_units"]


def test_recognize_lyrics_with_funasr_should_build_units_from_timestamps_dict(monkeypatch) -> None:
    """
    功能说明：验证 FunASR 返回 timestamps(dict[]) 时可正确生成句级歌词与 token_units。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：timestamps 键名为复数，且时间字段为 start_time/end_time。
    """
    class _FakeModel:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **_kwargs):
            return [
                {
                    "text": "テスト。次の行。",
                    "timestamps": [
                        {"token": "テ", "start_time": 0.1, "end_time": 0.2, "score": 0.9},
                        {"token": "ス", "start_time": 0.2, "end_time": 0.3, "score": 0.9},
                        {"token": "ト", "start_time": 0.3, "end_time": 0.4, "score": 0.9},
                        {"token": "。", "start_time": 0.4, "end_time": 0.5, "score": 0.9},
                        {"token": "次", "start_time": 0.6, "end_time": 0.7, "score": 0.9},
                        {"token": "の", "start_time": 0.7, "end_time": 0.8, "score": 0.9},
                        {"token": "行", "start_time": 0.8, "end_time": 0.9, "score": 0.9},
                        {"token": "。", "start_time": 0.9, "end_time": 1.0, "score": 0.9},
                    ],
                }
            ]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    _, lyric_units = _recognize_lyrics_with_funasr(
        audio_path=Path("dummy.wav"),
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="auto",
        funasr_language="ja",
        logger=logging.getLogger("funasr_timestamps_dict_test"),
    )

    assert len(lyric_units) == 2
    assert lyric_units[0]["text"] == "テスト。"
    assert lyric_units[0]["token_units"][0]["text"] == "テ"


def test_recognize_lyrics_with_funasr_should_raise_when_text_without_timestamp(monkeypatch) -> None:
    """
    功能说明：验证 FunASR 返回文本但无可用时间戳时会抛异常。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该行为用于 real_strict 的严格链路保障。
    """
    class _FakeModel:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **_kwargs):
            return [{"text": "有歌词但无时间戳"}]

    class _FakeFunASRModule:
        AutoModel = _FakeModel

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    with pytest.raises(RuntimeError):
        _recognize_lyrics_with_funasr(
            audio_path=Path("dummy.wav"),
            model_name="FunAudioLLM/Fun-ASR-Nano-2512",
            device="auto",
            funasr_language="auto",
            logger=logging.getLogger("funasr_no_timestamp_test"),
        )


def test_recognize_lyrics_with_funasr_should_raise_diagnostic_when_model_not_registered(monkeypatch) -> None:
    """
    功能说明：验证 Nano 未注册时会抛出包含版本信息的可执行诊断。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该诊断用于引导切换到 README 对齐的 Git 锁定版依赖。
    """
    class _FakeFunASRModule:
        __version__ = "1.3.1"

        class AutoModel:
            def __init__(self, **_kwargs):
                raise AssertionError("FunASRNano is not registered")

    monkeypatch.setitem(sys.modules, "funasr", _FakeFunASRModule())
    with pytest.raises(RuntimeError, match="FunASRNano 未注册"):
        _recognize_lyrics_with_funasr(
            audio_path=Path("dummy.wav"),
            model_name="FunAudioLLM/Fun-ASR-Nano-2512",
            device="auto",
            funasr_language="ja",
            logger=logging.getLogger("funasr_not_registered_test"),
        )


def test_run_real_pipeline_should_degrade_lyrics_chain_when_not_strict(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证非 strict 模式下歌词识别失败会降级为空歌词链且不阻断真实链路输出。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅模拟歌词链失败，其他链路最小打桩。
    """
    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.8, 0.9, 0.7]),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_recognize_lyrics_with_funasr",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("FunASRNano 未注册")),
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(
        module_a_impl,
        "_build_beats_from_segments",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("真实链路不应再调用 _build_beats_from_segments")),
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="ja",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_non_strict_test"),
    )
    assert output["lyric_units"] == []
    assert output["segments"]
    assert output["beats"]
    assert "big_segments_stage1" in output
    assert output["big_segments_stage1"] == output["big_segments"]
    assert all(str(item.get("source", "")) == "allin1" for item in output["beats"])


def test_run_real_pipeline_should_build_segments_from_big_v2(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证真实链路会用 big_v2 直接驱动分段主链，而非先按 stage1 分段后再后切。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该用例只验证调用路径，不验证分段算法细节。
    """
    big_stage1 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 5.0, "label": "verse"},
        {"segment_id": "big_002", "start_time": 5.0, "end_time": 10.0, "label": "chorus"},
    ]
    big_v2 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 4.2, "label": "verse"},
        {"segment_id": "big_002", "start_time": 4.2, "end_time": 10.0, "label": "chorus"},
    ]
    anchor_units = [
        {
            "start_time": 4.0,
            "end_time": 6.0,
            "text": "跨界歌词",
            "confidence": 0.9,
            "token_units": [
                {"text": "跨", "start_time": 4.0, "end_time": 4.5, "granularity": "char"},
                {"text": "界", "start_time": 4.5, "end_time": 5.0, "granularity": "char"},
                {"text": "歌", "start_time": 5.0, "end_time": 5.5, "granularity": "char"},
                {"text": "词", "start_time": 5.5, "end_time": 6.0, "granularity": "char"},
            ],
        }
    ]

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": big_stage1,
            "beat_times": [0.0, 2.5, 5.0, 7.5, 10.0],
            "beat_positions": [1, 2, 3, 4, 1],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.5, "type": "minor", "source": "allin1"},
                {"time": 5.0, "type": "minor", "source": "allin1"},
                {"time": 7.5, "type": "minor", "source": "allin1"},
                {"time": 10.0, "type": "major", "source": "allin1"},
            ],
        },
    )
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.5, 5.0, 7.5, 10.0], [1.0, 4.0, 6.0, 9.0], [0.0, 5.0, 10.0], [0.5, 0.7, 0.6]),
    )
    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", lambda *args, **kwargs: ([], anchor_units))
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: anchor_units)
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: anchor_units)
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: anchor_units)
    monkeypatch.setattr(module_a_impl, "_build_big_segments_v2_by_lyric_overlap", lambda **kwargs: big_v2)
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.2, 10.0])

    captured: dict[str, list[dict[str, float | str]]] = {}

    def _fake_build_segments_with_lyric_priority(**kwargs):
        captured["big_segments"] = kwargs["big_segments"]
        return [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.2, "label": "verse"},
            {"segment_id": "seg_0002", "big_segment_id": "big_002", "start_time": 4.2, "end_time": 10.0, "label": "chorus"},
        ]

    monkeypatch.setattr(module_a_impl, "_build_segments_with_lyric_priority", _fake_build_segments_with_lyric_priority)
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: anchor_units)
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 10.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=10.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_big_v2_path_test"),
    )

    assert captured["big_segments"] == big_v2
    assert output["big_segments"] == big_v2
    assert output["big_segments_stage1"] == big_stage1
    assert all(str(item.get("source", "")) == "allin1" for item in output["beats"])


def test_run_real_pipeline_should_raise_when_strict_and_registration_failed(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 strict 模式下歌词识别失败会直接抛错而非静默降级。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只验证 strict 分支错误传播，不依赖真实模型环境。
    """
    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.8, 0.9, 0.7]),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_recognize_lyrics_with_funasr",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("FunASRNano 未注册")),
    )

    with pytest.raises(RuntimeError, match="strict 模式要求歌词时间戳可用"):
        _run_real_pipeline(
            audio_path=tmp_path / "demo.wav",
            duration_seconds=4.0,
            work_dir=tmp_path / "work",
            snap_threshold_ms=200,
            instrumental_labels=["intro", "inst", "outro"],
            device="auto",
            funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
            funasr_language="ja",
            lyric_segment_policy="sentence_strict",
            comma_pause_seconds=0.45,
            long_pause_seconds=0.8,
            merge_gap_seconds=0.25,
            max_visual_unit_seconds=6.0,
            demucs_model="htdemucs",
            beat_interval_seconds=0.5,
            strict_lyric_timestamps=True,
            logger=logging.getLogger("run_real_pipeline_strict_test"),
        )


def test_run_callable_in_spawn_process_should_run_in_child_process() -> None:
    """
    功能说明：验证 spawn 隔离执行会在独立子进程中运行目标函数。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证进程隔离，不依赖模块A外部模型。
    """
    parent_pid = int(os.getpid())
    child_pid = _run_callable_in_spawn_process(
        callable_obj=_return_process_id_for_spawn_test,
        kwargs={},
        logger_name="module_a_spawn_pid_test",
        task_label="pid_probe",
    )
    assert isinstance(child_pid, int)
    assert child_pid != parent_pid


def test_run_real_pipeline_should_start_funasr_before_allin1_completes(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 Demucs 完成后 FunASR 可与 allin1 主分析并行启动，不等待 allin1 完成。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过事件栅栏验证并行时序，不依赖真实模型。
    """
    analyze_entered = threading.Event()
    funasr_started = threading.Event()
    allow_analyze_finish = threading.Event()

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )

    def _fake_analyze_with_allin1(*_args, **_kwargs):
        analyze_entered.set()
        assert funasr_started.wait(1.0)
        assert allow_analyze_finish.wait(1.0)
        return {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        }

    def _fake_recognize(*_args, **_kwargs):
        funasr_started.set()
        assert analyze_entered.wait(1.0)
        allow_analyze_finish.set()
        return [], []

    monkeypatch.setattr(module_a_impl, "_analyze_with_allin1", _fake_analyze_with_allin1)
    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", _fake_recognize)
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.6, 0.8, 0.7]),
    )
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_parallel_order_test"),
    )

    assert funasr_started.is_set()
    assert analyze_entered.is_set()
    assert output["beats"]
    assert all(str(item.get("source", "")) == "allin1" for item in output["beats"])


def test_run_real_pipeline_should_progress_non_lyric_steps_before_funasr_done(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 allin1 完成后可先推进无歌词依赖步骤，再等待 FunASR 收尾。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过事件栅栏约束执行顺序，不依赖真实模型。
    """
    allow_funasr_finish = threading.Event()
    acoustic_started = threading.Event()

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )

    def _fake_recognize(*_args, **_kwargs):
        assert allow_funasr_finish.wait(1.0)
        return [], []

    def _fake_extract_acoustic(*_args, **_kwargs):
        acoustic_started.set()
        allow_funasr_finish.set()
        return [0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.6, 0.8, 0.7]

    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", _fake_recognize)
    monkeypatch.setattr(module_a_impl, "_extract_acoustic_candidates_with_librosa", _fake_extract_acoustic)
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_parallel_non_lyric_progress_test"),
        skip_funasr_when_vocals_silent=False,
    )

    assert acoustic_started.is_set()
    assert output["beats"]
    assert all(str(item.get("source", "")) == "allin1" for item in output["beats"])


def test_run_real_pipeline_should_skip_funasr_when_vocals_silent(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 vocals 能量极低命中阈值时会主动跳过 FunASR。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证跳过判定，不依赖真实模型。
    """
    call_count = {"funasr": 0}

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )

    def _fake_extract_acoustic(audio_path: Path, *_args, **_kwargs):
        if str(audio_path).endswith("vocals.wav"):
            return [0.0, 2.0, 4.0], [1.0], [0.0, 2.0, 4.0], [0.001, 0.002, 0.001]
        return [0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.6, 0.7, 0.5]

    def _fake_recognize(*_args, **_kwargs):
        call_count["funasr"] += 1
        return [], []

    monkeypatch.setattr(module_a_impl, "_extract_acoustic_candidates_with_librosa", _fake_extract_acoustic)
    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", _fake_recognize)
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_skip_funasr_silent_test"),
        skip_funasr_when_vocals_silent=True,
        vocal_skip_peak_rms_threshold=0.01,
        vocal_skip_active_ratio_threshold=0.02,
    )

    assert call_count["funasr"] == 0
    assert output["lyric_units"] == []
    assert output["beats"]


def test_run_real_pipeline_should_allow_silent_skip_in_strict_mode(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 strict 模式下因无人声判定跳过 FunASR 时不抛错。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过将 FunASR 打桩为抛错来确认确实被跳过。
    """
    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0], [0.0, 2.0, 4.0], [0.001, 0.002, 0.001]),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_recognize_lyrics_with_funasr",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("strict 模式下此调用应被跳过")),
    )
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=True,
        logger=logging.getLogger("run_real_pipeline_strict_silent_skip_test"),
        skip_funasr_when_vocals_silent=True,
        vocal_skip_peak_rms_threshold=0.01,
        vocal_skip_active_ratio_threshold=0.02,
    )

    assert output["lyric_units"] == []
    assert output["segments"]


def test_run_real_pipeline_should_still_call_funasr_when_skip_disabled(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证关闭“静音跳过”后，即使低能量也会调用 FunASR。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验调用行为，不校验识别内容质量。
    """
    call_count = {"funasr": 0}

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )
    monkeypatch.setattr(
        module_a_impl,
        "_analyze_with_allin1",
        lambda *args, **kwargs: {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        },
    )
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0], [0.0, 2.0, 4.0], [0.001, 0.002, 0.001]),
    )

    def _fake_recognize(*_args, **_kwargs):
        call_count["funasr"] += 1
        return [], []

    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", _fake_recognize)
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_skip_disabled_test"),
        skip_funasr_when_vocals_silent=False,
    )

    assert call_count["funasr"] == 1


def test_run_real_pipeline_should_fallback_to_serial_when_parallel_failed(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证并行阶段异常后会自动回退串行执行并完成输出。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：并行 allin1 人为失败，串行重试应成功。
    """
    call_count = {"allin1": 0, "funasr": 0}
    main_thread_id = threading.main_thread().ident

    monkeypatch.setattr(
        module_a_impl,
        "_prepare_stems_with_allin1_demucs",
        lambda *args, **kwargs: (Path("vocals.wav"), Path("no_vocals.wav"), {"vocals": Path("vocals.wav"), "other": Path("other.wav"), "bass": Path("bass.wav"), "drums": Path("drums.wav")}),
    )

    def _fake_analyze_with_allin1(*_args, **_kwargs):
        call_count["allin1"] += 1
        if threading.get_ident() != main_thread_id:
            raise RuntimeError("parallel allin1 failed")
        return {
            "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
            "beat_times": [0.0, 2.0, 4.0],
            "beat_positions": [1, 2, 3],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "minor", "source": "allin1"},
            ],
        }

    monkeypatch.setattr(module_a_impl, "_analyze_with_allin1", _fake_analyze_with_allin1)

    def _fake_recognize(*args, **kwargs):
        call_count["funasr"] += 1
        return [], []

    monkeypatch.setattr(module_a_impl, "_recognize_lyrics_with_funasr", _fake_recognize)
    monkeypatch.setattr(
        module_a_impl,
        "_extract_acoustic_candidates_with_librosa",
        lambda *args, **kwargs: ([0.0, 2.0, 4.0], [1.0, 3.0], [0.0, 2.0, 4.0], [0.6, 0.8, 0.7]),
    )
    monkeypatch.setattr(module_a_impl, "_clean_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_visual_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(module_a_impl, "_build_segmentation_anchor_lyric_units", lambda **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_big_segments_v2_by_lyric_overlap",
        lambda **kwargs: kwargs["big_segments_stage1"],
    )
    monkeypatch.setattr(module_a_impl, "_select_small_timestamps", lambda **kwargs: [0.0, 4.0])
    monkeypatch.setattr(
        module_a_impl,
        "_build_segments_with_lyric_priority",
        lambda **kwargs: [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
    )
    monkeypatch.setattr(module_a_impl, "_attach_lyrics_to_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module_a_impl,
        "_build_energy_features",
        lambda *args, **kwargs: [{"start_time": 0.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )

    output = _run_real_pipeline(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=4.0,
        work_dir=tmp_path / "work",
        snap_threshold_ms=200,
        instrumental_labels=["intro", "inst", "outro"],
        device="auto",
        funasr_model="FunAudioLLM/Fun-ASR-Nano-2512",
        funasr_language="zh",
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        demucs_model="htdemucs",
        beat_interval_seconds=0.5,
        strict_lyric_timestamps=False,
        logger=logging.getLogger("run_real_pipeline_parallel_fallback_test"),
    )

    assert call_count["allin1"] == 2
    assert call_count["funasr"] == 1
    assert output["beats"]
    assert all(str(item.get("source", "")) == "allin1" for item in output["beats"])


def test_clean_lyric_units_should_filter_noise_and_normalize_vocalise() -> None:
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

    cleaned = _clean_lyric_units(
        lyric_units_raw=raw_units,
        big_segments=big_segments,
        instrumental_labels=["intro", "inst", "outro"],
        logger=logger,
    )
    assert len(cleaned) == 3
    assert cleaned[0]["text"] == "[未识别歌词]"
    assert cleaned[1]["text"] == "吟唱"
    assert cleaned[2]["text"] == "寂しいな"


def test_clean_lyric_units_should_mark_unknown_when_low_confidence_but_vocal() -> None:
    """
    功能说明：验证低置信但疑似有人声时会标记为“未识别歌词”。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：no_speech_prob 低时应优先保留人声存在信号。
    """
    cleaned = _clean_lyric_units(
        lyric_units_raw=[
            {"start_time": 3.0, "end_time": 3.8, "text": "？？", "confidence": 0.12, "no_speech_prob": 0.2},
        ],
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        logger=logging.getLogger("lyric_unknown_test"),
    )
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "[未识别歌词]"


def test_build_visual_lyric_units_sentence_strict_should_keep_sentence_units() -> None:
    """
    功能说明：验证 sentence_strict 策略不会改变句级歌词单元边界。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅测试策略层，不依赖外部模型。
    """
    sentence_units = [
        {
            "start_time": 10.0,
            "end_time": 11.2,
            "text": "第一句。",
            "confidence": 0.9,
            "token_units": [
                {"text": "第", "start_time": 10.0, "end_time": 10.2, "granularity": "char"},
                {"text": "一", "start_time": 10.2, "end_time": 10.4, "granularity": "char"},
                {"text": "句", "start_time": 10.4, "end_time": 10.8, "granularity": "char"},
                {"text": "。", "start_time": 10.8, "end_time": 11.2, "granularity": "char"},
            ],
        }
    ]
    visual_units = _build_visual_lyric_units(
        sentence_units=sentence_units,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 20.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        lyric_segment_policy="sentence_strict",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        logger=logging.getLogger("visual_unit_strict_test"),
    )
    assert len(visual_units) == 1
    assert visual_units[0]["start_time"] == 10.0
    assert visual_units[0]["end_time"] == 11.2
    assert visual_units[0]["text"] == "第一句。"
    assert visual_units[0]["source_sentence_index"] == 0
    assert visual_units[0]["unit_transform"] == "original"


def test_build_visual_lyric_units_adaptive_should_split_by_comma_pause() -> None:
    """
    功能说明：验证 adaptive_phrase 策略可按“逗号+停顿”将长句拆为多视觉单元。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：逗号前后停顿低于阈值时不应触发拆分。
    """
    sentence_units = [
        {
            "start_time": 0.0,
            "end_time": 4.0,
            "text": "甲乙、丙丁、戊己。",
            "confidence": 0.8,
            "token_units": [
                {"text": "甲", "start_time": 0.0, "end_time": 0.3, "granularity": "char"},
                {"text": "乙", "start_time": 0.3, "end_time": 0.6, "granularity": "char"},
                {"text": "、", "start_time": 1.2, "end_time": 1.3, "granularity": "char"},
                {"text": "丙", "start_time": 1.3, "end_time": 1.6, "granularity": "char"},
                {"text": "丁", "start_time": 1.6, "end_time": 1.9, "granularity": "char"},
                {"text": "、", "start_time": 2.5, "end_time": 2.6, "granularity": "char"},
                {"text": "戊", "start_time": 2.6, "end_time": 2.9, "granularity": "char"},
                {"text": "己", "start_time": 2.9, "end_time": 3.2, "granularity": "char"},
                {"text": "。", "start_time": 3.2, "end_time": 4.0, "granularity": "char"},
            ],
        }
    ]
    visual_units = _build_visual_lyric_units(
        sentence_units=sentence_units,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        lyric_segment_policy="adaptive_phrase",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.0,
        max_visual_unit_seconds=6.0,
        logger=logging.getLogger("visual_unit_split_test"),
    )
    assert len(visual_units) == 3
    assert all(item["unit_transform"] == "split" for item in visual_units)
    assert visual_units[0]["text"].endswith("、")
    assert visual_units[-1]["text"].endswith("。")


def test_build_visual_lyric_units_adaptive_should_left_attach_leading_punctuation_after_long_gap() -> None:
    """
    功能说明：验证 adaptive_phrase 在长停顿切分后，会将后句开头标点左归属到前句。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅调整 token 归属，不改标点 token 原始时间戳。
    """
    sentence_units = [
        {
            "start_time": 10.0,
            "end_time": 12.0,
            "text": "看这里，彩色",
            "confidence": 0.9,
            "token_units": [
                {"text": "看", "start_time": 10.00, "end_time": 10.10, "granularity": "char"},
                {"text": "这", "start_time": 10.12, "end_time": 10.22, "granularity": "char"},
                {"text": "里", "start_time": 10.24, "end_time": 10.34, "granularity": "char"},
                {"text": "，", "start_time": 11.30, "end_time": 11.35, "granularity": "char"},
                {"text": "彩", "start_time": 11.36, "end_time": 11.46, "granularity": "char"},
                {"text": "色", "start_time": 11.48, "end_time": 11.58, "granularity": "char"},
            ],
        }
    ]
    visual_units = _build_visual_lyric_units(
        sentence_units=sentence_units,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 20.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        lyric_segment_policy="adaptive_phrase",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.0,
        max_visual_unit_seconds=6.0,
        logger=logging.getLogger("visual_unit_punctuation_left_attach_test"),
    )
    assert len(visual_units) == 2
    assert [item["text"] for item in visual_units] == ["看这里，", "彩色"]
    assert all(item["unit_transform"] == "split" for item in visual_units)


def test_build_visual_lyric_units_adaptive_should_merge_short_sentences() -> None:
    """
    功能说明：验证 adaptive_phrase 策略可合并相邻短句，减少碎片化镜头。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：间隔超过 merge_gap_seconds 时不应合并。
    """
    sentence_units = [
        {
            "start_time": 10.0,
            "end_time": 11.0,
            "text": "第一句。",
            "confidence": 0.9,
            "token_units": [
                {"text": "第一句。", "start_time": 10.0, "end_time": 11.0, "granularity": "word"},
            ],
        },
        {
            "start_time": 11.1,
            "end_time": 12.0,
            "text": "第二句。",
            "confidence": 0.85,
            "token_units": [
                {"text": "第二句。", "start_time": 11.1, "end_time": 12.0, "granularity": "word"},
            ],
        },
    ]
    visual_units = _build_visual_lyric_units(
        sentence_units=sentence_units,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 20.0, "label": "verse"}],
        instrumental_labels=["intro", "inst", "outro"],
        lyric_segment_policy="adaptive_phrase",
        comma_pause_seconds=0.45,
        long_pause_seconds=0.8,
        merge_gap_seconds=0.25,
        max_visual_unit_seconds=6.0,
        logger=logging.getLogger("visual_unit_merge_test"),
    )
    assert len(visual_units) == 1
    assert visual_units[0]["unit_transform"] == "merged"
    assert visual_units[0]["start_time"] == 10.0
    assert visual_units[0]["end_time"] == 12.0
    assert "第一句" in visual_units[0]["text"] and "第二句" in visual_units[0]["text"]


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


def test_build_segments_with_lyric_priority_should_not_emit_inst_for_long_gap_inside_vocal_mid() -> None:
    """
    功能说明：验证 vocal mid 内即使歌词间隔较长，也不会再额外产出 inst 片段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该规则仅作用于已判定为 vocal 的中间段，不影响真实 inst mid。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        onset_candidates=[0.8, 1.6, 3.4, 4.8],
        lyric_units=[
            {"start_time": 1.0, "end_time": 1.4, "text": "第一句", "confidence": 0.9},
            {"start_time": 3.2, "end_time": 3.8, "text": "第二句", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )
    assert segments
    assert all(str(item.get("label", "")).lower() == "verse" for item in segments)


def test_build_segments_with_lyric_priority_should_absorb_head_and_tail_gap_as_vocal() -> None:
    """
    功能说明：验证句首/句尾无歌词空档会被并入 vocal，不再形成独立 inst 段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：单句歌词场景下应保持整段连续 vocal。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=4.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "chorus"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0],
        onset_candidates=[1.5, 2.2],
        lyric_units=[
            {"start_time": 1.2, "end_time": 2.1, "text": "一句", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
    )
    assert len(segments) == 1
    assert str(segments[0].get("label", "")).lower() == "chorus"
    assert abs(float(segments[0]["start_time"]) - 0.0) < 1e-6
    assert abs(float(segments[0]["end_time"]) - 4.0) < 1e-6


def test_merge_short_vocal_non_lyric_ranges_should_keep_continuous_cluster_when_over_1_2s() -> None:
    """
    功能说明：验证连续短歌词簇先合并为一段，且簇时长 >=1.2 秒时保持独立段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该场景用于避免“连续短词串”被误并到长歌词段。
    """
    merged = _merge_short_vocal_non_lyric_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 2.0, "big_segment_id": "big_001", "label": "chorus", "lyric_anchor": True, "lyric_text": "爸爸别争议啊，"},
            {"start_time": 2.0, "end_time": 2.7, "big_segment_id": "big_001", "label": "chorus", "lyric_anchor": True, "lyric_text": "啊，"},
            {"start_time": 2.7, "end_time": 3.3, "big_segment_id": "big_001", "label": "chorus", "lyric_anchor": True, "lyric_text": "爸爸，"},
            {"start_time": 3.3, "end_time": 6.0, "big_segment_id": "big_001", "label": "chorus", "lyric_anchor": True, "lyric_text": "我在这里，"},
        ],
        duration_seconds=6.0,
        instrumental_set={"intro", "inst", "outro"},
        min_duration_seconds=1.2,
    )
    assert len(merged) == 3
    assert abs(float(merged[1]["start_time"]) - 2.0) < 1e-6
    assert abs(float(merged[1]["end_time"]) - 3.3) < 1e-6


def test_merge_short_vocal_non_lyric_ranges_should_merge_continuous_cluster_under_1_2s_to_previous() -> None:
    """
    功能说明：验证连续短歌词簇时长 <1.2 秒时会并入前一个合格长歌词段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认并前规则在左右均可并时优先生效。
    """
    merged = _merge_short_vocal_non_lyric_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 2.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "看上去不费力气，"},
            {"start_time": 2.0, "end_time": 2.4, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "啊，"},
            {"start_time": 2.4, "end_time": 2.9, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "耶，"},
            {"start_time": 2.9, "end_time": 5.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "来个拥抱，"},
        ],
        duration_seconds=5.0,
        instrumental_set={"intro", "inst", "outro"},
        min_duration_seconds=1.2,
    )
    assert len(merged) == 2
    assert abs(float(merged[0]["end_time"]) - 2.9) < 1e-6


def test_merge_short_vocal_non_lyric_ranges_should_merge_single_short_target_to_previous() -> None:
    """
    功能说明：验证非连续短目标段会并入前一个合格段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前后都可并时，默认优先并入前段。
    """
    merged = _merge_short_vocal_non_lyric_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 2.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "看上去不费力气，"},
            {"start_time": 2.0, "end_time": 2.8, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "啊，"},
            {"start_time": 2.8, "end_time": 5.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "来个拥抱，"},
        ],
        duration_seconds=5.0,
        instrumental_set={"intro", "inst", "outro"},
        min_duration_seconds=1.2,
    )
    assert len(merged) == 2
    assert abs(float(merged[0]["end_time"]) - 2.8) < 1e-6


def test_merge_short_vocal_non_lyric_ranges_should_keep_short_vocal_with_long_lyric_text() -> None:
    """
    功能说明：验证歌词长度 >=3 的短 vocal 不属于短目标段，不应被该规则吞并。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：用于保护有效歌词可读性，避免过度收敛。
    """
    merged = _merge_short_vocal_non_lyric_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 1.5, "big_segment_id": "big_001", "label": "bridge", "lyric_anchor": True, "lyric_text": "我在这里。"},
            {"start_time": 1.5, "end_time": 2.2, "big_segment_id": "big_001", "label": "bridge", "lyric_anchor": True, "lyric_text": "YEAH，"},
            {"start_time": 2.2, "end_time": 4.0, "big_segment_id": "big_001", "label": "bridge", "lyric_anchor": True, "lyric_text": "秘密的。"},
        ],
        duration_seconds=4.0,
        instrumental_set={"intro", "inst", "outro"},
        min_duration_seconds=1.2,
    )
    assert len(merged) == 3
    assert abs(float(merged[1]["start_time"]) - 1.5) < 1e-6
    assert abs(float(merged[1]["end_time"]) - 2.2) < 1e-6


def test_merge_short_vocal_non_lyric_ranges_should_not_merge_into_inst_when_previous_is_inst() -> None:
    """
    功能说明：验证短目标段前一段为 inst 时，不会并入 inst，而是回退并入右侧合格 vocal。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：遵循“仅在非 inst 段之间收敛”的约束。
    """
    merged = _merge_short_vocal_non_lyric_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 1.8, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "看上去不费力气，"},
            {"start_time": 1.8, "end_time": 2.5, "big_segment_id": "big_001", "label": "inst", "lyric_anchor": False, "lyric_text": ""},
            {"start_time": 2.5, "end_time": 3.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "啊，"},
            {"start_time": 3.0, "end_time": 5.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True, "lyric_text": "来个拥抱，"},
        ],
        duration_seconds=5.0,
        instrumental_set={"intro", "inst", "outro"},
        min_duration_seconds=1.2,
    )
    assert len(merged) == 3
    assert str(merged[1].get("label", "")).lower() == "inst"
    assert abs(float(merged[2]["start_time"]) - 2.5) < 1e-6


def test_merge_short_inst_gaps_between_vocal_ranges_should_merge_cross_group_when_within_1_2s() -> None:
    """
    功能说明：验证跨标签/跨大段的短 inst 空挡（<=1.2s）会并入左侧人声段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该规则为去碎片优先策略，允许跨边界合并短空挡。
    """
    merged = _merge_short_inst_gaps_between_vocal_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 2.0, "big_segment_id": "big_001", "label": "bridge", "lyric_anchor": True},
            {"start_time": 2.0, "end_time": 2.7, "big_segment_id": "big_002", "label": "inst", "lyric_anchor": False},
            {"start_time": 2.7, "end_time": 5.0, "big_segment_id": "big_002", "label": "verse", "lyric_anchor": True},
        ],
        duration_seconds=5.0,
        instrumental_set={"intro", "inst", "outro"},
    )
    assert len(merged) == 2
    assert [str(item.get("label", "")).lower() for item in merged] == ["bridge", "verse"]
    assert abs(float(merged[0]["end_time"]) - 2.7) < 1e-6


def test_merge_short_inst_gaps_between_vocal_ranges_should_merge_0_917s_cross_group_gap() -> None:
    """
    功能说明：验证 0.917 秒跨标签短 inst 空挡会被并入左侧人声，覆盖 `_awuli_1` 典型场景。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：右侧为器乐时也应按“优先并左”收敛短 inst。
    """
    merged = _merge_short_inst_gaps_between_vocal_ranges(
        range_items=[
            {"start_time": 86.45, "end_time": 87.261, "big_segment_id": "big_007", "label": "bridge", "lyric_anchor": False},
            {"start_time": 87.261, "end_time": 88.178, "big_segment_id": "big_007", "label": "inst", "lyric_anchor": False},
            {"start_time": 88.178, "end_time": 94.563, "big_segment_id": "big_007", "label": "inst", "lyric_anchor": False},
        ],
        duration_seconds=94.563,
        instrumental_set={"intro", "inst", "outro"},
    )
    assert len(merged) == 2
    assert [str(item.get("label", "")).lower() for item in merged] == ["bridge", "inst"]
    assert abs(float(merged[0]["start_time"]) - 0.0) < 1e-6
    assert abs(float(merged[0]["end_time"]) - 88.178) < 1e-6


def test_merge_short_inst_gaps_between_vocal_ranges_should_merge_same_group_when_within_1_4s() -> None:
    """
    功能说明：验证同标签同大段的人声间 inst 空挡（<=1.4s）会并入左侧人声段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：用于覆盖 1.323s/1.370s 级别停顿并合场景。
    """
    merged = _merge_short_inst_gaps_between_vocal_ranges(
        range_items=[
            {"start_time": 10.0, "end_time": 12.3, "big_segment_id": "big_003", "label": "verse", "lyric_anchor": True},
            {"start_time": 12.3, "end_time": 13.623, "big_segment_id": "big_003", "label": "inst", "lyric_anchor": False},
            {"start_time": 13.623, "end_time": 16.0, "big_segment_id": "big_003", "label": "verse", "lyric_anchor": True},
        ],
        duration_seconds=16.0,
        instrumental_set={"intro", "inst", "outro"},
    )
    assert len(merged) == 2
    assert [str(item.get("label", "")).lower() for item in merged] == ["verse", "verse"]
    assert abs(float(merged[0]["end_time"]) - 13.623) < 1e-6


def test_merge_short_inst_gaps_between_vocal_ranges_should_keep_when_over_threshold() -> None:
    """
    功能说明：验证超过阈值的 inst 空挡不会被并合，应保留独立段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同标签同大段时使用 1.4s 阈值，超过后不得吞并。
    """
    merged = _merge_short_inst_gaps_between_vocal_ranges(
        range_items=[
            {"start_time": 20.0, "end_time": 22.0, "big_segment_id": "big_004", "label": "verse", "lyric_anchor": True},
            {"start_time": 22.0, "end_time": 23.45, "big_segment_id": "big_004", "label": "inst", "lyric_anchor": False},
            {"start_time": 23.45, "end_time": 26.0, "big_segment_id": "big_004", "label": "verse", "lyric_anchor": True},
        ],
        duration_seconds=26.0,
        instrumental_set={"intro", "inst", "outro"},
    )
    assert any(str(item.get("label", "")).lower() == "inst" for item in merged)


def test_merge_short_inst_gaps_between_vocal_ranges_should_merge_head_inst_to_right() -> None:
    """
    功能说明：验证开头短 inst 且无左侧人声时，会并入右侧人声段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅在右侧存在人声段时触发该并合兜底。
    """
    merged = _merge_short_inst_gaps_between_vocal_ranges(
        range_items=[
            {"start_time": 0.0, "end_time": 0.5, "big_segment_id": "big_001", "label": "inst", "lyric_anchor": False},
            {"start_time": 0.5, "end_time": 3.0, "big_segment_id": "big_001", "label": "verse", "lyric_anchor": True},
        ],
        duration_seconds=3.0,
        instrumental_set={"intro", "inst", "outro"},
    )
    assert len(merged) == 1
    assert str(merged[0].get("label", "")).lower() == "verse"
    assert abs(float(merged[0]["start_time"]) - 0.0) < 1e-6
    assert abs(float(merged[0]["end_time"]) - 3.0) < 1e-6


def test_build_segments_with_lyric_priority_should_merge_short_inst_gap_between_vocal_segments() -> None:
    """
    功能说明：验证全流程分段下，vocal-vocal 之间的短 inst 空挡会在后置阶段被并合。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该用例依赖能量阶段先产生短 inst 岛，再由第三阶段消除。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        onset_candidates=[0.5, 1.2, 2.4, 3.4, 4.4],
        lyric_units=[
            {"start_time": 0.6, "end_time": 1.2, "text": "第一句", "confidence": 0.9},
            {"start_time": 2.4, "end_time": 3.1, "text": "第二句", "confidence": 0.9},
        ],
        instrumental_labels=["intro", "inst", "outro"],
        vocal_rms_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.4, 3.0, 4.0, 5.0, 6.0],
        vocal_rms_values=[0.09, 0.10, 0.09, 0.001, 0.001, 0.10, 0.09, 0.08, 0.08, 0.08],
    )
    assert segments
    assert all(str(item.get("label", "")).lower() != "inst" for item in segments)


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


def test_build_segments_with_lyric_priority_should_prefer_vocal_under_high_recall_policy() -> None:
    """
    功能说明：验证高召回策略下，存在明显人声音量时优先保留人声标签。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即便中间存在短暂能量下探，也不应轻易切为 inst。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=10.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
        onset_candidates=[1.0, 2.0, 6.0, 8.0],
        lyric_units=[],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0],
        rms_values=[0.3, 0.4, 0.45, 0.2, 0.1, 0.3, 0.4, 0.35],
        vocal_onset_candidates=[1.0, 2.0, 6.0, 8.0],
        vocal_rms_times=[0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0],
        vocal_rms_values=[0.85, 0.92, 0.88, 0.10, 0.12, 0.86, 0.91, 0.87],
    )
    assert segments
    assert all(str(item.get("label", "")).lower() == "verse" for item in segments)


def test_build_mid_segments_by_vocal_energy_should_keep_continuous_vocal_when_short_silence_hole() -> None:
    """
    功能说明：验证连续人声中出现短时低能量空洞时，不会被切成“人声-器乐-人声”碎片。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：短空洞时长小于回填阈值，预期应被闭合处理。
    """
    mid_segments = _build_mid_segments_by_vocal_energy(
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "verse"}],
        instrumental_set={"intro", "inst", "outro"},
        vocal_rms_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.2, 3.8, 4.5, 5.2, 6.0, 7.0, 8.0],
        vocal_rms_values=[0.16, 0.22, 0.27, 0.31, 0.29, 0.03, 0.30, 0.33, 0.31, 0.28, 0.24, 0.21, 0.18, 0.16],
        min_mid_duration_seconds=0.8,
        enter_quantile=0.70,
        exit_quantile=0.45,
    )
    assert mid_segments
    vocal_segments = [item for item in mid_segments if bool(item.get("is_vocal"))]
    assert len(vocal_segments) == 1
    assert str(vocal_segments[0].get("label", "")).lower() == "verse"


def test_build_mid_segments_by_vocal_energy_should_mark_clear_silence_as_instrumental() -> None:
    """
    功能说明：验证全段仅有噪声底能量时，会稳定判定为器乐段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：big 标签为人声类型时，也不能被噪声底误判为 vocal。
    """
    mid_segments = _build_mid_segments_by_vocal_energy(
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "verse"}],
        instrumental_set={"intro", "inst", "outro"},
        vocal_rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vocal_rms_values=[0.0012, 0.0010, 0.0014, 0.0011, 0.0013, 0.0010, 0.0012],
        min_mid_duration_seconds=0.8,
        enter_quantile=0.70,
        exit_quantile=0.45,
    )
    assert mid_segments
    assert all(bool(item.get("is_vocal")) is False for item in mid_segments)
    assert all(str(item.get("label", "")).lower() == "inst" for item in mid_segments)


def test_build_mid_segments_by_vocal_energy_should_keep_weak_head_and_tail_as_vocal() -> None:
    """
    功能说明：验证字头/尾音处的弱能量在高召回策略下仍尽量保留为人声。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅绝对静音应判器乐，弱能量不应被提前切断。
    """
    mid_segments = _build_mid_segments_by_vocal_energy(
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 5.0, "label": "chorus"}],
        instrumental_set={"intro", "inst", "outro"},
        vocal_rms_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        vocal_rms_values=[0.0012, 0.0015, 0.0042, 0.0060, 0.0500, 0.0800, 0.0700, 0.0500, 0.0062, 0.0045, 0.0018],
        min_mid_duration_seconds=0.8,
        enter_quantile=0.70,
        exit_quantile=0.45,
    )
    vocal_segments = [item for item in mid_segments if bool(item.get("is_vocal"))]
    assert vocal_segments
    # 字头附近 1.0s 与尾音附近 4.5s 均应位于人声区间覆盖内。
    assert any(float(item["start_time"]) <= 1.0 <= float(item["end_time"]) for item in vocal_segments)
    assert any(float(item["start_time"]) <= 4.5 <= float(item["end_time"]) for item in vocal_segments)


def test_build_segments_with_lyric_priority_should_keep_no_lyric_vocal_mid_as_vocal() -> None:
    """
    功能说明：验证人声能量存在但无歌词时，区间仍保留人声标签。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅绝对静音区应输出 inst。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=4.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0],
        onset_candidates=[1.2, 2.0, 3.0],
        lyric_units=[],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 1.2, 2.0, 3.0, 4.0],
        rms_values=[0.3, 0.8, 0.7, 0.6, 0.55],
        vocal_onset_candidates=[1.2, 2.0, 3.0],
        vocal_rms_times=[0.0, 0.8, 1.2, 2.0, 3.0, 4.0],
        vocal_rms_values=[0.78, 0.80, 0.95, 0.86, 0.82, 0.80],
    )
    assert segments
    assert any(str(item.get("label", "")).lower() == "verse" for item in segments)


def test_build_segments_with_lyric_priority_should_keep_lyric_segment_as_vocal_and_no_cross_segment() -> None:
    """
    功能说明：验证歌词锚点分段下，一条歌词仅对应一个人声段且不跨多个 segment。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：相邻器乐段仍可存在，但歌词段必须唯一命中。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=6.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 6.0, "label": "chorus"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        onset_candidates=[1.0, 2.0, 4.0],
        lyric_units=[
            {
                "start_time": 1.2,
                "end_time": 2.4,
                "text": "第一句",
                "confidence": 0.9,
                "token_units": [
                    {"text": "第", "start_time": 1.2, "end_time": 1.5, "granularity": "char"},
                    {"text": "一", "start_time": 1.5, "end_time": 1.8, "granularity": "char"},
                    {"text": "句", "start_time": 1.8, "end_time": 2.2, "granularity": "char"},
                ],
            }
        ],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rms_values=[0.2, 0.4, 0.5, 0.45, 0.55, 0.5, 0.45],
        vocal_onset_candidates=[1.0, 2.0, 4.0],
        vocal_rms_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vocal_rms_values=[0.82, 0.88, 0.92, 0.90, 0.87, 0.89, 0.85],
    )
    lyric_overlap_segments = [
        item
        for item in segments
        if str(item.get("label", "")).lower() == "chorus"
        and max(0.0, min(float(item["end_time"]), 2.4) - max(float(item["start_time"]), 1.2)) > 1e-6
    ]
    assert len(lyric_overlap_segments) == 1


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
        {
            "start_time": 22.000,
            "end_time": 26.000,
            "text": "跨边界歌词",
            "confidence": 0.9,
            "token_units": [
                {"text": "跨", "start_time": 22.000, "end_time": 22.100, "granularity": "char"},
                {"text": "边", "start_time": 22.100, "end_time": 22.200, "granularity": "char"},
            ],
        },
    ]
    attached = _attach_lyrics_to_segments(lyric_units_raw=lyric_units, segments=segments)
    assert attached[0]["segment_id"] == "seg_0002"
    assert attached[0]["token_units"][0]["text"] == "跨"


def test_build_segmentation_anchor_lyric_units_should_split_by_comma_punctuation() -> None:
    """
    功能说明：验证分段锚点歌词单元会按逗号/句号拆分，避免长句跨多个镜头。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：拆分结果文本应去除句首标点并过滤纯标点，句尾标点可保留。
    """
    sentence_units = [
        {
            "start_time": 1.0,
            "end_time": 3.4,
            "text": "甲，乙。丙",
            "confidence": 0.9,
            "token_units": [
                {"text": "甲", "start_time": 1.0, "end_time": 1.2, "granularity": "char"},
                {"text": "，", "start_time": 1.2, "end_time": 1.3, "granularity": "char"},
                {"text": "乙", "start_time": 1.4, "end_time": 1.7, "granularity": "char"},
                {"text": "。", "start_time": 1.7, "end_time": 1.8, "granularity": "char"},
                {"text": "丙", "start_time": 2.0, "end_time": 2.4, "granularity": "char"},
            ],
        }
    ]
    anchors = _build_segmentation_anchor_lyric_units(
        sentence_units=sentence_units,
        logger=logging.getLogger("segmentation_anchor_split_test"),
    )
    assert len(anchors) == 3
    assert [item["text"] for item in anchors] == ["甲，", "乙。", "丙"]
    assert all(item["start_time"] < item["end_time"] for item in anchors)


def test_build_segmentation_anchor_lyric_units_should_split_by_long_gap_without_punctuation() -> None:
    """
    功能说明：验证分段锚点歌词在无标点场景下，仍会按超长停顿（gap>0.8s）强制断句。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅依赖 token 时间戳间隔，不依赖标点文本。
    """
    sentence_units = [
        {
            "start_time": 10.0,
            "end_time": 13.6,
            "text": "看这里一望无际彩色的世界",
            "confidence": 0.9,
            "token_units": [
                {"text": "看", "start_time": 10.00, "end_time": 10.10, "granularity": "char"},
                {"text": "这", "start_time": 10.12, "end_time": 10.22, "granularity": "char"},
                {"text": "里", "start_time": 10.24, "end_time": 10.34, "granularity": "char"},
                {"text": "一", "start_time": 10.36, "end_time": 10.46, "granularity": "char"},
                {"text": "望", "start_time": 10.48, "end_time": 10.58, "granularity": "char"},
                {"text": "无", "start_time": 10.60, "end_time": 10.70, "granularity": "char"},
                {"text": "际", "start_time": 10.72, "end_time": 10.82, "granularity": "char"},
                {"text": "彩", "start_time": 12.02, "end_time": 12.12, "granularity": "char"},
                {"text": "色", "start_time": 12.14, "end_time": 12.24, "granularity": "char"},
                {"text": "的", "start_time": 12.26, "end_time": 12.36, "granularity": "char"},
                {"text": "世", "start_time": 12.38, "end_time": 12.48, "granularity": "char"},
                {"text": "界", "start_time": 12.50, "end_time": 12.60, "granularity": "char"},
            ],
        }
    ]
    anchors = _build_segmentation_anchor_lyric_units(
        sentence_units=sentence_units,
        logger=logging.getLogger("segmentation_anchor_long_gap_split_test"),
    )
    assert len(anchors) == 2
    assert [item["text"] for item in anchors] == ["看这里一望无际", "彩色的世界"]
    assert all(float(item["start_time"]) < float(item["end_time"]) for item in anchors)


def test_build_segmentation_anchor_lyric_units_should_left_attach_leading_punctuation_after_long_gap() -> None:
    """
    功能说明：验证分段锚点拆分在长停顿切句后，会将后句开头标点左归属到前句。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：切分后若出现纯标点子句，应通过左归属消除该子句。
    """
    sentence_units = [
        {
            "start_time": 10.0,
            "end_time": 12.0,
            "text": "看这里，彩色",
            "confidence": 0.9,
            "token_units": [
                {"text": "看", "start_time": 10.00, "end_time": 10.10, "granularity": "char"},
                {"text": "这", "start_time": 10.12, "end_time": 10.22, "granularity": "char"},
                {"text": "里", "start_time": 10.24, "end_time": 10.34, "granularity": "char"},
                {"text": "，", "start_time": 11.30, "end_time": 11.35, "granularity": "char"},
                {"text": "彩", "start_time": 11.36, "end_time": 11.46, "granularity": "char"},
                {"text": "色", "start_time": 11.48, "end_time": 11.58, "granularity": "char"},
            ],
        }
    ]
    anchors = _build_segmentation_anchor_lyric_units(
        sentence_units=sentence_units,
        logger=logging.getLogger("segmentation_anchor_punctuation_left_attach_test"),
    )
    assert len(anchors) == 2
    assert [item["text"] for item in anchors] == ["看这里，", "彩色"]
    assert all(float(item["start_time"]) < float(item["end_time"]) for item in anchors)


def test_build_segments_with_lyric_priority_should_protect_boundary_short_word_in_inst() -> None:
    """
    功能说明：验证 inst 边界窗口命中的短词会被强制切出 vocal 微段，不再整段留在器乐。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：保护仅发生在边界窗口，区间内部歌词不由该规则处理。
    """
    segments = _build_segments_with_lyric_priority(
        duration_seconds=4.0,
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"}],
        beat_candidates=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
        onset_candidates=[0.6, 1.0, 2.0, 3.0],
        lyric_units=[
            {
                "start_time": 0.12,
                "end_time": 0.20,
                "text": "我",
                "confidence": 0.9,
                "token_units": [
                    {"text": "我", "start_time": 0.12, "end_time": 0.20, "granularity": "char"},
                ],
            }
        ],
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 0.6, 1.0, 2.0, 3.0, 4.0],
        rms_values=[0.20, 0.42, 0.45, 0.40, 0.38, 0.30],
        vocal_onset_candidates=[0.6, 1.0, 2.0, 3.0],
        vocal_rms_times=[0.0, 0.3, 0.6, 1.0, 2.0, 3.0, 3.5, 3.8, 4.0],
        vocal_rms_values=[0.0010, 0.0012, 0.0800, 0.1000, 0.0900, 0.0800, 0.0700, 0.0012, 0.0010],
    )
    assert segments
    assert any(
        str(item.get("label", "")).lower() == "verse"
        and float(item["start_time"]) <= 0.12 <= float(item["end_time"])
        and float(item["start_time"]) <= 0.20 <= float(item["end_time"])
        for item in segments
    )


def test_attach_lyrics_to_segments_should_drop_punctuation_only_clipped_tokens() -> None:
    """
    功能说明：验证歌词挂载切片时会过滤纯标点片段，避免逗号句号单独上屏。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当区间裁剪后仅剩标点 token，应直接丢弃该挂载单元。
    """
    segments = [
        {"segment_id": "seg_0001", "start_time": 0.0, "end_time": 0.4, "label": "inst"},
        {"segment_id": "seg_0002", "start_time": 0.4, "end_time": 1.2, "label": "verse"},
    ]
    lyric_units = [
        {
            "start_time": 0.3,
            "end_time": 1.0,
            "text": "，你好",
            "confidence": 0.9,
            "token_units": [
                {"text": "，", "start_time": 0.3, "end_time": 0.4, "granularity": "char"},
                {"text": "你", "start_time": 0.6, "end_time": 0.8, "granularity": "char"},
                {"text": "好", "start_time": 0.8, "end_time": 1.0, "granularity": "char"},
            ],
        }
    ]
    attached = _attach_lyrics_to_segments(lyric_units_raw=lyric_units, segments=segments)
    assert len(attached) == 1
    assert attached[0]["segment_id"] == "seg_0002"
    assert attached[0]["text"] == "你好"


def test_build_big_segments_v2_should_shift_boundary_to_right_when_right_overlap_is_larger() -> None:
    """
    功能说明：验证跨界歌词后半占比更高时，big_v2 边界向左移动并归后 big。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：big_v2 仅基于歌词跨界占比，不依赖 segment 边界吸附。
    """
    big_segments_stage1 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
        {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "verse"},
    ]
    lyric_units = [{"start_time": 9.0, "end_time": 13.0, "text": "跨界歌词", "confidence": 0.9}]
    big_v2 = _build_big_segments_v2_by_lyric_overlap(
        big_segments_stage1=big_segments_stage1,
        lyric_units=lyric_units,
        duration_seconds=20.0,
    )
    assert big_v2[0]["end_time"] == 9.0
    assert big_v2[1]["start_time"] == 9.0


def test_build_big_segments_v2_should_keep_left_when_left_overlap_is_larger() -> None:
    """
    功能说明：验证跨界歌词前半占比更高时，big_v2 边界向右移动并归前 big。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：边界重算后仍保持时间连续。
    """
    big_segments_stage1 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
        {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "verse"},
    ]
    lyric_units = [{"start_time": 7.0, "end_time": 11.0, "text": "跨界歌词", "confidence": 0.9}]
    big_v2 = _build_big_segments_v2_by_lyric_overlap(
        big_segments_stage1=big_segments_stage1,
        lyric_units=lyric_units,
        duration_seconds=20.0,
    )
    assert big_v2[0]["end_time"] == 11.0
    assert big_v2[1]["start_time"] == 11.0


def test_build_big_segments_v2_should_assign_tie_to_right_big() -> None:
    """
    功能说明：验证跨界歌词前后占比相等时，按规则归后 big（平局归后）。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：平局判定依赖边界两侧重叠时长相等。
    """
    big_segments_stage1 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
        {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "verse"},
    ]
    lyric_units = [{"start_time": 9.0, "end_time": 11.0, "text": "平局歌词", "confidence": 0.9}]
    big_v2 = _build_big_segments_v2_by_lyric_overlap(
        big_segments_stage1=big_segments_stage1,
        lyric_units=lyric_units,
        duration_seconds=20.0,
    )
    assert big_v2[0]["end_time"] == 9.0
    assert big_v2[1]["start_time"] == 9.0


def test_segments_built_by_big_v2_should_rebind_lyrics_to_valid_segment_ids() -> None:
    """
    功能说明：验证以 big_v2 重新分段后，歌词重新挂载得到的 segment_id 全部有效。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同一句跨界歌词允许拆分为多条挂载记录。
    """
    big_segments_stage1 = [
        {"segment_id": "big_001", "start_time": 0.0, "end_time": 10.0, "label": "verse"},
        {"segment_id": "big_002", "start_time": 10.0, "end_time": 20.0, "label": "verse"},
    ]
    lyric_units = [
        {
            "start_time": 9.0,
            "end_time": 13.0,
            "text": "跨界歌词",
            "confidence": 0.9,
            "token_units": [
                {"text": "跨", "start_time": 9.0, "end_time": 10.0, "granularity": "char"},
                {"text": "界", "start_time": 10.0, "end_time": 11.0, "granularity": "char"},
                {"text": "歌", "start_time": 11.0, "end_time": 12.0, "granularity": "char"},
                {"text": "词", "start_time": 12.0, "end_time": 13.0, "granularity": "char"},
            ],
        }
    ]
    big_v2 = _build_big_segments_v2_by_lyric_overlap(
        big_segments_stage1=big_segments_stage1,
        lyric_units=lyric_units,
        duration_seconds=20.0,
    )
    segments_v2 = _build_segments_with_lyric_priority(
        duration_seconds=20.0,
        big_segments=big_v2,
        beat_candidates=[0.0, 5.0, 10.0, 15.0, 20.0],
        onset_candidates=[4.5, 9.5, 12.5, 16.0],
        lyric_units=lyric_units,
        instrumental_labels=["intro", "inst", "outro"],
        rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        rms_values=[0.6, 0.7, 0.8, 0.7, 0.6],
        vocal_onset_candidates=[4.5, 9.5, 12.5, 16.0],
        vocal_rms_times=[0.0, 5.0, 10.0, 15.0, 20.0],
        vocal_rms_values=[0.5, 0.6, 0.75, 0.7, 0.5],
    )
    attached = _attach_lyrics_to_segments(lyric_units_raw=lyric_units, segments=segments_v2)
    segment_id_set = {str(item["segment_id"]) for item in segments_v2}
    assert attached
    assert all(str(item["segment_id"]) in segment_id_set for item in attached)
