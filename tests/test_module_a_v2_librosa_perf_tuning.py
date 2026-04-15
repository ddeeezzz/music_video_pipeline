"""
文件用途：验证模块A V2-Librosa提速参数是否按设计生效。
核心流程：构造短音频样本并打桩 librosa 关键算子，断言采样率与 hop 参数。
输入输出：输入临时音频文件，输出断言结果。
依赖说明：依赖 pytest、librosa 与模块A V2 librosa 后端。
维护说明：当特征提取参数策略调整时需同步更新断言。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于数学函数
import math
# 标准库：用于PCM波形写入
import struct
# 标准库：用于WAV写入
import wave
from pathlib import Path

# 第三方库：用于测试框架
import pytest

# 项目内模块：被测函数
from music_video_pipeline.modules.module_a_v2.backends.librosa import extract_acoustic_candidates_with_librosa


# 常量：测试音频默认时长（秒）
TEST_AUDIO_SECONDS = 0.24
# 常量：测试音频默认频率（Hz）
TEST_AUDIO_HZ = 440.0


def _write_sine_wav(path: Path, sample_rate: int, duration_seconds: float = TEST_AUDIO_SECONDS, frequency_hz: float = TEST_AUDIO_HZ) -> None:
    """
    功能说明：写入单声道16bit正弦波测试音频。
    参数说明：
    - path: 输出 wav 文件路径。
    - sample_rate: 采样率。
    - duration_seconds: 时长（秒）。
    - frequency_hz: 正弦频率（Hz）。
    返回值：无。
    异常说明：写入失败时抛异常。
    边界条件：最少写入1帧。
    """
    frame_count = max(1, int(sample_rate * duration_seconds))
    amplitude = 0.35
    pcm_frames = bytearray()
    for index in range(frame_count):
        phase = 2.0 * math.pi * frequency_hz * (index / float(sample_rate))
        value = int(max(-32767, min(32767, round(math.sin(phase) * 32767 * amplitude))))
        pcm_frames.extend(struct.pack("<h", value))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm_frames))


def test_librosa_feature_path_should_use_22050_and_pyin_1024_when_resample_needed(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证输入非22050采样率时，chroma/pyin路径会走22050并使用pyin hop=1024。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：beat/onset/rms 仍应保持基础 hop=512。
    """
    librosa = pytest.importorskip("librosa")
    np = pytest.importorskip("numpy")

    audio_path = tmp_path / "sr48000.wav"
    _write_sine_wav(path=audio_path, sample_rate=48000)

    calls: dict[str, float | int | bool] = {
        "resample_called": False,
        "pyin_sr": 0,
        "pyin_hop": 0,
        "chroma_sr": 0,
        "chroma_hop": 0,
        "onset_hop": 0,
        "rms_hop": 0,
    }

    original_resample = librosa.resample
    original_onset_strength = librosa.onset.onset_strength
    original_rms = librosa.feature.rms

    def _wrapped_resample(y, *, orig_sr, target_sr, **kwargs):
        calls["resample_called"] = True
        return original_resample(y, orig_sr=orig_sr, target_sr=target_sr, **kwargs)

    def _wrapped_onset_strength(*args, **kwargs):
        calls["onset_hop"] = int(kwargs.get("hop_length", 0))
        return original_onset_strength(*args, **kwargs)

    def _wrapped_rms(*args, **kwargs):
        calls["rms_hop"] = int(kwargs.get("hop_length", 0))
        return original_rms(*args, **kwargs)

    def _fake_chroma_stft(*, y, sr, hop_length, **_kwargs):
        calls["chroma_sr"] = int(sr)
        calls["chroma_hop"] = int(hop_length)
        frame_count = max(1, int(len(y) // max(1, int(hop_length))))
        return np.zeros((12, frame_count), dtype=float)

    def _fake_pyin(*, y, fmin, fmax, sr, hop_length, **_kwargs):
        del fmin, fmax
        calls["pyin_sr"] = int(sr)
        calls["pyin_hop"] = int(hop_length)
        frame_count = max(1, int(len(y) // max(1, int(hop_length))))
        return np.zeros(frame_count, dtype=float), np.ones(frame_count, dtype=bool), np.full(frame_count, 0.8, dtype=float)

    monkeypatch.setattr(librosa, "resample", _wrapped_resample)
    monkeypatch.setattr(librosa.onset, "onset_strength", _wrapped_onset_strength)
    monkeypatch.setattr(librosa.feature, "rms", _wrapped_rms)
    monkeypatch.setattr(librosa.feature, "chroma_stft", _fake_chroma_stft)
    monkeypatch.setattr(librosa, "pyin", _fake_pyin)

    result = extract_acoustic_candidates_with_librosa(
        audio_path=audio_path,
        duration_seconds=TEST_AUDIO_SECONDS,
        logger=logging.getLogger("test_module_a_v2_librosa_perf_tuning"),
        extract_beat=False,
        extract_onset=True,
        with_onset_points=True,
        with_chroma_points=True,
        with_f0_points=True,
    )

    assert len(result) == 7
    assert bool(calls["resample_called"])
    assert int(calls["pyin_sr"]) == 22050
    assert int(calls["pyin_hop"]) == 1024
    assert int(calls["chroma_sr"]) == 22050
    assert int(calls["chroma_hop"]) == 512
    assert int(calls["onset_hop"]) == 512
    assert int(calls["rms_hop"]) == 512


def test_librosa_feature_path_should_skip_resample_when_input_is_22050(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证输入已是22050采样率时，不会触发 feature 重采样。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅启用 chroma 特征也应走 feature 采样率分支但不重采样。
    """
    librosa = pytest.importorskip("librosa")
    np = pytest.importorskip("numpy")

    audio_path = tmp_path / "sr22050.wav"
    _write_sine_wav(path=audio_path, sample_rate=22050)

    calls = {"resample_count": 0, "chroma_sr": 0}

    def _wrapped_resample(*args, **kwargs):
        calls["resample_count"] += 1
        return np.zeros(10, dtype=float)

    def _fake_chroma_stft(*, y, sr, hop_length, **_kwargs):
        del hop_length
        calls["chroma_sr"] = int(sr)
        frame_count = max(1, int(len(y) // 512))
        return np.zeros((12, frame_count), dtype=float)

    monkeypatch.setattr(librosa, "resample", _wrapped_resample)
    monkeypatch.setattr(librosa.feature, "chroma_stft", _fake_chroma_stft)

    result = extract_acoustic_candidates_with_librosa(
        audio_path=audio_path,
        duration_seconds=TEST_AUDIO_SECONDS,
        logger=logging.getLogger("test_module_a_v2_librosa_perf_tuning"),
        extract_beat=False,
        extract_onset=False,
        with_chroma_points=True,
        with_f0_points=False,
    )

    assert len(result) == 5
    assert int(calls["resample_count"]) == 0
    assert int(calls["chroma_sr"]) == 22050
