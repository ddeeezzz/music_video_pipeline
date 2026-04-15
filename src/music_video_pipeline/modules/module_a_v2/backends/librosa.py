"""
文件用途：提供模块A V2的 Librosa 声学候选提取能力。
核心流程：提取 beat/onset/RMS，并归一化候选时间轴。
输入输出：输入音频路径与参数，输出候选时间戳与能量序列。
依赖说明：依赖 librosa、numpy 与 v2 时间工具。
维护说明：本文件仅负责信号提取，不承担后续分段决策。
"""

# 标准库：用于类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time


# 常量：基础特征提取 hop（beat/onset/rms）
BASE_HOP_LENGTH = 512
# 常量：chroma/pyin 专用采样率
FEATURE_SAMPLE_RATE = 22050
# 常量：pyin 专用 hop
PYIN_HOP_LENGTH = 1024


def extract_acoustic_candidates_with_librosa(
    audio_path,
    duration_seconds: float,
    logger,
    *,
    extract_beat: bool = True,
    extract_onset: bool = True,
    with_onset_points: bool = False,
    with_chroma_points: bool = False,
    with_f0_points: bool = False,
) -> tuple[Any, ...]:
    """
    功能说明：调用 Librosa 提取 beat/onset/RMS 候选池。
    参数说明：
    - audio_path: 输入音频文件路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器。
    - extract_beat: 是否执行 beat 检测。
    - extract_onset: 是否执行 onset 检测。
    - with_onset_points: 是否追加输出 onset 强度点（time+energy_raw）。
    - with_chroma_points: 是否追加输出 chroma 点（time+12维向量）。
    - with_f0_points: 是否追加输出 f0 点（time+f0_hz+voiced+confidence）。
    返回值：
    - tuple[Any, ...]:
      前4项固定为节拍候选、起音候选、RMS时间轴、RMS值序列；
      后续按参数顺序可选追加 onset_points / chroma_points / f0_points。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：RMS 为空时回退双点兜底。
    """
    try:
        # 第三方库：音频分析与节拍、起音、能量特征提取
        import librosa  # type: ignore
        # 第三方库：向量计算与帧索引构建
        import numpy as np  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"librosa/numpy 导入失败: {error}") from error

    logger.debug(
        (
            "模块A V2-Librosa底层提取开始，输入=%s，extract_beat=%s，extract_onset=%s，"
            "with_onset_points=%s，with_chroma_points=%s，with_f0_points=%s（RMS固定提取）"
        ),
        audio_path,
        bool(extract_beat),
        bool(extract_onset),
        bool(with_onset_points),
        bool(with_chroma_points),
        bool(with_f0_points),
    )
    y_native, sample_rate_native = librosa.load(str(audio_path), sr=None, mono=True)

    beat_times: list[float] = []
    if bool(extract_beat):
        _, beat_frames = librosa.beat.beat_track(y=y_native, sr=sample_rate_native)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate_native).tolist()

    onset_times: list[float] = []
    onset_points: list[dict[str, float]] = []
    if bool(extract_onset):
        onset_envelope = librosa.onset.onset_strength(
            y=y_native,
            sr=sample_rate_native,
            hop_length=BASE_HOP_LENGTH,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate_native,
            hop_length=BASE_HOP_LENGTH,
            units="frames",
        )
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=sample_rate_native,
            hop_length=BASE_HOP_LENGTH,
        ).tolist()
        onset_energy_by_time: dict[float, float] = {}
        for onset_time, onset_frame in zip(onset_times, onset_frames, strict=False):
            time_key = round_time(max(0.0, float(onset_time)))
            frame_index = int(max(0, min(int(onset_frame), len(onset_envelope) - 1)))
            energy_raw = max(0.0, float(onset_envelope[frame_index]))
            previous_value = onset_energy_by_time.get(time_key, 0.0)
            if energy_raw > previous_value:
                onset_energy_by_time[time_key] = energy_raw
        onset_points = [
            {
                "time": round(float(time_key), 6),
                "energy_raw": round(float(energy_raw), 6),
            }
            for time_key, energy_raw in sorted(onset_energy_by_time.items(), key=lambda pair: pair[0])
        ]

    y_feature = y_native
    sample_rate_feature = int(sample_rate_native)
    if bool(with_chroma_points) or bool(with_f0_points):
        sample_rate_feature = FEATURE_SAMPLE_RATE
        if int(sample_rate_native) != FEATURE_SAMPLE_RATE:
            y_feature = librosa.resample(
                y_native,
                orig_sr=sample_rate_native,
                target_sr=FEATURE_SAMPLE_RATE,
            )

    chroma_points: list[dict[str, Any]] = []
    if bool(with_chroma_points):
        chroma_matrix = librosa.feature.chroma_stft(
            y=y_feature,
            sr=sample_rate_feature,
            hop_length=BASE_HOP_LENGTH,
        )
        frame_count = int(chroma_matrix.shape[1]) if hasattr(chroma_matrix, "shape") and len(chroma_matrix.shape) >= 2 else 0
        chroma_times = librosa.frames_to_time(
            range(frame_count),
            sr=sample_rate_feature,
            hop_length=BASE_HOP_LENGTH,
        ).tolist()
        for frame_index in range(frame_count):
            chroma_vector = [
                round(float(max(0.0, chroma_matrix[chroma_index, frame_index])), 6)
                for chroma_index in range(12)
            ]
            chroma_points.append(
                {
                    "time": round(float(_clamp_time(chroma_times[frame_index], duration_seconds)), 6),
                    "chroma": chroma_vector,
                }
            )

    f0_points: list[dict[str, Any]] = []
    if bool(with_f0_points):
        try:
            f0_hz, voiced_flag, voiced_probs = librosa.pyin(
                y=y_feature,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate_feature,
                hop_length=PYIN_HOP_LENGTH,
            )
            frame_count = len(f0_hz) if f0_hz is not None else 0
            f0_times = librosa.frames_to_time(
                range(frame_count),
                sr=sample_rate_feature,
                hop_length=PYIN_HOP_LENGTH,
            ).tolist()
            for frame_index in range(frame_count):
                raw_f0 = float(f0_hz[frame_index]) if f0_hz[frame_index] == f0_hz[frame_index] else 0.0
                safe_f0 = max(0.0, raw_f0)
                voiced = bool(voiced_flag[frame_index]) if voiced_flag is not None else safe_f0 > 0.0
                confidence = 0.0
                if voiced_probs is not None and frame_index < len(voiced_probs):
                    confidence = max(0.0, min(1.0, float(voiced_probs[frame_index])))
                f0_points.append(
                    {
                        "time": round(float(_clamp_time(f0_times[frame_index], duration_seconds)), 6),
                        "f0_hz": round(float(safe_f0), 6),
                        "voiced": bool(voiced and safe_f0 > 0.0),
                        "confidence": round(float(confidence), 6),
                    }
                )
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A V2-Librosa提取F0失败，回退空列表，输入=%s，错误=%s", audio_path, error)
            f0_points = []

    rms_values_np = librosa.feature.rms(y=y_native, hop_length=BASE_HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms_values_np)),
        sr=sample_rate_native,
        hop_length=BASE_HOP_LENGTH,
    ).tolist()
    rms_values = [float(value) for value in rms_values_np.tolist()]

    beat_times = _normalize_timestamp_list(beat_times + [0.0, duration_seconds], duration_seconds)
    onset_times = _normalize_timestamp_list(onset_times + [0.0, duration_seconds], duration_seconds)
    if bool(with_onset_points):
        onset_energy_lookup = {
            round_time(_safe_time): float(max(0.0, _safe_energy))
            for _safe_time, _safe_energy in [
                (round_time(_safe_item.get("time", 0.0)), _safe_item.get("energy_raw", 0.0))
                for _safe_item in onset_points
            ]
        }
        onset_points = [
            {
                "time": round(round_time(onset_time), 6),
                "energy_raw": round(float(onset_energy_lookup.get(round_time(onset_time), 0.0)), 6),
            }
            for onset_time in onset_times
        ]
    if not rms_times:
        rms_times = [0.0, round_time(duration_seconds)]
        rms_values = [1.0, 1.0]

    output_items: list[Any] = [beat_times, onset_times, rms_times, rms_values]
    if bool(with_onset_points):
        output_items.append(onset_points)
    if bool(with_chroma_points):
        output_items.append(chroma_points)
    if bool(with_f0_points):
        output_items.append(f0_points)
    return tuple(output_items)  # type: ignore[return-value]


def _normalize_timestamp_list(timestamps: list[float], duration_seconds: float) -> list[float]:
    """
    功能说明：归一化时间戳（裁剪、去重、升序、最小间隔）。
    参数说明：
    - timestamps: 时间戳列表（秒）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - list[float]: 归一化后的结果值。
    异常说明：无。
    边界条件：空输入回退 [0, duration]。
    """
    clipped = sorted({_clamp_time(value, duration_seconds) for value in timestamps})
    if not clipped:
        return [0.0, round_time(duration_seconds)]

    filtered: list[float] = [clipped[0]]
    for value in clipped[1:]:
        if value - filtered[-1] >= 0.1:
            filtered.append(value)

    if filtered[0] > 0.0:
        filtered.insert(0, 0.0)
    else:
        filtered[0] = 0.0

    last_time = round_time(duration_seconds)
    if filtered[-1] < last_time:
        filtered.append(last_time)
    else:
        filtered[-1] = last_time

    dedup: list[float] = [filtered[0]]
    for value in filtered[1:]:
        if value - dedup[-1] >= 0.1:
            dedup.append(value)
    if dedup[-1] < last_time:
        dedup.append(last_time)

    return [round_time(value) for value in dedup]


def _clamp_time(time_value: float, duration_seconds: float) -> float:
    """
    功能说明：将时间戳限制在 [0, duration]。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - duration_seconds: 音频总时长（秒）。
    返回值：
    - float: 裁剪后的时间戳。
    异常说明：无。
    边界条件：duration 取最小 0.1 秒。
    """
    safe_duration = max(0.1, duration_seconds)
    return max(0.0, min(safe_duration, float(time_value)))
