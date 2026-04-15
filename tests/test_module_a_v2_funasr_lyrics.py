"""
文件用途：验证模块A V2 的 FunASR gap分句与离群剔除规则。
核心流程：构造可控 token/gap 样本，断言离群识别与分句阈值行为。
输入输出：输入伪造 FunASR 结果，输出断言结果。
依赖说明：依赖 module_a_v2.funasr_lyrics 纯函数。
维护说明：本测试不依赖真实模型推理，专注统计规则与切句行为。
"""

# 标准库：日志构造
import logging
# 标准库：模块注入
import sys
# 标准库：临时模块类型
import types

# 项目内模块：V2 FunASR歌词重建函数
from music_video_pipeline.modules.module_a_v2.funasr_lyrics import (
    _compute_dynamic_sentence_split_gap,
    _filter_high_outliers_by_log_mad,
    _resolve_funasr_model_and_vad,
    build_lyric_units_from_funasr_result,
    recognize_lyrics_with_funasr_v2,
)
# 项目内模块：窗口归一（验证句尾标点边界保留）
from music_video_pipeline.modules.module_a_v2.timeline.window_builder import normalize_sentence_units


def test_filter_high_outliers_by_log_mad_should_pick_extreme_gaps() -> None:
    """
    功能说明：验证 log1p+MAD 可自动识别超长离群空歇。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：离群应只剔除明显大值，不误伤主体样本。
    """
    raw_samples = [0.42, 0.55, 0.61, 0.74, 0.89, 10.98, 16.15]
    kept_samples, outlier_samples = _filter_high_outliers_by_log_mad(raw_samples)
    assert any(abs(item - 16.15) <= 1e-6 for item in outlier_samples)
    assert any(abs(item - 10.98) <= 1e-6 for item in outlier_samples)
    assert all(item < 3.0 for item in kept_samples)


def test_compute_dynamic_sentence_split_gap_should_use_punctuation_neighbor_samples() -> None:
    """
    功能说明：验证分句阈值使用“标点左右内容token边界间隔”样本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅标点样本参与阈值估计，sample_source 应为 punctuation_neighbor。
    """
    token_items = [
        {"text": "你", "start_time": 0.00, "end_time": 0.20},
        {"text": "，", "start_time": 0.20, "end_time": 0.24},
        {"text": "好", "start_time": 0.46, "end_time": 0.62},
        {"text": "，", "start_time": 0.62, "end_time": 0.66},
        {"text": "世", "start_time": 1.03, "end_time": 1.20},
        {"text": "，", "start_time": 1.20, "end_time": 1.24},
        {"text": "界", "start_time": 1.84, "end_time": 2.02},
    ]
    threshold, stats = _compute_dynamic_sentence_split_gap(token_items=token_items)
    assert stats["sample_source"] == "punctuation_neighbor"
    assert stats["sample_count_raw"] == 3
    assert stats["sample_count_kept"] >= 2
    assert threshold > 0.2


def test_compute_dynamic_sentence_split_gap_should_fallback_to_default_when_no_punctuation_samples() -> None:
    """
    功能说明：验证没有可用标点样本时，会回退默认阈值 0.35 秒。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：回退路径下 sample_source=none。
    """
    token_items = [
        {"text": "爱", "start_time": 0.00, "end_time": 0.20},
        {"text": "你", "start_time": 0.32, "end_time": 0.52},
        {"text": "我", "start_time": 0.64, "end_time": 0.84},
    ]
    threshold, stats = _compute_dynamic_sentence_split_gap(token_items=token_items)
    assert stats["sample_source"] == "none"
    assert stats["sample_count_raw"] == 0
    assert abs(threshold - 0.35) <= 1e-6


def test_build_lyric_units_from_funasr_result_should_split_by_gap_not_by_punctuation_only() -> None:
    """
    功能说明：验证分句触发基于 gap，不是标点本身。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅标点但无明显空歇时，不应被强制拆句。
    """
    raw_result = [
        {
            "text": "你，好，世界。",
            "timestamps": [
                {"text": "你", "start": 0.00, "end": 0.20, "granularity": "char"},
                {"text": "，", "start": 0.20, "end": 0.22, "granularity": "char"},
                {"text": "好", "start": 0.23, "end": 0.41, "granularity": "char"},
                {"text": "，", "start": 0.41, "end": 0.43, "granularity": "char"},
                {"text": "世", "start": 0.95, "end": 1.11, "granularity": "char"},
                {"text": "界", "start": 1.13, "end": 1.31, "granularity": "char"},
                {"text": "。", "start": 1.31, "end": 1.34, "granularity": "char"},
            ],
        }
    ]
    lyric_units_raw, split_stats = build_lyric_units_from_funasr_result(raw_result=raw_result)
    assert len(lyric_units_raw) == 2
    assert lyric_units_raw[0]["text"].endswith("，")
    assert split_stats["dynamic_gap_threshold_seconds"] > 0.0


def test_normalize_sentence_units_should_keep_tail_punctuation_right_boundary() -> None:
    """
    功能说明：验证句子以标点结尾时，句右边界保留到标点右边界。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：句尾标点晚于最后有效内容token时，不应被裁剪到内容token末尾。
    """
    sentence_units = [
        {
            "start_time": 10.00,
            "end_time": 10.92,
            "text": "我爱你。",
            "token_units": [
                {"text": "我", "start_time": 10.00, "end_time": 10.16},
                {"text": "爱", "start_time": 10.20, "end_time": 10.36},
                {"text": "你", "start_time": 10.40, "end_time": 10.58},
                {"text": "。", "start_time": 10.72, "end_time": 10.92},
            ],
        }
    ]
    normalized = normalize_sentence_units(sentence_units=sentence_units, duration_seconds=30.0)
    assert len(normalized) == 1
    assert abs(float(normalized[0]["end_time"]) - 10.92) <= 1e-6


def test_build_lyric_units_from_funasr_result_should_keep_english_space_semantics() -> None:
    """
    功能说明：验证英文token前导空格语义会保留，句文本不再连写。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：token_units 输出不再要求 granularity 字段。
    """
    raw_result = [
        {
            "text": "you sounds like a fuck.",
            "timestamps": [
                {"token": "you", "start_time": 0.00, "end_time": 0.08},
                {"token": " sounds", "start_time": 0.10, "end_time": 0.18},
                {"token": " like", "start_time": 0.20, "end_time": 0.28},
                {"token": " a", "start_time": 0.30, "end_time": 0.34},
                {"token": " fuck", "start_time": 0.36, "end_time": 0.44},
                {"token": ".", "start_time": 0.46, "end_time": 0.50},
            ],
        }
    ]
    lyric_units_raw, _split_stats = build_lyric_units_from_funasr_result(raw_result=raw_result)
    assert len(lyric_units_raw) == 1
    assert lyric_units_raw[0]["text"] == "you sounds like a fuck."
    assert lyric_units_raw[0]["token_units"][1]["text"].startswith(" ")
    assert "granularity" not in lyric_units_raw[0]["token_units"][0]


def test_resolve_funasr_model_and_vad_should_prefer_local_cache(monkeypatch, tmp_path) -> None:
    """
    功能说明：验证主模型与VAD命中本地缓存目录时优先返回本地路径。
    参数说明：
    - monkeypatch: pytest补丁工具。
    - tmp_path: 临时目录夹具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：缓存命中需包含 model.pt 文件。
    """
    cache_root = tmp_path / "models"
    main_model_dir = cache_root / "FunAudioLLM" / "Fun-ASR-Nano-2512"
    vad_model_dir = cache_root / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    main_model_dir.mkdir(parents=True, exist_ok=True)
    vad_model_dir.mkdir(parents=True, exist_ok=True)
    (main_model_dir / "model.pt").write_text("ok", encoding="utf-8")
    (vad_model_dir / "model.pt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.funasr_lyrics.MODELSCOPE_MODELS_DIR",
        cache_root,
    )

    resolved_model, resolved_vad, model_hit, vad_hit = _resolve_funasr_model_and_vad(
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model_name="fsmn-vad",
    )
    assert model_hit is True
    assert vad_hit is True
    assert resolved_model == str(main_model_dir)
    assert resolved_vad == str(vad_model_dir)


def test_recognize_lyrics_with_funasr_v2_should_set_disable_update_and_use_cached_paths(monkeypatch) -> None:
    """
    功能说明：验证 FunASR 初始化会开启 disable_update 且支持直接使用本地缓存路径。
    参数说明：
    - monkeypatch: pytest补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用伪模型避免真实推理与联网。
    """
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            captured_kwargs["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured_kwargs["generate_kwargs"] = kwargs
            return []

    fake_module = types.SimpleNamespace(AutoModel=_FakeModel, __version__="1.3.1")
    monkeypatch.setitem(sys.modules, "funasr", fake_module)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.funasr_lyrics._resolve_funasr_model_and_vad",
        lambda **_kwargs: ("/tmp/funasr_model", "/tmp/funasr_vad", True, True),
    )

    raw_result, lyric_units, split_stats = recognize_lyrics_with_funasr_v2(
        audio_path="dummy.wav",
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        device="auto",
        funasr_language="auto",
        logger=logging.getLogger("test_module_a_v2_funasr_lyrics"),
    )

    assert raw_result == []
    assert lyric_units == []
    assert split_stats["reason"] == "empty_records"
    assert captured_kwargs["model_kwargs"]["model"] == "/tmp/funasr_model"
    assert captured_kwargs["model_kwargs"]["vad_model"] == "/tmp/funasr_vad"
    assert captured_kwargs["model_kwargs"]["check_latest"] is False
    assert captured_kwargs["model_kwargs"]["disable_update"] is True
