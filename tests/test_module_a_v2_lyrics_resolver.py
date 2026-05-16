"""
文件用途：验证模块A V2 歌词来源优先级解析逻辑。
核心流程：打桩元数据、LRCLIB 与 FunASR 回调，断言 provider 决策与输出结构。
输入输出：输入伪造依赖与临时产物目录，输出断言结果。
依赖说明：依赖 pytest monkeypatch 与 lyrics_resolver 纯编排逻辑。
维护说明：若阶段一 provider 规则调整，需同步更新本测试。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：JSON读取
from music_video_pipeline.io_utils import read_json
# 项目内模块：V2产物路径
from music_video_pipeline.modules.module_a_v2.artifacts import build_module_a_v2_artifacts
# 项目内模块：歌词来源解析入口
from music_video_pipeline.modules.module_a_v2.lyrics_resolver import resolve_lyrics_with_priority


def test_resolve_lyrics_with_priority_should_stop_on_instrumental(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 LRCLIB 明确返回 instrumental 时，直接终止歌词链且不进入 FunASR。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：instrumental=true 时输出空歌词单元。
    """
    audio_path = tmp_path / "demo.m4a"
    audio_path.write_bytes(b"fake-audio")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_lyrics_resolver_instrumental")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "Artist",
            "title": "Song",
            "album": "",
            "duration_seconds": 120.0,
            "source": "embedded_tags",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.query_lrclib_lyrics",
        lambda **_kwargs: {
            "status": "instrumental",
            "artist": "Artist",
            "title": "Song",
            "duration_seconds": 120.0,
            "plain_lyrics": "",
            "synced_lyrics": "",
            "provider": "lrclib",
            "provider_id": "1",
            "instrumental": True,
            "error": "",
        },
    )

    funasr_called = {"count": 0}

    def _fake_funasr_runner():
        funasr_called["count"] += 1
        return None

    result = resolve_lyrics_with_priority(
        audio_path=audio_path,
        duration_seconds=120.0,
        artifacts=artifacts,
        logger=logger,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        enable_fingerprint_lookup=True,
        funasr_fallback_runner=_fake_funasr_runner,
    )

    assert result["provider"] == "lrclib"
    assert result["reason"] == "instrumental"
    assert result["lyric_sentence_units"] == []
    assert funasr_called["count"] == 0
    selected_provider_payload = read_json(artifacts.perception_model_lyrics_selected_provider_path)
    assert selected_provider_payload["provider"] == "lrclib"
    assert selected_provider_payload["reason"] == "instrumental"


def test_resolve_lyrics_with_priority_should_prefer_lrclib_synced(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 LRCLIB 命中同步歌词时优先使用主链结果，不再触发 FunASR。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：LRC 解析结果作为最终歌词句单元返回。
    """
    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_lyrics_resolver_synced")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "Artist",
            "title": "Song",
            "album": "",
            "duration_seconds": 100.0,
            "source": "embedded_tags",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.query_lrclib_lyrics",
        lambda **_kwargs: {
            "status": "synced",
            "artist": "Artist",
            "title": "Song",
            "duration_seconds": 100.0,
            "plain_lyrics": "hello",
            "synced_lyrics": "[00:00.00]hello\n[00:01.00]world",
            "provider": "lrclib",
            "provider_id": "2",
            "instrumental": False,
            "error": "",
        },
    )

    funasr_called = {"count": 0}

    def _fake_funasr_runner():
        funasr_called["count"] += 1
        return None

    result = resolve_lyrics_with_priority(
        audio_path=audio_path,
        duration_seconds=100.0,
        artifacts=artifacts,
        logger=logger,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        enable_fingerprint_lookup=True,
        funasr_fallback_runner=_fake_funasr_runner,
    )

    assert result["provider"] == "lrclib"
    assert result["reason"] == "metadata_synced"
    assert len(result["lyric_sentence_units"]) == 2
    assert funasr_called["count"] == 0


def test_resolve_lyrics_with_priority_should_use_fingerprint_when_metadata_not_found(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证元数据直查失败后，会继续走指纹补充链并命中 LRCLIB。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：阶段二只要求补齐“元数据失败 -> 指纹补充 -> LRCLIB”链路。
    """
    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    artifacts = build_module_a_v2_artifacts(tmp_path / "module_a_work_v2")
    logger = logging.getLogger("test_module_a_v2_lyrics_resolver_fingerprint")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "missing",
            "artist": "",
            "title": "",
            "album": "",
            "duration_seconds": 100.0,
            "source": "embedded_tags",
            "error": "artist/title 不完整",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.build_fingerprint_result",
        lambda **_kwargs: {
            "fingerprint": "abc",
            "duration_seconds": 100.0,
            "fingerprint_engine": "chromaprint",
            "status": "ok",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.query_acoustid_match",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "Artist 2",
            "title": "Song 2",
            "duration_seconds": 100.0,
            "score": 0.99,
            "acoustid_id": "ac_1",
            "recording_id": "rec_1",
            "raw_candidates": [],
            "error": "",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_resolver.query_lrclib_lyrics",
        lambda artist, title, duration_seconds, logger: {
            "status": "synced" if artist == "Artist 2" else "not_found",
            "artist": artist,
            "title": title,
            "duration_seconds": duration_seconds,
            "plain_lyrics": "",
            "synced_lyrics": "[00:00.00]foo\n[00:01.00]bar" if artist == "Artist 2" else "",
            "provider": "lrclib",
            "provider_id": "3",
            "instrumental": False,
            "error": "",
        },
    )

    funasr_called = {"count": 0}

    def _fake_funasr_runner():
        funasr_called["count"] += 1
        return None

    result = resolve_lyrics_with_priority(
        audio_path=audio_path,
        duration_seconds=100.0,
        artifacts=artifacts,
        logger=logger,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        enable_fingerprint_lookup=True,
        funasr_fallback_runner=_fake_funasr_runner,
    )

    assert result["provider"] == "lrclib"
    assert result["reason"] == "fingerprint_synced"
    assert len(result["lyric_sentence_units"]) == 2
    assert funasr_called["count"] == 0
