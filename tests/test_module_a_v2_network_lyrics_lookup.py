"""
文件用途：验证模块A页面联网歌词候选检索的优先级与兜底路径。
核心流程：覆盖元信息优先、指纹回退、手动歌曲名搜索与手动兜底提示。
输入输出：输入临时音频路径与 monkeypatch 补丁，输出查询摘要断言。
依赖说明：依赖 pytest、logging 与 network_lyrics_lookup 纯编排逻辑。
维护说明：当 Web 侧找词优先级调整时，需同步更新本测试。
"""

# 标准库：用于构造测试日志对象
import logging
# 标准库：用于临时文件路径类型
from pathlib import Path

# 第三方库：用于断言与补丁
import pytest

# 项目内模块：模块A页面联网歌词候选查询入口
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline import search_synced_lrc_candidates


def test_search_synced_lrc_candidates_should_prefer_metadata_search(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证自动链存在可用元信息候选时，不再进入指纹识别。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：元信息搜索直接命中同步歌词候选。
    """
    audio_path = tmp_path / "metadata-hit.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_search_synced_lrc_candidates_should_prefer_metadata_search")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "Singer",
            "title": "Song",
            "duration_seconds": 123.4,
        },
    )

    def _fake_search_lrclib_candidates(**kwargs):  # noqa: ANN003
        assert kwargs["query_text"] == "Song"
        return [
            {
                "status": "synced",
                "artist": "Singer",
                "title": "Song",
                "duration_seconds": 123.4,
                "plain_lyrics": "hello",
                "synced_lyrics": "[00:00.00]hello",
                "provider_id": "lrclib-meta-1",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        _fake_search_lrclib_candidates,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("元信息已命中时不应继续生成指纹"),
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=123.4,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
    )

    assert result["status"] == "ok"
    assert result["search_mode"] == "metadata"
    assert result["suggest_manual_query"] is False
    assert result["metadata_trace"]["embedded_status"] == "ok"
    assert result["metadata_trace"]["embedded_title"] == "Song"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["title"] == "Song"


def test_search_synced_lrc_candidates_should_use_raw_metadata_title_for_fuzzy_search(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证元信息模糊搜索优先使用原始标题，而不是额外净化标题。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使标题带括号前缀，也先把原始标题交给模糊搜索。
    """
    audio_path = tmp_path / "metadata-uploader-prefix.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_use_raw_metadata_title_for_fuzzy_search"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "凉祈言",
            "title": "【ABbbb君】孑然妒火",
            "duration_seconds": 111.0,
        },
    )

    observed_queries: list[str] = []

    def _fake_search_lrclib_candidates(**kwargs):  # noqa: ANN003
        observed_queries.append(str(kwargs["query_text"]))
        if kwargs["query_text"] == "【ABbbb君】孑然妒火":
            return [
                {
                    "status": "synced",
                    "artist": "凉祈言",
                    "title": "【ABbbb君】孑然妒火",
                    "duration_seconds": 111.0,
                    "plain_lyrics": "hello",
                    "synced_lyrics": "[00:00.00]hello",
                    "provider_id": "lrclib-meta-prefix-1",
                }
            ]
        return []

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        _fake_search_lrclib_candidates,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("净化后的标题已命中时不应继续生成指纹"),
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=111.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
    )

    assert observed_queries[0] == "【ABbbb君】孑然妒火"
    assert result["status"] == "ok"
    assert result["search_mode"] == "metadata"
    assert result["metadata_trace"]["embedded_title"] == "【ABbbb君】孑然妒火"
    assert result["candidates"][0]["title"] == "【ABbbb君】孑然妒火"


def test_search_synced_lrc_candidates_should_fallback_to_fingerprint(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证元信息未命中时会回退到音频指纹链继续查找。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：指纹识别命中后继续用 LRCLIB 严格查询歌词。
    """
    audio_path = tmp_path / "fingerprint-hit.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_search_synced_lrc_candidates_should_fallback_to_fingerprint")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: {"status": "missing", "artist": "", "title": "", "duration_seconds": 120.0},
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: {
            "status": "ok",
            "fingerprint": "fake-fingerprint",
            "duration_seconds": 120.0,
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.query_acoustid_match",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "Fingerprint Singer",
            "title": "Fingerprint Song",
            "score": 0.97,
            "raw_candidates": [
                {
                    "id": "acoustid-1",
                    "score": 0.97,
                    "recordings": [
                        {
                            "id": "recording-1",
                            "title": "Fingerprint Song",
                            "artists": [{"name": "Fingerprint Singer"}],
                        }
                    ],
                }
            ],
        },
    )

    def _fake_query_lrclib_lyrics(**kwargs):  # noqa: ANN003
        assert kwargs["artist"] == "Fingerprint Singer"
        assert kwargs["title"] == "Fingerprint Song"
        return {
            "status": "synced",
            "artist": "Fingerprint Singer",
            "title": "Fingerprint Song",
            "duration_seconds": 120.0,
            "plain_lyrics": "hello",
            "synced_lyrics": "[00:00.00]hello",
            "provider_id": "lrclib-fingerprint-1",
        }

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.query_lrclib_lyrics",
        _fake_query_lrclib_lyrics,
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=120.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
    )

    assert result["status"] == "ok"
    assert result["search_mode"] == "fingerprint"
    assert result["suggest_manual_query"] is False
    assert result["metadata_trace"]["embedded_status"] == "missing"
    assert result["metadata_trace"]["acoustid_status"] == "ok"
    assert result["metadata_trace"]["matched_title"] == "Fingerprint Song"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["artist"] == "Fingerprint Singer"


def test_search_synced_lrc_candidates_should_suggest_manual_query_when_auto_chain_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证自动链失败时会提示前端转入手动歌曲名搜索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：这里模拟 fpcalc 不可用，等价于当前机器缺少指纹工具。
    """
    audio_path = tmp_path / "auto-failed.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_search_synced_lrc_candidates_should_suggest_manual_query_when_auto_chain_failed")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: {"status": "missing", "artist": "", "title": "", "duration_seconds": 90.0},
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: {"status": "failed", "error": "fpcalc_not_found", "duration_seconds": 90.0},
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=90.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
    )

    assert result["status"] == "failed"
    assert result["search_mode"] == "automatic"
    assert result["suggest_manual_query"] is True
    assert result["metadata_trace"]["fingerprint_status"] == "failed"
    assert "手动输入歌曲名" in result["message"]


def test_search_synced_lrc_candidates_should_use_manual_query_first(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证用户手动输入歌曲名时，会直接走 LRCLIB 搜索而不触发自动链。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：手动输入支持自由文本，不要求 artist/title 结构化完整。
    """
    audio_path = tmp_path / "manual-query.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_search_synced_lrc_candidates_should_use_manual_query_first")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )

    def _fake_search_lrclib_candidates(**kwargs):  # noqa: ANN003
        assert kwargs["query_text"] == "郭顶 水星记"
        return [
            {
                "status": "synced",
                "artist": "郭顶",
                "title": "水星记",
                "duration_seconds": 100.0,
                "plain_lyrics": "着迷于你眼睛",
                "synced_lyrics": "[00:00.00]着迷于你眼睛",
                "provider_id": "lrclib-manual-1",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        _fake_search_lrclib_candidates,
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=100.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="郭顶 水星记",
    )

    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert result["suggest_manual_query"] is False
    assert result["metadata_trace"]["embedded_status"] == "skipped"
    assert any(item["title"] == "水星记" for item in result["candidates"])


def test_search_synced_lrc_candidates_should_parse_artist_title_manual_query_first(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动输入“歌手 - 歌名”时，会优先按 artist/title 结构化字段检索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：结构化 artist/title 命中后，不再回退到整句模糊搜索。
    """
    audio_path = tmp_path / "manual-query-artist-title.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_parse_artist_title_manual_query_first"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )

    observed_queries: list[dict[str, str]] = []

    def _fake_search_lrclib_candidates(**kwargs):  # noqa: ANN003
        observed_queries.append(
            {
                "query_text": str(kwargs.get("query_text", "")),
                "artist": str(kwargs.get("artist", "")),
                "title": str(kwargs.get("title", "")),
            }
        )
        if kwargs.get("artist") == "郭顶" and kwargs.get("title") == "水星记":
            return [
                {
                    "status": "synced",
                    "artist": "郭顶",
                    "title": "水星记",
                    "duration_seconds": 100.0,
                    "plain_lyrics": "着迷于你眼睛",
                    "synced_lyrics": "[00:00.00]着迷于你眼睛",
                    "provider_id": "lrclib-manual-artist-title-1",
                }
            ]
        return []

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        _fake_search_lrclib_candidates,
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=100.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="郭顶 - 水星记",
    )

    assert observed_queries[0] == {"query_text": "", "artist": "郭顶", "title": "水星记"}
    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert result["suggest_manual_query"] is False
    assert any(item["artist"] == "郭顶" and item["title"] == "水星记" for item in result["candidates"])


def test_search_synced_lrc_candidates_should_prefer_netease_music_when_manual_query_hits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动搜歌名时，会优先尝试网易云歌词搜索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：网易云命中后，不应继续回退到 LRCLIB 或 syncedlyrics。
    """
    audio_path = tmp_path / "manual-query-netease-hit.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_prefer_netease_music_when_manual_query_hits"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )

    observed_netease_queries: list[dict[str, str]] = []

    def _fake_search_netease_music_candidates(**kwargs):  # noqa: ANN003
        observed_netease_queries.append(
            {
                "query_text": str(kwargs.get("query_text", "")),
                "artist": str(kwargs.get("artist", "")),
                "title": str(kwargs.get("title", "")),
            }
        )
        return [
            {
                "status": "synced",
                "artist": "koyori/初音ミク",
                "title": "独りんぼエンヴィー",
                "duration_seconds": 204.506,
                "plain_lyrics": "",
                "synced_lyrics": "[00:00.55]独りんぼエンヴィー",
                "provider": "netease_music",
                "provider_id": "27515069",
                "instrumental": False,
                "error": "",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        _fake_search_netease_music_candidates,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=0.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="孑然妒火",
    )

    assert observed_netease_queries[0] == {"query_text": "孑然妒火", "artist": "", "title": ""}
    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "netease_music"
    assert result["candidates"][0]["title"] == "独りんぼエンヴィー"


def test_search_synced_lrc_candidates_should_fallback_to_qq_music_when_netease_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动搜歌名在网易云无结果时，会继续尝试 QQ 音乐歌词搜索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：QQ 音乐命中后，不应继续回退到 LRCLIB 或 syncedlyrics。
    """
    audio_path = tmp_path / "manual-query-qq-hit.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_fallback_to_qq_music_when_netease_empty"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )

    observed_qq_queries: list[dict[str, str]] = []

    def _fake_search_qq_music_candidates(**kwargs):  # noqa: ANN003
        observed_qq_queries.append(
            {
                "query_text": str(kwargs.get("query_text", "")),
                "artist": str(kwargs.get("artist", "")),
                "title": str(kwargs.get("title", "")),
            }
        )
        return [
            {
                "status": "synced",
                "artist": "初音未来",
                "title": "独りんぼエンヴィー (孑然妒火)",
                "duration_seconds": 0.0,
                "plain_lyrics": "",
                "synced_lyrics": "[00:01.36]独りんぼエンヴィー (孑然妒火) - 初音未来",
                "provider": "qq_music",
                "provider_id": "002VouWK1mlErw",
                "instrumental": False,
                "error": "",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        _fake_search_qq_music_candidates,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=0.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="独りんぼエンヴィー",
    )

    assert observed_qq_queries[0] == {"query_text": "独りんぼエンヴィー", "artist": "", "title": ""}
    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "qq_music"
    assert result["candidates"][0]["title"] == "独りんぼエンヴィー (孑然妒火)"


def test_search_synced_lrc_candidates_should_fallback_to_kugou_when_netease_and_qq_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动搜歌名在网易云和QQ都无结果时，会继续尝试酷狗歌词搜索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：酷狗命中后，不应继续回退到 LRCLIB 或 syncedlyrics。
    """
    audio_path = tmp_path / "manual-query-kugou-hit.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_fallback_to_kugou_when_netease_and_qq_empty"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )

    observed_kugou_queries: list[dict[str, str]] = []

    def _fake_search_kugou_music_candidates(**kwargs):  # noqa: ANN003
        observed_kugou_queries.append(
            {
                "query_text": str(kwargs.get("query_text", "")),
                "artist": str(kwargs.get("artist", "")),
                "title": str(kwargs.get("title", "")),
            }
        )
        return [
            {
                "status": "synced",
                "artist": "初音ミク",
                "title": "独りんぼエンヴィー",
                "duration_seconds": 202.109,
                "plain_lyrics": "",
                "synced_lyrics": "[00:21.75]悪戯は知らん顔で",
                "word_timed_lyrics": "[00:21.75]<00:21.75>悪戯<00:22.25><00:22.25>は知らん顔で<00:23.95>",
                "translated_lyrics": "[00:21.75]作恶作剧的孩子 摆出一副不知情的样子",
                "romanized_lyrics": "[00:21.75]i ta zu ra wa shi ra n ka o de",
                "provider": "kugou_music",
                "provider_id": "295011816",
                "provider_song_id": "31860080",
                "provider_accesskey": "ACCESSKEY",
                "provider_hash": "10f27d0dfb24b12275e41a2c9752b71c",
                "instrumental": False,
                "error": "",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        _fake_search_kugou_music_candidates,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=0.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="独りんぼエンヴィー",
    )

    assert observed_kugou_queries[0] == {"query_text": "独りんぼエンヴィー", "artist": "", "title": ""}
    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "kugou_music"
    assert result["candidates"][0]["provider_id"] == "295011816"


def test_search_synced_lrc_candidates_should_fallback_to_syncedlyrics_when_lrclib_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动搜歌名在 LRCLIB 无结果时，会继续尝试 syncedlyrics 聚合源。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：syncedlyrics 命中后，应直接返回可选同步歌词候选。
    """
    audio_path = tmp_path / "manual-query-syncedlyrics.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_fallback_to_syncedlyrics_when_lrclib_empty"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )

    observed_syncedlyrics_queries: list[dict[str, str]] = []

    def _fake_search_syncedlyrics_candidates(**kwargs):  # noqa: ANN003
        observed_syncedlyrics_queries.append(
            {
                "query_text": str(kwargs.get("query_text", "")),
                "artist": str(kwargs.get("artist", "")),
                "title": str(kwargs.get("title", "")),
            }
        )
        return [
            {
                "status": "synced",
                "artist": "",
                "title": "独りんぼエンヴィー",
                "duration_seconds": 0.0,
                "plain_lyrics": "",
                "synced_lyrics": "[00:22.75]悪戯は知らん顔で",
                "provider": "syncedlyrics",
                "provider_id": "独りんぼエンヴィー",
                "instrumental": False,
                "error": "",
            }
        ]

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        _fake_search_syncedlyrics_candidates,
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=0.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="独りんぼエンヴィー",
    )

    assert observed_syncedlyrics_queries[0] == {"query_text": "独りんぼエンヴィー", "artist": "", "title": ""}
    assert result["status"] == "ok"
    assert result["search_mode"] == "manual_query"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "syncedlyrics"
    assert result["candidates"][0]["title"] == "独りんぼエンヴィー"


def test_search_synced_lrc_candidates_should_sort_candidates_by_artist_title_similarity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证元信息命中时，会按歌手名与歌名字符相似度重排来源内候选。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：标题中包含真实歌名与歌手名的候选，应优先于无关视频标题候选。
    """
    audio_path = tmp_path / "metadata-similarity.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_search_synced_lrc_candidates_should_sort_candidates_by_artist_title_similarity")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: {
            "status": "ok",
            "artist": "凉祈言",
            "title": "【ABbbb君】孑然妒火",
            "duration_seconds": 111.0,
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [
            {
                "status": "synced",
                "artist": "未知UP主",
                "title": "ABbbb君投稿合集",
                "duration_seconds": 111.0,
                "synced_lyrics": "[00:01.00]foo",
                "provider": "netease_music",
                "provider_id": "netease-001",
            },
            {
                "status": "synced",
                "artist": "凉祈言",
                "title": "孑然妒火",
                "duration_seconds": 111.0,
                "synced_lyrics": "[00:01.00]hit",
                "provider": "netease_music",
                "provider_id": "netease-002",
            }
        ],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [
            {
                "status": "synced",
                "artist": "凉祈言",
                "title": "孑然妒火",
                "duration_seconds": 111.0,
                "synced_lyrics": "[00:01.00]bar",
                "provider": "qq_music",
                "provider_id": "qq-001",
            }
        ],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_syncedlyrics_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("元信息命中时不应继续生成指纹"),
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=111.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
    )

    assert result["status"] == "ok"
    assert result["provider_groups"][0]["provider"] == "netease_music"
    assert result["provider_groups"][1]["provider"] == "qq_music"
    assert result["provider_groups"][0]["candidates"][0]["title"] == "孑然妒火"
    assert result["provider_groups"][0]["candidates"][1]["title"] == "ABbbb君投稿合集"


def test_search_synced_lrc_candidates_should_strip_trailing_bracket_note_for_syncedlyrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证手动搜歌名带尾部括注时，会把净化后的标题交给 syncedlyrics 兜底搜索。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：原始查询无结果后，应继续尝试移除尾部括注的标题。
    """
    audio_path = tmp_path / "manual-query-syncedlyrics-bracket.mp3"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger(
        "test_search_synced_lrc_candidates_should_strip_trailing_bracket_note_for_syncedlyrics"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.read_embedded_metadata",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应读取自动链元信息"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.build_fingerprint_result",
        lambda **_kwargs: pytest.fail("手动搜歌名时不应继续生成指纹"),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_netease_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_qq_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_kugou_music_candidates",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline.search_lrclib_candidates",
        lambda **_kwargs: [],
    )

    observed_syncedlyrics_queries: list[str] = []

    class _FakeSyncedlyricsModule:
        """
        功能说明：模拟 syncedlyrics 第三方模块，仅记录搜索词并在净化后标题命中。
        参数说明：不适用。
        返回值：不适用。
        异常说明：不适用。
        边界条件：仅当搜索词等于净化标题时返回同步歌词。
        """

        @staticmethod
        def search(search_term: str, synced_only: bool = False) -> str | None:
            observed_syncedlyrics_queries.append(str(search_term))
            if synced_only and str(search_term) == "独りんぼエンヴィー":
                return "[00:22.75]悪戯は知らん顔で"
            return None

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.syncedlyrics._import_syncedlyrics_module",
        lambda logger: _FakeSyncedlyricsModule(),
    )

    result = search_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=0.0,
        fpcalc_bin="fpcalc",
        acoustid_api_key_file=".secrets/acoustid_api_key.txt",
        logger=logger,
        manual_query="独りんぼエンヴィー(充满嫉妒的一人捉迷藏)",
    )

    assert observed_syncedlyrics_queries == [
        "独りんぼエンヴィー(充满嫉妒的一人捉迷藏)",
        "独りんぼエンヴィー",
    ]
    assert result["status"] == "ok"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "syncedlyrics"
