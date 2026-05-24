"""
文件用途：验证酷狗歌词provider的KRC解析与候选归一化。
核心流程：覆盖KRC富歌词解析、同步歌词判定与逐字字段保留。
输入输出：输入最小KRC样例与候选样例，输出标准化结构断言。
依赖说明：依赖 pytest 与酷狗provider纯函数。
维护说明：当酷狗KRC适配规则调整时，需同步更新本测试。
"""

import base64
import json

from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.kugou_music import (
    _normalize_kugou_candidate,
    _parse_kugou_krc_bundle,
)


def test_parse_kugou_krc_bundle_should_extract_synced_word_timed_translation_and_romanized() -> None:
    """
    功能说明：验证KRC文本可解析出原文、逐字、翻译与罗马音。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：语言标签里同时含有翻译与罗马音。
    """
    language_tag = base64.b64encode(
        json.dumps(
            {
                "content": [
                    {"type": 0, "lyricContent": [["i ta ", "zu ra"]]},
                    {"type": 1, "lyricContent": [["恶作剧"]]},
                ],
                "version": 1,
            },
            ensure_ascii=False,
        ).encode("utf-8")
    ).decode("ascii")
    krc_text = (
        f"[language:{language_tag}]\n"
        "[1000,1000]<0,300,0>悪<300,700,0>戯"
    )

    result = _parse_kugou_krc_bundle(krc_text)

    assert result["synced_lyrics"] == "[00:01.00]悪戯"
    assert result["word_timed_lyrics"] == "[00:01.00]<00:01.00>悪<00:01.30><00:01.30>戯<00:02.00>"
    assert result["translated_lyrics"] == "[00:01.00]恶作剧"
    assert result["romanized_lyrics"] == "[00:01.00]i ta zu ra"


def test_normalize_kugou_candidate_should_preserve_provider_lookup_fields() -> None:
    """
    功能说明：验证酷狗候选归一化时会保留补拉所需的provider字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：provider_accesskey/provider_hash/provider_song_id 均应透传。
    """
    result = _normalize_kugou_candidate(
        song={
            "id": "31860080",
            "title": "独りんぼエンヴィー",
            "artist": "初音ミク",
            "duration_seconds": 202.109,
            "hash": "10f27d0dfb24b12275e41a2c9752b71c",
        },
        lyric_candidate={
            "id": "295011816",
            "accesskey": "ACCESSKEY",
            "score": 60,
            "duration_seconds": 202.109,
        },
        lyric_bundle={
            "synced_lyrics": "[00:21.75]悪戯は知らん顔で",
            "word_timed_lyrics": "[00:21.75]<00:21.75>悪戯<00:22.25>",
            "translated_lyrics": "[00:21.75]作恶作剧的孩子",
            "romanized_lyrics": "[00:21.75]i ta zu ra",
        },
    )

    assert result["status"] == "synced"
    assert result["provider"] == "kugou_music"
    assert result["provider_id"] == "295011816"
    assert result["provider_song_id"] == "31860080"
    assert result["provider_accesskey"] == "ACCESSKEY"
    assert result["provider_hash"] == "10f27d0dfb24b12275e41a2c9752b71c"
