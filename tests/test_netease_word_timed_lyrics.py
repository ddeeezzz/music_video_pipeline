"""
文件用途：验证网易云逐字歌词转换与增强 LRC 解析。
核心流程：覆盖 YRC -> 增强 LRC，以及增强 LRC -> token_units 两段关键能力。
输入输出：输入最小歌词样例，输出逐字时间轴断言。
依赖说明：依赖 pytest 与模块A内部歌词解析工具。
维护说明：当网易云逐字格式适配调整时，需同步更新本测试。
"""

import logging

from music_video_pipeline.modules.module_a_v2.lrc_parser import parse_synced_lyrics_to_sentence_units
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.netease_music import (
    _build_netease_word_timed_lyrics,
    _normalize_netease_candidate,
)


def test_build_netease_word_timed_lyrics_should_convert_yrc_to_enhanced_lrc() -> None:
    """
    功能说明：验证网易云 YRC 可以转换为增强 LRC。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：保留行级时间戳与词级 `<start>词<end>` 标记。
    """
    yrc_text = "[1000,1200](1000,300,0)独(1300,250,0)り(1550,650,0)んぼ"

    result = _build_netease_word_timed_lyrics(yrc_text)

    assert result == "[00:01.00]<00:01.00>独<00:01.30><00:01.30>り<00:01.55><00:01.55>んぼ<00:02.20>"


def test_parse_synced_lyrics_to_sentence_units_should_parse_enhanced_lrc_tokens() -> None:
    """
    功能说明：验证增强 LRC 能被模块A解析为句级与词级时间单元。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：句文本应移除增强时间标签，token_units 保留逐字时间。
    """
    logger = logging.getLogger("test_parse_synced_lyrics_to_sentence_units_should_parse_enhanced_lrc_tokens")
    enhanced_lrc = (
        "[00:01.00]<00:01.00>独<00:01.30><00:01.30>り<00:01.55><00:01.55>んぼ<00:02.20>\n"
        "[00:03.00]エンヴィー"
    )

    result = parse_synced_lyrics_to_sentence_units(enhanced_lrc, audio_duration=5.0, logger=logger)

    assert len(result) == 2
    assert result[0]["text"] == "独りんぼ"
    assert result[0]["token_units"] == [
        {"text": "独", "start_time": 1.0, "end_time": 1.3},
        {"text": "り", "start_time": 1.3, "end_time": 1.55},
        {"text": "んぼ", "start_time": 1.55, "end_time": 2.2},
    ]
    assert result[1]["text"] == "エンヴィー"
    assert result[1]["token_units"] == []


def test_normalize_netease_candidate_should_include_word_timed_translation_and_romanized() -> None:
    """
    功能说明：验证网易云候选归一化时会保留逐字歌词、翻译与罗马音。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：存在 yrc 时同步歌词仍使用普通 lrc，逐字歌词单独存入 word_timed_lyrics。
    """
    candidate = _normalize_netease_candidate(
        song={"id": "27515069", "artist": "初音ミク", "title": "独りんぼエンヴィー", "duration_seconds": 214.0},
        lyric_payload={
            "lrc": {"lyric": "[00:21.75]悪戯は知らん顔で"},
            "yrc": {"lyric": "[21750,2200](21750,500,0)悪戯(22250,1700,0)は知らん顔で"},
            "ytlrc": {"lyric": "[00:21.750]作恶作剧的孩子 摆出一副不知情的样子"},
            "romalrc": {"lyric": "[00:21.75]i ta zu ra wa shi ra n ka o de"},
        },
    )

    assert candidate["status"] == "synced"
    assert candidate["synced_lyrics"] == "[00:21.75]悪戯は知らん顔で"
    assert candidate["word_timed_lyrics"].startswith("[00:21.75]<00:21.75>悪戯")
    assert candidate["translated_lyrics"] == "[00:21.750]作恶作剧的孩子 摆出一副不知情的样子"
    assert candidate["romanized_lyrics"] == "[00:21.75]i ta zu ra wa shi ra n ka o de"
