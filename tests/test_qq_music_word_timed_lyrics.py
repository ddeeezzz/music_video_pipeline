"""
文件用途：验证 QQ 音乐 QRC 的增强 LRC 转换与歌词 bundle 输出。
核心流程：覆盖 QRC -> 增强 LRC，以及 musicu bundle -> word_timed_lyrics 两段关键能力。
输入输出：输入最小QRC样例与模拟payload，输出逐字时间轴断言。
依赖说明：依赖 pytest 与 QQ provider 纯函数。
维护说明：当 QQ QRC 格式适配调整时，需同步更新本测试。
"""

from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music import fetch_qq_music_lyrics_bundle
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music_qrc import (
    extract_enhanced_lrc_from_qq_music_qrc,
)


def test_extract_enhanced_lrc_from_qq_music_qrc_should_preserve_word_timing() -> None:
    """
    功能说明：验证 QQ QRC 可转换为增强 LRC。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：保留行级时间戳与词级 `<start>词<end>` 标记。
    """
    qrc_text = (
        '<QrcInfos><LyricInfo><Lyric_1 LyricType="1" '
        'LyricContent="[1000,1200]独(0,300)り(300,250)んぼ(550,650)&#10;[3000,600]エンヴィー(0,600)"/>'
        "</LyricInfo></QrcInfos>"
    )

    result = extract_enhanced_lrc_from_qq_music_qrc(qrc_text)

    assert result == (
        "[00:01.00]<00:01.00>独<00:01.30><00:01.30>り<00:01.55><00:01.55>んぼ<00:02.20>\n"
        "[00:03.00]<00:03.00>エンヴィー<00:03.60>"
    )


def test_fetch_qq_music_lyrics_bundle_should_include_word_timed_lyrics(monkeypatch) -> None:
    """
    功能说明：验证 QQ 富歌词 bundle 会返回逐字增强 LRC。
    参数说明：
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同步歌词仍保留普通LRC，逐字歌词单独存入 word_timed_lyrics。
    """
    qrc_text = (
        '<QrcInfos><LyricInfo><Lyric_1 LyricType="1" '
        'LyricContent="[1000,1200]独(0,300)り(300,250)んぼ(550,650)&#10;[3000,600]エンヴィー(0,600)"/>'
        "</LyricInfo></QrcInfos>"
    )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music.fetch_qq_music_synced_lyrics",
        lambda **_kwargs: "[00:01.00]独りんぼ\n[00:03.00]エンヴィー",
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music._fetch_qq_music_musicu_lyrics",
        lambda **_kwargs: {
            "lyric": "dummy-lyric",
            "qrc_t": "1",
            "trans": "dummy-trans",
            "trans_t": "1",
            "roma": "dummy-roma",
            "roma_t": "1",
        },
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music.decrypt_qq_music_qrc",
        lambda encrypted_text: qrc_text if encrypted_text == "dummy-lyric" else (
            '<QrcInfos><LyricInfo><Lyric_1 LyricType="1" LyricContent="[1000,1200]恶作剧(0,1200)"/></LyricInfo></QrcInfos>'
            if encrypted_text == "dummy-trans"
            else '<QrcInfos><LyricInfo><Lyric_1 LyricType="1" LyricContent="[1000,1200]i ta zu ra(0,1200)"/></LyricInfo></QrcInfos>'
        ),
    )

    bundle = fetch_qq_music_lyrics_bundle(song_mid="002VouWK1mlErw", song_id="12345", artist="初音未来", title="独りんぼエンヴィー", logger=None)

    assert bundle["synced_lyrics"] == "[00:01.00]独りんぼ\n[00:03.00]エンヴィー"
    assert bundle["word_timed_lyrics"].startswith("[00:01.00]<00:01.00>独<00:01.30>")
    assert bundle["translated_lyrics"] == "恶作剧"
    assert bundle["romanized_lyrics"] == "i ta zu ra"
