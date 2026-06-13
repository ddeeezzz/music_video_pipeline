"""
文件用途：提供 QQ 音乐歌词搜索的最小封装。
核心流程：先按关键词走 smartbox 搜歌，再按 songmid 获取歌词，并归一化为项目内部结构。
输入输出：输入查询词或歌手/歌名，输出标准化同步歌词候选列表。
依赖说明：依赖标准库 urllib/json 与少量文本规范化工具。
维护说明：本文件只负责 QQ 音乐歌词搜索，不承担上层优先级编排职责。
"""

# 标准库：用于 JSON 解析
import json
# 标准库：用于 base64 编码
from base64 import b64encode
# 标准库：用于正则清洗查询词
import re
# 标准库：用于 URL 编码与 HTTP 请求
from urllib.parse import urlencode
from urllib.request import Request, urlopen
# 标准库：用于 HTTP 异常识别
from urllib.error import HTTPError, URLError
# 标准库：用于类型提示
from typing import Any, Callable

from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music_qrc import (
    decrypt_qq_music_qrc,
    extract_enhanced_lrc_from_qq_music_qrc,
    extract_lrc_from_qq_music_qrc,
    extract_lrc_with_fallback_from_qq_music_qrc,
)

# 常量：QQ 音乐 smartbox 搜索接口
QQ_MUSIC_SMARTBOX_API_URL = "https://c6.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg"
# 常量：QQ 音乐歌词接口
QQ_MUSIC_LYRIC_API_URL = "https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg"
# 常量：QQ 音乐 musicu 接口
QQ_MUSIC_MUSICU_API_URL = "https://u.y.qq.com/cgi-bin/musicu.fcg"
# 常量：QQ 音乐 HTTP 超时时间（秒）
QQ_MUSIC_REQUEST_TIMEOUT_SECONDS = 15.0
# 常量：QQ 音乐 musicu 请求超时时间
QQ_MUSIC_MUSICU_TIMEOUT_SECONDS = 30.0
# 常量：标题尾部括注清理规则
TRAILING_BRACKET_NOTE_PATTERN = re.compile(r"\s*[\(（\[【].*?[\)）\]】]\s*$")


def search_qq_music_candidates(
    *,
    query_text: str = "",
    artist: str = "",
    title: str = "",
    logger,
    limit: int = 10,
    candidate_callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：使用 QQ 音乐搜索歌曲并抓取可用同步歌词候选。
    参数说明：
    - query_text: 自由文本查询词。
    - artist: 艺人名。
    - title: 曲名。
    - logger: 日志对象。
    - limit: 最多返回候选数。
    返回值：
    - list[dict[str, Any]]: 已标准化的同步歌词候选数组。
    异常说明：网络或解析异常时返回空数组。
    边界条件：仅返回包含时间戳歌词的候选。
    """
    safe_limit = max(1, int(limit))
    search_terms = _build_qq_music_search_terms(
        query_text=str(query_text).strip(),
        artist=str(artist).strip(),
        title=str(title).strip(),
    )
    if not search_terms:
        return []
    normalized_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for search_term in search_terms:
        songs = _search_qq_music_songs(search_term=search_term, logger=logger, limit=max(10, safe_limit * 3))
        for song in songs:
            song_mid = str(song.get("songmid", "")).strip()
            song_id = str(song.get("id", "")).strip()
            normalized_artist = str(song.get("artist", "")).strip()
            normalized_title = str(song.get("title", "")).strip()
            unique_key = (song_mid.lower(), normalized_artist.lower(), normalized_title.lower())
            if not song_mid or unique_key in seen_keys:
                continue
            lyric_payload = _fetch_qq_music_lyric(song_mid=song_mid, logger=logger)
            normalized_candidate = _normalize_qq_music_candidate(
                song=song,
                lyric_payload=lyric_payload,
            )
            if normalized_candidate.get("status") != "synced":
                continue
            if not str(normalized_candidate.get("synced_lyrics", "")).strip():
                continue
            lyrics_bundle = fetch_qq_music_lyrics_bundle(
                song_mid=song_mid,
                song_id=song_id,
                artist=normalized_artist,
                title=normalized_title,
                logger=logger,
            )
            for field_name in ["synced_lyrics", "word_timed_lyrics", "translated_lyrics", "romanized_lyrics"]:
                normalized_value = str(lyrics_bundle.get(field_name, "")).strip()
                if normalized_value:
                    normalized_candidate[field_name] = normalized_value
            seen_keys.add(unique_key)
            normalized_candidates.append(normalized_candidate)
            if candidate_callback is not None:
                candidate_callback(dict(normalized_candidate))
            if len(normalized_candidates) >= safe_limit:
                logger.info(
                    "模块A V2-QQ音乐搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
                    str(artist).strip() or "<empty>",
                    str(title).strip() or "<empty>",
                    search_term,
                    len(normalized_candidates),
                )
                return normalized_candidates
    logger.info(
        "模块A V2-QQ音乐搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
        str(artist).strip() or "<empty>",
        str(title).strip() or "<empty>",
        str(query_text).strip() or "<empty>",
        len(normalized_candidates),
    )
    return normalized_candidates


def _build_qq_music_search_terms(query_text: str, artist: str, title: str) -> list[str]:
    """
    功能说明：构造 QQ 音乐歌曲搜索词尝试序列。
    参数说明：
    - query_text: 自由文本查询词。
    - artist: 艺人名。
    - title: 曲名。
    返回值：
    - list[str]: 去重后的搜索词数组。
    异常说明：无。
    边界条件：优先完整词，再尝试去尾部括注后的标题。
    """
    normalized_query_text = str(query_text).strip()
    normalized_artist = str(artist).strip()
    normalized_title = str(title).strip()
    search_terms: list[str] = []
    if normalized_query_text:
        search_terms.append(normalized_query_text)
        stripped_query_text = _strip_trailing_bracket_note(normalized_query_text)
        if stripped_query_text and stripped_query_text != normalized_query_text:
            search_terms.append(stripped_query_text)
    if normalized_title:
        if normalized_artist:
            search_terms.append(f"{normalized_artist} {normalized_title}".strip())
        search_terms.append(normalized_title)
        stripped_title = _strip_trailing_bracket_note(normalized_title)
        if stripped_title and stripped_title != normalized_title:
            if normalized_artist:
                search_terms.append(f"{normalized_artist} {stripped_title}".strip())
            search_terms.append(stripped_title)
    return _dedupe_search_terms(search_terms)


def _search_qq_music_songs(*, search_term: str, logger, limit: int) -> list[dict[str, Any]]:
    """
    功能说明：按关键词检索 QQ 音乐歌曲列表（优先 MusicU，回退 smartbox）。
    参数说明：
    - search_term: 搜索词。
    - logger: 日志对象。
    - limit: 最多返回歌曲数。
    返回值：
    - list[dict[str, Any]]: 轻量歌曲摘要列表。
    异常说明：网络或解析异常时返回空数组。
    边界条件：先尝试 MusicU 搜索，失败时回退 smartbox。
    """
    # 方案一：MusicU 搜索
    payload = json.dumps(
        {
            "comm": {"ct": 11, "cv": "1003006"},
            "req_1": {
                "method": "DoSearchForQQMusicDesktop",
                "module": "music.search.SearchCgiService",
                "param": {
                    "num_per_page": max(1, int(limit)),
                    "page_num": 1,
                    "query": str(search_term).strip(),
                    "search_type": 0,
                },
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        url=QQ_MUSIC_MUSICU_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://y.qq.com/",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=QQ_MUSIC_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        response_payload = json.loads(raw_body)
    except Exception as error:
        logger.warning("模块A V2-QQ音乐搜索MusicU失败，回退smartbox，query=%s，错误=%s", search_term, error)
        return _search_qq_music_songs_smartbox(search_term=search_term, logger=logger, limit=limit)
    req_data = response_payload.get("req_1", {}) if isinstance(response_payload, dict) else {}
    if not isinstance(req_data, dict):
        return _search_qq_music_songs_smartbox(search_term=search_term, logger=logger, limit=limit)
    body_data = req_data.get("data", {}) if isinstance(req_data, dict) else {}
    song_data = body_data.get("body", {}) if isinstance(body_data, dict) else {}
    song_list = song_data.get("song", {}).get("list", []) if isinstance(song_data, dict) else []
    if not isinstance(song_list, list) or not song_list:
        return _search_qq_music_songs_smartbox(search_term=search_term, logger=logger, limit=limit)
    normalized_songs: list[dict[str, Any]] = []
    for item in song_list:
        if not isinstance(item, dict):
            continue
        singer_names: list[str] = []
        for singer_item in item.get("singer", []):
            if isinstance(singer_item, dict):
                singer_name = str(singer_item.get("name", "")).strip()
                if singer_name:
                    singer_names.append(singer_name)
        normalized_songs.append(
            {
                "songmid": str(item.get("mid", "")).strip(),
                "id": str(item.get("id", "")).strip(),
                "title": str(item.get("title", item.get("name", ""))).strip(),
                "artist": "/".join(singer_names),
                "duration_seconds": float(item.get("interval", 0) or 0),
            }
        )
        if len(normalized_songs) >= max(1, int(limit)):
            break
    return normalized_songs


def _search_qq_music_songs_smartbox(*, search_term: str, logger, limit: int) -> list[dict[str, Any]]:
    """
    功能说明：按关键词检索 QQ 音乐歌曲列表（smartbox 回退方案）。
    参数说明：
    - search_term: 搜索词。
    - logger: 日志对象。
    - limit: 最多返回歌曲数。
    返回值：
    - list[dict[str, Any]]: 轻量歌曲摘要列表。
    异常说明：网络或解析异常时返回空数组。
    边界条件：基于 smartbox 提示结果，只抽取上层需要的最小字段。
    """
    query_string = urlencode({"key": str(search_term).strip(), "format": "json"})
    request = Request(
        url=f"{QQ_MUSIC_SMARTBOX_API_URL}?{query_string}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://y.qq.com/",
        },
    )
    try:
        with urlopen(request, timeout=QQ_MUSIC_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        response_payload = json.loads(raw_body)
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-QQ音乐搜索smartbox失败，query=%s，错误=%s", search_term, error)
        return []
    except URLError as error:
        logger.warning("模块A V2-QQ音乐搜索smartbox网络异常，query=%s，错误=%s", search_term, error)
        return []
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-QQ音乐搜索smartbox解析失败，query=%s，错误=%s", search_term, error)
        return []
    data_payload = response_payload.get("data", {}) if isinstance(response_payload, dict) else {}
    song_payload = data_payload.get("song", {}) if isinstance(data_payload, dict) else {}
    item_list = song_payload.get("itemlist", []) if isinstance(song_payload, dict) else []
    if not isinstance(item_list, list):
        return []
    normalized_songs: list[dict[str, Any]] = []
    for item in item_list:
        if not isinstance(item, dict):
            continue
        normalized_songs.append(
            {
                "songmid": str(item.get("mid", "")).strip(),
                "id": str(item.get("id", item.get("docid", ""))).strip(),
                "title": str(item.get("name", "")).strip(),
                "artist": str(item.get("singer", "")).strip(),
                "duration_seconds": 0.0,
            }
        )
        if len(normalized_songs) >= max(1, int(limit)):
            break
    return normalized_songs


def _fetch_qq_music_lyric(*, song_mid: str, logger) -> dict[str, Any]:
    """
    功能说明：按 songmid 获取 QQ 音乐歌词响应。
    参数说明：
    - song_mid: QQ 音乐 songmid。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: 原始歌词响应；异常时返回空对象。
    异常说明：网络或解析异常时返回空对象。
    边界条件：请求明文歌词文本，避免再做 base64 解码。
    """
    query_string = urlencode(
        {
            "songmid": song_mid,
            "format": "json",
            "nobase64": "1",
            "g_tk": "5381",
            "loginUin": "0",
            "hostUin": "0",
            "inCharset": "utf8",
            "outCharset": "utf-8",
            "notice": "0",
            "platform": "yqq.json",
            "needNewCode": "0",
        }
    )
    request = Request(
        url=f"{QQ_MUSIC_LYRIC_API_URL}?{query_string}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://y.qq.com/",
        },
    )
    try:
        with urlopen(request, timeout=QQ_MUSIC_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw_body)
        return payload if isinstance(payload, dict) else {}
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-QQ音乐歌词获取失败，songmid=%s，错误=%s", song_mid, error)
        return {}
    except URLError as error:
        logger.warning("模块A V2-QQ音乐歌词获取网络异常，songmid=%s，错误=%s", song_mid, error)
        return {}
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-QQ音乐歌词获取解析失败，songmid=%s，错误=%s", song_mid, error)
        return {}


def fetch_qq_music_synced_lyrics(*, song_mid: str, logger) -> str:
    """
    功能说明：按 songmid 补拉 QQ 音乐同步歌词正文。
    参数说明：
    - song_mid: QQ 音乐 songmid。
    - logger: 日志对象。
    返回值：
    - str: 命中时返回同步歌词正文，否则返回空字符串。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：仅返回带时间戳的同步歌词，不返回纯文本歌词。
    """
    lyric_payload = _fetch_qq_music_lyric(song_mid=song_mid, logger=logger)
    song_stub = {"songmid": song_mid, "id": "", "artist": "", "title": "", "duration_seconds": 0.0}
    normalized_candidate = _normalize_qq_music_candidate(song=song_stub, lyric_payload=lyric_payload)
    return str(normalized_candidate.get("synced_lyrics", "")).strip()


def fetch_qq_music_lyrics_bundle(
    *,
    song_mid: str = "",
    song_id: str = "",
    artist: str = "",
    title: str = "",
    logger,
) -> dict[str, str]:
    """
    功能说明：补拉 QQ 音乐原文、翻译、罗马音三份歌词正文。
    参数说明：
    - song_mid: QQ 音乐 songmid，可选。
    - song_id: QQ 音乐 songID，可选，musicu 接口优先依赖该值。
    - artist: 歌手名，用于 songID 缺失时回查。
    - title: 曲名，用于 songID 缺失时回查。
    - logger: 日志对象。
    返回值：
    - dict[str, str]: `synced_lyrics`、`word_timed_lyrics`、`translated_lyrics`、`romanized_lyrics`。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：优先保留旧 Web 接口同步歌词，musicu 负责补翻译和罗马音。
    """
    safe_song_id = str(song_id).strip()
    safe_song_mid = str(song_mid).strip()
    safe_artist = str(artist).strip()
    safe_title = str(title).strip()
    bundle = {
        "synced_lyrics": "",
        "word_timed_lyrics": "",
        "translated_lyrics": "",
        "romanized_lyrics": "",
    }
    if safe_song_mid:
        bundle["synced_lyrics"] = fetch_qq_music_synced_lyrics(song_mid=safe_song_mid, logger=logger)
    if not safe_song_id:
        resolved_song = _resolve_qq_music_song(song_mid=safe_song_mid, artist=safe_artist, title=safe_title, logger=logger)
        safe_song_id = str(resolved_song.get("id", "")).strip()
        if not safe_artist:
            safe_artist = str(resolved_song.get("artist", "")).strip()
        if not safe_title:
            safe_title = str(resolved_song.get("title", "")).strip()
    if not safe_song_id:
        return bundle
    musicu_payload = _fetch_qq_music_musicu_lyrics(
        song_id=safe_song_id,
        artist=safe_artist,
        title=safe_title,
        logger=logger,
    )
    enhanced_lyrics = _extract_qq_music_musicu_text(
        payload=musicu_payload,
        field_name="lyric",
        logger=logger,
        enhanced=True,
    )
    if enhanced_lyrics:
        bundle["word_timed_lyrics"] = enhanced_lyrics
    for source_key, target_key in [("lyric", "synced_lyrics"), ("trans", "translated_lyrics"), ("roma", "romanized_lyrics")]:
        normalized_text = _extract_qq_music_musicu_text(payload=musicu_payload, field_name=source_key, logger=logger)
        if normalized_text:
            bundle[target_key] = normalized_text
    return bundle


def _normalize_qq_music_candidate(*, song: dict[str, Any], lyric_payload: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：把 QQ 音乐歌曲与歌词响应归一化为内部候选结构。
    参数说明：
    - song: 轻量歌曲摘要。
    - lyric_payload: 原始歌词响应。
    返回值：
    - dict[str, Any]: 标准化结果。
    异常说明：无。
    边界条件：无歌词或无时间戳歌词时视为不可用候选。
    """
    retcode = _extract_qq_music_retcode(lyric_payload=lyric_payload)
    synced_lyrics = str(lyric_payload.get("lyric", "") or "").strip() if isinstance(lyric_payload, dict) else ""
    has_timestamps = _looks_like_synced_lyrics(synced_lyrics)
    if retcode == 0 and has_timestamps:
        status = "synced"
    elif retcode == 0 and synced_lyrics:
        status = "plain"
    else:
        status = "not_found"
    return {
        "status": status,
        "artist": str(song.get("artist", "")).strip(),
        "title": str(song.get("title", "")).strip(),
        "duration_seconds": float(song.get("duration_seconds", 0.0) or 0.0),
        "plain_lyrics": "",
        "synced_lyrics": synced_lyrics if has_timestamps else "",
        "provider": "qq_music",
        "provider_id": str(song.get("songmid", "")).strip(),
        "provider_song_id": str(song.get("id", "")).strip(),
        "instrumental": False,
        "error": "",
    }


def _extract_qq_music_retcode(lyric_payload: dict[str, Any]) -> int:
    """
    功能说明：从 QQ 音乐歌词响应中提取返回码。
    参数说明：
    - lyric_payload: 原始歌词响应。
    返回值：
    - int: 返回码；无可用字段时返回 -1。
    异常说明：字段不可转整数时回退 -1。
    边界条件：retcode 为 0 时必须保留为成功值，不能被假值逻辑吞掉。
    """
    if not isinstance(lyric_payload, dict):
        return -1
    if "retcode" in lyric_payload:
        try:
            return int(lyric_payload.get("retcode"))
        except Exception:  # noqa: BLE001
            return -1
    if "code" in lyric_payload:
        try:
            return int(lyric_payload.get("code"))
        except Exception:  # noqa: BLE001
            return -1
    return -1


def _looks_like_synced_lyrics(text: str) -> bool:
    """
    功能说明：判断文本是否像标准时间轴歌词。
    参数说明：
    - text: 歌词文本。
    返回值：
    - bool: 是否包含常见 LRC 时间戳。
    异常说明：无。
    边界条件：只做轻量判断，不验证每一行格式。
    """
    normalized_text = str(text).strip()
    if not normalized_text:
        return False
    return bool(re.search(r"\[\d{2}:\d{2}(?:\.\d{1,3})?\]", normalized_text))


def _resolve_qq_music_song(*, song_mid: str, artist: str, title: str, logger) -> dict[str, Any]:
    """
    功能说明：当缓存缺少 songID 时，通过 songmid 或歌手曲名重新定位歌曲。
    参数说明：
    - song_mid: QQ 音乐 songmid。
    - artist: 歌手名。
    - title: 曲名。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: 命中的轻量歌曲摘要；失败时返回空对象。
    异常说明：无。
    边界条件：优先 songmid 精确匹配，找不到再回退首个同名候选。
    """
    search_term = " ".join(part for part in [str(artist).strip(), str(title).strip()] if part).strip() or str(title).strip()
    if not search_term:
        return {}
    songs = _search_qq_music_songs(search_term=search_term, logger=logger, limit=10)
    normalized_song_mid = str(song_mid).strip().lower()
    for song in songs:
        if normalized_song_mid and str(song.get("songmid", "")).strip().lower() == normalized_song_mid:
            return song
    return songs[0] if songs else {}


def _fetch_qq_music_musicu_lyrics(*, song_id: str, artist: str, title: str, logger) -> dict[str, Any]:
    """
    功能说明：调用 QQ musicu 接口获取原文/翻译/罗马音 QRC 密文。
    参数说明：
    - song_id: QQ 音乐 songID。
    - artist: 歌手名。
    - title: 曲名。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: musicu 返回的歌词数据对象；异常时返回空对象。
    异常说明：网络或解析异常时返回空对象。
    边界条件：仅请求当前项目使用的 lyric/trans/roma 三类歌词。
    """
    session_payload = _qq_music_musicu_request(
        method="GetSession",
        module="music.getSession.session",
        param={"caller": 0, "uid": "0", "vkey": 0},
        comm={
            "ct": 11,
            "cv": "1003006",
            "v": "1003006",
            "os_ver": "15",
            "phonetype": "24122RKC7C",
            "rom": "Redmi/miro/miro:15/AE3A.240806.005/OS2.0.105.0.VOMCNXM:user/release-keys",
            "tmeAppID": "qqmusiclight",
            "nettype": "NETWORK_WIFI",
            "udid": "0",
        },
        logger=logger,
    )
    session_data = session_payload.get("session", {}) if isinstance(session_payload, dict) else {}
    if not isinstance(session_data, dict):
        return {}
    comm_payload = {
        "ct": 11,
        "cv": "1003006",
        "v": "1003006",
        "os_ver": "15",
        "phonetype": "24122RKC7C",
        "rom": "Redmi/miro/miro:15/AE3A.240806.005/OS2.0.105.0.VOMCNXM:user/release-keys",
        "tmeAppID": "qqmusiclight",
        "nettype": "NETWORK_WIFI",
        "udid": "0",
        "uid": session_data.get("uid", 0),
        "sid": session_data.get("sid", ""),
        "userip": session_data.get("userip", ""),
    }
    return _qq_music_musicu_request(
        method="GetPlayLyricInfo",
        module="music.musichallSong.PlayLyricInfo",
        param={
            "albumName": b64encode(b"").decode(),
            "crypt": 1,
            "ct": 19,
            "cv": 2111,
            "interval": 0,
            "lrc_t": 0,
            "qrc": 1,
            "qrc_t": 0,
            "roma": 1,
            "roma_t": 0,
            "singerName": b64encode(str(artist).encode("utf-8")).decode(),
            "songID": int(song_id),
            "songName": b64encode(str(title).encode("utf-8")).decode(),
            "trans": 1,
            "trans_t": 0,
            "type": 0,
        },
        comm=comm_payload,
        logger=logger,
    )


def _qq_music_musicu_request(*, method: str, module: str, param: dict[str, Any], comm: dict[str, Any], logger) -> dict[str, Any]:
    """
    功能说明：发送 QQ musicu 请求并抽取 request.data。
    参数说明：
    - method: musicu 方法名。
    - module: musicu 模块名。
    - param: 请求参数。
    - comm: 公共上下文参数。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: request.data 数据体；异常时返回空对象。
    异常说明：网络或解析异常时返回空对象。
    边界条件：仅处理单 request 结构。
    """
    payload = json.dumps(
        {
            "comm": comm,
            "request": {
                "method": method,
                "module": module,
                "param": param,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        url=QQ_MUSIC_MUSICU_API_URL,
        data=payload,
        headers={
            "Cookie": "tmeLoginType=-1;",
            "Content-Type": "application/json",
            "User-Agent": "okhttp/3.14.9",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=QQ_MUSIC_MUSICU_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        response_payload = json.loads(raw_body)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-QQ音乐 musicu 请求失败，method=%s，module=%s，错误=%s", method, module, error)
        return {}
    if not isinstance(response_payload, dict):
        return {}
    request_payload = response_payload.get("request", {}) if isinstance(response_payload.get("request", {}), dict) else {}
    top_level_code = response_payload.get("code", -1)
    request_code = request_payload.get("code", -1)
    try:
        normalized_top_level_code = int(top_level_code)
    except Exception:  # noqa: BLE001
        normalized_top_level_code = -1
    try:
        normalized_request_code = int(request_code)
    except Exception:  # noqa: BLE001
        normalized_request_code = -1
    if normalized_top_level_code != 0 or normalized_request_code != 0:
        logger.warning(
            "模块A V2-QQ音乐 musicu 返回异常，method=%s，module=%s，top_code=%s，request_code=%s",
            method,
            module,
            top_level_code,
            request_code,
        )
        return {}
    data_payload = request_payload.get("data", {})
    return data_payload if isinstance(data_payload, dict) else {}


def _extract_qq_music_musicu_text(*, payload: dict[str, Any], field_name: str, logger, enhanced: bool = False) -> str:
    """
    功能说明：从 musicu 响应中提取指定歌词字段并转成可展示文本。
    参数说明：
    - payload: musicu 数据体。
    - field_name: `lyric`/`trans`/`roma`。
    - logger: 日志对象。
    返回值：
    - str: 解析后的歌词文本；失败时返回空字符串。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：翻译和罗马音也保留行级时间戳，便于前端按时间戳对齐。
    """
    encrypted_text = str(payload.get(field_name, "") or "").strip()
    timestamp_field_name = "qrc_t" if field_name == "lyric" else f"{field_name}_t"
    if not encrypted_text or str(payload.get(timestamp_field_name, "0")) == "0":
        return ""
    try:
        qrc_text = decrypt_qq_music_qrc(encrypted_text)
        if field_name == "lyric":
            if enhanced:
                return extract_enhanced_lrc_from_qq_music_qrc(qrc_text)
            return extract_lrc_from_qq_music_qrc(qrc_text)
        # 翻译和罗马音也保留时间戳，使前端可按时间戳对齐而非索引偏移
        return extract_lrc_with_fallback_from_qq_music_qrc(qrc_text)
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-QQ音乐 %s QRC 解析失败，错误=%s", field_name, error)
        return ""


def _strip_trailing_bracket_note(text: str) -> str:
    """
    功能说明：去除标题尾部括注说明，减少翻译备注对搜索命中的干扰。
    参数说明：
    - text: 原始文本。
    返回值：
    - str: 清理后的文本。
    异常说明：无。
    边界条件：仅移除尾部括注，不改动中间正文。
    """
    normalized_text = str(text).strip()
    if not normalized_text:
        return ""
    return TRAILING_BRACKET_NOTE_PATTERN.sub("", normalized_text).strip()


def _dedupe_search_terms(items: list[str]) -> list[str]:
    """
    功能说明：对搜索词序列去重并裁剪空白项。
    参数说明：
    - items: 原始搜索词数组。
    返回值：
    - list[str]: 去重后的搜索词数组。
    异常说明：无。
    边界条件：大小写无关去重，保留首次出现顺序。
    """
    normalized_items: list[str] = []
    seen_items: set[str] = set()
    for item in items:
        normalized_item = str(item).strip()
        if not normalized_item:
            continue
        dedupe_key = normalized_item.lower()
        if dedupe_key in seen_items:
            continue
        seen_items.add(dedupe_key)
        normalized_items.append(normalized_item)
    return normalized_items
