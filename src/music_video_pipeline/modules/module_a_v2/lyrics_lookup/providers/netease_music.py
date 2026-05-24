"""
文件用途：提供网易云音乐歌词搜索的最小封装。
核心流程：先按歌曲关键词搜索歌曲，再按歌曲 ID 获取 LRC，并归一化为项目内部结构。
输入输出：输入查询词或歌手/歌名，输出标准化同步歌词候选列表。
依赖说明：依赖标准库 urllib/json 与少量文本规范化工具。
维护说明：本文件只负责网易云歌词搜索，不承担上层优先级编排职责。
"""

# 标准库：用于 JSON 解析
import json
# 标准库：用于正则清洗查询词
import re
# 标准库：用于 URL 编码与 HTTP 请求
from urllib.parse import urlencode
from urllib.request import Request, urlopen
# 标准库：用于 HTTP 异常识别
from urllib.error import HTTPError, URLError
# 标准库：用于类型提示
from typing import Any, Callable


# 常量：网易云歌曲搜索接口
NETEASE_SEARCH_API_URL = "https://music.163.com/api/search/get/web"
# 常量：网易云歌词接口
NETEASE_LYRIC_API_URL = "https://music.163.com/api/song/lyric"
# 常量：网易云 HTTP 超时时间（秒）
NETEASE_REQUEST_TIMEOUT_SECONDS = 15.0
# 常量：标题尾部括注清理规则
TRAILING_BRACKET_NOTE_PATTERN = re.compile(r"\s*[\(（\[【].*?[\)）\]】]\s*$")
NETEASE_YRC_LINE_PATTERN = re.compile(r"^\[(?P<start>\d+),(?P<duration>\d+)\](?P<content>.*)$")
NETEASE_YRC_WORD_PATTERN = re.compile(r"\((?P<start>\d+),(?P<duration>\d+),\d+\)(?P<content>[^\(]*)")


def search_netease_music_candidates(
    *,
    query_text: str = "",
    artist: str = "",
    title: str = "",
    logger,
    limit: int = 10,
    candidate_callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：使用网易云音乐搜索歌曲并抓取可用同步歌词候选。
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
    search_terms = _build_netease_search_terms(
        query_text=str(query_text).strip(),
        artist=str(artist).strip(),
        title=str(title).strip(),
    )
    if not search_terms:
        return []
    normalized_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for search_term in search_terms:
        songs = _search_netease_songs(search_term=search_term, logger=logger, limit=max(10, safe_limit * 3))
        for song in songs:
            song_id = str(song.get("id", "")).strip()
            normalized_artist = str(song.get("artist", "")).strip()
            normalized_title = str(song.get("title", "")).strip()
            unique_key = (song_id.lower(), normalized_artist.lower(), normalized_title.lower())
            if not song_id or unique_key in seen_keys:
                continue
            lyric_payload = _fetch_netease_lyric(song_id=song_id, logger=logger)
            normalized_candidate = _normalize_netease_candidate(
                song=song,
                lyric_payload=lyric_payload,
            )
            if normalized_candidate.get("status") != "synced":
                continue
            if not str(normalized_candidate.get("synced_lyrics", "")).strip():
                continue
            seen_keys.add(unique_key)
            normalized_candidates.append(normalized_candidate)
            if candidate_callback is not None:
                candidate_callback(dict(normalized_candidate))
            if len(normalized_candidates) >= safe_limit:
                logger.info(
                    "模块A V2-网易云搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
                    str(artist).strip() or "<empty>",
                    str(title).strip() or "<empty>",
                    search_term,
                    len(normalized_candidates),
                )
                return normalized_candidates
    logger.info(
        "模块A V2-网易云搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
        str(artist).strip() or "<empty>",
        str(title).strip() or "<empty>",
        str(query_text).strip() or "<empty>",
        len(normalized_candidates),
    )
    return normalized_candidates


def _build_netease_search_terms(query_text: str, artist: str, title: str) -> list[str]:
    """
    功能说明：构造网易云歌曲搜索词尝试序列。
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


def _search_netease_songs(*, search_term: str, logger, limit: int) -> list[dict[str, Any]]:
    """
    功能说明：按关键词检索网易云歌曲列表。
    参数说明：
    - search_term: 搜索词。
    - logger: 日志对象。
    - limit: 最多返回歌曲数。
    返回值：
    - list[dict[str, Any]]: 轻量歌曲摘要列表。
    异常说明：网络或解析异常时返回空数组。
    边界条件：仅抽取上层需要的最小字段。
    """
    payload = urlencode(
        {
            "s": str(search_term).strip(),
            "type": "1",
            "limit": str(max(1, int(limit))),
            "offset": "0",
        }
    ).encode("utf-8")
    request = Request(
        url=NETEASE_SEARCH_API_URL,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://music.163.com/",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=NETEASE_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        response_payload = json.loads(raw_body)
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-网易云搜索失败，query=%s，错误=%s", search_term, error)
        return []
    except URLError as error:
        logger.warning("模块A V2-网易云搜索网络异常，query=%s，错误=%s", search_term, error)
        return []
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-网易云搜索解析失败，query=%s，错误=%s", search_term, error)
        return []
    result_payload = response_payload.get("result", {}) if isinstance(response_payload, dict) else {}
    songs = result_payload.get("songs", []) if isinstance(result_payload, dict) else []
    if not isinstance(songs, list):
        return []
    normalized_songs: list[dict[str, Any]] = []
    for item in songs:
        if not isinstance(item, dict):
            continue
        song_id = str(item.get("id", "")).strip()
        title = str(item.get("name", "")).strip()
        artists = item.get("artists", [])
        normalized_artist = ""
        if isinstance(artists, list):
            normalized_artist = "/".join(
                str(artist_item.get("name", "")).strip()
                for artist_item in artists
                if isinstance(artist_item, dict) and str(artist_item.get("name", "")).strip()
            )
        normalized_songs.append(
            {
                "id": song_id,
                "title": title,
                "artist": normalized_artist,
                "duration_seconds": float(item.get("duration", 0.0) or 0.0) / 1000.0,
            }
        )
    return normalized_songs


def _fetch_netease_lyric(*, song_id: str, logger) -> dict[str, Any]:
    """
    功能说明：按歌曲 ID 获取网易云歌词响应。
    参数说明：
    - song_id: 歌曲 ID。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: 原始歌词响应；异常时返回空对象。
    异常说明：网络或解析异常时返回空对象。
    边界条件：同时请求 LRC/YRC/翻译/罗马音字段。
    """
    query_string = urlencode({"id": song_id, "lv": "-1", "kv": "-1", "tv": "-1", "rv": "-1", "yv": "-1"})
    request = Request(
        url=f"{NETEASE_LYRIC_API_URL}?{query_string}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://music.163.com/",
        },
    )
    try:
        with urlopen(request, timeout=NETEASE_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw_body)
        return payload if isinstance(payload, dict) else {}
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-网易云歌词获取失败，song_id=%s，错误=%s", song_id, error)
        return {}
    except URLError as error:
        logger.warning("模块A V2-网易云歌词获取网络异常，song_id=%s，错误=%s", song_id, error)
        return {}
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-网易云歌词获取解析失败，song_id=%s，错误=%s", song_id, error)
        return {}


def fetch_netease_music_synced_lyrics(*, song_id: str, logger) -> str:
    """
    功能说明：按歌曲ID补拉网易云同步歌词正文。
    参数说明：
    - song_id: 网易云歌曲ID。
    - logger: 日志对象。
    返回值：
    - str: 命中时返回同步歌词正文，否则返回空字符串。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：仅返回带时间戳的同步歌词，不返回纯文本歌词。
    """
    lyric_payload = _fetch_netease_lyric(song_id=song_id, logger=logger)
    song_stub = {"id": song_id, "artist": "", "title": "", "duration_seconds": 0.0}
    normalized_candidate = _normalize_netease_candidate(song=song_stub, lyric_payload=lyric_payload)
    return str(normalized_candidate.get("synced_lyrics", "")).strip()


def fetch_netease_music_lyrics_bundle(*, song_id: str, logger) -> dict[str, str]:
    """
    功能说明：补拉网易云原文、逐字、翻译、罗马音歌词正文。
    参数说明：
    - song_id: 网易云歌曲ID。
    - logger: 日志对象。
    返回值：
    - dict[str, str]: `synced_lyrics`、`word_timed_lyrics`、`translated_lyrics`、`romanized_lyrics`。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：优先使用 yrc/ytlrc/yromalrc，缺失时回退到 lrc/tlyric/romalrc。
    """
    lyric_payload = _fetch_netease_lyric(song_id=song_id, logger=logger)
    song_stub = {"id": song_id, "artist": "", "title": "", "duration_seconds": 0.0}
    normalized_candidate = _normalize_netease_candidate(song=song_stub, lyric_payload=lyric_payload)
    return {
        "synced_lyrics": str(normalized_candidate.get("synced_lyrics", "")).strip(),
        "word_timed_lyrics": str(normalized_candidate.get("word_timed_lyrics", "")).strip(),
        "translated_lyrics": str(normalized_candidate.get("translated_lyrics", "")).strip(),
        "romanized_lyrics": str(normalized_candidate.get("romanized_lyrics", "")).strip(),
    }


def _normalize_netease_candidate(*, song: dict[str, Any], lyric_payload: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：把网易云歌曲与歌词响应归一化为内部候选结构。
    参数说明：
    - song: 轻量歌曲摘要。
    - lyric_payload: 原始歌词响应。
    返回值：
    - dict[str, Any]: 标准化结果。
    异常说明：无。
    边界条件：纯音乐或无时间戳歌词时视为不可用候选。
    """
    lrc_payload = lyric_payload.get("lrc", {}) if isinstance(lyric_payload, dict) else {}
    yrc_payload = lyric_payload.get("yrc", {}) if isinstance(lyric_payload, dict) else {}
    tlyric_payload = lyric_payload.get("tlyric", {}) if isinstance(lyric_payload, dict) else {}
    ytlrc_payload = lyric_payload.get("ytlrc", {}) if isinstance(lyric_payload, dict) else {}
    romalrc_payload = lyric_payload.get("romalrc", {}) if isinstance(lyric_payload, dict) else {}
    yromalrc_payload = lyric_payload.get("yromalrc", {}) if isinstance(lyric_payload, dict) else {}
    synced_lyrics = ""
    if isinstance(lrc_payload, dict):
        synced_lyrics = str(lrc_payload.get("lyric", "") or "").strip()
    word_timed_lyrics = _build_netease_word_timed_lyrics(
        str(yrc_payload.get("lyric", "") or "").strip() if isinstance(yrc_payload, dict) else ""
    )
    translated_lyrics = _pick_first_nonempty_lyrics(
        [
            str(ytlrc_payload.get("lyric", "") or "").strip() if isinstance(ytlrc_payload, dict) else "",
            str(tlyric_payload.get("lyric", "") or "").strip() if isinstance(tlyric_payload, dict) else "",
        ]
    )
    romanized_lyrics = _pick_first_nonempty_lyrics(
        [
            str(yromalrc_payload.get("lyric", "") or "").strip() if isinstance(yromalrc_payload, dict) else "",
            str(romalrc_payload.get("lyric", "") or "").strip() if isinstance(romalrc_payload, dict) else "",
        ]
    )
    pure_music = bool(lyric_payload.get("pureMusic", False))
    no_lyric = bool(lyric_payload.get("nolyric", False))
    has_timestamps = _looks_like_synced_lyrics(synced_lyrics)
    if pure_music or no_lyric:
        status = "instrumental" if pure_music else "not_found"
    elif has_timestamps:
        status = "synced"
    elif synced_lyrics:
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
        "word_timed_lyrics": word_timed_lyrics,
        "translated_lyrics": translated_lyrics,
        "romanized_lyrics": romanized_lyrics,
        "provider": "netease_music",
        "provider_id": str(song.get("id", "")).strip(),
        "instrumental": pure_music,
        "error": "",
    }


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


def _build_netease_word_timed_lyrics(yrc_text: str) -> str:
    """
    功能说明：把网易云 YRC 转成增强 LRC，供后续词级时间解析。
    参数说明：
    - yrc_text: 网易云 yrc.lyric 原文。
    返回值：
    - str: 含 `<start>词<end>` 标记的增强 LRC。
    异常说明：无；输入非法时回退空字符串。
    边界条件：行级仍保留 `[mm:ss.xx]`，方便沿用现有 LRC 解析链。
    """
    normalized_yrc_text = str(yrc_text).strip()
    if not normalized_yrc_text:
        return ""
    output_lines: list[str] = []
    for raw_line in normalized_yrc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_match = NETEASE_YRC_LINE_PATTERN.match(line)
        if line_match is None:
            continue
        line_start_ms = int(line_match.group("start") or 0)
        line_content = str(line_match.group("content") or "")
        enhanced_parts: list[str] = [f"[{_format_netease_timestamp(line_start_ms)}]"]
        for word_match in NETEASE_YRC_WORD_PATTERN.finditer(line_content):
            word_text = str(word_match.group("content") or "")
            word_start_ms = int(word_match.group("start") or 0)
            word_end_ms = word_start_ms + int(word_match.group("duration") or 0)
            if not word_text:
                continue
            enhanced_parts.append(f"<{_format_netease_timestamp(word_start_ms)}>{word_text}<{_format_netease_timestamp(word_end_ms)}>")
        if len(enhanced_parts) == 1:
            plain_line = re.sub(r"\(\d+,\d+,\d+\)", "", line_content).strip()
            if plain_line:
                enhanced_parts.append(plain_line)
        output_lines.append("".join(enhanced_parts))
    return "\n".join(output_lines).strip()


def _format_netease_timestamp(milliseconds: int) -> str:
    """
    功能说明：把毫秒时间转成增强 LRC 使用的 `MM:SS.xx` 文本。
    参数说明：
    - milliseconds: 毫秒值。
    返回值：
    - str: 时间戳文本。
    异常说明：无。
    边界条件：保留到厘秒精度。
    """
    total_centiseconds = max(0, int(milliseconds)) // 10
    minutes = total_centiseconds // 6000
    seconds = (total_centiseconds % 6000) // 100
    centiseconds = total_centiseconds % 100
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _pick_first_nonempty_lyrics(candidates: list[str]) -> str:
    """
    功能说明：从多种歌词字段候选中取第一个非空值。
    参数说明：
    - candidates: 候选文本列表。
    返回值：
    - str: 首个非空文本，未命中时返回空字符串。
    异常说明：无。
    边界条件：逐项 strip 后判断非空。
    """
    for item in candidates:
        normalized_item = str(item).strip()
        if normalized_item:
            return normalized_item
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
