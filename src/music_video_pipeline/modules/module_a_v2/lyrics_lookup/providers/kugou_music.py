"""
文件用途：提供酷狗音乐歌词搜索与KRC解析的最小封装。
核心流程：先搜歌曲摘要，再按歌曲信息搜歌词候选，最后下载并解析KRC为项目内部结构。
输入输出：输入查询词或歌手/歌名，输出标准化同步歌词候选列表。
依赖说明：依赖标准库 urllib/json/base64/zlib 等，不引入额外三方依赖。
维护说明：本文件只负责酷狗歌词搜索与解析，不承担上层优先级编排职责。
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zlib
from typing import Any, Callable


KUGOU_SONG_SEARCH_API_URL = "http://mobilecdnbj.kugou.com/api/v3/search/song"
KUGOU_LYRIC_SEARCH_API_URL = "https://lyrics.kugou.com/v1/search"
KUGOU_LYRIC_DOWNLOAD_API_URL = "http://lyrics.kugou.com/download"
KUGOU_REQUEST_TIMEOUT_SECONDS = 15.0
KUGOU_SIGN_SECRET = "LnT6xpN3khm36zse0QzvmgTZ3waWdRSA"
KUGOU_KRC_XOR_KEY = b"@Gaw^2tGQ61-\xce\xd2ni"
TRAILING_BRACKET_NOTE_PATTERN = re.compile(r"\s*[\(（\[【].*?[\)）\]】]\s*$")
KRC_TAG_PATTERN = re.compile(r"^\[(\w+):([^\]]*)\]$")
KRC_LINE_PATTERN = re.compile(r"^\[(?P<start>\d+),(?P<duration>\d+)\](?P<content>.*)$")
KRC_WORD_PATTERN = re.compile(r"(?:\[\d+,\d+\])?<(?P<start>\d+),(?P<duration>\d+),\d+>(?P<content>(?:.(?!\d+,\d+,\d+>))*)")


def search_kugou_music_candidates(
    *,
    query_text: str = "",
    artist: str = "",
    title: str = "",
    logger,
    limit: int = 10,
    candidate_callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> list[dict[str, Any]]:
    """
    功能说明：使用酷狗搜索歌曲并抓取可用同步歌词候选。
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
    search_terms = _build_kugou_search_terms(
        query_text=str(query_text).strip(),
        artist=str(artist).strip(),
        title=str(title).strip(),
    )
    if not search_terms:
        return []
    normalized_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for search_term in search_terms:
        songs = _search_kugou_songs(search_term=search_term, logger=logger, limit=max(10, safe_limit * 3))
        for song in songs:
            lyric_candidates = _search_kugou_lyrics(song=song, logger=logger, limit=3)
            for lyric_candidate in lyric_candidates:
                lyric_bundle = fetch_kugou_music_lyrics_bundle(
                    lyric_id=str(lyric_candidate.get("id", "")).strip(),
                    accesskey=str(lyric_candidate.get("accesskey", "")).strip(),
                    logger=logger,
                )
                normalized_candidate = _normalize_kugou_candidate(
                    song=song,
                    lyric_candidate=lyric_candidate,
                    lyric_bundle=lyric_bundle,
                )
                unique_key = (
                    str(normalized_candidate.get("provider_id", "")).strip().lower(),
                    str(normalized_candidate.get("artist", "")).strip().lower(),
                    str(normalized_candidate.get("title", "")).strip().lower(),
                )
                if unique_key in seen_keys:
                    continue
                if str(normalized_candidate.get("status", "")).strip().lower() != "synced":
                    continue
                if not str(normalized_candidate.get("synced_lyrics", "")).strip():
                    continue
                seen_keys.add(unique_key)
                normalized_candidates.append(normalized_candidate)
                if candidate_callback is not None:
                    candidate_callback(dict(normalized_candidate))
                if len(normalized_candidates) >= safe_limit:
                    logger.info(
                        "模块A V2-酷狗搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
                        str(artist).strip() or "<empty>",
                        str(title).strip() or "<empty>",
                        search_term,
                        len(normalized_candidates),
                    )
                    return normalized_candidates
    logger.info(
        "模块A V2-酷狗搜索完成，artist=%s，title=%s，query=%s，候选数=%s",
        str(artist).strip() or "<empty>",
        str(title).strip() or "<empty>",
        str(query_text).strip() or "<empty>",
        len(normalized_candidates),
    )
    return normalized_candidates


def fetch_kugou_music_synced_lyrics(*, lyric_id: str, accesskey: str, logger) -> str:
    """
    功能说明：按酷狗歌词ID与accesskey补拉同步歌词正文。
    参数说明：
    - lyric_id: 酷狗歌词ID。
    - accesskey: 酷狗歌词accesskey。
    - logger: 日志对象。
    返回值：
    - str: 命中时返回同步歌词正文，否则返回空字符串。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：仅返回带时间戳的同步歌词。
    """
    bundle = fetch_kugou_music_lyrics_bundle(lyric_id=lyric_id, accesskey=accesskey, logger=logger)
    return str(bundle.get("synced_lyrics", "")).strip()


def fetch_kugou_music_lyrics_bundle(*, lyric_id: str, accesskey: str, logger) -> dict[str, str]:
    """
    功能说明：补拉酷狗原文、逐字、翻译、罗马音歌词正文。
    参数说明：
    - lyric_id: 酷狗歌词ID。
    - accesskey: 酷狗歌词accesskey。
    - logger: 日志对象。
    返回值：
    - dict[str, str]: `synced_lyrics`、`word_timed_lyrics`、`translated_lyrics`、`romanized_lyrics`。
    异常说明：无；内部异常统一回退空字符串。
    边界条件：KRC缺失或解密失败时统一回退空字符串。
    """
    safe_lyric_id = str(lyric_id).strip()
    safe_accesskey = str(accesskey).strip()
    if not safe_lyric_id or not safe_accesskey:
        return {
            "synced_lyrics": "",
            "word_timed_lyrics": "",
            "translated_lyrics": "",
            "romanized_lyrics": "",
        }
    payload = _download_kugou_lyric(lyric_id=safe_lyric_id, accesskey=safe_accesskey, logger=logger)
    encrypted_content = str(payload.get("content", "") or "").strip() if isinstance(payload, dict) else ""
    content_type = int(payload.get("contenttype", 0) or 0) if isinstance(payload, dict) else 0
    if not encrypted_content:
        return {
            "synced_lyrics": "",
            "word_timed_lyrics": "",
            "translated_lyrics": "",
            "romanized_lyrics": "",
        }
    if content_type == 2:
        plain_text = _decode_kugou_plaintext_lyrics(encrypted_content)
        return {
            "synced_lyrics": plain_text if _looks_like_synced_lyrics(plain_text) else "",
            "word_timed_lyrics": "",
            "translated_lyrics": "",
            "romanized_lyrics": "",
        }
    krc_text = _decrypt_kugou_krc(encrypted_content, logger=logger)
    if not krc_text:
        return {
            "synced_lyrics": "",
            "word_timed_lyrics": "",
            "translated_lyrics": "",
            "romanized_lyrics": "",
        }
    return _parse_kugou_krc_bundle(krc_text)


def _build_kugou_search_terms(query_text: str, artist: str, title: str) -> list[str]:
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
            search_terms.append(f"{normalized_artist} - {normalized_title}".strip())
            search_terms.append(f"{normalized_artist} {normalized_title}".strip())
        search_terms.append(normalized_title)
        stripped_title = _strip_trailing_bracket_note(normalized_title)
        if stripped_title and stripped_title != normalized_title:
            if normalized_artist:
                search_terms.append(f"{normalized_artist} - {stripped_title}".strip())
                search_terms.append(f"{normalized_artist} {stripped_title}".strip())
            search_terms.append(stripped_title)
    return _dedupe_search_terms(search_terms)


def _search_kugou_songs(*, search_term: str, logger, limit: int) -> list[dict[str, Any]]:
    request_url = f"{KUGOU_SONG_SEARCH_API_URL}?{urlencode({'showtype': '14', 'highlight': '', 'pagesize': str(max(1, int(limit))), 'tag_aggr': '1', 'plat': '0', 'sver': '5', 'keyword': search_term, 'correct': '1', 'api_ver': '1', 'version': '9108', 'page': '1'})}"
    request = Request(
        url=request_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urlopen(request, timeout=KUGOU_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        response_payload = json.loads(raw_body)
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-酷狗搜歌失败，query=%s，错误=%s", search_term, error)
        return []
    except URLError as error:
        logger.warning("模块A V2-酷狗搜歌网络异常，query=%s，错误=%s", search_term, error)
        return []
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-酷狗搜歌解析失败，query=%s，错误=%s", search_term, error)
        return []
    song_items = response_payload.get("data", {}).get("info", []) if isinstance(response_payload, dict) else []
    if not isinstance(song_items, list):
        return []
    normalized_songs: list[dict[str, Any]] = []
    for item in song_items:
        if not isinstance(item, dict):
            continue
        normalized_songs.append(
            {
                "id": str(item.get("album_audio_id", "")).strip(),
                "title": _strip_trailing_bracket_note(str(item.get("songname", "")).strip()),
                "artist": str(item.get("singername", "")).strip(),
                "duration_seconds": float(item.get("duration", 0.0) or 0.0),
                "hash": str(item.get("hash", "")).strip(),
                "display_title": str(item.get("songname", "")).strip(),
            }
        )
    return normalized_songs


def _search_kugou_lyrics(*, song: dict[str, Any], logger, limit: int) -> list[dict[str, Any]]:
    params = {
        "album_audio_id": str(song.get("id", "")).strip(),
        "duration": str(int(round(float(song.get("duration_seconds", 0.0) or 0.0) * 1000.0))),
        "hash": str(song.get("hash", "")).strip(),
        "keyword": f"{str(song.get('artist', '')).strip()} - {str(song.get('display_title', song.get('title', ''))).strip()}".strip(),
        "lrctxt": "1",
        "man": "no",
    }
    payload = _kugou_signed_request(url=KUGOU_LYRIC_SEARCH_API_URL, params=params, module="Lyric", logger=logger)
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(candidates, list):
        return []
    normalized_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized_candidates.append(
            {
                "id": str(item.get("id", "")).strip(),
                "accesskey": str(item.get("accesskey", "")).strip(),
                "score": float(item.get("score", 0.0) or 0.0),
                "duration_seconds": float(item.get("duration", 0.0) or 0.0) / 1000.0,
            }
        )
        if len(normalized_candidates) >= max(1, int(limit)):
            break
    return normalized_candidates


def _download_kugou_lyric(*, lyric_id: str, accesskey: str, logger) -> dict[str, Any]:
    return _kugou_signed_request(
        url=KUGOU_LYRIC_DOWNLOAD_API_URL,
        params={
            "accesskey": accesskey,
            "charset": "utf8",
            "client": "mobi",
            "fmt": "krc",
            "id": lyric_id,
            "ver": "1",
        },
        module="Lyric",
        logger=logger,
    )


def _kugou_signed_request(*, url: str, params: dict[str, Any], module: str, logger) -> dict[str, Any]:
    mid = hashlib.md5(str(int(time.time() * 1000)).encode("utf-8")).hexdigest()
    safe_params = {
        "appid": "3116",
        "clientver": "11070",
        **params,
    }
    signature_source = (
        KUGOU_SIGN_SECRET
        + "".join(
            f"{key}={json.dumps(value) if isinstance(value, dict) else value}"
            for key, value in sorted(safe_params.items())
        )
        + KUGOU_SIGN_SECRET
    )
    safe_params["signature"] = hashlib.md5(signature_source.encode("utf-8")).hexdigest()
    request = Request(
        url=f"{url}?{urlencode(safe_params)}",
        headers={
            "User-Agent": f"Android14-1070-11070-201-0-{module}-wifi",
            "Connection": "Keep-Alive",
            "Accept-Encoding": "gzip, deflate",
            "KG-Rec": "1",
            "KG-RC": "1",
            "KG-CLIENTTIMEMS": str(int(time.time() * 1000)),
            "mid": mid,
        },
    )
    try:
        with urlopen(request, timeout=KUGOU_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw_body) if raw_body else {}
    except HTTPError as error:
        if int(getattr(error, "code", 0)) != 404:
            logger.warning("模块A V2-酷狗请求失败，url=%s，参数=%s，错误=%s", url, safe_params, error)
        return {}
    except URLError as error:
        logger.warning("模块A V2-酷狗请求网络异常，url=%s，参数=%s，错误=%s", url, safe_params, error)
        return {}
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-酷狗请求解析失败，url=%s，参数=%s，错误=%s", url, safe_params, error)
        return {}
    if not isinstance(payload, dict):
        return {}
    if int(payload.get("error_code", payload.get("status", 0)) or 0) not in {0, 200}:
        logger.warning("模块A V2-酷狗请求返回异常，url=%s，参数=%s，payload=%s", url, safe_params, payload)
        return {}
    return payload


def _decode_kugou_plaintext_lyrics(encoded_content: str) -> str:
    try:
        return base64.b64decode(str(encoded_content).strip()).decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""


def _decrypt_kugou_krc(encoded_content: str, logger) -> str:
    try:
        encrypted_bytes = base64.b64decode(str(encoded_content).strip())
        payload = encrypted_bytes[4:] if len(encrypted_bytes) >= 4 else b""
        decrypted_bytes = bytearray()
        for index, byte_value in enumerate(payload):
            decrypted_bytes.append(byte_value ^ KUGOU_KRC_XOR_KEY[index % len(KUGOU_KRC_XOR_KEY)])
        return zlib.decompress(bytes(decrypted_bytes)).decode("utf-8", errors="replace").strip()
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-酷狗KRC解密失败，错误=%s", error)
        return ""


def _parse_kugou_krc_bundle(krc_text: str) -> dict[str, str]:
    original_lines: list[dict[str, Any]] = []
    translation_lines: list[str] = []
    romanized_lines: list[str] = []
    tags: dict[str, str] = {}
    for raw_line in str(krc_text).splitlines():
        line = raw_line.strip()
        if not line.startswith("["):
            continue
        tag_match = KRC_TAG_PATTERN.match(line)
        if tag_match is not None:
            tags[str(tag_match.group(1)).strip()] = str(tag_match.group(2)).strip()
            continue
        line_match = KRC_LINE_PATTERN.match(line)
        if line_match is None:
            continue
        line_start_ms = int(line_match.group("start") or 0)
        line_duration_ms = int(line_match.group("duration") or 0)
        line_text = str(line_match.group("content") or "")
        words: list[dict[str, Any]] = []
        for word_match in KRC_WORD_PATTERN.finditer(line_text):
            word_text = str(word_match.group("content") or "")
            if not word_text:
                continue
            word_start_ms = line_start_ms + int(word_match.group("start") or 0)
            word_end_ms = word_start_ms + int(word_match.group("duration") or 0)
            words.append(
                {
                    "text": word_text,
                    "start_ms": word_start_ms,
                    "end_ms": word_end_ms,
                }
            )
        if not words:
            plain_text = re.sub(r"<\d+,\d+,\d+>", "", line_text).strip()
            if not plain_text:
                continue
            words = [{"text": plain_text, "start_ms": line_start_ms, "end_ms": line_start_ms + line_duration_ms}]
        original_lines.append(
            {
                "start_ms": line_start_ms,
                "end_ms": line_start_ms + line_duration_ms,
                "words": words,
            }
        )
    language_tag = str(tags.get("language", "")).strip()
    if language_tag:
        try:
            language_payload = json.loads(base64.b64decode(language_tag).decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            language_payload = {}
        language_items = language_payload.get("content", []) if isinstance(language_payload, dict) else []
        if isinstance(language_items, list):
            for item in language_items:
                if not isinstance(item, dict):
                    continue
                lyric_content = item.get("lyricContent", [])
                if not isinstance(lyric_content, list):
                    continue
                item_type = int(item.get("type", -1))
                if item_type == 1:
                    translation_lines = [
                        str(line_item[0]).strip()
                        for line_item in lyric_content
                        if isinstance(line_item, list) and line_item and str(line_item[0]).strip()
                    ]
                elif item_type == 0:
                    romanized_lines = [
                        "".join(str(token) for token in line_item if str(token))
                        for line_item in lyric_content
                        if isinstance(line_item, list)
                    ]
    synced_lines: list[str] = []
    word_timed_lines: list[str] = []
    translated_lines: list[str] = []
    romanized_output_lines: list[str] = []
    romanized_cursor = 0
    for line_index, original_line in enumerate(original_lines):
        line_start_ms = int(original_line.get("start_ms", 0) or 0)
        words = original_line.get("words", []) if isinstance(original_line.get("words", []), list) else []
        if not words:
            continue
        timestamp_text = f"[{_format_kugou_timestamp(line_start_ms)}]"
        plain_text = "".join(str(word.get("text", "")) for word in words).strip()
        if plain_text:
            synced_lines.append(f"{timestamp_text}{plain_text}")
        word_timed_lines.append(
            timestamp_text
            + "".join(
                f"<{_format_kugou_timestamp(int(word.get('start_ms', 0) or 0))}>{str(word.get('text', ''))}<"
                f"{_format_kugou_timestamp(int(word.get('end_ms', 0) or 0))}>"
                for word in words
                if str(word.get("text", ""))
            )
        )
        if line_index < len(translation_lines) and str(translation_lines[line_index]).strip():
            translated_lines.append(f"{timestamp_text}{str(translation_lines[line_index]).strip()}")
        if romanized_cursor < len(romanized_lines):
            current_romanized = str(romanized_lines[romanized_cursor]).strip()
            if current_romanized:
                romanized_output_lines.append(f"{timestamp_text}{current_romanized}")
            romanized_cursor += 1
    return {
        "synced_lyrics": "\n".join(synced_lines).strip(),
        "word_timed_lyrics": "\n".join(word_timed_lines).strip(),
        "translated_lyrics": "\n".join(translated_lines).strip(),
        "romanized_lyrics": "\n".join(romanized_output_lines).strip(),
    }


def _normalize_kugou_candidate(*, song: dict[str, Any], lyric_candidate: dict[str, Any], lyric_bundle: dict[str, str]) -> dict[str, Any]:
    synced_lyrics = str(lyric_bundle.get("synced_lyrics", "")).strip()
    has_timestamps = _looks_like_synced_lyrics(synced_lyrics)
    return {
        "status": "synced" if has_timestamps else "not_found",
        "artist": str(song.get("artist", "")).strip(),
        "title": str(song.get("title", "")).strip(),
        "duration_seconds": float(
            lyric_candidate.get("duration_seconds", song.get("duration_seconds", 0.0))
            or song.get("duration_seconds", 0.0)
            or 0.0
        ),
        "plain_lyrics": "",
        "synced_lyrics": synced_lyrics if has_timestamps else "",
        "word_timed_lyrics": str(lyric_bundle.get("word_timed_lyrics", "")).strip(),
        "translated_lyrics": str(lyric_bundle.get("translated_lyrics", "")).strip(),
        "romanized_lyrics": str(lyric_bundle.get("romanized_lyrics", "")).strip(),
        "provider": "kugou_music",
        "provider_id": str(lyric_candidate.get("id", "")).strip(),
        "provider_song_id": str(song.get("id", "")).strip(),
        "provider_accesskey": str(lyric_candidate.get("accesskey", "")).strip(),
        "provider_hash": str(song.get("hash", "")).strip(),
        "instrumental": False,
        "error": "",
        "score": float(lyric_candidate.get("score", 0.0) or 0.0),
    }


def _looks_like_synced_lyrics(text: str) -> bool:
    normalized_text = str(text).strip()
    if not normalized_text:
        return False
    return bool(re.search(r"\[\d{2}:\d{2}(?:\.\d{1,3})?\]", normalized_text))


def _format_kugou_timestamp(milliseconds: int) -> str:
    total_centiseconds = max(0, int(milliseconds)) // 10
    minutes = total_centiseconds // 6000
    seconds = (total_centiseconds % 6000) // 100
    centiseconds = total_centiseconds % 100
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _strip_trailing_bracket_note(text: str) -> str:
    normalized_text = str(text).strip()
    if not normalized_text:
        return ""
    return TRAILING_BRACKET_NOTE_PATTERN.sub("", normalized_text).strip()


def _dedupe_search_terms(items: list[str]) -> list[str]:
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
