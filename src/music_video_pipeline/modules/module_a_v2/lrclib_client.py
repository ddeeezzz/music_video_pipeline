"""
文件用途：提供模块A V2对 LRCLIB 的最小查询与标准化能力。
核心流程：使用 artist/title/duration 发起严格匹配查询，并归一化响应字段。
输入输出：输入查询键，输出标准化 LRCLIB 结果字典。
依赖说明：依赖标准库 urllib/json 完成 HTTP 调用。
维护说明：本文件只负责歌词服务查询，不承担上层编排职责。
"""

# 标准库：用于 JSON 解析
import json
# 标准库：用于 URL 编码与 HTTP 请求
from urllib.parse import urlencode
from urllib.request import Request, urlopen
# 标准库：用于 HTTP 异常识别
from urllib.error import HTTPError, URLError
# 标准库：用于类型提示
from typing import Any


# 常量：LRCLIB 查询接口地址
LRCLIB_GET_API_URL = "https://lrclib.net/api/get"
# 常量：LRCLIB HTTP 超时时间（秒）
LRCLIB_REQUEST_TIMEOUT_SECONDS = 15.0


def query_lrclib_lyrics(artist: str, title: str, duration_seconds: float, logger) -> dict[str, Any]:
    """
    功能说明：查询 LRCLIB 并标准化返回结果。
    参数说明：
    - artist: 艺术家名称。
    - title: 曲名。
    - duration_seconds: 音频时长（秒）。
    - logger: 日志记录器。
    返回值：
    - dict[str, Any]: 标准化 LRCLIB 结果。
    异常说明：网络或解析异常在函数内吞并并转为 failed/not_found。
    边界条件：当响应明确标记 instrumental 时，不再要求歌词字段存在。
    """
    query_params = {
        "artist_name": str(artist).strip(),
        "track_name": str(title).strip(),
        "duration": str(int(round(float(duration_seconds)))),
    }
    default_payload = {
        "status": "failed",
        "artist": str(artist).strip(),
        "title": str(title).strip(),
        "duration_seconds": float(duration_seconds),
        "plain_lyrics": "",
        "synced_lyrics": "",
        "provider": "lrclib",
        "provider_id": "",
        "instrumental": False,
        "error": "",
    }
    try:
        query_string = urlencode(query_params)
        request = Request(
            url=f"{LRCLIB_GET_API_URL}?{query_string}",
            headers={"User-Agent": "music-video-pipeline/1.0"},
        )
        with urlopen(request, timeout=LRCLIB_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw_body)
        normalized = _normalize_lrclib_payload(
            payload=payload,
            fallback_artist=str(artist).strip(),
            fallback_title=str(title).strip(),
            fallback_duration_seconds=float(duration_seconds),
        )
        logger.info(
            "模块A V2-LRCLIB查询完成，artist=%s，title=%s，status=%s，instrumental=%s",
            normalized["artist"] or "<empty>",
            normalized["title"] or "<empty>",
            normalized["status"],
            normalized["instrumental"],
        )
        return normalized
    except HTTPError as error:
        if int(getattr(error, "code", 0)) == 404:
            default_payload["status"] = "not_found"
            default_payload["error"] = "lrclib 404 not found"
            return default_payload
        default_payload["error"] = f"http_error:{error}"
        logger.warning("模块A V2-LRCLIB查询失败，artist=%s，title=%s，错误=%s", artist, title, error)
        return default_payload
    except URLError as error:
        default_payload["error"] = f"url_error:{error}"
        logger.warning("模块A V2-LRCLIB网络异常，artist=%s，title=%s，错误=%s", artist, title, error)
        return default_payload
    except Exception as error:  # noqa: BLE001
        default_payload["error"] = str(error)
        logger.warning("模块A V2-LRCLIB解析失败，artist=%s，title=%s，错误=%s", artist, title, error)
        return default_payload


def _normalize_lrclib_payload(
    payload: Any,
    fallback_artist: str,
    fallback_title: str,
    fallback_duration_seconds: float,
) -> dict[str, Any]:
    """
    功能说明：将 LRCLIB 原始响应归一化为内部最小结构。
    参数说明：
    - payload: LRCLIB 原始响应对象。
    - fallback_artist: 查询输入艺人名。
    - fallback_title: 查询输入曲名。
    - fallback_duration_seconds: 查询输入时长。
    返回值：
    - dict[str, Any]: 标准化结果。
    异常说明：无。
    边界条件：字段缺失时尽量回填查询输入，确保日志可读。
    """
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "artist": fallback_artist,
            "title": fallback_title,
            "duration_seconds": float(fallback_duration_seconds),
            "plain_lyrics": "",
            "synced_lyrics": "",
            "provider": "lrclib",
            "provider_id": "",
            "instrumental": False,
            "error": "payload_not_dict",
        }

    instrumental = bool(payload.get("instrumental", False))
    plain_lyrics = str(payload.get("plainLyrics", payload.get("plain_lyrics", "")) or "").strip()
    synced_lyrics = str(payload.get("syncedLyrics", payload.get("synced_lyrics", "")) or "").strip()
    if instrumental:
        status = "instrumental"
    elif synced_lyrics:
        status = "synced"
    elif plain_lyrics:
        status = "plain"
    else:
        status = "not_found"
    return {
        "status": status,
        "artist": str(payload.get("artistName", payload.get("artist", fallback_artist)) or fallback_artist).strip(),
        "title": str(payload.get("trackName", payload.get("title", fallback_title)) or fallback_title).strip(),
        "duration_seconds": float(payload.get("duration", fallback_duration_seconds) or fallback_duration_seconds),
        "plain_lyrics": plain_lyrics,
        "synced_lyrics": synced_lyrics,
        "provider": "lrclib",
        "provider_id": str(payload.get("id", "") or "").strip(),
        "instrumental": instrumental,
        "error": "",
    }
