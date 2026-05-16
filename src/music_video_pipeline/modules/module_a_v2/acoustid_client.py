"""
文件用途：提供模块A V2 的最小音频指纹与 AcoustID 查询能力。
核心流程：调用 fpcalc 生成指纹，再调用 AcoustID lookup 接口获取歌曲元数据候选。
输入输出：输入音频路径与 API Key 配置，输出标准化指纹结果与匹配结果。
依赖说明：依赖标准库 subprocess、urllib 与 json。
维护说明：本文件只负责识别补充链，不承担歌词查询与上层编排职责。
"""

# 标准库：用于 JSON 解析
import json
# 标准库：用于子进程调用
import subprocess
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any
# 标准库：用于 URL 编码与 HTTP 请求
from urllib.parse import urlencode
from urllib.request import Request, urlopen
# 标准库：用于 HTTP 异常识别
from urllib.error import HTTPError, URLError


# 常量：AcoustID lookup 接口地址
ACOUSTID_LOOKUP_API_URL = "https://api.acoustid.org/v2/lookup"
# 常量：AcoustID 查询超时时间（秒）
ACOUSTID_LOOKUP_TIMEOUT_SECONDS = 20.0
# 常量：项目根目录，用于解析相对 API Key 路径
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[4]


def build_fingerprint_result(audio_path: Path, duration_seconds: float, fpcalc_bin: str, logger) -> dict[str, Any]:
    """
    功能说明：调用 fpcalc 生成音频指纹并标准化。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - fpcalc_bin: fpcalc 可执行命令或绝对路径。
    - logger: 日志记录器。
    返回值：
    - dict[str, Any]: 标准化指纹结果。
    异常说明：异常在函数内吞并并转为 failed 结果。
    边界条件：当 fpcalc 缺失或输出格式异常时直接失败，不抛出到上层。
    """
    payload = {
        "fingerprint": "",
        "duration_seconds": float(duration_seconds),
        "fingerprint_engine": "chromaprint",
        "status": "failed",
        "error": "",
    }
    try:
        result = subprocess.run(
            [str(fpcalc_bin).strip() or "fpcalc", "-json", str(audio_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        raw_payload = json.loads(result.stdout)
        fingerprint = str(raw_payload.get("fingerprint", "")).strip()
        resolved_duration = float(raw_payload.get("duration", duration_seconds) or duration_seconds)
        if not fingerprint:
            payload["error"] = "fpcalc 返回空 fingerprint"
            return payload
        payload["fingerprint"] = fingerprint
        payload["duration_seconds"] = resolved_duration
        payload["status"] = "ok"
        return payload
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-AcoustID指纹生成失败，输入=%s，错误=%s", audio_path, error)
        payload["error"] = str(error)
        return payload


def query_acoustid_match(
    fingerprint_result: dict[str, Any],
    acoustid_api_key_file: str,
    logger,
) -> dict[str, Any]:
    """
    功能说明：使用指纹结果查询 AcoustID，并返回最小标准化歌曲匹配结果。
    参数说明：
    - fingerprint_result: 指纹结果标准化字典。
    - acoustid_api_key_file: AcoustID API Key 文件路径。
    - logger: 日志记录器。
    返回值：
    - dict[str, Any]: 标准化歌曲匹配结果。
    异常说明：异常在函数内吞并并转为 failed/no_match 结果。
    边界条件：当 API Key 缺失时直接失败，不进行网络调用。
    """
    payload = {
        "status": "failed",
        "artist": "",
        "title": "",
        "duration_seconds": float(fingerprint_result.get("duration_seconds", 0.0) or 0.0),
        "score": 0.0,
        "acoustid_id": "",
        "recording_id": "",
        "raw_candidates": [],
        "error": "",
    }
    if str(fingerprint_result.get("status", "")).strip().lower() != "ok":
        payload["error"] = "fingerprint_not_ready"
        return payload
    api_key = _read_acoustid_api_key(acoustid_api_key_file=acoustid_api_key_file)
    if not api_key:
        payload["error"] = "missing_acoustid_api_key"
        return payload
    try:
        query_params = {
            "client": api_key,
            "duration": str(int(round(float(fingerprint_result.get("duration_seconds", 0.0) or 0.0)))),
            "fingerprint": str(fingerprint_result.get("fingerprint", "")),
            "meta": "recordings releasegroups",
        }
        request = Request(
            url=f"{ACOUSTID_LOOKUP_API_URL}?{urlencode(query_params)}",
            headers={"User-Agent": "music-video-pipeline/1.0"},
        )
        with urlopen(request, timeout=ACOUSTID_LOOKUP_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_body = response.read().decode("utf-8", errors="replace")
        body = json.loads(raw_body)
        results = body.get("results", [])
        if not isinstance(results, list) or not results:
            payload["status"] = "no_match"
            return payload
        payload["raw_candidates"] = results
        best_match = _pick_best_acoustid_candidate(results=results)
        if best_match is None:
            payload["status"] = "no_match"
            return payload
        return {
            "status": "ok",
            "artist": str(best_match.get("artist", "")).strip(),
            "title": str(best_match.get("title", "")).strip(),
            "duration_seconds": float(fingerprint_result.get("duration_seconds", 0.0) or 0.0),
            "score": float(best_match.get("score", 0.0) or 0.0),
            "acoustid_id": str(best_match.get("acoustid_id", "")).strip(),
            "recording_id": str(best_match.get("recording_id", "")).strip(),
            "raw_candidates": results,
            "error": "",
        }
    except HTTPError as error:
        payload["error"] = f"http_error:{error}"
        logger.warning("模块A V2-AcoustID查询失败，错误=%s", error)
        return payload
    except URLError as error:
        payload["error"] = f"url_error:{error}"
        logger.warning("模块A V2-AcoustID网络异常，错误=%s", error)
        return payload
    except Exception as error:  # noqa: BLE001
        payload["error"] = str(error)
        logger.warning("模块A V2-AcoustID解析失败，错误=%s", error)
        return payload


def _pick_best_acoustid_candidate(results: list[Any]) -> dict[str, Any] | None:
    """
    功能说明：从 AcoustID 结果中挑选最小可用的最佳候选。
    参数说明：
    - results: AcoustID 原始 results 列表。
    返回值：
    - dict[str, Any] | None: 标准化最佳候选，未命中时返回 None。
    异常说明：无。
    边界条件：只取第一个具备 recording/title 的候选，不做复杂排序系统。
    """
    best_candidate: dict[str, Any] | None = None
    best_score = -1.0
    for result_item in results:
        if not isinstance(result_item, dict):
            continue
        recordings = result_item.get("recordings", [])
        if not isinstance(recordings, list) or not recordings:
            continue
        recording_item = recordings[0]
        if not isinstance(recording_item, dict):
            continue
        title = str(recording_item.get("title", "")).strip()
        artist = _extract_acoustid_artist(recording_item=recording_item)
        if not title or not artist:
            continue
        score = float(result_item.get("score", 0.0) or 0.0)
        if score <= best_score:
            continue
        best_score = score
        best_candidate = {
            "artist": artist,
            "title": title,
            "score": score,
            "acoustid_id": str(result_item.get("id", "")).strip(),
            "recording_id": str(recording_item.get("id", "")).strip(),
        }
    return best_candidate


def _extract_acoustid_artist(recording_item: dict[str, Any]) -> str:
    """
    功能说明：从 AcoustID recording 节点提取艺人名。
    参数说明：
    - recording_item: 单个 recording 节点。
    返回值：
    - str: 艺人名，未命中时返回空字符串。
    异常说明：无。
    边界条件：只取首个 artist 的 name 字段。
    """
    artists = recording_item.get("artists", [])
    if not isinstance(artists, list) or not artists:
        return ""
    first_artist = artists[0]
    if not isinstance(first_artist, dict):
        return ""
    return str(first_artist.get("name", "")).strip()


def _read_acoustid_api_key(acoustid_api_key_file: str) -> str:
    """
    功能说明：读取 AcoustID API Key 文件。
    参数说明：
    - acoustid_api_key_file: API Key 文件路径，支持相对项目根路径。
    返回值：
    - str: API Key 文本，缺失或为空时返回空字符串。
    异常说明：异常在函数内吞并。
    边界条件：不自动创建文件。
    """
    try:
        key_path = Path(str(acoustid_api_key_file).strip()).expanduser()
        if not key_path.is_absolute():
            key_path = PROJECT_ROOT_DIR / key_path
        if not key_path.exists():
            return ""
        return key_path.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return ""
