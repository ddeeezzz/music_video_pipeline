"""
文件用途：读取模块A V2歌词主链所需的音频内嵌元数据。
核心流程：使用 mutagen 读取 artist/title/album，并统一标准化输出。
输入输出：输入音频路径与时长，输出元数据标准化结果字典。
依赖说明：依赖 mutagen 读取音频标签。
维护说明：本文件只负责轻量元数据读取，不承担复杂清洗与编排职责。
"""

# 标准库：用于路径类型
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于读取音频元数据
from mutagen import File as MutagenFile


def read_embedded_metadata(audio_path: Path, duration_seconds: float, logger) -> dict[str, Any]:
    """
    功能说明：读取音频文件内嵌元数据并标准化。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - logger: 日志记录器。
    返回值：
    - dict[str, Any]: 标准化元数据结构。
    异常说明：异常在函数内吞并并转为 failed/missing 状态。
    边界条件：只要 artist/title 任一缺失，就不视为可用直查元数据。
    """
    default_payload = {
        "status": "missing",
        "artist": "",
        "title": "",
        "album": "",
        "duration_seconds": float(duration_seconds),
        "source": "embedded_tags",
        "error": "",
    }
    try:
        media_obj = MutagenFile(audio_path, easy=True)
        if media_obj is None:
            default_payload["error"] = "mutagen 返回空对象"
            return default_payload
        tags = getattr(media_obj, "tags", None)
        if tags is None:
            default_payload["error"] = "音频未携带可读取标签"
            return default_payload
        artist = _extract_first_tag_value(tags=tags, key_candidates=["artist", "albumartist"])
        title = _extract_first_tag_value(tags=tags, key_candidates=["title"])
        album = _extract_first_tag_value(tags=tags, key_candidates=["album"])
        payload = {
            "status": "ok" if artist and title else "missing",
            "artist": artist,
            "title": title,
            "album": album,
            "duration_seconds": float(duration_seconds),
            "source": "embedded_tags",
            "error": "",
        }
        if payload["status"] != "ok":
            payload["error"] = "artist/title 不完整"
        logger.info(
            "模块A V2-元数据读取完成，artist=%s，title=%s，status=%s",
            artist or "<empty>",
            title or "<empty>",
            payload["status"],
        )
        return payload
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-元数据读取失败，输入=%s，错误=%s", audio_path, error)
        default_payload["status"] = "failed"
        default_payload["error"] = str(error)
        return default_payload


def _extract_first_tag_value(tags: Any, key_candidates: list[str]) -> str:
    """
    功能说明：从 mutagen tags 中按候选键顺序提取首个非空字符串值。
    参数说明：
    - tags: mutagen 返回的标签对象。
    - key_candidates: 候选键名列表。
    返回值：
    - str: 提取到的标签文本，未命中时返回空字符串。
    异常说明：异常在函数内吞并。
    边界条件：列表值默认取第一项。
    """
    for key_name in key_candidates:
        try:
            raw_value = tags.get(key_name)
        except Exception:  # noqa: BLE001
            continue
        normalized = _normalize_tag_value(raw_value)
        if normalized:
            return normalized
    return ""


def _normalize_tag_value(raw_value: Any) -> str:
    """
    功能说明：将 mutagen 标签值归一化为单个字符串。
    参数说明：
    - raw_value: 原始标签值。
    返回值：
    - str: 清洗后的文本，空值返回空字符串。
    异常说明：无。
    边界条件：仅做首项提取与首尾空白清理，不做复杂清洗。
    """
    if raw_value is None:
        return ""
    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            return ""
        return str(raw_value[0]).strip()
    return str(raw_value).strip()
