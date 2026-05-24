"""
文件用途：管理模块A页面的联网歌词查找状态与手动启用歌词覆写。
核心流程：读写任务级状态文件，并在启用时将选中的同步歌词解析为句级单元。
输入输出：输入任务 artifacts 目录与状态内容，输出标准化状态字典或手动歌词结果。
依赖说明：依赖标准库 json/pathlib 与项目内 LRC 解析器。
维护说明：本文件只承载 Web 侧“手动联网歌词”状态，不修改主配置文件。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from music_video_pipeline.io_utils import write_json
from music_video_pipeline.modules.module_a_v2.lrc_parser import parse_synced_lyrics_to_sentence_units


MODULE_A_NETWORK_LYRICS_STATE_FILE_NAME = "module_a_network_lyrics_state.json"


def build_module_a_network_lyrics_state_path(artifacts_dir: Path) -> Path:
    """
    功能说明：返回模块A联网歌词状态文件路径。
    参数说明：
    - artifacts_dir: 任务 artifacts 根目录。
    返回值：
    - Path: 状态文件路径。
    异常说明：无。
    边界条件：仅拼路径，不创建文件。
    """
    return artifacts_dir / MODULE_A_NETWORK_LYRICS_STATE_FILE_NAME


def load_module_a_network_lyrics_state(artifacts_dir: Path) -> dict[str, Any]:
    """
    功能说明：读取模块A联网歌词状态文件并回退到默认结构。
    参数说明：
    - artifacts_dir: 任务 artifacts 根目录。
    返回值：
    - dict[str, Any]: 标准化状态字典。
    异常说明：无；文件缺失或损坏时回退默认值。
    边界条件：仅保留前端/执行链需要的最小字段。
    """
    state_path = build_module_a_network_lyrics_state_path(artifacts_dir=artifacts_dir)
    default_state = _build_default_state()
    if (not state_path.exists()) or (not state_path.is_file()):
        return default_state
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default_state
    if not isinstance(payload, dict):
        return default_state
    normalized_state = _build_default_state()
    normalized_state.update(
        {
            "display_status": str(payload.get("display_status", default_state["display_status"])).strip()
            or default_state["display_status"],
            "enabled": bool(payload.get("enabled", False)),
            "updated_at": str(payload.get("updated_at", "")).strip(),
            "last_search_at": str(payload.get("last_search_at", "")).strip(),
            "search_status": str(payload.get("search_status", "")).strip(),
            "lookup_error": str(payload.get("lookup_error", "")).strip(),
            "fingerprint_status": str(payload.get("fingerprint_status", "")).strip(),
            "acoustid_status": str(payload.get("acoustid_status", "")).strip(),
            "selected_candidate_id": str(payload.get("selected_candidate_id", "")).strip(),
        }
    )
    raw_metadata_trace = payload.get("metadata_trace", {})
    if isinstance(raw_metadata_trace, dict):
        normalized_state["metadata_trace"] = raw_metadata_trace
    raw_candidates = payload.get("candidates", [])
    if isinstance(raw_candidates, list):
        normalized_state["candidates"] = [item for item in raw_candidates if isinstance(item, dict)]
    raw_provider_groups = payload.get("provider_groups", [])
    if isinstance(raw_provider_groups, list):
        normalized_state["provider_groups"] = [item for item in raw_provider_groups if isinstance(item, dict)]
    raw_selected_candidate = payload.get("selected_candidate", {})
    if isinstance(raw_selected_candidate, dict):
        normalized_state["selected_candidate"] = raw_selected_candidate
    return normalized_state


def write_module_a_network_lyrics_state(artifacts_dir: Path, payload: dict[str, Any]) -> Path:
    """
    功能说明：写入模块A联网歌词状态文件。
    参数说明：
    - artifacts_dir: 任务 artifacts 根目录。
    - payload: 待写入状态字典。
    返回值：
    - Path: 实际写入路径。
    异常说明：写入失败时抛 OSError 或 TypeError。
    边界条件：自动创建父目录。
    """
    state_path = build_module_a_network_lyrics_state_path(artifacts_dir=artifacts_dir)
    write_json(state_path, payload)
    return state_path


def build_manual_network_lyrics_result(
    artifacts_dir: Path,
    audio_duration: float,
    logger,
) -> dict[str, Any] | None:
    """
    功能说明：读取已启用的联网歌词状态，并转换为模块A可直接使用的歌词结果。
    参数说明：
    - artifacts_dir: 任务 artifacts 根目录。
    - audio_duration: 当前音频时长（秒）。
    - logger: 日志对象。
    返回值：
    - dict[str, Any] | None: 命中已启用同步歌词时返回标准结果，否则返回 None。
    异常说明：无；非法状态统一回退 None。
    边界条件：仅支持已启用且携带 synced_lyrics 的候选。
    """
    state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
    if not bool(state.get("enabled", False)):
        return None
    selected_candidate = state.get("selected_candidate", {})
    if not isinstance(selected_candidate, dict):
        return None
    synced_lyrics = str(selected_candidate.get("synced_lyrics", "")).strip()
    word_timed_lyrics = str(selected_candidate.get("word_timed_lyrics", "")).strip()
    effective_synced_lyrics = word_timed_lyrics or synced_lyrics
    if not effective_synced_lyrics:
        return None
    lyric_sentence_units = parse_synced_lyrics_to_sentence_units(
        synced_lyrics=effective_synced_lyrics,
        audio_duration=audio_duration,
        logger=logger,
    )
    if not lyric_sentence_units:
        return None
    artist = str(selected_candidate.get("artist", "")).strip()
    title = str(selected_candidate.get("title", "")).strip()
    selected_provider = str(selected_candidate.get("provider", "lrclib")).strip() or "lrclib"
    manual_provider = f"{selected_provider}_manual"
    lrclib_result = {
        "status": "synced",
        "artist": artist,
        "title": title,
        "duration_seconds": float(selected_candidate.get("duration_seconds", audio_duration) or audio_duration),
        "plain_lyrics": str(selected_candidate.get("plain_lyrics", "")).strip(),
        "synced_lyrics": effective_synced_lyrics,
        "provider": selected_provider,
        "provider_id": str(selected_candidate.get("provider_id", "")).strip(),
        "instrumental": False,
        "error": "",
    }
    return {
        "provider": manual_provider,
        "reason": "manual_network_selected",
        "lyric_sentence_units": lyric_sentence_units,
        "sentence_split_stats": {
            "reason": "manual_network_selected",
            "sample_source": manual_provider,
            "sentence_count": len(lyric_sentence_units),
        },
        "funasr_raw_result": {"skipped": True, "reason": "manual_network_selected"},
        "lrclib_result": lrclib_result,
        "selected_candidate": selected_candidate,
    }


def current_time_text() -> str:
    """
    功能说明：返回当前北京时间（UTC+08:00）的 ISO 字符串。
    参数说明：无。
    返回值：
    - str: 当前时间文本。
    异常说明：无。
    边界条件：统一使用固定东八区。
    """
    china_tz = timezone(timedelta(hours=8))
    return datetime.now(china_tz).isoformat(timespec="seconds")


def _build_default_state() -> dict[str, Any]:
    """
    功能说明：构建模块A联网歌词状态的默认值。
    参数说明：无。
    返回值：
    - dict[str, Any]: 默认状态字典。
    异常说明：无。
    边界条件：默认不启用任何外部歌词。
    """
    return {
        "display_status": "idle",
        "enabled": False,
        "updated_at": "",
        "last_search_at": "",
        "search_status": "",
        "lookup_error": "",
        "fingerprint_status": "",
        "acoustid_status": "",
        "metadata_trace": {},
        "selected_candidate_id": "",
        "selected_candidate": {},
        "candidates": [],
        "provider_groups": [],
    }
