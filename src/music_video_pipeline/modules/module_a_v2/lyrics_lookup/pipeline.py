"""
文件用途：提供模块A页面的联网歌词候选查询能力。
核心流程：优先按元信息模糊搜索 -> 再按音频指纹补充 -> 仍未命中时支持手动歌曲名搜索。
输入输出：输入音频路径、联网配置与可选手动查询词，输出候选列表与查询摘要。
依赖说明：依赖模块A V2 既有元数据读取、AcoustID 与 LRCLIB 客户端。
维护说明：本文件只服务于 Web 手动选词入口，不改主链自动歌词优先级。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, signature
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable

from music_video_pipeline.modules.module_a_v2.acoustid_client import build_fingerprint_result, query_acoustid_match
from music_video_pipeline.modules.module_a_v2.lrclib_client import query_lrclib_lyrics, search_lrclib_candidates
from music_video_pipeline.modules.module_a_v2.metadata_reader import read_embedded_metadata
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.kugou_music import search_kugou_music_candidates
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.netease_music import search_netease_music_candidates
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music import search_qq_music_candidates
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.syncedlyrics import (
    SYNCEDLYRICS_PROVIDER_NAMES,
    search_syncedlyrics_candidates,
    search_syncedlyrics_candidates_by_provider,
)
PROVIDER_DISPLAY_NAMES = {
    "netease_music": "网易云音乐",
    "qq_music": "QQ音乐",
    "kugou_music": "酷狗音乐",
    "lrclib": "LRCLIB",
    "syncedlyrics": "syncedlyrics",
}
STREAM_PROVIDER_PAGE_SIZE = 10
STREAM_PROVIDER_PREFETCH_LIMIT = 30
NON_WORD_NORMALIZE_PATTERN = re.compile(r"[\s\-_/\|:：·,，.。!！?？'\"]+")
TRAILING_BRACKET_NOTE_PATTERN = re.compile(r"\s*[\(（\[【].*?[\)）\]】]\s*$")


def stream_synced_lrc_candidates(
    *,
    audio_path: Path,
    duration_seconds: float,
    fpcalc_bin: str,
    acoustid_api_key_file: str,
    logger,
    manual_query: str = "",
    emit_event: Callable[[str, dict[str, Any]], None] | None = None,
    max_candidates: int = STREAM_PROVIDER_PREFETCH_LIMIT,
    raw_candidate_limit: int = 30,
    split_syncedlyrics_providers: bool = False,
) -> dict[str, Any]:
    """
    功能说明：以事件流方式执行模块A歌词检索，并在阶段推进与来源返回时实时发出更新。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - fpcalc_bin: 指纹命令路径或命令名。
    - acoustid_api_key_file: AcoustID API Key 文件路径。
    - logger: 日志对象。
    - manual_query: 可选的手动歌曲名搜索词。
    - emit_event: 事件回调，接收 `(event_name, payload)`。
    - max_candidates: 每个来源预抓取上限。
    - raw_candidate_limit: 指纹原始候选上限。
    返回值：
    - dict[str, Any]: 最终搜索结果。
    异常说明：异常内部转为 failed 结果，不向上抛出。
    边界条件：provider_groups 内保留每个来源的完整预抓取结果，前端可自行分页展示。
    """
    started_at = time.perf_counter()

    def _emit(event_name: str, payload: dict[str, Any]) -> None:
        if emit_event is None:
            return
        emit_event(event_name, payload)

    def _elapsed_ms() -> int:
        return int(round((time.perf_counter() - started_at) * 1000.0))

    def _emit_stage(stage_key: str, status: str, message: str, **extra: Any) -> None:
        stage_payload = {
            "stage_key": stage_key,
            "status": status,
            "message": str(message).strip(),
            "elapsed_ms": _elapsed_ms(),
        }
        if extra:
            stage_payload.update(extra)
        _emit("stage", stage_payload)

    normalized_manual_query = str(manual_query).strip()
    _emit(
        "search_started",
        {
            "search_mode": "manual_query" if normalized_manual_query else "automatic",
            "manual_query": normalized_manual_query,
            "page_size": STREAM_PROVIDER_PAGE_SIZE,
            "prefetch_limit": max(1, int(max_candidates)),
        },
    )
    if normalized_manual_query:
        _emit_stage("manual_query", "running", "正在按歌曲名搜索多个来源")
        manual_artist_hint, manual_title_hint = _split_manual_artist_title(manual_query=normalized_manual_query)
        manual_metadata_trace = _build_metadata_trace(
            metadata_result={"status": "skipped", "source": "manual_query"},
            fingerprint_result={"status": "skipped", "reason": "manual_query"},
            acoustid_result={"status": "skipped", "reason": "manual_query"},
        )
        manual_provider_groups = _search_manual_network_candidates(
            manual_query=normalized_manual_query,
            duration_seconds=duration_seconds,
            logger=logger,
            limit=max(1, int(max_candidates)),
            progress_callback=lambda provider_group: _emit("provider_group", provider_group),
            started_at=started_at,
            split_syncedlyrics_providers=split_syncedlyrics_providers,
        )
        manual_provider_groups = _sort_provider_groups_candidates_by_similarity(
            provider_groups=manual_provider_groups,
            preferred_artist=manual_artist_hint,
            preferred_title=manual_title_hint,
        )
        manual_candidates = _flatten_provider_group_candidates(manual_provider_groups)
        _emit_stage(
            "manual_query",
            "completed",
            f"手动搜索完成，找到 {len(manual_candidates)} 个候选",
            candidate_count=len(manual_candidates),
        )
        final_result = {
            "status": "ok" if manual_candidates else "not_found",
            "error": "" if manual_candidates else "按歌曲名未找到可用的同步LRC歌词候选",
            "message": (
                f"已按歌曲名找到 {len(manual_candidates)} 个同步lrc歌词候选"
                if manual_candidates
                else "按歌曲名未找到可用的同步lrc歌词候选"
            ),
            "search_mode": "manual_query",
            "suggest_manual_query": False,
            "metadata_result": {"status": "skipped", "reason": "manual_query"},
            "fingerprint_result": {"status": "skipped", "reason": "manual_query"},
            "acoustid_result": {"status": "skipped", "reason": "manual_query"},
            "metadata_trace": manual_metadata_trace,
            "provider_groups": manual_provider_groups,
            "candidates": manual_candidates,
            "searched_candidate_count": len(manual_candidates),
        }
        _emit("complete", final_result)
        return final_result

    safe_max_candidates = max(1, int(max_candidates))
    safe_raw_candidate_limit = max(safe_max_candidates, int(raw_candidate_limit))
    _emit_stage("metadata_read", "running", "正在读取音频元信息")
    metadata_result = read_embedded_metadata(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        logger=logger,
    )
    _emit_stage(
        "metadata_read",
        "completed",
        "音频元信息读取完成",
        artist=str(metadata_result.get("artist", "")).strip(),
        title=str(metadata_result.get("title", "")).strip(),
        metadata_status=str(metadata_result.get("status", "")).strip(),
    )
    metadata_artist = str(metadata_result.get("artist", "")).strip()
    metadata_title = str(metadata_result.get("title", "")).strip()
    if metadata_artist or metadata_title:
        _emit_stage("metadata_search", "running", "正在按元信息搜索多个来源")
        metadata_duration_seconds = float(metadata_result.get("duration_seconds", duration_seconds) or duration_seconds)
        metadata_queries = _build_metadata_search_queries(artist=metadata_artist, title=metadata_title)
        metadata_attempts = _build_query_text_search_attempts(metadata_queries)
        metadata_provider_groups = _search_provider_groups(
            query_attempts=metadata_attempts,
            duration_seconds=metadata_duration_seconds,
            logger=logger,
            limit=safe_max_candidates,
            progress_callback=lambda provider_group: _emit("provider_group", provider_group),
            started_at=started_at,
            split_syncedlyrics_providers=split_syncedlyrics_providers,
        )
        metadata_provider_groups = _sort_provider_groups_candidates_by_similarity(
            provider_groups=metadata_provider_groups,
            preferred_artist=metadata_artist,
            preferred_title=metadata_title,
        )
        metadata_candidates = _flatten_provider_group_candidates(metadata_provider_groups)
        _emit_stage(
            "metadata_search",
            "completed",
            f"元信息搜索完成，找到 {len(metadata_candidates)} 个候选",
            candidate_count=len(metadata_candidates),
        )
        if metadata_candidates:
            final_result = {
                "status": "ok",
                "error": "",
                "message": f"已通过音频元信息找到 {len(metadata_candidates)} 个同步lrc歌词候选",
                "search_mode": "metadata",
                "suggest_manual_query": False,
                "metadata_result": metadata_result,
                "fingerprint_result": {"status": "skipped", "reason": "metadata_hit"},
                "acoustid_result": {"status": "skipped", "reason": "metadata_hit"},
                "metadata_trace": _build_metadata_trace(
                    metadata_result=metadata_result,
                    fingerprint_result={"status": "skipped", "reason": "metadata_hit"},
                    acoustid_result={"status": "skipped", "reason": "metadata_hit"},
                ),
                "provider_groups": metadata_provider_groups,
                "candidates": metadata_candidates,
                "searched_candidate_count": len(metadata_candidates),
            }
            _emit("complete", final_result)
            return final_result
    else:
        _emit_stage("metadata_search", "skipped", "音频元信息不足，跳过元信息搜索")

    _emit_stage("fingerprint_build", "running", "正在生成音频指纹")
    fingerprint_result = build_fingerprint_result(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        fpcalc_bin=fpcalc_bin,
        logger=logger,
    )
    fingerprint_status = str(fingerprint_result.get("status", "")).strip().lower()
    if fingerprint_status != "ok":
        metadata_trace = _build_metadata_trace(
            metadata_result=metadata_result,
            fingerprint_result=fingerprint_result,
            acoustid_result={"status": "failed", "error": "fingerprint_not_ready"},
        )
        _emit_stage(
            "fingerprint_build",
            "failed",
            "音频指纹生成失败",
            error=str(fingerprint_result.get("error", "fingerprint_failed")).strip() or "fingerprint_failed",
        )
        final_result = {
            "status": "failed",
            "error": str(fingerprint_result.get("error", "fingerprint_failed")).strip() or "fingerprint_failed",
            "message": "自动搜索未命中，且音乐指纹检索不可用，请手动输入歌曲名搜索",
            "search_mode": "automatic",
            "suggest_manual_query": True,
            "metadata_result": metadata_result,
            "fingerprint_result": fingerprint_result,
            "acoustid_result": {"status": "failed", "error": "fingerprint_not_ready"},
            "metadata_trace": metadata_trace,
            "provider_groups": _build_empty_provider_groups(),
            "candidates": [],
            "searched_candidate_count": 0,
        }
        _emit("complete", final_result)
        return final_result
    _emit_stage("fingerprint_build", "completed", "音频指纹生成完成")

    _emit_stage("acoustid_match", "running", "正在识别曲目")
    acoustid_result = query_acoustid_match(
        fingerprint_result=fingerprint_result,
        acoustid_api_key_file=acoustid_api_key_file,
        logger=logger,
    )
    raw_candidate_summaries = _extract_acoustid_candidate_summaries(
        results=acoustid_result.get("raw_candidates", []),
        limit=safe_raw_candidate_limit,
    )
    if str(acoustid_result.get("status", "")).strip().lower() != "ok":
        metadata_trace = _build_metadata_trace(
            metadata_result=metadata_result,
            fingerprint_result=fingerprint_result,
            acoustid_result=acoustid_result,
        )
        _emit_stage(
            "acoustid_match",
            "failed",
            "曲目识别失败",
            error=str(acoustid_result.get("error", "acoustid_failed")).strip() or "acoustid_failed",
        )
        final_result = {
            "status": "failed",
            "error": str(acoustid_result.get("error", "acoustid_failed")).strip() or "acoustid_failed",
            "message": "自动搜索未命中，且音乐指纹识别未完成，请手动输入歌曲名搜索",
            "search_mode": "automatic",
            "suggest_manual_query": True,
            "metadata_result": metadata_result,
            "fingerprint_result": fingerprint_result,
            "acoustid_result": acoustid_result,
            "metadata_trace": metadata_trace,
            "provider_groups": _build_empty_provider_groups(),
            "candidates": [],
            "searched_candidate_count": len(raw_candidate_summaries),
        }
        _emit("complete", final_result)
        return final_result
    _emit_stage(
        "acoustid_match",
        "completed",
        "曲目识别完成",
        matched_count=len(raw_candidate_summaries),
        matched_artist=str(acoustid_result.get("artist", "")).strip(),
        matched_title=str(acoustid_result.get("title", "")).strip(),
    )
    _emit_stage("fingerprint_lyrics", "running", "正在按识别结果补全歌词")

    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for candidate_summary in raw_candidate_summaries:
        artist = str(candidate_summary.get("artist", "")).strip()
        title = str(candidate_summary.get("title", "")).strip()
        unique_key = (artist.lower(), title.lower())
        if (not artist) or (not title) or unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)
        lrclib_result = query_lrclib_lyrics(
            artist=artist,
            title=title,
            duration_seconds=float(candidate_summary.get("duration_seconds", duration_seconds) or duration_seconds),
            logger=logger,
        )
        if str(lrclib_result.get("status", "")).strip().lower() != "synced":
            continue
        synced_lyrics = str(lrclib_result.get("synced_lyrics", "")).strip()
        if not synced_lyrics:
            continue
        preview_lines = _extract_preview_lines(synced_lyrics=synced_lyrics, limit=4)
        candidate_id = f"cand_{len(candidates) + 1:03d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "artist": artist,
                "title": title,
                "score": float(candidate_summary.get("score", 0.0) or 0.0),
                "acoustid_id": str(candidate_summary.get("acoustid_id", "")).strip(),
                "recording_id": str(candidate_summary.get("recording_id", "")).strip(),
                "duration_seconds": float(lrclib_result.get("duration_seconds", duration_seconds) or duration_seconds),
                "provider": "lrclib",
                "provider_id": str(lrclib_result.get("provider_id", "")).strip(),
                "lrclib_status": "synced",
                "plain_lyrics": str(lrclib_result.get("plain_lyrics", "")).strip(),
                "synced_lyrics": synced_lyrics,
                "preview_lines": preview_lines,
                "preview_text": "\n".join(preview_lines),
            }
        )
        if len(candidates) >= safe_max_candidates:
            break
    _emit_stage(
        "fingerprint_lyrics",
        "completed",
        f"指纹补词完成，找到 {len(candidates)} 个候选",
        candidate_count=len(candidates),
    )
    if candidates:
        _emit(
            "provider_group",
            _decorate_provider_group(
                provider_name="lrclib",
                candidates=candidates,
                first_result_at_ms=_elapsed_ms(),
            ),
        )
    status_text = "ok" if candidates else "not_found"
    error_text = "" if candidates else "自动搜索未找到可用的同步LRC歌词候选，请手动输入歌曲名搜索"
    metadata_trace = _build_metadata_trace(
        metadata_result=metadata_result,
        fingerprint_result=fingerprint_result,
        acoustid_result=acoustid_result,
    )
    final_result = {
        "status": status_text,
        "error": error_text,
        "message": (
            f"已通过音乐指纹找到 {len(candidates)} 个同步lrc歌词候选"
            if candidates
            else "自动搜索未找到可用的同步LRC歌词候选，请手动输入歌曲名搜索"
        ),
        "search_mode": "fingerprint" if candidates else "automatic",
        "suggest_manual_query": not candidates,
        "metadata_result": metadata_result,
        "fingerprint_result": fingerprint_result,
        "acoustid_result": acoustid_result,
        "metadata_trace": metadata_trace,
        "provider_groups": _build_single_provider_group("lrclib", candidates),
        "candidates": candidates,
        "searched_candidate_count": len(raw_candidate_summaries),
    }
    _emit("complete", final_result)
    return final_result


def search_synced_lrc_candidates(
    audio_path: Path,
    duration_seconds: float,
    fpcalc_bin: str,
    acoustid_api_key_file: str,
    logger,
    manual_query: str = "",
    max_candidates: int = 10,
    raw_candidate_limit: int = 30,
) -> dict[str, Any]:
    """
    功能说明：为模块A页面查询可用的同步歌词候选。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - fpcalc_bin: 指纹命令路径或命令名。
    - acoustid_api_key_file: AcoustID API Key 文件路径。
    - logger: 日志对象。
    - manual_query: 可选的手动歌曲名搜索词。
    - max_candidates: 最多返回候选数。
    - raw_candidate_limit: 最多尝试的 AcoustID 原始候选数。
    返回值：
    - dict[str, Any]: 查询摘要与候选数组。
    异常说明：异常内部转为 failed 结果，不向上抛出。
    边界条件：仅返回具备同步歌词的 LRCLIB 候选；自动链未命中时会提示前端转手动搜歌名。
    """
    return stream_synced_lrc_candidates(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        fpcalc_bin=fpcalc_bin,
        acoustid_api_key_file=acoustid_api_key_file,
        logger=logger,
        manual_query=manual_query,
        emit_event=None,
        max_candidates=max_candidates,
        raw_candidate_limit=raw_candidate_limit,
        split_syncedlyrics_providers=False,
    )


def _build_lrclib_search_candidates(
    lrclib_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """
    功能说明：把 LRCLIB 搜索结果转成前端候选结构。
    参数说明：
    - lrclib_results: 已标准化的 LRCLIB 结果列表。
    - limit: 最多保留多少项。
    返回值：
    - list[dict[str, Any]]: 前端可直接消费的候选数组。
    异常说明：无。
    边界条件：仅接收同步歌词结果，并按 artist/title 去重。
    """
    normalized_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for lrclib_result in lrclib_results:
        artist = str(lrclib_result.get("artist", "")).strip()
        title = str(lrclib_result.get("title", "")).strip()
        unique_key = (artist.lower(), title.lower())
        if (not artist and not title) or unique_key in seen_keys:
            continue
        synced_lyrics = str(lrclib_result.get("synced_lyrics", "")).strip()
        if str(lrclib_result.get("status", "")).strip().lower() != "synced" or not synced_lyrics:
            continue
        seen_keys.add(unique_key)
        preview_lines = _extract_preview_lines(synced_lyrics=synced_lyrics, limit=4)
        candidate_id = f"cand_{len(normalized_candidates) + 1:03d}"
        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "artist": artist,
                "title": title,
                "score": 0.0,
                "acoustid_id": "",
                "recording_id": "",
                "duration_seconds": float(lrclib_result.get("duration_seconds", 0.0) or 0.0),
                "provider": str(lrclib_result.get("provider", "lrclib")).strip() or "lrclib",
                "provider_id": str(lrclib_result.get("provider_id", "")).strip(),
                "provider_song_id": str(lrclib_result.get("provider_song_id", "")).strip(),
                "provider_accesskey": str(lrclib_result.get("provider_accesskey", "")).strip(),
                "provider_hash": str(lrclib_result.get("provider_hash", "")).strip(),
                "lrclib_status": str(lrclib_result.get("status", "synced")).strip() or "synced",
                "plain_lyrics": str(lrclib_result.get("plain_lyrics", "")).strip(),
                "synced_lyrics": synced_lyrics,
                "word_timed_lyrics": str(lrclib_result.get("word_timed_lyrics", "")).strip(),
                "translated_lyrics": str(lrclib_result.get("translated_lyrics", "")).strip(),
                "romanized_lyrics": str(lrclib_result.get("romanized_lyrics", "")).strip(),
                "preview_lines": preview_lines,
                "preview_text": "\n".join(preview_lines),
            }
        )
        if len(normalized_candidates) >= max(1, int(limit)):
            break
    return normalized_candidates


def _build_metadata_search_queries(artist: str, title: str) -> list[str]:
    """
    功能说明：基于音频元信息生成用于 LRCLIB 模糊搜索的查询词序列。
    参数说明：
    - artist: 元信息艺人名。
    - title: 元信息曲名。
    返回值：
    - list[str]: 按优先级排序的自由文本查询词。
    异常说明：无。
    边界条件：优先只用原始标题搜索，避免 artist 误伤模糊匹配。
    """
    normalized_artist = str(artist).strip()
    normalized_title = str(title).strip()
    query_items: list[str] = []
    if normalized_title:
        query_items.append(normalized_title)
    if normalized_artist:
        if normalized_title:
            query_items.append(f"{normalized_artist} {normalized_title}".strip())
        query_items.append(normalized_artist)
    return _dedupe_query_texts(query_items)


def _search_manual_lrclib_candidates(
    *,
    manual_query: str,
    duration_seconds: float,
    logger,
    limit: int,
) -> list[dict[str, Any]]:
    """
    功能说明：按手动输入内容生成多组 LRCLIB 查询，并返回首组命中的同步歌词候选。
    参数说明：
    - manual_query: 用户手动输入的搜歌文本。
    - duration_seconds: 音频时长（秒）。
    - logger: 日志对象。
    - limit: 最多返回候选数。
    返回值：
    - list[dict[str, Any]]: 标准化后的前端候选数组。
    异常说明：无。
    边界条件：当输入为“歌手 - 歌名”时优先按结构化 artist/title 检索，再回退到标题与整句模糊搜索。
    """
    manual_query_attempts = _build_manual_search_attempts(manual_query=manual_query)
    for query_attempt in manual_query_attempts:
        manual_candidates = _build_lrclib_search_candidates(
            lrclib_results=search_lrclib_candidates(
                query_text=str(query_attempt.get("query_text", "")).strip(),
                artist=str(query_attempt.get("artist", "")).strip(),
                title=str(query_attempt.get("title", "")).strip(),
                duration_seconds=duration_seconds,
                logger=logger,
                limit=limit,
            ),
            limit=limit,
        )
        if manual_candidates:
            return manual_candidates
    return []


def _search_manual_network_candidates(
    *,
    manual_query: str,
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    split_syncedlyrics_providers: bool = False,
) -> list[dict[str, Any]]:
    """
    功能说明：按手动输入内容并发尝试多来源搜索，并按来源分组返回候选。
    参数说明：
    - manual_query: 用户手动输入的搜歌文本。
    - duration_seconds: 音频时长（秒）。
    - logger: 日志对象。
    - limit: 最多返回候选数。
    返回值：
    - list[dict[str, Any]]: 按来源分组的候选数组。
    异常说明：无。
    边界条件：每个来源最多返回指定数量候选，来源之间并发执行。
    """
    manual_query_attempts = _build_manual_search_attempts(manual_query=manual_query)
    return _search_provider_groups(
        query_attempts=manual_query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        progress_callback=progress_callback,
        started_at=started_at,
        split_syncedlyrics_providers=split_syncedlyrics_providers,
    )


def _search_provider_groups(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    split_syncedlyrics_providers: bool = False,
) -> list[dict[str, Any]]:
    """
    功能说明：并发搜索多歌词来源，并返回按来源分组的结果。
    参数说明：
    - query_attempts: 查询尝试序列。
    - duration_seconds: 音频时长（秒）。
    - logger: 日志对象。
    - limit: 每个来源最多返回多少项。
    返回值：
    - list[dict[str, Any]]: 分组后的候选数组。
    异常说明：单个来源失败时吞并并保留其他来源结果。
    边界条件：始终按固定来源顺序返回。
    """
    provider_specs = [
        ("netease_music", _search_netease_provider_candidates),
        ("qq_music", _search_qq_music_provider_candidates),
        ("kugou_music", _search_kugou_music_provider_candidates),
        ("lrclib", _search_lrclib_provider_candidates),
    ]
    if split_syncedlyrics_providers:
        for syncedlyrics_provider_name in SYNCEDLYRICS_PROVIDER_NAMES:
            provider_specs.append(
                (
                    _build_syncedlyrics_provider_group_key(syncedlyrics_provider_name),
                    _build_syncedlyrics_provider_handler(syncedlyrics_provider_name),
                )
            )
    else:
        provider_specs.append(("syncedlyrics", _search_syncedlyrics_provider_candidates))
    grouped_results_map: dict[str, list[dict[str, Any]]] = {
        provider_name: [] for provider_name, _handler in provider_specs
    }
    first_result_at_ms_map: dict[str, int | None] = {
        provider_name: None for provider_name, _handler in provider_specs
    }
    progress_lock = Lock()

    def _emit_provider_progress(provider_name: str, candidates: list[dict[str, Any]]) -> None:
        with progress_lock:
            grouped_results_map[provider_name] = [dict(item) for item in candidates if isinstance(item, dict)]
            if started_at is not None and grouped_results_map.get(provider_name, []) and first_result_at_ms_map[provider_name] is None:
                first_result_at_ms_map[provider_name] = int(round((time.perf_counter() - started_at) * 1000.0))
            provider_group = _decorate_provider_group(
                provider_name=provider_name,
                candidates=grouped_results_map.get(provider_name, []),
                first_result_at_ms=first_result_at_ms_map.get(provider_name),
            )
        if progress_callback is not None:
            progress_callback(provider_group)

    with ThreadPoolExecutor(max_workers=len(provider_specs)) as executor:
        future_map = {
            executor.submit(
                provider_handler,
                query_attempts=query_attempts,
                duration_seconds=duration_seconds,
                logger=logger,
                limit=limit,
                progress_callback=lambda candidates, current_provider_name=provider_name: _emit_provider_progress(
                    current_provider_name,
                    candidates,
                ),
            ): provider_name
            for provider_name, provider_handler in provider_specs
        }
        for future in as_completed(future_map):
            provider_name = future_map[future]
            try:
                grouped_results_map[provider_name] = future.result()
            except Exception as error:  # noqa: BLE001
                logger.warning("模块A V2-%s 搜索失败，错误=%s", provider_name, error)
                grouped_results_map[provider_name] = []
            if started_at is not None and grouped_results_map.get(provider_name, []):
                first_result_at_ms_map[provider_name] = int(round((time.perf_counter() - started_at) * 1000.0))
            if progress_callback is not None and not grouped_results_map.get(provider_name, []):
                progress_callback(
                    _decorate_provider_group(
                        provider_name=provider_name,
                        candidates=[],
                        first_result_at_ms=first_result_at_ms_map.get(provider_name),
                    )
                )
    return [
        _decorate_provider_group(
            provider_name=provider_name,
            candidates=grouped_results_map.get(provider_name, []),
            first_result_at_ms=first_result_at_ms_map.get(provider_name),
        )
        for provider_name, _handler in provider_specs
    ]


def _search_netease_provider_candidates(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    return _search_provider_candidates_by_attempts(
        query_attempts=query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        provider_name="netease_music",
        provider_search_fn=search_netease_music_candidates,
        progress_callback=progress_callback,
        supports_candidate_callback=True,
    )


def _search_qq_music_provider_candidates(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    return _search_provider_candidates_by_attempts(
        query_attempts=query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        provider_name="qq_music",
        provider_search_fn=search_qq_music_candidates,
        progress_callback=progress_callback,
        supports_candidate_callback=True,
    )


def _search_kugou_music_provider_candidates(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    return _search_provider_candidates_by_attempts(
        query_attempts=query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        provider_name="kugou_music",
        provider_search_fn=search_kugou_music_candidates,
        progress_callback=progress_callback,
        supports_candidate_callback=True,
    )


def _search_lrclib_provider_candidates(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    return _search_provider_candidates_by_attempts(
        query_attempts=query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        provider_name="lrclib",
        provider_search_fn=search_lrclib_candidates,
        include_duration=True,
        progress_callback=progress_callback,
    )


def _search_syncedlyrics_provider_candidates(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    return _search_provider_candidates_by_attempts(
        query_attempts=query_attempts,
        duration_seconds=duration_seconds,
        logger=logger,
        limit=limit,
        provider_name="syncedlyrics",
        provider_search_fn=search_syncedlyrics_candidates,
        progress_callback=progress_callback,
    )


def _build_syncedlyrics_provider_handler(provider_name: str):
    normalized_provider_name = str(provider_name).strip()

    def _handler(
        *,
        query_attempts: list[dict[str, str]],
        duration_seconds: float,
        logger,
        limit: int,
        progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        return _search_provider_candidates_by_attempts(
            query_attempts=query_attempts,
            duration_seconds=duration_seconds,
            logger=logger,
            limit=limit,
            provider_name=_build_syncedlyrics_provider_group_key(normalized_provider_name),
            provider_search_fn=lambda **kwargs: search_syncedlyrics_candidates_by_provider(
                provider_name=normalized_provider_name,
                **kwargs,
            ),
            progress_callback=progress_callback,
        )

    return _handler


def _search_provider_candidates_by_attempts(
    *,
    query_attempts: list[dict[str, str]],
    duration_seconds: float,
    logger,
    limit: int,
    provider_name: str,
    provider_search_fn,
    include_duration: bool = False,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    supports_candidate_callback: bool = False,
) -> list[dict[str, Any]]:
    """
    功能说明：对单个来源按多组查询尝试执行搜索并聚合去重结果。
    参数说明：
    - query_attempts: 查询尝试序列。
    - duration_seconds: 音频时长（秒）。
    - logger: 日志对象。
    - limit: 最多返回候选数。
    - provider_search_fn: 实际来源搜索函数。
    - include_duration: 是否向搜索函数传入时长。
    返回值：
    - list[dict[str, Any]]: 标准化候选数组。
    异常说明：单次尝试异常时继续后续尝试。
    边界条件：仅保留具备同步歌词的候选。
    """
    aggregated_items: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    safe_limit = max(1, int(limit))

    def _append_candidate(candidate: dict[str, Any]) -> bool:
        normalized_candidate = dict(candidate)
        normalized_candidate["candidate_id"] = f"{provider_name}_{len(aggregated_items) + 1:03d}"
        unique_key = (
            str(normalized_candidate.get("provider", "")).strip().lower(),
            str(normalized_candidate.get("provider_id", "")).strip().lower(),
            build_candidate_label_key(candidate=normalized_candidate),
        )
        if unique_key in seen_keys:
            return False
        seen_keys.add(unique_key)
        aggregated_items.append(normalized_candidate)
        if progress_callback is not None:
            progress_callback(list(aggregated_items))
        return len(aggregated_items) >= safe_limit

    for query_attempt in query_attempts:
        kwargs = {
            "query_text": str(query_attempt.get("query_text", "")).strip(),
            "artist": str(query_attempt.get("artist", "")).strip(),
            "title": str(query_attempt.get("title", "")).strip(),
            "logger": logger,
            "limit": safe_limit,
        }
        if include_duration:
            kwargs["duration_seconds"] = duration_seconds
        if supports_candidate_callback and _provider_search_fn_accepts_candidate_callback(provider_search_fn):
            kwargs["candidate_callback"] = _append_candidate
        try:
            provider_results = provider_search_fn(**kwargs)
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A V2-provider 搜索尝试失败，query=%s，错误=%s", kwargs, error)
            continue
        normalized_candidates = _build_lrclib_search_candidates(
            lrclib_results=provider_results,
            limit=safe_limit,
        )
        for candidate in normalized_candidates:
            if _append_candidate(candidate):
                return aggregated_items
    return aggregated_items


def build_candidate_label_key(candidate: dict[str, Any]) -> str:
    """
    功能说明：为候选构造稳定的去重键片段。
    参数说明：
    - candidate: 候选对象。
    返回值：
    - str: 由 artist/title 组成的小写键。
    异常说明：无。
    边界条件：缺失字段时仍返回可比较字符串。
    """
    artist = str(candidate.get("artist", "")).strip().lower()
    title = str(candidate.get("title", "")).strip().lower()
    return f"{artist}::{title}"


def _provider_search_fn_accepts_candidate_callback(provider_search_fn) -> bool:
    try:
        provider_signature = signature(provider_search_fn)
    except (TypeError, ValueError):
        return False
    if "candidate_callback" in provider_signature.parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in provider_signature.parameters.values())


def _build_query_text_search_attempts(query_texts: list[str]) -> list[dict[str, str]]:
    """
    功能说明：把自由文本查询词序列转换为统一查询尝试结构。
    参数说明：
    - query_texts: 查询词数组。
    返回值：
    - list[dict[str, str]]: 统一查询尝试数组。
    异常说明：无。
    边界条件：自动去重空白项。
    """
    return _dedupe_search_attempts(
        [
            {
                "query_text": str(query_text).strip(),
                "artist": "",
                "title": "",
            }
            for query_text in query_texts
            if str(query_text).strip()
        ]
    )


def _ordered_provider_names() -> list[str]:
    return [
        "netease_music",
        "qq_music",
        "kugou_music",
        "lrclib",
        *[_build_syncedlyrics_provider_group_key(item) for item in SYNCEDLYRICS_PROVIDER_NAMES],
    ]


def _build_syncedlyrics_provider_group_key(provider_name: str) -> str:
    normalized_provider_name = str(provider_name).strip()
    return f"syncedlyrics::{normalized_provider_name.lower()}"


def _build_provider_display_name(provider_name: str) -> str:
    normalized_provider_name = str(provider_name).strip()
    if normalized_provider_name.startswith("syncedlyrics::"):
        raw_provider_name = normalized_provider_name.split("::", 1)[1].strip()
        return f"syncedlyrics: {raw_provider_name}"
    return PROVIDER_DISPLAY_NAMES.get(normalized_provider_name, normalized_provider_name)


def _decorate_provider_group(
    *,
    provider_name: str,
    candidates: list[dict[str, Any]],
    first_result_at_ms: int | None,
) -> dict[str, Any]:
    normalized_candidates = [item for item in candidates if isinstance(item, dict)]
    return {
        "provider": str(provider_name).strip(),
        "display_name": _build_provider_display_name(provider_name),
        "candidates": normalized_candidates,
        "first_result_at_ms": int(first_result_at_ms) if first_result_at_ms is not None and normalized_candidates else None,
        "page_size": STREAM_PROVIDER_PAGE_SIZE,
        "total_count": len(normalized_candidates),
        "has_more": len(normalized_candidates) > STREAM_PROVIDER_PAGE_SIZE,
    }


def _flatten_provider_group_candidates(provider_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：把按来源分组的候选拍平为单数组。
    参数说明：
    - provider_groups: 分组候选数组。
    返回值：
    - list[dict[str, Any]]: 拍平后的候选数组。
    异常说明：无。
    边界条件：自动跳过非法输入。
    """
    flattened_items: list[dict[str, Any]] = []
    for provider_group in provider_groups:
        if not isinstance(provider_group, dict):
            continue
        candidates = provider_group.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        flattened_items.extend(item for item in candidates if isinstance(item, dict))
    return flattened_items


def _sort_provider_groups_candidates_by_similarity(
    *,
    provider_groups: list[dict[str, Any]],
    preferred_artist: str,
    preferred_title: str,
) -> list[dict[str, Any]]:
    """
    功能说明：保持来源顺序不变，仅按歌手/歌名字符相似度重排来源内候选。
    参数说明：
    - provider_groups: 原始来源分组数组。
    - preferred_artist: 优先命中的歌手名。
    - preferred_title: 优先命中的歌名。
    返回值：
    - list[dict[str, Any]]: 仅来源内候选已重排的分组数组。
    异常说明：无。
    边界条件：空来源直接透传。
    """
    sorted_groups: list[dict[str, Any]] = []
    for provider_group in provider_groups:
        if not isinstance(provider_group, dict):
            continue
        raw_candidates = provider_group.get("candidates", [])
        if not isinstance(raw_candidates, list):
            sorted_groups.append(dict(provider_group))
            continue
        sorted_candidates = sorted(
            (dict(item) for item in raw_candidates if isinstance(item, dict)),
            key=lambda item: _build_candidate_similarity_sort_key(
                candidate=item,
                preferred_artist=preferred_artist,
                preferred_title=preferred_title,
            ),
            reverse=True,
        )
        sorted_groups.append(
            {
                **provider_group,
                "candidates": sorted_candidates,
            }
        )
    return sorted_groups


def _build_candidate_similarity_sort_key(
    *,
    candidate: dict[str, Any],
    preferred_artist: str,
    preferred_title: str,
) -> tuple[float, float, float]:
    normalized_artist = str(candidate.get("artist", "")).strip()
    normalized_title = str(candidate.get("title", "")).strip()
    title_match_score = _compute_exact_match_score(normalized_title, preferred_title)
    artist_match_score = _compute_exact_match_score(normalized_artist, preferred_artist)
    provider_score = float(candidate.get("score", 0.0) or 0.0)
    return (
        float(title_match_score),
        float(artist_match_score),
        provider_score,
    )


def _compute_exact_match_score(candidate_text: str, preferred_text: str) -> int:
    normalized_candidate_text = _normalize_match_text(candidate_text)
    normalized_preferred_text = _normalize_match_text(preferred_text)
    if not normalized_candidate_text or not normalized_preferred_text:
        return 0
    if normalized_candidate_text == normalized_preferred_text:
        return 3
    if normalized_preferred_text in normalized_candidate_text or normalized_candidate_text in normalized_preferred_text:
        return 2
    overlap_count = sum(1 for char in normalized_preferred_text if char in normalized_candidate_text)
    if overlap_count > 0:
        return 1
    return 0


def _normalize_match_text(text: str) -> str:
    normalized_text = str(text).strip().lower()
    if not normalized_text:
        return ""
    normalized_text = TRAILING_BRACKET_NOTE_PATTERN.sub("", normalized_text).strip()
    return NON_WORD_NORMALIZE_PATTERN.sub("", normalized_text)


def _build_empty_provider_groups() -> list[dict[str, Any]]:
    return [
        _decorate_provider_group(provider_name=provider_name, candidates=[], first_result_at_ms=None)
        for provider_name in _ordered_provider_names()
    ]


def _build_single_provider_group(provider_name: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    功能说明：把单来源候选包装为统一分组结构，其余来源保留空数组。
    参数说明：
    - provider_name: 来源名称。
    - candidates: 候选数组。
    返回值：
    - list[dict[str, Any]]: 分组结构。
    异常说明：无。
    边界条件：保持固定来源顺序。
    """
    groups = []
    for current_provider_name in _ordered_provider_names():
        groups.append(
            _decorate_provider_group(
                provider_name=current_provider_name,
                candidates=candidates if current_provider_name == provider_name else [],
                first_result_at_ms=None,
            )
        )
    return groups




def _build_manual_search_attempts(manual_query: str) -> list[dict[str, str]]:
    """
    功能说明：为手动搜歌名构造按优先级排序的查询尝试序列。
    参数说明：
    - manual_query: 用户原始输入。
    返回值：
    - list[dict[str, str]]: 每项包含 query_text/artist/title 的查询组合。
    异常说明：无。
    边界条件：支持“歌手 - 歌名”“歌手-歌名”“歌手/歌名”等常见写法。
    """
    normalized_manual_query = str(manual_query).strip()
    if not normalized_manual_query:
        return []
    manual_artist, manual_title = _split_manual_artist_title(manual_query=normalized_manual_query)
    search_attempts: list[dict[str, str]] = []
    if manual_artist and manual_title:
        search_attempts.append(
            {
                "query_text": "",
                "artist": manual_artist,
                "title": manual_title,
            }
        )
        search_attempts.append(
            {
                "query_text": "",
                "artist": "",
                "title": manual_title,
            }
        )
    search_attempts.append(
        {
            "query_text": normalized_manual_query,
            "artist": "",
            "title": "",
        }
    )
    return _dedupe_search_attempts(search_attempts)


def _split_manual_artist_title(manual_query: str) -> tuple[str, str]:
    """
    功能说明：尝试把手动搜歌文本拆解为歌手与歌名。
    参数说明：
    - manual_query: 用户原始输入。
    返回值：
    - tuple[str, str]: `(artist, title)`；无法识别时返回空字符串元组。
    异常说明：无。
    边界条件：只在分隔符两侧均有有效内容时视为成功拆分。
    """
    normalized_manual_query = str(manual_query).strip()
    split_separators = [" - ", " -", "- ", "-", " / ", "/", " | ", "|", "：", ":"]
    for separator in split_separators:
        if separator not in normalized_manual_query:
            continue
        artist_part, title_part = normalized_manual_query.split(separator, 1)
        normalized_artist = str(artist_part).strip()
        normalized_title = str(title_part).strip()
        if normalized_artist and normalized_title:
            return normalized_artist, normalized_title
    return "", ""


def _dedupe_search_attempts(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    功能说明：对手动查询尝试序列去重，避免重复请求同一组参数。
    参数说明：
    - items: 原始查询尝试数组。
    返回值：
    - list[dict[str, str]]: 去重后的查询尝试数组。
    异常说明：无。
    边界条件：空白查询会被移除。
    """
    normalized_items: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_query_text = str(item.get("query_text", "")).strip()
        normalized_artist = str(item.get("artist", "")).strip()
        normalized_title = str(item.get("title", "")).strip()
        if not normalized_query_text and not normalized_artist and not normalized_title:
            continue
        dedupe_key = (
            normalized_query_text.lower(),
            normalized_artist.lower(),
            normalized_title.lower(),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        normalized_items.append(
            {
                "query_text": normalized_query_text,
                "artist": normalized_artist,
                "title": normalized_title,
            }
        )
    return normalized_items


def _dedupe_query_texts(items: list[str]) -> list[str]:
    """
    功能说明：去重并裁剪搜索词序列。
    参数说明：
    - items: 原始搜索词数组。
    返回值：
    - list[str]: 过滤后的搜索词数组。
    异常说明：无。
    边界条件：空白项会被移除。
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


def _build_metadata_trace(
    metadata_result: Any,
    fingerprint_result: Any,
    acoustid_result: Any,
) -> dict[str, Any]:
    """
    功能说明：把文件元信息与指纹识别结果压缩成前端可直接展示的摘要。
    参数说明：
    - metadata_result: 文件内嵌元信息结果。
    - fingerprint_result: 指纹生成结果。
    - acoustid_result: 指纹匹配元信息结果。
    返回值：
    - dict[str, Any]: 联网找词诊断摘要。
    异常说明：无。
    边界条件：输入非法时自动回退为空字段。
    """
    normalized_metadata_result = metadata_result if isinstance(metadata_result, dict) else {}
    normalized_fingerprint_result = fingerprint_result if isinstance(fingerprint_result, dict) else {}
    normalized_acoustid_result = acoustid_result if isinstance(acoustid_result, dict) else {}
    return {
        "embedded_status": str(normalized_metadata_result.get("status", "")).strip(),
        "embedded_source": str(normalized_metadata_result.get("source", "")).strip(),
        "embedded_artist": str(normalized_metadata_result.get("artist", "")).strip(),
        "embedded_title": str(normalized_metadata_result.get("title", "")).strip(),
        "embedded_album": str(normalized_metadata_result.get("album", "")).strip(),
        "embedded_error": str(normalized_metadata_result.get("error", "")).strip(),
        "fingerprint_status": str(normalized_fingerprint_result.get("status", "")).strip(),
        "fingerprint_error": str(normalized_fingerprint_result.get("error", "")).strip(),
        "acoustid_status": str(normalized_acoustid_result.get("status", "")).strip(),
        "matched_artist": str(normalized_acoustid_result.get("artist", "")).strip(),
        "matched_title": str(normalized_acoustid_result.get("title", "")).strip(),
        "matched_score": float(normalized_acoustid_result.get("score", 0.0) or 0.0),
        "matched_error": str(normalized_acoustid_result.get("error", "")).strip(),
    }


def _extract_acoustid_candidate_summaries(results: Any, limit: int) -> list[dict[str, Any]]:
    """
    功能说明：从 AcoustID 原始结果提取可读候选摘要。
    参数说明：
    - results: AcoustID 原始 results。
    - limit: 最多返回数量。
    返回值：
    - list[dict[str, Any]]: 候选摘要数组。
    异常说明：无。
    边界条件：只收录具备 artist/title 的 recording。
    """
    if not isinstance(results, list):
        return []
    normalized_items: list[dict[str, Any]] = []
    for result_item in results:
        if not isinstance(result_item, dict):
            continue
        recordings = result_item.get("recordings", [])
        if not isinstance(recordings, list) or not recordings:
            continue
        first_recording = recordings[0]
        if not isinstance(first_recording, dict):
            continue
        artist = _extract_artist_name(recording_item=first_recording)
        title = str(first_recording.get("title", "")).strip()
        if (not artist) or (not title):
            continue
        normalized_items.append(
            {
                "artist": artist,
                "title": title,
                "score": float(result_item.get("score", 0.0) or 0.0),
                "acoustid_id": str(result_item.get("id", "")).strip(),
                "recording_id": str(first_recording.get("id", "")).strip(),
            }
        )
    normalized_items.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return normalized_items[: max(1, int(limit))]


def _extract_artist_name(recording_item: dict[str, Any]) -> str:
    """
    功能说明：从 AcoustID recording 节点提取首个艺人名。
    参数说明：
    - recording_item: 单个 recording 节点。
    返回值：
    - str: 艺人名。
    异常说明：无。
    边界条件：未命中时返回空字符串。
    """
    artists = recording_item.get("artists", [])
    if not isinstance(artists, list) or not artists:
        return ""
    first_artist = artists[0]
    if not isinstance(first_artist, dict):
        return ""
    return str(first_artist.get("name", "")).strip()


def _extract_preview_lines(synced_lyrics: str, limit: int) -> list[str]:
    """
    功能说明：从同步歌词原文中提取预览行。
    参数说明：
    - synced_lyrics: LRC 原文。
    - limit: 最多提取多少行。
    返回值：
    - list[str]: 预览文本行。
    异常说明：无。
    边界条件：保留原始时间戳，方便用户确认歌词是否对得上。
    """
    preview_lines: list[str] = []
    for raw_line in str(synced_lyrics).splitlines():
        line_text = raw_line.strip()
        if not line_text:
            continue
        preview_lines.append(line_text)
        if len(preview_lines) >= max(1, int(limit)):
            break
    return preview_lines
