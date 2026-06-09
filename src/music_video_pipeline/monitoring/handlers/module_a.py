"""
文件用途：模块 A handler mixin —— 模块 A 页面数据、联网歌词搜索/选择/详情。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖模块 A 歌词查找 pipeline、网络歌词状态、音频探测等。
维护说明：本文件仅包含模块 A 专属方法。
"""

import asyncio
import json
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline import (
    search_synced_lrc_candidates,
    stream_synced_lrc_candidates,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.kugou_music import (
    fetch_kugou_music_lyrics_bundle,
    fetch_kugou_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.netease_music import (
    fetch_netease_music_lyrics_bundle,
    fetch_netease_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music import (
    fetch_qq_music_lyrics_bundle,
    fetch_qq_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lrclib_client import query_lrclib_lyrics
from music_video_pipeline.modules.module_a_v2.network_lyrics_state import (
    current_time_text,
    load_module_a_network_lyrics_state,
    write_module_a_network_lyrics_state,
)
from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration
from music_video_pipeline.modules.module_a_v2.visualization import collect_visualization_payload
from music_video_pipeline.monitoring.routes import (
    TASK_MODULE_A_API_PATH,
    TASK_MODULE_A_CANDIDATE_LYRICS_API_PATH,
    TASK_MODULE_A_SEARCH_LYRICS_API_PATH,
    TASK_MODULE_A_SEARCH_LYRICS_WS_PATH,
    TASK_MODULE_A_SELECT_LYRICS_API_PATH,
    TASK_MODULE_A_VISUALIZATION_PAYLOAD_API_PATH,
)

try:
    from websockets.exceptions import ConnectionClosed
except Exception:
    ConnectionClosed = Exception


class ModuleAHandlers:
    """Mixin —— 模块 A 相关方法。"""

    def _build_module_a_stream_preview_provider_group(self, provider_group: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：为 WebSocket 实时流裁剪来源预览页，只推送当前页首屏所需内容。
        参数说明：
        - provider_group: 完整来源分组对象。
        返回值：
        - dict[str, Any]: 裁剪后的来源分组对象。
        异常说明：无。
        边界条件：保留 total_count/has_more，方便前端决定是否继续按需加载。
        """
        preview_group = dict(provider_group) if isinstance(provider_group, dict) else {}
        candidates = provider_group.get("candidates", []) if isinstance(provider_group, dict) else []
        normalized_candidates = [dict(item) for item in candidates[:10] if isinstance(item, dict)]
        preview_group["candidates"] = normalized_candidates
        preview_group["page_size"] = 10
        preview_group["total_count"] = int(provider_group.get("total_count", len(candidates)) or len(candidates))
        preview_group["has_more"] = preview_group["total_count"] > 10
        return preview_group

    def _build_module_a_stream_preview_result(self, search_result: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：为 WebSocket 完成事件裁剪来源预览页，避免前端在未翻页前接收全部候选。
        参数说明：
        - search_result: 完整搜索结果。
        返回值：
        - dict[str, Any]: 裁剪后的搜索结果。
        异常说明：无。
        边界条件：持久化仍使用完整结果，本函数仅影响实时推送。
        """
        preview_result = dict(search_result) if isinstance(search_result, dict) else {}
        provider_groups = search_result.get("provider_groups", []) if isinstance(search_result, dict) else []
        preview_groups = [
            self._build_module_a_stream_preview_provider_group(item)
            for item in provider_groups
            if isinstance(item, dict)
        ]
        preview_result["provider_groups"] = preview_groups
        preview_result["candidates"] = [
            dict(candidate)
            for provider_group in preview_groups
            for candidate in provider_group.get("candidates", [])
            if isinstance(candidate, dict)
        ]
        return preview_result

    def _build_module_a_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 A 页面所需的数据负载。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含模块A可视化与联网歌词状态的数据对象。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：不在此接口返回重型审阅时间线数据，避免页面重复拉取。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
            }
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        visualization_path = self._resolve_module_a_visualization_path(task_dir=task_dir, task_id=normalized_task_id)
        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:  # noqa: BLE001
            module_status_map = {}
        return {
            "ok": True,
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "module_a_status": str(module_status_map.get("A", "unknown")),
            "module_a_visualization": {
                "available": visualization_path is not None and visualization_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=visualization_path)
                if visualization_path
                else "",
                "path": str(visualization_path) if visualization_path else "",
            },
            "network_lrc_state": self._build_module_a_network_lyrics_summary(task_dir=task_dir),
        }

    def _handle_module_a_search_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 A 的联网同步歌词搜索请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：仅返回最多10个具备同步歌词的候选。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        manual_query = str(query.get("manual_query", [""])[0]).strip()
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"联网查找歌词失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        if self.app_config is None:
            return {"ok": False, "error": "当前监督服务缺少运行配置，无法联网查找歌词。"}, HTTPStatus.SERVICE_UNAVAILABLE

        if not str(task_record.get("audio_path", "")).strip():
            return {"ok": False, "error": f"联网查找歌词失败：任务缺少 audio_path，task_id={task_id}"}, HTTPStatus.BAD_REQUEST
        try:
            audio_path = self._resolve_task_audio_path_from_record(task_id=task_id, task_record=task_record, persist=True)
        except FileNotFoundError as error:
            return {"ok": False, "error": f"联网查找歌词失败：{error}"}, HTTPStatus.NOT_FOUND

        try:
            duration_seconds = probe_audio_duration(
                audio_path=audio_path,
                ffprobe_bin=str(getattr(getattr(self.app_config, "ffmpeg", None), "ffprobe_bin", "ffprobe")),
                logger=self.logger,
            )
            search_result = search_synced_lrc_candidates(
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                fpcalc_bin=str(getattr(getattr(self.app_config, "module_a", None), "fpcalc_bin", "fpcalc")),
                acoustid_api_key_file=str(
                    getattr(getattr(self.app_config, "module_a", None), "acoustid_api_key_file", "")
                ),
                logger=self.logger,
                manual_query=manual_query,
                max_candidates=10,
                raw_candidate_limit=30,
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning("[监督服务] 模块A联网歌词搜索失败，task_id=%s，错误=%s", task_id, error)
            return {"ok": False, "error": f"联网查找歌词失败：{error}", "task_id": task_id}, HTTPStatus.BAD_GATEWAY

        self._persist_module_a_search_result(task_id=task_id, search_result=search_result)
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        metadata_trace = self._build_module_a_metadata_trace_summary(raw_state.get("metadata_trace", {}))

        status_text = str(search_result.get("status", "")).strip().lower()
        if status_text == "failed":
            error_text = str(search_result.get("error", "")).strip() or "联网查找歌词失败"
            if bool(search_result.get("suggest_manual_query", False)):
                return {
                    "ok": True,
                    "task_id": task_id,
                    "search_status": "failed",
                    "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
                    "message": str(search_result.get("message", "")).strip() or error_text,
                    "error": error_text,
                    "suggest_manual_query": True,
                    "metadata_trace": metadata_trace,
                    "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
                    "candidates": [],
                }, HTTPStatus.OK
            return {"ok": False, "error": error_text, "task_id": task_id}, HTTPStatus.BAD_GATEWAY

        candidates = [
            self._build_module_a_candidate_summary(item)
            for item in raw_state.get("candidates", [])
            if isinstance(item, dict)
        ]
        if status_text == "not_found":
            return {
                "ok": True,
                "task_id": task_id,
                "search_status": "not_found",
                "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
                "message": str(search_result.get("message", "")).strip()
                or str(search_result.get("error", "")).strip()
                or "未找到可用的同步lrc歌词候选",
                "suggest_manual_query": bool(search_result.get("suggest_manual_query", False)),
                "metadata_trace": metadata_trace,
                "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
                "candidates": [],
            }, HTTPStatus.OK
        return {
            "ok": True,
            "task_id": task_id,
            "search_status": "ok",
            "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
            "message": str(search_result.get("message", "")).strip() or f"已找到 {len(candidates)} 个同步lrc歌词候选",
            "suggest_manual_query": False,
            "metadata_trace": metadata_trace,
            "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
            "candidates": candidates,
        }, HTTPStatus.OK

    def _persist_module_a_search_result(self, task_id: str, search_result: dict[str, Any]) -> None:
        """
        功能说明：把模块A联网歌词搜索结果写入任务状态文件。
        参数说明：
        - task_id: 任务唯一标识。
        - search_result: 搜索结果对象。
        返回值：无。
        异常说明：无；调用方负责外围异常处理。
        边界条件：仅持久化前端需要的最小字段。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        raw_state["updated_at"] = current_time_text()
        raw_state["last_search_at"] = raw_state["updated_at"]
        raw_state["search_status"] = str(search_result.get("status", "")).strip()
        raw_state["lookup_error"] = str(search_result.get("error", "")).strip()
        raw_state["fingerprint_status"] = str(
            search_result.get("fingerprint_result", {}).get("status", "")
            if isinstance(search_result.get("fingerprint_result", {}), dict)
            else ""
        ).strip()
        raw_state["acoustid_status"] = str(
            search_result.get("acoustid_result", {}).get("status", "")
            if isinstance(search_result.get("acoustid_result", {}), dict)
            else ""
        ).strip()
        raw_state["metadata_trace"] = (
            dict(search_result.get("metadata_trace", {}))
            if isinstance(search_result.get("metadata_trace", {}), dict)
            else {}
        )
        raw_state["candidates"] = [
            dict(item)
            for item in search_result.get("candidates", [])
            if isinstance(item, dict)
        ]
        raw_state["provider_groups"] = [
            dict(item)
            for item in search_result.get("provider_groups", [])
            if isinstance(item, dict)
        ]
        write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)

    def _handle_module_a_select_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 A 页面"选中候选歌词并决定是否启用"的请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：启用时会立即触发从模块A开始的后台重跑。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        candidate_id = str(query.get("candidate_id", [""])[0]).strip()
        enable_text = str(query.get("enable", ["0"])[0]).strip().lower()
        enable_lookup = enable_text in {"1", "true", "yes", "enabled"}
        if not candidate_id:
            return {"ok": False, "error": "候选歌词选择失败：candidate_id 不能为空。"}, HTTPStatus.BAD_REQUEST

        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"候选歌词选择失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND

        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        selected_candidate = self._find_module_a_candidate_by_id(
            candidates=raw_state.get("candidates", []),
            candidate_id=candidate_id,
        )
        if selected_candidate is None:
            return {
                "ok": False,
                "error": f"候选歌词选择失败：未找到 candidate_id={candidate_id}，请重新联网查找。",
            }, HTTPStatus.NOT_FOUND
        if enable_lookup and not str(selected_candidate.get("synced_lyrics", "")).strip():
            return {"ok": False, "error": "候选歌词选择失败：当前候选不包含可用的同步lrc歌词。"}, HTTPStatus.BAD_REQUEST

        raw_state["selected_candidate_id"] = candidate_id
        raw_state["selected_candidate"] = dict(selected_candidate)
        raw_state["enabled"] = bool(enable_lookup)
        raw_state["display_status"] = "enabled" if enable_lookup else "searched_not_enabled"
        raw_state["updated_at"] = current_time_text()
        raw_state["lookup_error"] = ""
        write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)

        if not enable_lookup:
            return {
                "ok": True,
                "task_id": task_id,
                "message": "已联网查找lrc但未启用",
            }, HTTPStatus.OK

        payload, status = self._submit_task_rerun_request(
            task_id=task_id,
            success_message="已经启用联网查找的lrc，并开始重跑模块A",
            log_reason=f"module_a_network_lrc:{candidate_id}",
        )
        return payload, status

    def _handle_module_a_candidate_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：按候选ID返回模块 A 联网歌词候选的完整同步歌词内容。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：仅从已缓存候选中读取，不触发新的联网搜索。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        candidate_id = str(query.get("candidate_id", [""])[0]).strip()
        if not candidate_id:
            return {"ok": False, "error": "候选歌词详情读取失败：candidate_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"候选歌词详情读取失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        candidate = self._find_module_a_candidate_by_id(
            candidates=raw_state.get("candidates", []),
            candidate_id=candidate_id,
        )
        if candidate is None:
            return {
                "ok": False,
                "error": f"候选歌词详情读取失败：未找到 candidate_id={candidate_id}。",
            }, HTTPStatus.NOT_FOUND
        candidate = self._hydrate_module_a_candidate_detail(task_id=task_id, artifacts_dir=artifacts_dir, raw_state=raw_state, candidate=candidate)
        return {
            "ok": True,
            "task_id": task_id,
            "candidate": self._build_module_a_candidate_summary(candidate),
            "synced_lyrics": str(candidate.get("synced_lyrics", "")).strip(),
            "word_timed_lyrics": str(candidate.get("word_timed_lyrics", "")).strip(),
            "translated_lyrics": str(candidate.get("translated_lyrics", "")).strip(),
            "romanized_lyrics": str(candidate.get("romanized_lyrics", "")).strip(),
        }, HTTPStatus.OK

    def _build_module_a_network_lyrics_summary(self, task_dir: Path) -> dict[str, Any]:
        """
        功能说明：构建模块 A 页面所需的联网歌词状态摘要。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, Any]: 轻量状态摘要。
        异常说明：无。
        边界条件：不返回完整歌词正文，避免页面重复拉取大字段。
        """
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        return {
            "display_status": str(raw_state.get("display_status", "idle")).strip() or "idle",
            "enabled": bool(raw_state.get("enabled", False)),
            "updated_at": str(raw_state.get("updated_at", "")).strip(),
            "last_search_at": str(raw_state.get("last_search_at", "")).strip(),
            "search_status": str(raw_state.get("search_status", "")).strip(),
            "lookup_error": str(raw_state.get("lookup_error", "")).strip(),
            "cached_candidates_count": len(raw_state.get("candidates", []))
            if isinstance(raw_state.get("candidates", []), list)
            else 0,
            "metadata_trace": self._build_module_a_metadata_trace_summary(raw_state.get("metadata_trace", {})),
            "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
            "selected_candidate": self._build_module_a_candidate_summary(raw_state.get("selected_candidate", {})),
        }

    def _handle_module_a_visualization_payload_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 A 可视化数据负载请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：任务不存在时返回 404，数据文件缺失时返回 404。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND
        task_dir = self._resolve_task_dir(task_id=task_id)
        try:
            payload = collect_visualization_payload(task_dir=task_dir)
        except Exception as error:  # noqa: BLE001
            self.logger.warning(
                "[监督服务] 模块A可视化负载加载失败，task_id=%s，错误=%s",
                task_id,
                error,
            )
            return {
                "ok": False,
                "error": f"可视化数据加载失败：{error}",
                "task_id": task_id,
            }, HTTPStatus.NOT_FOUND
        audio_path = Path(str(payload.get("audio_path", "")))
        audio_available = audio_path.exists() and audio_path.is_file()
        if audio_available:
            audio_url = self._build_task_file_url(task_id=task_id, file_path=audio_path)
        else:
            audio_url = ""
        payload["ok"] = True
        payload["task_id"] = task_id
        payload["task_dir"] = str(task_dir)
        payload["audio_url"] = audio_url
        payload["audio_available"] = audio_available
        return payload, HTTPStatus.OK

    def _fetch_module_a_candidate_synced_lyrics(self, candidate: dict[str, Any]) -> str:
        """
        功能说明：当缓存候选未带正文时，按来源即时补拉同步歌词。
        参数说明：
        - candidate: 候选对象。
        返回值：
        - str: 补拉到的同步歌词正文；失败时返回空字符串。
        异常说明：无；内部异常统一记录日志并回退空字符串。
        边界条件：优先复用 provider/provider_id，必要时回退 artist/title 检索。
        """
        if not isinstance(candidate, dict):
            return ""
        provider = str(candidate.get("provider", "")).strip().lower()
        provider_id = str(candidate.get("provider_id", "")).strip()
        artist = str(candidate.get("artist", "")).strip()
        title = str(candidate.get("title", "")).strip()
        duration_seconds = float(candidate.get("duration_seconds", 0.0) or 0.0)
        try:
            if provider == "qq_music" and provider_id:
                return fetch_qq_music_synced_lyrics(song_mid=provider_id, logger=self.logger)
            if provider == "netease_music" and provider_id:
                return fetch_netease_music_synced_lyrics(song_id=provider_id, logger=self.logger)
            if provider == "kugou_music" and provider_id:
                return fetch_kugou_music_synced_lyrics(
                    lyric_id=provider_id,
                    accesskey=str(candidate.get("provider_accesskey", "")).strip(),
                    logger=self.logger,
                )
            if provider.startswith("syncedlyrics"):
                from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.syncedlyrics import (
                    search_syncedlyrics_candidates,
                    search_syncedlyrics_candidates_by_provider,
                )

                if provider.startswith("syncedlyrics::"):
                    provider_name = provider.split("::", 1)[1].strip()
                    results = search_syncedlyrics_candidates_by_provider(
                        provider_name=provider_name,
                        artist=artist,
                        title=title,
                        logger=self.logger,
                        limit=1,
                    )
                else:
                    results = search_syncedlyrics_candidates(
                        artist=artist,
                        title=title,
                        logger=self.logger,
                        limit=1,
                    )
                if results:
                    return str(results[0].get("synced_lyrics", "")).strip()
            if provider == "lrclib" or (artist and title):
                result = query_lrclib_lyrics(
                    artist=artist,
                    title=title,
                    duration_seconds=duration_seconds,
                    logger=self.logger,
                )
                return str(result.get("synced_lyrics", "")).strip()
        except Exception as error:  # noqa: BLE001
            self.logger.warning(
                "[监督服务] 模块A候选歌词详情补拉失败，provider=%s，provider_id=%s，artist=%s，title=%s，错误=%s",
                provider,
                provider_id,
                artist,
                title,
                error,
            )
        return ""

    def _hydrate_module_a_candidate_detail(
        self,
        *,
        task_id: str,
        artifacts_dir: Path,
        raw_state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """
        功能说明：为候选详情补齐原文、翻译、罗马音并持久化回缓存。
        参数说明：
        - task_id: 任务ID。
        - artifacts_dir: 任务产物目录。
        - raw_state: 原始联网歌词缓存状态。
        - candidate: 当前候选对象。
        返回值：
        - dict[str, Any]: 补齐后的候选对象。
        异常说明：无；内部异常统一吞掉并返回原候选。
        边界条件：仅在缺字段时回源补拉，避免重复联网。
        """
        if not isinstance(candidate, dict):
            return {}
        enriched_candidate = dict(candidate)
        provider = str(enriched_candidate.get("provider", "")).strip().lower()
        needs_sync = not str(enriched_candidate.get("synced_lyrics", "")).strip()
        needs_translation = not str(enriched_candidate.get("translated_lyrics", "")).strip()
        needs_romanized = not str(enriched_candidate.get("romanized_lyrics", "")).strip()
        needs_word_timed = not str(enriched_candidate.get("word_timed_lyrics", "")).strip()
        if provider == "qq_music" and (needs_sync or needs_translation or needs_romanized):
            try:
                bundle = fetch_qq_music_lyrics_bundle(
                    song_mid=str(enriched_candidate.get("provider_id", "")).strip(),
                    song_id=str(enriched_candidate.get("provider_song_id", "")).strip(),
                    artist=str(enriched_candidate.get("artist", "")).strip(),
                    title=str(enriched_candidate.get("title", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选QQ富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if provider == "kugou_music" and (needs_sync or needs_word_timed or needs_translation or needs_romanized):
            try:
                bundle = fetch_kugou_music_lyrics_bundle(
                    lyric_id=str(enriched_candidate.get("provider_id", "")).strip(),
                    accesskey=str(enriched_candidate.get("provider_accesskey", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选酷狗富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_word_timed and str(bundle.get("word_timed_lyrics", "")).strip():
                enriched_candidate["word_timed_lyrics"] = str(bundle.get("word_timed_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if provider == "netease_music" and (needs_sync or needs_word_timed or needs_translation or needs_romanized):
            try:
                bundle = fetch_netease_music_lyrics_bundle(
                    song_id=str(enriched_candidate.get("provider_id", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选网易云富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_word_timed and str(bundle.get("word_timed_lyrics", "")).strip():
                enriched_candidate["word_timed_lyrics"] = str(bundle.get("word_timed_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if not str(enriched_candidate.get("synced_lyrics", "")).strip():
            synced_lyrics = self._fetch_module_a_candidate_synced_lyrics(candidate=enriched_candidate)
            if synced_lyrics:
                enriched_candidate["synced_lyrics"] = synced_lyrics
        if enriched_candidate != candidate:
            raw_candidates = raw_state.get("candidates", [])
            candidate_id = str(enriched_candidate.get("candidate_id", "")).strip()
            if isinstance(raw_candidates, list):
                for index, item in enumerate(raw_candidates):
                    if isinstance(item, dict) and str(item.get("candidate_id", "")).strip() == candidate_id:
                        raw_candidates[index] = dict(enriched_candidate)
                        break
            selected_candidate = raw_state.get("selected_candidate", {})
            if isinstance(selected_candidate, dict) and str(selected_candidate.get("candidate_id", "")).strip() == candidate_id:
                raw_state["selected_candidate"] = dict(enriched_candidate)
            write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)
        return enriched_candidate

    def _build_module_a_metadata_trace_summary(self, metadata_trace: Any) -> dict[str, Any]:
        """
        功能说明：将联网找词诊断摘要裁剪为模块A页面所需结构。
        参数说明：
        - metadata_trace: 原始诊断摘要对象。
        返回值：
        - dict[str, Any]: 前端稳定可用的摘要对象。
        异常说明：无。
        边界条件：非法输入时回退为空摘要。
        """
        if not isinstance(metadata_trace, dict):
            metadata_trace = {}
        return {
            "embedded_status": str(metadata_trace.get("embedded_status", "")).strip(),
            "embedded_source": str(metadata_trace.get("embedded_source", "")).strip(),
            "embedded_artist": str(metadata_trace.get("embedded_artist", "")).strip(),
            "embedded_title": str(metadata_trace.get("embedded_title", "")).strip(),
            "embedded_album": str(metadata_trace.get("embedded_album", "")).strip(),
            "embedded_error": str(metadata_trace.get("embedded_error", "")).strip(),
            "fingerprint_status": str(metadata_trace.get("fingerprint_status", "")).strip(),
            "fingerprint_error": str(metadata_trace.get("fingerprint_error", "")).strip(),
            "acoustid_status": str(metadata_trace.get("acoustid_status", "")).strip(),
            "matched_artist": str(metadata_trace.get("matched_artist", "")).strip(),
            "matched_title": str(metadata_trace.get("matched_title", "")).strip(),
            "matched_score": float(metadata_trace.get("matched_score", 0.0) or 0.0),
            "matched_error": str(metadata_trace.get("matched_error", "")).strip(),
        }

    def _build_module_a_candidate_summary(self, candidate: Any) -> dict[str, Any]:
        """
        功能说明：将模块 A 联网歌词候选裁剪为适合前端展示的摘要。
        参数说明：
        - candidate: 原始候选对象。
        返回值：
        - dict[str, Any]: 摘要对象。
        异常说明：无。
        边界条件：输入非法时返回空摘要。
        """
        if not isinstance(candidate, dict):
            return {
                "candidate_id": "",
                "artist": "",
                "title": "",
                "score": 0.0,
                "provider": "",
                "provider_id": "",
                "provider_song_id": "",
                "has_word_timed_lyrics": False,
                "has_translated_lyrics": False,
                "has_romanized_lyrics": False,
                "preview_lines": [],
                "preview_text": "",
            }
        preview_lines = candidate.get("preview_lines", [])
        normalized_preview_lines = [str(item).strip() for item in preview_lines if str(item).strip()] if isinstance(preview_lines, list) else []
        preview_text = str(candidate.get("preview_text", "")).strip()
        if not preview_text and normalized_preview_lines:
            preview_text = "\n".join(normalized_preview_lines)
        return {
            "candidate_id": str(candidate.get("candidate_id", "")).strip(),
            "artist": str(candidate.get("artist", "")).strip(),
            "title": str(candidate.get("title", "")).strip(),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "provider": str(candidate.get("provider", "lrclib")).strip(),
            "provider_id": str(candidate.get("provider_id", "")).strip(),
            "provider_song_id": str(candidate.get("provider_song_id", "")).strip(),
            "has_word_timed_lyrics": bool(str(candidate.get("word_timed_lyrics", "")).strip()),
            "has_translated_lyrics": bool(str(candidate.get("translated_lyrics", "")).strip()),
            "has_romanized_lyrics": bool(str(candidate.get("romanized_lyrics", "")).strip()),
            "preview_lines": normalized_preview_lines,
            "preview_text": preview_text,
        }

    def _build_module_a_provider_group_summaries(self, provider_groups: Any) -> list[dict[str, Any]]:
        """
        功能说明：将来源分组候选裁剪为适合前端展示的结构。
        参数说明：
        - provider_groups: 原始来源分组数组。
        返回值：
        - list[dict[str, Any]]: 前端稳定可用的来源分组摘要数组。
        异常说明：无。
        边界条件：非法输入时返回空数组。
        """
        if not isinstance(provider_groups, list):
            return []
        normalized_groups: list[dict[str, Any]] = []
        for provider_group in provider_groups:
            if not isinstance(provider_group, dict):
                continue
            candidates = provider_group.get("candidates", [])
            normalized_groups.append(
                {
                    "provider": str(provider_group.get("provider", "")).strip(),
                    "display_name": str(provider_group.get("display_name", "")).strip(),
                    "candidates": [
                        self._build_module_a_candidate_summary(item)
                        for item in candidates
                        if isinstance(item, dict)
                    ],
                }
            )
        return normalized_groups

    def _find_module_a_candidate_by_id(self, candidates: Any, candidate_id: str) -> dict[str, Any] | None:
        """
        功能说明：在缓存候选数组中按 candidate_id 查找目标项。
        参数说明：
        - candidates: 候选数组。
        - candidate_id: 目标候选ID。
        返回值：
        - dict[str, Any] | None: 命中返回候选对象，否则返回 None。
        异常说明：无。
        边界条件：仅接受字典数组。
        """
        if not isinstance(candidates, list):
            return None
        normalized_candidate_id = str(candidate_id).strip()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get("candidate_id", "")).strip() == normalized_candidate_id:
                return item
        return None

    async def _handle_module_a_search_lyrics_socket(self, websocket: Any, parsed: Any) -> None:
        """通过 WebSocket 实时推送模块A联网歌词搜索进度与来源结果。"""
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        manual_query = str(query.get("manual_query", [""])[0]).strip()
        self._connections.add(websocket)
        try:
            task_record = self.state_store.get_task(task_id=task_id)
            if task_record is None:
                await websocket.send(
                    json.dumps(
                        {
                            "event": "error",
                            "data": {"message": f"任务不存在：{task_id}"},
                        },
                        ensure_ascii=False,
                    )
                )
                return
            if self.app_config is None:
                await websocket.send(
                    json.dumps(
                        {
                            "event": "error",
                            "data": {"message": "当前监督服务缺少运行配置，无法联网查找歌词。"},
                        },
                        ensure_ascii=False,
                    )
                )
                return
            audio_path = self._resolve_task_audio_path_from_record(
                task_id=task_id, task_record=task_record, persist=False
            )
            duration_seconds = probe_audio_duration(
                audio_path=audio_path,
                ffprobe_bin=str(getattr(getattr(self.app_config, "ffmpeg", None), "ffprobe_bin", "ffprobe")),
                logger=self.logger,
            )
            event_loop = asyncio.get_running_loop()

            async def _send_event(event_name: str, payload: dict[str, Any]) -> None:
                await websocket.send(json.dumps({"event": event_name, "data": payload}, ensure_ascii=False))

            def _emit_event(event_name: str, payload: dict[str, Any]) -> None:
                stream_payload = payload
                if event_name == "provider_group":
                    stream_payload = self._build_module_a_stream_preview_provider_group(payload)
                elif event_name == "complete":
                    self._persist_module_a_search_result(task_id=task_id, search_result=payload)
                    stream_payload = self._build_module_a_stream_preview_result(payload)
                asyncio.run_coroutine_threadsafe(_send_event(event_name, stream_payload), event_loop)

            result = await asyncio.to_thread(
                stream_synced_lrc_candidates,
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                fpcalc_bin=str(getattr(getattr(self.app_config, "module_a", None), "fpcalc_bin", "fpcalc")),
                acoustid_api_key_file=str(
                    getattr(getattr(self.app_config, "module_a", None), "acoustid_api_key_file", "")
                ),
                logger=self.logger,
                manual_query=manual_query,
                emit_event=_emit_event,
                split_syncedlyrics_providers=True,
            )
            await asyncio.to_thread(self._persist_module_a_search_result, task_id, result)
        except ConnectionClosed:
            return
        except Exception as error:  # noqa: BLE001
            await websocket.send(
                json.dumps(
                    {
                        "event": "error",
                        "data": {"message": str(error).strip() or "module_a_search_stream_failed"},
                    },
                    ensure_ascii=False,
                )
            )
        finally:
            self._connections.discard(websocket)
            try:
                await websocket.close(code=1000, reason="module-a-search-complete")
            except Exception:  # noqa: BLE001
                pass
            if (
                self.auto_stop_on_terminal
                and self._task_terminal
                and not self._connections
                and self._async_stop_event
                and not self._async_stop_event.is_set()
            ):
                self._async_stop_event.set()
