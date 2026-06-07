"""
文件用途：审阅/Web 前端数据 handler mixin —— 构建 Web 前端负载、segment/lyric 时间线。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store、模块 B 产物解析、模块 C 关键帧记录。
维护说明：本文件仅包含审阅页数据聚合与前端静态文件管理方法。
"""

import json
import re
from pathlib import Path
from typing import Any

from music_video_pipeline.monitoring.routes import WEB_APP_STATIC_ROUTE_PREFIX


class ReviewHandlers:
    """Mixin —— 审阅页数据与 segment/lyric 时间线相关方法。"""

    def _build_web_payload(self, task_id: str) -> dict[str, Any]:
        """构建 Web 前端主页面所需的数据负载。"""
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id) or {}
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        video_path = self._resolve_output_video_path(task_dir=task_dir, task_record=task_record)
        visualization_path = self._resolve_module_a_visualization_path(task_dir=task_dir, task_id=normalized_task_id)
        segment_units = self._load_segment_units(task_dir=task_dir, task_id=normalized_task_id)
        lyric_units = self._load_lyric_units(task_dir=task_dir, review_segment_units=segment_units)
        return {
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "video": {
                "available": video_path is not None and video_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=video_path) if video_path else "",
                "path": str(video_path) if video_path else "",
            },
            "module_a_visualization": {
                "available": visualization_path is not None and visualization_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=visualization_path)
                if visualization_path
                else "",
                "path": str(visualization_path) if visualization_path else "",
            },
            "lyric_units": lyric_units,
            "segment_units": segment_units,
        }

    def _load_lyric_units(
        self,
        task_dir: Path,
        review_segment_units: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """读取歌词时间戳数组，并按审阅时间线重挂 segment_id。"""
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        try:
            payload = json.loads(module_a_output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        raw_lyric_units = payload.get("lyric_units", [])
        if not isinstance(raw_lyric_units, list):
            return []
        normalized_items: list[dict[str, Any]] = []
        for item in raw_lyric_units:
            if not isinstance(item, dict):
                continue
            token_units_payload = item.get("token_units", [])
            normalized_token_units: list[dict[str, Any]] = []
            if isinstance(token_units_payload, list):
                for token_item in token_units_payload:
                    if not isinstance(token_item, dict):
                        continue
                    normalized_token_units.append(
                        {
                            "text": str(token_item.get("text", "")),
                            "start_time": float(token_item.get("start_time", 0.0)),
                            "end_time": float(token_item.get("end_time", float(token_item.get("start_time", 0.0)))),
                        }
                    )
            normalized_items.append(
                {
                    "segment_id": str(item.get("segment_id", "")),
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "text": str(item.get("text", "")),
                    "confidence": float(item.get("confidence", 0.0)),
                    "token_units": normalized_token_units,
                }
            )
        if not review_segment_units:
            return normalized_items
        return self._reattach_lyric_units_to_review_segments(
            lyric_units=normalized_items,
            review_segment_units=review_segment_units,
        )

    def _load_segment_units(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """构建审阅页 segment 数组；优先使用当前成片对应的模块D时间线。"""
        module_d_segments = self._load_segment_units_from_module_d(task_dir=task_dir, task_id=task_id)
        if module_d_segments:
            return module_d_segments
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        payload = self._load_json_file(module_a_output_path)
        if not isinstance(payload, dict):
            return []
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list):
            return []
        segment_item_map = self._load_module_b_segment_item_map(task_dir=task_dir, task_id=task_id)
        frame_item_map = self._load_frame_item_map_by_shot_id(task_dir=task_dir, task_id=task_id)
        normalized_items: list[dict[str, Any]] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id", "")).strip()
            segment_item = segment_item_map.get(segment_id, {})
            shot_id = str(segment_item.get("shot_id", "")).strip()
            frame_item = frame_item_map.get(shot_id, {}) if shot_id else {}
            normalized_items.append(
                {
                    "segment_id": segment_id,
                    "big_segment_id": str(item.get("big_segment_id", "")),
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "label": str(item.get("label", "")),
                    "role": str(item.get("role", "")),
                    "scene_desc": str(segment_item.get("scene_desc", "")).strip(),
                    "shot_id": shot_id,
                    "camera_plan": segment_item.get("camera_plan", {}),
                    "keyframe_prompt_start_zh": str(frame_item.get("keyframe_prompt_start_zh", "") or segment_item.get("keyframe_prompt_start_zh", "")).strip(),
                    "keyframe_prompt_start_en": str(frame_item.get("keyframe_prompt_start_en", "") or segment_item.get("keyframe_prompt_start_en", "")).strip(),
                    "keyframe_prompt_end_zh": str(frame_item.get("keyframe_prompt_end_zh", "") or segment_item.get("keyframe_prompt_end_zh", "")).strip(),
                    "keyframe_prompt_end_en": str(frame_item.get("keyframe_prompt_end_en", "") or segment_item.get("keyframe_prompt_end_en", "")).strip(),
                    "video_prompt_zh": str(frame_item.get("video_prompt_zh", "") or segment_item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(frame_item.get("video_prompt_en", "") or segment_item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": str(frame_item.get("frame_path_start", "")).strip(),
                    "frame_path_end": str(frame_item.get("frame_path_end", "")).strip(),
                    "frame_url_start": str(frame_item.get("frame_url_start", "")).strip(),
                    "frame_url_end": str(frame_item.get("frame_url_end", "")).strip(),
                }
            )
        return normalized_items

    def _load_segment_units_from_module_d(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """从模块D标准输出恢复与当前成片一致的审阅页 segment 数组。"""
        module_d_output_path = task_dir / "artifacts" / "module_d_output.json"
        payload = self._load_json_file(module_d_output_path)
        raw_segment_items = payload.get("segment_items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_segment_items, list):
            return []
        frame_item_map = self._load_frame_item_map_by_shot_id(task_dir=task_dir, task_id=task_id)
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_segment_items):
            if not isinstance(item, dict):
                continue
            shot_id = str(item.get("shot_id", "")).strip()
            frame_item = frame_item_map.get(shot_id, {}) if shot_id else {}
            unit_index_raw = item.get("unit_index", index)
            try:
                unit_index = max(0, int(unit_index_raw))
            except (TypeError, ValueError):
                unit_index = max(0, index)
            segment_id = self._build_review_segment_id(
                explicit_segment_id=str(item.get("segment_id", "")).strip(),
                unit_index=unit_index,
                shot_id=shot_id,
                fallback_index=index,
            )
            segment_path = self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(item.get("segment_path", "")),
            )
            segment_frame_path_start, segment_frame_path_end = self._resolve_module_d_segment_frame_paths(
                task_id=task_id,
                shot_id=shot_id,
                segment_path=segment_path,
            )
            resolved_frame_path_start = segment_frame_path_start or self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(frame_item.get("frame_path_start", "")),
            )
            resolved_frame_path_end = segment_frame_path_end or self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(frame_item.get("frame_path_end", "")),
            )
            normalized_items.append(
                {
                    "segment_id": segment_id,
                    "big_segment_id": "",
                    "start_time": float(item.get("start_time", frame_item.get("start_time", 0.0))),
                    "end_time": float(item.get("end_time", frame_item.get("end_time", item.get("start_time", 0.0)))),
                    "label": "",
                    "role": "",
                    "scene_desc": str(item.get("scene_desc", "") or frame_item.get("scene_desc", "")).strip(),
                    "shot_id": shot_id,
                    "camera_plan": item.get("camera_plan", frame_item.get("camera_plan", {})),
                    "keyframe_prompt_start_zh": str(
                        item.get("keyframe_prompt_start_zh", "") or frame_item.get("keyframe_prompt_start_zh", "")
                    ).strip(),
                    "keyframe_prompt_start_en": str(
                        item.get("keyframe_prompt_start_en", "") or frame_item.get("keyframe_prompt_start_en", "")
                    ).strip(),
                    "keyframe_prompt_end_zh": str(
                        item.get("keyframe_prompt_end_zh", "") or frame_item.get("keyframe_prompt_end_zh", "")
                    ).strip(),
                    "keyframe_prompt_end_en": str(
                        item.get("keyframe_prompt_end_en", "") or frame_item.get("keyframe_prompt_end_en", "")
                    ).strip(),
                    "video_prompt_zh": str(item.get("video_prompt_zh", "") or frame_item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(item.get("video_prompt_en", "") or frame_item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": str(resolved_frame_path_start) if resolved_frame_path_start else "",
                    "frame_path_end": str(resolved_frame_path_end) if resolved_frame_path_end else "",
                    "frame_url_start": self._build_task_file_url(task_id=task_id, file_path=resolved_frame_path_start)
                    if resolved_frame_path_start
                    else "",
                    "frame_url_end": self._build_task_file_url(task_id=task_id, file_path=resolved_frame_path_end)
                    if resolved_frame_path_end
                    else "",
                }
            )
        return normalized_items

    def _resolve_module_d_segment_frame_paths(
        self,
        task_id: str,
        shot_id: str,
        segment_path: Path | None,
    ) -> tuple[Path | None, Path | None]:
        """优先从模块D段视频对应的逐帧目录中定位真实首尾帧。"""
        task_dir = self._resolve_task_dir(task_id=task_id)
        candidate_dirs: list[Path] = []
        if segment_path is not None:
            candidate_dirs.append((segment_path.parent / f".{str(shot_id).strip()}_frames").resolve())
        if str(shot_id).strip():
            candidate_dirs.append((task_dir / "artifacts" / "segments" / f".{str(shot_id).strip()}_frames").resolve())

        checked_dirs: set[Path] = set()
        for candidate_dir in candidate_dirs:
            if candidate_dir in checked_dirs:
                continue
            checked_dirs.add(candidate_dir)
            try:
                candidate_dir.relative_to(task_dir)
            except ValueError:
                continue
            if (not candidate_dir.exists()) or (not candidate_dir.is_dir()):
                continue
            frame_paths = sorted(
                [
                    file_path
                    for file_path in candidate_dir.iterdir()
                    if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                ],
                key=lambda file_path: file_path.name,
            )
            if not frame_paths:
                continue
            return frame_paths[0], frame_paths[-1]
        return None, None

    def _reattach_lyric_units_to_review_segments(
        self,
        lyric_units: list[dict[str, Any]],
        review_segment_units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """按当前审阅时间线重挂歌词 segment_id，避免旧成片混入新 A/B 的 segment 标识。"""
        if not lyric_units or not review_segment_units:
            return lyric_units
        normalized_items: list[dict[str, Any]] = []
        for item in lyric_units:
            lyric_start_time = float(item.get("start_time", 0.0))
            matched_segment = self._find_review_segment_for_time(
                review_segment_units=review_segment_units,
                current_time=lyric_start_time,
            )
            normalized_item = dict(item)
            if matched_segment:
                normalized_item["segment_id"] = str(matched_segment.get("segment_id", "")).strip()
            normalized_items.append(normalized_item)
        return normalized_items

    def _find_review_segment_for_time(
        self,
        review_segment_units: list[dict[str, Any]],
        current_time: float,
    ) -> dict[str, Any] | None:
        """按左闭右开规则查找给定时刻对应的审阅 segment。"""
        if (not review_segment_units) or (not isinstance(current_time, (int, float))):
            return None
        last_index = len(review_segment_units) - 1
        for index, item in enumerate(review_segment_units):
            try:
                start_time = float(item.get("start_time", 0.0))
                end_time = max(start_time, float(item.get("end_time", start_time)))
            except (TypeError, ValueError):
                continue
            is_last_item = index == last_index
            if current_time >= start_time and (current_time < end_time or (is_last_item and current_time <= end_time)):
                return item
        return None

    def _build_review_segment_id(
        self,
        explicit_segment_id: str,
        unit_index: int,
        shot_id: str,
        fallback_index: int,
    ) -> str:
        """为审阅页生成稳定的 seg_xxxx 标识。"""
        normalized_explicit_segment_id = str(explicit_segment_id).strip()
        if normalized_explicit_segment_id:
            return normalized_explicit_segment_id
        if unit_index >= 0:
            return f"seg_{unit_index + 1:04d}"
        matched = re.search(r"(\d+)", str(shot_id))
        if matched is not None:
            return f"seg_{int(matched.group(1)):04d}"
        return f"seg_{fallback_index + 1:04d}"

    def _load_module_b_segment_item_map(self, task_dir: Path, task_id: str) -> dict[str, dict[str, Any]]:
        """按 segment_id 加载模块B产物，供审阅页读取 scene_desc 与 prompt 数据。"""
        item_map: dict[str, dict[str, Any]] = {}
        try:
            state_rows = self.state_store.list_module_b_done_shot_items(task_id=task_id)
        except Exception:  # noqa: BLE001
            state_rows = []
        for row in state_rows:
            segment_id = str(row.get("unit_id", "")).strip()
            if not segment_id:
                continue
            artifact_path = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(row.get("artifact_path", "")))
            if artifact_path is None:
                continue
            payload = self._load_json_file(artifact_path)
            normalized_item = self._normalize_module_b_segment_item(payload=payload, segment_id=segment_id)
            if normalized_item:
                item_map[segment_id] = normalized_item

        module_b_units_dir = task_dir / "artifacts" / "module_b_units"
        if not module_b_units_dir.exists():
            return item_map
        for payload_path in sorted(module_b_units_dir.glob("*.json")):
            segment_id = self._derive_segment_id_from_name(name=payload_path.stem)
            if (not segment_id) or (segment_id in item_map):
                continue
            payload = self._load_json_file(payload_path)
            normalized_item = self._normalize_module_b_segment_item(payload=payload, segment_id=segment_id)
            if normalized_item:
                item_map[segment_id] = normalized_item
        return item_map

    def _normalize_module_b_segment_item(self, payload: Any, segment_id: str) -> dict[str, Any]:
        """把模块B单元载荷收敛成审阅页需要的最小字段集合。"""
        if not isinstance(payload, dict):
            return {}
        normalized_segment_id = str(payload.get("segment_id", "")).strip() or str(segment_id).strip()
        if not normalized_segment_id:
            return {}
        camera_plan_payload = payload.get("camera_plan", {})
        return {
            "segment_id": normalized_segment_id,
            "shot_id": str(payload.get("shot_id", "")).strip(),
            "scene_desc": str(payload.get("scene_desc", "")).strip(),
            "camera_plan": camera_plan_payload if isinstance(camera_plan_payload, dict) else {},
            "keyframe_prompt_start_zh": str(payload.get("keyframe_prompt_start_zh", "")).strip(),
            "keyframe_prompt_start_en": str(payload.get("keyframe_prompt_start_en", "")).strip(),
            "keyframe_prompt_end_zh": str(payload.get("keyframe_prompt_end_zh", "")).strip(),
            "keyframe_prompt_end_en": str(payload.get("keyframe_prompt_end_en", "")).strip(),
            "video_prompt_zh": str(payload.get("video_prompt_zh", "")).strip(),
            "video_prompt_en": str(payload.get("video_prompt_en", "")).strip(),
        }

    def _load_frame_item_map_by_shot_id(self, task_dir: Path, task_id: str) -> dict[str, dict[str, Any]]:
        """按 shot_id 加载模块C关键帧产物，并补齐前端可直接访问的 URL。"""
        item_map: dict[str, dict[str, Any]] = {}
        try:
            state_frame_items = self.state_store.list_module_c_done_frame_items(task_id=task_id)
        except Exception:  # noqa: BLE001
            state_frame_items = []
        for item in state_frame_items:
            normalized_item = self._normalize_frame_item(task_id=task_id, payload=item)
            shot_id = str(normalized_item.get("shot_id", "")).strip()
            if shot_id:
                item_map[shot_id] = normalized_item

        module_c_output_path = task_dir / "artifacts" / "module_c_output.json"
        payload = self._load_json_file(module_c_output_path)
        raw_frame_items = payload.get("frame_items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_frame_items, list):
            return item_map
        for item in raw_frame_items:
            normalized_item = self._normalize_frame_item(task_id=task_id, payload=item)
            shot_id = str(normalized_item.get("shot_id", "")).strip()
            if not shot_id:
                continue
            existing_item = item_map.get(shot_id)
            if existing_item is None:
                item_map[shot_id] = normalized_item
                continue
            for field_name in ("frame_path", "frame_path_start", "frame_path_end", "frame_url_start", "frame_url_end"):
                if (not str(existing_item.get(field_name, "")).strip()) and str(normalized_item.get(field_name, "")).strip():
                    existing_item[field_name] = normalized_item[field_name]
        return item_map

    def _normalize_frame_item(self, task_id: str, payload: Any) -> dict[str, Any]:
        """把模块C关键帧记录规整成审阅页需要的路径与URL字段。"""
        if not isinstance(payload, dict):
            return {}
        shot_id = str(payload.get("shot_id", "")).strip()
        if not shot_id:
            return {}
        frame_path = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path", "")))
        frame_path_start = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path_start", "")))
        frame_path_end = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path_end", "")))
        normalized_frame_path_start = frame_path_start or frame_path
        normalized_frame_path_end = frame_path_end or frame_path or frame_path_start
        normalized_frame_path = frame_path or normalized_frame_path_start or normalized_frame_path_end
        return {
            "shot_id": shot_id,
            "frame_path": str(normalized_frame_path) if normalized_frame_path else "",
            "frame_path_start": str(normalized_frame_path_start) if normalized_frame_path_start else "",
            "frame_path_end": str(normalized_frame_path_end) if normalized_frame_path_end else "",
            "frame_url_start": self._build_task_file_url(task_id=task_id, file_path=normalized_frame_path_start)
            if normalized_frame_path_start
            else "",
            "frame_url_end": self._build_task_file_url(task_id=task_id, file_path=normalized_frame_path_end)
            if normalized_frame_path_end
            else "",
        }

    def _resolve_task_artifact_path(self, task_id: str, raw_path: str) -> Path | None:
        """把产物中记录的绝对或相对路径回映射到当前任务目录下的真实文件。"""
        normalized_raw_path = str(raw_path).strip()
        if not normalized_raw_path:
            return None
        task_dir = self._resolve_task_dir(task_id=task_id)
        raw_candidate = Path(normalized_raw_path)
        candidate_paths: list[Path] = []
        if not raw_candidate.is_absolute():
            candidate_paths.append((task_dir / raw_candidate).resolve())
        else:
            candidate_paths.append(raw_candidate.resolve())

        raw_parts = [str(part) for part in raw_candidate.parts if str(part).strip()]
        normalized_task_id = str(task_id).strip()
        if normalized_task_id in raw_parts:
            task_index = raw_parts.index(normalized_task_id)
            candidate_paths.append(task_dir.joinpath(*raw_parts[task_index + 1:]).resolve())
        if "artifacts" in raw_parts:
            artifacts_index = max(index for index, part_text in enumerate(raw_parts) if part_text == "artifacts")
            candidate_paths.append(task_dir.joinpath("artifacts", *raw_parts[artifacts_index + 1:]).resolve())

        for candidate_path in candidate_paths:
            try:
                candidate_path.relative_to(task_dir)
            except ValueError:
                continue
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path
        return None

    @staticmethod
    def _derive_segment_id_from_name(name: str) -> str:
        """从模块B单元文件名中提取标准 segment_id。"""
        matched = re.search(r"(seg_\d+)", str(name))
        if matched is None:
            return ""
        return str(matched.group(1)).strip()
