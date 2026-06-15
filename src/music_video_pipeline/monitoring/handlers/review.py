"""
文件用途：审阅/Web 前端数据 handler mixin —— 构建 Web 前端负载、segment/lyric 时间线。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store、模块 B 流式产物（role3/role4 streaming）、frames 目录。
维护说明：本文件仅包含审阅页数据聚合方法。
"""

import json
import re
from pathlib import Path
from typing import Any


class ReviewHandlers:
    """Mixin —— 审阅页数据与 segment/lyric 时间线相关方法。"""

    def _build_web_payload(self, task_id: str) -> dict[str, Any]:
        """构建 Web 前端主页面所需的数据负载。"""
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id) or {}
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        video_path = self._resolve_output_video_path(task_dir=task_dir, task_record=task_record)
        segment_units = self._load_segment_units_from_streaming(task_dir=task_dir, task_id=normalized_task_id)
        lyric_units = self._load_lyric_units(task_dir=task_dir, review_segment_units=segment_units)
        return {
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "video": {
                "available": video_path is not None and video_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=video_path) if video_path else "",
                "path": str(video_path) if video_path else "",
            },
            "module_a_visualization": {"available": False, "url": "", "path": ""},
            "lyric_units": lyric_units,
            "segment_units": segment_units,
        }

    # ── 新方法：从流式文件 + frames 目录构建 segment_units ──

    def _load_segment_units_from_streaming(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """从模块 B 流式文件 + frames 目录构建审阅页 segment 数组（shot 粒度展开）。"""
        module_a_segments = self._read_module_a_segments(task_dir)
        role3_map = self._parse_role3_streaming(task_dir)
        role4_map = self._parse_role4_streaming(task_dir)
        frames_map = self._scan_frames_dir(task_dir, task_id)

        shot_prefix_index = self._build_shot_prefix_index(list(role4_map.keys()))

        normalized_items: list[dict[str, Any]] = []
        for segment in module_a_segments:
            segment_id = str(segment.get("segment_id", "")).strip()
            seg_number = self._extract_seg_number(segment_id)

            big_segment_id = str(segment.get("big_segment_id", "")).strip()
            role3_data = role3_map.get(segment_id, {})
            if not big_segment_id:
                big_segment_id = str(role3_data.get("big_segment_id", "")).strip()

            scene_desc = str(role3_data.get("scene_desc", "") or segment.get("scene_desc", "")).strip()
            remotion_id = str(role3_data.get("remotion_id", "")).strip()

            shot_prefix = f"shot_{seg_number}_" if seg_number else ""
            shot_ids = shot_prefix_index.get(shot_prefix, [])
            if not shot_ids:
                shot_ids = [f"shot_{seg_number}_1"] if seg_number else []

            for shot_id in shot_ids:
                role4_data = role4_map.get(shot_id, {})
                frame_data = frames_map.get(shot_id, {})
                normalized_items.append({
                    "segment_id": segment_id,
                    "big_segment_id": big_segment_id,
                    "start_time": float(segment.get("start_time", 0.0)),
                    "end_time": float(segment.get("end_time", float(segment.get("start_time", 0.0)))),
                    "label": str(segment.get("label", "")),
                    "role": str(segment.get("role", "")),
                    "scene_desc": scene_desc,
                    "shot_id": shot_id,
                    "camera_plan": {"remotion_id": remotion_id} if remotion_id else {},
                    "keyframe_prompt_start_zh": str(role4_data.get("keyframe_prompt_start_zh", "")).strip(),
                    "keyframe_prompt_start_en": str(role4_data.get("keyframe_prompt_start_en", "")).strip(),
                    "keyframe_prompt_end_zh": str(role4_data.get("keyframe_prompt_end_zh", "")).strip(),
                    "keyframe_prompt_end_en": str(role4_data.get("keyframe_prompt_end_en", "")).strip(),
                    "video_prompt_zh": str(role4_data.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(role4_data.get("video_prompt_en", "")).strip(),
                    "frame_path_start": "",
                    "frame_path_end": "",
                    "frame_url_start": str(frame_data.get("frame_url_start", "")).strip(),
                    "frame_url_end": str(frame_data.get("frame_url_end", "")).strip(),
                })
        return normalized_items

    @staticmethod
    def _read_module_a_segments(task_dir: Path) -> list[dict[str, Any]]:
        """读取 module_a_output.json 中的 segments 数组（基础时间信息）。"""
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        try:
            payload = json.loads(module_a_output_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list):
            return []
        return [item for item in raw_segments if isinstance(item, dict)]

    def _parse_role3_streaming(self, task_dir: Path) -> dict[str, dict[str, Any]]:
        """解析 Role3 流式文件，按 segment_id 返回 scene_desc / remotion_id / big_segment_id。"""
        streaming_dir = task_dir / "artifacts" / "module_b_work" / "role3" / "streaming"
        if not streaming_dir.exists():
            return {}

        result: dict[str, dict[str, Any]] = {}
        for file_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            big_id_match = re.search(r"big_\d+", file_path.stem)
            big_segment_id = big_id_match.group(0) if big_id_match else ""
            content = file_path.read_text(encoding="utf-8")

            seg_sections = re.split(r"\n(?=###\s+seg_\d+)", content)
            for section in seg_sections:
                seg_match = re.search(r"###\s+(seg_\d+)", section)
                if not seg_match:
                    continue
                segment_id = seg_match.group(1)
                segment_data: dict[str, Any] = {"big_segment_id": big_segment_id}

                for line in section.strip().split("\n"):
                    line = line.strip()
                    field_match = re.match(r"^-\s+(\w+):\s*(.*)", line)
                    if field_match:
                        field = field_match.group(1)
                        value = field_match.group(2).strip()
                        if field == "scene_desc_zh":
                            segment_data["scene_desc"] = value
                        elif field == "remotion_id":
                            segment_data["remotion_id"] = value
                        elif field == "shot_subject_kind":
                            segment_data["shot_subject_kind"] = value

                if segment_data.get("scene_desc") or segment_data.get("remotion_id"):
                    result[segment_id] = segment_data

        return result

    def _parse_role4_streaming(self, task_dir: Path) -> dict[str, dict[str, Any]]:
        """解析 Role4 流式文件，按 shot_id 返回 prompts。"""
        streaming_dir = task_dir / "artifacts" / "module_b_work" / "role4" / "streaming"
        if not streaming_dir.exists():
            return {}

        result: dict[str, dict[str, Any]] = {}
        for file_path in streaming_dir.glob("role4_prompt_output.streaming.*.md"):
            shot_id_match = re.search(r"shot_\d+_\d+", file_path.stem)
            if not shot_id_match:
                continue
            shot_id = shot_id_match.group(0)
            content = file_path.read_text(encoding="utf-8")

            shot_data: dict[str, str] = {}
            for line in content.strip().split("\n"):
                line = line.strip()
                field_match = re.match(r"^-\s+(\w+):\s*(.*)", line)
                if field_match:
                    field = field_match.group(1)
                    value = field_match.group(2).strip()
                    if field in {
                        "keyframe_prompt_start_zh", "keyframe_prompt_start_en",
                        "keyframe_prompt_end_zh", "keyframe_prompt_end_en",
                        "video_prompt_zh", "video_prompt_en",
                    }:
                        shot_data[field] = value

            if shot_data:
                result[shot_id] = shot_data

        return result

    def _scan_frames_dir(self, task_dir: Path, task_id: str) -> dict[str, dict[str, Any]]:
        """扫描 artifacts/frames/ 目录，按 shot_id 返回首尾帧 URL。"""
        frames_dir = task_dir / "artifacts" / "frames"
        if not frames_dir.exists():
            return {}

        result: dict[str, dict[str, Any]] = {}
        for file_path in sorted(frames_dir.iterdir()):
            if not file_path.is_file():
                continue
            match = re.match(
                r"(shot_\d+_\d+)_(start|end)\.(png|jpg|jpeg|webp)$",
                file_path.name.lower(),
            )
            if not match:
                continue
            shot_id = match.group(1)
            frame_type = match.group(2)

            if shot_id not in result:
                result[shot_id] = {"frame_url_start": "", "frame_url_end": ""}

            url = self._build_task_file_url(task_id=task_id, file_path=file_path)
            if frame_type == "start":
                result[shot_id]["frame_url_start"] = url
            elif frame_type == "end":
                result[shot_id]["frame_url_end"] = url

        return result

    @staticmethod
    def _build_shot_prefix_index(shot_ids: list[str]) -> dict[str, list[str]]:
        """按 shot_XXXX_ 前缀索引 shot_id 列表。"""
        prefix_index: dict[str, list[str]] = {}
        for shot_id in shot_ids:
            match = re.match(r"(shot_\d+_)", shot_id)
            prefix = match.group(1) if match else shot_id
            prefix_index.setdefault(prefix, []).append(shot_id)
        for prefix in prefix_index:
            prefix_index[prefix].sort()
        return prefix_index

    @staticmethod
    def _extract_seg_number(segment_id: str) -> str:
        """从 segment_id 中提取纯数字部分。"""
        match = re.search(r"(\d+)", segment_id)
        return match.group(1) if match else ""

    # ── 保留方法：歌词处理 ──

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
        except Exception:
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
