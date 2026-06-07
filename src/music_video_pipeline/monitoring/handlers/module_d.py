"""
文件用途：模块 D handler mixin —— 构建按 big_segment 分组的页面负载与 segment 重跑。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store、项目内路径工具与 Remotion 渲染模块。
维护说明：本文件仅包含模块 D 专属方法，不引入其他模块的耦合。
"""

import json
import re as _re
import subprocess
import time
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_streaming_dir
from music_video_pipeline.monitoring.routes import (
    TASK_MODULE_D_API_PATH,
    TASK_MODULE_D_RERUN_SEGMENT_API_PATH,
    TASK_MODULE_D_RERUN_BOTH_FRAMES_API_PATH,
    TASK_MODULE_D_RERUN_MODULE_API_PATH,
)

# ========== 模板注册（唯一入口）==========
# 键名即 composition_id，直接用于 Remotion 渲染；新增模板只需在此加一条
_REMOTION_TEMPLATES: dict[str, dict[str, str]] = {
    "CenterTemplate":    {"short": "center",    "category": "single"},
    "GridTemplate":      {"short": "grid",      "category": "multi"},
    "ScrollTemplate":    {"short": "scroll",    "category": "multi"},
    "TiltUpTemplate":    {"short": "tilt_up",   "category": "transition"},
    "TiltDownTemplate":  {"short": "tilt_down", "category": "transition"},
    "PanRightTemplate":  {"short": "pan_right", "category": "transition"},
}

_TRANSITION_TEMPLATES = frozenset(
    k for k, v in _REMOTION_TEMPLATES.items() if v["category"] == "transition"
)
_MULTI_SUBJECT_TEMPLATES = frozenset(
    k for k, v in _REMOTION_TEMPLATES.items() if v["category"] == "multi"
)

# 透明 1×1 GIF，用于白屏/黑屏过渡时占位 symbol（不可见）
_TRANSPARENT_PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def _build_module_d_rerun_key(task_id: str, ref_id: str) -> str:
    """构建模块 D 重跑线程唯一键。"""
    return f"{str(task_id).strip()}|module_d|segment|{str(ref_id).strip()}"


class ModuleDHandlers:
    """Mixin —— 模块 D 相关方法。"""

    # ------------------------------------------------------------------
    # 角色 3 流式文件解析
    # ------------------------------------------------------------------

    def _load_role3_big_segment_metas(self, task_dir: Path) -> dict[str, dict[str, str]]:
        """
        功能说明：从 role3 流式文件读取每个 big_segment 的 remotion_id 和 scene_desc_zh。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, dict[str, str]]: {big_segment_id: {remotion_id, scene_desc_zh}}。
        异常说明：无；缺失 role3 流式产物时返回空字典。
        """
        artifacts_dir = (task_dir / "artifacts").resolve()
        streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role3")
        result: dict[str, dict[str, str]] = {}
        if not streaming_dir.exists():
            return result

        for stream_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            except Exception:
                continue
            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()
            first_seg_desc = ""

            for block in _re.split(r"\n(?=### )", text):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n")
                heading = lines[0].strip()
                if heading.startswith("## "):
                    current_big = heading[3:].strip().split(" / ")[0].strip()
                    continue
                if not heading.startswith("### "):
                    continue

                seg_desc = ""
                remotion_id = ""
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith("- scene_desc_zh:"):
                        seg_desc = stripped[len("- scene_desc_zh:"):].strip()
                    elif stripped.startswith("- remotion_id:"):
                        remotion_id = stripped[len("- remotion_id:"):].strip()

                if current_big not in result:
                    result[current_big] = {
                        "remotion_id": remotion_id,
                        "scene_desc_zh": seg_desc,
                    }
                if not first_seg_desc and seg_desc:
                    first_seg_desc = seg_desc

            if current_big not in result:
                result[current_big] = {
                    "remotion_id": "",
                    "scene_desc_zh": first_seg_desc,
                }

        return result

    def _load_role3_seg_details(self, task_dir: Path) -> dict[str, dict[str, Any]]:
        """
        功能说明：从 role3 流式文件加载每个 big 的完整 seg 列表。
        返回值：{big_id: {remotion_id: str, segs: [{seg_id, scene_desc_zh}]}}
        """
        artifacts_dir = (task_dir / "artifacts").resolve()
        streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role3")
        result: dict[str, dict[str, Any]] = {}
        if not streaming_dir.exists():
            return result

        for stream_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            except Exception:
                continue

            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()
            segs: list[dict[str, str]] = []
            remotion_id = ""

            for block in _re.split(r"\n(?=### )", text):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n")
                heading = lines[0].strip()

                if heading.startswith("## "):
                    current_big = heading[3:].strip().split(" / ")[0].strip()
                    continue
                if not heading.startswith("### "):
                    continue

                seg_id = heading[4:].strip()
                seg_desc = ""
                rid = ""
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith("- scene_desc_zh:"):
                        seg_desc = stripped[len("- scene_desc_zh:"):].strip()
                    elif stripped.startswith("- remotion_id:"):
                        rid = stripped[len("- remotion_id:"):].strip()

                segs.append({"seg_id": seg_id, "scene_desc_zh": seg_desc, "remotion_id": rid})
                if not remotion_id and rid:
                    remotion_id = rid

            if segs:
                result[current_big] = {"remotion_id": remotion_id, "segs": segs}

        return result

    def _build_seg_to_big_mapping(self, task_dir: Path) -> dict[str, str]:
        """
        功能说明：从 role3 流式文件构建 seg_id → big_segment_id 映射。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, str]: {seg_id: big_segment_id}。
        """
        artifacts_dir = (task_dir / "artifacts").resolve()
        streaming_dir = get_module_b_streaming_dir(artifacts_dir, "role3")
        mapping: dict[str, str] = {}
        if not streaming_dir.exists():
            return mapping

        for stream_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            except Exception:
                continue
            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()

            for block in _re.split(r"\n(?=### )", text):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n")
                heading = lines[0].strip()
                if heading.startswith("## "):
                    current_big = heading[3:].strip().split(" / ")[0].strip()
                    continue
                if not heading.startswith("### "):
                    continue
                seg_id = heading[4:].strip()
                if seg_id:
                    mapping[seg_id] = current_big

        return mapping

    @staticmethod
    def _shot_id_to_seg_id(shot_id: str) -> str:
        """从 shot_id（如 shot_0001_1）反推 seg_id（如 seg_0001）。"""
        m = _re.match(r'^shot_(\d+)_\d+$', str(shot_id).strip())
        return f"seg_{m.group(1)}" if m else ""

    @staticmethod
    def _count_subjects_for_seg(scene_desc_zh: str, remotion_id: str) -> int:
        """根据 scene_desc_zh 和 remotion_id 确定一个 seg 应生成的 shot 数量。"""
        desc = str(scene_desc_zh or "").strip()
        if not desc:
            return 1
        rid = str(remotion_id or "").strip()
        if rid in _MULTI_SUBJECT_TEMPLATES:
            m = _re.search(r'出现(.+)', desc)
            if m:
                part = m.group(1)
                part = _re.split(r'(?:，|,|；|;)?\s*(?:背景|场景|环境)\s*(?:为|是|：|:)', part, maxsplit=1)[0]
                part = _re.sub(r'[。；;]$', '', part).strip()
                subjects = _re.split(r'[、，,]', part)
                subjects = [s.strip() for s in subjects if s.strip()]
                return max(len(subjects), 1)
        return 1

    @staticmethod
    def _build_shot_id(segment_id: str, subject_index: int) -> str:
        """从 segment_id（如 seg_0001）和主体序号（1-based）构建 shot_id（如 shot_0001_1）。"""
        seg_number = str(segment_id).strip().replace("seg_", "")
        return f"shot_{seg_number}_{int(subject_index)}"

    # ------------------------------------------------------------------
    # 页面负载构建
    # ------------------------------------------------------------------

    def _build_module_d_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 D 页面所需的数据负载，按 big_segment 分组。
        从 role3 流式文件（模块 B 产物）构建 big_segment → seg → shot 层次结构，
        state_store 与 module_d_output.json 仅补充状态和元数据。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 含 big_segments 分组与最终视频 URL 的数据对象。
        异常说明：无；任务不存在时返回 ok=false；role3 数据不存在时 big_segments 为空。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
                "module_d_status": "unknown",
                "unit_summary": self._build_empty_module_unit_summary(module_name="D"),
                "big_segments": [],
                "output_video_url": "",
                "active_rerun": None,
            }

        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:
            module_status_map = {}
        try:
            unit_summary = self.state_store.get_module_unit_status_summary(
                task_id=normalized_task_id, module_name="D",
            )
        except Exception:
            unit_summary = self._build_empty_module_unit_summary(module_name="D")

        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        artifacts_dir = task_dir / "artifacts"

        # --- PRIMARY：role3 流式文件构建 big_segment 层次 ---
        role3_details = self._load_role3_seg_details(task_dir=task_dir)

        # --- module_d_output.json 补充输出视频 URL ---
        module_d_output_path = artifacts_dir / "module_d_output.json"
        output_video_url = ""

        if module_d_output_path.exists():
            try:
                d_output = json.loads(module_d_output_path.read_text(encoding="utf-8"))
                if isinstance(d_output, dict):
                    output_video_path = str(d_output.get("output_video_path", "")).strip()
                    if output_video_path:
                        p = Path(output_video_path)
                        if p.exists():
                            output_video_url = self._build_task_file_url(
                                task_id=normalized_task_id, file_path=p,
                            )
                            output_video_url += f"?t={int(p.stat().st_mtime)}"
            except Exception:
                pass

        # 回退找 final_output.mp4
        if not output_video_url:
            for candidate in [task_dir / "final_output.mp4", artifacts_dir / "final_output.mp4"]:
                if candidate.exists():
                    output_video_url = self._build_task_file_url(
                        task_id=normalized_task_id, file_path=candidate,
                    )
                    output_video_url += f"?t={int(candidate.stat().st_mtime)}"
                    break

        # --- state_store 单元状态 ---
        all_units = self.state_store.list_module_units_by_status(
            task_id=task_id, module_name="D",
            statuses=["pending", "running", "done", "failed"],
        )
        unit_status_map: dict[str, dict[str, Any]] = {}
        segment_unit_map: dict[str, dict[str, Any]] = {}
        for unit in all_units:
            uid = str(unit.get("unit_id", "")).strip()
            if uid:
                unit_status_map[uid] = unit
            segid = str(unit.get("segment_id", "")).strip()
            if segid:
                segment_unit_map[segid] = unit

        frames_dir = artifacts_dir / "frames"

        def _video_url(seg_id: str) -> str:
            """从 segments 目录取对应 segment 的视频文件 URL。"""
            seg_match = _re.search(r"(\d+)", seg_id)
            if seg_match:
                seg_num = int(seg_match.group(1))
                mp4 = artifacts_dir / "segments" / f"segment_{seg_num:03d}.mp4"
                if mp4.exists():
                    try:
                        url = self._build_task_file_url(task_id=normalized_task_id, file_path=mp4)
                        return url + f"?t={int(mp4.stat().st_mtime)}"
                    except Exception:
                        pass
            return ""

        def _keyframe_url(shot_id: str, frame_type: str) -> str:
            """获取 shot 的 keyframe 图片 URL（frame_type: start/end）。"""
            fp = frames_dir / f"{shot_id}_{frame_type}.png"
            if fp.exists():
                try:
                    url = self._build_task_file_url(task_id=normalized_task_id, file_path=fp)
                    return url + f"?t={int(fp.stat().st_mtime)}"
                except Exception:
                    pass
            return ""

        def _probe_video_duration(seg_id: str) -> float:
            """用 ffprobe 读取 segment 视频文件的实际时长（秒），失败返回 0。"""
            seg_match = _re.search(r"(\d+)", seg_id)
            if not seg_match:
                return 0.0
            seg_num = int(seg_match.group(1))
            mp4 = artifacts_dir / "segments" / f"segment_{seg_num:03d}.mp4"
            if not mp4.exists():
                return 0.0
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    check=True, timeout=10,
                )
                return max(0.0, float(result.stdout.strip()))
            except Exception:
                return 0.0

        def _shot_data(shot_id: str, big_id: str, seg_id: str, include_video: bool = True) -> dict[str, Any]:
            unit = unit_status_map.get(shot_id)
            vurl = _video_url(seg_id) if include_video else ""
            if unit is not None:
                status = str(unit.get("status", "pending")).strip()
                start_time = float(unit.get("start_time", 0) or 0)
                end_time = float(unit.get("end_time", 0) or 0)
                unit_seg_id = str(unit.get("segment_id", "")).strip()
                unit_index = int(unit.get("unit_index", 0))
                error_message = str(unit.get("error_message", "")).strip()
            else:
                status = "pending"
                start_time = 0.0
                end_time = 0.0
                unit_seg_id = seg_id
                unit_index = 0
                error_message = ""
            # state store 中没有记录时，从 mp4 存在性推断已完成（兼容旧版无 state 的产物）
            if unit is None and vurl:
                status = "done"
            # state store 无时长数据时，从视频文件本身读
            if start_time <= 0 and end_time <= 0 and vurl:
                duration = _probe_video_duration(seg_id)
                if duration > 0:
                    start_time = 0.0
                    end_time = duration
            return {
                "shot_id": shot_id,
                "unit_index": unit_index,
                "segment_id": unit_seg_id,
                "status": status,
                "video_url": vurl,
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(max(0.0, end_time - start_time), 2),
                "error_message": error_message,
                "scene_desc": "",
                "big_segment_id": big_id,
                "keyframe_start_url": _keyframe_url(shot_id, "start"),
                "keyframe_end_url": _keyframe_url(shot_id, "end"),
                "keyframe_prompt_start_zh": "",
                "keyframe_prompt_start_en": "",
                "keyframe_prompt_end_zh": "",
                "keyframe_prompt_end_en": "",
                "video_prompt_zh": "",
                "video_prompt_en": "",
            }

        # --- 构建 segments（按 seg_XXXX 平铺，每个 seg 一组 shots） ---
        segments: list[dict[str, Any]] = []

        if role3_details:
            for big_id in sorted(role3_details, key=lambda bid: (
                0, int(_re.search(r'(\d+)', bid).group(1))
            ) if _re.search(r'(\d+)', bid) else (1, 0)):
                info = role3_details[big_id]
                segs = info.get("segs", [])
                for seg in segs:
                    seg_id = seg.get("seg_id", "")
                    scene_desc_zh = seg.get("scene_desc_zh", "")
                    # 优先使用 segment 自己的 remotion_id，fallback 到 big 级别
                    seg_remotion_id = str(seg.get("remotion_id", "")).strip() or str(info.get("remotion_id", "")).strip()
                    subject_count = self._count_subjects_for_seg(scene_desc_zh, seg_remotion_id)
                    is_multi_subject = seg_remotion_id in _MULTI_SUBJECT_TEMPLATES
                    segment_video_url = _video_url(seg_id)
                    segment_unit = segment_unit_map.get(seg_id)
                    if segment_unit is None:
                        # 兼容旧任务无 segment_id：用首 shot 的 unit_id 查找
                        first_shot_id = self._build_shot_id(seg_id, 1)
                        segment_unit = unit_status_map.get(first_shot_id)
                    if segment_unit is not None:
                        segment_status = str(segment_unit.get("status", "pending")).strip()
                        segment_start_time = float(segment_unit.get("start_time", 0) or 0)
                        segment_end_time = float(segment_unit.get("end_time", 0) or 0)
                        segment_error_message = str(segment_unit.get("error_message", "")).strip()
                    else:
                        segment_status = "done" if segment_video_url else "pending"
                        segment_start_time = 0.0
                        segment_end_time = 0.0
                        segment_error_message = ""
                    shots: list[dict[str, Any]] = []
                    for i in range(1, subject_count + 1):
                        shot_id = self._build_shot_id(seg_id, i)
                        shots.append(_shot_data(shot_id, big_id, seg_id, include_video=not is_multi_subject))

                    segments.append({
                        "segment_id": seg_id,
                        "big_segment_id": big_id,
                        "remotion_id": seg_remotion_id,
                        "scene_desc_zh": scene_desc_zh,
                        "status": segment_status,
                        "video_url": segment_video_url,
                        "start_time": segment_start_time,
                        "end_time": segment_end_time,
                        "duration": round(max(0.0, segment_end_time - segment_start_time), 2),
                        "error_message": segment_error_message,
                        "shots": shots,
                    })

        # 用 module_a_output.json 的 segment 绝对时间覆盖 state store 的相对时间
        module_a_path = artifacts_dir / "module_a_output.json"
        if module_a_path.exists():
            try:
                ma_data = json.loads(module_a_path.read_text(encoding="utf-8"))
                ma_seg_map: dict[str, tuple[float, float]] = {}
                for ma_seg in ma_data.get("segments", []):
                    sid = str(ma_seg.get("segment_id", "")).strip()
                    if sid:
                        ma_seg_map[sid] = (
                            float(ma_seg.get("start_time", 0) or 0),
                            float(ma_seg.get("end_time", 0) or 0),
                        )
                for seg in segments:
                    seg_id = str(seg.get("segment_id", "")).strip()
                    if seg_id in ma_seg_map:
                        st, et = ma_seg_map[seg_id]
                        seg["start_time"] = round(st, 2)
                        seg["end_time"] = round(et, 2)
                        seg["duration"] = round(max(0.0, et - st), 2)
            except Exception:
                pass

        # 活跃重跑：取最新提交且 active=True 的条目（忽略已完成的旧记录）
        active_rerun: dict[str, Any] | None = None
        latest_ms: int = 0
        for key, meta in self._rerun_thread_meta.items():
            if not key.startswith(normalized_task_id + "|module_d"):
                continue
            if not bool(meta.get("active")):
                continue
            submitted_ms = int(meta.get("submitted_at_ms", 0) or 0)
            if submitted_ms >= latest_ms:
                latest_ms = submitted_ms
                active_rerun = {
                    "active": True,
                    "status": str(meta.get("status", "")).strip(),
                    "big_segment_id": str(meta.get("big_segment_id", "")).strip(),
                    "segment_id": str(meta.get("segment_id", "")).strip(),
                    "frame_type": str(meta.get("frame_type", "")).strip(),
                    "submitted_at": str(meta.get("submitted_at", "")).strip(),
                    "submitted_at_ms": submitted_ms,
                    "started_at_ms": int(meta.get("started_at_ms", 0) or 0),
                    "last_error": str(meta.get("last_error", "")).strip(),
                    "failure_reason": str(meta.get("failure_reason", "")).strip(),
                }

        # 如果 state store 无数据，从 segments/shot 状态推算 unit_summary
        if unit_summary.get("total_units", 0) == 0 and segments:
            shot_statuses = [str(seg.get("status", "pending")) for seg in segments]
            unit_summary = {
                "module_name": "D",
                "total_units": len(shot_statuses),
                "status_counts": {
                    "done": shot_statuses.count("done"),
                    "running": shot_statuses.count("running"),
                    "pending": shot_statuses.count("pending"),
                    "failed": shot_statuses.count("failed"),
                },
                "pending_unit_ids": [],
                "running_unit_ids": [],
                "failed_unit_ids": [],
                "done_unit_ids": [],
                "problem_unit_ids": [],
            }

        return {
            "ok": True,
            "task_id": normalized_task_id,
            "module_d_status": str(module_status_map.get("D", "unknown")),
            "unit_summary": unit_summary,
            "segments": segments,
            "output_video_url": output_video_url,
            "active_rerun": active_rerun,
        }

    @staticmethod
    def _segment_id_from_segment_mp4_name(file_name: str) -> str:
        """从 segment_XXX.mp4 文件名反推 seg_XXXX 标识。"""
        matched = _re.match(r"^segment_(\d+)\.mp4$", str(file_name).strip())
        if not matched:
            return ""
        return f"seg_{int(matched.group(1)):04d}"

    def _build_module_d_segment_videos_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：直接扫描 artifacts/segments 目录，返回各 segment 视频文件的存在性与时间戳。
        参数说明：
        - task_id: 目标任务 ID。
        返回值：
        - dict[str, Any]: 含 items 映射的轻量负载，供前端独立轮询视频文件变化。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：不依赖 role3/state_store/ffprobe，仅反映磁盘文件事实。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        if not task_dir.exists():
            return {
                "ok": False,
                "error": f"任务目录不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
                "items": {},
            }
        segments_dir = task_dir / "artifacts" / "segments"
        items: dict[str, dict[str, Any]] = {}

        if segments_dir.exists():
            for mp4_path in sorted(segments_dir.glob("segment_*.mp4")):
                seg_id = self._segment_id_from_segment_mp4_name(mp4_path.name)
                if not seg_id:
                    continue
                try:
                    stat_result = mp4_path.stat()
                except OSError:
                    continue
                mtime_sec = int(stat_result.st_mtime)
                items[seg_id] = {
                    "segment_id": seg_id,
                    "exists": True,
                    "mtime": mtime_sec,
                    "size_bytes": int(stat_result.st_size),
                    "video_url": (
                        self._build_task_file_url(task_id=normalized_task_id, file_path=mp4_path)
                        + f"?t={mtime_sec}"
                    ),
                }

        return {
            "ok": True,
            "task_id": normalized_task_id,
            "items": items,
        }

    # ------------------------------------------------------------------
    # Remotion 模板 props 构建
    # ------------------------------------------------------------------

    def _build_remotion_request_props(
        self,
        task_id: str,
        segment_id: str,
        frame_type: str = "both",
        transition_bg: str | None = None,
    ) -> dict[str, Any]:
        """
        功能说明：为 Remotion 模板构建渲染 props。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识（如 seg_0001）。
        - frame_type: "start" / "end" / "both"，决定 frames 数组的元素个数。
        - transition_bg: 过渡模板背景 "white" / "black" / None（取上一个 segment 尾帧）。
        返回值：
        - dict[str, Any]: 含 frames / slots 数组的 Remotion props。
        异常说明：
        - ValueError: segment 数据缺失或 remotion_id 不支持时抛出。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        frames_dir = task_dir / "artifacts" / "frames"
        payload = self._build_module_d_payload(task_id=task_id)
        segments = payload.get("segments", [])

        # 找到目标 segment
        target_seg = None
        for seg in segments:
            if str(seg.get("segment_id", "")).strip() == segment_id:
                target_seg = seg
                break
        if not target_seg:
            raise ValueError(f"segment {segment_id} 不存在于页面负载中")
        remotion_id = str(target_seg.get("remotion_id", "")).strip()
        if not remotion_id:
            raise ValueError(f"segment {segment_id} 缺少 remotion_id，无法构建模板请求。")
        target_shots = target_seg.get("shots", [])
        if not target_shots:
            raise ValueError(f"segment {segment_id} 没有 shot 数据")

        def _keyframe_url_ft(shot_id: str, ft: str) -> str:
            """获取 shot 指定帧类型的图片 URL。"""
            frame_path = frames_dir / f"{shot_id}_{ft}.png"
            if frame_path.exists():
                path = self._build_task_file_url(task_id=task_id, file_path=frame_path)
                port = self._bound_port or self.port
                url = f"http://{self.host}:{port}{path}?_t={int(frame_path.stat().st_mtime)}"
                return url
            return ""

        def _make_symbol(src_url: str, w: float = 0.42, h: float = 0.42) -> dict[str, Any]:
            return {"src": src_url, "width_ratio": w, "height_ratio": h}

        # 根据 frame_type 确定每 slot 要取的帧类型列表
        if frame_type not in ("start", "end", "both"):
            frame_type = "both"
        frame_keys = ["start", "end"] if frame_type == "both" else [frame_type]

        # 从 Module A 加载 energy/rhythm 特征
        energy_level, rhythm_tension = self._load_segment_energy(task_id=task_id, segment_id=segment_id)

        base_props = {
            "template": remotion_id,
            "fps": 24,
            "duration_in_frames": 48,
            "bpm": 120,
            "energy_level": energy_level,
            "rhythm_tension": rhythm_tension,
        }

        if remotion_id == "CenterTemplate":
            first_shot = target_shots[0]
            sid = str(first_shot.get("shot_id", "")).strip()
            frames = [
                _make_symbol(_keyframe_url_ft(sid, k), 1.0, 1.0)
                for k in frame_keys
            ]
            if not frames[0]["src"]:
                raise ValueError(f"segment {segment_id} 下没有可用的首帧图片。")
            # 首尾帧模式：尾帧缺失则复用首帧
            if len(frames) > 1 and not frames[1]["src"]:
                frames[1] = frames[0]
            return {
                **base_props,
                "background": {"kind": "solid", "color": "#ffffff"},
                "frames": frames,
                "motion": {"breathe": True},
            }

        if remotion_id in _TRANSITION_TEMPLATES:
            curr_shot = target_shots[0]
            curr_sid = str(curr_shot.get("shot_id", "")).strip()
            start_src = _keyframe_url_ft(curr_sid, "start") if curr_sid else ""
            end_src = _keyframe_url_ft(curr_sid, "end") if curr_sid else ""

            # scene_after 根据 frame_type 决定使用首帧还是尾帧
            if frame_type == "end":
                scene_after_src = end_src or start_src
            else:
                scene_after_src = start_src
            if not scene_after_src:
                raise ValueError(f"segment {segment_id} 下没有可用的{'尾' if frame_type == 'end' else '首'}帧图片。")

            # scene_before：由 transition_bg 决定
            if transition_bg == "white":
                bf_bg: dict[str, Any] = {"kind": "solid", "color": "#ffffff"}
                bf_sym: dict[str, Any] = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
            elif transition_bg == "black":
                bf_bg = {"kind": "solid", "color": "#000000"}
                bf_sym = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
            else:
                prev_shot_id = ""
                prev_seg_multi = False
                prev_seg_id = ""
                for i, seg in enumerate(segments):
                    if str(seg.get("segment_id", "")).strip() == segment_id and i > 0:
                        prev_seg = segments[i - 1]
                        prev_seg_id = str(prev_seg.get("segment_id", "")).strip()
                        prev_remotion_id = str(prev_seg.get("remotion_id", "")).strip()
                        prev_seg_multi = prev_remotion_id in _MULTI_SUBJECT_TEMPLATES
                        if not prev_seg_multi:
                            prev_shots = prev_seg.get("shots", [])
                            if prev_shots:
                                prev_shot_id = str(prev_shots[-1].get("shot_id", "")).strip()
                        break
                prev_src = ""
                if prev_seg_multi and prev_seg_id:
                    # 多主体模板：shot 帧是格子局部图，需从渲染视频提取末帧
                    seg_match = _re.search(r"(\d+)", prev_seg_id)
                    if seg_match:
                        prev_mp4 = task_dir / "artifacts" / "segments" / f"segment_{int(seg_match.group(1)):03d}.mp4"
                        if prev_mp4.exists():
                            out_frame = frames_dir / f"{prev_seg_id}_end_from_video.png"
                            if not out_frame.exists() or prev_mp4.stat().st_mtime > out_frame.stat().st_mtime:
                                import shutil as _shutil
                                ffmpeg_bin = str(getattr(getattr(getattr(self, "app_config", None), "ffmpeg", None), "ffmpeg_bin", "")).strip() or _shutil.which("ffmpeg") or "ffmpeg"
                                subprocess.run(
                                    [ffmpeg_bin, "-y", "-sseof", "-0.1", "-i", str(prev_mp4),
                                     "-vsync", "0", "-vframes", "1", str(out_frame)],
                                    capture_output=True, timeout=30,
                                )
                            if out_frame.exists():
                                port = self._bound_port or self.port
                                path = self._build_task_file_url(task_id=task_id, file_path=out_frame)
                                prev_src = f"http://{self.host}:{port}{path}?_t={int(out_frame.stat().st_mtime)}"
                elif prev_shot_id:
                    prev_frame_path = frames_dir / f"{prev_shot_id}_end.png"
                    if prev_frame_path.exists():
                        port = self._bound_port or self.port
                        path = self._build_task_file_url(task_id=task_id, file_path=prev_frame_path)
                        prev_src = f"http://{self.host}:{port}{path}?_t={int(prev_frame_path.stat().st_mtime)}"
                bf_bg = {"kind": "solid", "color": "#ffffff"}
                bf_sym = _make_symbol(prev_src or scene_after_src, 1.0, 1.0)

            # frames 数组携带额外帧
            extra_frames: list[dict[str, Any]] = []
            if frame_type == "both":
                if end_src and end_src != scene_after_src:
                    extra_frames = [_make_symbol(end_src, 1.0, 1.0)]

            result: dict[str, Any] = {
                **base_props,
                "scene_before": {"background": bf_bg, "symbol": bf_sym},
                "scene_after": {"background": {"kind": "solid", "color": "#ffffff"}, "symbol": _make_symbol(scene_after_src, 1.0, 1.0)},
                "motion": {"travel_px": 512 if remotion_id == "PanRightTemplate" else 320, "easing": "ease_in_out"},
            }
            if extra_frames:
                result["frames"] = extra_frames
            return result

        if remotion_id in _MULTI_SUBJECT_TEMPLATES:
            # 每个 shot 对应一个 slot，每个 slot 有 frames 数组
            normalized_shots = target_shots[:3]
            while len(normalized_shots) < 3:
                normalized_shots.append(normalized_shots[-1] if normalized_shots else normalized_shots[0])

            slots: list[dict[str, Any]] = []
            for shot in normalized_shots:
                sid = str(shot.get("shot_id", "")).strip()
                slot_frames = [
                    _make_symbol(_keyframe_url_ft(sid, k), 0.26, 0.52)
                    for k in frame_keys
                ]
                slots.append({"frames": slot_frames})

            result: dict[str, Any] = {
                **base_props,
                "background": {"kind": "solid", "color": "#ffffff"},
                "slots": slots,
            }
            if remotion_id == "GridTemplate":
                result["layout"] = {"visible_cell_count": 3}
                result["motion"] = {"active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 72}
            else:
                result["layout"] = {"visible_cell_count": 3}
                result["motion"] = {"loop": False}
            return result

        raise ValueError(f"不支持的 remotion_id：{remotion_id}")

    # ------------------------------------------------------------------
    # Segment 重跑 handler（统一入口）
    # ------------------------------------------------------------------

    def _handle_module_d_rerun(
        self,
        task_id: str,
        segment_id: str,
        frame_type: str,
        transition_bg: str | None = None,
    ) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：统一处理模块 D segment 单帧/首尾帧重跑请求。不直接暴露给 HTTP 路由，
        由 _handle_module_d_rerun_segment 和 _handle_module_d_rerun_both_frames 作为 thin wrapper 调用。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识。
        - frame_type: "start" / "end" / "both"。
        - transition_bg: 过渡模板背景 "white" / "black" / None（取上一个 segment 尾帧）。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        if not segment_id:
            return {"ok": False, "error": "缺少 segment_id 参数。"}, HTTPStatus.BAD_REQUEST
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST
        if frame_type not in ("start", "end", "both"):
            return {"ok": False, "error": "frame_type 必须为 start / end / both。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, segment_id) + f"_{frame_type}"
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"segment {segment_id} 正在重跑中（{frame_type}），请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_segment_rerun_in_background,
            args=(task_id, segment_id, frame_type, rerun_key, transition_bg),
            name=f"module-d-rerun-{segment_id}-{frame_type}",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "segment_id": segment_id,
            "frame_type": frame_type,
            "transition_bg": transition_bg,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        type_label = {"start": "首帧", "end": "尾帧", "both": "首尾帧"}.get(frame_type, frame_type)
        return {
            "ok": True,
            "message": f"segment {segment_id} {type_label}重跑已提交。",
            "segment_id": segment_id,
            "frame_type": frame_type,
        }, HTTPStatus.OK

    # ------------------------------------------------------------------
    # Segment 重跑 handler（单帧）
    # ------------------------------------------------------------------

    def _handle_module_d_rerun_segment(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D segment 单帧重跑请求（thin wrapper，统一走 _handle_module_d_rerun）。
        参数说明：
        - parsed: 已解析的 URL 对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        frame_type = str(query.get("frame_type", [""])[0]).strip()
        if frame_type not in ("start", "end"):
            return {"ok": False, "error": "frame_type 必须为 start 或 end。"}, HTTPStatus.BAD_REQUEST
        transition_bg_raw = str(query.get("transition_bg", [""])[0]).strip()
        transition_bg: str | None = transition_bg_raw if transition_bg_raw in ("white", "black") else None
        return self._handle_module_d_rerun(
            task_id=task_id, segment_id=segment_id, frame_type=frame_type, transition_bg=transition_bg,
        )

    def _run_module_d_segment_rerun_in_background(
        self,
        task_id: str,
        segment_id: str,
        frame_type: str,
        rerun_key: str,
        transition_bg: str | None = None,
    ) -> None:
        """
        功能说明：后台线程执行 Remotion 模板渲染来重跑 segment（只使用目标 segment 的数据）。
        时长严格按 Module A 的 segment 时长设置 duration_in_frames，与首尾帧重跑及正式流水线一致。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: 具体 segment 标识（如 seg_0001）。
        - frame_type: "start" 或 "end"。
        - rerun_key: 线程唯一键。
        - transition_bg: 过渡模板首帧背景，"white"/"black"/None（默认上一个 segment 尾帧）。
        返回值：无。
        异常说明：异常会记录到 _rerun_thread_meta。
        """
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)

        try:
            # 1. 构建 Remotion props
            props = self._build_remotion_request_props(
                task_id=task_id,
                segment_id=segment_id,
                frame_type=frame_type,
                transition_bg=transition_bg,
            )
            props, seg_duration, total_frames = self._apply_module_a_segment_duration_to_props(
                task_id=task_id,
                segment_id=segment_id,
                props=props,
            )
            remotion_id = str(props.get("template", "")).strip()
            composition_id = remotion_id
            if not composition_id:
                raise ValueError(f"remotion_id 为空。")

            # 2. 解析 Remotion 项目目录（固定为项目根目录下的 remotion_templates）
            project_root = Path(__file__).resolve().parents[4]
            remotion_project_dir = (project_root / "remotion_templates").resolve()
            if not remotion_project_dir.exists():
                raise FileNotFoundError(f"Remotion 模板工程目录不存在：{remotion_project_dir}")

            # 3. 写 props JSON
            task_dir = self._resolve_task_dir(task_id=task_id)
            artifact_props_dir = task_dir / "artifacts" / "remotion_reruns"
            artifact_props_dir.mkdir(parents=True, exist_ok=True)
            props_path = artifact_props_dir / f"{segment_id}_{frame_type}_props.json"
            props_path.write_text(
                json.dumps(props, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # 4. 渲染输出路径（直接写入 segments，与前端预览一致）
            segments_dir = task_dir / "artifacts" / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            seg_match = _re.search(r"(\d+)", segment_id)
            if not seg_match:
                raise ValueError(f"无法从 segment_id 解析序号：{segment_id}")
            seg_num = int(seg_match.group(1))
            output_path = segments_dir / f"segment_{seg_num:03d}.mp4"

            # 5. 调用 Remotion 渲染
            from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment
            render_template_segment(
                remotion_project_dir=remotion_project_dir,
                composition_id=composition_id,
                props_json_path=props_path,
                output_path=output_path,
            )

            if logger:
                logger.info(
                    "模块D segment 重跑完成，task_id=%s，segment_id=%s，"
                    "seg_duration=%ss，total_frames=%s，output=%s",
                    task_id, segment_id, seg_duration, total_frames, output_path,
                )

            if meta:
                meta["status"] = "done"
                meta["active"] = False

        except Exception as error:
            error_text = str(error)
            if logger:
                logger.error(
                    "模块D segment 重跑失败，task_id=%s，segment_id=%s，error=%s",
                    task_id, segment_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "Remotion 渲染失败"

        finally:
            # 清理线程引用与元数据
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)

    def _load_segment_energy(self, task_id: str, segment_id: str) -> tuple[str, float]:
        """
        功能说明：从 module_a_output.json 读取指定 segment 的 energy_level 和 rhythm_tension。
        通过 energy_features 列表与 segment 时间范围的重叠匹配。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识（如 seg_0001）。
        返回值：
        - tuple[str, float]: (energy_level, rhythm_tension)，数据缺失时默认返回 ("mid", 0.5)。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        module_a_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_path.exists():
            return "mid", 0.5
        try:
            data = json.loads(module_a_path.read_text(encoding="utf-8"))
            # 找到 segment 的起止时间
            seg_start = 0.0
            seg_end = 0.0
            for seg in data.get("segments", []):
                if str(seg.get("segment_id", "")).strip() == segment_id:
                    seg_start = float(seg.get("start_time", 0) or 0)
                    seg_end = float(seg.get("end_time", 0) or 0)
                    break
            if seg_start <= 0 and seg_end <= 0:
                return "mid", 0.5
            # 在 energy_features 中找时间重叠的条目
            for feat in data.get("energy_features", []):
                f_start = float(feat.get("start_time", 0) or 0)
                f_end = float(feat.get("end_time", 0) or 0)
                if f_start < seg_end and f_end > seg_start:
                    energy_level = str(feat.get("energy_level", "mid")).strip().lower()
                    if energy_level not in ("low", "mid", "high"):
                        energy_level = "mid"
                    rhythm_tension = float(feat.get("rhythm_tension", 0.5) or 0.5)
                    rhythm_tension = max(0.0, min(1.0, rhythm_tension))
                    return energy_level, rhythm_tension
        except Exception:
            pass
        return "mid", 0.5

    def _load_module_a_segment_duration(self, task_id: str, segment_id: str) -> float | None:
        """
        功能说明：从 module_a_output.json 读取指定 segment 的时长（秒）。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识（如 seg_0001）。
        返回值：
        - float | None: 时长（秒），数据缺失时返回 None。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        module_a_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_path.exists():
            return None
        try:
            data = json.loads(module_a_path.read_text(encoding="utf-8"))
            for seg in data.get("segments", []):
                if str(seg.get("segment_id", "")).strip() == segment_id:
                    start = float(seg.get("start_time", 0) or 0)
                    end = float(seg.get("end_time", 0) or 0)
                    duration = round(max(0.0, end - start), 3)
                    return duration if duration > 0 else None
        except Exception:
            pass
        return None

    def _apply_module_a_segment_duration_to_props(
        self,
        task_id: str,
        segment_id: str,
        props: dict[str, Any],
    ) -> tuple[dict[str, Any], float, int]:
        """
        功能说明：用 Module A segment 时长覆盖 Remotion props 的 duration_in_frames。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识（如 seg_0001）。
        - props: 待渲染的 Remotion props。
        返回值：
        - tuple[dict, float, int]: (更新后的 props, 时长秒, 总帧数)。
        异常说明：
        - ValueError: module_a_output.json 中缺少 segment 时长时抛出。
        """
        seg_duration = self._load_module_a_segment_duration(task_id=task_id, segment_id=segment_id)
        if seg_duration is None:
            raise ValueError(f"segment {segment_id} 在 module_a_output.json 中无时长数据")
        fps = int(props.get("fps", 24) or 24)
        total_frames = max(1, round(seg_duration * fps))
        props["duration_in_frames"] = total_frames
        return props, seg_duration, total_frames

    # ------------------------------------------------------------------
    # Segment 首尾帧重跑 handler（thin wrapper）
    # ------------------------------------------------------------------

    def _handle_module_d_rerun_both_frames(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D segment 首尾帧重跑请求（thin wrapper，统一走 _handle_module_d_rerun）。
        参数说明：
        - parsed: 已解析的 URL 对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        transition_bg_raw = str(query.get("transition_bg", [""])[0]).strip()
        transition_bg: str | None = transition_bg_raw if transition_bg_raw in ("white", "black") else None
        return self._handle_module_d_rerun(
            task_id=task_id, segment_id=segment_id, frame_type="both", transition_bg=transition_bg,
        )

    # ------------------------------------------------------------------
    # Module 级批量重跑
    # ------------------------------------------------------------------

    def _handle_module_d_rerun_module(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D 批量重跑请求 —— 遍历所有 segment 逐一提交重跑。
        参数说明：
        - parsed: 已解析的 URL 对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        frame_type = str(query.get("frame_type", [""])[0]).strip()

        if frame_type not in ("start", "end", "both"):
            return {"ok": False, "error": "frame_type 必须为 start / end / both。"}, HTTPStatus.BAD_REQUEST
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST

        # 解析 role3 获取所有 segment
        task_dir = self._resolve_task_dir(task_id=task_id)
        role3_details = self._load_role3_seg_details(task_dir=task_dir)
        if not role3_details:
            return {"ok": False, "error": "无 role3 数据，无法批量重跑。"}, HTTPStatus.BAD_REQUEST

        all_segments: list[dict[str, str]] = []
        for info in role3_details.values():
            for seg in info.get("segs", []):
                all_segments.append(seg)

        if not all_segments:
            return {"ok": False, "error": "无 segment 数据。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, f"module_{frame_type}")
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"模块 D 正在批量重跑中（{frame_type}），请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_rerun_module_in_background,
            args=(task_id, all_segments, frame_type, rerun_key),
            name=f"module-d-rerun-module-{frame_type}",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "frame_type": frame_type,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        return {
            "ok": True,
            "message": f"模块 D 批量重跑已提交（{frame_type}），共 {len(all_segments)} 个 segment。",
            "segment_count": len(all_segments),
            "frame_type": frame_type,
        }, HTTPStatus.OK

    def _run_module_d_rerun_module_in_background(
        self,
        task_id: str,
        segments: list[dict[str, str]],
        frame_type: str,
        rerun_key: str,
    ) -> None:
        """
        功能说明：后台线程逐 segment 提交重跑。
        """
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)

        try:
            for seg in segments:
                segment_id = seg.get("seg_id", "")
                if not segment_id:
                    continue

                seg_rerun_key = _build_module_d_rerun_key(task_id, segment_id) + f"_{frame_type}"

                # 跳过已在运行中的 segment
                active_seg = self._rerun_threads.get(seg_rerun_key)
                if active_seg and active_seg.is_alive():
                    if logger:
                        logger.info("模块D 批量跳过正在重跑的 segment=%s", segment_id)
                    continue

                self._rerun_threads.pop(seg_rerun_key, None)
                self._rerun_thread_meta.pop(seg_rerun_key, None)

                t = threading.Thread(
                    target=self._run_module_d_segment_rerun_in_background,
                    args=(task_id, segment_id, frame_type, seg_rerun_key),
                    name=f"module-d-rerun-{segment_id}-{frame_type}",
                    daemon=True,
                )

                self._rerun_threads[seg_rerun_key] = t
                self._rerun_thread_meta[seg_rerun_key] = {
                    "active": True,
                    "status": "queued",
                    "segment_id": segment_id,
                    "frame_type": frame_type,
                    "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "submitted_at_ms": int(time.time() * 1000),
                    "started_at_ms": 0,
                    "last_error": "",
                    "failure_reason": "",
                }
                t.start()

            if meta:
                meta["status"] = "done"
                meta["active"] = False

        except Exception as error:
            error_text = str(error)
            if logger:
                logger.error(
                    "模块D 批量重跑失败，task_id=%s，error=%s",
                    task_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "批量重跑失败"

        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)
