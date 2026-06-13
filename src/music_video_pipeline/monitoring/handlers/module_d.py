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
from urllib.parse import parse_qs, quote

from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_streaming_dir
from music_video_pipeline.monitoring.routes import (
    TASK_MODULE_D_API_PATH,
    TASK_MODULE_D_RERUN_SEGMENT_API_PATH,
    TASK_MODULE_D_RERUN_BOTH_FRAMES_API_PATH,
    TASK_MODULE_D_RERUN_MODULE_API_PATH,
    TASK_MODULE_D_RERUN_TOONCRAFTER_API_PATH,
    TASK_MODULE_D_RERUN_TOONCRAFTER_MODULE_API_PATH,
    TASK_MODULE_D_RERUN_REMOTION_API_PATH,
    TASK_MODULE_D_RERUN_REMOTION_MODULE_API_PATH,
    TASK_MODULE_D_TOONCRAFTER_MODE_API_PATH,
    TASK_MODULE_D_REBUILD_FINAL_API_PATH,
    TOONCRAFTER_MODE_FILE_NAME,
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


def _resolve_shot_mode_for_segment(task_dir: Path, segment_id: str, shot_id: str, fallback_mode: str) -> str:
    """
    功能说明：从 tooncrafter_mode.json 读取指定 shot 的帧填充模式。
    参数说明：
    - task_dir: 任务根目录。
    - segment_id: segment 标识。
    - shot_id: shot 标识。
    - fallback_mode: 未保存时的默认值。
    返回值：
    - str: "slow" / "pingpong" / "holdtail"。
    """
    mode_file = task_dir / "artifacts" / TOONCRAFTER_MODE_FILE_NAME
    if not mode_file.exists():
        return fallback_mode
    try:
        data = json.loads(mode_file.read_text(encoding="utf-8"))
        seg_entry = data.get(segment_id) if isinstance(data, dict) else None
        if isinstance(seg_entry, dict):
            saved = str(seg_entry.get(shot_id, "")).strip()
            if saved in ("slow", "pingpong", "holdtail"):
                return saved
        if isinstance(seg_entry, str) and seg_entry in ("slow", "pingpong", "holdtail"):
            return seg_entry
    except Exception:
        pass
    return fallback_mode


def _expand_frames_by_mode(
    frames: list[Path], mode: str, total_frames: int,
) -> list[Path]:
    """
    功能说明：根据帧填充模式扩展帧序列到目标帧数。
    参数说明：
    - frames: 输入帧序列（通常 16 帧）。
    - mode: "slow"(慢放) / "pingpong"(往返循环) / "holdtail"(尾帧保持)。
    - total_frames: 目标总帧数。
    返回值：
    - list[Path]: 扩展后的帧序列。
    """
    if mode == "pingpong" and frames:
        pingpong_base = list(frames) + list(reversed(frames))[1:-1]
        repeated: list[Path] = []
        while len(repeated) < total_frames:
            repeated.extend(pingpong_base)
        return repeated[:total_frames]
    if mode == "holdtail" and frames:
        result = list(frames)
        if len(result) < total_frames:
            result.extend([frames[-1]] * (total_frames - len(result)))
        return result[:total_frames]
    # mode == "slow": 走默认重采样，不做扩展
    return frames


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
                task_id=normalized_task_id, module_name="D")
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
            frame_type = str(meta.get("frame_type", "")).strip()
            if not bool(meta.get("active")) and frame_type != "rebuild_final":
                continue
            submitted_ms = int(meta.get("submitted_at_ms", 0) or 0)
            if submitted_ms >= latest_ms:
                latest_ms = submitted_ms
                stored_active = bool(meta.get("active", False))
                active_rerun = {
                    "active": stored_active,
                    "status": str(meta.get("status", "")).strip(),
                    "big_segment_id": str(meta.get("big_segment_id", "")).strip(),
                    "segment_id": str(meta.get("segment_id", "")).strip(),
                    "frame_type": str(meta.get("frame_type", "")).strip(),
                    "submitted_at": str(meta.get("submitted_at", "")).strip(),
                    "submitted_at_ms": submitted_ms,
                    "started_at_ms": int(meta.get("started_at_ms", 0) or 0),
                    "last_error": str(meta.get("last_error", "")).strip(),
                    "failure_reason": str(meta.get("failure_reason", "")).strip(),
                    "video_url": str(meta.get("video_url", "")).strip(),
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
    # ToonCrafter 帧填充模式持久化
    # ------------------------------------------------------------------

    def _handle_module_d_tooncrafter_mode(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理 segment/shot 级帧填充模式的读/写请求。
        支持：
          GET  ?task_id=xxx&segment_id=seg_XXXX
               → 返回整体模式 {"mode": "slow"} 或逐 shot 模式 {"modes": {...}}
          GET  ?task_id=xxx&segment_id=seg_XXXX&mode=slow
               → 写入 segment 整体模式
          GET  ?task_id=xxx&segment_id=seg_XXXX&shot_id=shot_X_Y&mode=slow
               → 写入 shot 级模式
        存储结构: {"seg_XXXX": "slow", "seg_YYYY": {"shot_Y_1": "pingpong", ...}}
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        shot_id = str(query.get("shot_id", [""])[0]).strip()
        write_mode = str(query.get("mode", [""])[0]).strip()
        if not segment_id:
            return {"ok": False, "error": "缺少 segment_id 参数。"}, HTTPStatus.BAD_REQUEST

        task_dir = self._resolve_task_dir(task_id=task_id)
        mode_file = task_dir / "artifacts" / TOONCRAFTER_MODE_FILE_NAME

        # --- 写入 ---
        if write_mode in ("slow", "pingpong", "holdtail"):
            modes: dict[str, Any] = {}
            if mode_file.exists():
                try:
                    modes = json.loads(mode_file.read_text(encoding="utf-8"))
                except Exception:
                    modes = {}
                if not isinstance(modes, dict):
                    modes = {}
            if shot_id:
                # shot 级写入
                seg_entry = modes.get(segment_id)
                if not isinstance(seg_entry, dict):
                    seg_entry = {}
                seg_entry[shot_id] = write_mode
                modes[segment_id] = seg_entry
            else:
                # segment 整体写入
                modes[segment_id] = write_mode
            mode_file.parent.mkdir(parents=True, exist_ok=True)
            mode_file.write_text(json.dumps(modes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {"ok": True, "mode": write_mode, "segment_id": segment_id, "task_id": task_id}, HTTPStatus.OK

        # --- 读取 ---
        mode = "slow"
        seg_modes: dict[str, str] = {}
        if mode_file.exists():
            try:
                data = json.loads(mode_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    seg_entry = data.get(segment_id)
                    if isinstance(seg_entry, dict):
                        seg_modes = {k: v for k, v in seg_entry.items() if v in ("slow", "pingpong", "holdtail")}
                    elif isinstance(seg_entry, str) and seg_entry in ("slow", "pingpong", "holdtail"):
                        mode = seg_entry
            except Exception:
                pass

        result: dict[str, Any] = {"ok": True, "segment_id": segment_id, "task_id": task_id}
        if seg_modes:
            result["modes"] = seg_modes
            result["mode"] = mode  # fallback
        else:
            result["mode"] = mode
        return result, HTTPStatus.OK

    # ------------------------------------------------------------------
    # 转场辅助：提取上一个 segment 的尾帧（每次重新截取）
    # ------------------------------------------------------------------

    def _extract_prev_segment_tail_frame(
        self, task_id: str, segment_id: str, task_dir: Path,
    ) -> Path | None:
        """从上一个 segment 的视频文件中提取尾帧（每次重新截取）。"""
        logger = getattr(self, "logger", None)
        segments_dir = task_dir / "artifacts" / "segments"
        seg_match = _re.search(r"(\d+)", segment_id)
        if not seg_match:
            return None
        prev_num = int(seg_match.group(1)) - 1
        if prev_num < 1:
            return None
        prev_mp4 = segments_dir / f"segment_{prev_num:03d}.mp4"
        if not prev_mp4.exists():
            return None
        out_frame = task_dir / "artifacts" / "frames" / f"seg_{prev_num:04d}_end_from_video.png"
        out_frame.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        ffmpeg_bin = str(getattr(getattr(getattr(self, "app_config", None), "ffmpeg", None), "ffmpeg_bin", "")).strip() or _shutil.which("ffmpeg") or "ffmpeg"
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-sseof", "-0.1", "-i", str(prev_mp4),
                 "-vsync", "0", "-vframes", "1", str(out_frame)],
                capture_output=True, timeout=30,
            )
        except Exception as exc:
            if logger:
                logger.warning("提取上一个 segment 尾帧失败：%s", exc)
            return None
        return out_frame if out_frame.exists() else None

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
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] _build_remotion_request_props 开始，task_id=%s，segment_id=%s，frame_type=%s，segments_count=%s",
                task_id, segment_id, frame_type, len(payload.get("segments", [])),
            )
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

        if logger:
            logger.info(
                "[模块D] _build_remotion_request_props props 构建完成，template=%s",
                remotion_id,
            )
        
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

            # frames 数组：ToonCrafter 插值帧序列
            extra_frames: list[dict[str, Any]] = []
            tooncrafter_base = task_dir / "artifacts" / "tooncrafter_frames" / segment_id
            if tooncrafter_base.is_dir():
                tc_shot_dirs = sorted(tooncrafter_base.iterdir())
                for shot_dir in tc_shot_dirs:
                    if not shot_dir.is_dir():
                        continue
                    tc_files = sorted(shot_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1]))
                    for fp in tc_files:
                        port = self._bound_port or self.port
                        path = self._build_task_file_url(task_id=task_id, file_path=fp)
                        url = f"http://{self.host}:{port}{path}?_t={int(fp.stat().st_mtime)}"
                        extra_frames.append({"src": url, "width_ratio": 1.0, "height_ratio": 1.0})

            result: dict[str, Any] = {
                **base_props,
                "scene_before": {"background": bf_bg, "symbol": bf_sym},
                "scene_after": {"background": {"kind": "solid", "color": "#ffffff"}, "symbol": _make_symbol(scene_after_src, 1.0, 1.0)},
                "motion": {"travel_px": 1920 if remotion_id == "PanRightTemplate" else 1080, "easing": "ease_in_out"},
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
                    _make_symbol(_keyframe_url_ft(sid, k), 0.28, 0.80)
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
                result["motion"] = {"active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 240}
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

        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] rerun handler 收到请求，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )
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
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

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
                encoding="utf-8")
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
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 首尾帧重跑请求已到达，task_id=%s，segment_id=%s，transition_bg=%s",
                task_id, segment_id, transition_bg,
            )
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
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

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

    # ------------------------------------------------------------------
    # ToonCrafter + Remotion 重跑
    # ------------------------------------------------------------------

    def _handle_module_d_rerun_tooncrafter(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D segment 的 ToonCrafter + Remotion 重跑请求（thin wrapper）。
        参数说明：
        - parsed: 已解析的 URL 对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        mode = str(query.get("mode", [""])[0]).strip() or "slow"
        if mode not in ("slow", "pingpong", "holdtail"):
            mode = "slow"
        transition_bg_raw = str(query.get("transition_bg", [""])[0]).strip()
        transition_bg: str | None = transition_bg_raw if transition_bg_raw in ("white", "black") else None
        if not segment_id:
            return {"ok": False, "error": "缺少 segment_id 参数。"}, HTTPStatus.BAD_REQUEST
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, segment_id) + "_tooncrafter"
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"segment {segment_id} 正在 ToonCrafter 重跑中，请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_segment_rerun_tooncrafter_in_background,
            args=(task_id, segment_id, rerun_key, mode, transition_bg),
            name=f"module-d-tooncrafter-{segment_id}",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "segment_id": segment_id,
            "frame_type": "tooncrafter",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        return {
            "ok": True,
            "message": f"segment {segment_id} ToonCrafter 重跑已提交。",
            "segment_id": segment_id,
            "frame_type": "tooncrafter",
        }, HTTPStatus.OK

    def _handle_module_d_rerun_tooncrafter_module(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D 全量 ToonCrafter 重跑请求。
        参数说明：
        - parsed: 已解析的 URL 对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: (payload, status_code)。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST

        task_dir = self._resolve_task_dir(task_id=task_id)
        role3_details = self._load_role3_seg_details(task_dir=task_dir)
        if not role3_details:
            return {"ok": False, "error": "无 role3 数据，无法执行 ToonCrafter 重跑。"}, HTTPStatus.BAD_REQUEST

        all_segments: list[dict[str, str]] = []
        for info in role3_details.values():
            for seg in info.get("segs", []):
                all_segments.append(seg)
        if not all_segments:
            return {"ok": False, "error": "无 segment 数据。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, "module_tooncrafter")
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": "模块 D 正在 ToonCrafter 批量重跑中，请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_rerun_tooncrafter_module_in_background,
            args=(task_id, all_segments, rerun_key),
            name="module-d-tooncrafter-module",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "frame_type": "tooncrafter",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        return {
            "ok": True,
            "message": f"模块 D ToonCrafter 批量重跑已提交，共 {len(all_segments)} 个 segment。",
            "segment_count": len(all_segments),
            "frame_type": "tooncrafter",
        }, HTTPStatus.OK

    def _run_module_d_segment_rerun_tooncrafter_in_background(
        self,
        task_id: str,
        segment_id: str,
        rerun_key: str,
        mode: str = "slow",
        transition_bg: str | None = None,
    ) -> None:
        """
        功能说明：后台线程执行 ToonCrafter + Remotion 渲染。
        先为每个 shot 跑 ToonCrafter 生成中间帧序列，再用所有帧构建 Remotion props 渲染最终视频。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: segment 标识。
        - rerun_key: 线程唯一键。
        - mode: 帧填充模式 "slow"(慢放) / "pingpong"(往返循环) / "relay"(接力生成)。
        返回值：无。
        """
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

        try:
            task_dir = self._resolve_task_dir(task_id=task_id)
            artifacts_dir = task_dir / "artifacts"
            frames_dir = artifacts_dir / "frames"

            # 1. 加载 segment 数据
            payload = self._build_module_d_payload(task_id=task_id)
            segments = payload.get("segments", [])
            target_seg = next(
                (seg for seg in segments if str(seg.get("segment_id", "")).strip() == segment_id),
                None,
            )
            if not target_seg:
                raise ValueError(f"segment {segment_id} 不存在")
            remotion_id = str(target_seg.get("remotion_id", "")).strip()
            if not remotion_id:
                raise ValueError(f"segment {segment_id} 缺少 remotion_id")
            target_shots = target_seg.get("shots", [])
            if not target_shots:
                raise ValueError(f"segment {segment_id} 没有 shot 数据")

            # 2. 从 module_a_output.json 读取时长
            seg_duration = self._load_module_a_segment_duration(task_id=task_id, segment_id=segment_id)
            if seg_duration is None:
                raise ValueError(f"segment {segment_id} 在 module_a_output.json 中无时长数据")
            fps = 24
            total_frames = max(1, round(seg_duration * fps))

            # 3. 加载 module_b_output.json 获取 video_prompt_en
            module_b_output_path = artifacts_dir / "module_b_output.json"
            module_b_shot_map: dict[str, str] = {}
            if module_b_output_path.exists():
                try:
                    b_data = json.loads(module_b_output_path.read_text(encoding="utf-8"))
                    if isinstance(b_data, list):
                        for item in b_data:
                            sid = str(item.get("shot_id", "")).strip()
                            prompt = str(item.get("video_prompt_en", "")).strip()
                            if sid and prompt:
                                module_b_shot_map[sid] = prompt
                except Exception:
                    pass

            # 4. 每个 shot 跑 ToonCrafter 生成帧
            from music_video_pipeline.modules.module_d.backends.comfyui_renderer import (
                generate_tooncrafter_frames,
            )
            from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit

            is_multi = remotion_id in _MULTI_SUBJECT_TEMPLATES
            is_transition = remotion_id in _TRANSITION_TEMPLATES
            tooncrafter_dir = artifacts_dir / "tooncrafter_frames" / segment_id
            tooncrafter_dir.mkdir(parents=True, exist_ok=True)

            # 读取 ComfyUI 配置中 ToonCrafter 原生帧数
            comfy_frames = 16
            try:
                comfy_frames = int(
                    getattr(
                        getattr(getattr(self, "app_config", None), "module_d", None), "comfyui", None
                    ).generation_frames or 16
                )
            except Exception:
                pass

            shot_frame_map: dict[str, list[Path]] = {}

            if is_multi:
                # 多主体：每个 slot 独立跑 ToonCrafter
                normalized_shots = target_shots[:3]
                while len(normalized_shots) < 3:
                    normalized_shots.append(normalized_shots[-1] if normalized_shots else {"shot_id": ""})

                for slot_idx, shot in enumerate(normalized_shots):
                    shot_id = str(shot.get("shot_id", "")).strip()
                    if not shot_id:
                        continue
                    start_path = frames_dir / f"{shot_id}_start.png"
                    end_path = frames_dir / f"{shot_id}_end.png"
                    if not start_path.exists():
                        if logger:
                            logger.warning("ToonCrafter 跳过 shot=%s：首帧不存在", shot_id)
                        continue
                    prompt = module_b_shot_map.get(shot_id, "animation transition")
                    shot_tooncrafter_dir = tooncrafter_dir / shot_id
                    shot_tooncrafter_dir.mkdir(parents=True, exist_ok=True)
                    temp_unit = ModuleDUnit(
                        unit_id=shot_id,
                        unit_index=slot_idx,
                        shot={
                            "frame_path_start": str(start_path),
                            "frame_path_end": str(end_path) if end_path.exists() else str(start_path),
                            "video_prompt_en": prompt,
                        },
                        start_time=0,
                        end_time=0,
                        duration=0,
                        exact_frames=comfy_frames,
                        segment_path=shot_tooncrafter_dir / f"{shot_id}.mp4",
                        temp_segment_path=shot_tooncrafter_dir / f"{shot_id}.tmp.mp4")
                    try:
                        comfy_cfg = getattr(getattr(self, "app_config", None), "module_d", None).comfyui  # type: ignore[union-attr]
                        comfyui_global_cfg = getattr(self, "app_config", None).comfyui  # type: ignore[union-attr]
                        contract_path = str(comfy_cfg.contract_file or "configs/comfyui/module_d.contract.json")
                    except Exception:
                        raise RuntimeError("模块D ToonCrafter 重跑失败：无法读取 ComfyUI 配置（app_config）")
                    frames = generate_tooncrafter_frames(
                        temp_unit,
                        comfy_cfg=comfy_cfg,
                        comfyui_global_cfg=comfyui_global_cfg,
                        contract_path=contract_path,
                        frames_output_dir=shot_tooncrafter_dir,
                        pad_to_fit=is_multi,
                    )
                    if total_frames > 0:
                        shot_mode = _resolve_shot_mode_for_segment(task_dir, segment_id, shot_id, mode)
                        frames = _expand_frames_by_mode(frames, shot_mode, total_frames)
                    shot_frame_map[shot_id] = frames

            else:
                # 单主体/转场：每个 shot 独立跑 ToonCrafter
                for shot in target_shots:
                    shot_id = str(shot.get("shot_id", "")).strip()
                    if not shot_id:
                        continue
                    start_path = frames_dir / f"{shot_id}_start.png"
                    end_path = frames_dir / f"{shot_id}_end.png"
                    if not start_path.exists():
                        if logger:
                            logger.warning("ToonCrafter 跳过 shot=%s：首帧不存在", shot_id)
                        continue
                    prompt = module_b_shot_map.get(shot_id, "animation transition")
                    shot_tooncrafter_dir = tooncrafter_dir / shot_id
                    shot_tooncrafter_dir.mkdir(parents=True, exist_ok=True)
                    temp_unit = ModuleDUnit(
                        unit_id=shot_id,
                        unit_index=0,
                        shot={
                            "frame_path_start": str(start_path),
                            "frame_path_end": str(end_path) if end_path.exists() else str(start_path),
                            "video_prompt_en": prompt,
                        },
                        start_time=0,
                        end_time=0,
                        duration=0,
                        exact_frames=comfy_frames,
                        segment_path=shot_tooncrafter_dir / f"{shot_id}.mp4",
                        temp_segment_path=shot_tooncrafter_dir / f"{shot_id}.tmp.mp4")
                    try:
                        comfy_cfg = getattr(getattr(self, "app_config", None), "module_d", None).comfyui  # type: ignore[union-attr]
                        comfyui_global_cfg = getattr(self, "app_config", None).comfyui  # type: ignore[union-attr]
                        contract_path = str(comfy_cfg.contract_file or "configs/comfyui/module_d.contract.json")
                    except Exception:
                        raise RuntimeError("模块D ToonCrafter 重跑失败：无法读取 ComfyUI 配置（app_config）")
                    frames = generate_tooncrafter_frames(
                        temp_unit,
                        comfy_cfg=comfy_cfg,
                        comfyui_global_cfg=comfyui_global_cfg,
                        contract_path=contract_path,
                        frames_output_dir=shot_tooncrafter_dir,
                        pad_to_fit=is_multi,
                    )
                    if total_frames > 0:
                        frames = _expand_frames_by_mode(frames, mode, total_frames)
                    shot_frame_map[shot_id] = frames

            if not shot_frame_map:
                raise ValueError(f"segment {segment_id} 下没有可用的 shot 帧")

            # 5. 构建 Remotion props
            def _make_symbol_from_path(png_path: Path, w: float = 1.0, h: float = 1.0) -> dict[str, Any]:
                try:
                    rel_url = self._build_task_file_url(task_id=task_id, file_path=Path(png_path))
                    port = self._bound_port or self.port
                    src = f"http://{self.host}:{port}{rel_url}?_t={int(Path(png_path).stat().st_mtime)}"
                except Exception:
                    src = Path(str(png_path)).resolve().as_uri()
                return {"src": src, "width_ratio": w, "height_ratio": h}

            base_props: dict[str, Any] = {
                "template": remotion_id,
                "fps": fps,
                "duration_in_frames": total_frames,
                "bpm": 120,
                "background": {"kind": "solid", "color": "#ffffff"},
            }

            if remotion_id == "CenterTemplate":
                first_shot_id = str(target_shots[0].get("shot_id", "")).strip()
                tc_frames = shot_frame_map.get(first_shot_id, [])
                if not tc_frames:
                    raise ValueError(f"shot {first_shot_id} 无 ToonCrafter 帧")
                base_props["frames"] = [_make_symbol_from_path(f, 1.0, 1.0) for f in tc_frames]
                base_props["motion"] = {"breathe": True}

            elif is_transition:
                curr_shot_id = str(target_shots[0].get("shot_id", "")).strip()
                tc_frames = shot_frame_map.get(curr_shot_id, [])
                end_path = frames_dir / f"{curr_shot_id}_end.png"
                # 白屏 / 黑屏 / 上一个 segment 尾帧
                if transition_bg == "white":
                    bf_bg: dict[str, Any] = {"kind": "solid", "color": "#ffffff"}
                    bf_sym_src = None
                elif transition_bg == "black":
                    bf_bg = {"kind": "solid", "color": "#000000"}
                    bf_sym_src = None
                else:
                    bf_bg = {"kind": "solid", "color": "#ffffff"}
                    bf_sym_src = self._extract_prev_segment_tail_frame(
                        task_id=task_id, segment_id=segment_id, task_dir=task_dir,
                    )
                after_src = _make_symbol_from_path(end_path, 1.0, 1.0) if end_path.exists() else (
                    _make_symbol_from_path(tc_frames[-1], 1.0, 1.0) if tc_frames else None
                )
                if bf_sym_src:
                    before_src = _make_symbol_from_path(bf_sym_src, 1.0, 1.0)
                else:
                    before_src = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
                if not after_src:
                    raise ValueError(f"shot {curr_shot_id} 缺少尾帧")
                travel_px = 1920 if remotion_id == "PanRightTemplate" else 1080
                base_props["scene_before"] = {
                    "background": bf_bg, "symbol": before_src,
                }
                after_src = _make_symbol_from_path(end_path, 1.0, 1.0) if end_path.exists() else (
                    _make_symbol_from_path(tc_frames[-1], 1.0, 1.0) if tc_frames else None
                )
                if bf_sym_src:
                    before_src = _make_symbol_from_path(bf_sym_src, 1.0, 1.0)
                else:
                    before_src = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
                if not after_src:
                    raise ValueError(f"shot {curr_shot_id} 缺少尾帧")
                travel_px = 1920 if remotion_id == "PanRightTemplate" else 1080
                base_props["scene_before"] = {
                    "background": bf_bg, "symbol": before_src,
                }
                base_props["scene_after"] = {
                    "background": {"kind": "solid", "color": "#ffffff"}, "symbol": after_src,
                }
                base_props["motion"] = {"travel_px": travel_px, "easing": "ease_in_out"}
                if tc_frames:
                    base_props["frames"] = [_make_symbol_from_path(f, 1.0, 1.0) for f in tc_frames]

            elif is_multi:
                normalized_shots = target_shots[:3]
                while len(normalized_shots) < 3:
                    normalized_shots.append(normalized_shots[-1] if normalized_shots else {"shot_id": ""})
                slots: list[dict[str, Any]] = []
                for shot in normalized_shots:
                    sid = str(shot.get("shot_id", "")).strip()
                    tc_frames = shot_frame_map.get(sid, [])
                    if not tc_frames:
                        # 无 ToonCrafter 帧时使用原始帧文件
                        raw_start = frames_dir / f"{sid}_start.png"
                        if raw_start.exists():
                            tc_frames = [raw_start]
                        else:
                            continue
                    slot_frames = [_make_symbol_from_path(f, 0.28, 0.80) for f in tc_frames]
                    slots.append({"frames": slot_frames})
                base_props["slots"] = slots
                base_props["layout"] = {"visible_cell_count": 3}
                if remotion_id == "GridTemplate":
                    base_props["motion"] = {
                        "active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 240,
                    }
                else:
                    base_props["motion"] = {"loop": False}

            else:
                raise ValueError(f"不支持的 remotion_id：{remotion_id}")

            # 6. 写 props JSON
            props_dir = artifacts_dir / "remotion_reruns"
            props_dir.mkdir(parents=True, exist_ok=True)
            props_path = props_dir / f"{segment_id}_tooncrafter_props.json"
            props_path.write_text(
                json.dumps(base_props, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            # 7. 调用 Remotion 渲染
            seg_match = _re.search(r"(\d+)", segment_id)
            if not seg_match:
                raise ValueError(f"无法从 segment_id 解析序号：{segment_id}")
            seg_num = int(seg_match.group(1))
            output_path = artifacts_dir / "segments" / f"segment_{seg_num:03d}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            project_root = Path(__file__).resolve().parents[4]
            remotion_project_dir = (project_root / "remotion_templates").resolve()
            from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment

            render_template_segment(
                remotion_project_dir=remotion_project_dir,
                composition_id=remotion_id,
                props_json_path=props_path,
                output_path=output_path,
            )

            if logger:
                logger.info(
                    "模块D ToonCrafter 重跑完成，task_id=%s，segment_id=%s，"
                    "duration=%ss，frames=%s，shots_with_tc=%s，output=%s",
                    task_id, segment_id, seg_duration, total_frames,
                    list(shot_frame_map.keys()), output_path,
                )

            # 8. 更新 state store
            for shot in target_shots:
                shot_id = str(shot.get("shot_id", "")).strip()
                if shot_id:
                    self.state_store.set_module_unit_status(
                        task_id=task_id,
                        module_name="D",
                        unit_id=shot_id,
                        status="done",
                        artifact_path=str(output_path),
                        error_message="")
            if meta:
                meta["status"] = "done"
                meta["active"] = False

        except Exception as error:
            error_text = str(error)
            if logger:
                logger.error(
                    "模块D ToonCrafter 重跑失败，task_id=%s，segment_id=%s，error=%s",
                    task_id, segment_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "ToonCrafter 渲染失败"

        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)

    def _run_module_d_rerun_tooncrafter_module_in_background(
        self,
        task_id: str,
        segments: list[dict[str, str]],
        rerun_key: str,
    ) -> None:
        """
        功能说明：后台线程逐 segment 提交 ToonCrafter 重跑。
        参数说明：
        - task_id: 任务唯一标识。
        - segments: segment 列表。
        - rerun_key: 线程唯一键。
        返回值：无。
        """
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

        try:
            for seg in segments:
                segment_id = seg.get("seg_id", "")
                if not segment_id:
                    continue

                seg_rerun_key = _build_module_d_rerun_key(task_id, segment_id) + "_tooncrafter"

                active_seg = self._rerun_threads.get(seg_rerun_key)
                if active_seg and active_seg.is_alive():
                    if logger:
                        logger.info("模块D ToonCrafter 批量跳过正在重跑的 segment=%s", segment_id)
                    continue

                self._rerun_threads.pop(seg_rerun_key, None)
                self._rerun_thread_meta.pop(seg_rerun_key, None)

                t = threading.Thread(
                    target=self._run_module_d_segment_rerun_tooncrafter_in_background,
                    args=(task_id, segment_id, seg_rerun_key),
                    name=f"module-d-tooncrafter-{segment_id}",
                    daemon=True,
                )
                self._rerun_threads[seg_rerun_key] = t
                self._rerun_thread_meta[seg_rerun_key] = {
                    "active": True,
                    "status": "queued",
                    "segment_id": segment_id,
                    "frame_type": "tooncrafter",
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
                    "模块D ToonCrafter 批量重跑失败，task_id=%s，error=%s",
                    task_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "ToonCrafter 批量重跑失败"

        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)

    # ------------------------------------------------------------------
    # Remotion 重渲（复用已有 ToonCrafter 帧，不重新跑 ToonCrafter）
    # ------------------------------------------------------------------

    def _handle_module_d_rerun_remotion(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 D segment Remotion 重渲请求（复用已有 ToonCrafter 帧，不重新跑 ToonCrafter）。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        mode = str(query.get("mode", [""])[0]).strip() or "slow"
        if mode not in ("slow", "pingpong", "holdtail"):
            mode = "slow"
        transition_bg_raw = str(query.get("transition_bg", [""])[0]).strip()
        transition_bg: str | None = transition_bg_raw if transition_bg_raw in ("white", "black") else None
        if not segment_id:
            return {"ok": False, "error": "缺少 segment_id 参数。"}, HTTPStatus.BAD_REQUEST
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, segment_id) + "_remotion"
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"segment {segment_id} 正在 Remotion 重渲中，请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_segment_rerun_remotion_in_background,
            args=(task_id, segment_id, rerun_key, mode, transition_bg),
            name=f"module-d-remotion-{segment_id}",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "segment_id": segment_id,
            "frame_type": "remotion",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        return {
            "ok": True,
            "message": f"segment {segment_id} Remotion 重渲已提交。",
            "segment_id": segment_id,
            "frame_type": "remotion",
        }, HTTPStatus.OK

    def _handle_module_d_rerun_remotion_module(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST

        task_dir = self._resolve_task_dir(task_id=task_id)
        role3_details = self._load_role3_seg_details(task_dir=task_dir)
        if not role3_details:
            return {"ok": False, "error": "无 role3 数据，无法执行 Remotion 重渲。"}, HTTPStatus.BAD_REQUEST

        all_segments: list[dict[str, str]] = []
        for info in role3_details.values():
            for seg in info.get("segs", []):
                all_segments.append(seg)
        if not all_segments:
            return {"ok": False, "error": "无 segment 数据。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, "module_remotion")
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": "模块 D 正在 Remotion 批量重渲中，请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_rerun_remotion_module_in_background,
            args=(task_id, all_segments, rerun_key),
            name="module-d-remotion-module",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "frame_type": "remotion",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        return {
            "ok": True,
            "message": f"模块 D Remotion 批量重渲已提交，共 {len(all_segments)} 个 segment。",
            "segment_count": len(all_segments),
            "frame_type": "remotion",
        }, HTTPStatus.OK

    def _run_module_d_segment_rerun_remotion_in_background(
        self,
        task_id: str,
        segment_id: str,
        rerun_key: str,
        mode: str = "slow",
        transition_bg: str | None = None,
    ) -> None:
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

        try:
            task_dir = self._resolve_task_dir(task_id=task_id)
            artifacts_dir = task_dir / "artifacts"
            frames_dir = artifacts_dir / "frames"
            tooncrafter_base_dir = artifacts_dir / "tooncrafter_frames" / segment_id

            payload = self._build_module_d_payload(task_id=task_id)
            segments = payload.get("segments", [])
            target_seg = next(
                (seg for seg in segments if str(seg.get("segment_id", "")).strip() == segment_id),
                None,
            )
            if not target_seg:
                raise ValueError(f"segment {segment_id} 不存在")
            remotion_id = str(target_seg.get("remotion_id", "")).strip()
            if not remotion_id:
                raise ValueError(f"segment {segment_id} 缺少 remotion_id")
            target_shots = target_seg.get("shots", [])
            if not target_shots:
                raise ValueError(f"segment {segment_id} 没有 shot 数据")

            is_multi = remotion_id in _MULTI_SUBJECT_TEMPLATES
            is_transition = remotion_id in _TRANSITION_TEMPLATES

            if not tooncrafter_base_dir.exists():
                raise ValueError(
                    f"segment {segment_id} 下没有 ToonCrafter 帧数据，"
                    f"请先点击 ToonCrafter 按钮生成帧。path={tooncrafter_base_dir}"
                )

            fps = 24
            seg_duration = self._load_module_a_segment_duration(task_id=task_id, segment_id=segment_id)
            if seg_duration is None:
                raise ValueError(f"segment {segment_id} 在 module_a_output.json 中无时长数据")
            total_frames = max(1, round(seg_duration * fps))

            shot_frame_map: dict[str, list[Path]] = {}
            if is_multi:
                normalized_shots = target_shots[:3]
                while len(normalized_shots) < 3:
                    normalized_shots.append(normalized_shots[-1] if normalized_shots else {"shot_id": ""})
                for shot in normalized_shots:
                    sid = str(shot.get("shot_id", "")).strip()
                    if not sid:
                        continue
                    shot_frames_dir = tooncrafter_base_dir / sid
                    if shot_frames_dir.exists():
                        png_files = sorted(shot_frames_dir.glob("frame_*.png"))
                        if png_files:
                            shot_frame_map[sid] = png_files
            else:
                for shot in target_shots:
                    sid = str(shot.get("shot_id", "")).strip()
                    if not sid:
                        continue
                    shot_frames_dir = tooncrafter_base_dir / sid
                    if shot_frames_dir.exists():
                        png_files = sorted(shot_frames_dir.glob("frame_*.png"))
                        if png_files:
                            shot_frame_map[sid] = png_files

            if not shot_frame_map:
                raise ValueError(
                    f"segment {segment_id} 下没有可用的 ToonCrafter 帧，"
                    f"请先点击 ToonCrafter 按钮生成帧。path={tooncrafter_base_dir}"
                )

            # 在各 shot 帧上应用帧填充模式（逐 shot 读取保存的模式）
            if total_frames > 0:
                for sid in shot_frame_map:
                    shot_mode = _resolve_shot_mode_for_segment(task_dir, segment_id, sid, mode)
                    shot_frame_map[sid] = _expand_frames_by_mode(
                        shot_frame_map[sid], shot_mode, total_frames
                    )

            def _make_symbol_from_path(png_path: Path, w: float = 1.0, h: float = 1.0) -> dict[str, Any]:
                try:
                    rel_url = self._build_task_file_url(task_id=task_id, file_path=Path(png_path))
                    port = self._bound_port or self.port
                    src = f"http://{self.host}:{port}{rel_url}?_t={int(Path(png_path).stat().st_mtime)}"
                except Exception:
                    src = Path(str(png_path)).resolve().as_uri()
                return {"src": src, "width_ratio": w, "height_ratio": h}

            base_props: dict[str, Any] = {
                "template": remotion_id,
                "fps": fps,
                "duration_in_frames": total_frames,
                "bpm": 120,
                "background": {"kind": "solid", "color": "#ffffff"},
            }

            if remotion_id == "CenterTemplate":
                first_shot_id = str(target_shots[0].get("shot_id", "")).strip()
                tc_frames = shot_frame_map.get(first_shot_id, [])
                if not tc_frames:
                    raise ValueError(f"shot {first_shot_id} 无 ToonCrafter 帧")
                base_props["frames"] = [_make_symbol_from_path(f, 1.0, 1.0) for f in tc_frames]
                base_props["motion"] = {"breathe": True}

            elif is_transition:
                curr_shot_id = str(target_shots[0].get("shot_id", "")).strip()
                tc_frames = shot_frame_map.get(curr_shot_id, [])
                end_path = frames_dir / f"{curr_shot_id}_end.png"
                # 白屏 / 黑屏 / 上一个 segment 尾帧
                if transition_bg == "white":
                    bf_bg: dict[str, Any] = {"kind": "solid", "color": "#ffffff"}
                    bf_sym_src = None
                elif transition_bg == "black":
                    bf_bg = {"kind": "solid", "color": "#000000"}
                    bf_sym_src = None
                else:
                    bf_bg = {"kind": "solid", "color": "#ffffff"}
                    bf_sym_src = self._extract_prev_segment_tail_frame(
                        task_id=task_id, segment_id=segment_id, task_dir=task_dir,
                    )
                after_src = _make_symbol_from_path(end_path, 1.0, 1.0) if end_path.exists() else (
                    _make_symbol_from_path(tc_frames[-1], 1.0, 1.0) if tc_frames else None
                )
                if bf_sym_src:
                    before_src = _make_symbol_from_path(bf_sym_src, 1.0, 1.0)
                else:
                    before_src = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
                if not after_src:
                    raise ValueError(f"shot {curr_shot_id} 缺少尾帧")
                travel_px = 1920 if remotion_id == "PanRightTemplate" else 1080
                base_props["scene_before"] = {
                    "background": bf_bg, "symbol": before_src,
                }
                after_src = _make_symbol_from_path(end_path, 1.0, 1.0) if end_path.exists() else (
                    _make_symbol_from_path(tc_frames[-1], 1.0, 1.0) if tc_frames else None
                )
                if bf_sym_src:
                    before_src = _make_symbol_from_path(bf_sym_src, 1.0, 1.0)
                else:
                    before_src = {"src": _TRANSPARENT_PIXEL, "width_ratio": 0.01, "height_ratio": 0.01}
                if not after_src:
                    raise ValueError(f"shot {curr_shot_id} 缺少尾帧")
                travel_px = 1920 if remotion_id == "PanRightTemplate" else 1080
                base_props["scene_before"] = {
                    "background": bf_bg, "symbol": before_src,
                }
                base_props["scene_after"] = {
                    "background": {"kind": "solid", "color": "#ffffff"}, "symbol": after_src,
                }
                base_props["motion"] = {"travel_px": travel_px, "easing": "ease_in_out"}
                if tc_frames:
                    base_props["frames"] = [_make_symbol_from_path(f, 1.0, 1.0) for f in tc_frames]

            elif is_multi:
                normalized_shots = target_shots[:3]
                while len(normalized_shots) < 3:
                    normalized_shots.append(normalized_shots[-1] if normalized_shots else {"shot_id": ""})
                slots: list[dict[str, Any]] = []
                for shot in normalized_shots:
                    sid = str(shot.get("shot_id", "")).strip()
                    tc_frames = shot_frame_map.get(sid, [])
                    if not tc_frames:
                        raw_start = frames_dir / f"{sid}_start.png"
                        if raw_start.exists():
                            tc_frames = [raw_start]
                        else:
                            continue
                    slot_frames = [_make_symbol_from_path(f, 0.28, 0.80) for f in tc_frames]
                    slots.append({"frames": slot_frames})
                base_props["slots"] = slots
                base_props["layout"] = {"visible_cell_count": 3}
                if remotion_id == "GridTemplate":
                    base_props["motion"] = {
                        "active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 240,
                    }
                else:
                    base_props["motion"] = {"loop": False}
            else:
                raise ValueError(f"不支持的 remotion_id：{remotion_id}")

            props_dir = artifacts_dir / "remotion_reruns"
            props_dir.mkdir(parents=True, exist_ok=True)
            props_path = props_dir / f"{segment_id}_remotion_props.json"
            props_path.write_text(
                json.dumps(base_props, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            seg_match = _re.search(r"(\d+)", segment_id)
            if not seg_match:
                raise ValueError(f"无法从 segment_id 解析序号：{segment_id}")
            seg_num = int(seg_match.group(1))
            output_path = artifacts_dir / "segments" / f"segment_{seg_num:03d}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            project_root = Path(__file__).resolve().parents[4]
            remotion_project_dir = (project_root / "remotion_templates").resolve()
            from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment

            render_template_segment(
                remotion_project_dir=remotion_project_dir,
                composition_id=remotion_id,
                props_json_path=props_path,
                output_path=output_path,
            )

            if logger:
                logger.info(
                    "模块D Remotion 重渲完成，task_id=%s，segment_id=%s，"
                    "duration=%ss，frames=%s，shots=%s，output=%s",
                    task_id, segment_id, seg_duration, total_frames,
                    list(shot_frame_map.keys()), output_path,
                )

            for shot in target_shots:
                shot_id = str(shot.get("shot_id", "")).strip()
                if shot_id:
                    self.state_store.set_module_unit_status(
                        task_id=task_id,
                        module_name="D",
                        unit_id=shot_id,
                        status="done",
                        artifact_path=str(output_path),
                        error_message="")
            if meta:
                meta["status"] = "done"
                meta["active"] = False

        except Exception as error:
            error_text = str(error)
            if logger:
                logger.error(
                    "模块D Remotion 重渲失败，task_id=%s，segment_id=%s，error=%s",
                    task_id, segment_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "Remotion 重渲失败"

        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)

    def _run_module_d_rerun_remotion_module_in_background(
        self,
        task_id: str,
        segments: list[dict[str, str]],
        rerun_key: str,
    ) -> None:
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

        try:
            for seg in segments:
                segment_id = seg.get("seg_id", "")
                if not segment_id:
                    continue

                seg_rerun_key = _build_module_d_rerun_key(task_id, segment_id) + "_remotion"

                active_seg = self._rerun_threads.get(seg_rerun_key)
                if active_seg and active_seg.is_alive():
                    if logger:
                        logger.info("模块D Remotion 批量跳过正在重渲的 segment=%s", segment_id)
                    continue

                self._rerun_threads.pop(seg_rerun_key, None)
                self._rerun_thread_meta.pop(seg_rerun_key, None)

                t = threading.Thread(
                    target=self._run_module_d_segment_rerun_remotion_in_background,
                    args=(task_id, segment_id, seg_rerun_key),
                    name=f"module-d-remotion-{segment_id}",
                    daemon=True,
                )
                self._rerun_threads[seg_rerun_key] = t
                self._rerun_thread_meta[seg_rerun_key] = {
                    "active": True,
                    "status": "queued",
                    "segment_id": segment_id,
                    "frame_type": "remotion",
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
                    "模块D Remotion 批量重渲失败，task_id=%s，error=%s",
                    task_id, error_text,
                )
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "Remotion 批量重渲失败"

        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)

    # ------------------------------------------------------------------
    # 按选中的 segment 拼接输出成片
    # ------------------------------------------------------------------

    def _handle_module_d_rebuild_final(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：提交按 segment 选择输出成片请求（后台线程执行 FFmpeg 拼接）。
        可选 audio_path 参数指定使用的音频文件。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        segment_ids_raw = str(query.get("segment_ids", [""])[0]).strip()
        audio_path_param = str(query.get("audio_path", [""])[0]).strip() or ""
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数。"}, HTTPStatus.BAD_REQUEST
        if not segment_ids_raw:
            return {"ok": False, "error": "缺少 segment_ids 参数。"}, HTTPStatus.BAD_REQUEST

        rerun_key = _build_module_d_rerun_key(task_id, "rebuild_final") + f"_{abs(hash(segment_ids_raw))}"
        active_thread = self._rerun_threads.get(rerun_key)
        if active_thread and active_thread.is_alive():
            return {
                "ok": False,
                "error": "输出成片任务正在执行中，请等待完成。",
            }, HTTPStatus.CONFLICT

        self._rerun_threads.pop(rerun_key, None)
        self._rerun_thread_meta.pop(rerun_key, None)

        rerun_thread = threading.Thread(
            target=self._run_module_d_rebuild_final_in_background,
            args=(task_id, segment_ids_raw, rerun_key, audio_path_param),
            name="module-d-rebuild-final",
            daemon=True,
        )
        submitted_at_ms = int(time.time() * 1000)
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "frame_type": "rebuild_final",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000)),
            "submitted_at_ms": submitted_at_ms,
            "started_at_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()

        seg_count = len([s for s in segment_ids_raw.split(",") if s.strip()])
        return {
            "ok": True,
            "message": f"输出成片已提交，共 {seg_count} 段。",
            "segment_count": seg_count,
        }, HTTPStatus.OK

    def _handle_module_d_rebuild_audio_candidates(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：返回输出成片可用的音频文件候选列表。
        候选来源：resources/ 目录下的 mp3/wav/m4a + runs/{task_id}/*_visualization_audio.mp3（置顶默认选中）。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        candidates: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        project_root = Path(__file__).resolve().parents[4]

        # 1. 项目根目录 resources/ 下的音频文件
        resources_dir = project_root / "resources"
        if resources_dir.exists():
            for ext in ("*.mp3", "*.wav", "*.m4a"):
                for f in sorted(resources_dir.glob(ext)):
                    key = f.name.lower()
                    if key in seen_names:
                        continue
                    seen_names.add(key)
                    candidates.append({
                        "label": f.name,
                        "path": str(f.resolve()),
                        "size_bytes": f.stat().st_size,
                    })

        # 2. runs/{task_id}/*_visualization_audio.mp3（默认选中置顶）
        task_dir = self._resolve_task_dir(task_id=task_id)
        default_mp3 = task_dir / f"{task_id}_module_a_v2_visualization_audio.mp3"
        if default_mp3.exists():
            key = default_mp3.name.lower()
            if key in seen_names:
                # 从 candidates 移除同名旧条目，把这条置顶
                candidates = [c for c in candidates if c["path"] != str(default_mp3.resolve())]
            else:
                seen_names.add(key)
            candidates.insert(0, {
                "label": f"{default_mp3.name}（默认）",
                "path": str(default_mp3.resolve()),
                "size_bytes": default_mp3.stat().st_size,
                "default": True,
            })

        return {
            "ok": True,
            "candidates": candidates,
            "default_path": str(default_mp3.resolve()) if default_mp3.exists() else "",
        }, HTTPStatus.OK

    def _run_module_d_rebuild_final_in_background(
        self,
        task_id: str,
        segment_ids_raw: str,
        rerun_key: str,
        audio_path_param: str = "",
    ) -> None:
        meta = self._rerun_thread_meta.get(rerun_key)
        if meta:
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at_ms"] = int(time.time() * 1000)
        logger = getattr(self, "logger", None)
        if logger:
            logger.info(
                "[模块D] 后台线程开始执行重跑，task_id=%s，segment_id=%s，frame_type=%s",
                task_id, segment_id, frame_type,
            )

        try:
            task_dir = self._resolve_task_dir(task_id=task_id)
            artifacts_dir = task_dir / "artifacts"
            segments_dir = artifacts_dir / "segments"

            selected_ids = [sid.strip() for sid in segment_ids_raw.split(",") if sid.strip()]
            if logger:
                logger.info("模块D 成片原始 selected_ids=%s", selected_ids)
            # 去重 + 按数字排序
            seen: set[str] = set()
            deduped: list[str] = []
            for sid in selected_ids:
                if sid not in seen:
                    seen.add(sid)
                    deduped.append(sid)
            selected_ids = deduped
            if logger:
                logger.info("模块D 成片去重后 selected_ids=%s", selected_ids)
            selected_ids.sort(
                key=lambda _sid: int(_re.search(r"(\d+)", _sid).group(1)) if _re.search(r"(\d+)", _sid) else 0
            )
            # 从 payload 获取每个 segment 的起止时间
            payload = self._build_module_d_payload(task_id=task_id)
            seg_times: dict[str, tuple[float, float]] = {}
            for seg_item in payload.get("segments", []):
                sid = str(seg_item.get("segment_id", "")).strip()
                if sid:
                    st = float(seg_item.get("start_time", 0) or 0)
                    et = float(seg_item.get("end_time", 0) or 0)
                    if et > st:
                        seg_times[sid] = (st, et)
                    else:
                        shots_data = seg_item.get("shots", [])
                        if shots_data:
                            st = min(float(s.get("start_time", 0) or 0) for s in shots_data)
                            et = max(float(s.get("end_time", 0) or 0) for s in shots_data)
                            if et > st:
                                seg_times[sid] = (st, et)

            # 预查 mp4 + seg_times，只保留影片和元数据都完整的条目
            # 同时保证 selected_ids 和 seg_paths 并行、有序
            valid_segments: list[tuple[str, Path]] = []
            for sid in selected_ids:
                seg_match = _re.search(r"(\d+)", sid)
                if not seg_match:
                    continue
                mp4 = segments_dir / f"segment_{int(seg_match.group(1)):03d}.mp4"
                if mp4.exists() and sid in seg_times:
                    valid_segments.append((sid, mp4))
            if not valid_segments:
                raise RuntimeError("没有找到可用的 segment mp4 文件。")
            selected_ids = [vs[0] for vs in valid_segments]
            seg_paths = [vs[1] for vs in valid_segments]

            # 决定音频路径：优先使用前端传入的，否则自愈
            audio_path: Path | None = None
            if audio_path_param:
                ap = Path(audio_path_param)
                if ap.exists():
                    audio_path = ap
                else:
                    project_root = Path(__file__).resolve().parents[4]
                    ap2 = (project_root / audio_path_param).resolve()
                    if ap2.exists():
                        audio_path = ap2
            if audio_path is None:
                task_record = self.state_store.get_task(task_id=task_id)
                if not task_record:
                    raise RuntimeError(f"任务不存在：{task_id}")
                try:
                    audio_path = self._resolve_task_audio_path_from_record(
                        task_id=task_id, task_record=task_record, persist=False,
                    )
                except (FileNotFoundError, Exception) as path_err:
                    raise RuntimeError(f"音频文件不存在或无法自愈：{path_err}")
            if audio_path is None or not audio_path.exists():
                raise RuntimeError(f"音频文件不存在：{audio_path}")

            ffmpeg_bin = "ffmpeg"
            ffprobe_bin = "ffprobe"
            video_codec = "libx264"
            video_preset = "medium"
            video_crf = 18
            audio_codec = "aac"
            try:
                app_config = getattr(self, "app_config", None)
                if app_config and app_config.ffmpeg:
                    fc = app_config.ffmpeg
                    ffmpeg_bin = str(getattr(fc, "ffmpeg_bin", ffmpeg_bin))
                    ffprobe_bin = str(getattr(fc, "ffprobe_bin", ffprobe_bin))
                    video_codec = str(getattr(fc, "video_codec", video_codec))
                    video_preset = str(getattr(fc, "video_preset", video_preset))
                    video_crf = int(getattr(fc, "video_crf", video_crf))
                    audio_codec = str(getattr(fc, "audio_codec", audio_codec))
            except Exception:
                pass

            if logger:
                for sid in selected_ids:
                    if sid in seg_times:
                        st, et = seg_times[sid]
                        logger.info("模块D 成片 segment=%s time=[%.3f, %.3f] duration=%.3f", sid, st, et, et - st)

            def _seg_sort_key(sid: str) -> int:
                m = _re.search(r"(\d+)", sid)
                return int(m.group(1)) if m else 0

            sorted_ids = sorted([s for s in selected_ids if s in seg_times], key=_seg_sort_key)
            merged_ranges: list[tuple[float, float]] = []
            if sorted_ids:
                cur_start, cur_end = seg_times[sorted_ids[0]]
                for sid in sorted_ids[1:]:
                    st, et = seg_times[sid]
                    if st <= cur_end:
                        cur_end = max(cur_end, et)
                    else:
                        merged_ranges.append((cur_start, cur_end))
                        cur_start, cur_end = st, et
                merged_ranges.append((cur_start, cur_end))

                audio_parts_dir = artifacts_dir / "rebuild_audio_parts"
                audio_parts_dir.mkdir(parents=True, exist_ok=True)
                audio_parts: list[Path] = []
                for idx, (astart, aend) in enumerate(merged_ranges):
                    part_path = audio_parts_dir / f"audio_part_{idx:03d}.mp3"
                    cut_cmd = [
                        ffmpeg_bin, "-nostdin", "-y",
                        "-ss", f"{astart:.3f}",
                        "-i", str(audio_path),
                        "-to", f"{aend:.3f}",
                        "-c", "copy",
                        str(part_path),
                    ]
                    subprocess.run(cut_cmd, capture_output=True, text=True, check=True, timeout=300)
                    audio_parts.append(part_path)

                audio_concat_file = audio_parts_dir / "audio_concat.txt"
                audio_concat_lines = [f"file '{p.resolve()}'" for p in audio_parts]
                audio_concat_file.write_text("\n".join(audio_concat_lines) + "\n", encoding="utf-8")
                merged_audio = audio_parts_dir / "merged_audio.mp3"
                subprocess.run([
                    ffmpeg_bin, "-nostdin", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(audio_concat_file),
                    "-c", "copy",
                    str(merged_audio),
                ], capture_output=True, text=True, check=True, timeout=300)
            else:
                merged_audio = audio_path

            if logger:
                logger.info("模块D 成片输出开始，task_id=%s，segments=%s，seg_count=%s",
                    task_id, selected_ids, len(seg_paths),
                )

            # 每段视频剪到音频时长：每段按 seg_times 中的 et-st 截断
            # 各段实际视频因帧率取整普遍比音频时长多 30-60ms，累积后会严重漂移
            trimmed_dir = artifacts_dir / "concat_trimmed"
            trimmed_dir.mkdir(parents=True, exist_ok=True)
            trimmed_paths: list[Path] = []
            for idx, (sid, seg_path) in enumerate(zip(selected_ids, seg_paths)):
                st, et = seg_times[sid]
                dur = et - st
                if dur <= 0.0:
                    trimmed_paths.append(seg_path)
                    continue
                trimmed = trimmed_dir / f"trimmed_{idx:03d}_{dur:.3f}s.mp4"
                if not trimmed.exists() or seg_path.stat().st_mtime > trimmed.stat().st_mtime:
                    trim_cmd = [
                        ffmpeg_bin, "-nostdin", "-y",
                        "-i", str(seg_path),
                        "-t", f"{dur:.3f}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(trimmed),
                    ]
                    subprocess.run(trim_cmd, capture_output=True, text=True, check=True, timeout=300)
                trimmed_paths.append(trimmed)
            if not trimmed_paths:
                raise RuntimeError("裁剪后的视频片段为空。")
            total_video_dur = sum(
                seg_times[sid][1] - seg_times[sid][0]
                for sid in selected_ids if sid in seg_times
            )

            if logger:
                logger.info("模块D 成片视频裁剪完成，task_id=%s，trims=%s，total_video=%.3fs，total_audio=%.3fs",
                    task_id, len(trimmed_paths), total_video_dur,
                    sum(met - mst for mst, met in merged_ranges),
                )

            # 音频总时长（用于 -t 精确截断输出视频）
            audio_duration = sum(max(0.0, seg_times[sid][1] - seg_times[sid][0]) for sid in selected_ids)

            concat_file = artifacts_dir / "concat_selected.txt"
            concat_lines = [f"file '{p.resolve()}'" for p in trimmed_paths]
            concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

            output_path = task_dir / "final_output.mp4"
            encode_command = [
                ffmpeg_bin, "-nostdin", "-y",
                "-fflags", "+genpts", "-safe", "0",
                "-f", "concat", "-i", str(concat_file),
                "-i", str(merged_audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", video_codec, "-preset", video_preset, "-crf", str(video_crf),
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec, "-b:a", "192k",
                "-vsync", "cfr", "-af", "aresample=async=1000",
                "-t", f"{audio_duration:.3f}",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(encode_command, capture_output=True, text=True, check=True, timeout=600)

            if not output_path.exists():
                raise RuntimeError("输出文件未生成")

            out_dur_result = subprocess.run(
                [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
                capture_output=True, text=True, timeout=15,
            )
            out_duration = max(0.0, float(out_dur_result.stdout.strip() or 0))

            if logger:
                logger.info("模块D 成片输出完成，task_id=%s，audio=%.3fs，output=%.3fs，diff=%.3fs，path=%s",
                    task_id,
                    audio_duration, out_duration, abs(out_duration - audio_duration),
                    output_path,
                )

            if meta:
                meta["status"] = "done"
                meta["active"] = False
                meta["video_url"] = f"/task/{quote(task_id)}/final_output.mp4?t={int(output_path.stat().st_mtime)}"

        except Exception as error:
            error_text = str(error)
            if logger:
                logger.error("模块D 成片输出失败，task_id=%s，error=%s", task_id, error_text)
            if meta:
                meta["status"] = "failed"
                meta["active"] = False
                meta["last_error"] = error_text
                meta["failure_reason"] = "成片输出失败"

        finally:
            # 保留 meta 供前端轮询（下次提交时会覆盖），只清理线程引用
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread:
                self._rerun_threads.pop(rerun_key, None)