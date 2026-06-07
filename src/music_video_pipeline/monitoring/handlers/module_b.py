"""
文件用途：Module B handler mixin —— 模块 B 页面数据、role 重跑、segment 重跑。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store、模块 B 重跑回调 handler 与项目内路径工具。
维护说明：本文件仅包含模块 B 页面展示与重跑编排方法。
"""

import json
import re
import subprocess
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from music_video_pipeline.modules.module_a_v2.network_lyrics_state import current_time_text
from music_video_pipeline.modules.module_b.artifact_paths import (
    get_module_b_prompt_dir,
    get_module_b_role_result_path,
    get_module_b_streaming_dir,
)
from music_video_pipeline.modules.module_b.markdown_contracts import parse_scene_plans
from music_video_pipeline.modules.module_b.orchestrator import (
    _build_shot_id,
    _parse_subject_descriptions,
)
from music_video_pipeline.monitoring.routes import (
    ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME,
    COMPLETED_MODULE_B_RERUN_META_FILE_NAME,
)


class ModuleBHandlers:
    """Mixin —— 模块 B 页面数据构建与 role/segment 重跑相关方法。"""

    def _build_module_b_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 B 页面所需的数据负载，仅围绕当前 module_b 源码方案。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含 role 模板、当前实现状态与 segment 入口的数据对象。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：仅展示当前 module_b 源码方案与当前任务上下文。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
                "roles": [],
                "segment_items": [],
            }
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:  # noqa: BLE001
            module_status_map = {}
        try:
            module_b_unit_summary = self.state_store.get_module_unit_status_summary(
                task_id=normalized_task_id,
                module_name="B",
            )
        except Exception:  # noqa: BLE001
            module_b_unit_summary = self._build_empty_module_unit_summary(module_name="B")
        return {
            "ok": True,
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "module_status": module_status_map,
            "module_b_status": module_status_map.get("B", "unknown"),
            "module_b_unit_summary": module_b_unit_summary,
            "active_rerun": self._build_module_b_active_rerun_payload(task_id=normalized_task_id),
            "aggregate_output": self._build_task_file_asset(
                task_id=normalized_task_id,
                file_path=(task_dir / "artifacts" / "module_b_output.json").resolve(),
            ),
            "roles": self._build_module_b_role_payloads(task_id=normalized_task_id, task_dir=task_dir),
            "segment_items": self._load_module_b_segment_selector_items(task_dir=task_dir, task_id=normalized_task_id),
        }
    def _build_module_b_role_payloads(self, task_id: str, task_dir: Path) -> list[dict[str, Any]]:
        """
        功能说明：构建当前 module_b 四个 role 的展示数据。
        参数说明：
        - task_id: 任务唯一标识。
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: role 页面卡片数据数组。
        异常说明：无。
        边界条件：模板缺失或角色尚未实现时也返回稳定结构，供前端呈现占位态。
        """
        project_root = self._resolve_project_root()
        role_specs = [
            {
                "role_name": "role1",
                "title": "Role 1",
                "description": "视觉意象细化",
                "template_relpath": Path("configs/prompts/module_b.role1_visual_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role1_imagery_describer.py"),
                "contract_fields": ["pos_zh", "pos_en"],
                "supports_segment_retry": False,
            },
            {
                "role_name": "role2",
                "title": "Role 2",
                "description": "大段剧情规划",
                "template_relpath": Path("configs/prompts/module_b.role2_big_segment_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role2_story_planner.py"),
                "contract_fields": ["imagery_used", "story_outline_zh"],
                "supports_segment_retry": False,
            },
            {
                "role_name": "role3",
                "title": "Role 3",
                "description": "镜头规划",
                "template_relpath": Path("configs/prompts/module_b.role3_segment_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role3_shot_planner.py"),
                "contract_fields": ["scene_desc_zh", "remotion_id"],
                "supports_segment_retry": True,
            },
            {
                "role_name": "role4",
                "title": "Role 4",
                "description": "提示词规划",
                "template_relpath": Path("configs/prompts/module_b.role4_prompt_builder.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role4_prompt_builder.py"),
                "contract_fields": [
                    "keyframe_prompt_start_zh",
                    "keyframe_prompt_start_en",
                    "keyframe_prompt_end_zh",
                    "keyframe_prompt_end_en",
                    "video_prompt_zh",
                    "video_prompt_en",
                ],
                "supports_segment_retry": True,
            },
        ]
        # 预加载 role4 的 shot 选择列表（需要解析 role3 产物）
        role4_shot_items: list[dict[str, Any]] = []
        try:
            role4_shot_items = self._load_role4_shot_selector_items(task_dir=task_dir)
        except Exception:
            pass

        role_payloads: list[dict[str, Any]] = []
        for role_spec in role_specs:
            role_name = str(role_spec["role_name"]).strip()
            source_path = (project_root / Path(role_spec["source_relpath"])).resolve()
            template_path = (project_root / Path(role_spec["template_relpath"])).resolve()
            contract_fields = [str(item) for item in role_spec["contract_fields"]]
            result_file_path = self._find_task_local_module_b_role_artifact(task_dir=task_dir, role_name=role_name)
            implementation_status = self._describe_module_b_role_implementation(result_file_path=result_file_path)
            is_implemented = implementation_status == "implemented"
            role_active_rerun_payload = self._build_module_b_active_rerun_payload(task_id=task_id)
            is_role_active = (
                role_active_rerun_payload.get("active")
                and role_active_rerun_payload.get("role_name") == role_name
            )

            # 用 artifact_paths 获取各 role 的 streaming 固定路径
            stream_preview_path, stream_preview_meta_path = self._resolve_role_stream_preview_paths(
                task_dir=task_dir, role_name=role_name
            )

            # role3/role4 per-segment/shot streaming 预览段
            stream_preview_segments: list[dict[str, Any]] = []
            rendered_prompt_segments: list[dict[str, Any]] = []
            segment_items: list[dict[str, Any]] = []
            if role_name == "role3":
                segment_items = self._load_module_b_segment_selector_items(task_dir=task_dir, task_id=task_id)
                if segment_items:
                    stream_preview_segments = self._build_role3_stream_preview_segments(
                        task_dir=task_dir, segment_items=segment_items
                    )
                    rendered_prompt_segments = self._build_role3_rendered_prompt_segments(
                        task_dir=task_dir, segment_items=segment_items
                    )
            elif role_name == "role4":
                segment_items = role4_shot_items
                if role4_shot_items:
                    stream_preview_segments = self._build_role4_stream_preview_segments(
                        task_dir=task_dir, shot_items=role4_shot_items
                    )
                    rendered_prompt_segments = self._build_role4_rendered_prompt_segments(
                        task_dir=task_dir, shot_items=role4_shot_items
                    )

            role_payload = {
                "role_name": role_name,
                "title": str(role_spec["title"]).strip(),
                "description": str(role_spec["description"]).strip(),
                "source_path": str(source_path),
                "contract_fields": contract_fields,
                "implementation_status": implementation_status,
                "supports_role_rerun": is_implemented and role_name in {"role1", "role2", "role3", "role4"},
                "supports_segment_retry": bool(role_spec["supports_segment_retry"]) and is_implemented,
                "segment_items": segment_items,
                "active_rerun": role_active_rerun_payload if is_role_active else {
                    "active": False,
                    "status": "",
                    "mode": "",
                    "role_name": "",
                    "segment_id": "",
                    "shot_id": "",
                    "submitted_at": "",
                    "submitted_at_ms": 0,
                    "started_at": "",
                    "started_at_ms": 0,
                    "finished_at": "",
                    "finished_at_ms": 0,
                    "duration_ms": 0,
                    "last_error": "",
                    "failure_reason": "",
                },
                "prompt_template": self._build_text_file_asset(file_path=template_path),
                "rendered_prompt": self._build_task_text_file_asset(
                    task_id=task_id,
                    file_path=self._find_role_rendered_prompt(task_dir=task_dir, role_name=role_name),
                ),
                "rendered_prompt_segments": rendered_prompt_segments,
                "stream_preview_segments": stream_preview_segments,
                "stream_preview": self._build_task_text_file_asset(
                    task_id=task_id,
                    file_path=stream_preview_path,
                ),
                "stream_preview_meta": self._build_task_json_asset(
                    task_id=task_id,
                    file_path=stream_preview_meta_path,
                ),
                "result": self._build_task_file_asset(
                    task_id=task_id,
                    file_path=result_file_path,
                ),
                "result_text": self._build_task_text_file_asset(
                    task_id=task_id,
                    file_path=result_file_path,
                ),
            }
            role_payloads.append(role_payload)
        return role_payloads
    def _build_module_b_active_rerun_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 B 当前活跃重跑动作摘要，供前端显示进度态。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, Any]: 活跃重跑摘要。
        异常说明：无。
        边界条件：无活跃动作时返回 active=false 的稳定结构。
        """
        meta = self._rerun_thread_meta.get(str(task_id).strip(), {})
        if not meta:
            persisted_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
            if persisted_meta:
                return persisted_meta
        if not meta:
            completed_meta = self._load_completed_module_b_rerun_meta(task_id=task_id)
            if completed_meta:
                return completed_meta
            return {
                "active": False,
                "status": "",
                "mode": "",
                "role_name": "",
                "segment_id": "",
                "shot_id": "",
                "submitted_at": "",
                "submitted_at_ms": 0,
                "started_at": "",
                "started_at_ms": 0,
                "finished_at": "",
                "finished_at_ms": 0,
                "duration_ms": 0,
                "last_error": "",
                "failure_reason": "",
            }
        return {
            "active": bool(meta.get("active", False)),
            "status": str(meta.get("status", "")).strip(),
            "mode": str(meta.get("mode", "")).strip(),
            "role_name": str(meta.get("role_name", "")).strip(),
            "segment_id": str(meta.get("segment_id", "")).strip(),
            "shot_id": str(meta.get("shot_id", "")).strip(),
            "submitted_at": str(meta.get("submitted_at", "")).strip(),
            "submitted_at_ms": int(meta.get("submitted_at_ms", 0) or 0),
            "started_at": str(meta.get("started_at", "")).strip(),
            "started_at_ms": int(meta.get("started_at_ms", 0) or 0),
            "finished_at": str(meta.get("finished_at", "")).strip(),
            "finished_at_ms": int(meta.get("finished_at_ms", 0) or 0),
            "duration_ms": int(meta.get("duration_ms", 0) or 0),
            "last_error": str(meta.get("last_error", "")).strip(),
            "failure_reason": str(meta.get("failure_reason", "")).strip(),
        }
    def _build_active_module_b_rerun_process_path(self, task_id: str) -> Path:
        """
        功能说明：构建模块 B 活跃重跑子进程状态文件路径。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - Path: 状态文件绝对路径。
        异常说明：无。
        边界条件：文件固定写在 runs/<task_id>/ 目录下。
        """
        return self._resolve_task_dir(task_id=task_id) / ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME
    @staticmethod
    def _parse_truthy_flag(value: str) -> bool:
        """
        功能说明：把查询字符串中的布尔开关解析为真值。
        参数说明：
        - value: 原始字符串。
        返回值：
        - bool: 常见 true/1/yes/on 视为真。
        异常说明：无。
        边界条件：空字符串返回 False。
        """
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
    def _load_active_module_b_rerun_process_meta(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：从子进程状态文件恢复模块 B 活跃重跑摘要。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, Any]: 若存在活跃子进程则返回前端可用摘要，否则返回空字典。
        异常说明：无；文件损坏时自动清理并回退为空。
        边界条件：仅在 PID 仍存活时视为 active。
        """
        process_file_path = self._build_active_module_b_rerun_process_path(task_id=task_id)
        if not process_file_path.exists():
            return {}
        try:
            payload = json.loads(process_file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            process_file_path.unlink(missing_ok=True)
            return {}
        pid = int(payload.get("pid", 0) or 0)
        if pid <= 0 or not self._is_process_alive(pid):
            process_file_path.unlink(missing_ok=True)
            return {}
        submitted_at_ms = int(payload.get("submitted_at_ms", 0) or 0)
        submitted_at = str(payload.get("submitted_at", "")).strip()
        if submitted_at_ms > 0 and not submitted_at:
            submitted_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000))
        return {
            "active": True,
            "status": "running",
            "mode": str(payload.get("mode", "")).strip(),
            "role_name": str(payload.get("role_name", "")).strip(),
            "segment_id": str(payload.get("segment_id", "")).strip(),
            "shot_id": str(payload.get("shot_id", "")).strip(),
            "submitted_at": submitted_at,
            "submitted_at_ms": submitted_at_ms,
            "started_at": submitted_at,
            "started_at_ms": submitted_at_ms,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": max(0, int(time.time() * 1000) - submitted_at_ms) if submitted_at_ms > 0 else 0,
            "last_error": "",
            "failure_reason": "",
        }
    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """
        功能说明：在 Windows 上检查指定 PID 是否仍然存活。
        参数说明：
        - pid: 目标进程 PID。
        返回值：
        - bool: 进程存在且仍运行返回 True，否则返回 False。
        异常说明：无；底层探测异常时回退为 False。
        边界条件：依赖 tasklist 输出，适用于当前本地 Windows 环境。
        """
        normalized_pid = int(pid or 0)
        if normalized_pid <= 0:
            return False
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {normalized_pid}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
        except Exception:  # noqa: BLE001
            return False
        output_text = f"{completed.stdout}\n{completed.stderr}".lower()
        if "no tasks are running" in output_text or "没有运行的任务" in output_text:
            return False
        return str(normalized_pid) in output_text
    def _build_completed_module_b_rerun_meta_path(self, task_id: str) -> Path:
        return self._resolve_task_dir(task_id=task_id) / COMPLETED_MODULE_B_RERUN_META_FILE_NAME
    def _save_completed_module_b_rerun_meta(self, task_id: str, meta: dict) -> None:
        """持久化已完成重跑的 meta（duration_ms 等），不依赖内存。"""
        path = self._build_completed_module_b_rerun_meta_path(task_id=task_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("保存模块B已完成重跑meta失败，task_id=%s，error=%s", task_id, exc)
    def _load_completed_module_b_rerun_meta(self, task_id: str) -> dict:
        """从持久化文件读取已完成重跑的 meta。"""
        path = self._build_completed_module_b_rerun_meta_path(task_id=task_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            path.unlink(missing_ok=True)
            return {}
    def _terminate_active_module_b_rerun_process(self, task_id: str) -> bool:
        """
        功能说明：终止当前任务的模块 B 活跃重跑子进程树。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - bool: 成功发起终止返回 True；无活跃进程或终止失败返回 False。
        异常说明：无；错误由调用方转成冲突提示。
        边界条件：当前实现依赖 Windows `taskkill /T /F`。
        """
        process_file_path = self._build_active_module_b_rerun_process_path(task_id=task_id)
        if not process_file_path.exists():
            return False
        try:
            payload = json.loads(process_file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            process_file_path.unlink(missing_ok=True)
            return False
        pid = int(payload.get("pid", 0) or 0)
        if pid <= 0:
            process_file_path.unlink(missing_ok=True)
            return False
        if not self._is_process_alive(pid):
            process_file_path.unlink(missing_ok=True)
            return False
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        return completed.returncode == 0
    def _load_module_b_segment_selector_items(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：为 role3/role4 的按 segment 操作构建选择列表。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: segment 入口数组。
        异常说明：无；缺失上游产物时回退为空数组。
        边界条件：优先复用当前审阅页 segment 数据，确保 segment_id/shot_id 对齐页面观察上下文。
        """
        try:
            module_a_path = (task_dir / "artifacts" / "module_a_output.json").resolve()
            if not module_a_path.exists():
                return []
            ma_payload = json.loads(module_a_path.read_text(encoding="utf-8"))
            raw_segments = ma_payload.get("segments", []) if isinstance(ma_payload, dict) else []
            segment_units = []
            for item in raw_segments:
                if not isinstance(item, dict):
                    continue
                segment_id = str(item.get("segment_id", "")).strip()
                if not segment_id:
                    continue
                segment_units.append({
                    "segment_id": segment_id,
                    "big_segment_id": str(item.get("big_segment_id", "")).strip(),
                    "start_time": float(item.get("start_time", 0.0) or 0.0),
                    "end_time": float(item.get("end_time", 0.0) or 0.0),
                    "label": str(item.get("label", "")).strip(),
                    "role": str(item.get("role", "")).strip(),
                })
        except Exception:
            return []
        # 加载 role2 数据，构建 big_segment_id → story_outline_zh 映射
        role2_outline_map: dict[str, str] = {}
        try:
            role2_path = get_module_b_role_result_path((task_dir / "artifacts").resolve(), "role2")
            if role2_path.exists():
                for plan in parse_scene_plans(role2_path.read_text(encoding="utf-8")):
                    bid = str(plan.big_segment_id).strip()
                    if bid:
                        role2_outline_map[bid] = str(plan.story_outline_zh or "").strip()
        except Exception:
            pass
        # 解析 role3 输出，构建 seg_xxxx → big_xxx 映射（用于补充 Module D 缺失的 big_segment_id）
        seg_to_big_map: dict[str, str] = {}
        try:
            role3_path = get_module_b_role_result_path((task_dir / "artifacts").resolve(), "role3")
            if role3_path.exists():
                role3_text = role3_path.read_text(encoding="utf-8").replace("\r\n", "\n")
                current_bid = ""
                for line in role3_text.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("## ") and not stripped.startswith("### "):
                        current_bid = stripped[3:].strip()
                    elif stripped.startswith("### "):
                        seg = stripped[4:].strip()
                        if seg and current_bid:
                            seg_to_big_map[seg] = current_bid
        except Exception:
            pass
        selector_items: list[dict[str, Any]] = []
        for index, item in enumerate(segment_units, start=1):
            segment_id = str(item.get("segment_id", "")).strip()
            if not segment_id:
                continue
            big_segment_id = str(item.get("big_segment_id", "")).strip()
            if not big_segment_id:
                big_segment_id = seg_to_big_map.get(segment_id, "")
            story_outline_zh = role2_outline_map.get(big_segment_id, "")
            display_title = f"{big_segment_id} / {story_outline_zh}" if big_segment_id and story_outline_zh else ""
            selector_items.append(
                {
                    "segment_id": segment_id,
                    "shot_id": str(item.get("shot_id", "")).strip() or f"shot_{index:03d}",
                    "start_time": float(item.get("start_time", 0.0) or 0.0),
                    "end_time": float(item.get("end_time", 0.0) or 0.0),
                    "label": str(item.get("label", "")).strip(),
                    "role": str(item.get("role", "")).strip(),
                    "scene_desc": str(item.get("scene_desc", "")).strip(),
                    "big_segment_id": big_segment_id,
                    "story_outline_zh": story_outline_zh,
                    "display_title": display_title,
                    "display_subtitle": story_outline_zh,
                }
            )
        return selector_items
    @staticmethod
    def _classify_module_b_rerun_failure_reason(error: Exception) -> str:
        """
        功能说明：把模块 B 重跑异常归类为更易读的失败原因。
        参数说明：
        - error: 后台线程捕获到的异常对象。
        返回值：
        - str: 适合前端展示的失败原因摘要。
        异常说明：无。
        边界条件：无法归类时回退为原始异常文本。
        """
        error_text = str(error).strip()
        lowered_text = error_text.lower()
        if (
            "必须至少包含一个 `##` 条目" in error_text
            or "缺失字段" in error_text
            or "未定义字段" in error_text
            or "重复字段" in error_text
            or "模块 b role1 输出不符合契约" in lowered_text
        ):
            return "未通过契约校验"
        if "timeout" in lowered_text or "timed out" in lowered_text or "超时" in error_text:
            return "LLM 超时"
        return error_text or "模块B重跑失败"
    @staticmethod
    def _describe_module_b_role_implementation(result_file_path: Path | None) -> str:
        if result_file_path is None:
            return "missing"
        return "implemented"
    def _find_task_local_module_b_role_artifact(self, task_dir: Path, role_name: str) -> Path | None:
        """
        功能说明：返回模块 B 指定 role 的正式产物路径。
        产物固定位于 artifacts/module_b_work/<role_name>/ 下。
        """
        try:
            result_path = get_module_b_role_result_path((task_dir / "artifacts").resolve(), str(role_name).strip())
        except ValueError:
            return None
        return result_path.resolve() if result_path.exists() else None

    def _resolve_role_stream_preview_paths(
        self, task_dir: Path, role_name: str
    ) -> tuple[Path | None, Path | None]:
        """
        功能说明：返回各 role 的全局 streaming 文件路径与 meta 路径。
        role1/role2 有全局 streaming 文件；role3/role4 只有 per-segment/shot 文件，返回 (None, None)。
        """
        safe_role_name = str(role_name or "").strip().lower()
        if safe_role_name in {"role3", "role4"}:
            return None, None
        try:
            streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), safe_role_name)
        except ValueError:
            return None, None
        streaming_filename_map = {
            "role1": "role1_visual_output.streaming.md",
            "role2": "role2_story_output.streaming.md",
        }
        meta_filename_map = {
            "role1": "role1_visual_output.streaming.meta.json",
            "role2": "role2_story_output.streaming.meta.json",
        }
        streaming_filename = streaming_filename_map.get(safe_role_name)
        meta_filename = meta_filename_map.get(safe_role_name)
        stream_path = (streaming_dir / streaming_filename).resolve() if streaming_filename else None
        meta_path = (streaming_dir / meta_filename).resolve() if meta_filename else None
        return stream_path, meta_path

    def _load_role4_shot_selector_items(self, task_dir: Path) -> list[dict[str, Any]]:
        """
        功能说明：从 role3 流式文件解析 shot 列表，作为 role4 的按 shot 操作选择项。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: shot 入口数组，每项含 segment_id/shot_id/scene_desc/big_segment_id。
        异常说明：无；缺失 role3 流式产物时回退为空数组。
        边界条件：逐个大段流式文件独立读取并解析。
        """

        # 从模块 A 输出读取 segment 时间
        segment_times: dict[str, tuple[float, float]] = {}
        module_a_path = task_dir / "artifacts" / "module_a_output.json"
        if module_a_path.exists():
            try:
                ma_payload = json.loads(module_a_path.read_text(encoding="utf-8"))
                for seg in (ma_payload.get("segments", []) if isinstance(ma_payload, dict) else []):
                    sid = str(seg.get("segment_id", "")).strip()
                    if sid:
                        st = float(seg.get("start_time", 0) or 0)
                        et = float(seg.get("end_time", st) or 0)
                        segment_times[sid] = (st, et)
            except Exception:
                pass

        streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), "role3")
        items: list[dict[str, Any]] = []
        if not streaming_dir.exists():
            return items

        for stream_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            except Exception:
                continue
            # 从文件名提取 big_segment_id: role3_segment_output.streaming.big_001.md
            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()

            for block in re.split(r"\n(?=### )", text):
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
                if not seg_id:
                    continue
                scene_desc = ""
                remotion_id = ""
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith("- scene_desc_zh:"):
                        scene_desc = stripped[len("- scene_desc_zh:"):].strip()
                    elif stripped.startswith("- remotion_id:"):
                        remotion_id = stripped[len("- remotion_id:"):].strip()

                subjects = _parse_subject_descriptions(scene_desc, remotion_id)
                start_time, end_time = segment_times.get(seg_id, (0.0, 0.0))

                for subj_idx, subject_desc in enumerate(subjects, start=1):
                    shot_id = _build_shot_id(seg_id, subj_idx)

                    items.append({
                        "segment_id": shot_id,
                        "shot_id": shot_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": scene_desc,
                        "display_title": f"{shot_id} / {scene_desc} / {current_big}",
                        "display_subtitle": scene_desc,
                        "role": "role4",
                        "scene_desc": scene_desc,
                        "big_segment_id": current_big,
                        "remotion_id": remotion_id,
                    })
        return items

    @staticmethod
    def _derive_legacy_segment_id_from_shot_id(shot_id: str) -> str:
        """从新格式 shot_id（如 shot_0001_1）反推旧格式 segment_id（如 seg_0001）。"""
        m = re.match(r'^shot_(\d+)_\d+$', str(shot_id).strip())
        return f"seg_{m.group(1)}" if m else ""

    def _build_role3_stream_preview_segments(
        self, task_dir: Path, segment_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        功能说明：为 role3 构建 per-big-segment 流式预览段数据。
        参数说明：
        - task_dir: 任务目录。
        - segment_items: role3 segment 选择列表（含 big_segment_id）。
        返回值：
        - list[dict[str, Any]]: 每项含 segment_id/content/updated_at_ms，segment_id 为 big_segment_id。
        异常说明：无。
        """
        streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), "role3")
        seen_bids: set[str] = set()
        segments: list[dict[str, Any]] = []
        for item in segment_items:
            bid = str(item.get("big_segment_id", "")).strip()
            if not bid or bid in seen_bids:
                continue
            seen_bids.add(bid)
            stream_path = streaming_dir / f"role3_segment_output.streaming.{bid}.md"
            content = ""
            updated_at_ms = 0
            if stream_path.exists():
                try:
                    content = stream_path.read_text(encoding="utf-8")
                    updated_at_ms = int(stream_path.stat().st_mtime * 1000)
                except Exception:
                    pass
            segments.append({
                "segment_id": bid,
                "content": content,
                "updated_at_ms": updated_at_ms,
            })
        return segments

    def _build_role3_rendered_prompt_segments(
        self, task_dir: Path, segment_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        功能说明：为 role3 构建 per-big-segment 渲染后 prompt 段数据。
        参数说明：
        - task_dir: 任务目录。
        - segment_items: role3 segment 选择列表（含 big_segment_id）。
        返回值：
        - list[dict[str, Any]]: 每项含 segment_id/content/updated_at_ms，segment_id 为 big_segment_id。
        异常说明：无。
        """
        prompt_dir = get_module_b_prompt_dir((task_dir / "artifacts").resolve(), "role3")
        seen_bids: set[str] = set()
        segments: list[dict[str, Any]] = []
        for item in segment_items:
            bid = str(item.get("big_segment_id", "")).strip()
            if not bid or bid in seen_bids:
                continue
            seen_bids.add(bid)
            prompt_path = prompt_dir / f"role3_rendered_prompt.{bid}.md"
            content = ""
            updated_at_ms = 0
            if prompt_path.exists():
                try:
                    content = prompt_path.read_text(encoding="utf-8")
                    updated_at_ms = int(prompt_path.stat().st_mtime * 1000)
                except Exception:
                    pass
            segments.append({
                "segment_id": bid,
                "content": content,
                "updated_at_ms": updated_at_ms,
            })
        return segments

    def _build_role4_stream_preview_segments(
        self, task_dir: Path, shot_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        功能说明：为 role4 构建 per-shot 流式预览段数据。
        参数说明：
        - task_dir: 任务目录。
        - shot_items: role4 shot 列表。
        返回值：
        - list[dict[str, Any]]: 每项含 segment_id/content/updated_at_ms。
        异常说明：无。
        边界条件：优先尝试新 shot_id 文件名，回退旧 segment_id 文件名。
        """

        streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), "role4")
        segments: list[dict[str, Any]] = []
        for item in shot_items:
            sid = str(item.get("shot_id", "")).strip()
            if not sid:
                continue
            stream_path = streaming_dir / f"role4_prompt_output.streaming.{sid}.md"
            if not stream_path.exists():
                legacy_seg_id = self._derive_legacy_segment_id_from_shot_id(sid)
                if legacy_seg_id:
                    legacy_path = streaming_dir / f"role4_prompt_output.streaming.{legacy_seg_id}.md"
                    if legacy_path.exists():
                        stream_path = legacy_path
            content = ""
            updated_at_ms = 0
            if stream_path.exists():
                try:
                    content = stream_path.read_text(encoding="utf-8")
                    updated_at_ms = int(stream_path.stat().st_mtime * 1000)
                except Exception:
                    pass
            segments.append({
                "segment_id": sid,
                "content": content,
                "updated_at_ms": updated_at_ms,
            })
        return segments

    def _build_role4_rendered_prompt_segments(
        self, task_dir: Path, shot_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        功能说明：为 role4 构建 per-shot 渲染后 prompt 段数据。
        参数说明：
        - task_dir: 任务目录。
        - shot_items: role4 shot 列表。
        返回值：
        - list[dict[str, Any]]: 每项含 segment_id/content/updated_at_ms。
        异常说明：无。
        边界条件：优先尝试新 shot_id 文件名，回退旧 segment_id 文件名。
        """

        prompt_dir = get_module_b_prompt_dir((task_dir / "artifacts").resolve(), "role4")
        segments: list[dict[str, Any]] = []
        for item in shot_items:
            sid = str(item.get("shot_id", "")).strip()
            if not sid:
                continue
            prompt_path = prompt_dir / f"role4_rendered_prompt.{sid}.md"
            if not prompt_path.exists():
                legacy_seg_id = self._derive_legacy_segment_id_from_shot_id(sid)
                if legacy_seg_id:
                    legacy_path = prompt_dir / f"role4_rendered_prompt.{legacy_seg_id}.md"
                    if legacy_path.exists():
                        prompt_path = legacy_path
            content = ""
            updated_at_ms = 0
            if prompt_path.exists():
                try:
                    content = prompt_path.read_text(encoding="utf-8")
                    updated_at_ms = int(prompt_path.stat().st_mtime * 1000)
                except Exception:
                    pass
            segments.append({
                "segment_id": sid,
                "content": content,
                "updated_at_ms": updated_at_ms,
            })
        return segments

    def _find_role_rendered_prompt(self, task_dir: Path, role_name: str) -> Path | None:
        """
        功能说明：查找指定 role 最近的渲染后 prompt 文件。
        参数说明：
        - task_dir: 任务目录。
        - role_name: 角色名。
        返回值：
        - Path | None: 找到则返回文件路径，否则返回 None。
        异常说明：无。
        边界条件：优先 glob 匹配最新文件。
        """

        prompt_dir = get_module_b_prompt_dir((task_dir / "artifacts").resolve(), role_name)
        if not prompt_dir.exists():
            return None
        candidates = sorted(
            [item for item in prompt_dir.glob("*.md") if item.is_file()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return candidates[0].resolve() if candidates else None

    def _handle_module_b_role_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 B role 级重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：当前默认仅提供占位入口，未来接入真实 handler 后复用同一 API。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        role_name = str(query.get("role_name", [""])[0]).strip().lower()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if role_name not in {"role1", "role2", "role3", "role4"}:
            return {"ok": False, "error": f"模块B role 重跑失败：非法 role_name={role_name or '<empty>'}。"}, HTTPStatus.BAD_REQUEST
        if self.module_b_role_rerun_handler is None:
            return {
                "ok": False,
                "error": f"当前 module_b 新方案 role 重跑尚未接通，role_name={role_name}。",
            }, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块B role 重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        active_process_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
        has_active_process = bool(active_process_meta.get("active"))
        if replace_running and (has_active_process or (active_thread is not None and active_thread.is_alive())):
            terminated = self._terminate_active_module_b_rerun_process(task_id=task_id)
            if active_thread is not None and active_thread.is_alive():
                active_thread.join(timeout=5.0)
            if not terminated and active_thread is not None and active_thread.is_alive():
                return {
                    "ok": False,
                    "error": f"模块B role 重跑失败：旧进程仍在退出中，task_id={task_id}",
                }, HTTPStatus.CONFLICT
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"模块B role 重跑失败：任务已有后台动作执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        if self._load_active_module_b_rerun_process_meta(task_id=task_id).get("active"):
            return {"ok": False, "error": f"模块B role 重跑失败：任务已有后台子进程执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        rerun_thread = threading.Thread(
            target=self._run_module_b_role_rerun_in_background,
            name=f"module-b-role-rerun-{task_id}-{role_name}",
            args=(task_id, role_name),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        self._rerun_thread_meta[task_id] = {
            "active": True,
            "status": "queued",
            "mode": "role",
            "role_name": role_name,
            "segment_id": "",
            "shot_id": "",
            "submitted_at": current_time_text(),
            "submitted_at_ms": int(time.time() * 1000),
            "started_at": "",
            "started_at_ms": 0,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()
        self.logger.info("[监督服务] 模块B role 重跑已提交，task_id=%s，role_name=%s", task_id, role_name)
        return {
            "ok": True,
            "task_id": task_id,
            "message": f"模块B role 重跑已提交，task_id={task_id}，role_name={role_name}",
        }, HTTPStatus.OK
    def _handle_module_b_role_segment_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 B role 内 segment 级重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：当前仅开放 role3/role4；shot_id 由当前任务 segment 映射推导。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        role_name = str(query.get("role_name", [""])[0]).strip().lower()
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if role_name not in {"role3", "role4"}:
            return {
                "ok": False,
                "error": f"模块B segment 重跑失败：仅支持 role3/role4，role_name={role_name or '<empty>'}。",
            }, HTTPStatus.BAD_REQUEST
        if not segment_id:
            return {"ok": False, "error": "模块B segment 重跑失败：segment_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        if self.module_b_role_segment_rerun_handler is None:
            return {
                "ok": False,
                "error": f"当前 module_b 新方案 segment 重跑尚未接通，role_name={role_name}，segment_id={segment_id}。",
            }, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块B segment 重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        active_process_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
        has_active_process = bool(active_process_meta.get("active"))
        if replace_running and (has_active_process or (active_thread is not None and active_thread.is_alive())):
            terminated = self._terminate_active_module_b_rerun_process(task_id=task_id)
            if active_thread is not None and active_thread.is_alive():
                active_thread.join(timeout=5.0)
            if not terminated and active_thread is not None and active_thread.is_alive():
                return {
                    "ok": False,
                    "error": f"模块B segment 重跑失败：旧进程仍在退出中，task_id={task_id}",
                }, HTTPStatus.CONFLICT
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"模块B segment 重跑失败：任务已有后台动作执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        if self._load_active_module_b_rerun_process_meta(task_id=task_id).get("active"):
            return {"ok": False, "error": f"模块B segment 重跑失败：任务已有后台子进程执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        task_dir = self._resolve_task_dir(task_id=task_id)
        shot_id = self._resolve_module_b_shot_id_from_segment(task_dir=task_dir, task_id=task_id, segment_id=segment_id, role_name=role_name)
        if not shot_id:
            segment_display = "big_segment" if role_name == "role3" else "shot"
            segment_key = "big_segment_id" if role_name == "role3" else "shot_id"
            return {
                "ok": False,
                "error": f"模块B role{role_name} {segment_display} 重跑失败：无法从当前任务解析 {segment_key}，{segment_key}={segment_id}。",
            }, HTTPStatus.NOT_FOUND
        if role_name == "role3":
            shot_id = ""
        rerun_thread = threading.Thread(
            target=self._run_module_b_role_segment_rerun_in_background,
            name=f"module-b-role-segment-rerun-{task_id}-{role_name}-{segment_id}",
            args=(task_id, role_name, segment_id, shot_id),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        self._rerun_thread_meta[task_id] = {
            "active": True,
            "status": "queued",
            "mode": "segment",
            "role_name": role_name,
            "segment_id": segment_id,
            "shot_id": shot_id,
            "submitted_at": current_time_text(),
            "submitted_at_ms": int(time.time() * 1000),
            "started_at": "",
            "started_at_ms": 0,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        # 提交后立即清空 streaming 文件，避免轮询读到旧内容
        _clear_role_streaming_file(task_dir=task_dir, role_name=role_name, segment_id=segment_id, shot_id=shot_id)

        rerun_thread.start()
        if role_name == "role3":
            self.logger.info(
                "[监督服务] 模块B segment 重跑已提交，task_id=%s，role_name=%s，big_segment_id=%s",
                task_id,
                role_name,
                segment_id,
            )
            return {
                "ok": True,
                "task_id": task_id,
                "message": f"模块B segment 重跑已提交，task_id={task_id}，role_name={role_name}，big_segment_id={segment_id}",
            }, HTTPStatus.OK
        self.logger.info(
            "[监督服务] 模块B segment 重跑已提交，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
            task_id,
            role_name,
            segment_id,
            shot_id,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": (
                f"模块B segment 重跑已提交，task_id={task_id}，role_name={role_name}，"
                f"segment_id={segment_id}，shot_id={shot_id}"
            ),
        }, HTTPStatus.OK
    def _handle_module_b_rebuild_output_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：根据已有的 role3/role4 markdown 产物重新生成 module_b_output.json，不调用 LLM。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：执行成功会覆盖已有的 module_b_output.json。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        if not artifacts_dir.is_dir():
            return {"ok": False, "error": f"任务产物目录不存在：{artifacts_dir}"}, HTTPStatus.NOT_FOUND

        # 读取 module_a_output.json（需要 segment 时间信息）
        module_a_path = artifacts_dir / "module_a_output.json"
        try:
            module_a_output = json.loads(module_a_path.read_text(encoding="utf-8"))
        except Exception:
            module_a_output = {}

        # 获取已完成单元记录
        done_unit_records = self.state_store.list_module_units_by_status(
            task_id=task_id,
            module_name="B",
            statuses=["done"],
        )

        # 重建输出
        from music_video_pipeline.modules.module_b.output_builder import build_module_b_output
        output = build_module_b_output(
            done_unit_records=list(done_unit_records or []),
            module_a_output=module_a_output,
            instrumental_labels=[],
            artifacts_dir=artifacts_dir,
        )

        output_path = artifacts_dir / "module_b_output.json"
        try:
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            return {
                "ok": False,
                "error": f"重建模块 B 输出失败：写文件出错，{error}",
            }, HTTPStatus.INTERNAL_SERVER_ERROR

        self.logger.info(
            "[监督服务] 模块 B 输出已重建，task_id=%s，shot_count=%s",
            task_id,
            len(output),
        )

        # 同步重建模块 C 单元，使 module-c 页面立即以 shot_id 展示
        try:
            from music_video_pipeline.modules.module_c.unit_models import (
                build_module_c_units,
                build_unit_sync_payload,
            )
            c_units = build_module_c_units(shots=output)
            self.state_store.sync_module_units(
                task_id=task_id,
                module_name="C",
                units=build_unit_sync_payload(units=c_units),
            )
            self.logger.info(
                "[监督服务] 模块 C 单元已同步重建，task_id=%s，unit_count=%s",
                task_id,
                len(c_units),
            )
        except Exception as sync_error:
            self.logger.warning(
                "[监督服务] 重建模块 B 输出后同步模块 C 单元失败（可忽略），task_id=%s，错误=%s",
                task_id,
                sync_error,
            )

        return {
            "ok": True,
            "task_id": task_id,
            "message": f"模块 B 输出已重建，shot_count={len(output)}",
        }, HTTPStatus.OK
    def _handle_module_b_resume_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 B 断点续跑请求（扫描 role3 输出的所有 shot，对缺少 role4 产物的 shot 逐个补跑）。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：已有产物的 shot 会跳过。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND

        active_thread = self._rerun_threads.get(task_id)
        if active_thread is not None and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"模块B 断点续跑失败：任务已有后台动作执行中，task_id={task_id}",
            }, HTTPStatus.CONFLICT

        config_path_text = str(task_record.get("config_path", "")).strip()
        if not config_path_text:
            return {"ok": False, "error": f"任务缺少 config_path，task_id={task_id}"}, HTTPStatus.NOT_FOUND

        from pathlib import Path as _Path
        workspace_root = _Path(task_record.get("workspace_root", "")) if task_record.get("workspace_root") else None
        if workspace_root is None:
            workspace_root = self._resolve_project_root()

        rerun_thread = threading.Thread(
            target=self._run_module_b_resume_in_background,
            name=f"module-b-resume-{task_id}",
            args=(task_id, config_path_text, workspace_root),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        self._rerun_thread_meta[task_id] = {
            "active": True,
            "status": "queued",
            "mode": "resume",
            "submitted_at": current_time_text(),
            "submitted_at_ms": int(time.time() * 1000),
        }
        rerun_thread.start()
        self.logger.info("[监督服务] 模块B 断点续跑已提交，task_id=%s", task_id)
        return {
            "ok": True,
            "task_id": task_id,
            "message": "模块 B 断点续跑已提交，后台开始扫描缺失 shot 并逐个补跑。",
        }, HTTPStatus.OK
    def _resolve_module_b_shot_id_from_segment(self, task_dir: Path, task_id: str, segment_id: str, role_name: str = "") -> str:
        """
        功能说明：根据当前任务的 segment 上下文解析对应 shot_id。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        - segment_id: 目标 segment 标识。
        - role_name: 角色名；为 "role3" 时按 big_segment_id 匹配。
        返回值：
        - str: 命中的 shot_id；找不到时返回空字符串。
        异常说明：无。
        边界条件：优先复用当前模块 B 页面使用的 segment 列表。
        """
        normalized_segment_id = str(segment_id).strip()
        if not normalized_segment_id:
            return ""
        normalized_role_name = str(role_name).strip().lower()
        for item in self._load_module_b_segment_selector_items(task_dir=task_dir, task_id=task_id):
            if normalized_role_name == "role3":
                if str(item.get("big_segment_id", "")).strip() == normalized_segment_id:
                    return str(item.get("shot_id", "")).strip()
            if str(item.get("segment_id", "")).strip() == normalized_segment_id:
                return str(item.get("shot_id", "")).strip()
        # role4 的 segment_id 本身就是 shot_id（如 shot_0001_1），直接返回
        for item in self._load_role4_shot_selector_items(task_dir=task_dir):
            if str(item.get("segment_id", "")).strip() == normalized_segment_id:
                return str(item.get("shot_id", "")).strip()
        return ""
    def _run_module_b_role_rerun_in_background(self, task_id: str, role_name: str) -> None:
        """
        功能说明：在后台线程中执行模块 B role 级重跑。
        参数说明：
        - task_id: 任务唯一标识。
        - role_name: 模块 B 角色名。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        started_at_ms = int(time.time() * 1000)
        meta = self._rerun_thread_meta.get(task_id)
        if isinstance(meta, dict):
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at"] = current_time_text()
            meta["started_at_ms"] = started_at_ms
            meta["finished_at"] = ""
            meta["finished_at_ms"] = 0
            meta["duration_ms"] = 0
            meta["last_error"] = ""
            meta["failure_reason"] = ""
        try:
            self.logger.info("[监督服务] 后台开始执行模块B role 重跑，task_id=%s，role_name=%s", task_id, role_name)
            if self.module_b_role_rerun_handler is None:
                raise RuntimeError(f"模块B role 重跑 handler 缺失，task_id={task_id}，role_name={role_name}")
            self.module_b_role_rerun_handler(task_id, role_name)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                self._save_completed_module_b_rerun_meta(task_id=task_id, meta=meta)
            self.logger.info("[监督服务] 后台模块B role 重跑执行结束，task_id=%s，role_name=%s", task_id, role_name)
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = self._classify_module_b_rerun_failure_reason(error)
                self._save_completed_module_b_rerun_meta(task_id=task_id, meta=meta)
            self.logger.error(
                "[监督服务] 后台模块B role 重跑失败，task_id=%s，role_name=%s，错误信息=%s",
                task_id,
                role_name,
                error,
            )
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
    def _run_module_b_role_segment_rerun_in_background(
        self,
        task_id: str,
        role_name: str,
        segment_id: str,
        shot_id: str,
    ) -> None:
        """
        功能说明：在后台线程中执行模块 B role 内 segment 级重跑。
        参数说明：
        - task_id: 任务唯一标识。
        - role_name: 模块 B 角色名。
        - segment_id: 目标 segment 标识。
        - shot_id: 解析后的目标 shot_id。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        started_at_ms = int(time.time() * 1000)
        meta = self._rerun_thread_meta.get(task_id)
        if isinstance(meta, dict):
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at"] = current_time_text()
            meta["started_at_ms"] = started_at_ms
            meta["finished_at"] = ""
            meta["finished_at_ms"] = 0
            meta["duration_ms"] = 0
            meta["last_error"] = ""
            meta["failure_reason"] = ""
        try:
            if role_name == "role3":
                self.logger.info(
                    "[监督服务] 后台开始执行模块B segment 重跑，task_id=%s，role_name=%s，big_segment_id=%s",
                    task_id,
                    role_name,
                    segment_id,
                )
            else:
                self.logger.info(
                    "[监督服务] 后台开始执行模块B segment 重跑，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
                    task_id,
                    role_name,
                    segment_id,
                    shot_id,
                )
            if self.module_b_role_segment_rerun_handler is None:
                raise RuntimeError(
                    f"模块B segment 重跑 handler 缺失，task_id={task_id}，role_name={role_name}，shot_id={shot_id}"
                )
            # role3 按 big_segment_id 重跑，不再走 shot 级路由
            handler_shot_id = segment_id if role_name == "role3" else shot_id
            self.module_b_role_segment_rerun_handler(task_id, role_name, handler_shot_id)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                self._save_completed_module_b_rerun_meta(task_id=task_id, meta=meta)
            if role_name == "role3":
                self.logger.info(
                    "[监督服务] 后台模块B segment 重跑执行结束，task_id=%s，role_name=%s，big_segment_id=%s",
                    task_id,
                    role_name,
                    segment_id,
                )
            else:
                self.logger.info(
                    "[监督服务] 后台模块B segment 重跑执行结束，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
                    task_id,
                    role_name,
                    segment_id,
                    shot_id,
                )
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = self._classify_module_b_rerun_failure_reason(error)
                self._save_completed_module_b_rerun_meta(task_id=task_id, meta=meta)
            self.logger.error(
                "[监督服务] 后台模块B segment 重跑失败，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s，错误信息=%s",
                task_id,
                role_name,
                segment_id,
                shot_id,
                error,
            )
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
    def _run_module_b_resume_in_background(self, task_id: str, config_path_text: str, workspace_root: Path) -> None:
        """
        功能说明：在后台线程中执行模块 B 断点续跑，通过 CLI 子进程执行 resume 命令。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path_text: 配置文件路径。
        - workspace_root: 工作区根目录。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        import sys
        started_at_ms = int(time.time() * 1000)
        meta = self._rerun_thread_meta.get(task_id)
        if isinstance(meta, dict):
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at"] = current_time_text()
            meta["started_at_ms"] = started_at_ms
        try:
            self.logger.info("[监督服务] 后台开始执行模块B 断点续跑，task_id=%s", task_id)
            command = [
                sys.executable,
                "-m",
                "music_video_pipeline.cli",
                "resume",
                "--task-id",
                task_id,
                "--config",
                config_path_text,
            ]
            completed = subprocess.run(
                command,
                cwd=str(workspace_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if completed.returncode != 0:
                error_excerpt = (completed.stderr or "").strip() or (completed.stdout or "").strip()
                raise RuntimeError(f"断点续跑子进程退出码={completed.returncode}，{error_excerpt[:500]}")
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - started_at_ms)
            self.logger.info("[监督服务] 后台模块B 断点续跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - started_at_ms)
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = self._classify_module_b_rerun_failure_reason(error)
            self.logger.error(
                "[监督服务] 后台模块B 断点续跑失败，task_id=%s，错误=%s",
                task_id,
                error,
            )
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)


def _clear_role_streaming_file(task_dir: Path, role_name: str, segment_id: str, shot_id: str) -> None:
    """
    功能说明：清空 role3/role4 对应 segment/shot 的 streaming 文件，避免轮询读到旧内容。
    参数说明：
    - task_dir: 任务目录。
    - role_name: 角色名（role3/role4）。
    - segment_id: 目标 segment 标识（role3 为 big_segment_id，role4 为 segment_id）。
    - shot_id: 目标 shot 标识（role4 使用）。
    返回值：无。
    异常说明：文件不存在或写失败时静默忽略。
    """
    safe_role_name = str(role_name or "").strip().lower()
    if safe_role_name not in {"role3", "role4"}:
        return
    try:
        streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), safe_role_name)
    except Exception:
        return
    if safe_role_name == "role3":
        stream_path = streaming_dir / f"role3_segment_output.streaming.{segment_id}.md"
    else:
        if not shot_id:
            return
        stream_path = streaming_dir / f"role4_prompt_output.streaming.{shot_id}.md"
    if stream_path.exists():
        try:
            stream_path.write_text("", encoding="utf-8")
        except Exception:
            pass
