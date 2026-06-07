"""
文件用途：模块 C handler mixin —— 构建模块 C 页面负载、处理 shot/帧重跑请求。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store、模块 C 执行器、项目内工具函数。
维护说明：本文件仅包含模块 C 专属方法，不引入其他模块的耦合。
"""

import json
import re
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import requests

from music_video_pipeline.modules.module_a_v2.network_lyrics_state import current_time_text
from music_video_pipeline.monitoring.routes import (
    TASK_MODULE_C_API_PATH,
    TASK_MODULE_C_RERUN_FRAME_API_PATH,
    TASK_MODULE_C_REBUILD_UNITS_API_PATH,
    TASK_MODULE_C_RERUN_SHOT_API_PATH,
)



def _build_module_c_frame_rerun_key(task_id: str, shot_id: str, frame_type: str) -> str:
    """构建模块 C 单帧重跑的独立并发键。"""
    return f"{str(task_id).strip()}|module_c_frame|{str(shot_id).strip()}|{str(frame_type).strip().lower()}"


def _build_module_c_shot_rerun_key(task_id: str, shot_id: str) -> str:
    """构建模块 C shot 级重跑的独立并发键。"""
    return f"{str(task_id).strip()}|module_c_shot|{str(shot_id).strip()}"


def _classify_module_c_rerun_failure_reason(error: Exception) -> str:
    """把模块 C 重跑异常归类为更易读的失败原因。"""
    error_text = str(error).strip()
    lowered_text = error_text.lower()
    if "shot_id 不存在" in error_text or "尚未建立单元状态" in error_text:
        return "目标镜头尚未建立模块C单元"
    if "exit_code=1" in error_text:
        return "模块C子进程执行失败"
    if "timeout" in lowered_text or "timed out" in lowered_text or "超时" in error_text:
        return "模块C执行超时"
    return error_text or "模块C重跑失败"


class ModuleCHandlers:
    """Mixin —— 模块 C 相关方法。"""

    @staticmethod
    def _interrupt_comfyui(app_config: Any) -> None:
        """发送中断请求到 ComfyUI，使其正在执行的 prompt 立即停止。"""
        try:
            server_url = str(getattr(getattr(app_config, "comfyui", None), "server_url", "http://127.0.0.1:8188"))
            requests.post(f"{server_url.rstrip('/')}/interrupt", timeout=5.0)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _parse_role4_streaming_prompt(role4_streaming_dir: Path, shot_id: str) -> dict[str, str]:
        """
        功能说明：读取 per-shot streaming 文件，提取 6 个 prompt 字段。
        参数说明：
        - role4_streaming_dir: role4 streaming 目录。
        - shot_id: 目标 shot 标识（如 shot_0004_1）。
        返回值：
        - dict[str, str]: 包含 prompts 的字典；文件不存在或解析失败时返回空 dict。
        边界条件：per-shot 文件由 role4 直接写入，格式为 `- field_name: content`。
        """
        if not str(shot_id).strip():
            return {}
        shot_path = role4_streaming_dir / f"role4_prompt_output.streaming.{shot_id.strip()}.md"
        if not shot_path.exists():
            return {}
        try:
            text = shot_path.read_text(encoding="utf-8")
        except Exception:
            return {}
        result: dict[str, str] = {}
        fields = [
            "subject_kind",
            "keyframe_prompt_start_zh",
            "keyframe_prompt_start_en",
            "keyframe_prompt_end_zh",
            "keyframe_prompt_end_en",
            "video_prompt_zh",
            "video_prompt_en",
        ]
        for field in fields:
            m = re.search(rf"^- {re.escape(field)}:\s*(.*)", text, re.MULTILINE)
            result[field] = m.group(1).strip() if m else ""
        return result

    def _build_module_c_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 C 页面所需的数据负载，含 shot 列表与首尾帧状态。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含 shot 列表、帧URL与活跃重跑状态的数据对象。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：首尾帧文件缺失时 frame_url 置空。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
                "module_c_status": "unknown",
                "unit_summary": self._build_empty_module_unit_summary(module_name="C"),
                "shots": [],
                "active_rerun": None,
            }
        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:  # noqa: BLE001
            module_status_map = {}
        try:
            unit_summary = self.state_store.get_module_unit_status_summary(
                task_id=normalized_task_id,
                module_name="C",
            )
        except Exception:  # noqa: BLE001
            unit_summary = self._build_empty_module_unit_summary(module_name="C")

        # 收集当前正在被重跑的 shot_id 集合，自愈逻辑跳过它们
        active_rerun_shot_ids: set[str] = set()
        for key, meta in self._rerun_thread_meta.items():
            if key.startswith(normalized_task_id + "|module_c") and meta.get("active"):
                active_rerun_shot_ids.add(str(meta.get("shot_id", "")).strip())

        all_units = self.state_store.list_module_units_by_status(
            task_id=task_id,
            module_name="C",
            statuses=["pending", "running", "done", "failed"],
        )
        shots: list[dict[str, Any]] = []
        for unit in sorted(all_units, key=lambda row: int(row.get("unit_index", 0))):
            shot_id = str(unit.get("unit_id", "")).strip()
            status = str(unit.get("status", "pending")).strip()
            start_time = float(unit.get("start_time", 0) or 0)
            end_time = float(unit.get("end_time", 0) or 0)
            duration = round(max(0.0, end_time - start_time), 2)
            error_message = str(unit.get("error_message", "")).strip()

            frame_status_start = str(unit.get("frame_status_start", "pending")).strip()
            frame_status_end = str(unit.get("frame_status_end", "pending")).strip()
            segment_id = str(unit.get("segment_id", "")).strip()

            frame_url_start = ""
            frame_url_end = ""
            task_dir = self._resolve_task_dir(task_id=normalized_task_id)
            frames_dir = task_dir / "artifacts" / "frames"
            frame_path_start = frames_dir / f"{shot_id}_start.png"
            frame_path_end = frames_dir / f"{shot_id}_end.png"
            try:
                if frame_path_start.exists():
                    frame_url_start = self._build_task_file_url(
                        task_id=normalized_task_id,
                        file_path=frame_path_start,
                    )
                    # 用文件修改时间做缓存破坏
                    frame_url_start += f"?t={int(frame_path_start.stat().st_mtime)}"
                if frame_path_end.exists():
                    frame_url_end = self._build_task_file_url(
                        task_id=normalized_task_id,
                        file_path=frame_path_end,
                    )
                    frame_url_end += f"?t={int(frame_path_end.stat().st_mtime)}"
            except Exception:  # noqa: BLE001
                pass

            # 读取 role4 streaming prompt（按 seg 文件 + subject_index）
            role4_streaming_dir = task_dir / "artifacts" / "module_b_work" / "role4" / "streaming"
            role4_prompt = self._parse_role4_streaming_prompt(role4_streaming_dir, shot_id)

            # 从 config 读取正向 prompt 前缀/后缀（与 ComfyUIFrameGenerator._assemble_prompt 保持一致）
            comfyui_cfg = self.app_config.module_c.comfyui if hasattr(self, "app_config") else None
            prompt_prefix = str(getattr(comfyui_cfg, "prompt_prefix", "")).strip() if comfyui_cfg else ""
            prompt_suffix = str(getattr(comfyui_cfg, "prompt_suffix", "")).strip() if comfyui_cfg else ""

            # 场景类主体跳过白色背景 suffix，避免与场景描述冲突
            shot_subject_kind = str(role4_prompt.get("subject_kind", "character") or "character").strip().lower()
            effective_suffix = prompt_suffix if shot_subject_kind != "scene" else ""

            # 组装完整 prompt：前缀 + LLM 输出 + （可选后缀）
            assembled_prompt_start = prompt_prefix
            if role4_prompt.get("keyframe_prompt_start_en", "").strip():
                assembled_prompt_start += "\n" + role4_prompt["keyframe_prompt_start_en"].strip()
            if effective_suffix:
                assembled_prompt_start += "\n" + effective_suffix

            assembled_prompt_end = prompt_prefix
            if role4_prompt.get("keyframe_prompt_end_en", "").strip():
                assembled_prompt_end += "\n" + role4_prompt["keyframe_prompt_end_en"].strip()
            if effective_suffix:
                assembled_prompt_end += "\n" + effective_suffix

            # 自愈：首尾帧都已 done 但 unit 状态不对时，覆盖为 done
            # 跳过正在被重跑的 shot，避免把 running 状态改回去
            if (
                shot_id not in active_rerun_shot_ids
                and frame_status_start == "done"
                and frame_status_end == "done"
                and status != "done"
            ):
                status = "done"

            shots.append({
                "shot_id": shot_id,
                "unit_index": int(unit.get("unit_index", 0)),
                "segment_id": segment_id,
                "status": status,
                "frame_status_start": frame_status_start,
                "frame_status_end": frame_status_end,
                "frame_url_start": frame_url_start,
                "frame_url_end": frame_url_end,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "error_message": error_message,
                "role4_prompt": role4_prompt,
                "assembled_prompt_start": assembled_prompt_start,
                "assembled_prompt_end": assembled_prompt_end,
            })

        active_rerun: dict[str, Any] | None = None
        for key, meta in self._rerun_thread_meta.items():
            if key.startswith(normalized_task_id + "|module_c"):
                active_rerun = {
                    "active": bool(meta.get("active")),
                    "status": str(meta.get("status", "")).strip(),
                    "shot_id": str(meta.get("shot_id", "")).strip(),
                    "frame_type": str(meta.get("frame_type", "")).strip(),
                    "submitted_at": str(meta.get("submitted_at", "")).strip(),
                    "submitted_at_ms": int(meta.get("submitted_at_ms", 0) or 0),
                    "started_at_ms": int(meta.get("started_at_ms", 0) or 0),
                    "last_error": str(meta.get("last_error", "")).strip(),
                    "failure_reason": str(meta.get("failure_reason", "")).strip(),
                }
                break

        if active_rerun and str(active_rerun.get("shot_id", "")).strip():
            active_shot_id = str(active_rerun.get("shot_id", "")).strip()
            active_status = str(active_rerun.get("status", "")).strip().lower()
            if active_status in {"queued", "running"}:
                for shot_item in shots:
                    if str(shot_item.get("shot_id", "")).strip() != active_shot_id:
                        continue
                    shot_item["status"] = "running"
                    if not str(shot_item.get("error_message", "")).strip():
                        shot_item["error_message"] = ""
                    break

        return {
            "ok": True,
            "task_id": normalized_task_id,
            "module_c_status": str(module_status_map.get("C", "unknown")),
            "unit_summary": unit_summary,
            "shots": shots,
            "active_rerun": active_rerun,
        }

    def _handle_module_c_shot_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 C shot 级重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：同一 shot 不允许并发重复触发。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        shot_id = str(query.get("shot_id", [""])[0]).strip()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if not shot_id:
            return {"ok": False, "error": "模块C shot 重跑失败：shot_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        if self.module_c_shot_rerun_handler is None:
            return {"ok": False, "error": "当前模块 C shot 重跑尚未接通。"}, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块C shot 重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND

        rerun_key = _build_module_c_shot_rerun_key(task_id, shot_id)
        active_thread = self._rerun_threads.get(rerun_key)
        if replace_running and active_thread is not None and active_thread.is_alive():
            self._interrupt_comfyui(self.app_config)
            active_thread.join(timeout=10.0)
            self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)
        if active_thread is not None and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"模块C shot 重跑失败：该 shot 已有后台动作执行中，task_id={task_id}，shot_id={shot_id}",
            }, HTTPStatus.CONFLICT

        rerun_thread = threading.Thread(
            target=self._run_module_c_shot_rerun_in_background,
            name=f"module-c-shot-rerun-{task_id}-{shot_id}",
            args=(task_id, shot_id),
            daemon=True,
        )
        try:
            self.state_store.set_module_unit_status(
                task_id=task_id,
                module_name="C",
                unit_id=shot_id,
                status="running",
                artifact_path="",
                error_message="",
            )
            self.state_store.set_module_status(
                task_id=task_id,
                module_name="C",
                status="running",
                artifact_path="",
                error_message="",
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning(
                "[监督服务] 模块C shot 重跑提交时预写 running 状态失败，task_id=%s，shot_id=%s，错误=%s",
                task_id,
                shot_id,
                error,
            )
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "mode": "shot",
            "role_name": "module_c",
            "segment_id": "",
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
        rerun_thread.start()
        self.logger.info("[监督服务] 模块C shot 重跑已提交，task_id=%s，shot_id=%s", task_id, shot_id)
        return {
            "ok": True,
            "task_id": task_id,
            "shot_id": shot_id,
            "message": f"模块C shot 重跑已提交，task_id={task_id}，shot_id={shot_id}",
        }, HTTPStatus.OK

    def _handle_module_c_frame_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 C 单帧重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：frame_type 保留在 meta 中供后续精细化。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        shot_id = str(query.get("shot_id", [""])[0]).strip()
        frame_type = str(query.get("frame_type", [""])[0]).strip()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if not shot_id:
            return {"ok": False, "error": "模块C 帧重跑失败：shot_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        if frame_type not in {"start", "end"}:
            return {"ok": False, "error": "模块C 帧重跑失败：frame_type 必须为 start 或 end。"}, HTTPStatus.BAD_REQUEST
        if self.module_c_frame_rerun_handler is None:
            return {"ok": False, "error": "当前模块 C 帧重跑尚未接通。"}, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块C 帧重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND

        rerun_key = _build_module_c_frame_rerun_key(task_id, shot_id, frame_type)
        active_thread = self._rerun_threads.get(rerun_key)
        if replace_running and active_thread is not None and active_thread.is_alive():
            self._interrupt_comfyui(self.app_config)
            active_thread.join(timeout=10.0)
            self._rerun_threads.pop(rerun_key, None)
            self._rerun_thread_meta.pop(rerun_key, None)
        if active_thread is not None and active_thread.is_alive():
            return {
                "ok": False,
                "error": f"模块C 帧重跑失败：该 shot 已有后台动作执行中，task_id={task_id}，shot_id={shot_id}",
            }, HTTPStatus.CONFLICT

        rerun_thread = threading.Thread(
            target=self._run_module_c_frame_rerun_in_background,
            name=f"module-c-frame-rerun-{task_id}-{shot_id}-{frame_type}",
            args=(task_id, shot_id, frame_type),
            daemon=True,
        )
        try:
            self.state_store.set_module_unit_status(
                task_id=task_id,
                module_name="C",
                unit_id=shot_id,
                status="running",
                artifact_path="",
                error_message="",
            )
            # 重置被重跑帧的状态，让前端立即反映"正在重跑"
            self.state_store.set_module_unit_frame_status(
                task_id=task_id, module_name="C", unit_id=shot_id,
                frame_type=normalized_frame_type, status="running",
            )
            self.state_store.set_module_status(
                task_id=task_id,
                module_name="C",
                status="running",
                artifact_path="",
                error_message="",
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning(
                "[监督服务] 模块C单帧重跑提交时预写 running 状态失败，task_id=%s，shot_id=%s，frame_type=%s，错误=%s",
                task_id,
                shot_id,
                frame_type,
                error,
            )
        self._rerun_threads[rerun_key] = rerun_thread
        self._rerun_thread_meta[rerun_key] = {
            "active": True,
            "status": "queued",
            "mode": "frame",
            "role_name": "module_c",
            "segment_id": "",
            "shot_id": shot_id,
            "frame_type": frame_type,
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
        self.logger.info(
            "[监督服务] 模块C 帧重跑已提交，task_id=%s，shot_id=%s，frame_type=%s",
            task_id, shot_id, frame_type,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "shot_id": shot_id,
            "frame_type": frame_type,
            "message": f"模块C 帧重跑已提交，task_id={task_id}，shot_id={shot_id}，frame_type={frame_type}",
        }, HTTPStatus.OK

    def _run_module_c_shot_rerun_in_background(self, task_id: str, shot_id: str) -> None:
        """在后台线程中执行模块 C shot 重跑。"""
        started_at_ms = int(time.time() * 1000)
        rerun_key = _build_module_c_shot_rerun_key(task_id, shot_id)
        meta = self._rerun_thread_meta.get(rerun_key)
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
            self.logger.info(
                "[监督服务] 后台开始执行模块C shot 重跑，task_id=%s，shot_id=%s",
                task_id, shot_id,
            )
            if self.module_c_shot_rerun_handler is None:
                raise RuntimeError(f"模块C shot 重跑 handler 缺失，task_id={task_id}，shot_id={shot_id}")
            self.module_c_shot_rerun_handler(task_id, shot_id)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(rerun_key)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
            self.logger.info(
                "[监督服务] 后台模块C shot 重跑执行结束，task_id=%s，shot_id=%s",
                task_id, shot_id,
            )
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(rerun_key)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = _classify_module_c_rerun_failure_reason(error)
            try:
                self.state_store.set_module_unit_status(
                    task_id=task_id,
                    module_name="C",
                    unit_id=shot_id,
                    status="failed",
                    error_message=str(error).strip(),
                )
                self.state_store.set_module_status(
                    task_id=task_id,
                    module_name="C",
                    status="failed",
                    artifact_path="",
                    error_message=str(error).strip(),
                )
            except Exception as persist_error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 回写模块C shot 重跑失败状态时出错，task_id=%s，shot_id=%s，错误=%s",
                    task_id,
                    shot_id,
                    persist_error,
                )
            self.logger.error(
                "[监督服务] 后台模块C shot 重跑失败，task_id=%s，shot_id=%s，错误信息=%s",
                task_id, shot_id, error,
            )
        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(rerun_key, None)

    def _run_module_c_frame_rerun_in_background(self, task_id: str, shot_id: str, frame_type: str) -> None:
        """在后台线程中执行模块 C 单帧重跑。"""
        started_at_ms = int(time.time() * 1000)
        rerun_key = _build_module_c_frame_rerun_key(task_id, shot_id, frame_type)
        meta = self._rerun_thread_meta.get(rerun_key)
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
            self.logger.info(
                "[监督服务] 后台开始执行模块C单帧重跑，task_id=%s，shot_id=%s，frame_type=%s",
                task_id,
                shot_id,
                frame_type,
            )
            if self.module_c_frame_rerun_handler is None:
                raise RuntimeError(
                    f"模块C单帧重跑 handler 缺失，task_id={task_id}，shot_id={shot_id}，frame_type={frame_type}"
                )
            self.module_c_frame_rerun_handler(task_id, shot_id, frame_type)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(rerun_key)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
            self.logger.info(
                "[监督服务] 后台模块C单帧重跑执行结束，task_id=%s，shot_id=%s，frame_type=%s",
                task_id,
                shot_id,
                frame_type,
            )
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(rerun_key)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = _classify_module_c_rerun_failure_reason(error)
            try:
                self.state_store.set_module_unit_status(
                    task_id=task_id,
                    module_name="C",
                    unit_id=shot_id,
                    status="failed",
                    error_message=str(error).strip(),
                )
                self.state_store.set_module_status(
                    task_id=task_id,
                    module_name="C",
                    status="failed",
                    artifact_path="",
                    error_message=str(error).strip(),
                )
            except Exception as persist_error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 回写模块C单帧重跑失败状态时出错，task_id=%s，shot_id=%s，frame_type=%s，错误=%s",
                    task_id,
                    shot_id,
                    frame_type,
                    persist_error,
                )
            self.logger.error(
                "[监督服务] 后台模块C单帧重跑失败，task_id=%s，shot_id=%s，frame_type=%s，错误信息=%s",
                task_id,
                shot_id,
                frame_type,
                error,
            )
        finally:
            current_thread = self._rerun_threads.get(rerun_key)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(rerun_key, None)

    def _handle_module_c_rebuild_units_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：从当前 module_b_output.json 重建模块 C 单元列表并同步到 state_store。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：同步时会清除 state_store 中不再存在的旧 unit_id。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        module_b_path = artifacts_dir / "module_b_output.json"
        if not module_b_path.exists():
            return {"ok": False, "error": f"模块 B 输出不存在：{module_b_path}"}, HTTPStatus.NOT_FOUND
        try:
            module_b_output = json.loads(module_b_path.read_text(encoding="utf-8"))
        except Exception as error:
            return {"ok": False, "error": f"读取 module_b_output.json 失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
        if not isinstance(module_b_output, list):
            return {"ok": False, "error": "module_b_output.json 格式错误：应为数组"}, HTTPStatus.INTERNAL_SERVER_ERROR

        from music_video_pipeline.modules.module_c.unit_models import (
            build_module_c_units,
            build_unit_sync_payload,
        )
        try:
            units = build_module_c_units(shots=module_b_output)
        except Exception as error:
            return {"ok": False, "error": f"构建模块 C 单元失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR

        try:
            self.state_store.sync_module_units(
                task_id=task_id,
                module_name="C",
                units=build_unit_sync_payload(units=units),
            )
        except Exception as error:
            return {"ok": False, "error": f"同步模块 C 单元失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR

        self.logger.info(
            "[监督服务] 模块 C 单元已重建，task_id=%s，unit_count=%s",
            task_id,
            len(units),
        )
        return {
            "ok": True,
            "task_id": task_id,
            "unit_count": len(units),
            "message": f"模块 C 单元已重建，共 {len(units)} 个单元",
        }, HTTPStatus.OK
