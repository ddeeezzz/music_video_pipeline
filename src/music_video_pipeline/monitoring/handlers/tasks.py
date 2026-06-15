"""
文件用途：任务 handler mixin —— 任务列表、详情、新建、改名、复制、重跑。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 state_store 与项目内路径工具。
维护说明：本文件仅包含任务 CRUD 与重跑编排方法。
"""

import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
import subprocess as _subprocess

from music_video_pipeline.monitoring.routes import (
    TASK_COPY_API_PATH,
    TASK_CREATE_API_PATH,
    TASK_DETAIL_API_PATH,
    TASK_LIST_API_PATH,
    TASK_RENAME_API_PATH,
    TASK_RERUN_API_PATH,
    TASK_STATUS_RESET_API_PATH,
)


class TaskHandlers:
    """Mixin —— 任务列表/详情/CRUD/重跑相关方法。"""

    def _build_display_audio_path(self, task_id: str, task_record: dict[str, Any]) -> str:
        """为任务列表/详情优先展示当前机器可访问的音频路径。"""
        try:
            return str(self._resolve_task_audio_path_from_record(task_id=task_id, task_record=task_record, persist=False))
        except Exception:  # noqa: BLE001
            return str(task_record.get("audio_path", ""))

    def _build_task_list_payload(self) -> dict[str, Any]:
        """构建主页任务列表所需的任务概览与模块状态摘要。"""
        task_rows = self.state_store.list_tasks()
        task_ids = [str(item.get("task_id", "")).strip() for item in task_rows if str(item.get("task_id", "")).strip()]
        module_status_map = self.state_store.list_task_module_status_map(task_ids=task_ids)
        normalized_tasks: list[dict[str, Any]] = []
        for item in task_rows:
            task_id = str(item.get("task_id", "")).strip()
            normalized_tasks.append(
                {
                    "task_id": task_id,
                    "status": str(item.get("status", "unknown")),
                    "audio_path": self._build_display_audio_path(task_id=task_id, task_record=item),
                    "config_path": str(item.get("config_path", "")),
                    "output_video_path": str(item.get("output_video_path", "")),
                    "updated_at": str(item.get("updated_at", "")),
                    "module_status": module_status_map.get(task_id, {}),
                }
            )
        return {"ok": True, "current_task_id": self.task_id, "tasks": normalized_tasks}

    def _build_task_detail_payload(self, task_id: str) -> dict[str, Any]:
        """构建单任务详情面板所需的数据对象。"""
        normalized_task_id = str(task_id).strip()
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{normalized_task_id}", "task": None}
        module_status_map = self.state_store.list_task_module_status_map([normalized_task_id]).get(normalized_task_id, {})
        # 尝试加载合并配置；失败时返回空
        try:
            merged_config = self._merge_task_config(task_id=normalized_task_id, task_record=task_record)
        except Exception:  # noqa: BLE001
            merged_config = {}
        return {
            "ok": True,
            "task": {
                "task_id": normalized_task_id,
                "status": str(task_record.get("status", "unknown")),
                "audio_path": self._build_display_audio_path(task_id=normalized_task_id, task_record=task_record),
                "config_path": str(task_record.get("config_path", "")),
                "output_video_path": str(task_record.get("output_video_path", "")),
                "updated_at": str(task_record.get("updated_at", "")),
                "created_at": str(task_record.get("created_at", "")),
                "error_message": str(task_record.get("error_message", "")),
                "module_status": module_status_map,
                "config": merged_config,
                "save_lyrics_port": int(getattr(self, "_save_lyrics_post_port", 0)),
            },
        }

    def _handle_create_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理主页新建任务请求，仅写入状态记录，不触发实际运行。"""
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [""])[0]).strip()
        audio_path = str(query.get("audio_path", [""])[0]).strip()
        config_path = str(query.get("config_path", [""])[0]).strip()
        if not task_id or not audio_path or not config_path:
            return {"ok": False, "error": "新建任务失败：task_id、audio_path、config_path 不能为空。"}, HTTPStatus.BAD_REQUEST
        if self.state_store.task_exists(task_id=task_id):
            return {"ok": False, "error": f"新建任务失败：task_id 已存在，task_id={task_id}"}, HTTPStatus.CONFLICT
        self.state_store.init_task(task_id=task_id, audio_path=audio_path, config_path=config_path)
        return {
            "ok": True,
            "task_id": task_id,
            "task": self._build_task_detail_payload(task_id=task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_rename_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理主页任务改名请求，并同步重命名任务目录。"""
        query = parse_qs(parsed.query)
        old_task_id = str(query.get("old_task_id", [""])[0]).strip()
        new_task_id = str(query.get("new_task_id", [""])[0]).strip()
        if not old_task_id or not new_task_id:
            return {"ok": False, "error": "任务改名失败：old_task_id 与 new_task_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        try:
            self._rename_task_with_artifacts(old_task_id=old_task_id, new_task_id=new_task_id)
        except ValueError as error:
            return {"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST
        except RuntimeError as error:
            return {"ok": False, "error": str(error)}, HTTPStatus.CONFLICT
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"任务改名失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
        return {
            "ok": True,
            "task_id": new_task_id,
            "task": self._build_task_detail_payload(task_id=new_task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_copy_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理基于现有任务复制为新任务的请求，仅创建新记录，不自动运行。"""
        query = parse_qs(parsed.query)
        source_task_id = str(query.get("source_task_id", [""])[0]).strip()
        new_task_id = str(query.get("new_task_id", [""])[0]).strip()
        if not source_task_id or not new_task_id:
            return {"ok": False, "error": "复制任务失败：source_task_id 与 new_task_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        source_task = self.state_store.get_task(task_id=source_task_id)
        if source_task is None:
            return {"ok": False, "error": f"复制任务失败：源任务不存在，task_id={source_task_id}"}, HTTPStatus.NOT_FOUND
        if self.state_store.task_exists(task_id=new_task_id):
            return {"ok": False, "error": f"复制任务失败：目标 task_id 已存在，task_id={new_task_id}"}, HTTPStatus.CONFLICT
        audio_path = str(query.get("audio_path", [str(source_task.get("audio_path", ""))])[0]).strip()
        config_path = str(query.get("config_path", [str(source_task.get("config_path", ""))])[0]).strip()
        if not audio_path or not config_path:
            return {"ok": False, "error": "复制任务失败：audio_path 与 config_path 不能为空。"}, HTTPStatus.BAD_REQUEST
        self.state_store.init_task(task_id=new_task_id, audio_path=audio_path, config_path=config_path)
        return {
            "ok": True,
            "task_id": new_task_id,
            "task": self._build_task_detail_payload(task_id=new_task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_rerun_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理主页"生成"按钮触发的强制全链路重跑请求。支持 force=true 参数跳过 running 检查。"""
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        force = str(query.get("force", ["false"])[0]).strip().lower() == "true"
        return self._submit_task_rerun_request(
            task_id=task_id,
            success_message=f"任务已开始生成，task_id={task_id}，模式=强制从A模块开始覆盖式重跑",
            log_reason="manual_rerun",
            force=force,
        )

    def _submit_task_rerun_request(
        self,
        *,
        task_id: str,
        success_message: str,
        log_reason: str,
        force: bool = False,
    ) -> tuple[dict[str, Any], HTTPStatus]:
        """提交一次"从模块A开始"的后台重跑请求。

        当 force=True 时自动重置过期 running 状态，跳过检查直接提交。
        """
        if self.rerun_handler is None:
            return {"ok": False, "error": "当前监督服务未配置生成能力。"}, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"生成失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        if not force:
            active_thread = self._rerun_threads.get(task_id)
            if active_thread is not None and active_thread.is_alive():
                return {"ok": False, "error": f"生成失败：任务已在后台启动中，task_id={task_id}"}, HTTPStatus.CONFLICT
            task_status = str(task_record.get("status", "")).strip().lower()
            if task_status == "running":
                return {"ok": False, "error": f"生成失败：任务当前正在运行，task_id={task_id}"}, HTTPStatus.CONFLICT
        else:
            # force 模式：自动重置过期 running 状态
            task_status = str(task_record.get("status", "")).strip().lower()
            if task_status == "running":
                self.state_store.update_task_status(
                    task_id=task_id, status="failed",
                    error_message="自动重置：用户触发强制重跑",
                )
                self.logger.warning(
                    "[监督服务] 强制重跑检测到过期running状态，已自动重置为failed，task_id=%s",
                    task_id,
                )
            # 重置模块 A 状态为 pending（确保从 A 开始重新跑）
            try:
                module_status_map = self.state_store.get_module_status_map(task_id=task_id)
                if module_status_map.get("A") == "running":
                    self.state_store.set_module_status(task_id=task_id, module_name="A", status="pending")
                    self.logger.warning(
                        "[监督服务] 强制重跑检测到模块A running状态，已重置为pending，task_id=%s",
                        task_id,
                    )
            except Exception:  # noqa: BLE001
                pass

        rerun_thread = threading.Thread(
            target=self._run_rerun_task_in_background,
            name=f"task-rerun-{task_id}",
            args=(task_id,),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        rerun_thread.start()
        self.logger.info(
            "[监督服务] 任务强制重跑已提交，task_id=%s，from_module=A，reason=%s",
            task_id,
            log_reason,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": success_message,
        }, HTTPStatus.OK

    def _run_rerun_task_in_background(self, task_id: str) -> None:
        """在后台线程中执行任务强制重跑。"""
        try:
            self.logger.info("[监督服务] 后台开始执行任务强制重跑，task_id=%s，from_module=A", task_id)
            self.rerun_handler(task_id)
            self.logger.info("[监督服务] 后台任务强制重跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            self.logger.error("[监督服务] 后台任务强制重跑失败，task_id=%s，错误信息=%s", task_id, error)
            try:
                self.state_store.update_task_status(task_id=task_id, status="failed", error_message=str(error)[:500])
            except Exception:  # noqa: BLE001
                pass
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
                self._rerun_thread_meta.pop(task_id, None)

    def _handle_resume_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理详情页"全链路续跑"请求：从断点恢复任务（A→B→C→D），不强制重置。"""
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        if self.rerun_handler is None:
            return {"ok": False, "error": "当前监督服务未配置生成能力。"}, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"续跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"续跑失败：任务已在后台启动中，task_id={task_id}"}, HTTPStatus.CONFLICT
        task_status = str(task_record.get("status", "")).strip().lower()
        if task_status == "running":
            return {"ok": False, "error": f"续跑失败：任务当前正在运行，task_id={task_id}"}, HTTPStatus.CONFLICT

        rerun_thread = threading.Thread(
            target=self._run_resume_task_in_background,
            name=f"task-resume-{task_id}",
            args=(task_id,),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        rerun_thread.start()
        self.logger.info(
            "[监督服务] 任务全链路续跑已提交，task_id=%s",
            task_id,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": f"全链路续跑已提交，task_id={task_id}，后台从断点继续执行。",
        }, HTTPStatus.OK

    def _run_resume_task_in_background(self, task_id: str) -> None:
        """在后台线程中执行任务全链路续跑（通过 CLI resume 子进程）。"""
        import sys
        from pathlib import Path
        try:
            self.logger.info("[监督服务] 后台开始执行全链路续跑，task_id=%s", task_id)
            task_record = self.state_store.get_task(task_id=task_id) or {}
            config_path_text = str(task_record.get("config_path", "")).strip()
            workspace_root = self._resolve_project_root()
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
            completed = _subprocess.run(
                command,
                cwd=str(workspace_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if completed.returncode != 0:
                error_excerpt = (completed.stderr or "").strip() or (completed.stdout or "").strip()
                raise RuntimeError(f"全链路续跑子进程退出码={completed.returncode}，{error_excerpt[:500]}")
            self.logger.info("[监督服务] 后台全链路续跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            self.logger.error("[监督服务] 后台全链路续跑失败，task_id=%s，错误信息=%s", task_id, error)
            try:
                self.state_store.update_task_status(task_id=task_id, status="failed", error_message=str(error)[:500])
            except Exception:  # noqa: BLE001
                pass
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
                self._rerun_thread_meta.pop(task_id, None)
    def _submit_task_rerun_lyrics_only_request(
        self,
        *,
        task_id: str,
        success_message: str,
        log_reason: str,
    ) -> tuple[dict[str, Any], HTTPStatus]:
        """提交一次"仅更新歌词→算法层"的轻量重跑请求（跳过信号处理）。"""
        if self.lyrics_only_rerun_handler is None:
            return {
                "ok": False,
                "error": "当前监督服务未配置轻量重跑能力。",
            }, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"轻量重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"轻量重跑失败：任务已在后台启动中，task_id={task_id}"}, HTTPStatus.CONFLICT
        task_status = str(task_record.get("status", "")).strip().lower()
        if task_status == "running":
            # 状态为 running 但无活跃后台线程 → 上次运行异常退出，视为过期状态自动重置
            self.state_store.update_task_status(task_id=task_id, status="failed",
                                                 error_message="自动重置：前一次运行异常退出（任务卡死在running状态）")
            self.logger.warning(
                "[监督服务] 轻量重跑检测到过期running状态，已自动重置为failed，task_id=%s",
                task_id,
            )

        rerun_thread = threading.Thread(
            target=self._run_rerun_task_lyrics_only_in_background,
            name=f"task-lyrics-only-{task_id}",
            args=(task_id,),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        rerun_thread.start()
        self.logger.info(
            "[监督服务] 任务轻量重跑已提交，task_id=%s，reason=%s",
            task_id,
            log_reason,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": success_message,
        }, HTTPStatus.OK

    def _run_rerun_task_lyrics_only_in_background(self, task_id: str) -> None:
        """在后台线程中执行轻量重跑（仅歌词→算法层，跳过信号处理）。"""
        try:
            self.logger.info("[监督服务] 后台开始执行轻量重跑，task_id=%s", task_id)
            self.lyrics_only_rerun_handler(task_id)
            self.logger.info("[监督服务] 后台轻量重跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            self.logger.error("[监督服务] 后台轻量重跑失败，task_id=%s，错误信息=%s", task_id, error)
            try:
                self.state_store.update_task_status(task_id=task_id, status="failed", error_message=str(error)[:500])
            except Exception:  # noqa: BLE001
                pass
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
                self._rerun_thread_meta.pop(task_id, None)

    def _rename_task_with_artifacts(self, old_task_id: str, new_task_id: str) -> None:
        """协调状态库改名与 runs 目录改名，确保任务上下文一致。"""
        normalized_old_task_id = str(old_task_id).strip()
        normalized_new_task_id = str(new_task_id).strip()
        old_task_dir = self._resolve_task_dir(task_id=normalized_old_task_id)
        new_task_dir = self._resolve_task_dir(task_id=normalized_new_task_id)
        if old_task_dir.exists() and new_task_dir.exists():
            raise RuntimeError(f"任务改名失败：目标任务目录已存在，path={new_task_dir}")

        self.state_store.rename_task(old_task_id=normalized_old_task_id, new_task_id=normalized_new_task_id)
        try:
            if old_task_dir.exists():
                old_task_dir.rename(new_task_dir)
        except Exception as error:  # noqa: BLE001
            try:
                self.state_store.rename_task(old_task_id=normalized_new_task_id, new_task_id=normalized_old_task_id)
            except Exception as rollback_error:  # noqa: BLE001
                raise RuntimeError(
                    f"任务改名失败：目录改名出错且数据库回滚失败，dir_error={error}，rollback_error={rollback_error}"
                ) from rollback_error
            raise RuntimeError(f"任务改名失败：目录改名出错，已回滚数据库，error={error}") from error

    def _handle_task_status_reset_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """处理任务状态手动重置请求，支持 pending / running / done / failed。"""
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        target_status = str(query.get("status", ["pending"])[0]).strip()
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"重置失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        try:
            self.state_store.update_task_status(task_id=task_id, status=target_status)
            self.logger.info(
                "[任务] 手动重置任务状态（调试），task_id=%s，status=%s",
                task_id,
                target_status,
            )
            return {"ok": True, "task_id": task_id, "status": target_status}, HTTPStatus.OK
        except Exception as error:
            return {"ok": False, "error": f"重置任务状态失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
