"""
文件用途：提供可复用的命令服务层，统一执行 mvpl 命令请求。
核心流程：接收结构化请求 -> 归一化参数 -> 调用 PipelineRunner 对应方法。
输入输出：输入 CommandRequest，输出与 CLI 一致的摘要 dict。
依赖说明：依赖 pathlib/dataclasses 与项目内 PipelineRunner/AppConfig。
维护说明：CLI 参数模式、交互模式与未来 API 均应复用本层，避免分发逻辑分叉。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from music_video_pipeline.config import AppConfig
from music_video_pipeline.task_audio_path import remap_windows_absolute_path


MonitorHandler = Callable[[str | None, Any, Any], dict]


@dataclass(slots=True)
class CommandRequest:
    """结构化命令请求对象。"""

    command: str
    config_path: Path
    runs_dir: Path | None = None
    monitor_host: str | None = None
    monitor_port: int | None = None
    task_id: str | None = None
    audio_path: Path | None = None
    module: str | None = None
    force_module: str | None = None
    force: bool = False
    lyrics_only: bool = False
    role_name: str | None = None
    shot_id: str | None = None
    segment_id: str | None = None
    frame_type: str | None = None
    user_custom_prompt_override: str | None = None
    storyboard_template_file_override: str | None = None
    run_name: str | None = None
    template_name: str | None = None
    symbol_src: str | None = None
    symbol_src_list: list[str] | None = None
    background_kind: str | None = None
    background_color: str | None = None
    background_src: str | None = None
    grid_direction: str | None = None
    scroll_direction: str | None = None


class MvplCommandService:
    """统一执行 mvpl 命令请求的服务层。"""

    def __init__(
        self,
        *,
        runner: Any,
        workspace_root: Path,
        config: AppConfig,
        logger: Any | None = None,
        monitor_handler: MonitorHandler | None = None,
    ) -> None:
        self.runner = runner
        self.workspace_root = workspace_root
        self.config = config
        self.logger = logger
        self.monitor_handler = monitor_handler

    def execute(self, request: CommandRequest) -> dict:
        """执行结构化命令请求并返回摘要。"""
        command = str(request.command).strip()

        if command == "run":
            task_id = self._require_text(request.task_id, field_name="task_id")
            audio_path = self._resolve_audio_path(request.audio_path)
            if request.lyrics_only:
                # 轻量重跑：调用 runner 的 lyrics_only 专用方法
                if not hasattr(self.runner, "_rerun_task_from_module_a_lyrics_only_for_monitor"):
                    raise RuntimeError("当前运行器未支持轻量重跑能力。")
                return self.runner._rerun_task_from_module_a_lyrics_only_for_monitor(task_id=task_id)
            return self.runner.run(
                task_id=task_id,
                audio_path=audio_path,
                config_path=request.config_path,
                force_module=request.force_module,
            )

        if command == "resume":
            task_id = self._require_text(request.task_id, field_name="task_id")
            return self.runner.resume(
                task_id=task_id,
                config_path=request.config_path,
                force_module=request.force_module,
            )

        if command == "run-module":
            task_id = self._require_text(request.task_id, field_name="task_id")
            module_name = self._require_text(request.module, field_name="module")
            audio_path = self._resolve_path(request.audio_path) if request.audio_path is not None else None
            return self.runner.run_single_module(
                task_id=task_id,
                module_name=module_name,
                audio_path=audio_path,
                force=bool(request.force),
                config_path=request.config_path,
            )

        if command == "c-task-status":
            task_id = self._require_text(request.task_id, field_name="task_id")
            return self.runner.get_module_c_status_summary(task_id=task_id, config_path=request.config_path)

        if command == "c-retry-shot":
            task_id = self._require_text(request.task_id, field_name="task_id")
            shot_id = self._require_text(request.shot_id, field_name="shot_id")
            return self.runner.retry_module_c_shot(task_id=task_id, shot_id=shot_id, config_path=request.config_path)

        if command == "c-retry-frame":
            task_id = self._require_text(request.task_id, field_name="task_id")
            shot_id = self._require_text(request.shot_id, field_name="shot_id")
            frame_type = self._require_text(request.frame_type, field_name="frame_type")
            return self.runner.retry_module_c_frame(
                task_id=task_id,
                shot_id=shot_id,
                frame_type=frame_type,
                config_path=request.config_path,
            )

        if command == "b-task-status":
            task_id = self._require_text(request.task_id, field_name="task_id")
            return self.runner.get_module_b_status_summary(task_id=task_id, config_path=request.config_path)

        if command == "b-retry-segment":
            task_id = self._require_text(request.task_id, field_name="task_id")
            segment_id = self._require_text(request.segment_id, field_name="segment_id")
            return self.runner.retry_module_b_segment(
                task_id=task_id,
                segment_id=segment_id,
                config_path=request.config_path,
            )

        if command == "b-retry-role":
            task_id = self._require_text(request.task_id, field_name="task_id")
            role_name = self._require_text(request.role_name, field_name="role_name")
            return self.runner.retry_module_b_role(
                task_id=task_id,
                role_name=role_name,
                config_path=request.config_path,
            )

        if command == "b-retry-role-shot":
            task_id = self._require_text(request.task_id, field_name="task_id")
            role_name = self._require_text(request.role_name, field_name="role_name")
            shot_id = self._require_text(request.shot_id, field_name="shot_id")
            return self.runner.retry_module_b_role_shot(
                task_id=task_id,
                role_name=role_name,
                shot_id=shot_id,
                config_path=request.config_path,
            )

        if command == "d-task-status":
            task_id = self._require_text(request.task_id, field_name="task_id")
            return self.runner.get_module_d_status_summary(task_id=task_id, config_path=request.config_path)

        if command == "d-retry-shot":
            task_id = self._require_text(request.task_id, field_name="task_id")
            shot_id = self._require_text(request.shot_id, field_name="shot_id")
            return self.runner.retry_module_d_shot(task_id=task_id, shot_id=shot_id, config_path=request.config_path)

        if command == "bcd-task-status":
            task_id = self._require_text(request.task_id, field_name="task_id")
            return self.runner.get_bcd_status_summary(task_id=task_id, config_path=request.config_path)

        if command == "bcd-retry-segment":
            task_id = self._require_text(request.task_id, field_name="task_id")
            segment_id = self._require_text(request.segment_id, field_name="segment_id")
            return self.runner.retry_bcd_segment(task_id=task_id, segment_id=segment_id, config_path=request.config_path)

        if command == "web":
            if self.monitor_handler is None:
                raise RuntimeError("web 命令未配置 monitor_handler。")
            dispatch_logger = self.logger
            if dispatch_logger is None:
                raise RuntimeError("web 命令缺少日志对象。")
            normalized_task_id = str(request.task_id or "").strip() or None
            return self.monitor_handler(normalized_task_id, self.runner, dispatch_logger)

        if command == "template-render":
            from music_video_pipeline.template_render_cli import render_center_template

            template_name = str(request.template_name or "").strip() or "center"
            if template_name == "grid":
                from music_video_pipeline.template_render_cli import render_grid_template

                return render_grid_template(
                    project_root=self.workspace_root,
                    run_name=str(request.run_name or "").strip(),
                    symbol_src_list=request.symbol_src_list or [],
                    background_kind=str(request.background_kind or "").strip() or "solid",
                    background_color=str(request.background_color or "").strip() or "#FFFFFF",
                    background_src=str(request.background_src or "").strip(),
                    direction=str(request.grid_direction or "").strip() or "left_to_right",
                )
            if template_name == "scroll":
                from music_video_pipeline.template_render_cli import render_scroll_template

                return render_scroll_template(
                    project_root=self.workspace_root,
                    run_name=str(request.run_name or "").strip(),
                    symbol_src=str(request.symbol_src or "").strip() or "/fixtures/center-symbol.svg",
                    symbol_src_list=request.symbol_src_list or [],
                    background_kind=str(request.background_kind or "").strip() or "solid",
                    background_color=str(request.background_color or "").strip() or "#FFFFFF",
                    background_src=str(request.background_src or "").strip(),
                    direction=str(request.scroll_direction or "").strip() or "right_to_left",
                )

            return render_center_template(
                project_root=self.workspace_root,
                run_name=str(request.run_name or "").strip() or "center_template_render",
                symbol_src=str(request.symbol_src or "").strip() or "/fixtures/center-symbol.svg",
                background_kind=str(request.background_kind or "").strip() or "solid",
                background_color=str(request.background_color or "").strip() or "#FFFFFF",
                background_src=str(request.background_src or "").strip(),
            )

        raise RuntimeError(f"未知命令: {command}")

    def _resolve_audio_path(self, audio_path: Path | None) -> Path:
        if audio_path is None:
            return self._resolve_path(Path(self.config.paths.default_audio_path))
        return self._resolve_path(audio_path)

    def _resolve_path(self, input_path: Path) -> Path:
        path_text = str(input_path)
        remapped = remap_windows_absolute_path(workspace_root=self.workspace_root, path_text=path_text)
        if remapped is not None:
            return remapped
        if input_path.is_absolute():
            return input_path.resolve()
        return (self.workspace_root / input_path).resolve()

    @staticmethod
    def _require_text(value: str | None, *, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise RuntimeError(f"命令参数缺失：{field_name}")
        return text
