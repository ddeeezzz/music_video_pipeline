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


MonitorHandler = Callable[[str, Any, Any], dict]


@dataclass(slots=True)
class CommandRequest:
    """结构化命令请求对象。"""

    command: str
    config_path: Path
    task_id: str | None = None
    audio_path: Path | None = None
    module: str | None = None
    force_module: str | None = None
    force: bool = False
    shot_id: str | None = None
    segment_id: str | None = None
    user_custom_prompt_override: str | None = None


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

        if command == "monitor":
            if self.monitor_handler is None:
                raise RuntimeError("monitor 命令未配置 monitor_handler。")
            task_id = self._require_text(request.task_id, field_name="task_id")
            dispatch_logger = self.logger
            if dispatch_logger is None:
                raise RuntimeError("monitor 命令缺少日志对象。")
            return self.monitor_handler(task_id, self.runner, dispatch_logger)

        raise RuntimeError(f"未知命令: {command}")

    def _resolve_audio_path(self, audio_path: Path | None) -> Path:
        if audio_path is None:
            return self._resolve_path(Path(self.config.paths.default_audio_path))
        return self._resolve_path(audio_path)

    def _resolve_path(self, input_path: Path) -> Path:
        if input_path.is_absolute():
            return input_path.resolve()
        return (self.workspace_root / input_path).resolve()

    @staticmethod
    def _require_text(value: str | None, *, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise RuntimeError(f"命令参数缺失：{field_name}")
        return text
