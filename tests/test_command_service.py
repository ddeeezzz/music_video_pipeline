"""
文件用途：验证命令服务层（MvplCommandService）的参数归一化与分发行为。
核心流程：构造 CommandRequest 与 FakeRunner，断言执行路径和参数一致性。
"""

from __future__ import annotations

from pathlib import Path

from music_video_pipeline.command_service import CommandRequest, MvplCommandService
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, task_id: str, audio_path: Path, config_path: Path, force_module: str | None) -> dict:
        self.calls.append(
            (
                "run",
                {
                    "task_id": task_id,
                    "audio_path": str(audio_path),
                    "config_path": str(config_path),
                    "force_module": force_module,
                },
            )
        )
        return {"kind": "run"}

    def resume(self, task_id: str, config_path: Path, force_module: str | None) -> dict:
        self.calls.append(
            (
                "resume",
                {
                    "task_id": task_id,
                    "config_path": str(config_path),
                    "force_module": force_module,
                },
            )
        )
        return {"kind": "resume"}

    def run_single_module(
        self,
        task_id: str,
        module_name: str,
        audio_path: Path | None,
        force: bool,
        config_path: Path,
    ) -> dict:
        self.calls.append(
            (
                "run-module",
                {
                    "task_id": task_id,
                    "module_name": module_name,
                    "audio_path": str(audio_path) if audio_path else "",
                    "force": force,
                    "config_path": str(config_path),
                },
            )
        )
        return {"kind": "run-module"}

    def get_module_b_status_summary(self, task_id: str, config_path: Path) -> dict:
        self.calls.append(("b-task-status", {"task_id": task_id, "config_path": str(config_path)}))
        return {"kind": "b-status"}

    def retry_module_b_segment(self, task_id: str, segment_id: str, config_path: Path) -> dict:
        self.calls.append(
            (
                "b-retry-segment",
                {
                    "task_id": task_id,
                    "segment_id": segment_id,
                    "config_path": str(config_path),
                },
            )
        )
        return {"kind": "b-retry"}


def test_command_service_should_use_default_audio_from_config(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(runs_dir=str(tmp_path / "runs"), default_audio="resources/default_audio.mp3")
    runner = _FakeRunner()

    service = MvplCommandService(
        runner=runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=config,
        logger=None,
    )
    request = CommandRequest(
        command="run",
        task_id="task_001",
        config_path=(workspace_root / "configs" / "default.json").resolve(),
        audio_path=None,
    )

    result = service.execute(request)
    assert result["kind"] == "run"
    assert runner.calls[0][0] == "run"
    assert runner.calls[0][1]["audio_path"].endswith("workspace/resources/default_audio.mp3")


def test_command_service_should_call_monitor_handler(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(runs_dir=str(tmp_path / "runs"), default_audio="resources/default_audio.mp3")
    runner = _FakeRunner()

    called: list[tuple[str, object, object]] = []

    def _monitor_handler(task_id: str, monitor_runner, logger) -> dict:  # noqa: ANN001
        called.append((task_id, monitor_runner, logger))
        return {"kind": "monitor", "task_id": task_id}

    service = MvplCommandService(
        runner=runner,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        config=config,
        logger="logger_obj",
        monitor_handler=_monitor_handler,
    )
    request = CommandRequest(
        command="monitor",
        task_id="task_monitor_001",
        config_path=(workspace_root / "configs" / "default.json").resolve(),
    )

    result = service.execute(request)
    assert result["kind"] == "monitor"
    assert called[0][0] == "task_monitor_001"
    assert called[0][1] is runner
    assert called[0][2] == "logger_obj"


def _build_test_config(runs_dir: str, default_audio: str) -> AppConfig:
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=runs_dir, default_audio_path=default_audio),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        mock=MockConfig(beat_interval_seconds=0.5, video_width=960, video_height=540),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
