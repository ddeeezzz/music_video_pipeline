"""
文件用途：验证上传队列下线后，流水线主链路不会再触发自动上传。
核心流程：执行 run/status 等命令，断言行为不依赖 upload-worker 队列能力。
输入输出：输入临时任务环境，输出“无自动上传副作用”的断言结果。
依赖说明：依赖 pytest 与项目内 PipelineRunner / 配置数据类。
维护说明：若未来重新引入自动上传，应新增独立入口并重新设计测试。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：配置数据类
from music_video_pipeline.config import (
    AppConfig,
    BypyUploadConfig,
    FfmpegConfig,
    LoggingConfig,
    MockConfig,
    ModeConfig,
    PathsConfig,
)
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner
# 项目内模块：上传执行器（仅用于打桩验证“不会被调用”）
from music_video_pipeline.upload import runner as upload_runner


def test_pipeline_run_should_not_call_bypy_syncup_anymore(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 run 命令完成后不会触发任何自动 bypy 上传调用。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest monkeypatch 工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使 bypy_upload.enabled=true，也不应触发上传。
    """
    workspace_root = tmp_path / "workspace_run_no_auto_upload"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")

    config = _build_test_config(tmp_path=tmp_path, upload_enabled=True)
    logger = logging.getLogger("pipeline_no_auto_upload")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)
    runner.module_runners = {
        "A": _build_success_runner("A"),
        "B": _build_success_runner("B"),
        "C": _build_success_runner("C"),
        "D": _build_success_runner("D"),
    }

    def _fail_if_called(*args, **kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("不应触发自动上传：run_bypy_syncup 被调用")

    monkeypatch.setattr(upload_runner, "run_bypy_syncup", _fail_if_called)

    summary = runner.run(task_id="task_no_auto_upload", audio_path=audio_path, config_path=tmp_path / "config.json")
    assert summary["task_status"] == "done"


def test_pipeline_runner_should_not_expose_upload_worker_api(tmp_path: Path) -> None:
    """
    功能说明：验证 PipelineRunner 不再暴露 upload-worker 队列执行入口。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：该断言用于防止旧入口回归。
    """
    workspace_root = tmp_path / "workspace_api_check"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path=tmp_path, upload_enabled=False)
    runner = PipelineRunner(
        workspace_root=workspace_root,
        config=config,
        logger=logging.getLogger("pipeline_api_check"),
    )
    assert not hasattr(runner, "run_upload_worker")


def _build_success_runner(module_name: str):
    """
    功能说明：构造始终成功的假模块执行器。
    参数说明：
    - module_name: 模块名。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：无。
    边界条件：会在 artifacts 目录写入占位产物文件。
    """

    def _runner(context) -> Path:
        artifact_path = context.artifacts_dir / f"{module_name.lower()}_artifact.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(f"{module_name} done", encoding="utf-8")
        return artifact_path

    return _runner


def _build_test_config(tmp_path: Path, upload_enabled: bool) -> AppConfig:
    """
    功能说明：构建测试专用配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    - upload_enabled: 是否启用 bypy 上传配置（仅用于验证“不会自动触发”）。
    返回值：
    - AppConfig: 配置对象。
    异常说明：无。
    边界条件：runs_dir 指向临时目录，避免污染仓库。
    """
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path="demo.mp3"),
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
        mock=MockConfig(beat_interval_seconds=0.5, video_width=640, video_height=360),
        bypy_upload=BypyUploadConfig(
            enabled=upload_enabled,
            bypy_bin="bypy",
            remote_runs_dir="/runs",
            retry_times=1,
            timeout_seconds=30.0,
            config_dir=str(tmp_path / "bypy_cfg"),
            require_auth_file=False,
            selection_profile="whitelist_v1",
        ),
    )
