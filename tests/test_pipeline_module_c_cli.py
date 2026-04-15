"""
文件用途：验证模块C排障CLI对应的调度层行为。
核心流程：构造任务状态与模块C单元状态，调用PipelineRunner新接口并断言结果。
输入输出：输入临时任务环境，输出状态摘要与重试行为断言。
依赖说明：依赖 pytest 与项目内 PipelineRunner/StateStore。
维护说明：若模块C定向重试语义调整，需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：JSON写入工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


def test_get_module_c_status_summary_should_return_counts_and_problem_units(tmp_path: Path) -> None:
    """
    功能说明：验证模块C状态摘要接口能返回任务与单元状态聚合信息。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例覆盖 done/failed/running 三种单元状态。
    """
    runner, workspace_root, config = _build_runner(tmp_path=tmp_path, logger_name="pipeline_c_status_test")
    task_id = "task_c_status_001"
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    runner.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
    runner.state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="B", status="done", artifact_path="b.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="C", status="failed", artifact_path="", error_message="mock failed")
    runner.state_store.set_module_status(task_id=task_id, module_name="D", status="pending")
    runner.state_store.sync_module_units(
        task_id=task_id,
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_001", status="done", artifact_path="/tmp/001.png")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_002", status="failed", error_message="boom")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_003", status="running")

    summary = runner.get_module_c_status_summary(task_id=task_id, config_path=config_path)
    assert summary["task_id"] == task_id
    assert summary["module_c_status"] == "failed"
    assert summary["module_status"]["A"] == "done"
    assert summary["module_status"]["B"] == "done"
    assert summary["module_status"]["C"] == "failed"
    assert summary["module_c_unit_summary"]["total_units"] == 3
    assert summary["module_c_unit_summary"]["status_counts"]["done"] == 1
    assert summary["module_c_unit_summary"]["status_counts"]["failed"] == 1
    assert summary["module_c_unit_summary"]["status_counts"]["running"] == 1
    assert summary["module_c_unit_summary"]["failed_unit_ids"] == ["shot_002"]
    assert summary["module_c_unit_summary"]["running_unit_ids"] == ["shot_003"]
    assert summary["module_c_unit_summary"]["problem_unit_ids"] == ["shot_002", "shot_003"]

    log_dir = runner.runs_dir / task_id / "log"
    assert sorted(log_dir.glob("c_task_status_*.log")), "c-task-status 未生成任务日志文件"


def test_retry_module_c_shot_should_only_rerun_target_unit_then_run_module_d(tmp_path: Path) -> None:
    """
    功能说明：验证模块C定向重试仅处理目标shot，并在成功后执行模块D。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：A/B 状态预置为 done，不参与重跑。
    """
    runner, workspace_root, _config = _build_runner(tmp_path=tmp_path, logger_name="pipeline_c_retry_test")
    task_id = "task_c_retry_001"
    target_shot_id = "shot_002"
    audio_path = workspace_root / "demo_retry.mp3"
    audio_path.write_bytes(b"fake-audio")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    _init_done_task_with_c_units(runner=runner, task_id=task_id, audio_path=audio_path, config_path=config_path)

    executed_modules: list[str] = []

    def _fake_run_module_c(context) -> Path:
        """
        功能说明：测试桩模块C，仅把待跑单元标记为done并产出占位清单。
        参数说明：
        - context: 运行上下文对象。
        返回值：
        - Path: module_c_output.json 路径。
        异常说明：无。
        边界条件：只处理 pending/running/failed 单元。
        """
        executed_modules.append("C")
        pending_records = context.state_store.list_module_units_by_status(
            task_id=context.task_id,
            module_name="C",
            statuses=["pending", "running", "failed"],
        )
        for record in pending_records:
            unit_id = str(record["unit_id"])
            frame_path = context.artifacts_dir / "frames" / f"retry_{unit_id}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(b"fake-frame")
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="C",
                unit_id=unit_id,
                status="done",
                artifact_path=str(frame_path),
                error_message="",
            )
        output_path = context.artifacts_dir / "module_c_output.json"
        write_json(
            output_path,
            {
                "task_id": context.task_id,
                "frames_dir": str(context.artifacts_dir / "frames"),
                "frame_items": context.state_store.list_module_c_done_frame_items(task_id=context.task_id),
            },
        )
        return output_path

    def _fake_run_module_d(context) -> Path:
        """
        功能说明：测试桩模块D，产出占位视频文件。
        参数说明：
        - context: 运行上下文对象。
        返回值：
        - Path: final_output.mp4 路径。
        异常说明：无。
        边界条件：不依赖真实ffmpeg环境。
        """
        executed_modules.append("D")
        output_path = context.task_dir / "final_output.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")
        return output_path

    runner.module_runners["C"] = _fake_run_module_c
    runner.module_runners["D"] = _fake_run_module_d

    summary = runner.retry_module_c_shot(task_id=task_id, shot_id=target_shot_id, config_path=config_path)
    assert summary["task_id"] == task_id
    assert summary["retry_shot_id"] == target_shot_id
    assert summary["task_status"] == "done"
    assert executed_modules == ["C", "D"]

    target_record = runner.state_store.get_module_unit_record(task_id=task_id, module_name="C", unit_id=target_shot_id)
    assert target_record is not None
    assert target_record["status"] == "done"
    assert "retry_shot_002.png" in str(target_record["artifact_path"])

    untouched_record = runner.state_store.get_module_unit_record(task_id=task_id, module_name="C", unit_id="shot_001")
    assert untouched_record is not None
    assert untouched_record["status"] == "done"
    assert "original_shot_001.png" in str(untouched_record["artifact_path"])

    log_dir = runner.runs_dir / task_id / "log"
    assert sorted(log_dir.glob("c_retry_shot_*.log")), "c-retry-shot 未生成任务日志文件"


def test_retry_module_c_shot_should_fail_when_task_or_shot_not_found(tmp_path: Path) -> None:
    """
    功能说明：验证不存在任务或shot时会抛出可定位错误。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：错误路径不应修改其他任务状态。
    """
    runner, workspace_root, _config = _build_runner(tmp_path=tmp_path, logger_name="pipeline_c_retry_error_test")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="任务不存在"):
        runner.retry_module_c_shot(task_id="task_not_exist", shot_id="shot_001", config_path=config_path)

    task_id = "task_c_retry_error_001"
    audio_path = workspace_root / "demo_retry_error.mp3"
    audio_path.write_bytes(b"fake-audio")
    _init_done_task_with_c_units(runner=runner, task_id=task_id, audio_path=audio_path, config_path=config_path)

    with pytest.raises(RuntimeError, match="shot_id 不存在"):
        runner.retry_module_c_shot(task_id=task_id, shot_id="shot_404", config_path=config_path)


def test_retry_module_c_shot_should_reject_when_other_non_done_units_exist(tmp_path: Path) -> None:
    """
    功能说明：验证定向重试会拒绝“目标外仍有非done单元”的任务，避免扩大重试范围。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：目标shot本身可为done，阻塞来自其他shot。
    """
    runner, workspace_root, _config = _build_runner(tmp_path=tmp_path, logger_name="pipeline_c_retry_blocking_test")
    task_id = "task_c_retry_blocking_001"
    audio_path = workspace_root / "demo_retry_blocking.mp3"
    audio_path.write_bytes(b"fake-audio")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    _init_done_task_with_c_units(runner=runner, task_id=task_id, audio_path=audio_path, config_path=config_path)
    runner.state_store.set_module_unit_status(
        task_id=task_id,
        module_name="C",
        unit_id="shot_001",
        status="failed",
        artifact_path="",
        error_message="mock failed",
    )

    with pytest.raises(RuntimeError, match="存在其他非done单元"):
        runner.retry_module_c_shot(task_id=task_id, shot_id="shot_002", config_path=config_path)


def _build_runner(tmp_path: Path, logger_name: str) -> tuple[PipelineRunner, Path, AppConfig]:
    """
    功能说明：构造测试用PipelineRunner及其依赖目录。
    参数说明：
    - tmp_path: pytest 临时目录。
    - logger_name: 日志对象名称。
    返回值：
    - tuple[PipelineRunner, Path, AppConfig]: runner、workspace_root、配置对象。
    异常说明：无。
    边界条件：runs_dir 使用相对路径，确保落在 workspace_root 内。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir="runs", default_audio_path="resources/demo.mp3"),
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
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)
    return runner, workspace_root, config


def _init_done_task_with_c_units(runner: PipelineRunner, task_id: str, audio_path: Path, config_path: Path) -> None:
    """
    功能说明：初始化一个A/B/C/D均done且包含模块C单元状态的任务。
    参数说明：
    - runner: 测试用流水线调度器。
    - task_id: 任务标识。
    - audio_path: 音频路径。
    - config_path: 配置路径。
    返回值：无。
    异常说明：无。
    边界条件：模块C初始两个shot均为done，便于验证定向重试覆盖。
    """
    runner.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
    runner.state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="B", status="done", artifact_path="b.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="C", status="done", artifact_path="c.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="D", status="done", artifact_path="final_output.mp4")
    runner.state_store.mark_task_done_if_possible(task_id=task_id, output_video_path="final_output.mp4")

    runner.state_store.sync_module_units(
        task_id=task_id,
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
        ],
    )
    runner.state_store.set_module_unit_status(
        task_id=task_id,
        module_name="C",
        unit_id="shot_001",
        status="done",
        artifact_path=str(runner.runs_dir / task_id / "artifacts" / "frames" / "original_shot_001.png"),
        error_message="",
    )
    runner.state_store.set_module_unit_status(
        task_id=task_id,
        module_name="C",
        unit_id="shot_002",
        status="done",
        artifact_path=str(runner.runs_dir / task_id / "artifacts" / "frames" / "original_shot_002.png"),
        error_message="",
    )
