"""
文件用途：验证跨模块B/C/D排障CLI对应的调度层行为。
核心流程：构造任务状态与链路单元状态，调用PipelineRunner新接口并断言结果。
输入输出：输入临时任务环境，输出链路状态摘要与定向重试行为断言。
依赖说明：依赖 pytest 与项目内 PipelineRunner/StateStore。
维护说明：跨模块定向重试语义调整时需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


def test_get_bcd_status_summary_should_return_chain_aggregation(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨模块链路状态摘要接口能返回链路级聚合信息。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例覆盖 done/failed/running 三种链路状态。
    """
    runner, workspace_root = _build_runner(tmp_path=tmp_path, logger_name="pipeline_bcd_status_test")
    task_id = "task_bcd_status_001"
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    runner.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
    runner.state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="B", status="running")
    runner.state_store.set_module_status(task_id=task_id, module_name="C", status="running")
    runner.state_store.set_module_status(task_id=task_id, module_name="D", status="running")

    _seed_chain_units(runner=runner, task_id=task_id)
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0001", status="done", artifact_path="/tmp/b1.json")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_001", status="done", artifact_path="/tmp/c1.png")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id="shot_001", status="done", artifact_path="/tmp/d1.mp4")

    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0002", status="done", artifact_path="/tmp/b2.json")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_002", status="running")

    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="mock failed")
    monkeypatch.setattr(
        "music_video_pipeline.pipeline.collect_adaptive_window_status_snapshot",
        lambda context: {"enabled": True, "c_dynamic_limit": 4, "d_dynamic_limit": 1},
    )

    summary = runner.get_bcd_status_summary(task_id=task_id, config_path=config_path)
    assert summary["task_id"] == task_id
    assert summary["module_b_status"] == "running"
    assert summary["bcd_chain_count"] == 3
    assert summary["bcd_chain_status_counts"]["done"] == 1
    assert summary["bcd_chain_status_counts"]["running"] == 1
    assert summary["bcd_chain_status_counts"]["failed"] == 1
    assert len(summary["bcd_problem_chains"]) == 2
    assert summary["adaptive_window"]["enabled"] is True
    assert summary["adaptive_window"]["c_dynamic_limit"] == 4
    assert summary["adaptive_window"]["d_dynamic_limit"] == 1


def test_retry_bcd_segment_should_only_reset_target_chain_and_rerun(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨模块链路重试仅影响目标segment链路。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过补丁隔离真实跨模块执行器。
    """
    runner, workspace_root = _build_runner(tmp_path=tmp_path, logger_name="pipeline_bcd_retry_test")
    task_id = "task_bcd_retry_001"
    audio_path = workspace_root / "demo_retry.mp3"
    audio_path.write_bytes(b"fake-audio")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    runner.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
    runner.state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="B", status="done", artifact_path="b.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="C", status="done", artifact_path="c.json")
    runner.state_store.set_module_status(task_id=task_id, module_name="D", status="done", artifact_path="final_output.mp4")
    _seed_chain_units(runner=runner, task_id=task_id)
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0001", status="done", artifact_path="/tmp/original_b1.json")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0002", status="done", artifact_path="/tmp/original_b2.json")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="seg_0003", status="done", artifact_path="/tmp/original_b3.json")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_001", status="done", artifact_path="/tmp/original_c1.png")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_002", status="done", artifact_path="/tmp/original_c2.png")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="C", unit_id="shot_003", status="done", artifact_path="/tmp/original_c3.png")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id="shot_001", status="done", artifact_path="/tmp/original_d1.mp4")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id="shot_002", status="done", artifact_path="/tmp/original_d2.mp4")
    runner.state_store.set_module_unit_status(task_id=task_id, module_name="D", unit_id="shot_003", status="done", artifact_path="/tmp/original_d3.mp4")

    invoked: dict[str, str] = {}

    def _fake_run_cross_module_bcd(context, target_segment_id=None):
        invoked["target_segment_id"] = str(target_segment_id)
        unit_index = 1
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id="seg_0002",
            status="done",
            artifact_path="/tmp/retry_b2.json",
        )
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id="shot_002",
            status="done",
            artifact_path="/tmp/retry_c2.png",
        )
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id="shot_002",
            status="done",
            artifact_path="/tmp/retry_d2.mp4",
        )
        for module_name, artifact_path in [("B", "b.json"), ("C", "c.json"), ("D", "final_output.mp4")]:
            context.state_store.set_module_status(
                task_id=context.task_id,
                module_name=module_name,
                status="done",
                artifact_path=artifact_path,
                error_message="",
            )
        output_path = context.task_dir / "final_output.mp4"
        output_path.write_bytes(b"fake-video")
        return {
            "task_id": context.task_id,
            "output_video_path": str(output_path),
            "failed_chain_indexes": [],
            "retry_unit_index": unit_index,
        }

    monkeypatch.setattr("music_video_pipeline.pipeline.run_cross_module_bcd", _fake_run_cross_module_bcd)

    summary = runner.retry_bcd_segment(task_id=task_id, segment_id="seg_0002", config_path=config_path)
    assert invoked["target_segment_id"] == "seg_0002"
    assert summary["retry_segment_id"] == "seg_0002"
    assert summary["retry_unit_index"] == 1
    assert summary["retry_shot_id"] == "shot_002"

    b1 = runner.state_store.get_module_unit_record(task_id=task_id, module_name="B", unit_id="seg_0001")
    b2 = runner.state_store.get_module_unit_record(task_id=task_id, module_name="B", unit_id="seg_0002")
    assert b1 is not None and b1["artifact_path"] == "/tmp/original_b1.json"
    assert b2 is not None and b2["artifact_path"] == "/tmp/retry_b2.json"


def _build_runner(tmp_path: Path, logger_name: str) -> tuple[PipelineRunner, Path]:
    """
    功能说明：构造测试用PipelineRunner及其依赖目录。
    参数说明：
    - tmp_path: pytest 临时目录。
    - logger_name: 日志对象名称。
    返回值：
    - tuple[PipelineRunner, Path]: runner 与 workspace_root。
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
    return runner, workspace_root


def _seed_chain_units(runner: PipelineRunner, task_id: str) -> None:
    """
    功能说明：写入三条链路的 B/C/D 单元基础记录。
    参数说明：
    - runner: 测试用流水线调度器。
    - task_id: 任务标识。
    返回值：无。
    异常说明：无。
    边界条件：unit_index 与 shot_id 映射保持一致。
    """
    runner.state_store.sync_module_units(
        task_id=task_id,
        module_name="B",
        units=[
            {"unit_id": "seg_0001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "seg_0002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "seg_0003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    runner.state_store.sync_module_units(
        task_id=task_id,
        module_name="C",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )
    runner.state_store.sync_module_units(
        task_id=task_id,
        module_name="D",
        units=[
            {"unit_id": "shot_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0},
            {"unit_id": "shot_002", "unit_index": 1, "start_time": 1.0, "end_time": 2.0, "duration": 1.0},
            {"unit_id": "shot_003", "unit_index": 2, "start_time": 2.0, "end_time": 3.0, "duration": 1.0},
        ],
    )


def test_retry_bcd_segment_should_fail_when_task_or_segment_not_found(tmp_path: Path) -> None:
    """
    功能说明：验证跨模块定向重试在非法 task_id/segment_id 下会抛可定位错误。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：错误路径不应污染其他任务状态。
    """
    runner, workspace_root = _build_runner(tmp_path=tmp_path, logger_name="pipeline_bcd_retry_error_test")
    config_path = workspace_root / "configs" / "wuli_v2.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="任务不存在"):
        runner.retry_bcd_segment(task_id="task_not_exist", segment_id="seg_0001", config_path=config_path)

    task_id = "task_bcd_retry_error_001"
    audio_path = workspace_root / "demo_retry_error.mp3"
    audio_path.write_bytes(b"fake-audio")
    runner.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
    runner.state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path="a.json")
    _seed_chain_units(runner=runner, task_id=task_id)

    with pytest.raises(RuntimeError, match="segment_id 不存在"):
        runner.retry_bcd_segment(task_id=task_id, segment_id="seg_9999", config_path=config_path)
