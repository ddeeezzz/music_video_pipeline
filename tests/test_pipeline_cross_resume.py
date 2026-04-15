"""
文件用途：验证跨模块并行路径下的 run/resume 恢复语义。
核心流程：构造可控模块A与跨模块调度桩，模拟首次失败后 resume 恢复。
输入输出：输入临时任务环境，输出恢复行为断言。
依赖说明：依赖 pytest 与项目内 PipelineRunner。
维护说明：run/resume 调度入口变更时需同步更新本测试。
"""

# 标准库：用于日志初始化
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


def test_resume_should_continue_cross_module_without_rerun_a(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨模块并行模式下，首次失败后 resume 不会重跑模块 A。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：跨模块执行通过桩函数模拟，不依赖真实模型与ffmpeg。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")

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
    logger = logging.getLogger("pipeline_cross_resume_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)

    call_stats = {"a": 0, "cross": 0}

    def _fake_run_a(context) -> Path:
        call_stats["a"] += 1
        output_path = context.artifacts_dir / "module_a_output.json"
        write_json(
            output_path,
            {
                "task_id": context.task_id,
                "audio_path": str(context.audio_path),
                "segments": [
                    {"segment_id": "seg_0001", "start_time": 0.0, "end_time": 1.0, "label": "verse"},
                ],
                "beats": [{"time": 0.5, "type": "major", "source": "onset"}],
                "lyric_units": [],
                "energy_features": [{"start_time": 0.0, "end_time": 1.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
            },
        )
        return output_path

    runner.module_runners["A"] = _fake_run_a

    def _fake_run_cross_module_bcd(context, target_segment_id=None):
        call_stats["cross"] += 1
        if call_stats["cross"] == 1:
            context.state_store.set_module_status(task_id=context.task_id, module_name="B", status="failed", error_message="mock cross fail")
            context.state_store.set_module_status(task_id=context.task_id, module_name="C", status="failed", error_message="mock cross fail")
            context.state_store.set_module_status(task_id=context.task_id, module_name="D", status="failed", error_message="mock cross fail")
            context.state_store.update_task_status(task_id=context.task_id, status="failed", error_message="mock cross fail")
            raise RuntimeError("mock cross fail")
        context.state_store.set_module_status(task_id=context.task_id, module_name="B", status="done", artifact_path="b.json")
        context.state_store.set_module_status(task_id=context.task_id, module_name="C", status="done", artifact_path="c.json")
        final_output_path = context.task_dir / "final_output.mp4"
        final_output_path.write_bytes(b"fake-video")
        context.state_store.set_module_status(task_id=context.task_id, module_name="D", status="done", artifact_path=str(final_output_path))
        return {
            "task_id": context.task_id,
            "output_video_path": str(final_output_path),
            "failed_chain_indexes": [],
            "target_segment_id": target_segment_id,
        }

    monkeypatch.setattr("music_video_pipeline.pipeline.run_cross_module_bcd", _fake_run_cross_module_bcd)

    with pytest.raises(RuntimeError, match="mock cross fail"):
        runner.run(task_id="task_cross_resume_001", audio_path=audio_path, config_path=tmp_path / "config.json")

    status_after_fail = runner.state_store.get_module_status_map(task_id="task_cross_resume_001")
    assert status_after_fail["A"] == "done"
    assert status_after_fail["B"] == "failed"

    summary = runner.resume(task_id="task_cross_resume_001", config_path=tmp_path / "config.json")
    assert summary["task_status"] == "done"
    assert call_stats["a"] == 1
    assert call_stats["cross"] == 2
