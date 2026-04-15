"""
文件用途：验证模块D最小单元并行执行、失败重试与断点恢复行为。
核心流程：构造模块C帧输入，打桩ffmpeg执行器，检查单元状态与恢复行为。
输入输出：输入临时任务目录，输出模块D执行结果断言。
依赖说明：依赖 pytest 与项目内模块D编排实现。
维护说明：当模块D单元级调度策略变更时需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleDConfig, PathsConfig
# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON读写工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块D编排入口
from music_video_pipeline.modules.module_d import orchestrator as module_d_orchestrator
# 项目内模块：模块D执行器
from music_video_pipeline.modules.module_d import executor as module_d_executor
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


class _ScriptedFfmpegRunner:
    """
    功能说明：测试用ffmpeg执行桩，可按片段名预设失败次数。
    参数说明：
    - fail_plan: 单元失败计划，键为 segment_XXX，值为剩余失败次数。
    返回值：不适用。
    异常说明：当命中失败计划时抛 RuntimeError。
    边界条件：未配置失败计划的片段始终成功。
    """

    def __init__(self, fail_plan: dict[str, int] | None = None) -> None:
        self.fail_plan = dict(fail_plan or {})
        self.calls: list[str] = []

    def run(self, command: list[str], command_name: str) -> None:
        """
        功能说明：模拟执行ffmpeg命令并按计划触发失败。
        参数说明：
        - command: ffmpeg命令数组。
        - command_name: 命令用途说明（测试中不使用）。
        返回值：无。
        异常说明：命中失败计划时抛 RuntimeError。
        边界条件：输出文件路径默认取命令最后一个参数。
        """
        _ = command_name
        output_path = Path(command[-1])
        segment_key = output_path.name.replace(".tmp.mp4", "").replace(".mp4", "")
        self.calls.append(segment_key)

        remaining_failures = int(self.fail_plan.get(segment_key, 0))
        if remaining_failures > 0:
            self.fail_plan[segment_key] = remaining_failures - 1
            raise RuntimeError(f"mock failure for {segment_key}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-segment")


def test_run_module_d_should_retry_failed_unit_and_keep_output_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单元失败后会按配置重试，最终输出顺序仍稳定。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例串行执行（segment_workers=1）以简化打桩。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_retry_order", segment_workers=1, unit_retry_times=1)
    _write_module_c_output(context=context)

    scripted_ffmpeg = _ScriptedFfmpegRunner(fail_plan={"segment_002": 1})
    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", scripted_ffmpeg.run)
    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)
    monkeypatch.setattr(
        module_d_executor,
        "_resolve_video_encoder_profile",
        lambda **kwargs: {
            "use_gpu": False,
            "name": "cpu",
            "codec": "libx264",
            "command_args": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
            "fallback_cpu_profile": None,
        },
    )

    def _fake_concat(**kwargs) -> dict:
        output_video_path = kwargs["output_video_path"]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-video")
        return {"mode": "copy", "copy_fallback_triggered": False}

    monkeypatch.setattr(module_d_orchestrator, "_concat_segment_videos", _fake_concat)

    output_path = module_d_orchestrator.run_module_d(context)
    module_d_output = read_json(context.artifacts_dir / "module_d_output.json")

    assert output_path.exists()
    assert module_d_output["concat_mode"] == "copy"
    assert [item["shot_id"] for item in module_d_output["segment_items"]] == ["shot_001", "shot_002", "shot_003"]
    assert scripted_ffmpeg.calls.count("segment_001") == 1
    assert scripted_ffmpeg.calls.count("segment_002") == 2
    assert scripted_ffmpeg.calls.count("segment_003") == 1

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_002", "shot_003"]


def test_run_module_d_should_resume_only_failed_units_after_strict_failure(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证严格失败后再次执行仅补跑failed单元，done单元不重跑。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：第一次执行设定 segment_002 持续失败，第二次改为成功。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_resume_failed_only", segment_workers=1, unit_retry_times=1)
    _write_module_c_output(context=context)

    fail_ffmpeg = _ScriptedFfmpegRunner(fail_plan={"segment_002": 100})
    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", fail_ffmpeg.run)
    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)
    monkeypatch.setattr(
        module_d_executor,
        "_resolve_video_encoder_profile",
        lambda **kwargs: {
            "use_gpu": False,
            "name": "cpu",
            "codec": "libx264",
            "command_args": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
            "fallback_cpu_profile": None,
        },
    )
    monkeypatch.setattr(
        module_d_orchestrator,
        "_concat_segment_videos",
        lambda **kwargs: {"mode": "copy", "copy_fallback_triggered": False},
    )

    with pytest.raises(RuntimeError):
        module_d_orchestrator.run_module_d(context)

    failed_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["failed"],
    )
    assert [item["unit_id"] for item in failed_units] == ["shot_002"]

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_003"]

    resume_ffmpeg = _ScriptedFfmpegRunner()
    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", resume_ffmpeg.run)

    def _fake_concat_success(**kwargs) -> dict:
        output_video_path = kwargs["output_video_path"]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-video")
        return {"mode": "copy", "copy_fallback_triggered": False}

    monkeypatch.setattr(module_d_orchestrator, "_concat_segment_videos", _fake_concat_success)
    module_d_orchestrator.run_module_d(context)

    assert resume_ffmpeg.calls == ["segment_002"]
    done_units_after_resume = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units_after_resume] == ["shot_001", "shot_002", "shot_003"]


def _build_context(tmp_path: Path, task_id: str, segment_workers: int, unit_retry_times: int) -> RuntimeContext:
    """
    功能说明：构建模块D测试用运行上下文。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - task_id: 任务唯一标识。
    - segment_workers: 模块D并行worker数。
    - unit_retry_times: 模块D单元重试次数。
    返回值：
    - RuntimeContext: 测试用上下文对象。
    异常说明：无。
    边界条件：状态库与产物目录均在临时目录内隔离。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")

    runs_dir = tmp_path / "runs"
    task_dir = runs_dir / task_id
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(runs_dir), default_audio_path=str(audio_path)),
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
        module_d=ModuleDConfig(segment_workers=segment_workers, unit_retry_times=unit_retry_times),
    )
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(tmp_path / "config.json"))
    logger = logging.getLogger(f"test_module_d_{task_id}")
    logger.setLevel(logging.INFO)
    return RuntimeContext(
        task_id=task_id,
        audio_path=audio_path,
        task_dir=task_dir,
        artifacts_dir=artifacts_dir,
        config=config,
        logger=logger,
        state_store=state_store,
    )


def _write_module_c_output(context: RuntimeContext) -> None:
    """
    功能说明：写入模块D测试所需的模块C输入文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：分镜顺序固定为 shot_001 -> shot_002 -> shot_003。
    """
    frames_dir = context.artifacts_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_items: list[dict] = []
    for index in range(3):
        shot_id = f"shot_{index + 1:03d}"
        frame_path = frames_dir / f"frame_{index + 1:03d}.png"
        frame_path.write_bytes(b"fake-frame")
        frame_items.append(
            {
                "shot_id": shot_id,
                "frame_path": str(frame_path),
                "start_time": float(index),
                "end_time": float(index + 1),
                "duration": 1.0,
            }
        )

    write_json(
        context.artifacts_dir / "module_c_output.json",
        {
            "task_id": context.task_id,
            "frames_dir": str(frames_dir),
            "frame_items": frame_items,
        },
    )
