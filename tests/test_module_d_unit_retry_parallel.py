"""
文件用途：验证模块 D 在纯 ComfyUI/ToonCrafter 路径下的重试、恢复与状态写回行为。
核心流程：构造模块 C 双关键帧输入，打桩模块 D 渲染器与终拼器，断言单元状态与输出顺序。
输入输出：输入临时任务目录，输出模块 D 编排与执行结果断言。
依赖说明：依赖 pytest 与项目内模块 D 编排/执行实现。
维护说明：本文件只覆盖当前真实后端，不保留旧视频后端测试语义。
"""

# 标准库：用于日志对象构建。
import logging
# 标准库：用于路径处理。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest

# 项目内模块：应用配置数据类。
from music_video_pipeline.config import (
    AppConfig,
    BypyUploadConfig,
    FfmpegConfig,
    LoggingConfig,
    ModuleAConfig,
    ModuleDConfig,
    PathsConfig,
)
# 项目内模块：运行上下文定义。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具。
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块 D 编排入口。
from music_video_pipeline.modules.module_d import orchestrator as module_d_orchestrator
# 项目内模块：模块 D 执行器。
from music_video_pipeline.modules.module_d import executor as module_d_executor
# 项目内模块：模块 D 单元模型。
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit
# 项目内模块：状态存储。
from music_video_pipeline.state_store import StateStore



def test_run_module_d_should_retry_failed_unit_and_keep_output_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单元失败后会按配置重试，最终输出顺序仍保持稳定。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例通过打桩 ComfyUI 渲染器避免真实服务依赖。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_retry_order", segment_workers=1, unit_retry_times=1)
    _write_module_c_output(context=context)

    attempt_counts: dict[str, int] = {}

    def _fake_render_one_unit_comfyui(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, object]:
        attempt_counts[unit.unit_id] = attempt_counts.get(unit.unit_id, 0) + 1
        if unit.unit_id == "shot_002" and attempt_counts[unit.unit_id] == 1:
            raise RuntimeError("comfyui failed once")
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(f"segment:{unit.unit_id}".encode("utf-8"))
        return {
            "segment_path": str(unit.segment_path),
            "backend": "comfyui-tooncrafter",
            "frame_count_used": int(unit.exact_frames),
        }

    monkeypatch.setattr(module_d_executor, "render_one_unit_comfyui", _fake_render_one_unit_comfyui)
    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)

    def _fake_concat_segment_videos(**kwargs) -> dict[str, object]:
        output_video_path = kwargs["output_video_path"]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-video")
        return {"mode": "copy", "copy_fallback_triggered": False}

    monkeypatch.setattr(module_d_orchestrator, "_concat_segment_videos", _fake_concat_segment_videos)

    output_path = module_d_orchestrator.run_module_d(context)
    module_d_output = read_json(context.artifacts_dir / "module_d_output.json")

    assert output_path.exists()
    assert module_d_output["concat_mode"] == "copy"
    assert [item["shot_id"] for item in module_d_output["segment_items"]] == ["shot_001", "shot_002", "shot_003"]
    assert attempt_counts["shot_001"] == 1
    assert attempt_counts["shot_002"] == 2
    assert attempt_counts["shot_003"] == 1

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_002", "shot_003"]



def test_run_module_d_should_resume_only_failed_units_after_strict_failure(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证严格失败后再次执行仅补跑 failed 单元，done 单元直接复用。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：第一次执行让 shot_002 持续失败，第二次改为成功。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_resume_failed_only", segment_workers=1, unit_retry_times=1)
    _write_module_c_output(context=context)
    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)
    monkeypatch.setattr(
        module_d_orchestrator,
        "_concat_segment_videos",
        lambda **kwargs: {"mode": "copy", "copy_fallback_triggered": False},
    )

    first_attempts: list[str] = []

    def _always_fail_shot_002(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, object]:
        first_attempts.append(unit.unit_id)
        if unit.unit_id == "shot_002":
            raise RuntimeError("comfyui keeps failing")
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(f"segment:{unit.unit_id}".encode("utf-8"))
        return {"segment_path": str(unit.segment_path)}

    monkeypatch.setattr(module_d_executor, "render_one_unit_comfyui", _always_fail_shot_002)

    with pytest.raises(RuntimeError, match="shot_002"):
        module_d_orchestrator.run_module_d(context)

    failed_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["failed"],
    )
    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in failed_units] == ["shot_002"]
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_003"]

    resumed_attempts: list[str] = []

    def _success_on_resume(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, object]:
        resumed_attempts.append(unit.unit_id)
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(f"resume:{unit.unit_id}".encode("utf-8"))
        return {"segment_path": str(unit.segment_path)}

    def _fake_concat_segment_videos(**kwargs) -> dict[str, object]:
        output_video_path = kwargs["output_video_path"]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-video")
        return {"mode": "copy", "copy_fallback_triggered": False}

    monkeypatch.setattr(module_d_executor, "render_one_unit_comfyui", _success_on_resume)
    monkeypatch.setattr(module_d_orchestrator, "_concat_segment_videos", _fake_concat_segment_videos)

    output_path = module_d_orchestrator.run_module_d(context)

    assert output_path.exists()
    assert resumed_attempts == ["shot_002"]
    done_units_after_resume = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units_after_resume] == ["shot_001", "shot_002", "shot_003"]



def test_execute_one_unit_with_retry_should_retry_comfyui_renderer_once(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 execute_one_unit_with_retry 会在 ComfyUI 单元失败后按次数重试。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只验证执行器层，不触发模块 D 总编排。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_single_retry", segment_workers=1, unit_retry_times=1)
    unit = _build_single_unit(tmp_path=tmp_path)
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="D",
        units=[
            {
                "unit_id": unit.unit_id,
                "unit_index": unit.unit_index,
                "start_time": unit.start_time,
                "end_time": unit.end_time,
                "duration": unit.duration,
            }
        ],
    )

    attempt_counter = {"count": 0}

    def _fail_once_then_succeed(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, object]:
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            raise RuntimeError("first failure")
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(b"ok")
        return {"segment_path": str(unit.segment_path)}

    monkeypatch.setattr(module_d_executor, "render_one_unit_comfyui", _fail_once_then_succeed)

    output_path = module_d_executor.execute_one_unit_with_retry(context=context, unit=unit)

    record = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id=unit.unit_id)
    assert output_path == unit.segment_path
    assert attempt_counter["count"] == 2
    assert record is not None and record["status"] == "done"
    assert str(record["artifact_path"]) == str(unit.segment_path)



def test_execute_one_unit_with_retry_should_raise_after_retry_exhausted(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 execute_one_unit_with_retry 在重试耗尽后会抛错并写入 failed 状态。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：重试次数读取模块 D 当前配置。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_d_single_fail", segment_workers=1, unit_retry_times=1)
    unit = _build_single_unit(tmp_path=tmp_path)
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="D",
        units=[
            {
                "unit_id": unit.unit_id,
                "unit_index": unit.unit_index,
                "start_time": unit.start_time,
                "end_time": unit.end_time,
                "duration": unit.duration,
            }
        ],
    )

    def _always_fail(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, object]:
        raise RuntimeError(f"always fail:{unit.unit_id}")

    monkeypatch.setattr(module_d_executor, "render_one_unit_comfyui", _always_fail)

    with pytest.raises(RuntimeError, match="always fail"):
        module_d_executor.execute_one_unit_with_retry(context=context, unit=unit)

    record = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id=unit.unit_id)
    assert record is not None and record["status"] == "failed"
    assert "always fail" in str(record["error_message"])



def _build_context(
    tmp_path: Path,
    task_id: str,
    segment_workers: int,
    unit_retry_times: int,
) -> RuntimeContext:
    """
    功能说明：构建模块 D 测试用运行上下文。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - task_id: 任务唯一标识。
    - segment_workers: 模块 D 并发 worker 数量。
    - unit_retry_times: 模块 D 单元重试次数。
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
        module_d=ModuleDConfig(
            segment_workers=segment_workers,
            unit_retry_times=unit_retry_times,
            render_backend="comfyui",
        ),
        bypy_upload=BypyUploadConfig(enabled=False),
        module_a=ModuleAConfig(funasr_language="auto"),
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
    功能说明：写入模块 D 测试所需的模块 C 双关键帧输出文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：分镜顺序固定为 shot_001 -> shot_002 -> shot_003。
    """
    frames_dir = context.artifacts_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_items: list[dict[str, object]] = []
    for index in range(3):
        shot_id = f"shot_{index + 1:03d}"
        frame_path_start = frames_dir / f"frame_{index + 1:03d}_start.png"
        frame_path_end = frames_dir / f"frame_{index + 1:03d}_end.png"
        frame_path_start.write_bytes(b"fake-frame-start")
        frame_path_end.write_bytes(b"fake-frame-end")
        frame_items.append(
            {
                "shot_id": shot_id,
                "frame_path": str(frame_path_start),
                "frame_path_start": str(frame_path_start),
                "frame_path_end": str(frame_path_end),
                "control_frame_paths": [str(frame_path_start), str(frame_path_end)],
                "video_prompt_en": f"video prompt {shot_id}",
                "video_prompt_zh": f"视频提示词 {shot_id}",
                "scene_desc": f"scene {shot_id}",
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



def _build_single_unit(tmp_path: Path) -> ModuleDUnit:
    """
    功能说明：构造单元级执行器测试所需的最小模块 D 单元。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：
    - ModuleDUnit: 可直接交给执行器的测试单元。
    异常说明：无。
    边界条件：双关键帧文件会一并落到临时目录中。
    """
    start_frame = tmp_path / "single_start.png"
    end_frame = tmp_path / "single_end.png"
    start_frame.write_bytes(b"single-start")
    end_frame.write_bytes(b"single-end")
    return ModuleDUnit(
        unit_id="shot_001",
        unit_index=0,
        shot={
            "shot_id": "shot_001",
            "frame_path": str(start_frame),
            "frame_path_start": str(start_frame),
            "frame_path_end": str(end_frame),
            "control_frame_paths": [str(start_frame), str(end_frame)],
            "video_prompt_en": "video prompt from start to end",
            "start_time": 0.0,
            "end_time": 1.0,
            "duration": 1.0,
        },
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        exact_frames=24,
        segment_path=tmp_path / "segment_001.mp4",
        temp_segment_path=tmp_path / "segment_001.tmp.mp4",
    )
