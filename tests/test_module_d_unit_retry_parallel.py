"""
文件用途：验证模块D最小单元并行执行、失败重试与断点恢复行为。
核心流程：构造模块C帧输入，打桩ffmpeg执行器，检查单元状态与恢复行为。
输入输出：输入临时任务目录，输出模块D执行结果断言。
依赖说明：依赖 pytest 与项目内模块D编排实现。
维护说明：当模块D单元级调度策略变更时需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于并发验证
import threading
# 标准库：用于制造可观测并发窗口
import time
# 标准库：用于轻量上下文桩
from types import SimpleNamespace
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import (
    AppConfig,
    BypyUploadConfig,
    FfmpegConfig,
    LoggingConfig,
    MockConfig,
    ModeConfig,
    ModuleDConfig,
    PathsConfig,
)
# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON读写工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块D编排入口
from music_video_pipeline.modules.module_d import orchestrator as module_d_orchestrator
# 项目内模块：模块D执行器
from music_video_pipeline.modules.module_d import executor as module_d_executor
# 项目内模块：模块D单元模型
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit
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


def test_run_module_d_should_render_with_animatediff_backend_without_clip_upload(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 animatediff 后端可完成单元渲染，且不会触发 clip 级上传消费。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩避免真实模型推理与真实 bypy 上传。
    """
    context = _build_context(
        tmp_path=tmp_path,
        task_id="task_d_animatediff_success",
        segment_workers=2,
        unit_retry_times=1,
        render_backend="animatediff",
        upload_enabled=True,
    )
    _write_module_c_output(context=context)
    _write_module_b_output(context=context)

    prompts: list[str] = []
    upload_call_count = {"count": 0}

    def _fake_denoise_stage(*, context, unit, prompt, device_override=None):  # noqa: ANN001
        _ = (context, device_override)
        prompts.append(str(prompt))
        return {
            "shot_id": str(unit.unit_id),
            "shot_index": int(unit.unit_index),
            "frames": [b"fake-frame"],
            "target_effective_fps": 8,
            "target_effective_frames": int(unit.exact_frames),
            "inference_frames": int(unit.exact_frames),
            "exact_frames": int(unit.exact_frames),
        }

    def _fake_post_stage(*, context, unit, denoise_summary, encoder_command_args, profile_name="animatediff"):  # noqa: ANN001
        _ = (context, denoise_summary, encoder_command_args, profile_name)
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(b"fake-animatediff-segment")
        return {"segment_path": str(unit.segment_path)}

    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_denoise_stage", _fake_denoise_stage)
    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_post_stage", _fake_post_stage)
    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)

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
    assert len(prompts) == 3
    assert all(str(prompt).strip() for prompt in prompts)
    assert upload_call_count["count"] == 0


def test_run_module_d_should_fail_without_ffmpeg_fallback_when_animatediff_failed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 animatediff 渲染失败时严格失败且不回退 ffmpeg。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩模拟 AnimateDiff 全失败，确保 ffmpeg 不会被调用。
    """
    context = _build_context(
        tmp_path=tmp_path,
        task_id="task_d_animatediff_fallback",
        segment_workers=1,
        unit_retry_times=1,
        render_backend="animatediff",
        upload_enabled=False,
    )
    _write_module_c_output(context=context)
    _write_module_b_output(context=context)

    fallback_ffmpeg = _ScriptedFfmpegRunner()
    monkeypatch.setattr(
        module_d_executor,
        "run_one_unit_animatediff_denoise_stage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mock animatediff failed")),
    )
    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", fallback_ffmpeg.run)
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

    with pytest.raises(RuntimeError, match="模块D单元渲染失败"):
        module_d_orchestrator.run_module_d(context)

    failed_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["failed"],
    )
    assert [item["unit_id"] for item in failed_units] == ["shot_001", "shot_002", "shot_003"]
    assert fallback_ffmpeg.calls == []


def test_run_module_d_should_start_next_denoise_before_previous_post_done(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证纯模块D入口下，下一单元去噪可在上一单元后处理结束前启动。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过两阶段打桩构造“去噪短、后处理长”的可观测窗口。
    """
    context = _build_context(
        tmp_path=tmp_path,
        task_id="task_d_stage_overlap",
        segment_workers=2,
        unit_retry_times=0,
        render_backend="animatediff",
        upload_enabled=False,
    )
    _write_module_c_output(context=context)
    _write_module_b_output(context=context)

    monkeypatch.setattr(module_d_orchestrator, "_probe_media_duration", lambda media_path, ffprobe_bin: 3.0)

    def _fake_concat(**kwargs) -> dict:
        output_video_path = kwargs["output_video_path"]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-video")
        return {"mode": "copy", "copy_fallback_triggered": False}

    monkeypatch.setattr(module_d_orchestrator, "_concat_segment_videos", _fake_concat)

    timeline_lock = threading.Lock()
    stage_timeline: dict[str, dict[str, float]] = {}

    def _mark_time(unit_id: str, key: str, value: float) -> None:
        with timeline_lock:
            stage_timeline.setdefault(unit_id, {})
            stage_timeline[unit_id][key] = float(value)

    def _fake_denoise_stage(*, context, unit, prompt, device_override=None):  # noqa: ANN001
        _ = (context, prompt, device_override)
        _mark_time(unit.unit_id, "denoise_start", time.perf_counter())
        time.sleep(0.04)
        _mark_time(unit.unit_id, "denoise_end", time.perf_counter())
        return {
            "shot_id": str(unit.unit_id),
            "shot_index": int(unit.unit_index),
            "frames": [b"fake-frame"],
            "target_effective_fps": 8,
            "target_effective_frames": int(unit.exact_frames),
            "inference_frames": int(unit.exact_frames),
            "exact_frames": int(unit.exact_frames),
        }

    def _fake_post_stage(*, context, unit, denoise_summary, encoder_command_args, profile_name="animatediff"):  # noqa: ANN001
        _ = (context, denoise_summary, encoder_command_args, profile_name)
        _mark_time(unit.unit_id, "post_start", time.perf_counter())
        time.sleep(0.12)
        unit.segment_path.parent.mkdir(parents=True, exist_ok=True)
        unit.segment_path.write_bytes(b"ok")
        _mark_time(unit.unit_id, "post_end", time.perf_counter())
        return {"segment_path": str(unit.segment_path)}

    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_denoise_stage", _fake_denoise_stage)
    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_post_stage", _fake_post_stage)

    output_path = module_d_orchestrator.run_module_d(context)

    assert output_path.exists()
    ordered_unit_ids = sorted(stage_timeline.keys(), key=lambda item: float(stage_timeline[item]["denoise_start"]))
    assert len(ordered_unit_ids) >= 2
    first_unit_id = ordered_unit_ids[0]
    second_unit_id = ordered_unit_ids[1]
    assert float(stage_timeline[second_unit_id]["denoise_start"]) < float(stage_timeline[first_unit_id]["post_end"])


def test_execute_one_unit_with_retry_should_only_serialize_animatediff_denoise_stage(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证跨线程并发触发 D 单元时，AnimateDiff 仅去噪阶段串行，后处理可并发。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证执行器阶段互斥行为，不依赖真实模型与状态库。
    """
    context = SimpleNamespace(
        task_id="task_d_lock",
        config=SimpleNamespace(
            module_d=SimpleNamespace(
                render_backend="animatediff",
                unit_retry_times=0,
            ),
            ffmpeg=SimpleNamespace(
                video_codec="libx264",
                video_preset="veryfast",
                video_crf=30,
            ),
        ),
        state_store=SimpleNamespace(set_module_unit_status=lambda **kwargs: None),
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
    )
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    units = [
        ModuleDUnit(
            unit_id="shot_001",
            unit_index=0,
            shot={"video_prompt_en": "p1", "video_prompt": "p1"},
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            exact_frames=24,
            segment_path=segments_dir / "segment_001.mp4",
            temp_segment_path=segments_dir / "segment_001.tmp.mp4",
        ),
        ModuleDUnit(
            unit_id="shot_002",
            unit_index=1,
            shot={"video_prompt_en": "p2", "video_prompt": "p2"},
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            exact_frames=24,
            segment_path=segments_dir / "segment_002.mp4",
            temp_segment_path=segments_dir / "segment_002.tmp.mp4",
        ),
    ]

    state_lock = threading.Lock()
    denoise_in_flight = 0
    max_denoise_in_flight = 0
    post_in_flight = 0
    max_post_in_flight = 0
    stage_timeline: dict[str, dict[str, float]] = {}

    def _mark_time(unit_id: str, key: str, value: float) -> None:
        with state_lock:
            stage_timeline.setdefault(unit_id, {})
            stage_timeline[unit_id][key] = float(value)

    def _fake_denoise_stage(*, context, unit, prompt, device_override=None):  # noqa: ANN001
        nonlocal denoise_in_flight, max_denoise_in_flight
        _ = (context, prompt, device_override)
        _mark_time(unit.unit_id, "denoise_start", time.perf_counter())
        with state_lock:
            denoise_in_flight += 1
            if denoise_in_flight > max_denoise_in_flight:
                max_denoise_in_flight = denoise_in_flight
        time.sleep(0.05)
        with state_lock:
            denoise_in_flight -= 1
        _mark_time(unit.unit_id, "denoise_end", time.perf_counter())
        return {
            "shot_id": str(unit.unit_id),
            "shot_index": int(unit.unit_index),
            "frames": [b"fake-frame"],
            "target_effective_fps": 8,
            "target_effective_frames": int(unit.exact_frames),
            "inference_frames": int(unit.exact_frames),
            "exact_frames": int(unit.exact_frames),
        }

    def _fake_post_stage(*, context, unit, denoise_summary, encoder_command_args, profile_name="animatediff"):  # noqa: ANN001
        nonlocal post_in_flight, max_post_in_flight
        _ = (context, denoise_summary, encoder_command_args, profile_name)
        _mark_time(unit.unit_id, "post_start", time.perf_counter())
        with state_lock:
            post_in_flight += 1
            if post_in_flight > max_post_in_flight:
                max_post_in_flight = post_in_flight
        time.sleep(0.12)
        unit.segment_path.write_bytes(b"ok")
        with state_lock:
            post_in_flight -= 1
        _mark_time(unit.unit_id, "post_end", time.perf_counter())
        return {"segment_path": str(unit.segment_path)}

    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_denoise_stage", _fake_denoise_stage)
    monkeypatch.setattr(module_d_executor, "run_one_unit_animatediff_post_stage", _fake_post_stage)

    errors: list[Exception] = []

    def _run(unit: ModuleDUnit) -> None:
        try:
            module_d_executor.execute_one_unit_with_retry(context=context, unit=unit)
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread_1 = threading.Thread(target=_run, args=(units[0],))
    thread_2 = threading.Thread(target=_run, args=(units[1],))
    thread_1.start()
    thread_2.start()
    thread_1.join(timeout=2.0)
    thread_2.join(timeout=2.0)

    assert not errors
    assert max_denoise_in_flight == 1
    assert max_post_in_flight >= 1
    first_unit_id, second_unit_id = sorted(
        [unit.unit_id for unit in units],
        key=lambda item: float(stage_timeline[item]["denoise_start"]),
    )
    assert float(stage_timeline[second_unit_id]["denoise_start"]) < float(stage_timeline[first_unit_id]["post_end"])
    assert units[0].segment_path.exists()
    assert units[1].segment_path.exists()


def _build_context(
    tmp_path: Path,
    task_id: str,
    segment_workers: int,
    unit_retry_times: int,
    render_backend: str = "ffmpeg",
    upload_enabled: bool = False,
) -> RuntimeContext:
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

    module_d_config = ModuleDConfig(
        segment_workers=segment_workers,
        unit_retry_times=unit_retry_times,
        render_backend=render_backend,
        animatediff=ModuleDConfig.AnimateDiffConfig(
            device="cpu",
            fallback_to_ffmpeg=False,
        ),
    )

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
        module_d=module_d_config,
        bypy_upload=BypyUploadConfig(enabled=upload_enabled),
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


def _write_module_b_output(context: RuntimeContext) -> None:
    """
    功能说明：写入模块D animatediff 测试所需的模块B分镜提示词文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：shot_id 顺序与 module_c_output 保持一致。
    """
    shots: list[dict] = []
    for index in range(3):
        shot_id = f"shot_{index + 1:03d}"
        shots.append(
            {
                "shot_id": shot_id,
                "start_time": float(index),
                "end_time": float(index + 1),
                "scene_desc": f"scene {shot_id}",
                "keyframe_prompt_en": f"keyframe {shot_id}",
                "video_prompt_en": f"video prompt {shot_id}",
                "keyframe_prompt": f"keyframe {shot_id}",
                "video_prompt": f"video prompt {shot_id}",
                "camera_motion": "static",
                "transition": "cut",
                "constraints": {"safe": True},
            }
        )
    write_json(context.artifacts_dir / "module_b_output.json", shots)
