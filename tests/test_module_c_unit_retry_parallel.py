"""
文件用途：验证模块 C 最小视觉单元在 ComfyUI 单一路径下的重试与断点恢复行为。
核心流程：构造模块 B 分镜输入，打桩 ComfyUI 关键帧生成器，检查单元状态与输出顺序。
输入输出：输入临时任务目录，输出模块 C 执行结果断言。
依赖说明：依赖 pytest 与项目内模块 C 编排实现。
维护说明：本文件只覆盖当前 ComfyUI 常驻服务路径，不再保留旧本地后端测试。
"""

# 标准库：用于日志对象构建。
import logging
# 标准库：用于路径处理。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest

# 项目内模块：配置数据类。
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, ModuleAConfig, ModuleCConfig, PathsConfig
# 项目内模块：运行上下文定义。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：关键帧生成器抽象。
from music_video_pipeline.generators import FrameGenerator
# 项目内模块：JSON 读写工具。
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块 C 编排入口。
from music_video_pipeline.modules.module_c import orchestrator as module_c_orchestrator
# 项目内模块：状态存储。
from music_video_pipeline.state_store import StateStore



def _build_prompt_fields(prompt_en: str) -> dict[str, str]:
    """
    功能说明：构造模块 B 提示词字段。
    参数说明：
    - prompt_en: 起始态英文提示词。
    返回值：
    - dict[str, str]: 可合并到 shot 的提示词字段。
    异常说明：无。
    边界条件：视频提示词固定使用单轨字段。
    """
    normalized_prompt = str(prompt_en).strip()
    return {
        "keyframe_prompt_start_zh": f"关键帧起始：{normalized_prompt}",
        "keyframe_prompt_start_en": normalized_prompt,
        "keyframe_negative_prompt_start_zh": "负面起始：彩色污染，噪点，水印，文字，额外人物",
        "keyframe_negative_prompt_start_en": "color contamination, noise, watermark, text, extra person",
        "keyframe_prompt_end_zh": f"关键帧结束：{normalized_prompt}",
        "keyframe_prompt_end_en": normalized_prompt,
        "keyframe_negative_prompt_end_zh": "负面结束：彩色污染，噪点，水印，文字，额外人物",
        "keyframe_negative_prompt_end_en": "color contamination, noise, watermark, text, extra person",
        "video_prompt_zh": f"视频提示词：{normalized_prompt}",
        "video_prompt_en": normalized_prompt,
    }


class _ScriptedFrameGenerator(FrameGenerator):
    """
    功能说明：测试用关键帧生成器，可按 shot_id 预设失败次数。
    参数说明：
    - fail_plan: 单元失败计划，键为 shot_id，值为剩余失败次数。
    返回值：不适用。
    异常说明：当命中失败计划时抛 RuntimeError。
    边界条件：未配置失败计划的单元始终成功。
    """

    def __init__(self, fail_plan: dict[str, int] | None = None) -> None:
        self.fail_plan = dict(fail_plan or {})
        self.calls: list[str] = []

    def generate_one(
        self,
        shot: dict,
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict:
        """
        功能说明：执行单元渲染，按预设决定抛错或生成占位文件。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 帧输出目录。
        - width: 图像宽度（测试中不使用）。
        - height: 图像高度（测试中不使用）。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict: 兼容模块 D 消费的 frame_item。
        异常说明：命中失败计划时抛 RuntimeError。
        边界条件：输出文件名固定使用 frame_{index:03d}.png。
        """
        _ = (width, height)
        shot_id = str(shot["shot_id"])
        self.calls.append(shot_id)

        remaining_failures = int(self.fail_plan.get(shot_id, 0))
        if remaining_failures > 0:
            self.fail_plan[shot_id] = remaining_failures - 1
            raise RuntimeError(f"simulated failure for {shot_id}")

        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path_start = output_dir / f"frame_{shot_index + 1:03d}.png"
        frame_path_end = output_dir / f"frame_{shot_index + 1:03d}_end.png"
        frame_path_start.write_bytes(b"fake-frame-start")
        frame_path_end.write_bytes(b"fake-frame-end")
        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)
        return {
            "shot_id": shot_id,
            "frame_path": str(frame_path_start),
            "frame_path_start": str(frame_path_start),
            "frame_path_end": str(frame_path_end),
            "control_frame_paths": [str(frame_path_start), str(frame_path_end)],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "video_prompt_en": str(shot.get("video_prompt_en", "")),
        }


class _PrewarmFailingFrameGenerator(_ScriptedFrameGenerator):
    """
    功能说明：测试用关键帧生成器，在 prewarm 阶段直接失败。
    参数说明：无。
    返回值：不适用。
    异常说明：prewarm 固定抛 RuntimeError。
    边界条件：generate_one 不应被调用。
    """

    def prewarm(self) -> None:
        raise RuntimeError("comfyui prewarm failed")



def test_run_module_c_should_retry_failed_unit_and_keep_output_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单元失败后会按配置重试，最终输出顺序仍稳定。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例开启并行执行（render_workers=3）。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_c_retry_order", render_workers=3, unit_retry_times=1)
    _write_module_b_output(context=context)
    scripted_generator = _ScriptedFrameGenerator(fail_plan={"shot_002": 1})
    monkeypatch.setattr(
        module_c_orchestrator,
        "build_keyframe_generator",
        lambda mode, logger, app_config=None: scripted_generator,
    )

    output_path = module_c_orchestrator.run_module_c(context)
    output_data = read_json(output_path)
    frame_items = output_data["frame_items"]

    assert [item["shot_id"] for item in frame_items] == ["shot_001", "shot_002", "shot_003"]
    assert scripted_generator.calls.count("shot_001") == 1
    assert scripted_generator.calls.count("shot_002") == 2
    assert scripted_generator.calls.count("shot_003") == 1

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_002", "shot_003"]



def test_run_module_c_should_resume_only_failed_units_after_strict_failure(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证严格失败后再次执行仅补跑 failed 单元，done 单元不重跑。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：第一次执行设定 shot_002 持续失败，第二次改为成功。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_c_resume_failed_only", render_workers=2, unit_retry_times=1)
    _write_module_b_output(context=context)

    fail_generator = _ScriptedFrameGenerator(fail_plan={"shot_002": 100})
    monkeypatch.setattr(
        module_c_orchestrator,
        "build_keyframe_generator",
        lambda mode, logger, app_config=None: fail_generator,
    )
    with pytest.raises(RuntimeError):
        module_c_orchestrator.run_module_c(context)

    failed_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["failed"],
    )
    assert [item["unit_id"] for item in failed_units] == ["shot_002"]
    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["shot_001", "shot_003"]

    resume_generator = _ScriptedFrameGenerator()
    monkeypatch.setattr(
        module_c_orchestrator,
        "build_keyframe_generator",
        lambda mode, logger, app_config=None: resume_generator,
    )
    module_c_orchestrator.run_module_c(context)

    assert resume_generator.calls == ["shot_002"]
    done_units_after_resume = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units_after_resume] == ["shot_001", "shot_002", "shot_003"]


def test_run_module_c_should_fail_fast_when_prewarm_failed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证模块 C 会在批量生成前先执行 prewarm，失败时立即退出。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：prewarm 失败时不应有任何单元进入 done。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_c_prewarm_fail", render_workers=2, unit_retry_times=1)
    _write_module_b_output(context=context)
    generator = _PrewarmFailingFrameGenerator()
    monkeypatch.setattr(
        module_c_orchestrator,
        "build_keyframe_generator",
        lambda mode, logger, app_config=None: generator,
    )

    with pytest.raises(RuntimeError, match="prewarm failed"):
        module_c_orchestrator.run_module_c(context)

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["done"],
    )
    assert done_units == []



def _build_context(
    tmp_path: Path,
    task_id: str,
    render_workers: int,
    unit_retry_times: int,
) -> RuntimeContext:
    """
    功能说明：构建模块 C 测试用运行上下文。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - task_id: 任务唯一标识。
    - render_workers: 模块 C 并行 worker 数。
    - unit_retry_times: 模块 C 单元重试次数。
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
        module_c=ModuleCConfig(render_workers=render_workers, unit_retry_times=unit_retry_times),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(tmp_path / "config.json"))
    logger = logging.getLogger(f"test_module_c_{task_id}")
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



def _write_module_b_output(context: RuntimeContext) -> None:
    """
    功能说明：写入模块 C 测试所需的模块 B 分镜输入文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：分镜顺序固定为 shot_001 -> shot_002 -> shot_003。
    """
    module_b_output = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.0,
            "scene_desc": "scene-1",
            **_build_prompt_fields(prompt_en="prompt-1"),
            "camera_plan": {
                "preset_id": "none",
                "mode": "none",
                "direction": "center",
                "strength": "none",
                "easing": "linear",
            },
            "transition_plan": {
                "preset_id": "crossfade_160",
                "kind": "crossfade",
                "duration_ms": 160,
                "easing": "ease_in_out",
            },
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
        },
        {
            "shot_id": "shot_002",
            "start_time": 1.0,
            "end_time": 2.0,
            "scene_desc": "scene-2",
            **_build_prompt_fields(prompt_en="prompt-2"),
            "camera_plan": {
                "preset_id": "zoom_in_s",
                "mode": "zoom",
                "direction": "center",
                "strength": "small",
                "easing": "ease_in_out",
            },
            "transition_plan": {
                "preset_id": "crossfade_160",
                "kind": "crossfade",
                "duration_ms": 160,
                "easing": "ease_in_out",
            },
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
        },
        {
            "shot_id": "shot_003",
            "start_time": 2.0,
            "end_time": 3.0,
            "scene_desc": "scene-3",
            **_build_prompt_fields(prompt_en="prompt-3"),
            "camera_plan": {
                "preset_id": "none",
                "mode": "none",
                "direction": "center",
                "strength": "none",
                "easing": "linear",
            },
            "transition_plan": {
                "preset_id": "hard_cut_0",
                "kind": "hard_cut",
                "duration_ms": 0,
                "easing": "linear",
            },
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
        },
    ]
    write_json(context.artifacts_dir / "module_b_output.json", module_b_output)
