"""
文件用途：验证流水线失败后可从断点恢复。
核心流程：构造可控的模块执行器，在模块 C 首次失败后执行 resume。
输入输出：输入临时任务环境，输出恢复行为断言。
依赖说明：依赖 pytest 与项目内 PipelineRunner。
维护说明：恢复语义调整时需同步更新本测试。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleCConfig, PathsConfig
# 项目内模块：关键帧生成器抽象
from music_video_pipeline.generators import FrameGenerator
# 项目内模块：JSON写入工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：模块C编排入口
from music_video_pipeline.modules.module_c import orchestrator as module_c_orchestrator
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


def test_resume_should_continue_from_first_non_done_module(tmp_path: Path) -> None:
    """
    功能说明：验证模块 C 失败后 resume 会从 C 继续而非重跑 A/B。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：测试使用假模块执行器，不依赖 ffmpeg。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config = _build_test_config(tmp_path=tmp_path)

    logger = logging.getLogger("resume_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)

    executed_modules: list[str] = []
    fail_flag = {"fail_c_once": True}
    runner.module_runners = {
        "A": _build_success_runner(module_name="A", executed_modules=executed_modules),
        "B": _build_success_runner(module_name="B", executed_modules=executed_modules),
        "C": _build_fail_once_runner(module_name="C", executed_modules=executed_modules, fail_flag=fail_flag),
        "D": _build_success_runner(module_name="D", executed_modules=executed_modules),
    }

    with pytest.raises(RuntimeError):
        runner.run(task_id="resume_task", audio_path=audio_path, config_path=tmp_path / "config.json")

    first_round = executed_modules.copy()
    assert first_round == ["A", "B", "C"]

    status_map_after_fail = runner.state_store.get_module_status_map(task_id="resume_task")
    assert status_map_after_fail["A"] == "done"
    assert status_map_after_fail["B"] == "done"
    assert status_map_after_fail["C"] == "failed"
    assert status_map_after_fail["D"] == "pending"

    runner.resume(task_id="resume_task", config_path=tmp_path / "config.json")
    second_round = executed_modules
    assert second_round == ["A", "B", "C", "C", "D"]

    task_record = runner.state_store.get_task(task_id="resume_task")
    assert task_record is not None
    assert task_record["status"] == "done"


def test_resume_should_only_rerun_failed_module_c_units(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证模块C失败后 resume 仅重跑 failed 单元，不重跑已 done 单元。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模块级恢复仍从 C 开始，单元级恢复由模块 C 内部完成。
    """
    workspace_root = tmp_path / "workspace_c_units"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config = _build_test_config(tmp_path=tmp_path)

    logger = logging.getLogger("resume_module_c_units_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)

    fail_generator = _ScriptedFrameGenerator(fail_plan={"shot_002": 100})
    resume_generator = _ScriptedFrameGenerator()
    generator_holder = {"current": fail_generator}
    monkeypatch.setattr(module_c_orchestrator, "build_frame_generator", lambda mode, logger: generator_holder["current"])

    runner.module_runners = {
        "A": _build_module_a_placeholder_runner(),
        "B": _build_module_b_placeholder_runner(),
        "C": module_c_orchestrator.run_module_c,
        "D": _build_success_runner(module_name="D", executed_modules=[]),
    }

    with pytest.raises(RuntimeError):
        runner.run(task_id="resume_task_module_c_units", audio_path=audio_path, config_path=tmp_path / "config.json")

    done_units_after_fail = runner.state_store.list_module_units_by_status(
        task_id="resume_task_module_c_units",
        module_name="C",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units_after_fail] == ["shot_001", "shot_003"]

    generator_holder["current"] = resume_generator
    runner.resume(task_id="resume_task_module_c_units", config_path=tmp_path / "config.json")
    assert resume_generator.calls == ["shot_002"]


def test_resume_should_fail_fast_for_legacy_module_b_output_contract(tmp_path: Path) -> None:
    """
    功能说明：验证旧版 image_prompt 产物在 resume 时会明确报契约不匹配并提示从B重跑。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本测试仅验证迁移期契约硬切行为，不验证模块C渲染细节。
    """
    workspace_root = tmp_path / "workspace_legacy_b_contract"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config = _build_test_config(tmp_path=tmp_path)

    logger = logging.getLogger("resume_legacy_b_contract_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)

    executed_modules: list[str] = []
    module_a_runner = _build_module_a_placeholder_runner()
    module_b_legacy_runner = _build_legacy_module_b_placeholder_runner()
    runner.module_runners = {
        "A": lambda context: _record_runner_call(executed_modules, "A", module_a_runner, context),
        "B": lambda context: _record_runner_call(executed_modules, "B", module_b_legacy_runner, context),
        "C": lambda context: _record_runner_call(executed_modules, "C", module_c_orchestrator.run_module_c, context),
        "D": _build_success_runner(module_name="D", executed_modules=executed_modules),
    }

    expected_error = "keyframe_prompt/video_prompt"
    with pytest.raises(RuntimeError, match=expected_error):
        runner.run(task_id="resume_task_legacy_b", audio_path=audio_path, config_path=tmp_path / "config.json")

    assert executed_modules == ["A", "B", "C"]
    status_map_after_fail = runner.state_store.get_module_status_map(task_id="resume_task_legacy_b")
    assert status_map_after_fail["A"] == "done"
    assert status_map_after_fail["B"] == "done"
    assert status_map_after_fail["C"] == "failed"

    with pytest.raises(RuntimeError, match=expected_error):
        runner.resume(task_id="resume_task_legacy_b", config_path=tmp_path / "config.json")

    assert executed_modules == ["A", "B", "C", "C"]


def _build_success_runner(module_name: str, executed_modules: list[str]):
    """
    功能说明：构造始终成功的假模块执行器。
    参数说明：
    - module_name: 模块名。
    - executed_modules: 执行记录列表。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：无。
    边界条件：会在 artifacts 目录生成占位产物文件。
    """

    def _runner(context) -> Path:
        """
        功能说明：写入占位产物并返回产物路径。
        参数说明：
        - context: 运行上下文对象。
        返回值：
        - Path: 占位文件路径。
        异常说明：文件写入失败时抛 OSError。
        边界条件：无。
        """
        executed_modules.append(module_name)
        artifact_path = context.artifacts_dir / f"{module_name.lower()}_artifact.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(f"{module_name} done", encoding="utf-8")
        return artifact_path

    return _runner


def _record_runner_call(executed_modules: list[str], module_name: str, runner, context) -> Path:
    """
    功能说明：记录模块执行顺序并透传调用真实测试 runner。
    参数说明：
    - executed_modules: 模块执行顺序记录列表。
    - module_name: 当前模块名。
    - runner: 实际执行函数。
    - context: 运行上下文对象。
    返回值：
    - Path: runner 返回的产物路径。
    异常说明：由下游 runner 决定。
    边界条件：仅用于测试观测，不改写 runner 行为。
    """
    executed_modules.append(module_name)
    return runner(context)


def _build_fail_once_runner(module_name: str, executed_modules: list[str], fail_flag: dict[str, bool]):
    """
    功能说明：构造首次失败、后续成功的假模块执行器。
    参数说明：
    - module_name: 模块名。
    - executed_modules: 执行记录列表。
    - fail_flag: 控制首次失败的布尔字典。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：首次执行抛 RuntimeError。
    边界条件：第二次开始返回成功产物路径。
    """

    def _runner(context) -> Path:
        """
        功能说明：首次抛错，后续写产物并返回。
        参数说明：
        - context: 运行上下文对象。
        返回值：
        - Path: 成功时返回占位产物路径。
        异常说明：首次执行抛 RuntimeError。
        边界条件：失败标记通过闭包共享。
        """
        executed_modules.append(module_name)
        if fail_flag["fail_c_once"]:
            fail_flag["fail_c_once"] = False
            raise RuntimeError("模拟模块C首次失败")
        artifact_path = context.artifacts_dir / f"{module_name.lower()}_artifact.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(f"{module_name} recovered", encoding="utf-8")
        return artifact_path

    return _runner


def _build_test_config(tmp_path: Path) -> AppConfig:
    """
    功能说明：构建测试专用配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
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
        module_c=ModuleCConfig(render_workers=2, unit_retry_times=1),
    )


class _ScriptedFrameGenerator(FrameGenerator):
    """
    功能说明：测试用关键帧生成器，可按 shot_id 预设失败次数。
    参数说明：
    - fail_plan: 单元失败计划，键为 shot_id，值为剩余失败次数。
    返回值：不适用。
    异常说明：命中失败计划时抛 RuntimeError。
    边界条件：未命中失败计划时会正常写出占位帧。
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
        功能说明：按预设失败策略执行单元渲染。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 帧输出目录。
        - width: 图像宽度（测试中不使用）。
        - height: 图像高度（测试中不使用）。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict: 单元 frame_item。
        异常说明：命中失败计划时抛 RuntimeError。
        边界条件：输出路径按索引命名，确保可重复覆盖。
        """
        _ = (width, height)
        shot_id = str(shot["shot_id"])
        self.calls.append(shot_id)
        if self.fail_plan.get(shot_id, 0) > 0:
            self.fail_plan[shot_id] -= 1
            raise RuntimeError(f"mock failure: {shot_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / f"frame_{shot_index + 1:03d}.png"
        frame_path.write_bytes(b"fake-frame")
        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        return {
            "shot_id": shot_id,
            "frame_path": str(frame_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": round(max(0.5, end_time - start_time), 3),
        }


def _build_module_a_placeholder_runner():
    """
    功能说明：构造模块A占位执行器，仅用于满足流水线产物路径约束。
    参数说明：无。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：无。
    边界条件：不参与模块C业务逻辑，仅写占位 JSON。
    """

    def _runner(context) -> Path:
        artifact_path = context.artifacts_dir / "module_a_output.json"
        write_json(
            artifact_path,
            {
                "task_id": context.task_id,
                "audio_path": str(context.audio_path),
                "big_segments": [{"segment_id": "big_001", "start_time": 0.0, "end_time": 3.0, "label": "verse"}],
                "segments": [{"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 3.0, "label": "verse"}],
                "beats": [{"time": 0.0, "type": "major", "source": "beat"}, {"time": 3.0, "type": "major", "source": "beat"}],
                "lyric_units": [],
                "energy_features": [{"start_time": 0.0, "end_time": 3.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
            },
        )
        return artifact_path

    return _runner


def _build_module_b_placeholder_runner():
    """
    功能说明：构造模块B占位执行器，输出固定3个分镜供模块C消费。
    参数说明：无。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：无。
    边界条件：分镜顺序固定，便于断言单元恢复行为。
    """

    def _runner(context) -> Path:
        artifact_path = context.artifacts_dir / "module_b_output.json"
        write_json(
            artifact_path,
            [
                {
                    "shot_id": "shot_001",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "scene_desc": "scene-1",
                    "keyframe_prompt": "prompt-1", "video_prompt": "prompt-1",
                    "camera_motion": "slow_pan",
                    "transition": "crossfade",
                },
                {
                    "shot_id": "shot_002",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "scene_desc": "scene-2",
                    "keyframe_prompt": "prompt-2", "video_prompt": "prompt-2",
                    "camera_motion": "zoom_in",
                    "transition": "crossfade",
                },
                {
                    "shot_id": "shot_003",
                    "start_time": 2.0,
                    "end_time": 3.0,
                    "scene_desc": "scene-3",
                    "keyframe_prompt": "prompt-3", "video_prompt": "prompt-3",
                    "camera_motion": "none",
                    "transition": "hard_cut",
                },
            ],
        )
        return artifact_path

    return _runner


def _build_legacy_module_b_placeholder_runner():
    """
    功能说明：构造模块B旧版占位执行器，仅输出 image_prompt 用于迁移回归测试。
    参数说明：无。
    返回值：
    - callable: 可供 PipelineRunner 调用的模块函数。
    异常说明：无。
    边界条件：该产物按新契约应在模块C阶段被拒绝。
    """

    def _runner(context) -> Path:
        artifact_path = context.artifacts_dir / "module_b_output.json"
        write_json(
            artifact_path,
            [
                {
                    "shot_id": "shot_001",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "scene_desc": "legacy-scene-1",
                    "image_prompt": "legacy-image-prompt-1",
                    "camera_motion": "slow_pan",
                    "transition": "crossfade",
                }
            ],
        )
        return artifact_path

    return _runner
