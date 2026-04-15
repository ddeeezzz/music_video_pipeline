"""
文件用途：验证模块B最小视觉单元并行执行、失败重试与断点恢复行为。
核心流程：构造模块A输入，打桩分镜生成器，检查单元状态与输出顺序。
输入输出：输入临时任务目录，输出模块B执行结果断言。
依赖说明：依赖 pytest 与项目内模块B编排实现。
维护说明：当模块B单元级调度策略变更时需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import (
    AppConfig,
    FfmpegConfig,
    LoggingConfig,
    MockConfig,
    ModeConfig,
    ModuleAConfig,
    ModuleBConfig,
    ModuleCConfig,
    PathsConfig,
)
# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON读写工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块B编排入口
from music_video_pipeline.modules.module_b import orchestrator as module_b_orchestrator
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


class _ScriptedScriptGenerator:
    """
    功能说明：测试用分镜生成器，可按 segment_id 预设失败次数。
    参数说明：
    - fail_plan: 单元失败计划，键为 segment_id，值为剩余失败次数。
    返回值：不适用。
    异常说明：命中失败计划时抛 RuntimeError。
    边界条件：未配置失败计划的单元始终成功。
    """

    def __init__(self, fail_plan: dict[str, int] | None = None) -> None:
        self.fail_plan = dict(fail_plan or {})
        self.calls: list[str] = []

    def generate(self, module_a_output: dict) -> list[dict]:
        """
        功能说明：批量接口测试占位实现，当前测试不使用该路径。
        参数说明：
        - module_a_output: 模块A输出字典。
        返回值：
        - list[dict]: 空数组。
        异常说明：无。
        边界条件：模块B单元测试仅调用 generate_one。
        """
        _ = module_a_output
        return []

    def generate_one(
        self,
        module_a_output: dict,
        segment: dict,
        segment_index: int,
    ) -> dict:
        """
        功能说明：执行单元分镜生成，按预设决定抛错或返回分镜。
        参数说明：
        - module_a_output: 模块A输出字典（测试中不使用）。
        - segment: 当前segment对象。
        - segment_index: 分镜顺序索引（0 基）。
        返回值：
        - dict: 兼容模块B契约的shot对象。
        异常说明：命中失败计划时抛 RuntimeError。
        边界条件：shot_id 固定按 segment_index 命名。
        """
        _ = module_a_output
        segment_id = str(segment["segment_id"])
        self.calls.append(segment_id)

        remaining_failures = int(self.fail_plan.get(segment_id, 0))
        if remaining_failures > 0:
            self.fail_plan[segment_id] = remaining_failures - 1
            raise RuntimeError(f"mock failure for {segment_id}")

        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        return {
            "shot_id": f"shot_{segment_index + 1:03d}",
            "start_time": start_time,
            "end_time": end_time,
            "scene_desc": f"scene-{segment_id}",
            "keyframe_prompt": f"prompt-{segment_id}", "video_prompt": f"prompt-{segment_id}",
            "camera_motion": "zoom_in",
            "transition": "crossfade",
            "constraints": {"must_keep_style": True, "must_align_to_beat": True},
            "lyric_text": "",
            "lyric_units": [],
        }


def test_run_module_b_should_retry_failed_unit_and_keep_output_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单元失败后会按配置重试，最终输出顺序仍稳定。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例开启并行执行（script_workers=3）。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_b_retry_order", script_workers=3, unit_retry_times=1)
    _write_module_a_output(context=context)
    scripted_generator = _ScriptedScriptGenerator(fail_plan={"seg_0002": 1})
    monkeypatch.setattr(module_b_orchestrator, "build_script_generator", lambda mode, logger, module_b_config=None: scripted_generator)

    output_path = module_b_orchestrator.run_module_b(context)
    output_data = read_json(output_path)

    assert [item["shot_id"] for item in output_data] == ["shot_001", "shot_002", "shot_003"]
    assert scripted_generator.calls.count("seg_0001") == 1
    assert scripted_generator.calls.count("seg_0002") == 2
    assert scripted_generator.calls.count("seg_0003") == 1

    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["seg_0001", "seg_0002", "seg_0003"]


def test_run_module_b_should_resume_only_failed_units_after_strict_failure(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证严格失败后再次执行仅补跑 failed 单元，done 单元不重跑。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：第一次执行设定 seg_0002 持续失败，第二次改为成功。
    """
    context = _build_context(tmp_path=tmp_path, task_id="task_b_resume_failed_only", script_workers=2, unit_retry_times=1)
    _write_module_a_output(context=context)

    fail_generator = _ScriptedScriptGenerator(fail_plan={"seg_0002": 100})
    monkeypatch.setattr(module_b_orchestrator, "build_script_generator", lambda mode, logger, module_b_config=None: fail_generator)
    with pytest.raises(RuntimeError):
        module_b_orchestrator.run_module_b(context)

    failed_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["failed"],
    )
    assert [item["unit_id"] for item in failed_units] == ["seg_0002"]
    done_units = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units] == ["seg_0001", "seg_0003"]

    resume_generator = _ScriptedScriptGenerator()
    monkeypatch.setattr(module_b_orchestrator, "build_script_generator", lambda mode, logger, module_b_config=None: resume_generator)
    module_b_orchestrator.run_module_b(context)

    assert resume_generator.calls == ["seg_0002"]
    done_units_after_resume = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["done"],
    )
    assert [item["unit_id"] for item in done_units_after_resume] == ["seg_0001", "seg_0002", "seg_0003"]


def _build_context(tmp_path: Path, task_id: str, script_workers: int, unit_retry_times: int) -> RuntimeContext:
    """
    功能说明：构建模块B测试用运行上下文。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - task_id: 任务唯一标识。
    - script_workers: 模块B并行worker数。
    - unit_retry_times: 模块B单元重试次数。
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
        module_b=ModuleBConfig(script_workers=script_workers, unit_retry_times=unit_retry_times),
        module_c=ModuleCConfig(render_workers=2, unit_retry_times=1),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(tmp_path / "config.json"))
    logger = logging.getLogger(f"test_module_b_{task_id}")
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


def _write_module_a_output(context: RuntimeContext) -> None:
    """
    功能说明：写入模块B测试所需的模块A输入文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：segments 时间轴连续且包含3个 segment。
    """
    module_a_output = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "big_segments": [
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 3.0, "label": "verse"},
        ],
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"},
            {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 1.0, "end_time": 2.0, "label": "verse"},
            {"segment_id": "seg_0003", "big_segment_id": "big_001", "start_time": 2.0, "end_time": 3.0, "label": "verse"},
        ],
        "beats": [
            {"time": 0.0, "type": "major", "source": "beat"},
            {"time": 3.0, "type": "major", "source": "beat"},
        ],
        "lyric_units": [],
        "energy_features": [
            {"start_time": 0.0, "end_time": 1.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.4},
            {"start_time": 1.0, "end_time": 2.0, "energy_level": "high", "trend": "up", "rhythm_tension": 0.8},
            {"start_time": 2.0, "end_time": 3.0, "energy_level": "low", "trend": "down", "rhythm_tension": 0.2},
        ],
    }
    write_json(context.artifacts_dir / "module_a_output.json", module_a_output)
