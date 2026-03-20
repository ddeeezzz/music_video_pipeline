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
from mvp_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, PathsConfig
# 项目内模块：流水线调度器
from mvp_pipeline.pipeline import PipelineRunner


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
    )
