"""
文件用途：验证调度层会在任务目录下写入每次执行的日志文件。
核心流程：分别执行 run/resume/run-module，并断言 task_dir/log 下存在对应日志文件。
输入输出：输入临时任务环境，输出日志文件存在性与关键内容断言。
依赖说明：依赖 pytest 与项目内 PipelineRunner。
维护说明：任务日志命名策略调整时需同步更新本测试前缀断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, PathsConfig
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


def test_pipeline_should_write_task_log_for_run_and_resume(tmp_path: Path) -> None:
    """
    功能说明：验证 run/resume 均会在任务目录 log 文件夹写入单次执行日志。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：run 首次失败后 resume 成功，便于同时验证两种命令。
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("pipeline_task_logging_run_resume_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)

    fail_flag = {"fail_c_once": True}
    runner.module_runners = {
        "A": _build_success_runner("A"),
        "B": _build_success_runner("B"),
        "C": _build_fail_once_runner("C", fail_flag=fail_flag),
        "D": _build_success_runner("D"),
    }

    with pytest.raises(RuntimeError):
        runner.run(task_id="task_log_resume", audio_path=audio_path, config_path=tmp_path / "config.json")

    task_dir = Path(config.paths.runs_dir) / "task_log_resume"
    log_dir = task_dir / "log"
    run_logs = sorted(log_dir.glob("run_*.log"))
    assert run_logs, "run 命令未生成任务日志文件"
    assert "模块A准备执行" in run_logs[-1].read_text(encoding="utf-8")

    runner.resume(task_id="task_log_resume", config_path=tmp_path / "config.json")
    resume_logs = sorted(log_dir.glob("resume_*.log"))
    assert resume_logs, "resume 命令未生成任务日志文件"
    assert "模块C准备执行" in resume_logs[-1].read_text(encoding="utf-8")


def test_pipeline_should_write_task_log_for_run_single_module(tmp_path: Path) -> None:
    """
    功能说明：验证 run-module 会在任务目录 log 文件夹写入单次执行日志。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅执行模块 A，避免引入上游状态依赖。
    """
    workspace_root = tmp_path / "workspace_single"
    workspace_root.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("pipeline_task_logging_single_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=workspace_root, config=config, logger=logger)
    runner.module_runners = {
        "A": _build_success_runner("A"),
        "B": _build_success_runner("B"),
        "C": _build_success_runner("C"),
        "D": _build_success_runner("D"),
    }

    runner.run_single_module(
        task_id="task_log_single_module",
        module_name="A",
        audio_path=audio_path,
        config_path=tmp_path / "config.json",
        force=False,
    )

    task_dir = Path(config.paths.runs_dir) / "task_log_single_module"
    log_dir = task_dir / "log"
    single_logs = sorted(log_dir.glob("run_module_a_*.log"))
    assert single_logs, "run-module 未生成任务日志文件"
    assert "模块A准备执行" in single_logs[-1].read_text(encoding="utf-8")


def _build_success_runner(module_name: str):
    """
    功能说明：构造始终成功的假模块执行器。
    参数说明：
    - module_name: 模块名。
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
        artifact_path = context.artifacts_dir / f"{module_name.lower()}_artifact.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(f"{module_name} done", encoding="utf-8")
        return artifact_path

    return _runner


def _build_fail_once_runner(module_name: str, fail_flag: dict[str, bool]):
    """
    功能说明：构造首次失败、后续成功的假模块执行器。
    参数说明：
    - module_name: 模块名。
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
        if fail_flag.get("fail_c_once", False):
            fail_flag["fail_c_once"] = False
            raise RuntimeError(f"模拟模块{module_name}首次失败")
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
