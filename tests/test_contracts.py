"""
文件用途：验证模块 A/B 输出契约最低字段。
核心流程：在临时目录执行模块函数并检查输出结构。
输入输出：输入测试上下文，输出契约断言结果。
依赖说明：依赖 pytest 与项目内模块实现。
维护说明：契约字段变更时需同步更新测试断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：目录工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块实现
from music_video_pipeline.modules.module_a import run_module_a
# 项目内模块：模块实现
from music_video_pipeline.modules.module_b import run_module_b
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def test_module_a_and_b_outputs_should_match_contracts(tmp_path: Path) -> None:
    """
    功能说明：执行模块 A/B 并验证输出满足最低契约。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入音频为临时占位文件，仅用于流程验证。
    """
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake-audio-content")

    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("contract_test")
    logger.setLevel(logging.INFO)
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")

    context = RuntimeContext(
        task_id="contract_task",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "contract_task",
        artifacts_dir=tmp_path / "runs" / "contract_task" / "artifacts",
        config=config,
        logger=logger,
        state_store=state_store,
    )
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)

    module_a_path = run_module_a(context)
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)
    assert module_a_output["task_id"] == "contract_task"

    module_b_path = run_module_b(context)
    module_b_output = read_json(module_b_path)
    validate_module_b_output(module_b_output)
    assert len(module_b_output) > 0


def _build_test_config(tmp_path: Path) -> AppConfig:
    """
    功能说明：构建用于测试的最小配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - AppConfig: 测试配置对象。
    异常说明：无。
    边界条件：ffmpeg 配置在本测试中不会被实际调用。
    """
    return AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path="input.mp3"),
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
