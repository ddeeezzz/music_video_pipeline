"""
文件用途：验证模块A双时间戳与小段落契约的关键行为。
核心流程：在 fallback_only 模式执行模块A，检查大段落/小段落/小时戳的一致性。
输入输出：输入临时音频与运行上下文，输出断言结果。
依赖说明：依赖 pytest 与项目内 run_module_a 实现。
维护说明：当模块A契约变更时需同步更新本测试断言。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：配置对象
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块A实现
from music_video_pipeline.modules.module_a import run_module_a
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def test_module_a_fallback_should_output_big_and_small_timestamps(tmp_path: Path) -> None:
    """
    功能说明：验证模块A在规则链下仍能输出大段落、小段落与小时戳。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输入音频可为占位字节，测试目标是时间轴结构完整性。
    """
    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio-content")

    config = _build_test_config(tmp_path=tmp_path)
    logger = logging.getLogger("module_a_algorithm_test")
    logger.setLevel(logging.INFO)

    context = RuntimeContext(
        task_id="algo_task",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "algo_task",
        artifacts_dir=tmp_path / "runs" / "algo_task" / "artifacts",
        config=config,
        logger=logger,
        state_store=StateStore(db_path=tmp_path / "state.sqlite3"),
    )
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_module_a(context)
    output_data = read_json(output_path)

    big_segments = output_data["big_segments"]
    segments = output_data["segments"]
    beats = output_data["beats"]
    lyric_units = output_data["lyric_units"]

    assert len(big_segments) > 0
    assert len(segments) > 0
    assert len(beats) >= 2
    assert all("big_segment_id" in item for item in segments)
    assert all("segment_id" in item for item in lyric_units) or not lyric_units

    # 小段落必须连续覆盖，不应出现倒序与重叠
    for index in range(1, len(segments)):
        assert float(segments[index]["start_time"]) == float(segments[index - 1]["end_time"])

    # beats 语义为最终小时戳，必须升序
    beat_times = [float(item["time"]) for item in beats]
    assert beat_times == sorted(beat_times)


def _build_test_config(tmp_path: Path) -> AppConfig:
    """
    功能说明：构建模块A算法测试专用配置对象。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - AppConfig: 配置对象。
    异常说明：无。
    边界条件：module_a 使用 fallback_only，确保不依赖外部模型环境。
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
        module_a=ModuleAConfig(mode="fallback_only"),
    )
