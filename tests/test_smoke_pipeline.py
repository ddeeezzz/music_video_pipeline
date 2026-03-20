"""
文件用途：执行端到端冒烟测试，验证 MVP 全链路可生成视频。
核心流程：使用真实模块 A/B/C/D/E 运行一次 20 秒样片任务。
输入输出：输入资源音频，输出视频文件存在性断言。
依赖说明：依赖 pytest、ffmpeg 可执行环境与项目内 PipelineRunner。
维护说明：该测试默认标记为 smoke，可按需选择执行。
"""

# 标准库：用于日志初始化
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于检查外部命令是否存在
import shutil

# 第三方库：用于测试标记与跳过
import pytest

# 项目内模块：配置数据类
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, PathsConfig
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


@pytest.mark.smoke
def test_smoke_run_should_generate_playable_mp4(tmp_path: Path) -> None:
    """
    功能说明：验证 run 命令主流程可产出 MP4 文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：若 ffmpeg 或示例音频缺失则跳过。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("本机未安装 ffmpeg，跳过 smoke 测试。")

    project_root = Path(__file__).resolve().parents[1]
    audio_path = project_root / "resources" / "juebieshu20s.mp3"
    if not audio_path.exists():
        pytest.skip(f"示例音频不存在，跳过 smoke 测试: {audio_path}")

    config = AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path=str(audio_path)),
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
    logger = logging.getLogger("smoke_test")
    logger.setLevel(logging.INFO)
    runner = PipelineRunner(workspace_root=project_root, config=config, logger=logger)

    summary = runner.run(task_id="smoke_task", audio_path=audio_path, config_path=tmp_path / "config.json")
    assert summary["task_status"] == "done"
    output_video_path = Path(summary["output_video_path"])
    assert output_video_path.exists()
    assert output_video_path.stat().st_size > 0
