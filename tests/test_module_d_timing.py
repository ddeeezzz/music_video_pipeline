"""
文件用途：验证模块D按全局时间轴分配帧数，避免片段拼接累积漂移。
核心流程：对内部帧分配函数与片段渲染命令进行单元测试。
输入输出：输入伪造 frame_items，输出帧分配与命令参数断言结果。
依赖说明：依赖 pytest 与项目内 module_d 函数。
维护说明：当模块D渲染策略调整时需同步更新本测试断言。
"""

# 标准库：用于日志对象构造
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：模块D内部函数
from music_video_pipeline.modules import module_d


def test_allocate_segment_frames_should_match_audio_target_total() -> None:
    """
    功能说明：验证全局帧分配总和严格等于 round(audio_duration * fps)。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：每段应至少分配 1 帧。
    """
    frame_items = [
        {"frame_path": "a.png", "start_time": 0.0, "end_time": 1.15, "duration": 1.15},
        {"frame_path": "b.png", "start_time": 1.15, "end_time": 2.63, "duration": 1.48},
        {"frame_path": "c.png", "start_time": 2.63, "end_time": 4.96, "duration": 2.33},
    ]
    audio_duration = 4.963
    fps = 24

    allocated_frames = module_d._allocate_segment_frames_by_timeline(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=fps,
    )

    assert len(allocated_frames) == len(frame_items)
    assert sum(allocated_frames) == round(audio_duration * fps)
    assert all(item > 0 for item in allocated_frames)


def test_render_segment_videos_should_use_frames_v_instead_of_t(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证片段渲染命令使用 -frames:v，不再依赖片段级 -t。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过 monkeypatch 跳过真实 ffmpeg 执行。
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for index in range(3):
        frame_path = frames_dir / f"frame_{index + 1:03d}.png"
        frame_path.write_bytes(b"fake-image")
        frame_paths.append(frame_path)

    frame_items = [
        {"frame_path": str(frame_paths[0]), "start_time": 0.0, "end_time": 1.15, "duration": 1.15},
        {"frame_path": str(frame_paths[1]), "start_time": 1.15, "end_time": 2.63, "duration": 1.48},
        {"frame_path": str(frame_paths[2]), "start_time": 2.63, "end_time": 4.96, "duration": 2.33},
    ]
    audio_duration = 4.963
    fps = 24

    expected_frames = module_d._allocate_segment_frames_by_timeline(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=fps,
    )

    commands: list[list[str]] = []

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        """
        功能说明：记录 ffmpeg 命令用于断言，不执行真实渲染。
        参数说明：
        - command: ffmpeg 命令数组。
        - command_name: 命令用途名称。
        返回值：无。
        异常说明：无。
        边界条件：测试阶段不依赖外部 ffmpeg 环境。
        """
        _ = command_name
        commands.append(command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    segment_paths = module_d._render_segment_videos(
        frame_items=frame_items,
        segments_dir=tmp_path / "segments",
        ffmpeg_bin="ffmpeg",
        fps=fps,
        video_codec="libx264",
        video_preset="veryfast",
        video_crf=30,
        audio_duration=audio_duration,
        logger=logging.getLogger("module_d_timing_test"),
    )

    assert len(segment_paths) == len(frame_items)
    assert len(commands) == len(frame_items)
    for index, command in enumerate(commands):
        assert "-frames:v" in command
        assert "-t" not in command
        frames_index = command.index("-frames:v") + 1
        assert int(command[frames_index]) == expected_frames[index]
