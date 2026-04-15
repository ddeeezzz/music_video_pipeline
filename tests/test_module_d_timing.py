"""
文件用途：验证模块D时间轴帧分配与两阶段终拼关键行为。
核心流程：覆盖帧分配、单段原子写入、copy失败回退reencode。
输入输出：输入伪造 frame_items，输出命令断言与执行结果断言。
依赖说明：依赖 pytest 与项目内 module_d 子模块函数。
维护说明：当模块D渲染/终拼策略调整时需同步更新本测试。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：模块D执行器
from music_video_pipeline.modules.module_d import executor as module_d_executor
# 项目内模块：模块D终拼器
from music_video_pipeline.modules.module_d import finalizer as module_d_finalizer
# 项目内模块：模块D单元模型
from music_video_pipeline.modules.module_d import unit_models as module_d_unit_models


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

    allocated_frames = module_d_unit_models._allocate_segment_frames_by_timeline(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=fps,
    )

    assert len(allocated_frames) == len(frame_items)
    assert sum(allocated_frames) == round(audio_duration * fps)
    assert all(item > 0 for item in allocated_frames)


def test_render_single_segment_worker_should_commit_atomically_on_success(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单段渲染成功时先写临时文件再原子替换到最终文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过 mock ffmpeg 写出临时文件。
    """
    frame_path = tmp_path / "frame_success.png"
    frame_path.write_bytes(b"fake-image")
    temp_path = tmp_path / "segment_001.tmp.mp4"
    final_path = tmp_path / "segment_001.mp4"

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = module_d_executor._render_single_segment_worker(
        ffmpeg_bin="ffmpeg",
        frame_path=str(frame_path),
        exact_frames=24,
        fps=24,
        encoder_command_args=["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
        segment_index=1,
        temp_output_path=str(temp_path),
        final_output_path=str(final_path),
        profile_name="cpu",
    )

    assert result["segment_index"] == 1
    assert final_path.exists()
    assert not temp_path.exists()


def test_render_single_segment_worker_should_cleanup_temp_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单段渲染失败时会清理临时文件，且不生成最终成品文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模拟 ffmpeg 失败前已写临时文件场景。
    """
    frame_path = tmp_path / "frame_failure.png"
    frame_path.write_bytes(b"fake-image")
    temp_path = tmp_path / "segment_002.tmp.mp4"
    final_path = tmp_path / "segment_002.mp4"

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")
        raise RuntimeError("mock ffmpeg failed")

    monkeypatch.setattr(module_d_executor, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    try:
        module_d_executor._render_single_segment_worker(
            ffmpeg_bin="ffmpeg",
            frame_path=str(frame_path),
            exact_frames=24,
            fps=24,
            encoder_command_args=["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
            segment_index=2,
            temp_output_path=str(temp_path),
            final_output_path=str(final_path),
            profile_name="cpu",
        )
        assert False, "预期抛出 RuntimeError"
    except RuntimeError:
        pass

    assert not temp_path.exists()
    assert not final_path.exists()


def test_concat_segment_videos_should_fallback_to_reencode_when_copy_failed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 concat copy 失败后会触发一次 reencode 回退。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证策略切换，不执行真实 ffmpeg 命令。
    """
    segment_paths = [tmp_path / "segment_001.mp4", tmp_path / "segment_002.mp4"]
    for segment_path in segment_paths:
        segment_path.write_bytes(b"fake-segment")

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-audio")
    output_video_path = tmp_path / "final_output.mp4"
    commands: list[list[str]] = []
    call_count = {"copy": 0}

    monkeypatch.setattr(
        module_d_finalizer,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": True, "hevc_nvenc": False},
    )

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        commands.append(command)
        if "（copy）" in command_name:
            call_count["copy"] += 1
            raise RuntimeError("mock copy failed")
        output_video_path.write_bytes(b"fake-video")

    monkeypatch.setattr(module_d_finalizer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = module_d_finalizer._concat_segment_videos(
        segment_paths=segment_paths,
        concat_file_path=tmp_path / "segments_concat.txt",
        ffmpeg_bin="ffmpeg",
        audio_path=audio_path,
        output_video_path=output_video_path,
        audio_duration=2.0,
        fps=24,
        video_codec="libx264",
        audio_codec="aac",
        video_preset="veryfast",
        video_crf=30,
        video_accel_mode="auto",
        gpu_video_codec="h264_nvenc",
        concat_video_mode="copy",
        concat_copy_fallback_reencode=True,
    )

    assert result["mode"] == "copy_with_reencode_fallback"
    assert result["copy_fallback_triggered"] is True
    assert call_count["copy"] == 1
    assert len(commands) == 2
    assert output_video_path.exists()
