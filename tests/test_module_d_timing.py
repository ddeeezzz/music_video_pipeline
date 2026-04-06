"""
文件用途：验证模块D时间轴帧分配与受控并行渲染行为。
核心流程：覆盖单段命令构建、并行顺序稳定、失败隔离与原子写入。
输入输出：输入伪造 frame_items，输出命令断言与渲染结果断言。
依赖说明：依赖 pytest 与项目内 module_d 函数。
维护说明：当模块D渲染调度策略调整时需同步更新本测试。
"""

# 标准库：用于日志对象构造
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：模块D内部函数
from music_video_pipeline.modules import module_d


class _ImmediateFuture:
    """
    功能说明：测试用同步 Future，延迟到 result() 执行任务。
    参数说明：
    - fn/args/kwargs: 待执行函数与参数。
    返回值：不适用。
    异常说明：沿用被执行函数抛出的异常。
    边界条件：同一个 Future 多次调用 result() 只执行一次。
    """

    def __init__(self, fn, args, kwargs) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._done = False
        self._value = None

    def result(self):
        """
        功能说明：返回任务结果并在首次调用时执行任务。
        参数说明：无。
        返回值：任务函数返回值。
        异常说明：任务函数异常会原样抛出。
        边界条件：重复调用不重复执行任务。
        """
        if not self._done:
            self._value = self._fn(*self._args, **self._kwargs)
            self._done = True
        return self._value


class _FakeProcessPoolExecutor:
    """
    功能说明：测试用伪进程池执行器，在当前进程内提交任务。
    参数说明：
    - max_workers: 接口兼容参数。
    - mp_context: 接口兼容参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：不模拟真实进程隔离，仅用于调度逻辑断言。
    """

    def __init__(self, max_workers: int, mp_context=None) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = (exc_type, exc, tb)
        return False

    def submit(self, fn, *args, **kwargs):
        """
        功能说明：提交任务并返回测试 Future。
        参数说明：
        - fn: 任务函数。
        - args/kwargs: 任务参数。
        返回值：
        - _ImmediateFuture: 同步 Future 对象。
        异常说明：无。
        边界条件：任务执行延迟到 result()。
        """
        return _ImmediateFuture(fn=fn, args=args, kwargs=kwargs)


def _write_fake_output(command: list[str]) -> None:
    """
    功能说明：根据 ffmpeg 命令末尾输出路径写入假视频文件。
    参数说明：
    - command: ffmpeg 命令参数数组。
    返回值：无。
    异常说明：无。
    边界条件：自动创建输出目录，覆盖已有同名文件。
    """
    output_path = Path(command[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake-video")


def _build_three_frame_items(tmp_path: Path) -> list[dict[str, float | str]]:
    """
    功能说明：生成三段连续时间轴 frame_items 测试数据。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - list[dict]: 可直接用于模块D渲染函数的 frame_items。
    异常说明：无。
    边界条件：每段 duration 为 1 秒，时间戳连续。
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_items: list[dict[str, float | str]] = []
    for index in range(3):
        frame_path = frames_dir / f"frame_{index + 1:03d}.png"
        frame_path.write_bytes(b"fake-image")
        frame_items.append(
            {
                "frame_path": str(frame_path),
                "start_time": float(index),
                "end_time": float(index + 1),
                "duration": 1.0,
            }
        )
    return frame_items


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


def test_render_segment_videos_should_use_single_output_commands_and_warn_legacy_batch_size(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """
    功能说明：验证新渲染路径固定单段命令，旧 batch_size 配置仅作兼容并告警。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    - caplog: pytest 日志捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：render_workers=1 时不依赖进程池。
    """
    frame_items = _build_three_frame_items(tmp_path=tmp_path)
    audio_duration = 3.0
    commands: list[list[str]] = []

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        commands.append(command)
        _write_fake_output(command=command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)
    with caplog.at_level(logging.WARNING):
        segment_paths = module_d._render_segment_videos(
            frame_items=frame_items,
            segments_dir=tmp_path / "segments",
            ffmpeg_bin="ffmpeg",
            fps=24,
            video_codec="libx264",
            video_preset="veryfast",
            video_crf=30,
            render_batch_size=12,
            render_workers=1,
            audio_duration=audio_duration,
            logger=logging.getLogger("module_d_single_output_test"),
        )

    assert len(segment_paths) == 3
    assert len(commands) == 3
    assert [path.name for path in segment_paths] == ["segment_001.mp4", "segment_002.mp4", "segment_003.mp4"]
    for command in commands:
        assert command.count("-loop") == 1
        assert command.count("-map") == 0
        assert "-frames:v" in command
        assert "-t" not in command
    assert "render_batch_size=12" in caplog.text


def test_render_segment_videos_should_use_gpu_speed_profile_when_auto_and_nvenc_available(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 auto 模式且 NVENC 可用时，单段命令采用速度优先 GPU 参数。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：render_workers=1 时命令在当前进程执行，便于断言。
    """
    frame_items = _build_three_frame_items(tmp_path=tmp_path)[:1]
    commands: list[list[str]] = []

    monkeypatch.setattr(
        module_d,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": True, "hevc_nvenc": False},
    )

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        commands.append(command)
        _write_fake_output(command=command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    module_d._render_segment_videos(
        frame_items=frame_items,
        segments_dir=tmp_path / "segments_gpu",
        ffmpeg_bin="ffmpeg",
        fps=24,
        video_codec="libx264",
        video_preset="veryfast",
        video_crf=30,
        render_batch_size=1,
        render_workers=1,
        audio_duration=1.0,
        logger=logging.getLogger("module_d_gpu_auto_test"),
        video_accel_mode="auto",
        gpu_video_codec="h264_nvenc",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p1"
    assert command[command.index("-rc") + 1] == "vbr"
    assert command[command.index("-cq") + 1] == "34"


def test_render_segment_videos_should_use_cpu_codec_when_cpu_only(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 cpu_only 模式下即使 NVENC 可用也走 CPU 编码参数。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：断言命令中包含 libx264 与 crf。
    """
    frame_items = _build_three_frame_items(tmp_path=tmp_path)[:1]
    commands: list[list[str]] = []

    monkeypatch.setattr(
        module_d,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": True, "hevc_nvenc": True},
    )

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        commands.append(command)
        _write_fake_output(command=command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    module_d._render_segment_videos(
        frame_items=frame_items,
        segments_dir=tmp_path / "segments_cpu",
        ffmpeg_bin="ffmpeg",
        fps=24,
        video_codec="libx264",
        video_preset="veryfast",
        video_crf=30,
        render_batch_size=1,
        render_workers=1,
        audio_duration=1.0,
        logger=logging.getLogger("module_d_cpu_only_test"),
        video_accel_mode="cpu_only",
        gpu_video_codec="h264_nvenc",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert "-crf" in command


def test_render_segment_videos_should_keep_order_when_futures_complete_out_of_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证并行结果乱序完成时仍按 segment_index 输出路径顺序。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过伪执行器与反序 as_completed 模拟乱序完成。
    """
    frame_items = _build_three_frame_items(tmp_path=tmp_path)

    monkeypatch.setattr(module_d, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(module_d, "as_completed", lambda futures: list(reversed(list(futures))))

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        _ = command_name
        _write_fake_output(command=command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)
    monkeypatch.setattr(
        module_d,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": False, "hevc_nvenc": False},
    )

    segment_paths = module_d._render_segment_videos(
        frame_items=frame_items,
        segments_dir=tmp_path / "segments_order",
        ffmpeg_bin="ffmpeg",
        fps=24,
        video_codec="libx264",
        video_preset="veryfast",
        video_crf=30,
        render_batch_size=1,
        render_workers=2,
        audio_duration=3.0,
        logger=logging.getLogger("module_d_order_test"),
        video_accel_mode="auto",
    )

    assert [path.name for path in segment_paths] == ["segment_001.mp4", "segment_002.mp4", "segment_003.mp4"]


def test_render_segment_videos_should_retry_failed_gpu_segment_with_cpu_once(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证单段 GPU 失败时仅该段触发一次 CPU 重试，其他段不重跑。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩 worker 函数模拟 segment_002 的 GPU 失败。
    """
    frame_items = _build_three_frame_items(tmp_path=tmp_path)
    call_records: list[tuple[int, str]] = []

    def _fake_worker(
        ffmpeg_bin: str,
        frame_path: str,
        exact_frames: int,
        fps: int,
        encoder_command_args: list[str],
        segment_index: int,
        temp_output_path: str,
        final_output_path: str,
        profile_name: str,
    ) -> dict[str, float | int | str]:
        _ = (ffmpeg_bin, frame_path, exact_frames, fps, encoder_command_args, temp_output_path)
        call_records.append((segment_index, profile_name))
        if profile_name == "gpu" and segment_index == 2:
            raise RuntimeError("mock gpu failure")
        final_path = Path(final_output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"fake-video")
        return {
            "segment_index": segment_index,
            "segment_path": str(final_path),
            "elapsed": 0.01,
            "profile_name": profile_name,
        }

    monkeypatch.setattr(module_d, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(module_d, "_render_single_segment_worker", _fake_worker)
    monkeypatch.setattr(
        module_d,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": True, "hevc_nvenc": False},
    )

    segment_paths = module_d._render_segment_videos(
        frame_items=frame_items,
        segments_dir=tmp_path / "segments_retry",
        ffmpeg_bin="ffmpeg",
        fps=24,
        video_codec="libx264",
        video_preset="veryfast",
        video_crf=30,
        render_batch_size=1,
        render_workers=2,
        audio_duration=3.0,
        logger=logging.getLogger("module_d_retry_test"),
        video_accel_mode="auto",
        gpu_video_codec="h264_nvenc",
    )

    assert len(segment_paths) == 3
    assert call_records.count((2, "gpu")) == 1
    assert call_records.count((2, "cpu")) == 1
    assert (1, "cpu") not in call_records
    assert (3, "cpu") not in call_records


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
        _write_fake_output(command=command)

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = module_d._render_single_segment_worker(
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
        _write_fake_output(command=command)
        raise RuntimeError("mock ffmpeg failed")

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    with pytest.raises(RuntimeError):
        module_d._render_single_segment_worker(
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
    audio_path = tmp_path / "audio.mp3"
    output_video_path = tmp_path / "final_output.mp4"
    commands: list[list[str]] = []
    call_count = {"copy": 0}

    monkeypatch.setattr(
        module_d,
        "_probe_ffmpeg_encoder_capabilities",
        lambda ffmpeg_bin: {"h264_nvenc": True, "hevc_nvenc": False},
    )

    def _fake_run_ffmpeg_command(command: list[str], command_name: str) -> None:
        commands.append(command)
        if "（copy）" in command_name:
            call_count["copy"] += 1
            raise RuntimeError("copy failed for test")

    monkeypatch.setattr(module_d, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = module_d._concat_segment_videos(
        segment_paths=segment_paths,
        concat_file_path=tmp_path / "segments_concat.txt",
        ffmpeg_bin="ffmpeg",
        audio_path=audio_path,
        output_video_path=output_video_path,
        audio_duration=10.0,
        fps=24,
        video_codec="libx264",
        audio_codec="aac",
        video_preset="veryfast",
        video_crf=30,
        video_accel_mode="auto",
        gpu_video_codec="h264_nvenc",
        concat_video_mode="copy",
        concat_copy_fallback_reencode=True,
        logger=logging.getLogger("module_d_concat_fallback_test"),
    )

    assert call_count["copy"] == 1
    assert len(commands) == 2
    assert result["copy_fallback_triggered"] is True
    assert result["mode"] == "copy_with_reencode_fallback"
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert commands[1][commands[1].index("-c:v") + 1] == "h264_nvenc"
