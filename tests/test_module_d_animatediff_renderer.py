"""
文件用途：验证模块D AnimateDiff 渲染器的关键辅助逻辑。
核心流程：覆盖 Motion Adapter 下载缓存策略与设备解析策略。
输入输出：输入临时目录与打桩依赖，输出断言结果。
依赖说明：依赖 pytest 与模块 D AnimateDiff 渲染器实现。
维护说明：若下载器或设备策略变更，应同步更新本测试。
"""

# 标准库：用于模块打桩
import logging
import sys
from types import ModuleType
from types import SimpleNamespace
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于伪造帧
from PIL import Image
# 第三方库：用于异常断言
import pytest

# 项目内模块：AnimateDiff 渲染器
from music_video_pipeline.modules.module_d.backends import animatediff_renderer as renderer


def test_ensure_motion_adapter_dir_should_download_when_cache_missing(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证缓存缺失时会触发 Motion Adapter 下载。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过 fake snapshot_download 避免真实网络请求。
    """
    calls: dict[str, str] = {}
    fake_hf_module = ModuleType("huggingface_hub")

    def _fake_snapshot_download(repo_id: str, revision: str, local_dir: str, local_dir_use_symlinks: bool = False) -> str:
        _ = local_dir_use_symlinks
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        calls["local_dir"] = local_dir
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "model_index.json").write_text("{}", encoding="utf-8")
        return str(target_dir)

    fake_hf_module.snapshot_download = _fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    adapter_dir = renderer._ensure_motion_adapter_dir(
        project_root=tmp_path,
        repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
        revision="main",
        local_dir_text="models/motion_adapter/15/test_adapter",
        hf_endpoint="https://hf-mirror.com",
    )
    assert adapter_dir.exists()
    assert (adapter_dir / "model_index.json").exists()
    assert calls["repo_id"] == "guoyww/animatediff-motion-adapter-v1-5-2"
    assert calls["revision"] == "main"


def test_ensure_motion_adapter_dir_should_skip_download_when_cache_exists(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证本地缓存存在时不会重复触发下载。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：目录存在且非空即视为缓存命中。
    """
    cached_dir = tmp_path / "models/motion_adapter/15/cached_adapter"
    cached_dir.mkdir(parents=True, exist_ok=True)
    (cached_dir / "adapter.bin").write_bytes(b"cached")

    fake_hf_module = ModuleType("huggingface_hub")

    def _fake_snapshot_download(*args, **kwargs):  # noqa: ANN001, ANN002
        raise RuntimeError("snapshot_download should not be called when cache exists")

    fake_hf_module.snapshot_download = _fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    adapter_dir = renderer._ensure_motion_adapter_dir(
        project_root=tmp_path,
        repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
        revision="main",
        local_dir_text="models/motion_adapter/15/cached_adapter",
        hf_endpoint="",
    )
    assert adapter_dir == cached_dir.resolve()


def test_ensure_motion_adapter_dir_should_raise_when_download_failed(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 Motion Adapter 下载失败时会抛出可定位错误。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：下载异常应被包装为 RuntimeError 并包含 repo_id 信息。
    """
    fake_hf_module = ModuleType("huggingface_hub")

    def _fake_snapshot_download(*args, **kwargs):  # noqa: ANN001, ANN002
        raise RuntimeError("mock download failed")

    fake_hf_module.snapshot_download = _fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    with pytest.raises(RuntimeError, match="Motion Adapter 下载失败"):
        renderer._ensure_motion_adapter_dir(
            project_root=tmp_path,
            repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
            revision="main",
            local_dir_text="models/motion_adapter/15/failed_adapter",
            hf_endpoint="https://hf-mirror.com",
        )


def test_resolve_device_should_choose_cuda1_when_two_gpus_available(caplog) -> None:
    """
    功能说明：验证 auto 设备策略在双卡场景下优先选择 cuda:1（C/D 分卡策略）。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅测试设备解析，不依赖真实 GPU。
    """

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

    class _FakeTorch:
        cuda = _FakeCuda()

    assert renderer._resolve_device(device_text="auto", torch_module=_FakeTorch()) == "cuda:1"
    assert renderer._resolve_device(device_text="cuda:0", torch_module=_FakeTorch()) == "cuda:0"
    with caplog.at_level(logging.WARNING):
        assert renderer._resolve_device(device_text="cuda:3", torch_module=_FakeTorch()) == "cuda:0"
    assert "设备索引越界" in caplog.text


def test_render_one_unit_animatediff_should_use_8fps_density_and_ignore_num_frames_cap(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证普通段按 8fps 计算推理帧，且不再受 num_frames 上限约束。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输出编码仍应使用 exact_frames，保持时间轴长度一致。
    """
    captured: dict[str, int] = {}
    temp_segment_path = tmp_path / "segments" / "segment_001.tmp.mp4"
    final_segment_path = tmp_path / "segments" / "segment_001.mp4"

    context = SimpleNamespace(
        task_id="task_ad_density_default",
        config=SimpleNamespace(
            ffmpeg=SimpleNamespace(ffmpeg_bin="ffmpeg", fps=24),
            render=SimpleNamespace(video_width=848, video_height=480),
            module_d=SimpleNamespace(
                animatediff=SimpleNamespace(
                    seed_mode="shot_index",
                    num_frames=4,
                    negative_prompt="",
                    guidance_scale=7.0,
                    steps=24,
                    binding_name="xiantiao_style",
                )
            ),
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    unit = SimpleNamespace(
        unit_id="shot_001",
        unit_index=0,
        duration=4.0,
        shot={"big_segment_label": "verse", "segment_label": "verse"},
        exact_frames=98,
        temp_segment_path=temp_segment_path,
        segment_path=final_segment_path,
    )

    monkeypatch.setattr(
        renderer,
        "_ensure_runtime",
        lambda context, device_override=None: {
            "assets": {
                "base_model_key": "x",
                "lora_file_path": "/tmp/fake_lora.safetensors",
            },
            "motion_adapter_repo_id": "m",
            "motion_adapter_dir": "d",
            "device": str(device_override or "cuda:1"),
        },
    )

    def _fake_generate_mv_clip(*, prompt, num_frames, runtime, width, height, negative_prompt, guidance_scale, steps, seed):  # noqa: ANN001
        _ = (prompt, runtime, width, height, negative_prompt, guidance_scale, steps, seed)
        captured["inference_frames"] = int(num_frames)
        return [Image.new(mode="RGB", size=(64, 64), color=(0, 0, 0)) for _ in range(int(num_frames))]

    def _fake_run_ffmpeg_command(*, command, command_name):  # noqa: ANN001
        _ = command_name
        captured["encoded_frames"] = int(command[command.index("-frames:v") + 1])
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")

    monkeypatch.setattr(renderer, "generate_mv_clip", _fake_generate_mv_clip)
    monkeypatch.setattr(renderer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = renderer.render_one_unit_animatediff(
        context=context,
        unit=unit,
        prompt="line art city",
        encoder_command_args=["-c:v", "libx264"],
    )

    assert captured["inference_frames"] == 32
    assert captured["encoded_frames"] == 98
    assert final_segment_path.exists()
    assert result["segment_path"] == str(final_segment_path)
    assert result["target_effective_fps"] == 8
    assert result["target_effective_frames"] == 32
    assert result["inference_frames"] == 32
    assert result["exact_frames"] == 98
    assert result["density_label_source"] == "big_segment_label"
    assert result["density_label_value"] == "verse"


def test_render_one_unit_animatediff_should_use_16fps_when_big_segment_label_is_solo(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 big_segment_label=solo 时推理帧密度提升到 16fps。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输出编码仍按 exact_frames 固定时间轴。
    """
    captured: dict[str, int] = {}
    temp_segment_path = tmp_path / "segments" / "segment_001.tmp.mp4"
    final_segment_path = tmp_path / "segments" / "segment_001.mp4"

    context = SimpleNamespace(
        task_id="task_ad_density_solo",
        config=SimpleNamespace(
            ffmpeg=SimpleNamespace(ffmpeg_bin="ffmpeg", fps=24),
            render=SimpleNamespace(video_width=848, video_height=480),
            module_d=SimpleNamespace(
                animatediff=SimpleNamespace(
                    seed_mode="shot_index",
                    num_frames=2,
                    negative_prompt="",
                    guidance_scale=7.0,
                    steps=24,
                    binding_name="xiantiao_style",
                )
            ),
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    unit = SimpleNamespace(
        unit_id="shot_001",
        unit_index=0,
        duration=4.0,
        shot={"big_segment_label": "solo", "segment_label": "verse"},
        exact_frames=98,
        temp_segment_path=temp_segment_path,
        segment_path=final_segment_path,
    )

    monkeypatch.setattr(
        renderer,
        "_ensure_runtime",
        lambda context, device_override=None: {
            "assets": {"base_model_key": "x", "lora_file_path": "/tmp/fake_lora.safetensors"},
            "motion_adapter_repo_id": "m",
            "motion_adapter_dir": "d",
            "device": str(device_override or "cuda:1"),
        },
    )

    def _fake_generate_mv_clip(*, prompt, num_frames, runtime, width, height, negative_prompt, guidance_scale, steps, seed):  # noqa: ANN001
        _ = (prompt, runtime, width, height, negative_prompt, guidance_scale, steps, seed)
        captured["inference_frames"] = int(num_frames)
        return [Image.new(mode="RGB", size=(64, 64), color=(0, 0, 0)) for _ in range(int(num_frames))]

    def _fake_run_ffmpeg_command(*, command, command_name):  # noqa: ANN001
        _ = command_name
        captured["encoded_frames"] = int(command[command.index("-frames:v") + 1])
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")

    monkeypatch.setattr(renderer, "generate_mv_clip", _fake_generate_mv_clip)
    monkeypatch.setattr(renderer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    result = renderer.render_one_unit_animatediff(
        context=context,
        unit=unit,
        prompt="line art city",
        encoder_command_args=["-c:v", "libx264"],
    )

    assert captured["inference_frames"] == 64
    assert captured["encoded_frames"] == 98
    assert result["target_effective_fps"] == 16


def test_render_one_unit_animatediff_should_use_16fps_when_big_segment_label_is_chorus(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 big_segment_label=chorus 时推理帧密度提升到 16fps。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：输出编码仍按 exact_frames 固定时间轴。
    """
    captured: dict[str, int] = {}
    temp_segment_path = tmp_path / "segments" / "segment_001.tmp.mp4"
    final_segment_path = tmp_path / "segments" / "segment_001.mp4"

    context = SimpleNamespace(
        task_id="task_ad_density_chorus",
        config=SimpleNamespace(
            ffmpeg=SimpleNamespace(ffmpeg_bin="ffmpeg", fps=24),
            render=SimpleNamespace(video_width=848, video_height=480),
            module_d=SimpleNamespace(
                animatediff=SimpleNamespace(
                    seed_mode="shot_index",
                    num_frames=3,
                    negative_prompt="",
                    guidance_scale=7.0,
                    steps=24,
                    binding_name="xiantiao_style",
                )
            ),
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    unit = SimpleNamespace(
        unit_id="shot_001",
        unit_index=0,
        duration=4.0,
        shot={"big_segment_label": "chorus", "segment_label": "verse"},
        exact_frames=98,
        temp_segment_path=temp_segment_path,
        segment_path=final_segment_path,
    )

    monkeypatch.setattr(
        renderer,
        "_ensure_runtime",
        lambda context, device_override=None: {
            "assets": {"base_model_key": "x", "lora_file_path": "/tmp/fake_lora.safetensors"},
            "motion_adapter_repo_id": "m",
            "motion_adapter_dir": "d",
            "device": str(device_override or "cuda:1"),
        },
    )
    def _fake_generate_mv_clip(*, prompt, num_frames, runtime, width, height, negative_prompt, guidance_scale, steps, seed):  # noqa: ANN001
        _ = (prompt, runtime, width, height, negative_prompt, guidance_scale, steps, seed)
        captured["inference_frames"] = int(num_frames)
        return [Image.new(mode="RGB", size=(64, 64), color=(0, 0, 0)) for _ in range(int(num_frames))]

    monkeypatch.setattr(renderer, "generate_mv_clip", _fake_generate_mv_clip)

    def _fake_run_ffmpeg_command(*, command, command_name):  # noqa: ANN001
        _ = command_name
        captured["encoded_frames"] = int(command[command.index("-frames:v") + 1])
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")

    monkeypatch.setattr(renderer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)
    result = renderer.render_one_unit_animatediff(
        context=context,
        unit=unit,
        prompt="line art city",
        encoder_command_args=["-c:v", "libx264"],
    )

    assert captured["inference_frames"] == 64
    assert result["segment_path"] == str(final_segment_path)
    assert captured["encoded_frames"] == 98


def test_render_one_unit_animatediff_should_prioritize_big_segment_label_over_segment_label(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 big_segment_label 与 segment_label 同时存在时，优先使用 big_segment_label。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当 big_segment_label 未命中高密度时，不应被 segment_label 覆盖。
    """
    captured: dict[str, int] = {}
    temp_segment_path = tmp_path / "segments" / "segment_001.tmp.mp4"
    final_segment_path = tmp_path / "segments" / "segment_001.mp4"
    context = SimpleNamespace(
        task_id="task_ad_density_priority",
        config=SimpleNamespace(
            ffmpeg=SimpleNamespace(ffmpeg_bin="ffmpeg", fps=24),
            render=SimpleNamespace(video_width=848, video_height=480),
            module_d=SimpleNamespace(
                animatediff=SimpleNamespace(
                    seed_mode="shot_index",
                    num_frames=1,
                    negative_prompt="",
                    guidance_scale=7.0,
                    steps=24,
                    binding_name="xiantiao_style",
                )
            ),
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    unit = SimpleNamespace(
        unit_id="shot_001",
        unit_index=0,
        duration=4.0,
        shot={"big_segment_label": "verse", "segment_label": "solo"},
        exact_frames=98,
        temp_segment_path=temp_segment_path,
        segment_path=final_segment_path,
    )

    monkeypatch.setattr(
        renderer,
        "_ensure_runtime",
        lambda context, device_override=None: {
            "assets": {"base_model_key": "x", "lora_file_path": "/tmp/fake_lora.safetensors"},
            "motion_adapter_repo_id": "m",
            "motion_adapter_dir": "d",
            "device": str(device_override or "cuda:1"),
        },
    )

    def _fake_generate_mv_clip(*, prompt, num_frames, runtime, width, height, negative_prompt, guidance_scale, steps, seed):  # noqa: ANN001
        _ = (prompt, runtime, width, height, negative_prompt, guidance_scale, steps, seed)
        captured["inference_frames"] = int(num_frames)
        return [Image.new(mode="RGB", size=(64, 64), color=(0, 0, 0)) for _ in range(int(num_frames))]

    def _fake_run_ffmpeg_command(*, command, command_name):  # noqa: ANN001
        _ = command_name
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")

    monkeypatch.setattr(renderer, "generate_mv_clip", _fake_generate_mv_clip)
    monkeypatch.setattr(renderer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    renderer.render_one_unit_animatediff(
        context=context,
        unit=unit,
        prompt="line art city",
        encoder_command_args=["-c:v", "libx264"],
    )
    assert captured["inference_frames"] == 32


def test_render_one_unit_animatediff_should_forward_device_override_to_runtime(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证渲染入口可透传 device_override 到 runtime 初始化层。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不依赖真实模型，仅验证参数传递。
    """
    captured: dict[str, str] = {}
    temp_segment_path = tmp_path / "segments" / "segment_001.tmp.mp4"
    final_segment_path = tmp_path / "segments" / "segment_001.mp4"

    context = SimpleNamespace(
        task_id="task_ad_device_override",
        config=SimpleNamespace(
            ffmpeg=SimpleNamespace(ffmpeg_bin="ffmpeg", fps=24),
            render=SimpleNamespace(video_width=848, video_height=480),
            module_d=SimpleNamespace(
                animatediff=SimpleNamespace(
                    seed_mode="shot_index",
                    num_frames=16,
                    negative_prompt="",
                    guidance_scale=7.0,
                    steps=24,
                    binding_name="xiantiao_style",
                )
            ),
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    unit = SimpleNamespace(
        unit_id="shot_001",
        unit_index=0,
        duration=0.5,
        shot={"big_segment_label": "verse", "segment_label": "verse"},
        exact_frames=12,
        temp_segment_path=temp_segment_path,
        segment_path=final_segment_path,
    )

    def _fake_ensure_runtime(*, context, device_override=None):  # noqa: ANN001
        _ = context
        captured["device_override"] = str(device_override)
        return {
            "assets": {
                "base_model_key": "x",
                "lora_file_path": "/tmp/fake_lora.safetensors",
            },
            "motion_adapter_repo_id": "m",
            "motion_adapter_dir": "d",
            "device": str(device_override or "cuda:1"),
        }

    monkeypatch.setattr(renderer, "_ensure_runtime", _fake_ensure_runtime)
    monkeypatch.setattr(
        renderer,
        "generate_mv_clip",
        lambda **kwargs: [Image.new(mode="RGB", size=(32, 32), color=(0, 0, 0)) for _ in range(12)],
    )
    def _fake_run_ffmpeg_command(*, command, command_name):  # noqa: ANN001
        _ = command_name
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")

    monkeypatch.setattr(renderer, "_run_ffmpeg_command", _fake_run_ffmpeg_command)

    renderer.render_one_unit_animatediff(
        context=context,
        unit=unit,
        prompt="line art city",
        encoder_command_args=["-c:v", "libx264"],
        device_override="cuda:0",
    )

    assert captured["device_override"] == "cuda:0"


def test_build_frames_encode_command_should_include_nostdin() -> None:
    """
    功能说明：验证 AnimateDiff 帧编码命令默认带 -nostdin，防止交互式卡死。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验命令拼装，不执行 ffmpeg。
    """
    command = renderer._build_frames_encode_command(
        ffmpeg_bin="ffmpeg",
        frames_pattern="/tmp/f_%05d.png",
        exact_frames=24,
        fps=24,
        encoder_command_args=["-c:v", "libx264"],
        output_path="/tmp/out.mp4",
    )
    assert command[0] == "ffmpeg"
    assert "-nostdin" in command


def test_generate_mv_clip_should_use_inference_mode_and_empty_cuda_cache() -> None:
    """
    功能说明：验证 AnimateDiff 推理阶段会进入 inference_mode 并尝试执行 empty_cache。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过最小 fake runtime 覆盖推理调用，不依赖真实 torch/diffusers。
    """
    called = {
        "inference_mode_entered": False,
        "empty_cache_called": False,
    }

    class _FakeInferenceMode:
        def __enter__(self):  # noqa: ANN204
            called["inference_mode_entered"] = True
            return None

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            _ = (exc_type, exc, tb)
            return False

    class _FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            called["empty_cache_called"] = True

    class _FakeTorch:
        cuda = _FakeCuda()

        @staticmethod
        def inference_mode() -> _FakeInferenceMode:
            return _FakeInferenceMode()

    class _FakePipelineOutput:
        def __init__(self) -> None:
            self.frames = [[Image.new(mode="RGB", size=(32, 32), color=(0, 0, 0))]]

    class _FakePipeline:
        def __call__(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return _FakePipelineOutput()

    runtime = {
        "pipeline": _FakePipeline(),
        "torch": _FakeTorch(),
        "device": "cuda:1",
    }

    frames = renderer.generate_mv_clip(
        prompt="line art city",
        num_frames=1,
        runtime=runtime,
        width=64,
        height=64,
        negative_prompt="",
        guidance_scale=7.0,
        steps=12,
        seed=None,
    )

    assert len(frames) == 1
    assert called["inference_mode_entered"] is True
    assert called["empty_cache_called"] is True


def test_resample_frames_uniform_should_output_exact_target_length() -> None:
    """
    功能说明：验证均匀重采样后输出帧数与目标帧数严格一致。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同时覆盖上采样与下采样。
    """
    source_frames = [Image.new(mode="RGB", size=(8, 8), color=(index, index, index)) for index in range(5)]
    upsampled = renderer._resample_frames_uniform(frames=source_frames, target_frames=12)
    downsampled = renderer._resample_frames_uniform(frames=source_frames, target_frames=3)
    assert len(upsampled) == 12
    assert len(downsampled) == 3


def test_prewarm_animatediff_runtime_should_only_call_ensure_runtime(monkeypatch) -> None:
    """
    功能说明：验证预热入口仅调用 runtime 初始化缓存，不触发推理路径。
    参数说明：
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩 _ensure_runtime 验证调用参数与返回摘要。
    """
    captured: dict[str, str] = {}

    def _fake_ensure_runtime(*, context, device_override=None):  # noqa: ANN001
        _ = context
        captured["device_override"] = str(device_override)
        return {
            "device": str(device_override or "cuda:1"),
            "cache_key": "mock-cache-key",
        }

    monkeypatch.setattr(renderer, "_ensure_runtime", _fake_ensure_runtime)
    monkeypatch.setattr(
        renderer,
        "generate_mv_clip",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("prewarm should not call generate_mv_clip")),
    )

    result = renderer.prewarm_animatediff_runtime(
        context=SimpleNamespace(logger=SimpleNamespace(info=lambda *args, **kwargs: None)),
        device_override="cuda:0",
    )

    assert captured["device_override"] == "cuda:0"
    assert result["device"] == "cuda:0"
    assert result["cache_key"] == "mock-cache-key"
