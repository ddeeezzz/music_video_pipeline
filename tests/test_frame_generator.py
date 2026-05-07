"""
文件用途：验证模块 C 关键帧生成器工厂已收口为 ComfyUI 唯一路径。
核心流程：构造最小配置对象，断言工厂拒绝旧模式并返回 ComfyUI 生成器。
输入输出：输入最小配置对象，输出工厂行为断言。
依赖说明：依赖 pytest 与项目内模块 C 关键帧生成器工厂。
维护说明：本文件只覆盖当前真实路径，不再保留旧占位/diffusion 旧测试。
"""

# 标准库：用于日志对象构建。
import logging

# 第三方库：用于异常断言。
import pytest

# 项目内模块：配置数据类。
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, ModuleAConfig, ModuleCConfig, PathsConfig
# 项目内模块：关键帧生成器工厂。
from music_video_pipeline.generators.frame_generator import build_keyframe_generator
# 项目内模块：ComfyUI 关键帧生成器实现。
from music_video_pipeline.generators.comfyui_frame_generator import ComfyUIFrameGenerator



def test_build_keyframe_generator_should_reject_non_comfyui_mode() -> None:
    """
    功能说明：验证模块 C 工厂会拒绝非 comfyui 模式。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧模式名仅用于验证硬切行为。
    """
    with pytest.raises(RuntimeError, match="仅支持 comfyui"):
        build_keyframe_generator(mode="legacy_placeholder", logger=logging.getLogger("test_frame_generator_reject"), app_config=_build_app_config())



def test_build_keyframe_generator_should_require_app_config() -> None:
    """
    功能说明：验证模块 C 工厂缺少 app_config 时会直接失败。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：防止调用方误以为存在本地无配置回退路径。
    """
    with pytest.raises(RuntimeError, match="缺少 app_config"):
        build_keyframe_generator(mode="comfyui", logger=logging.getLogger("test_frame_generator_require_config"), app_config=None)



def test_build_keyframe_generator_should_return_comfyui_generator() -> None:
    """
    功能说明：验证模块 C 工厂会构造 ComfyUIFrameGenerator。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证工厂收口，不触发真实 ComfyUI 请求。
    """
    generator = build_keyframe_generator(
        mode="comfyui",
        logger=logging.getLogger("test_frame_generator_comfyui"),
        app_config=_build_app_config(),
    )
    assert isinstance(generator, ComfyUIFrameGenerator)


def test_comfyui_frame_generator_should_prewarm_service_and_assets(tmp_path, monkeypatch) -> None:
    """
    功能说明：验证 ComfyUI 关键帧生成器的 prewarm 会先探活服务并校验关键模型资产。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不触发真实 ComfyUI 请求。
    """
    app_config = _build_app_config()
    generator = build_keyframe_generator(
        mode="comfyui",
        logger=logging.getLogger("test_frame_generator_prewarm"),
        app_config=app_config,
    )
    called = {"ready": 0}
    monkeypatch.setattr(generator._client, "ensure_service_ready", lambda: called.__setitem__("ready", called["ready"] + 1))
    generator._project_root = tmp_path
    for relative_path in [
        app_config.module_c.comfyui.unet_file,
        app_config.module_c.comfyui.clip_file,
        app_config.module_c.comfyui.vae_file,
        app_config.module_c.comfyui.style_lora_file,
    ]:
        asset_path = tmp_path / str(relative_path)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(b"ok")

    generator.prewarm()

    assert called["ready"] == 1



def _build_app_config() -> AppConfig:
    """
    功能说明：构造关键帧生成器测试所需的最小配置对象。
    参数说明：无。
    返回值：
    - AppConfig: 最小可用配置对象。
    异常说明：无。
    边界条件：不要求真实 ComfyUI 服务在线。
    """
    return AppConfig(
        paths=PathsConfig(runs_dir="runs", default_audio_path="demo.mp3"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=24,
        ),
        logging=LoggingConfig(level="INFO"),
        module_c=ModuleCConfig(),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
