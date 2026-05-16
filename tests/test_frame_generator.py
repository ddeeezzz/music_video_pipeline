"""
文件用途：验证模块 C 关键帧生成器工厂与 ComfyUI 路由行为。
核心流程：构造最小配置对象，断言工厂收口、预热校验与 asset_kind 路由逻辑。
输入输出：输入最小配置对象与临时目录，输出工厂和工作流绑定行为断言。
依赖说明：依赖 pytest 与项目内模块 C 关键帧生成器实现。
维护说明：本文件只覆盖当前真实 ComfyUI 路径与素材类型路由，不保留旧字段测试。
"""

# 标准库：用于日志对象构建。
from dataclasses import replace
import logging
# 标准库：用于路径处理。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest

# 项目内模块：配置数据类。
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, ModuleAConfig, ModuleCConfig, PathsConfig
# 项目内模块：ComfyUI 契约工具。
from music_video_pipeline.comfyui import load_workflow_contract, render_workflow_from_contract
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
        app_config.module_c.comfyui.checkpoint_file,
        app_config.module_c.comfyui.scene_lora_file,
        app_config.module_c.comfyui.char_lora_file,
    ]:
        asset_path = tmp_path / str(relative_path)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(b"ok")

    generator.prewarm()

    assert called["ready"] == 1


def test_comfyui_frame_generator_should_route_prop_without_any_lora(tmp_path, monkeypatch) -> None:
    """
    功能说明：验证 asset_kind=prop 时会切换到 prop contract，且不注入任何 LoRA 绑定。
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
        logger=logging.getLogger("test_frame_generator_prop"),
        app_config=app_config,
    )
    generator._project_root = tmp_path
    generator._contract_prop_start = replace(generator._contract_prop_start, output_node_id="901")
    generator._contract_prop_end = replace(generator._contract_prop_end, output_node_id="902")

    for relative_path in [
        app_config.module_c.comfyui.checkpoint_file,
        app_config.module_c.comfyui.scene_lora_file,
        app_config.module_c.comfyui.char_lora_file,
    ]:
        asset_path = tmp_path / str(relative_path)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(b"ok")

    captured_prompts: list[dict] = []

    def _fake_execute_prompt(*, workflow_prompt, output_node_id):
        captured_prompts.append({"prompt": workflow_prompt, "output_node_id": output_node_id})
        output_file = tmp_path / f"output_{len(captured_prompts)}.png"
        output_file.write_bytes(b"png")
        return [output_file]

    monkeypatch.setattr(generator._client, "ensure_service_ready", lambda: None)
    monkeypatch.setattr(generator._client, "execute_prompt", _fake_execute_prompt)
    monkeypatch.setattr(generator._client, "stage_input_image", lambda source_path, prefix: f"mvpl/{prefix}.png")

    shot = {
        "shot_id": "shot_prop_001",
        "asset_kind": "prop",
        "start_time": 0.0,
        "end_time": 1.0,
        "scene_desc": "symbolic mirror",
        "keyframe_prompt_start_zh": "镜子",
        "keyframe_prompt_start_en": "broken mirror",
        "keyframe_negative_prompt_start_zh": "彩色污染",
        "keyframe_negative_prompt_start_en": "color contamination",
        "keyframe_prompt_end_zh": "镜子裂开",
        "keyframe_prompt_end_en": "broken mirror cracking",
        "keyframe_negative_prompt_end_zh": "彩色污染",
        "keyframe_negative_prompt_end_en": "color contamination",
        "video_prompt_zh": "镜子视频",
        "video_prompt_en": "mirror video",
    }

    result = generator.generate_one(
        shot=shot,
        output_dir=tmp_path / "frames",
        width=768,
        height=432,
        shot_index=0,
    )

    assert result["asset_kind"] == "prop"
    assert len(captured_prompts) == 2
    start_prompt = captured_prompts[0]["prompt"]
    end_prompt = captured_prompts[1]["prompt"]
    assert captured_prompts[0]["output_node_id"] == "901"
    assert captured_prompts[1]["output_node_id"] == "902"
    assert "4" not in start_prompt
    assert "5" not in start_prompt
    assert "4" not in end_prompt
    assert "5" not in end_prompt


def test_comfyui_character_workflows_should_keep_start_dual_lora_and_end_single_lora_chain() -> None:
    """
    功能说明：验证人物 start workflow 保持双 LoRA 链，end workflow 与 GUI 一致走单 LoRA + 规则型去背后处理链。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只验证节点连线与关键输入，不触发真实 ComfyUI 请求。
    """
    start_contract = load_workflow_contract("configs/comfyui/module_c_start.contract.json")
    end_contract = load_workflow_contract("configs/comfyui/module_c_end.contract.json")

    start_workflow = render_workflow_from_contract(
        contract=start_contract,
        binding_values={
            "checkpoint_name": "base.safetensors",
            "scene_lora_name": "scene/test_scene.safetensors",
            "scene_lora_strength_model": 0.9,
            "scene_lora_strength_clip": 0.9,
            "char_lora_name": "char/test_char.safetensors",
            "char_lora_strength_model": 0.8,
            "char_lora_strength_clip": 0.8,
            "positive_prompt": "face close-up",
            "negative_prompt": "bad quality",
            "width": 768,
            "height": 432,
            "seed": 123,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "filename_prefix": "test/start",
        },
    )
    end_workflow = render_workflow_from_contract(
        contract=end_contract,
        binding_values={
            "checkpoint_name": "base.safetensors",
            "char_lora_name": "char/test_char.safetensors",
            "char_lora_strength_model": 0.8,
            "char_lora_strength_clip": 0.8,
            "init_image": "mvpl/init.png",
            "positive_prompt": "face close-up end",
            "negative_prompt": "bad quality",
            "seed": 456,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.55,
            "filename_prefix": "test/end",
        },
    )

    assert start_workflow["5"]["inputs"]["model"] == ["4", 0]
    assert start_workflow["20"]["inputs"]["clip"] == ["5", 1]
    assert start_workflow["40"]["inputs"]["model"] == ["5", 0]
    assert start_workflow["85"]["class_type"] == "ImageScale"
    assert start_workflow["91"]["class_type"] == "MVPL: ExtractComicAlpha"
    assert start_workflow["91"]["inputs"]["image"] == ["85", 0]
    assert start_workflow["99"]["inputs"]["images"] == ["100", 0]

    assert "4" not in end_workflow
    assert end_workflow["5"]["inputs"]["model"] == ["1", 0]
    assert end_workflow["20"]["inputs"]["clip"] == ["5", 1]
    assert end_workflow["10"]["inputs"]["image"] == "mvpl/init.png"
    assert end_workflow["11"]["inputs"]["pixels"] == ["10", 0]
    assert end_workflow["40"]["inputs"]["latent_image"] == ["11", 0]
    assert end_workflow["85"]["class_type"] == "ImageScale"
    assert end_workflow["147"]["class_type"] == "MVPL: ConvertToGrayscale"
    assert end_workflow["147"]["inputs"]["image"] == ["85", 0]
    assert end_workflow["91"]["class_type"] == "MVPL: ExtractComicAlpha"
    assert end_workflow["91"]["inputs"]["image"] == ["147", 0]
    assert end_workflow["99"]["inputs"]["images"] == ["91", 0]


def test_comfyui_prop_workflows_should_skip_scene_lora_and_character_lora() -> None:
    """
    功能说明：验证 prop start/end workflow 的 contract 绑定后不加载任何 LoRA，并走规则型去背后处理链。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只验证节点连线与关键输入，不触发真实 ComfyUI 请求。
    """
    start_contract = load_workflow_contract("configs/comfyui/module_c_prop_start.contract.json")
    end_contract = load_workflow_contract("configs/comfyui/module_c_prop_end.contract.json")

    start_workflow = render_workflow_from_contract(
        contract=start_contract,
        binding_values={
            "checkpoint_name": "base.safetensors",
            "positive_prompt": "broken mirror",
            "negative_prompt": "portrait",
            "width": 768,
            "height": 432,
            "seed": 123,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "filename_prefix": "test/prop_start",
        },
    )
    end_workflow = render_workflow_from_contract(
        contract=end_contract,
        binding_values={
            "checkpoint_name": "base.safetensors",
            "init_image": "mvpl/init_prop.png",
            "positive_prompt": "broken mirror end",
            "negative_prompt": "portrait",
            "seed": 456,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.55,
            "filename_prefix": "test/prop_end",
        },
    )

    assert "4" not in start_workflow
    assert "5" not in start_workflow
    assert start_workflow["20"]["inputs"]["clip"] == ["1", 1]
    assert start_workflow["40"]["inputs"]["model"] == ["1", 0]
    assert start_workflow["85"]["class_type"] == "ImageScale"
    assert start_workflow["91"]["class_type"] == "MVPL: ExtractComicAlpha"
    assert start_workflow["91"]["inputs"]["image"] == ["85", 0]
    assert start_workflow["99"]["inputs"]["images"] == ["100", 0]

    assert "4" not in end_workflow
    assert "5" not in end_workflow
    assert end_workflow["20"]["inputs"]["clip"] == ["1", 1]
    assert end_workflow["40"]["inputs"]["model"] == ["1", 0]
    assert end_workflow["10"]["inputs"]["image"] == "mvpl/init_prop.png"
    assert end_workflow["11"]["inputs"]["pixels"] == ["10", 0]
    assert end_workflow["40"]["inputs"]["latent_image"] == ["11", 0]
    assert end_workflow["85"]["class_type"] == "ImageScale"
    assert end_workflow["91"]["class_type"] == "MVPL: ExtractComicAlpha"
    assert end_workflow["91"]["inputs"]["image"] == ["85", 0]
    assert end_workflow["99"]["inputs"]["images"] == ["100", 0]


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
