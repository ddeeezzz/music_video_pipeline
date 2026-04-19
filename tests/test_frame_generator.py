"""
文件用途：验证模块C占位图的中文字体与自适应排版行为。
核心流程：构造中文分镜数据，检查图片生成、字体优先加载、场景文本换行截断。
输入输出：输入临时目录与伪造分镜，输出断言结果。
依赖说明：依赖 pytest、Pillow 与项目内 frame_generator。
维护说明：若占位图布局策略调整，需要同步更新断言。
"""

# 标准库：用于路径处理
import logging
from pathlib import Path
# 标准库：用于JSON写入
import json

# 第三方库：用于图像绘制对象构建
from PIL import Image, ImageDraw
# 第三方库：用于异常断言
import pytest

# 项目内模块：占位图生成器实现与内部排版工具
from music_video_pipeline.generators import frame_generator as frame_generator_module
from music_video_pipeline.generators.frame_generator import (
    DiffusionFrameGenerator,
    MockFrameGenerator,
    _extract_audio_role_display_for_shot,
    _extract_big_segment_display_for_shot,
    _extract_lyric_text_for_shot,
    _load_chinese_font,
    _load_module_c_real_profile,
    _measure_text_pixel_width,
    _resolve_binding_runtime_assets,
    _resolve_runtime_device,
    _resolve_torch_dtype,
    _wrap_text_by_pixel_width,
)


def test_mock_frame_generator_should_render_chinese_scene_text(tmp_path: Path) -> None:
    """
    功能说明：验证中文场景文案可生成占位图且不抛异常。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证生成成功与图片尺寸，不做像素级 OCR 检查。
    """
    generator = MockFrameGenerator()
    shots = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 2.5,
            "scene_desc": "夜色城市街景，霓虹灯与雨幕交织，人物缓慢前行",
            "camera_motion": "slow_pan",
            "transition": "crossfade",
        }
    ]

    frame_item = generator.generate_one(
        shot=shots[0],
        output_dir=tmp_path / "frames",
        width=960,
        height=540,
        shot_index=0,
    )

    frame_path = Path(frame_item["frame_path"])
    assert frame_path.exists()

    image = Image.open(frame_path)
    assert image.size == (960, 540)


def test_mock_frame_generator_should_render_lyrics_text_with_new_fields(tmp_path: Path) -> None:
    """
    功能说明：验证分镜包含歌词扩展字段时，占位图可正常生成。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：歌词超长时应通过自动换行与截断处理。
    """
    generator = MockFrameGenerator()
    shots = [
        {
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 3.0,
            "scene_desc": "雨夜下的城市街口，镜头缓慢推进",
            "camera_motion": "zoom_in",
            "transition": "crossfade",
            "lyric_text": "这是很长的一句歌词用于测试自动换行与布局保护 这是第二句歌词继续测试",
            "lyric_units": [
                {"start_time": 0.1, "end_time": 1.2, "text": "这是很长的一句歌词用于测试自动换行与布局保护", "confidence": 0.9},
                {"start_time": 1.3, "end_time": 2.7, "text": "这是第二句歌词继续测试", "confidence": 0.85},
            ],
            "big_segment_id": "big_003",
            "big_segment_label": "chorus",
            "segment_label": "chorus",
            "audio_role": "vocal",
        }
    ]

    frame_item = generator.generate_one(
        shot=shots[0],
        output_dir=tmp_path / "frames",
        width=960,
        height=540,
        shot_index=0,
    )
    assert Path(frame_item["frame_path"]).exists()


def test_extract_lyric_text_for_shot_should_fallback_to_lyric_units() -> None:
    """
    功能说明：验证当 lyric_text 为空时可从 lyric_units 聚合展示文本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：lyric_units 中空文本会被忽略。
    """
    shot = {
        "lyric_text": "",
        "lyric_units": [
            {"text": "第一句"},
            {"text": "  "},
            {"text": "第二句"},
        ],
    }
    assert _extract_lyric_text_for_shot(shot=shot) == "第一句 第二句"


def test_extract_lyric_text_for_shot_should_return_unknown_marker_when_no_reliable_text() -> None:
    """
    功能说明：验证 lyric_units 仅包含未识别标记时，返回“未识别歌词”。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：兼容 lyric_text 缺失的旧分镜结构。
    """
    shot = {
        "lyric_units": [
            {"text": "吟唱"},
            {"text": "[未识别歌词]"},
        ],
    }
    assert _extract_lyric_text_for_shot(shot=shot) == "[未识别歌词]"


def test_extract_lyric_text_for_shot_should_filter_punctuation_only_text() -> None:
    """
    功能说明：验证占位图歌词提取会过滤纯标点文本，避免标点单独上屏。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当 lyric_text 为纯标点时，应回退到 lyric_units 中的有效文本。
    """
    shot = {
        "lyric_text": "，",
        "lyric_units": [
            {"text": "。"},
            {"text": "  你好  "},
        ],
    }
    assert _extract_lyric_text_for_shot(shot=shot) == "你好"


def test_extract_lyric_text_for_shot_should_append_note_for_instrumental_with_lyrics() -> None:
    """
    功能说明：验证器乐段存在歌词文本时，会在歌词后追加固定说明文案。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅对有效歌词文本追加说明，不影响纯标点过滤。
    """
    shot = {
        "audio_role": "instrumental",
        "lyric_text": "当白天像是",
    }
    lyric_text = _extract_lyric_text_for_shot(shot=shot)
    assert lyric_text.startswith("当白天像是")
    assert "根据音源分离后的能量检测" in lyric_text
    assert "Fun-ASR 识别到了歌词" in lyric_text


def test_extract_big_segment_display_for_shot_should_format_label_and_id() -> None:
    """
    功能说明：验证大段落展示文本按“标签+ID”格式输出。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：字段缺失时应返回“<未知>”。
    """
    assert _extract_big_segment_display_for_shot(
        shot={"big_segment_label": "chorus", "big_segment_id": "big_003"}
    ) == "chorus (big_003)"
    assert _extract_big_segment_display_for_shot(shot={}) == "<未知>"


def test_extract_audio_role_display_for_shot_should_map_role_text() -> None:
    """
    功能说明：验证 audio_role 到中文段落类型文案的映射。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：非法值应降级为“<未知>”。
    """
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "instrumental"}) == "器乐段"
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "vocal"}) == "人声段"
    assert _extract_audio_role_display_for_shot(shot={"audio_role": "other"}) == "<未知>"


def test_load_chinese_font_should_prioritize_repo_bundled_font() -> None:
    """
    功能说明：验证字体加载优先使用仓库内置字体文件。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当仓库字体文件存在时必须命中该路径。
    """
    _, font_source = _load_chinese_font(size=28)
    assert font_source.endswith("resources/fonts/NotoSansCJKsc-Regular.otf")


def test_wrap_text_by_pixel_width_should_clip_to_max_lines_with_ellipsis() -> None:
    """
    功能说明：验证超长中文文本会按像素宽度换行并在超限时追加省略号。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：逐字换行需兼容中文无空格文本。
    """
    font_obj, _ = _load_chinese_font(size=24)
    image = Image.new(mode="RGB", size=(500, 300), color=(0, 0, 0))
    drawer = ImageDraw.Draw(image)
    long_scene_text = "场景：" + "这是一个用于测试中文自动换行与截断行为的超长描述文本" * 6

    max_width = 260
    lines = _wrap_text_by_pixel_width(
        drawer=drawer,
        text=long_scene_text,
        font_obj=font_obj,
        max_width=max_width,
        max_lines=3,
    )

    assert len(lines) == 3
    assert lines[-1].endswith("...")
    assert all(_measure_text_pixel_width(drawer, line, font_obj) <= max_width for line in lines)


def test_load_module_c_real_profile_should_fill_defaults(tmp_path: Path) -> None:
    """
    功能说明：验证模块C真实配置可在缺省可选字段时自动回填默认值。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：binding_name/model_series 为必填。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    profile_path = config_dir / "module_c_real_default.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "module_c_real": {
                    "binding_name": "xiantiao_style",
                    "model_series": "15",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profile = _load_module_c_real_profile(project_root=project_root)
    assert profile.binding_name == "xiantiao_style"
    assert profile.model_series == "15"
    assert profile.steps == 24
    assert profile.seed_mode == "shot_index"
    assert profile.scheduler == "default"


def test_load_module_c_real_profile_should_fail_when_required_field_missing(tmp_path: Path) -> None:
    """
    功能说明：验证模块C真实配置缺失必填字段时会抛错。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证 binding_name 缺失场景。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    profile_path = config_dir / "module_c_real_default.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "module_c_real": {
                    "model_series": "15",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="binding_name"):
        _load_module_c_real_profile(project_root=project_root)


def test_load_module_c_real_profile_should_fail_when_unknown_field_exists(tmp_path: Path) -> None:
    """
    功能说明：验证模块C真实配置含未知字段时会抛错。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证 module_c_real 内未知字段。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    profile_path = config_dir / "module_c_real_default.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "module_c_real": {
                    "binding_name": "xiantiao_style",
                    "model_series": "15",
                    "unknown_field": "x",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="未知字段"):
        _load_module_c_real_profile(project_root=project_root)


def test_resolve_binding_runtime_assets_should_resolve_lora_and_base_model_paths(tmp_path: Path) -> None:
    """
    功能说明：验证 binding_name 到 base_model_key 与路径解析链路可成功。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：底模路径以 base_model_registry 为准。
    """
    project_root = tmp_path / "project"
    lora_file = project_root / "models" / "lora" / "15" / "xiantiao_style" / "xiantiao_style.safetensors"
    base_model_path = project_root / "models" / "base_model" / "15" / "diffusers" / "revAnimated_v122"
    lora_file.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.mkdir(parents=True, exist_ok=True)
    lora_file.write_bytes(b"fake-lora")

    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "lora_bindings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "bindings": [
                    {
                        "binding_name": "xiantiao_style",
                        "model_series": "15",
                        "base_model_key": "base_15_diffusers_revanimated_v122",
                        "lora_file": "models/lora/15/xiantiao_style/xiantiao_style.safetensors",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "base_models": [
                    {
                        "key": "base_15_diffusers_revanimated_v122",
                        "series": "15",
                        "enabled": True,
                        "path": "models/base_model/15/diffusers/revAnimated_v122",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    assets = _resolve_binding_runtime_assets(
        project_root=project_root,
        binding_name="xiantiao_style",
        model_series="15",
    )
    assert assets["base_model_key"] == "base_15_diffusers_revanimated_v122"
    assert Path(assets["base_model_path"]).resolve() == base_model_path.resolve()
    assert Path(assets["lora_file_path"]).resolve() == lora_file.resolve()


def test_resolve_binding_runtime_assets_should_fail_when_binding_missing(tmp_path: Path) -> None:
    """
    功能说明：验证 binding 不存在时会抛出明确错误。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证 binding 缺失分支。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "lora_bindings.json").write_text(
        json.dumps({"version": 1, "bindings": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps({"version": 1, "base_models": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="绑定不存在"):
        _resolve_binding_runtime_assets(
            project_root=project_root,
            binding_name="xiantiao_style",
            model_series="15",
        )


def test_resolve_binding_runtime_assets_should_fail_when_base_model_key_missing(tmp_path: Path) -> None:
    """
    功能说明：验证绑定指向的 base_model_key 不存在时会抛出明确错误。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证 base_model_key 不存在分支。
    """
    project_root = tmp_path / "project"
    lora_file = project_root / "models" / "lora" / "15" / "xiantiao_style" / "xiantiao_style.safetensors"
    lora_file.parent.mkdir(parents=True, exist_ok=True)
    lora_file.write_bytes(b"fake-lora")
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "lora_bindings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "bindings": [
                    {
                        "binding_name": "xiantiao_style",
                        "model_series": "15",
                        "base_model_key": "missing_base_key",
                        "lora_file": "models/lora/15/xiantiao_style/xiantiao_style.safetensors",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps({"version": 1, "base_models": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="底模 key 不存在"):
        _resolve_binding_runtime_assets(
            project_root=project_root,
            binding_name="xiantiao_style",
            model_series="15",
        )


def test_resolve_binding_runtime_assets_should_fail_when_base_model_path_missing(tmp_path: Path) -> None:
    """
    功能说明：验证底模路径不存在时会抛出明确错误。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证底模路径不存在分支。
    """
    project_root = tmp_path / "project"
    lora_file = project_root / "models" / "lora" / "15" / "xiantiao_style" / "xiantiao_style.safetensors"
    lora_file.parent.mkdir(parents=True, exist_ok=True)
    lora_file.write_bytes(b"fake-lora")
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "lora_bindings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "bindings": [
                    {
                        "binding_name": "xiantiao_style",
                        "model_series": "15",
                        "base_model_key": "base_15_diffusers_revanimated_v122",
                        "lora_file": "models/lora/15/xiantiao_style/xiantiao_style.safetensors",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "base_models": [
                    {
                        "key": "base_15_diffusers_revanimated_v122",
                        "series": "15",
                        "enabled": True,
                        "path": "models/base_model/15/diffusers/not_exist_model",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="底模路径不存在"):
        _resolve_binding_runtime_assets(
            project_root=project_root,
            binding_name="xiantiao_style",
            model_series="15",
        )


def test_diffusion_frame_generator_should_run_real_generation_with_runtime_assets(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证扩散生成器会按绑定加载资产并输出真实关键帧清单结构。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过桩替换 diffusers/torch，避免真实模型加载。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "module_c_real_default.json").write_text(
        json.dumps(
            {
                "version": 1,
                "module_c_real": {
                    "binding_name": "xiantiao_style",
                    "model_series": "15",
                    "lora_scale": 0.8,
                    "steps": 12,
                    "guidance_scale": 5.5,
                    "negative_prompt": "bad",
                    "device": "auto",
                    "torch_dtype": "auto",
                    "scheduler": "ddim",
                    "seed_mode": "shot_index",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lora_file = project_root / "models" / "lora" / "15" / "xiantiao_style" / "xiantiao_style.safetensors"
    base_model_path = project_root / "models" / "base_model" / "15" / "diffusers" / "revAnimated_v122"
    lora_file.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.mkdir(parents=True, exist_ok=True)
    lora_file.write_bytes(b"fake-lora")
    (config_dir / "lora_bindings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "bindings": [
                    {
                        "binding_name": "xiantiao_style",
                        "model_series": "15",
                        "base_model_key": "base_15_diffusers_revanimated_v122",
                        "lora_file": "models/lora/15/xiantiao_style/xiantiao_style.safetensors",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "base_models": [
                    {
                        "key": "base_15_diffusers_revanimated_v122",
                        "series": "15",
                        "enabled": True,
                        "path": "models/base_model/15/diffusers/revAnimated_v122",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(frame_generator_module, "_resolve_project_root", lambda: project_root)

    class _FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorchGenerator:
        def __init__(self, device: str) -> None:
            self.device = device
            self.seed_value: int | None = None

        def manual_seed(self, seed_value: int) -> "_FakeTorchGenerator":
            self.seed_value = seed_value
            return self

    class _FakeTorch:
        cuda = _FakeTorchCuda()
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"
        Generator = _FakeTorchGenerator

    class _FakeScheduler:
        @classmethod
        def from_config(cls, config: dict) -> dict:
            return {"name": cls.__name__, "config": dict(config)}

    class _FakePipeline:
        last_from_pretrained_kwargs: dict | None = None

        def __init__(self) -> None:
            self.scheduler = type("_S", (), {"config": {"origin": "default"}})()
            self.device = "cpu"
            self.loaded_lora: tuple[str, str] | None = None
            self.last_call_kwargs: dict[str, object] | None = None

        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: object) -> "_FakePipeline":
            cls.last_from_pretrained_kwargs = {"model_path": model_path, **kwargs}
            return cls()

        def to(self, device: str) -> "_FakePipeline":
            self.device = device
            return self

        def load_lora_weights(self, lora_dir: str, weight_name: str) -> None:
            self.loaded_lora = (lora_dir, weight_name)

        def __call__(self, **kwargs: object) -> object:
            self.last_call_kwargs = kwargs
            image_obj = Image.new(mode="RGB", size=(int(kwargs["width"]), int(kwargs["height"])), color=(1, 2, 3))
            return type("_Output", (), {"images": [image_obj]})()

    monkeypatch.setattr(
        frame_generator_module,
        "_import_diffusion_runtime_dependencies",
        lambda: {
            "torch": _FakeTorch,
            "StableDiffusionPipeline": _FakePipeline,
            "EulerAncestralDiscreteScheduler": _FakeScheduler,
            "DDIMScheduler": _FakeScheduler,
        },
    )

    generator = DiffusionFrameGenerator(logger=frame_generator_module.logging.getLogger("test_diffusion_real"))
    frame_item = generator.generate_one(
        shot={
            "shot_id": "shot_001",
            "start_time": 0.0,
            "end_time": 1.5,
            "scene_desc": "real scene",
            "keyframe_prompt_en": "line art girl",
            "camera_motion": "none",
            "transition": "hard_cut",
        },
        output_dir=tmp_path / "frames",
        width=128,
        height=96,
        shot_index=0,
    )

    assert Path(frame_item["frame_path"]).exists()
    assert frame_item["binding_name"] == "xiantiao_style"
    assert frame_item["base_model_key"] == "base_15_diffusers_revanimated_v122"
    assert frame_item["lora_file"].endswith("xiantiao_style.safetensors")


def test_diffusion_frame_generator_should_fail_immediately_when_binding_missing(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 diffusion 生成器在 binding 缺失时会立即抛错，不会降级到 Mock。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：依赖导入会被桩替换，确保错误来源仅为绑定缺失。
    """
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "module_c_real_default.json").write_text(
        json.dumps(
            {
                "version": 1,
                "module_c_real": {
                    "binding_name": "not_exist_binding",
                    "model_series": "15",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "lora_bindings.json").write_text(
        json.dumps({"version": 1, "bindings": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config_dir / "base_model_registry.json").write_text(
        json.dumps({"version": 1, "base_models": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_generator_module, "_resolve_project_root", lambda: project_root)
    monkeypatch.setattr(
        frame_generator_module,
        "_import_diffusion_runtime_dependencies",
        lambda: {
            "torch": object(),
            "StableDiffusionPipeline": object(),
            "EulerAncestralDiscreteScheduler": object(),
            "DDIMScheduler": object(),
        },
    )
    generator = DiffusionFrameGenerator(logger=frame_generator_module.logging.getLogger("test_diffusion_binding_missing"))
    with pytest.raises(RuntimeError, match="绑定不存在"):
        generator.generate_one(
            shot={
                "shot_id": "shot_001",
                "start_time": 0.0,
                "end_time": 1.0,
                "keyframe_prompt_en": "line art",
                "scene_desc": "x",
                "camera_motion": "none",
                "transition": "hard_cut",
            },
            output_dir=tmp_path / "frames",
            width=128,
            height=128,
            shot_index=0,
        )


def test_resolve_runtime_device_should_support_explicit_cuda_index() -> None:
    """
    功能说明：验证模块C扩散设备解析支持 cuda:N 形式，满足 C/D 分卡策略。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证解析逻辑，不依赖真实 GPU。
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

    profile_obj = type("_Profile", (), {"device": "cuda:0"})()
    assert _resolve_runtime_device(profile=profile_obj, torch_module=_FakeTorch()) == "cuda:0"


def test_resolve_runtime_device_should_fallback_to_cuda0_when_index_out_of_range(caplog) -> None:
    """
    功能说明：验证模块C扩散在单卡场景收到 cuda:N 越界配置时自动回退到 cuda:0。
    参数说明：
    - caplog: pytest 日志捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证解析与日志，不依赖真实 GPU。
    """

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    class _FakeTorch:
        cuda = _FakeCuda()

    profile_obj = type("_Profile", (), {"device": "cuda:1"})()
    with caplog.at_level(logging.WARNING):
        resolved_device = _resolve_runtime_device(profile=profile_obj, torch_module=_FakeTorch())
    assert resolved_device == "cuda:0"
    assert "设备索引越界" in caplog.text


def test_resolve_torch_dtype_auto_should_use_float16_on_cuda_index_device() -> None:
    """
    功能说明：验证模块C扩散在 device=cuda:N 且 torch_dtype=auto 时仍使用 float16。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证 dtype 解析逻辑，不依赖真实 torch。
    """

    class _FakeTorch:
        float16 = "fp16"
        float32 = "fp32"
        bfloat16 = "bf16"

    profile_obj = type("_Profile", (), {"torch_dtype": "auto"})()
    resolved = _resolve_torch_dtype(profile=profile_obj, torch_module=_FakeTorch(), device="cuda:0")
    assert resolved == "fp16"
