"""
文件用途：验证模块A V2中 allin1 后端调用的设备与精度兜底逻辑。
核心流程：打桩 allin1 后端，覆盖 device 透传与 CUDA 下模型 FP32 强制加载路径。
输入输出：输入临时音频与伪后端，输出解析后的标准化段落与节拍结果。
依赖说明：依赖 pytest、logging 与 module_a_v2.backends.allin1。
维护说明：若后端调用签名调整，需要同步更新本文件中的桩函数。
"""

# 标准库：日志对象
import logging
# 标准库：简单对象容器
from types import SimpleNamespace
# 标准库：路径处理
from pathlib import Path

# 第三方库：张量计算
import torch

# 项目内模块：allin1 封装入口
from music_video_pipeline.modules.module_a_v2.backends.allin1 import (
    analyze_with_allin1,
)


def test_analyze_with_allin1_should_run_once_on_cuda_when_backend_success(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 CUDA 场景会透传设备参数并单次完成调用。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证调用顺序与返回结构，不覆盖真实模型推理。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_allin1_backend_retry")
    call_devices: list[str] = []

    def _fake_analyze(*_args, **kwargs):
        runtime_device = str(kwargs.get("device", ""))
        call_devices.append(runtime_device)
        return {
            "segments": [{"start": 0.0, "end": 2.0, "label": "verse"}],
            "beats": [0.0, 1.0],
            "beat_positions": [1, 2],
        }

    fake_backend = SimpleNamespace(analyze=_fake_analyze)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.backends.allin1._import_allin1_backend",
        lambda: ("allin1fix", fake_backend),
    )

    output = analyze_with_allin1(
        audio_path=audio_path,
        duration_seconds=2.0,
        logger=logger,
        stems_input={
            "vocals": str(tmp_path / "vocals.wav"),
            "bass": str(tmp_path / "bass.wav"),
            "drums": str(tmp_path / "drums.wav"),
            "other": str(tmp_path / "other.wav"),
            "identifier": "demo",
        },
        runtime_device="cuda",
    )

    assert call_devices == ["cuda"]
    assert output["big_segments"] == [{"segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"}]
    assert output["beat_times"] == [0.0, 1.0]
    assert output["beats"] == [
        {"time": 0.0, "type": "major", "source": "allin1"},
        {"time": 1.0, "type": "minor", "source": "allin1"},
    ]


def test_analyze_with_allin1_should_call_backend_without_device_when_signature_not_supported(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证后端签名不支持 device 参数时，调用可自动剔除该参数继续执行。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：使用仅支持 path 的简化桩函数验证兼容逻辑。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_allin1_backend_compat")

    def _fake_analyze_no_device(path: str):
        assert Path(path).name == "demo.wav"
        return {
            "segments": [{"start": 0.0, "end": 3.0, "label": "chorus"}],
            "beats": [0.0, 1.5],
            "beat_positions": [1, 2],
        }

    fake_backend = SimpleNamespace(analyze=_fake_analyze_no_device)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.backends.allin1._import_allin1_backend",
        lambda: ("allin1fix", fake_backend),
    )

    output = analyze_with_allin1(
        audio_path=audio_path,
        duration_seconds=3.0,
        logger=logger,
        runtime_device="cpu",
    )

    assert output["big_segments"] == [{"segment_id": "big_001", "start_time": 0.0, "end_time": 3.0, "label": "chorus"}]
    assert output["beat_times"] == [0.0, 1.5]


def test_analyze_with_allin1_should_force_model_fp32_on_cuda(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 CUDA + allin1fix 场景下会临时强制模型加载为 float32。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过测试模块内桩函数模拟 allin1 analyze 模块级加载器。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_allin1_backend_fp32")

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.float_called = False

        def float(self):  # noqa: D401
            self.float_called = True
            self.dummy_weight = torch.nn.Parameter(self.dummy_weight.detach().to(torch.float32))
            return self

    state_holder: dict[str, _FakeModel | None] = {"model": None}
    current_module = __import__(__name__, fromlist=["_placeholder"])

    def _fake_loader(*_args, **_kwargs):
        model = _FakeModel()
        state_holder["model"] = model
        return model

    monkeypatch.setattr(current_module, "load_pretrained_model", _fake_loader, raising=False)

    def _fake_analyze(*_args, **_kwargs):
        model = load_pretrained_model()
        assert isinstance(model, _FakeModel)
        return {
            "segments": [{"start": 0.0, "end": 2.0, "label": "verse"}],
            "beats": [0.0, 1.0],
            "beat_positions": [1, 2],
        }

    fake_backend = SimpleNamespace(analyze=_fake_analyze)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.backends.allin1._import_allin1_backend",
        lambda: ("allin1fix", fake_backend),
    )

    output = analyze_with_allin1(
        audio_path=audio_path,
        duration_seconds=2.0,
        logger=logger,
        runtime_device="cuda",
    )

    fake_model = state_holder["model"]
    assert fake_model is not None
    assert fake_model.float_called
    assert fake_model.dummy_weight.dtype == torch.float32
    assert output["big_segments"] == [{"segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"}]


def test_analyze_with_allin1_should_force_fp32_for_models_list_tree(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 CUDA + allin1fix 下会递归处理 models 列表内子模型，避免残留 bfloat16。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模拟 allin1fix Ensemble 使用普通 list 持有子模型的结构。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    logger = logging.getLogger("test_module_a_v2_allin1_backend_fp32_tree")

    class _FakeSubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_bias = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.float_called = False

        def float(self):  # noqa: D401
            self.float_called = True
            self.dummy_bias = torch.nn.Parameter(self.dummy_bias.detach().to(torch.float32))
            return self

    class _FakeEnsembleModel:
        def __init__(self):
            self.models = [_FakeSubModel(), _FakeSubModel()]
            self.float_called = False

        def float(self):  # noqa: D401
            self.float_called = True
            # 模拟 allin1fix Ensemble：根节点 float 不会自动递归到 list 子模型
            return self

    state_holder: dict[str, _FakeEnsembleModel | None] = {"model": None}
    current_module = __import__(__name__, fromlist=["_placeholder"])

    def _fake_loader(*_args, **_kwargs):
        model = _FakeEnsembleModel()
        state_holder["model"] = model
        return model

    monkeypatch.setattr(current_module, "load_pretrained_model", _fake_loader, raising=False)

    def _fake_analyze(*_args, **_kwargs):
        model = load_pretrained_model()
        assert isinstance(model, _FakeEnsembleModel)
        return {
            "segments": [{"start": 0.0, "end": 2.0, "label": "verse"}],
            "beats": [0.0, 1.0],
            "beat_positions": [1, 2],
        }

    fake_backend = SimpleNamespace(analyze=_fake_analyze)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.backends.allin1._import_allin1_backend",
        lambda: ("allin1fix", fake_backend),
    )

    output = analyze_with_allin1(
        audio_path=audio_path,
        duration_seconds=2.0,
        logger=logger,
        runtime_device="cuda",
    )

    fake_model = state_holder["model"]
    assert fake_model is not None
    assert fake_model.float_called
    assert all(child.float_called for child in fake_model.models)
    assert all(child.dummy_bias.dtype == torch.float32 for child in fake_model.models)
    assert output["big_segments"] == [{"segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"}]
