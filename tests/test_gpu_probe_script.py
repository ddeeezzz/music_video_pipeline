"""
文件用途：验证 scripts/gpu_probe.py 的解析与异常降级行为。
核心流程：直接调用脚本内部函数，打桩 subprocess，断言结构化输出。
输入输出：输入伪造 nvidia-smi 输出，输出 JSON 结构断言。
依赖说明：依赖 pytest 与标准库 importlib。
维护说明：脚本字段变更时需同步更新本测试。
"""

# 标准库：用于动态加载脚本模块。
import importlib.util
# 标准库：用于路径处理。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest


def _load_gpu_probe_module():
    """
    功能说明：按文件路径动态加载 gpu_probe.py。
    参数说明：无。
    返回值：脚本模块对象。
    异常说明：模块加载失败时抛异常。
    边界条件：固定从仓库 scripts 目录加载。
    """
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_probe.py"
    spec = importlib.util.spec_from_file_location("gpu_probe_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本模块：{script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_probe_parse_should_parse_standard_nvidia_smi_csv() -> None:
    """
    功能说明：验证脚本可正确解析标准 nvidia-smi CSV 输出。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：used_ratio 需按 used/total 计算。
    """
    module = _load_gpu_probe_module()
    rows = module._parse_nvidia_smi_csv("0, 15356, 4014\n1, 15356, 14930\n")

    assert len(rows) == 2
    assert rows[0]["index"] == 0
    assert rows[0]["total_mb"] == 15356
    assert rows[0]["used_mb"] == 4014
    assert rows[0]["used_ratio"] == pytest.approx(4014 / 15356, rel=1e-6)
    assert rows[1]["index"] == 1
    assert rows[1]["used_ratio"] == pytest.approx(14930 / 15356, rel=1e-6)


def test_gpu_probe_parse_should_raise_when_csv_columns_invalid() -> None:
    """
    功能说明：验证脚本在 CSV 列数非法时抛出 ValueError。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：错误信息应包含输出非法提示。
    """
    module = _load_gpu_probe_module()
    with pytest.raises(ValueError, match="输出列数非法"):
        module._parse_nvidia_smi_csv("0, 15356\n")


def test_gpu_probe_run_should_return_structured_error_when_exec_failed(monkeypatch) -> None:
    """
    功能说明：验证脚本在 subprocess 执行异常时返回结构化错误。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：失败时应返回 ok=False 且 gpus 为空数组。
    """
    module = _load_gpu_probe_module()

    def _raise_exec_error(*args, **kwargs):
        raise RuntimeError("mock exec failed")

    monkeypatch.setattr(module.subprocess, "run", _raise_exec_error)
    payload = module._run_probe(timeout_seconds=2.0)

    assert payload["ok"] is False
    assert "probe_exec_failed" in str(payload["error"])
    assert payload["gpus"] == []
