"""
文件用途：验证跨模块 diffusers 导入守卫的串行化与缓存行为。
核心流程：并发调用模块C/D依赖加载入口，断言导入不会并发重入且缓存生效。
输入输出：输入 monkeypatch 打桩依赖，输出断言结果。
依赖说明：依赖 pytest 与项目内 diffusers_runtime 模块。
维护说明：若导入守卫实现策略变更，需要同步更新本测试。
"""

# 标准库：用于构造并发测试场景。
import threading
# 标准库：用于模拟导入耗时。
import time
# 标准库：用于构造模块对象。
from types import ModuleType

# 项目内模块：跨模块导入守卫实现。
from music_video_pipeline import diffusers_runtime


def _build_fake_torch_module() -> ModuleType:
    """
    功能说明：构造最小可用的 fake torch 模块对象。
    参数说明：无。
    返回值：
    - ModuleType: 带 cuda.empty_cache 的模块桩。
    异常说明：无。
    边界条件：仅覆盖当前测试所需最小接口。
    """
    module = ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            return None

    module.cuda = _FakeCuda()  # type: ignore[attr-defined]
    return module


def _build_fake_diffusers_module() -> ModuleType:
    """
    功能说明：构造同时包含模块C/D所需符号的 fake diffusers 模块对象。
    参数说明：无。
    返回值：
    - ModuleType: fake diffusers 模块。
    异常说明：无。
    边界条件：符号值仅用于断言引用存在，不参与真实推理。
    """
    module = ModuleType("diffusers")
    module.StableDiffusionPipeline = object()  # type: ignore[attr-defined]
    module.EulerAncestralDiscreteScheduler = object()  # type: ignore[attr-defined]
    module.DDIMScheduler = object()  # type: ignore[attr-defined]
    module.AnimateDiffPipeline = object()  # type: ignore[attr-defined]
    module.AnimateDiffSDXLPipeline = object()  # type: ignore[attr-defined]
    module.MotionAdapter = object()  # type: ignore[attr-defined]
    return module


def test_diffusers_runtime_should_serialize_cross_module_imports(monkeypatch) -> None:
    """
    功能说明：验证模块C/D并发加载依赖时，导入阶段由全局锁串行化。
    参数说明：
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过耗时导入桩放大竞态窗口，确保测试可观测。
    """
    diffusers_runtime._RUNTIME_DEPS_CACHE.clear()
    fake_torch = _build_fake_torch_module()
    fake_diffusers = _build_fake_diffusers_module()

    state_lock = threading.Lock()
    in_flight_imports = 0
    max_in_flight_imports = 0

    def _fake_import_module(name: str):  # noqa: ANN001
        nonlocal in_flight_imports, max_in_flight_imports
        with state_lock:
            in_flight_imports += 1
            if in_flight_imports > max_in_flight_imports:
                max_in_flight_imports = in_flight_imports
        time.sleep(0.03)
        with state_lock:
            in_flight_imports -= 1
        if name == "torch":
            return fake_torch
        if name == "diffusers":
            return fake_diffusers
        raise ImportError(name)

    monkeypatch.setattr(diffusers_runtime.importlib, "import_module", _fake_import_module)

    errors: list[Exception] = []
    barrier = threading.Barrier(3)

    def _run_c() -> None:
        try:
            barrier.wait(timeout=1.0)
            diffusers_runtime.load_module_c_diffusion_dependencies()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    def _run_d() -> None:
        try:
            barrier.wait(timeout=1.0)
            diffusers_runtime.load_module_d_animatediff_dependencies()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread_c = threading.Thread(target=_run_c)
    thread_d = threading.Thread(target=_run_d)
    thread_c.start()
    thread_d.start()
    barrier.wait(timeout=1.0)
    thread_c.join(timeout=2.0)
    thread_d.join(timeout=2.0)

    assert not errors
    assert max_in_flight_imports == 1


def test_diffusers_runtime_should_cache_module_d_dependencies(monkeypatch) -> None:
    """
    功能说明：验证模块D依赖首次加载后会命中缓存，避免重复导入。
    参数说明：
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证模块D入口缓存行为。
    """
    diffusers_runtime._RUNTIME_DEPS_CACHE.clear()
    fake_torch = _build_fake_torch_module()
    fake_diffusers = _build_fake_diffusers_module()
    counters = {"torch": 0, "diffusers": 0}

    def _fake_import_module(name: str):  # noqa: ANN001
        if name == "torch":
            counters["torch"] += 1
            return fake_torch
        if name == "diffusers":
            counters["diffusers"] += 1
            return fake_diffusers
        raise ImportError(name)

    monkeypatch.setattr(diffusers_runtime.importlib, "import_module", _fake_import_module)

    first = diffusers_runtime.load_module_d_animatediff_dependencies()
    second = diffusers_runtime.load_module_d_animatediff_dependencies()

    assert first is second
    assert counters["torch"] == 1
    assert counters["diffusers"] == 1
