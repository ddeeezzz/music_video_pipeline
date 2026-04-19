"""
文件用途：提供跨模块共享的 diffusers/torch 惰性导入守卫。
核心流程：通过全局锁串行导入依赖，并按模块场景缓存已解析类型。
输入输出：输入无，输出模块 C/D 所需依赖对象字典。
依赖说明：依赖标准库 importlib/threading 与运行环境中的 torch、diffusers。
维护说明：若新增 diffusers 依赖场景，应在本文件集中扩展，避免重复并发导入。
"""

# 标准库：用于动态导入第三方模块。
import importlib
# 标准库：用于导入期全局互斥。
import threading
# 标准库：用于类型提示。
from typing import Any

# 全局锁：确保 diffusers 相关导入在进程内串行进行，避免并发导入竞态。
_DIFFUSERS_IMPORT_LOCK = threading.Lock()
# 全局缓存：按场景缓存解析后的依赖对象，避免重复导入与属性解析。
_RUNTIME_DEPS_CACHE: dict[str, dict[str, Any]] = {}


def _load_torch_and_diffusers_modules() -> tuple[Any, Any]:
    """
    功能说明：动态加载 torch 与 diffusers 模块对象。
    参数说明：无。
    返回值：
    - tuple[Any, Any]: (torch_module, diffusers_module)。
    异常说明：
    - ImportError: 任一模块不可用时抛出。
    边界条件：依赖 Python 模块缓存，重复调用成本可忽略。
    """
    torch_module = importlib.import_module("torch")
    diffusers_module = importlib.import_module("diffusers")
    return torch_module, diffusers_module


def load_module_c_diffusion_dependencies() -> dict[str, Any]:
    """
    功能说明：加载模块 C 扩散推理所需依赖并缓存。
    参数说明：无。
    返回值：
    - dict[str, Any]: torch 与 StableDiffusionPipeline/调度器类型映射。
    异常说明：
    - ImportError: 依赖不存在或版本不兼容时抛出。
    边界条件：首次成功后复用缓存，不再重复解析属性。
    """
    cache_key = "module_c_diffusion"
    with _DIFFUSERS_IMPORT_LOCK:
        cached = _RUNTIME_DEPS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        torch_module, diffusers_module = _load_torch_and_diffusers_modules()
        try:
            stable_diffusion_pipeline = getattr(diffusers_module, "StableDiffusionPipeline")
            euler_ancestral_discrete_scheduler = getattr(diffusers_module, "EulerAncestralDiscreteScheduler")
            ddim_scheduler = getattr(diffusers_module, "DDIMScheduler")
        except AttributeError as error:
            raise ImportError(f"diffusers 缺少模块C所需符号：{error}") from error

        runtime_dependencies = {
            "torch": torch_module,
            "StableDiffusionPipeline": stable_diffusion_pipeline,
            "EulerAncestralDiscreteScheduler": euler_ancestral_discrete_scheduler,
            "DDIMScheduler": ddim_scheduler,
        }
        _RUNTIME_DEPS_CACHE[cache_key] = runtime_dependencies
        return runtime_dependencies


def load_module_d_animatediff_dependencies() -> dict[str, Any]:
    """
    功能说明：加载模块 D AnimateDiff 推理所需依赖并缓存。
    参数说明：无。
    返回值：
    - dict[str, Any]: torch 与 AnimateDiff 相关 pipeline/adapter 类型映射。
    异常说明：
    - ImportError: 依赖不存在或版本不兼容时抛出。
    边界条件：首次成功后复用缓存，不再重复解析属性。
    """
    cache_key = "module_d_animatediff"
    with _DIFFUSERS_IMPORT_LOCK:
        cached = _RUNTIME_DEPS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        torch_module, diffusers_module = _load_torch_and_diffusers_modules()
        try:
            animatediff_pipeline = getattr(diffusers_module, "AnimateDiffPipeline")
            animatediff_sdxl_pipeline = getattr(diffusers_module, "AnimateDiffSDXLPipeline")
            motion_adapter = getattr(diffusers_module, "MotionAdapter")
        except AttributeError as error:
            raise ImportError(f"diffusers 缺少模块D所需符号：{error}") from error

        runtime_dependencies = {
            "torch": torch_module,
            "AnimateDiffPipeline": animatediff_pipeline,
            "AnimateDiffSDXLPipeline": animatediff_sdxl_pipeline,
            "MotionAdapter": motion_adapter,
        }
        _RUNTIME_DEPS_CACHE[cache_key] = runtime_dependencies
        return runtime_dependencies
