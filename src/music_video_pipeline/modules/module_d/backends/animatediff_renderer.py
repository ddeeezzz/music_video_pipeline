"""
文件用途：提供模块 D 的 AnimateDiff 渲染实现。
核心流程：解析底模与 LoRA 资产 -> 自动准备 Motion Adapter -> 生成视频帧 -> 编码为片段文件。
输入输出：输入 RuntimeContext/ModuleDUnit/prompt，输出单段渲染结果摘要。
依赖说明：依赖 Pillow、torch、diffusers、huggingface_hub 与模块D ffmpeg 执行工具。
维护说明：本文件只处理 AnimateDiff 渲染，不处理模块D状态机重试。
"""

# 标准库：用于稳定随机种子
import hashlib
# 标准库：用于JSON读取
import json
# 标准库：用于日志输出
import logging
# 标准库：用于环境变量控制
import os
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于目录清理
import shutil
# 标准库：用于临时目录
import tempfile
# 标准库：用于线程锁
import threading
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于保存图像帧
from PIL import Image

# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：跨模块 diffusers 导入守卫
from music_video_pipeline.diffusers_runtime import load_module_d_animatediff_dependencies
# 项目内模块：模块D ffmpeg 执行工具
from music_video_pipeline.modules.module_d.finalizer import _run_ffmpeg_command
# 项目内模块：模块D单元模型
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit

# 常量：底模注册表相对路径。
BASE_MODEL_REGISTRY_RELATIVE_PATH = Path("configs/base_model_registry.json")
# 常量：LoRA 绑定清单相对路径。
LORA_BINDINGS_RELATIVE_PATH = Path("configs/lora_bindings.json")
# 常量：默认 Motion Adapter HF 仓库（SD1.5）。
DEFAULT_MOTION_ADAPTER_REPO_ID_15 = "guoyww/animatediff-motion-adapter-v1-5-2"
# 常量：默认 Motion Adapter 本地缓存目录（SD1.5）。
DEFAULT_MOTION_ADAPTER_LOCAL_DIR_15 = "models/motion_adapter/15/diffusers/guoyww_animatediff_motion_adapter_v1_5_2"

# 全局缓存：避免每个 shot 重复加载 pipeline。
_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
# 全局锁：保护 runtime 缓存与模型初始化。
_RUNTIME_LOCK = threading.Lock()


def _resolve_project_root() -> Path:
    """
    功能说明：解析项目根目录（t1）。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：假设本文件位于 src/music_video_pipeline/modules/module_d/backends 下。
    """
    return Path(__file__).resolve().parents[5]


def _load_json_object(path: Path, source_name: str) -> dict[str, Any]:
    """
    功能说明：读取并校验 JSON 顶层对象。
    参数说明：
    - path: JSON 文件路径。
    - source_name: 来源名称（用于报错上下文）。
    返回值：
    - dict[str, Any]: JSON 顶层对象。
    异常说明：
    - RuntimeError: 文件不存在、读取失败或顶层结构非法时抛出。
    边界条件：使用 utf-8-sig 兼容 BOM 文件。
    """
    if not path.exists():
        raise RuntimeError(f"{source_name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file_obj:
            data = json.load(file_obj)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"{source_name} 读取失败：{path}，错误={error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"{source_name} 顶层结构必须是对象：{path}")
    return data


def _has_fp16_variant_weights(local_dir: Path) -> bool:
    """
    功能说明：检测 Motion Adapter 目录是否存在 fp16 safetensors 变体文件。
    参数说明：
    - local_dir: Motion Adapter 本地目录。
    返回值：
    - bool: 存在 fp16 变体返回 True。
    异常说明：无。
    边界条件：仅检测常见命名 diffusion_pytorch_model.fp16.safetensors。
    """
    return (local_dir / "diffusion_pytorch_model.fp16.safetensors").exists()


def _resolve_device(device_text: str, torch_module: Any) -> str:
    """
    功能说明：解析设备字符串并校验可用性。
    参数说明：
    - device_text: 配置中的设备字符串（auto/cpu/cuda/cuda:N）。
    - torch_module: torch 模块对象。
    返回值：
    - str: 可用于 .to(...) 的设备字符串。
    异常说明：
    - RuntimeError: 指定 cuda 但不可用或索引越界时抛出。
    边界条件：auto 在多卡时优先 cuda:1，单卡时回退 cuda:0。
    """
    normalized = str(device_text).strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    cuda_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    if normalized == "auto":
        if cuda_available and cuda_count >= 2:
            return "cuda:1"
        if cuda_available:
            return "cuda:0"
        return "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not cuda_available:
            raise RuntimeError("AnimateDiff 配置要求 cuda，但当前环境不可用。")
        return "cuda:0"
    if normalized.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(f"AnimateDiff 配置要求设备 {normalized}，但当前环境无可用 CUDA。")
        try:
            device_index = int(normalized.split(":", 1)[1])
        except ValueError as error:
            raise RuntimeError(f"AnimateDiff 设备配置非法：{normalized}") from error
        if device_index < 0 or device_index >= cuda_count:
            if cuda_count >= 1:
                logging.getLogger("D").warning(
                    "AnimateDiff 设备索引越界，已自动回退到 cuda:0，原始设备=%s，可用GPU数量=%s",
                    normalized,
                    cuda_count,
                )
                return "cuda:0"
            raise RuntimeError(f"AnimateDiff 设备索引越界：{normalized}，当前可用 GPU 数量={cuda_count}")
        return normalized
    raise RuntimeError(f"AnimateDiff 设备配置非法：{normalized}，支持 auto/cpu/cuda/cuda:N")


def _resolve_torch_dtype(torch_module: Any, torch_dtype_text: str, device: str) -> Any:
    """
    功能说明：根据配置解析 torch dtype。
    参数说明：
    - torch_module: torch 模块对象。
    - torch_dtype_text: 配置精度字符串。
    - device: 已解析设备字符串。
    返回值：
    - Any: torch dtype 对象。
    异常说明：
    - RuntimeError: 配置非法时抛出。
    边界条件：auto 在 cuda 下使用 float16，cpu 下使用 float32。
    """
    normalized = str(torch_dtype_text).strip().lower()
    if normalized == "auto":
        return torch_module.float16 if str(device).startswith("cuda") else torch_module.float32
    if normalized == "float16":
        return torch_module.float16
    if normalized == "float32":
        return torch_module.float32
    if normalized == "bfloat16":
        return torch_module.bfloat16
    raise RuntimeError(f"AnimateDiff torch_dtype 非法：{normalized}")


def _resolve_binding_and_base_assets(
    *,
    project_root: Path,
    binding_name: str,
    model_series: str,
) -> dict[str, Any]:
    """
    功能说明：解析 LoRA 绑定与底模资产。
    参数说明：
    - project_root: 项目根目录。
    - binding_name: LoRA 绑定名。
    - model_series: 模型系列（15/xl）。
    返回值：
    - dict[str, Any]: 含 base_model_path/lora_file_path/base_model_key 的字典。
    异常说明：
    - RuntimeError: 绑定或底模不存在、禁用、路径缺失时抛出。
    边界条件：底模由 lora_bindings.json 的 base_model_key 反查 registry。
    """
    bindings_data = _load_json_object(
        path=(project_root / LORA_BINDINGS_RELATIVE_PATH).resolve(),
        source_name="LoRA绑定清单",
    )
    bindings = bindings_data.get("bindings")
    if not isinstance(bindings, list):
        raise RuntimeError("LoRA绑定清单非法：bindings 必须是数组。")
    matched_binding: dict[str, Any] | None = None
    for item in bindings:
        if not isinstance(item, dict):
            continue
        if str(item.get("binding_name", "")).strip() != str(binding_name).strip():
            continue
        if str(item.get("model_series", "")).strip().lower() != str(model_series).strip().lower():
            continue
        matched_binding = item
        break
    if matched_binding is None:
        raise RuntimeError(f"AnimateDiff 绑定不存在：binding_name={binding_name}，model_series={model_series}")

    base_model_key = str(matched_binding.get("base_model_key", "")).strip()
    if not base_model_key:
        raise RuntimeError(f"AnimateDiff 绑定缺失 base_model_key：binding_name={binding_name}")
    lora_file_text = str(matched_binding.get("lora_file", "")).strip()
    if not lora_file_text:
        raise RuntimeError(f"AnimateDiff 绑定缺失 lora_file：binding_name={binding_name}")
    lora_file_path = (project_root / lora_file_text).resolve()
    if not lora_file_path.exists():
        raise RuntimeError(f"AnimateDiff LoRA 文件不存在：{lora_file_path}")

    registry_data = _load_json_object(
        path=(project_root / BASE_MODEL_REGISTRY_RELATIVE_PATH).resolve(),
        source_name="底模注册表",
    )
    base_models = registry_data.get("base_models")
    if not isinstance(base_models, list):
        raise RuntimeError("底模注册表非法：base_models 必须是数组。")
    matched_base_model: dict[str, Any] | None = None
    for item in base_models:
        if not isinstance(item, dict):
            continue
        if str(item.get("key", "")).strip() == base_model_key:
            matched_base_model = item
            break
    if matched_base_model is None:
        raise RuntimeError(f"AnimateDiff 底模 key 不存在：{base_model_key}")
    if not bool(matched_base_model.get("enabled", False)):
        raise RuntimeError(f"AnimateDiff 底模已禁用：{base_model_key}")
    record_series = str(matched_base_model.get("series", "")).strip().lower()
    if record_series != str(model_series).strip().lower():
        raise RuntimeError(
            f"AnimateDiff 底模系列不匹配：base_model_key={base_model_key}，series={record_series}，expected={model_series}"
        )
    base_model_path_text = str(matched_base_model.get("path", "")).strip()
    if not base_model_path_text:
        raise RuntimeError(f"AnimateDiff 底模记录缺失 path：{base_model_key}")
    base_model_path = (project_root / base_model_path_text).resolve()
    if not base_model_path.exists():
        raise RuntimeError(f"AnimateDiff 底模路径不存在：{base_model_path}")

    return {
        "base_model_key": base_model_key,
        "base_model_path": base_model_path,
        "lora_file_path": lora_file_path,
    }


def _ensure_motion_adapter_dir(
    *,
    project_root: Path,
    repo_id: str,
    revision: str,
    local_dir_text: str,
    hf_endpoint: str,
) -> Path:
    """
    功能说明：确保 Motion Adapter 已下载到本地目录。
    参数说明：
    - project_root: 项目根目录。
    - repo_id: HF 仓库标识。
    - revision: 仓库 revision。
    - local_dir_text: 本地缓存目录（相对项目根）。
    - hf_endpoint: HF 镜像地址（可空）。
    返回值：
    - Path: Motion Adapter 本地目录。
    异常说明：
    - RuntimeError: 下载失败时抛出。
    边界条件：目录存在且非空时视为缓存命中，不重复下载。
    """
    local_dir = (project_root / local_dir_text).resolve()
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"AnimateDiff 依赖导入失败：huggingface_hub.snapshot_download 不可用，错误={error}") from error

    previous_hf_endpoint = os.environ.get("HF_ENDPOINT")
    should_restore_env = False
    if str(hf_endpoint).strip():
        if not previous_hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint).strip()
            should_restore_env = True
    try:
        snapshot_download(
            repo_id=str(repo_id).strip(),
            revision=str(revision).strip() or "main",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"AnimateDiff Motion Adapter 下载失败：repo_id={repo_id}，revision={revision}，local_dir={local_dir}，错误={error}"
        ) from error
    finally:
        if should_restore_env:
            os.environ.pop("HF_ENDPOINT", None)
    return local_dir


def _build_runtime_cache_key(
    *,
    base_model_path: Path,
    lora_file_path: Path,
    motion_adapter_dir: Path,
    model_series: str,
    device: str,
    torch_dtype_text: str,
) -> str:
    """
    功能说明：构建 runtime 缓存 key。
    参数说明：
    - base_model_path/lora_file_path/motion_adapter_dir/model_series/device/torch_dtype_text: 运行关键参数。
    返回值：
    - str: 稳定缓存 key。
    异常说明：无。
    边界条件：同参数组合复用同一 pipeline。
    """
    key_source = "|".join(
        [
            str(base_model_path),
            str(lora_file_path),
            str(motion_adapter_dir),
            str(model_series).strip().lower(),
            str(device).strip().lower(),
            str(torch_dtype_text).strip().lower(),
        ]
    )
    return hashlib.sha256(key_source.encode("utf-8")).hexdigest()


def _ensure_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, Any]:
    """
    功能说明：初始化并缓存 AnimateDiff runtime。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 可选，强制绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - dict[str, Any]: runtime 字典（pipeline/torch/device/dtype/assets）。
    异常说明：
    - RuntimeError: 资产解析、依赖导入、模型加载失败时抛出。
    边界条件：首次加载后复用，避免每个 shot 重新加载模型。
    """
    anim_cfg = context.config.module_d.animatediff
    project_root = _resolve_project_root()
    assets = _resolve_binding_and_base_assets(
        project_root=project_root,
        binding_name=anim_cfg.binding_name,
        model_series=anim_cfg.model_series,
    )
    repo_id = str(anim_cfg.motion_adapter_repo_id).strip() or DEFAULT_MOTION_ADAPTER_REPO_ID_15
    local_dir_text = str(anim_cfg.motion_adapter_local_dir).strip() or DEFAULT_MOTION_ADAPTER_LOCAL_DIR_15
    motion_adapter_dir = _ensure_motion_adapter_dir(
        project_root=project_root,
        repo_id=repo_id,
        revision=anim_cfg.motion_adapter_revision,
        local_dir_text=local_dir_text,
        hf_endpoint=anim_cfg.hf_endpoint,
    )

    try:
        runtime_dependencies = load_module_d_animatediff_dependencies()
        torch = runtime_dependencies["torch"]
        AnimateDiffPipeline = runtime_dependencies["AnimateDiffPipeline"]
        AnimateDiffSDXLPipeline = runtime_dependencies["AnimateDiffSDXLPipeline"]
        MotionAdapter = runtime_dependencies["MotionAdapter"]
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"AnimateDiff 依赖导入失败，请检查 diffusers/torch 环境：{error}") from error

    device_text = str(device_override).strip() if device_override is not None else str(anim_cfg.device).strip()
    device = _resolve_device(device_text=device_text, torch_module=torch)
    resolved_dtype = _resolve_torch_dtype(torch_module=torch, torch_dtype_text=anim_cfg.torch_dtype, device=device)
    cache_key = _build_runtime_cache_key(
        base_model_path=assets["base_model_path"],
        lora_file_path=assets["lora_file_path"],
        motion_adapter_dir=motion_adapter_dir,
        model_series=anim_cfg.model_series,
        device=device,
        torch_dtype_text=anim_cfg.torch_dtype,
    )
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached

        motion_adapter_load_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "local_files_only": True,
            "use_safetensors": True,
        }
        has_fp16_variant = _has_fp16_variant_weights(local_dir=motion_adapter_dir)
        if has_fp16_variant:
            motion_adapter_load_kwargs["variant"] = "fp16"

        try:
            motion_adapter = MotionAdapter.from_pretrained(
                str(motion_adapter_dir),
                **motion_adapter_load_kwargs,
            )
        except Exception as first_error:  # noqa: BLE001
            if has_fp16_variant:
                retry_kwargs = dict(motion_adapter_load_kwargs)
                retry_kwargs.pop("variant", None)
                try:
                    motion_adapter = MotionAdapter.from_pretrained(
                        str(motion_adapter_dir),
                        **retry_kwargs,
                    )
                except Exception as second_error:  # noqa: BLE001
                    raise RuntimeError(
                        "AnimateDiff Motion Adapter 加载失败："
                        f"{motion_adapter_dir}，首次参数={motion_adapter_load_kwargs}，"
                        f"重试参数={retry_kwargs}，首次错误={first_error}，重试错误={second_error}"
                    ) from second_error
            else:
                raise RuntimeError(
                    "AnimateDiff Motion Adapter 加载失败："
                    f"{motion_adapter_dir}，加载参数={motion_adapter_load_kwargs}，错误={first_error}"
                ) from first_error

        try:
            if str(anim_cfg.model_series).strip().lower() == "xl":
                pipeline = AnimateDiffSDXLPipeline.from_pretrained(
                    str(assets["base_model_path"]),
                    motion_adapter=motion_adapter,
                    torch_dtype=resolved_dtype,
                    local_files_only=True,
                )
            else:
                pipeline = AnimateDiffPipeline.from_pretrained(
                    str(assets["base_model_path"]),
                    motion_adapter=motion_adapter,
                    torch_dtype=resolved_dtype,
                    local_files_only=True,
                )
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"AnimateDiff pipeline 初始化失败：{assets['base_model_path']}，错误={error}") from error

        try:
            pipeline = pipeline.to(device)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"AnimateDiff 切换设备失败：device={device}，错误={error}") from error

        try:
            pipeline.load_lora_weights(
                str(assets["lora_file_path"].parent),
                weight_name=assets["lora_file_path"].name,
            )
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"AnimateDiff 加载 LoRA 失败：{assets['lora_file_path']}，错误={error}") from error

        runtime = {
            "cache_key": cache_key,
            "pipeline": pipeline,
            "torch": torch,
            "device": device,
            "dtype": resolved_dtype,
            "assets": assets,
            "motion_adapter_repo_id": repo_id,
            "motion_adapter_dir": motion_adapter_dir,
        }
        _RUNTIME_CACHE[cache_key] = runtime
        context.logger.info(
            "AnimateDiff runtime 初始化完成，series=%s，binding=%s，base_model_key=%s，device=%s，motion_adapter=%s",
            anim_cfg.model_series,
            anim_cfg.binding_name,
            assets["base_model_key"],
            device,
            repo_id,
        )
        return runtime


def prewarm_animatediff_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, str]:
    """
    功能说明：预热指定设备上的 AnimateDiff runtime（仅加载并缓存，不执行推理）。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 可选，强制绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - dict[str, str]: 预热摘要（device/cache_key）。
    异常说明：
    - RuntimeError: runtime 初始化失败时抛出。
    边界条件：命中缓存时不会重复加载模型。
    """
    runtime = _ensure_runtime(context=context, device_override=device_override)
    return {
        "device": str(runtime.get("device", str(device_override or ""))),
        "cache_key": str(runtime.get("cache_key", "")),
    }


def _resolve_seed_value(shot_id: str, shot_index: int) -> int:
    """
    功能说明：根据 shot 信息生成稳定随机种子。
    参数说明：
    - shot_id: 镜头ID。
    - shot_index: 镜头索引（0基）。
    返回值：
    - int: 32位种子值。
    异常说明：无。
    边界条件：同输入组合应稳定复现。
    """
    digest = hashlib.sha256(f"{shot_id}|{shot_index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _extract_frames_from_pipeline_output(output_obj: Any) -> list[Image.Image]:
    """
    功能说明：从 diffusers pipeline 输出中提取帧图像数组。
    参数说明：
    - output_obj: pipeline 推理返回对象。
    返回值：
    - list[Image.Image]: PIL 图像帧数组。
    异常说明：
    - RuntimeError: 输出结构无法识别时抛出。
    边界条件：兼容 frames 为 list[list[PIL]] 或 list[PIL] 两种常见结构。
    """
    frames_obj = getattr(output_obj, "frames", None)
    if isinstance(frames_obj, list) and frames_obj:
        candidate = frames_obj[0] if isinstance(frames_obj[0], list) else frames_obj
    else:
        raise RuntimeError("AnimateDiff 输出中未找到有效 frames。")
    if not isinstance(candidate, list) or not candidate:
        raise RuntimeError("AnimateDiff 输出 frames 为空。")
    normalized: list[Image.Image] = []
    for frame in candidate:
        if isinstance(frame, Image.Image):
            normalized.append(frame)
            continue
        try:
            normalized.append(Image.fromarray(frame))
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"AnimateDiff 帧转换失败：{error}") from error
    return normalized


def generate_mv_clip(
    prompt: str,
    num_frames: int = 16,
    *,
    runtime: dict[str, Any],
    width: int,
    height: int,
    negative_prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int | None,
) -> list[Image.Image]:
    """
    功能说明：调用 AnimateDiff 生成短视频帧序列。
    参数说明：
    - prompt: 正向提示词。
    - num_frames: 目标帧数。
    - runtime: runtime 字典（包含 pipeline/torch/device）。
    - width/height: 输出分辨率。
    - negative_prompt: 负向提示词。
    - guidance_scale: CFG 引导系数。
    - steps: 推理步数。
    - seed: 可选随机种子。
    返回值：
    - list[Image.Image]: 生成帧序列。
    异常说明：
    - RuntimeError: 推理失败或帧提取失败时抛出。
    边界条件：num_frames 小于 1 时自动归一化为 1。
    """
    normalized_prompt = str(prompt).strip()
    if not normalized_prompt:
        raise RuntimeError("AnimateDiff 生成失败：prompt 不能为空。")
    normalized_frames = max(1, int(num_frames))
    pipeline = runtime["pipeline"]
    torch_module = runtime["torch"]
    device = str(runtime["device"])
    call_kwargs: dict[str, Any] = {
        "prompt": normalized_prompt,
        "negative_prompt": str(negative_prompt),
        "num_frames": normalized_frames,
        "width": int(width),
        "height": int(height),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
    }
    if seed is not None:
        generator = torch_module.Generator(device=device)
        generator.manual_seed(int(seed))
        call_kwargs["generator"] = generator
    try:
        # 推理阶段禁用梯度图，避免长链路运行时显存逐步抬升。
        with torch_module.inference_mode():
            output_obj = pipeline(**call_kwargs)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"AnimateDiff 推理失败：{error}") from error
    finally:
        # 主动触发缓存回收，降低长任务中的显存碎片化与峰值累积风险。
        if str(device).startswith("cuda"):
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass
    frames = _extract_frames_from_pipeline_output(output_obj=output_obj)
    if len(frames) >= normalized_frames:
        return frames[:normalized_frames]
    # 对不足帧数场景做末帧补齐，避免时间轴错位。
    padded = list(frames)
    while len(padded) < normalized_frames:
        padded.append(padded[-1].copy())
    return padded


def _build_frames_encode_command(
    *,
    ffmpeg_bin: str,
    frames_pattern: str,
    exact_frames: int,
    fps: int,
    encoder_command_args: list[str],
    output_path: str,
) -> list[str]:
    """
    功能说明：构建“图像序列 -> 视频片段”的 ffmpeg 命令。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - frames_pattern: 帧文件通配模式（如 frame_%05d.png）。
    - exact_frames: 目标帧数。
    - fps: 输出帧率。
    - encoder_command_args: 编码参数数组。
    - output_path: 输出片段路径。
    返回值：
    - list[str]: ffmpeg 命令数组。
    异常说明：无。
    边界条件：固定输出 yuv420p，无音频。
    """
    return [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_pattern),
        "-frames:v",
        str(int(exact_frames)),
        "-r",
        str(int(fps)),
        *list(encoder_command_args),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ] 


def _resolve_effective_density_fps(unit: ModuleDUnit) -> tuple[int, str, str]:
    """
    功能说明：根据分段标签解析 AnimateDiff 有效帧密度（fps）。
    参数说明：
    - unit: 模块 D 单元对象（需包含 shot 标签字段）。
    返回值：
    - tuple[int, str, str]: (target_effective_fps, label_source, label_value)。
    异常说明：无。
    边界条件：仅识别 solo/chorus（大小写不敏感），未命中默认 8fps。
    """
    shot_payload = getattr(unit, "shot", {})
    if not isinstance(shot_payload, dict):
        shot_payload = {}
    high_density_labels = {"solo", "chorus"}
    big_segment_label = str(shot_payload.get("big_segment_label", "")).strip().lower()
    segment_label = str(shot_payload.get("segment_label", "")).strip().lower()

    if big_segment_label:
        if big_segment_label in high_density_labels:
            return 16, "big_segment_label", big_segment_label
        return 8, "big_segment_label", big_segment_label
    if segment_label:
        if segment_label in high_density_labels:
            return 16, "segment_label", segment_label
        return 8, "segment_label", segment_label
    return 8, "default", ""


def _resolve_target_effective_frames(unit: ModuleDUnit, output_fps: int) -> tuple[int, int, str, str]:
    """
    功能说明：计算单元目标有效帧数（不使用 num_frames 上限）。
    参数说明：
    - unit: 模块 D 单元对象。
    - output_fps: 全局输出帧率（用于兜底时长计算）。
    返回值：
    - tuple[int, int, str, str]: (target_effective_frames, target_effective_fps, label_source, label_value)。
    异常说明：无。
    边界条件：返回值始终在 [1, exact_frames]。
    """
    exact_frames = max(1, int(getattr(unit, "exact_frames", 1)))
    target_effective_fps, label_source, label_value = _resolve_effective_density_fps(unit=unit)
    unit_duration_seconds = float(getattr(unit, "duration", 0.0))
    if unit_duration_seconds <= 0.0:
        safe_output_fps = max(1, int(output_fps))
        unit_duration_seconds = float(exact_frames) / float(safe_output_fps)
    target_effective_frames = max(
        1,
        min(
            exact_frames,
            int(round(float(unit_duration_seconds) * float(target_effective_fps))),
        ),
    )
    return target_effective_frames, target_effective_fps, label_source, label_value


def _resample_frames_uniform(frames: list[Image.Image], target_frames: int) -> list[Image.Image]:
    """
    功能说明：将帧序列均匀重采样到指定帧数。
    参数说明：
    - frames: 原始帧序列。
    - target_frames: 目标帧数。
    返回值：
    - list[Image.Image]: 重采样后帧序列。
    异常说明：
    - RuntimeError: 输入帧为空时抛出。
    边界条件：target_frames 小于 1 时归一化为 1。
    """
    if not frames:
        raise RuntimeError("AnimateDiff 重采样失败：输入帧为空。")
    normalized_target_frames = max(1, int(target_frames))
    source_frames = list(frames)
    source_count = len(source_frames)
    if source_count == normalized_target_frames:
        return [frame.copy() for frame in source_frames]
    if source_count == 1:
        return [source_frames[0].copy() for _ in range(normalized_target_frames)]
    if normalized_target_frames == 1:
        return [source_frames[0].copy()]

    resampled_frames: list[Image.Image] = []
    for output_index in range(normalized_target_frames):
        mapped_position = float(output_index) * float(source_count - 1) / float(normalized_target_frames - 1)
        mapped_index = int(round(mapped_position))
        mapped_index = max(0, min(source_count - 1, mapped_index))
        resampled_frames.append(source_frames[mapped_index].copy())
    return resampled_frames


def run_one_unit_animatediff_denoise_stage(
    *,
    context: RuntimeContext,
    unit: ModuleDUnit,
    prompt: str,
    device_override: str | None = None,
) -> dict[str, Any]:
    """
    功能说明：执行单个模块 D 单元的去噪阶段（仅生成并重采样帧）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元。
    - prompt: 视频提示词。
    - device_override: 可选，强制绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - dict[str, Any]: 去噪阶段摘要（含 frames/帧密度信息/运行资产信息）。
    异常说明：
    - RuntimeError: runtime 初始化或推理失败时抛出。
    边界条件：本函数不写文件，不执行 ffmpeg 编码。
    """
    runtime = _ensure_runtime(context=context, device_override=device_override)
    anim_cfg = context.config.module_d.animatediff
    shot_id = str(unit.unit_id)
    shot_index = int(unit.unit_index)
    seed: int | None = None
    if str(anim_cfg.seed_mode).strip().lower() == "shot_index":
        seed = _resolve_seed_value(shot_id=shot_id, shot_index=shot_index)
    exact_frames = max(1, int(unit.exact_frames))
    target_effective_frames, target_effective_fps, label_source, label_value = _resolve_target_effective_frames(
        unit=unit,
        output_fps=int(context.config.ffmpeg.fps),
    )
    inference_frames = int(target_effective_frames)
    context.logger.info(
        "AnimateDiff 帧密度策略生效，shot_id=%s，label_source=%s，label_value=%s，target_effective_fps=%s，"
        "target_effective_frames=%s，inference_frames=%s，exact_frames=%s",
        shot_id,
        label_source,
        label_value,
        target_effective_fps,
        target_effective_frames,
        inference_frames,
        exact_frames,
    )
    frames = generate_mv_clip(
        prompt=prompt,
        num_frames=inference_frames,
        runtime=runtime,
        width=int(context.config.render.video_width),
        height=int(context.config.render.video_height),
        negative_prompt=str(anim_cfg.negative_prompt),
        guidance_scale=float(anim_cfg.guidance_scale),
        steps=int(anim_cfg.steps),
        seed=seed,
    )
    frames = _resample_frames_uniform(frames=frames, target_frames=exact_frames)
    return {
        "shot_id": shot_id,
        "shot_index": int(shot_index),
        "frames": frames,
        "target_effective_fps": int(target_effective_fps),
        "target_effective_frames": int(target_effective_frames),
        "inference_frames": int(inference_frames),
        "exact_frames": int(exact_frames),
        "density_label_source": str(label_source),
        "density_label_value": str(label_value),
        "binding_name": str(anim_cfg.binding_name),
        "base_model_key": str(runtime["assets"]["base_model_key"]),
        "lora_file": str(runtime["assets"]["lora_file_path"]),
        "motion_adapter_repo_id": str(runtime["motion_adapter_repo_id"]),
        "motion_adapter_dir": str(runtime["motion_adapter_dir"]),
        "runtime_device": str(runtime.get("device", str(device_override or ""))),
    }


def _write_frames_to_temp_dir(
    *,
    task_id: str,
    shot_id: str,
    frames: list[Image.Image],
) -> Path:
    """
    功能说明：将帧数组写入临时目录，作为编码阶段输入。
    参数说明：
    - task_id: 任务ID（用于目录前缀）。
    - shot_id: 镜头ID（用于目录前缀）。
    - frames: 待写入帧数组。
    返回值：
    - Path: 临时帧目录路径。
    异常说明：
    - RuntimeError: 帧写入失败时抛出。
    边界条件：目录由调用方负责清理。
    """
    temp_frames_dir = Path(tempfile.mkdtemp(prefix=f"ad_frames_{task_id}_{shot_id}_"))
    try:
        for frame_index, frame_obj in enumerate(frames):
            frame_path = temp_frames_dir / f"frame_{frame_index + 1:05d}.png"
            frame_obj.save(frame_path)
    except Exception as error:  # noqa: BLE001
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        raise RuntimeError(f"AnimateDiff 帧落盘失败：shot_id={shot_id}，错误={error}") from error
    return temp_frames_dir


def run_one_unit_animatediff_post_stage(
    *,
    context: RuntimeContext,
    unit: ModuleDUnit,
    denoise_summary: dict[str, Any],
    encoder_command_args: list[str],
    profile_name: str = "animatediff",
) -> dict[str, Any]:
    """
    功能说明：执行单个模块 D 单元的后处理阶段（帧落盘 + ffmpeg 编码）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元。
    - denoise_summary: 去噪阶段输出摘要。
    - encoder_command_args: ffmpeg 编码参数。
    - profile_name: 渲染 profile 名称。
    返回值：
    - dict[str, Any]: 渲染摘要（segment_path + 去噪阶段关键元信息）。
    异常说明：
    - RuntimeError: 帧落盘、编码或原子替换失败时抛出。
    边界条件：编码成功后始终返回最终片段路径。
    """
    shot_id = str(denoise_summary.get("shot_id", str(unit.unit_id)))
    shot_index = int(denoise_summary.get("shot_index", int(unit.unit_index)))
    exact_frames = int(denoise_summary.get("exact_frames", int(unit.exact_frames)))
    frames_obj = denoise_summary.get("frames")
    if not isinstance(frames_obj, list) or not frames_obj:
        raise RuntimeError(f"AnimateDiff 后处理失败：去噪阶段帧数据为空，shot_id={shot_id}")
    frames = frames_obj

    temp_path = Path(str(unit.temp_segment_path))
    final_path = Path(str(unit.segment_path))
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass

    temp_frames_dir = _write_frames_to_temp_dir(task_id=context.task_id, shot_id=shot_id, frames=frames)
    try:
        command = _build_frames_encode_command(
            ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
            frames_pattern=str(temp_frames_dir / "frame_%05d.png"),
            exact_frames=exact_frames,
            fps=context.config.ffmpeg.fps,
            encoder_command_args=list(encoder_command_args),
            output_path=str(temp_path),
        )
        try:
            _run_ffmpeg_command(
                command=command,
                command_name=f"渲染小片段 segment_{shot_index + 1:03d}（{profile_name}）",
            )
            temp_path.replace(final_path)
        except Exception:  # noqa: BLE001
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise
    finally:
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

    return {
        "segment_index": int(shot_index + 1),
        "segment_path": str(final_path),
        "profile_name": str(profile_name),
        "target_effective_fps": int(denoise_summary.get("target_effective_fps", 0)),
        "target_effective_frames": int(denoise_summary.get("target_effective_frames", 0)),
        "inference_frames": int(denoise_summary.get("inference_frames", 0)),
        "exact_frames": int(exact_frames),
        "density_label_source": str(denoise_summary.get("density_label_source", "")),
        "density_label_value": str(denoise_summary.get("density_label_value", "")),
        "binding_name": str(denoise_summary.get("binding_name", "")),
        "base_model_key": str(denoise_summary.get("base_model_key", "")),
        "lora_file": str(denoise_summary.get("lora_file", "")),
        "motion_adapter_repo_id": str(denoise_summary.get("motion_adapter_repo_id", "")),
        "motion_adapter_dir": str(denoise_summary.get("motion_adapter_dir", "")),
        "runtime_device": str(denoise_summary.get("runtime_device", "")),
    }


def render_one_unit_animatediff(
    *,
    context: RuntimeContext,
    unit: ModuleDUnit,
    prompt: str,
    encoder_command_args: list[str],
    profile_name: str = "animatediff",
    device_override: str | None = None,
) -> dict[str, Any]:
    """
    功能说明：渲染单个模块 D 单元（AnimateDiff 路径）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元。
    - prompt: 视频提示词。
    - encoder_command_args: ffmpeg 编码参数。
    - profile_name: 渲染 profile 名称。
    - device_override: 可选，强制绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - dict[str, Any]: 渲染摘要（segment_index/segment_path/profile_name）。
    异常说明：
    - RuntimeError: runtime 初始化、推理、帧落盘或编码失败时抛出。
    边界条件：输出文件采用 temp -> replace 原子提交。
    """
    denoise_summary = run_one_unit_animatediff_denoise_stage(
        context=context,
        unit=unit,
        prompt=prompt,
        device_override=device_override,
    )
    return run_one_unit_animatediff_post_stage(
        context=context,
        unit=unit,
        denoise_summary=denoise_summary,
        encoder_command_args=encoder_command_args,
        profile_name=profile_name,
    )
