"""
文件用途：实现模块 D 的 ToonCrafter ComfyUI 视频工作流执行。
核心流程：读取双关键帧与视频提示词 -> 缩放到 ToonCrafter 固定分辨率 -> 渲染 16 帧原生序列 -> 重采样到目标帧数 -> 编码为 mp4。
输入输出：输入 RuntimeContext/ModuleDUnit，输出单元渲染摘要字典。
依赖说明：依赖项目内 ComfyUI 客户端、工作流契约工具、PIL 图像处理与模块 D FFmpeg 工具。
维护说明：本文件固定承载 ToonCrafter v1 路径，不再扩展其他视频后端。
"""

# 标准库：用于稳定随机种子。
import hashlib
# 标准库：用于文件复制与目录操作。
import shutil
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any

# 第三方库：用于图像缩放与格式统一。
from PIL import Image

# 项目内模块：ComfyUI API 客户端。
from music_video_pipeline.comfyui import (
    ComfyUIClient,
    ComfyUIServiceOptions,
    load_workflow_contract,
    render_workflow_from_contract,
)
# 项目内模块：运行上下文。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：模块 D FFmpeg 工具。
from music_video_pipeline.modules.module_d.finalizer import _run_ffmpeg_command
# 项目内模块：模块 D 单元模型。
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit


# 常量：ToonCrafter 内部 clip 模型名称，交给 wrapper 首次自动下载。
TOONCRAFTER_CLIP_MODEL_NAME = "stable-diffusion-2-1-clip-fp16.safetensors"
# 常量：ToonCrafter 内部 clip vision 模型名称，交给 wrapper 首次自动下载。
TOONCRAFTER_CLIP_VISION_MODEL_NAME = "CLIP-ViT-H-fp16.safetensors"


def prewarm_comfyui_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, str]:
    """
    功能说明：预热模块 D 的 ToonCrafter ComfyUI 后端（服务探活 + 契约加载 + 关键模型存在性校验）。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 保留参数；当前 ComfyUI HTTP 路径不直接消费该值。
    返回值：
    - dict[str, str]: 预热摘要。
    异常说明：
    - RuntimeError: 服务不可用、契约读取失败或关键模型缺失时抛出。
    边界条件：本预热不触发真实推理，只校验工作流执行前置条件。
    """
    client = _build_comfyui_client(context=context)
    client.ensure_service_ready()
    contract = load_workflow_contract(context.config.module_d.comfyui.contract_file)
    checkpoint_path = _resolve_tooncrafter_checkpoint_path(context=context)
    if not checkpoint_path.exists():
        raise RuntimeError(f"模块D ToonCrafter 预热失败：主模型不存在，path={checkpoint_path}")
    sketch_encoder_path = _resolve_sketch_encoder_path(context=context)
    if not sketch_encoder_path.exists():
        raise RuntimeError(f"模块D ToonCrafter 预热失败：sketch encoder 不存在，path={sketch_encoder_path}")
    return {
        "backend": "comfyui-tooncrafter",
        "device": str(device_override or "comfyui-http"),
        "contract": str(contract.workflow_api_file),
        "checkpoint": str(checkpoint_path),
    }



def render_one_unit_comfyui(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, Any]:
    """
    功能说明：执行单个模块 D 单元的 ToonCrafter ComfyUI 视频工作流。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - dict[str, Any]: 渲染摘要（segment_path/target_effective_fps/frame_count_used/native_frame_count）。
    异常说明：
    - RuntimeError: 契约不完整、服务失败、提示词缺失、关键帧缺失或编码失败时抛出。
    边界条件：模块 D 已彻底收口为 ToonCrafter；camera_motion 仅作为元数据保留，不参与渲染控制。
    """
    contract = load_workflow_contract(context.config.module_d.comfyui.contract_file)
    client = _build_comfyui_client(context=context)
    client.ensure_service_ready()

    comfy_cfg = context.config.module_d.comfyui
    prompt_text = str(unit.shot.get("video_prompt_en", "")).strip()
    if not prompt_text:
        raise RuntimeError(f"模块D ToonCrafter 渲染失败：缺失 video_prompt_en，unit_id={unit.unit_id}")
    start_image = str(unit.shot.get("frame_path_start", "")).strip()
    end_image = str(unit.shot.get("frame_path_end", "")).strip()
    if (not start_image) or (not end_image):
        raise RuntimeError(
            "模块D ToonCrafter 渲染失败：缺失双关键帧字段，"
            f"unit_id={unit.unit_id}，frame_path_start={start_image}，frame_path_end={end_image}"
        )

    sequence_dir = unit.segment_path.parent / f".{unit.unit_id}_frames"
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    resized_dir = unit.segment_path.parent / f".{unit.unit_id}_tooncrafter_inputs"
    if resized_dir.exists():
        shutil.rmtree(resized_dir)
    resized_dir.mkdir(parents=True, exist_ok=True)

    resized_start_path = resized_dir / "start_512x320.png"
    resized_end_path = resized_dir / "end_512x320.png"
    _resize_image_for_tooncrafter(
        source_path=Path(start_image),
        target_path=resized_start_path,
        width=int(comfy_cfg.generation_width),
        height=int(comfy_cfg.generation_height),
    )
    _resize_image_for_tooncrafter(
        source_path=Path(end_image),
        target_path=resized_end_path,
        width=int(comfy_cfg.generation_width),
        height=int(comfy_cfg.generation_height),
    )

    workflow_prompt = render_workflow_from_contract(
        contract=contract,
        binding_values={
            "checkpoint_name": str(comfy_cfg.checkpoint_name),
            "start_image": client.stage_input_image(resized_start_path, prefix=f"{unit.unit_id}_start"),
            "end_image": client.stage_input_image(resized_end_path, prefix=f"{unit.unit_id}_end"),
            "positive_prompt": prompt_text if bool(comfy_cfg.use_video_prompt_as_positive) else "",
            "negative_prompt": str(comfy_cfg.negative_prompt),
            "steps": int(comfy_cfg.steps),
            "cfg": float(comfy_cfg.cfg),
            "eta": float(comfy_cfg.eta),
            "frames": int(comfy_cfg.generation_frames),
            "seed": _resolve_seed_value(unit=unit),
            "fs": int(comfy_cfg.generation_fps),
            "vae_dtype": str(comfy_cfg.vae_dtype),
            "image_embed_ratio": float(comfy_cfg.image_embed_ratio),
            "augmentation_level": float(comfy_cfg.augmentation_level),
            "filename_prefix": f"mvpl/module_d/{unit.unit_id}/tooncrafter",
        },
    )
    native_output_files = client.execute_prompt(
        workflow_prompt=workflow_prompt,
        output_node_id=contract.output_node_id,
    )
    expected_native_frames = int(comfy_cfg.generation_frames)
    if len(native_output_files) != expected_native_frames:
        raise RuntimeError(
            "模块D ToonCrafter 渲染失败：原生输出帧数异常，"
            f"unit_id={unit.unit_id}，expected={expected_native_frames}，actual={len(native_output_files)}"
        )

    resampled_files = _resample_frame_sequence(
        source_files=native_output_files,
        target_count=int(unit.exact_frames),
        sequence_dir=sequence_dir,
    )
    frame_pattern = sequence_dir / "frame_%04d.png"
    command = [
        context.config.ffmpeg.ffmpeg_bin,
        "-nostdin",
        "-y",
        "-framerate",
        str(int(context.config.ffmpeg.fps)),
        "-i",
        str(frame_pattern),
        "-c:v",
        str(context.config.ffmpeg.video_codec),
        "-preset",
        str(context.config.ffmpeg.video_preset),
        "-crf",
        str(int(context.config.ffmpeg.video_crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(unit.segment_path),
    ]
    _run_ffmpeg_command(command=command, command_name=f"模块D ToonCrafter 帧序列编码[{unit.unit_id}]")
    return {
        "backend": "comfyui-tooncrafter",
        "segment_path": str(unit.segment_path),
        "target_effective_fps": int(context.config.ffmpeg.fps),
        "frame_count_used": int(len(resampled_files)),
        "native_frame_count": int(len(native_output_files)),
        "generation_width": int(comfy_cfg.generation_width),
        "generation_height": int(comfy_cfg.generation_height),
    }



def _resize_image_for_tooncrafter(source_path: Path, target_path: Path, width: int, height: int) -> None:
    """
    功能说明：将输入关键帧统一缩放到 ToonCrafter v1 固定分辨率。
    参数说明：
    - source_path: 原始关键帧路径。
    - target_path: 缩放后输出路径。
    - width: 目标宽度。
    - height: 目标高度。
    返回值：无。
    异常说明：
    - RuntimeError: 原图不存在时抛出。
    边界条件：当前按固定尺寸直接缩放，不做留边与裁切策略。
    """
    if not source_path.exists():
        raise RuntimeError(f"模块D ToonCrafter 关键帧不存在：path={source_path}")
    with Image.open(source_path) as image_obj:
        resized = image_obj.convert("RGB").resize((int(width), int(height)), resample=Image.Resampling.LANCZOS)
        resized.save(target_path)



def _resample_frame_sequence(source_files: list[Path], target_count: int, sequence_dir: Path) -> list[Path]:
    """
    功能说明：将 ToonCrafter 原生 16 帧序列确定性重采样为模块 D 目标帧数。
    参数说明：
    - source_files: 原生输出帧路径数组。
    - target_count: 目标帧数。
    - sequence_dir: 重采样后序列目录。
    返回值：
    - list[Path]: 重采样后的目标帧路径数组。
    异常说明：
    - RuntimeError: 输入序列为空时抛出。
    边界条件：当 target_count 大于原始帧数时允许重复采样；当小于原始帧数时按等间隔抽帧。
    """
    if not source_files:
        raise RuntimeError("模块D ToonCrafter 重采样失败：原生帧序列为空。")
    normalized_target = max(1, int(target_count))
    source_count = len(source_files)
    copied_files: list[Path] = []
    for target_index in range(normalized_target):
        source_index = _map_resample_index(
            target_index=target_index,
            target_count=normalized_target,
            source_count=source_count,
        )
        source_file = source_files[source_index]
        target_file = sequence_dir / f"frame_{target_index + 1:04d}{source_file.suffix.lower() or '.png'}"
        shutil.copy2(source_file, target_file)
        copied_files.append(target_file)
    return copied_files



def _map_resample_index(target_index: int, target_count: int, source_count: int) -> int:
    """
    功能说明：计算目标序列第 N 帧应映射到原序列的哪个整数索引。
    参数说明：
    - target_index: 目标序列索引（0 基）。
    - target_count: 目标总帧数。
    - source_count: 原始总帧数。
    返回值：
    - int: 原始序列索引（0 基）。
    异常说明：无。
    边界条件：target_count=1 时固定返回 0；其余情况按首尾对齐的线性整数映射。
    """
    if int(source_count) <= 1 or int(target_count) <= 1:
        return 0
    ratio = float(target_index) * float(source_count - 1) / float(target_count - 1)
    mapped_index = int(round(ratio))
    return max(0, min(int(source_count - 1), mapped_index))



def _build_comfyui_client(context: RuntimeContext) -> ComfyUIClient:
    """
    功能说明：按全局配置构造 ComfyUI 客户端。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - ComfyUIClient: 客户端实例。
    异常说明：无。
    边界条件：root_dir 相对路径按项目根解析。
    """
    project_root = Path(__file__).resolve().parents[5]
    comfyui_cfg = context.config.comfyui
    return ComfyUIClient(
        ComfyUIServiceOptions(
            root_dir=(project_root / str(comfyui_cfg.root_dir)).resolve(),
            server_url=str(comfyui_cfg.server_url),
            request_timeout_seconds=float(comfyui_cfg.request_timeout_seconds),
            poll_interval_seconds=float(comfyui_cfg.poll_interval_seconds),
            execution_timeout_seconds=float(comfyui_cfg.execution_timeout_seconds),
        )
    )



def _resolve_seed_value(unit: ModuleDUnit) -> int:
    """
    功能说明：根据 unit 生成稳定种子。
    参数说明：
    - unit: 模块 D 单元对象。
    返回值：
    - int: 32 位正整数种子。
    异常说明：无。
    边界条件：同 unit_id 在重复执行时返回相同结果。
    """
    seed_source = f"{unit.unit_id}|{unit.unit_index}|{unit.exact_frames}".encode("utf-8")
    return int(hashlib.sha256(seed_source).hexdigest()[:8], 16)



def _resolve_tooncrafter_checkpoint_path(context: RuntimeContext) -> Path:
    """
    功能说明：解析 ToonCrafter 主模型在项目模型目录中的真实路径。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 主模型绝对路径。
    异常说明：无。
    边界条件：模型资产统一放在 /root/data/t1/models/tooncrafter/checkpoints 下。
    """
    project_root = Path(__file__).resolve().parents[5]
    return (
        project_root
        / "models"
        / "tooncrafter"
        / "checkpoints"
        / str(context.config.module_d.comfyui.checkpoint_name)
    ).resolve()



def _resolve_sketch_encoder_path(context: RuntimeContext) -> Path:
    """
    功能说明：解析 sketch encoder 在项目模型目录中的真实路径。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: sketch encoder 绝对路径。
    异常说明：无。
    边界条件：v1 虽不接入 workflow，但保留文件存在性校验，避免资产不完整。
    """
    project_root = Path(__file__).resolve().parents[5]
    return (
        project_root
        / "models"
        / "tooncrafter"
        / "checkpoints"
        / str(context.config.module_d.comfyui.sketch_encoder_name)
    ).resolve()
