"""
文件用途：提供模块 C 的关键帧生成器抽象与实现。
核心流程：根据分镜列表生成关键帧，支持 mock 与 diffusion 两种生成路径。
输入输出：输入分镜数组与输出目录，输出帧清单。
依赖说明：依赖标准库 abc/logging/pathlib/json，以及第三方 Pillow、diffusers、torch。
维护说明：DiffusionFrameGenerator 依赖本地配置与模型绑定清单，改动时需同步测试。
"""

# 标准库：定义抽象基类
from abc import ABC, abstractmethod
# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于稳定哈希种子
import hashlib
# 标准库：用于JSON配置读取
import json
# 标准库：用于日志输出
import logging
# 标准库：用于正则处理
import re
# 标准库：用于线程互斥锁
import threading
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 第三方库：用于创建占位图像与加载字体
from PIL import Image, ImageDraw, ImageFont
# 项目内模块：跨模块 diffusers 导入守卫
from music_video_pipeline.diffusers_runtime import load_module_c_diffusion_dependencies

UNKNOWN_LYRIC_TEXT = "[未识别歌词]"
CHANT_LYRIC_TEXT = "吟唱"
INSTRUMENTAL_LYRIC_NOTE = "（说明：根据音源分离后的能量检测，此处为器乐段，但 Fun-ASR 识别到了歌词）"
# 常量：用于清理歌词文本句首标点（含中英文常见符号）的正则。
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+")
# 常量：用于识别“纯标点文本”，避免将其作为可展示歌词。
PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\s，。、；：！？!?,.;:]+$")
# 常量：模块C真实生成配置文件相对路径（项目根目录下）。
MODULE_C_REAL_PROFILE_RELATIVE_PATH = Path("configs/module_c_real_default.json")
# 常量：LoRA绑定清单相对路径（项目根目录下）。
LORA_BINDINGS_RELATIVE_PATH = Path("configs/lora_bindings.json")
# 常量：底模注册表相对路径（项目根目录下）。
BASE_MODEL_REGISTRY_RELATIVE_PATH = Path("configs/base_model_registry.json")
# 常量：模块C真实生成配置顶层字段集合。
MODULE_C_REAL_TOP_LEVEL_KEYS = {"version", "module_c_real"}
# 常量：模块C真实生成配置内部字段集合。
MODULE_C_REAL_SECTION_KEYS = {
    "binding_name",
    "model_series",
    "lora_scale",
    "steps",
    "guidance_scale",
    "negative_prompt",
    "device",
    "torch_dtype",
    "scheduler",
    "seed_mode",
}
# 常量：模块C真实生成配置必填字段集合。
MODULE_C_REAL_REQUIRED_SECTION_KEYS = {"binding_name", "model_series"}
# 常量：允许的设备策略。
MODULE_C_REAL_ALLOWED_DEVICE = {"auto", "cpu", "cuda"}
# 常量：允许的精度策略。
MODULE_C_REAL_ALLOWED_TORCH_DTYPE = {"auto", "float16", "float32", "bfloat16"}
# 常量：允许的调度器策略。
MODULE_C_REAL_ALLOWED_SCHEDULER = {"default", "euler_a", "ddim"}
# 常量：允许的种子策略。
MODULE_C_REAL_ALLOWED_SEED_MODE = {"shot_index", "none"}
# 常量：cuda 设备字符串模式（如 cuda:0、cuda:1）。
CUDA_DEVICE_PATTERN = re.compile(r"^cuda:\d+$")


@dataclass(frozen=True)
class ModuleCRealProfile:
    """
    功能说明：定义模块C真实扩散生成配置结构。
    参数说明：
    - binding_name: LoRA 绑定名（映射 lora_bindings.json）。
    - model_series: 模型系列（当前主要使用 15）。
    - lora_scale: LoRA 权重缩放。
    - steps: 推理步数。
    - guidance_scale: CFG 引导系数。
    - negative_prompt: 负向提示词。
    - device: 设备策略（auto/cpu/cuda）。
    - torch_dtype: 精度策略（auto/float16/float32/bfloat16）。
    - scheduler: 采样器策略（default/euler_a/ddim）。
    - seed_mode: 种子策略（shot_index/none）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：字段合法性由加载器统一校验。
    """

    binding_name: str
    model_series: str
    lora_scale: float
    steps: int
    guidance_scale: float
    negative_prompt: str
    device: str
    torch_dtype: str
    scheduler: str
    seed_mode: str


def _resolve_project_root() -> Path:
    """
    功能说明：解析项目根目录路径。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：假设本文件位于 src/music_video_pipeline/generators 目录。
    """
    return Path(__file__).resolve().parents[3]


def _load_json_object(path: Path, source_name: str) -> dict[str, Any]:
    """
    功能说明：读取并校验 JSON 顶层对象结构。
    参数说明：
    - path: JSON 文件路径。
    - source_name: 数据来源名称（用于错误提示）。
    返回值：
    - dict[str, Any]: JSON 对象。
    异常说明：
    - RuntimeError: 文件不存在、读取失败或结构非法时抛出。
    边界条件：读取编码使用 utf-8-sig，兼容 BOM。
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


def _load_module_c_real_profile(project_root: Path) -> ModuleCRealProfile:
    """
    功能说明：加载并校验模块C真实扩散生成配置。
    参数说明：
    - project_root: 项目根目录。
    返回值：
    - ModuleCRealProfile: 校验通过后的配置对象。
    异常说明：
    - RuntimeError: 配置缺失、字段非法或值超范围时抛出。
    边界条件：仅 binding_name/model_series 必填，其余字段允许按默认值回填。
    """
    profile_path = (project_root / MODULE_C_REAL_PROFILE_RELATIVE_PATH).resolve()
    raw_data = _load_json_object(path=profile_path, source_name="模块C真实生成配置")

    raw_top_keys = set(raw_data.keys())
    unknown_top_keys = raw_top_keys.difference(MODULE_C_REAL_TOP_LEVEL_KEYS)
    if unknown_top_keys:
        raise RuntimeError(f"模块C真实生成配置存在未知顶层字段：{sorted(unknown_top_keys)}")
    for required_top_key in MODULE_C_REAL_TOP_LEVEL_KEYS:
        if required_top_key not in raw_data:
            raise RuntimeError(f"模块C真实生成配置缺失顶层字段：{required_top_key}")

    raw_version = raw_data["version"]
    if not isinstance(raw_version, int):
        raise RuntimeError("模块C真实生成配置 version 必须是 int。")
    if raw_version != 1:
        raise RuntimeError(f"模块C真实生成配置 version 非法：{raw_version}，当前仅支持 1。")

    section_raw = raw_data["module_c_real"]
    if not isinstance(section_raw, dict):
        raise RuntimeError("模块C真实生成配置 module_c_real 必须是对象。")
    section_top_keys = set(section_raw.keys())
    unknown_section_keys = section_top_keys.difference(MODULE_C_REAL_SECTION_KEYS)
    if unknown_section_keys:
        raise RuntimeError(f"模块C真实生成配置存在未知字段：{sorted(unknown_section_keys)}")
    for required_section_key in MODULE_C_REAL_REQUIRED_SECTION_KEYS:
        if required_section_key not in section_raw:
            raise RuntimeError(f"模块C真实生成配置缺失字段：module_c_real.{required_section_key}")

    defaults = {
        "lora_scale": 1.0,
        "steps": 24,
        "guidance_scale": 7.0,
        "negative_prompt": "lowres, blurry, bad anatomy",
        "device": "auto",
        "torch_dtype": "auto",
        "scheduler": "default",
        "seed_mode": "shot_index",
    }
    merged_section = {**defaults, **section_raw}

    binding_name = str(merged_section["binding_name"]).strip()
    model_series = str(merged_section["model_series"]).strip()
    if not binding_name:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.binding_name 不能为空。")
    if not model_series:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.model_series 不能为空。")

    try:
        lora_scale = float(merged_section["lora_scale"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.lora_scale 必须是数字。") from error
    if lora_scale <= 0:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.lora_scale 必须大于 0。")

    try:
        steps = int(merged_section["steps"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.steps 必须是整数。") from error
    if steps <= 0:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.steps 必须大于 0。")

    try:
        guidance_scale = float(merged_section["guidance_scale"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.guidance_scale 必须是数字。") from error
    if guidance_scale < 0:
        raise RuntimeError("模块C真实生成配置非法：module_c_real.guidance_scale 不能小于 0。")

    negative_prompt = str(merged_section["negative_prompt"])
    device = str(merged_section["device"]).strip().lower()
    torch_dtype = str(merged_section["torch_dtype"]).strip().lower()
    scheduler = str(merged_section["scheduler"]).strip().lower()
    seed_mode = str(merged_section["seed_mode"]).strip().lower()

    if (device not in MODULE_C_REAL_ALLOWED_DEVICE) and (not bool(CUDA_DEVICE_PATTERN.fullmatch(device))):
        raise RuntimeError(
            f"模块C真实生成配置非法：module_c_real.device={device}，"
            f"合法值={sorted(MODULE_C_REAL_ALLOWED_DEVICE)} 或 cuda:N。"
        )
    if torch_dtype not in MODULE_C_REAL_ALLOWED_TORCH_DTYPE:
        raise RuntimeError(
            f"模块C真实生成配置非法：module_c_real.torch_dtype={torch_dtype}，"
            f"合法值={sorted(MODULE_C_REAL_ALLOWED_TORCH_DTYPE)}。"
        )
    if scheduler not in MODULE_C_REAL_ALLOWED_SCHEDULER:
        raise RuntimeError(
            f"模块C真实生成配置非法：module_c_real.scheduler={scheduler}，"
            f"合法值={sorted(MODULE_C_REAL_ALLOWED_SCHEDULER)}。"
        )
    if seed_mode not in MODULE_C_REAL_ALLOWED_SEED_MODE:
        raise RuntimeError(
            f"模块C真实生成配置非法：module_c_real.seed_mode={seed_mode}，"
            f"合法值={sorted(MODULE_C_REAL_ALLOWED_SEED_MODE)}。"
        )

    return ModuleCRealProfile(
        binding_name=binding_name,
        model_series=model_series,
        lora_scale=lora_scale,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        device=device,
        torch_dtype=torch_dtype,
        scheduler=scheduler,
        seed_mode=seed_mode,
    )


def _resolve_binding_runtime_assets(project_root: Path, binding_name: str, model_series: str) -> dict[str, Any]:
    """
    功能说明：根据 binding_name 与 model_series 解析 LoRA 与底模运行时资产。
    参数说明：
    - project_root: 项目根目录。
    - binding_name: LoRA 绑定名。
    - model_series: 模型系列（如 15）。
    返回值：
    - dict[str, Any]: 含 base_model_key/base_model_path/lora_file_path 的资产字典。
    异常说明：
    - RuntimeError: 绑定、底模或路径非法时抛出。
    边界条件：底模路径以 base_model_registry.json 为权威来源。
    """
    binding_path = (project_root / LORA_BINDINGS_RELATIVE_PATH).resolve()
    binding_data = _load_json_object(path=binding_path, source_name="LoRA绑定清单")
    bindings = binding_data.get("bindings")
    if not isinstance(bindings, list):
        raise RuntimeError(f"LoRA绑定清单字段 bindings 非法，需为数组：{binding_path}")

    matched_binding: dict[str, Any] | None = None
    for item in bindings:
        if not isinstance(item, dict):
            continue
        item_binding_name = str(item.get("binding_name", "")).strip()
        item_model_series = str(item.get("model_series", "")).strip()
        if item_binding_name == binding_name and item_model_series == model_series:
            matched_binding = item
            break
    if matched_binding is None:
        raise RuntimeError(
            f"模块C真实生成绑定不存在：binding_name={binding_name}，model_series={model_series}。"
        )

    base_model_key = str(matched_binding.get("base_model_key", "")).strip()
    if not base_model_key:
        raise RuntimeError(
            f"模块C真实生成绑定缺失 base_model_key：binding_name={binding_name}，model_series={model_series}。"
        )

    lora_file_text = str(matched_binding.get("lora_file", "")).strip()
    if not lora_file_text:
        raise RuntimeError(
            f"模块C真实生成绑定缺失 lora_file：binding_name={binding_name}，model_series={model_series}。"
        )
    lora_file_path = (project_root / lora_file_text).resolve()
    if not lora_file_path.exists():
        raise RuntimeError(
            f"模块C真实生成 LoRA 文件不存在：binding_name={binding_name}，lora_file={lora_file_path}。"
        )

    registry_path = (project_root / BASE_MODEL_REGISTRY_RELATIVE_PATH).resolve()
    registry_data = _load_json_object(path=registry_path, source_name="底模注册表")
    base_models = registry_data.get("base_models")
    if not isinstance(base_models, list):
        raise RuntimeError(f"底模注册表字段 base_models 非法，需为数组：{registry_path}")

    matched_base_model: dict[str, Any] | None = None
    for item in base_models:
        if not isinstance(item, dict):
            continue
        item_key = str(item.get("key", "")).strip()
        if item_key == base_model_key:
            matched_base_model = item
            break
    if matched_base_model is None:
        raise RuntimeError(f"模块C真实生成底模 key 不存在：base_model_key={base_model_key}。")

    record_series = str(matched_base_model.get("series", "")).strip()
    if record_series != model_series:
        raise RuntimeError(
            f"模块C真实生成底模系列不匹配：base_model_key={base_model_key}，series={record_series}，"
            f"expected={model_series}。"
        )
    if not bool(matched_base_model.get("enabled", False)):
        raise RuntimeError(f"模块C真实生成底模 key 已禁用：base_model_key={base_model_key}。")

    base_model_path_text = str(matched_base_model.get("path", "")).strip()
    if not base_model_path_text:
        raise RuntimeError(f"模块C真实生成底模记录缺失 path：base_model_key={base_model_key}。")
    base_model_path = (project_root / base_model_path_text).resolve()
    if not base_model_path.exists():
        raise RuntimeError(
            f"模块C真实生成底模路径不存在：base_model_key={base_model_key}，path={base_model_path}。"
        )

    return {
        "binding_name": binding_name,
        "model_series": model_series,
        "base_model_key": base_model_key,
        "base_model_path": base_model_path,
        "lora_file_path": lora_file_path,
    }


def resolve_module_c_diffusion_trace_metadata() -> dict[str, str]:
    """
    功能说明：解析模块C diffusion 运行所需可追溯元信息。
    参数说明：无。
    返回值：
    - dict[str, str]: 含 binding_name/base_model_key/lora_file 字段的字典。
    异常说明：
    - RuntimeError: 配置或资产解析失败时抛出。
    边界条件：仅在 diffusion 模式下调用。
    """
    project_root = _resolve_project_root()
    profile = _load_module_c_real_profile(project_root=project_root)
    assets = _resolve_binding_runtime_assets(
        project_root=project_root,
        binding_name=profile.binding_name,
        model_series=profile.model_series,
    )
    return {
        "binding_name": str(assets["binding_name"]),
        "base_model_key": str(assets["base_model_key"]),
        "lora_file": str(assets["lora_file_path"]),
    }


def _import_diffusion_runtime_dependencies() -> dict[str, Any]:
    """
    功能说明：惰性导入扩散推理依赖，避免 mock 模式无谓开销。
    参数说明：无。
    返回值：
    - dict[str, Any]: torch 与 diffusers 关键类型映射。
    异常说明：
    - RuntimeError: 导入失败时抛出。
    边界条件：仅在 diffusion 模式实际生成时调用。
    """
    try:
        return load_module_c_diffusion_dependencies()
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"模块C真实扩散依赖导入失败，请检查 diffusers/torch 环境：{error}") from error


def _resolve_runtime_device(profile: ModuleCRealProfile, torch_module: Any) -> str:
    """
    功能说明：解析扩散推理设备。
    参数说明：
    - profile: 模块C真实配置对象。
    - torch_module: torch 模块对象。
    返回值：
    - str: 设备字符串（cpu/cuda）。
    异常说明：
    - RuntimeError: 配置要求 cuda 但不可用时抛出。
    边界条件：auto 优先选择 cuda，不可用时回退 cpu。
    """
    normalized_device = str(profile.device).strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    cuda_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    if normalized_device == "auto":
        if cuda_available and cuda_count >= 2:
            return "cuda:0"
        if cuda_available:
            return "cuda:0"
        return "cpu"
    if normalized_device == "cuda":
        if not cuda_available:
            raise RuntimeError("模块C真实扩散配置要求使用 cuda，但当前环境不可用。")
        return "cuda:0"
    if CUDA_DEVICE_PATTERN.fullmatch(normalized_device):
        if not cuda_available:
            raise RuntimeError(f"模块C真实扩散配置要求使用 {normalized_device}，但当前环境无可用 CUDA。")
        try:
            device_index = int(normalized_device.split(":", 1)[1])
        except ValueError as error:
            raise RuntimeError(f"模块C真实扩散配置非法设备：{normalized_device}") from error
        if device_index < 0 or device_index >= cuda_count:
            if cuda_count >= 1:
                logging.getLogger("C").warning(
                    "模块C真实扩散配置设备索引越界，已自动回退到 cuda:0，原始设备=%s，可用GPU数量=%s",
                    normalized_device,
                    cuda_count,
                )
                return "cuda:0"
            raise RuntimeError(f"模块C真实扩散配置设备索引越界：{normalized_device}，当前可用 GPU 数量={cuda_count}")
        return normalized_device
    return normalized_device


def _resolve_torch_dtype(profile: ModuleCRealProfile, torch_module: Any, device: str) -> Any:
    """
    功能说明：根据配置与设备解析 torch 精度类型。
    参数说明：
    - profile: 模块C真实配置对象。
    - torch_module: torch 模块对象。
    - device: 运行设备。
    返回值：
    - Any: torch dtype 对象。
    异常说明：
    - RuntimeError: 精度字符串非法时抛出。
    边界条件：auto 在 cuda 下选 float16，cpu 下选 float32。
    """
    if profile.torch_dtype == "auto":
        normalized_device = str(device).strip().lower()
        return torch_module.float16 if normalized_device.startswith("cuda") else torch_module.float32
    if profile.torch_dtype == "float16":
        return torch_module.float16
    if profile.torch_dtype == "float32":
        return torch_module.float32
    if profile.torch_dtype == "bfloat16":
        return torch_module.bfloat16
    raise RuntimeError(f"模块C真实扩散配置 torch_dtype 非法：{profile.torch_dtype}")


def _resolve_seed_value(shot_id: str, shot_index: int) -> int:
    """
    功能说明：根据 shot 生成稳定随机种子，保证可复现。
    参数说明：
    - shot_id: 镜头ID。
    - shot_index: 镜头索引（0 基）。
    返回值：
    - int: 32位种子值。
    异常说明：无。
    边界条件：不同 shot_id/index 组合应稳定地产生不同种子。
    """
    source_text = f"{shot_id}|{shot_index}"
    digest_text = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return int(digest_text[:8], 16)


class FrameGenerator(ABC):
    """
    功能说明：关键帧生成器接口定义。
    参数说明：无。
    返回值：不适用。
    异常说明：子类可抛出实现相关异常。
    边界条件：输出单元结构需包含 frame_path 与时间字段。
    """

    @abstractmethod
    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：根据单个分镜生成关键帧。
        参数说明：
        - shot: 模块 B 输出的单个分镜对象。
        - output_dir: 图像输出目录。
        - width: 输出宽度。
        - height: 输出高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个 frame_item。
        异常说明：由子类决定。
        边界条件：width/height 建议为偶数。
        """
        raise NotImplementedError


class MockFrameGenerator(FrameGenerator):
    """
    功能说明：生成占位关键帧（MVP 默认实现）。
    参数说明：无。
    返回值：不适用。
    异常说明：磁盘写入失败时抛出 OSError。
    边界条件：若分镜无效时自动使用最小时长 0.5 秒。
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        功能说明：初始化占位关键帧生成器。
        参数说明：
        - logger: 可选日志对象，用于输出字体降级告警。
        返回值：无。
        异常说明：无。
        边界条件：logger 为空时仅跳过告警输出。
        """
        self.logger = logger
        # 标记：避免中文字体缺失告警被重复刷屏。
        self._font_warning_emitted = False

    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：为单个分镜生成一张带文字的占位图。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 帧输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个分镜对应的 frame_item。
        异常说明：目录不可写时抛 OSError。
        边界条件：时长 <= 0 时自动修正为 0.5 秒。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        title_font_size = max(28, int(width * 0.035))
        body_font_size = max(24, int(width * 0.028))
        title_font_obj, title_font_source = _load_chinese_font(size=title_font_size)
        body_font_obj, body_font_source = _load_chinese_font(size=body_font_size)
        if self.logger and ("pil_default" in {title_font_source, body_font_source}) and (not self._font_warning_emitted):
            self.logger.warning("模块C未加载到可用中文字体，当前使用默认字体，中文可能无法正常显示。")
            self._font_warning_emitted = True

        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)

        image_path = output_dir / f"frame_{shot_index + 1:03d}.png"
        self._build_placeholder_image(
            image_path=image_path,
            width=width,
            height=height,
            shot=shot,
            title_font_obj=title_font_obj,
            body_font_obj=body_font_obj,
            title_font_size=title_font_size,
            body_font_size=body_font_size,
        )
        return {
            "shot_id": str(shot["shot_id"]),
            "frame_path": str(image_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "scene_desc": str(shot.get("scene_desc", "")),
            "keyframe_prompt_zh": str(shot.get("keyframe_prompt_zh", "")),
            "keyframe_prompt_en": str(shot.get("keyframe_prompt_en", "")),
            "keyframe_prompt": str(shot.get("keyframe_prompt", "")),
            "video_prompt_zh": str(shot.get("video_prompt_zh", "")),
            "video_prompt_en": str(shot.get("video_prompt_en", "")),
            "video_prompt": str(shot.get("video_prompt", "")),
        }

    def _build_placeholder_image(
        self,
        image_path: Path,
        width: int,
        height: int,
        shot: dict[str, Any],
        title_font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        body_font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        title_font_size: int,
        body_font_size: int,
    ) -> None:
        """
        功能说明：生成单张占位图并写入分镜信息文字。
        参数说明：
        - image_path: 输出图像路径。
        - width: 图像宽度。
        - height: 图像高度。
        - shot: 当前分镜字典。
        - title_font_obj: 标题字体对象。
        - body_font_obj: 正文字体对象。
        - title_font_size: 标题字号（像素）。
        - body_font_size: 正文字号（像素）。
        返回值：无。
        异常说明：文件写入失败时抛 OSError。
        边界条件：段落/scene/lyrics 文本按像素宽度自动换行，超限时追加省略号。
        """
        background_color = (26, 51, 77)
        image = Image.new(mode="RGB", size=(width, height), color=background_color)
        drawer = ImageDraw.Draw(image)
        text_color = (255, 255, 255)
        margin = max(24, int(width * 0.04))
        line_gap = max(8, int(body_font_size * 0.45))
        line_height = body_font_size + line_gap
        max_text_width = width - margin * 2
        cursor_y = margin

        drawer.text((margin, cursor_y), f"镜头ID：{shot['shot_id']}", fill=text_color, font=title_font_obj)
        cursor_y += title_font_size + line_gap

        drawer.text(
            (margin, cursor_y),
            f"时间：{float(shot['start_time']):.2f}-{float(shot['end_time']):.2f}s",
            fill=text_color,
            font=body_font_obj,
        )
        cursor_y += line_height

        big_segment_text = _extract_big_segment_display_for_shot(shot=shot)
        big_segment_line = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=f"大段落：{big_segment_text}",
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=1,
        )
        drawer.text((margin, cursor_y), big_segment_line[0], fill=text_color, font=body_font_obj)
        cursor_y += line_height

        role_text = _extract_audio_role_display_for_shot(shot=shot)
        role_line = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=f"段落类型：{role_text}",
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=1,
        )
        drawer.text((margin, cursor_y), role_line[0], fill=text_color, font=body_font_obj)
        cursor_y += line_height

        footer_rows = 2
        footer_reserved_height = footer_rows * body_font_size + (footer_rows - 1) * line_gap
        lyric_min_reserved_height = line_height
        scene_available_height = max(
            line_height,
            height - margin - footer_reserved_height - lyric_min_reserved_height - cursor_y,
        )
        max_scene_lines = max(1, min(3, scene_available_height // line_height))

        scene_text = f"场景：{str(shot['scene_desc'])}"
        scene_lines = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=scene_text,
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=max_scene_lines,
        )
        for line in scene_lines:
            drawer.text((margin, cursor_y), line, fill=text_color, font=body_font_obj)
            cursor_y += line_height

        lyric_text = _extract_lyric_text_for_shot(shot=shot)
        lyric_render_text = f"歌词：{lyric_text}" if lyric_text else "歌词：<无>"
        footer_y = height - margin - footer_reserved_height
        lyric_available_height = max(
            line_height,
            footer_y - cursor_y - line_gap,
        )
        max_lyric_lines = max(1, min(4, lyric_available_height // line_height))
        lyric_lines = _wrap_text_by_pixel_width(
            drawer=drawer,
            text=lyric_render_text,
            font_obj=body_font_obj,
            max_width=max_text_width,
            max_lines=max_lyric_lines,
        )
        for line in lyric_lines:
            drawer.text((margin, cursor_y), line, fill=text_color, font=body_font_obj)
            cursor_y += line_height

        drawer.text((margin, footer_y), f"运镜：{shot['camera_motion']}", fill=text_color, font=body_font_obj)
        drawer.text((margin, footer_y + line_height), f"转场：{shot['transition']}", fill=text_color, font=body_font_obj)
        image.save(image_path)


class DiffusionFrameGenerator(FrameGenerator):
    """
    功能说明：扩散模型关键帧真实生成实现。
    参数说明：
    - logger: 日志对象。
    返回值：不适用。
    异常说明：依赖缺失、配置错误或推理失败时抛 RuntimeError。
    边界条件：同一实例内串行执行推理，避免并发访问同一 pipeline。
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        功能说明：初始化扩散生成器运行时状态。
        参数说明：
        - logger: 日志对象。
        返回值：无。
        异常说明：无。
        边界条件：pipeline 与权重在首次 generate_one 时惰性加载。
        """
        self.logger = logger
        # 互斥锁：保护 pipeline 初始化与推理调用，避免多线程并发冲突。
        self._runtime_lock = threading.Lock()
        # 缓存：首次成功加载后复用，避免每个 shot 重复装载模型。
        self._pipeline: Any | None = None
        # 缓存：torch 模块对象。
        self._torch_module: Any | None = None
        # 缓存：模块C真实配置对象。
        self._profile: ModuleCRealProfile | None = None
        # 缓存：运行设备字符串。
        self._device: str = "cpu"
        # 缓存：绑定解析结果（用于可追溯输出）。
        self._binding_assets: dict[str, Any] | None = None

    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：根据分镜调用扩散模型生成真实关键帧。
        参数说明：
        - shot: 分镜对象。
        - output_dir: 输出目录。
        - width: 图像宽度。
        - height: 图像高度。
        - shot_index: 分镜顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 单个 frame_item。
        异常说明：配置、依赖、资产或推理失败时抛 RuntimeError。
        边界条件：prompt 为空会直接失败，不做静默降级。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)
        shot_id = str(shot.get("shot_id", "")).strip()
        if not shot_id:
            raise RuntimeError("模块C扩散生成失败：shot_id 不能为空。")
        prompt = str(shot.get("keyframe_prompt_en", "")).strip()
        if not prompt:
            raise RuntimeError(f"模块C扩散生成失败：keyframe_prompt_en 为空（当前策略要求必须英文），shot_id={shot_id}")

        with self._runtime_lock:
            self._ensure_runtime_loaded()
            if self._pipeline is None or self._profile is None or self._binding_assets is None:
                raise RuntimeError("模块C扩散生成失败：运行时尚未完成初始化。")
            runtime_kwargs = self._build_runtime_kwargs(
                prompt=prompt,
                shot_id=shot_id,
                shot_index=shot_index,
                width=width,
                height=height,
            )
            try:
                pipeline_output = self._pipeline(**runtime_kwargs)
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"模块C扩散生成失败，shot_id={shot_id}，错误={error}") from error

            generated_images = getattr(pipeline_output, "images", None)
            if not isinstance(generated_images, list) or not generated_images:
                raise RuntimeError(f"模块C扩散生成失败：未返回有效图像，shot_id={shot_id}")
            image_obj = generated_images[0]

            image_path = output_dir / f"frame_{shot_index + 1:03d}.png"
            try:
                image_obj.save(image_path)
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"模块C扩散生成失败：图像保存失败，shot_id={shot_id}，错误={error}") from error

            return {
                "shot_id": shot_id,
                "frame_path": str(image_path),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "scene_desc": str(shot.get("scene_desc", "")),
                "keyframe_prompt_zh": str(shot.get("keyframe_prompt_zh", "")),
                "keyframe_prompt_en": str(shot.get("keyframe_prompt_en", "")),
                "keyframe_prompt": prompt,
                "video_prompt_zh": str(shot.get("video_prompt_zh", "")),
                "video_prompt_en": str(shot.get("video_prompt_en", "")),
                "video_prompt": str(shot.get("video_prompt", "")),
                "binding_name": str(self._binding_assets["binding_name"]),
                "base_model_key": str(self._binding_assets["base_model_key"]),
                "lora_file": str(self._binding_assets["lora_file_path"]),
            }

    def _ensure_runtime_loaded(self) -> None:
        """
        功能说明：初始化扩散推理运行时（仅首次执行）。
        参数说明：无。
        返回值：无。
        异常说明：配置、绑定、底模或依赖异常时抛 RuntimeError。
        边界条件：初始化成功后复用 pipeline，不重复加载。
        """
        if self._pipeline is not None:
            return
        project_root = _resolve_project_root()
        profile = _load_module_c_real_profile(project_root=project_root)
        binding_assets = _resolve_binding_runtime_assets(
            project_root=project_root,
            binding_name=profile.binding_name,
            model_series=profile.model_series,
        )
        runtime_dependencies = _import_diffusion_runtime_dependencies()
        torch_module = runtime_dependencies["torch"]
        stable_diffusion_pipeline = runtime_dependencies["StableDiffusionPipeline"]
        resolved_device = _resolve_runtime_device(profile=profile, torch_module=torch_module)
        resolved_dtype = _resolve_torch_dtype(profile=profile, torch_module=torch_module, device=resolved_device)
        base_model_path = Path(binding_assets["base_model_path"]).resolve()
        lora_file_path = Path(binding_assets["lora_file_path"]).resolve()

        try:
            pipeline = stable_diffusion_pipeline.from_pretrained(
                str(base_model_path),
                torch_dtype=resolved_dtype,
                local_files_only=True,
            )
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"模块C扩散初始化失败：加载底模失败，base_model_path={base_model_path}，错误={error}"
            ) from error
        try:
            pipeline = pipeline.to(resolved_device)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"模块C扩散初始化失败：切换设备失败，device={resolved_device}，错误={error}"
            ) from error
        try:
            pipeline.load_lora_weights(
                str(lora_file_path.parent),
                weight_name=lora_file_path.name,
            )
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"模块C扩散初始化失败：加载LoRA失败，lora_file={lora_file_path}，错误={error}"
            ) from error

        self._apply_scheduler_if_needed(
            pipeline=pipeline,
            scheduler_name=profile.scheduler,
            runtime_dependencies=runtime_dependencies,
        )
        self._pipeline = pipeline
        self._torch_module = torch_module
        self._profile = profile
        self._device = resolved_device
        self._binding_assets = binding_assets
        self.logger.info(
            "模块C扩散运行时初始化完成，binding_name=%s，base_model_key=%s，device=%s，dtype=%s",
            profile.binding_name,
            binding_assets["base_model_key"],
            resolved_device,
            profile.torch_dtype,
        )

    def _apply_scheduler_if_needed(
        self,
        pipeline: Any,
        scheduler_name: str,
        runtime_dependencies: dict[str, Any],
    ) -> None:
        """
        功能说明：按配置切换扩散采样器。
        参数说明：
        - pipeline: 扩散推理管线实例。
        - scheduler_name: 采样器策略名。
        - runtime_dependencies: 运行时依赖映射。
        返回值：无。
        异常说明：切换失败时抛 RuntimeError。
        边界条件：default 不做替换，沿用模型原始 scheduler。
        """
        if scheduler_name == "default":
            return
        try:
            if scheduler_name == "euler_a":
                scheduler_class = runtime_dependencies["EulerAncestralDiscreteScheduler"]
                pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                return
            if scheduler_name == "ddim":
                scheduler_class = runtime_dependencies["DDIMScheduler"]
                pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                return
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"模块C扩散初始化失败：采样器切换失败，scheduler={scheduler_name}，错误={error}") from error
        raise RuntimeError(f"模块C扩散初始化失败：不支持的采样器 {scheduler_name}")

    def _build_runtime_kwargs(
        self,
        prompt: str,
        shot_id: str,
        shot_index: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """
        功能说明：构建扩散推理调用参数。
        参数说明：
        - prompt: 正向提示词。
        - shot_id: 镜头ID。
        - shot_index: 镜头索引（0 基）。
        - width: 图像宽度。
        - height: 图像高度。
        返回值：
        - dict[str, Any]: 可传入 pipeline 的参数字典。
        异常说明：运行时未初始化时抛 RuntimeError。
        边界条件：seed_mode=none 时不传 generator，走随机采样。
        """
        if self._profile is None:
            raise RuntimeError("模块C扩散生成失败：配置尚未初始化。")
        runtime_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": self._profile.negative_prompt,
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(self._profile.steps),
            "guidance_scale": float(self._profile.guidance_scale),
            "cross_attention_kwargs": {"scale": float(self._profile.lora_scale)},
        }
        if self._profile.seed_mode == "shot_index":
            if self._torch_module is None:
                raise RuntimeError("模块C扩散生成失败：torch 运行时尚未初始化。")
            seed_value = _resolve_seed_value(shot_id=shot_id, shot_index=shot_index)
            generator = self._torch_module.Generator(device=self._device)
            generator.manual_seed(seed_value)
            runtime_kwargs["generator"] = generator
        return runtime_kwargs


def build_frame_generator(mode: str, logger: logging.Logger) -> FrameGenerator:
    """
    功能说明：根据模式构建关键帧生成器实例。
    参数说明：
    - mode: 生成模式（mock/diffusion）。
    - logger: 日志对象。
    返回值：
    - FrameGenerator: 对应生成器实例。
    异常说明：无。
    边界条件：未知模式将降级到 Mock。
    """
    mode_text = mode.lower().strip()
    if mode_text == "diffusion":
        return DiffusionFrameGenerator(logger=logger)
    if mode_text != "mock":
        logger.warning("未知关键帧生成模式: %s，已降级为 mock。", mode)
    return MockFrameGenerator(logger=logger)


def _load_chinese_font(size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    """
    功能说明：加载可显示中文的字体，若不可用则回退到默认字体。
    参数说明：
    - size: 字号大小。
    返回值：
    - tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]: (字体对象, 字体来源标识)。
    异常说明：内部捕获字体加载异常并继续尝试下一个候选。
    边界条件：所有候选失败时返回 PIL 默认字体（可能不支持中文）。
    """
    font_candidates = _resolve_chinese_font_candidates()
    for font_path in font_candidates:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size), str(font_path)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default(), "pil_default"


def _resolve_chinese_font_candidates() -> list[Path]:
    """
    功能说明：返回中文字体候选列表（按优先级排序）。
    参数说明：无。
    返回值：
    - list[Path]: 字体路径候选。
    异常说明：无。
    边界条件：优先仓库内置字体，其次 Linux/Windows 系统字体。
    """
    project_root = Path(__file__).resolve().parents[3]
    return [
        project_root / "resources" / "fonts" / "NotoSansCJKsc-Regular.otf",
        project_root / "resources" / "fonts" / "msyh.ttc",
        project_root / "resources" / "fonts" / "simhei.ttf",
        project_root / "resources" / "fonts" / "simsun.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]


def _measure_text_pixel_width(
    drawer: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    """
    功能说明：测量文本在指定字体下的像素宽度。
    参数说明：
    - drawer: Pillow 绘制对象。
    - text: 待测量文本。
    - font_obj: 字体对象。
    返回值：
    - int: 文本像素宽度。
    异常说明：无。
    边界条件：空文本宽度为 0。
    """
    if not text:
        return 0
    left, _, right, _ = drawer.textbbox((0, 0), text, font=font_obj)
    return max(0, right - left)


def _wrap_text_by_pixel_width(
    drawer: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """
    功能说明：按像素宽度逐字换行，并限制最大行数。
    参数说明：
    - drawer: Pillow 绘制对象。
    - text: 待换行文本。
    - font_obj: 字体对象。
    - max_width: 允许的最大像素宽度。
    - max_lines: 最大行数。
    返回值：
    - list[str]: 换行结果。
    异常说明：无。
    边界条件：超出最大行数时最后一行自动追加省略号。
    """
    if max_lines <= 0:
        return []
    if not text:
        return [""]

    lines: list[str] = []
    current_line = ""
    for char_text in text:
        trial_line = f"{current_line}{char_text}"
        if _measure_text_pixel_width(drawer, trial_line, font_obj) <= max_width:
            current_line = trial_line
            continue
        if current_line:
            lines.append(current_line)
            current_line = char_text
        else:
            lines.append(char_text)
            current_line = ""
    if current_line:
        lines.append(current_line)

    if len(lines) <= max_lines:
        return lines

    clipped_lines = lines[:max_lines]
    clipped_lines[-1] = _append_ellipsis_to_line(
        drawer=drawer,
        line_text=clipped_lines[-1],
        font_obj=font_obj,
        max_width=max_width,
    )
    return clipped_lines


def _append_ellipsis_to_line(
    drawer: ImageDraw.ImageDraw,
    line_text: str,
    font_obj: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    """
    功能说明：在行尾追加省略号，必要时裁剪文本保证宽度合法。
    参数说明：
    - drawer: Pillow 绘制对象。
    - line_text: 原始行文本。
    - font_obj: 字体对象。
    - max_width: 最大像素宽度。
    返回值：
    - str: 处理后的行文本。
    异常说明：无。
    边界条件：若宽度极小，至少返回 "..."。
    """
    suffix = "..."
    if _measure_text_pixel_width(drawer, f"{line_text}{suffix}", font_obj) <= max_width:
        return f"{line_text}{suffix}"

    trimmed_text = line_text
    while trimmed_text and _measure_text_pixel_width(drawer, f"{trimmed_text}{suffix}", font_obj) > max_width:
        trimmed_text = trimmed_text[:-1]
    return f"{trimmed_text}{suffix}" if trimmed_text else suffix


def _extract_lyric_text_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：从分镜中提取歌词展示文本，兼容新旧字段结构。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 可渲染的歌词文本；无歌词时返回空字符串。
    异常说明：无。
    边界条件：优先使用 lyric_text，缺失时尝试从 lyric_units 聚合。
    """
    lyric_text = _clean_lyric_render_text(str(shot.get("lyric_text", "")).strip())
    if lyric_text:
        return _append_instrumental_lyric_note_if_needed(shot=shot, lyric_text=lyric_text)

    lyric_units = shot.get("lyric_units", [])
    if not isinstance(lyric_units, list):
        return ""

    reliable_text_items: list[str] = []
    has_unknown = False
    has_chant = False
    for item in lyric_units:
        if not isinstance(item, dict):
            continue
        text = _clean_lyric_render_text(str(item.get("text", "")).strip())
        if not text:
            continue
        if text == UNKNOWN_LYRIC_TEXT:
            has_unknown = True
            continue
        if text == CHANT_LYRIC_TEXT:
            has_chant = True
            continue
        reliable_text_items.append(text)

    if reliable_text_items:
        lyric_text_joined = " ".join(reliable_text_items)
        return _append_instrumental_lyric_note_if_needed(shot=shot, lyric_text=lyric_text_joined)
    if has_unknown:
        return UNKNOWN_LYRIC_TEXT
    if has_chant:
        return CHANT_LYRIC_TEXT
    return ""


def _append_instrumental_lyric_note_if_needed(shot: dict[str, Any], lyric_text: str) -> str:
    """
    功能说明：在“器乐段但有有效歌词”场景追加固定说明文案，避免误判导致吞字。
    参数说明：
    - shot: 分镜字典，读取 audio_role 字段。
    - lyric_text: 已提取且清洗后的歌词文本。
    返回值：
    - str: 可能追加说明文案后的歌词文本。
    异常说明：无。
    边界条件：未识别标记/吟唱标记不追加说明文案。
    """
    cleaned_text = str(lyric_text).strip()
    if not cleaned_text:
        return ""
    if cleaned_text in {UNKNOWN_LYRIC_TEXT, CHANT_LYRIC_TEXT}:
        return cleaned_text
    audio_role = str(shot.get("audio_role", "")).strip().lower()
    if audio_role != "instrumental":
        return cleaned_text
    if INSTRUMENTAL_LYRIC_NOTE in cleaned_text:
        return cleaned_text
    return f"{cleaned_text}{INSTRUMENTAL_LYRIC_NOTE}"


def _clean_lyric_render_text(text: str) -> str:
    """
    功能说明：清洗占位图歌词文本，避免句首标点或纯标点上屏。
    参数说明：
    - text: 原始歌词文本。
    返回值：
    - str: 清洗后的可渲染文本；若仅标点返回空字符串。
    异常说明：无。
    边界条件：仅移除句首标点，句中与句尾标点保留。
    """
    cleaned_text = EDGE_PUNCTUATION_PATTERN.sub("", str(text).strip()).strip()
    if not cleaned_text:
        return ""
    if PUNCTUATION_ONLY_PATTERN.fullmatch(cleaned_text):
        return ""
    return cleaned_text


def _extract_big_segment_display_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：提取分镜对应的大段落展示文本（标签+ID）。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 大段落展示字符串。
    异常说明：无。
    边界条件：缺失字段时返回“<未知>”。
    """
    big_label = str(shot.get("big_segment_label", "")).strip()
    big_id = str(shot.get("big_segment_id", "")).strip()
    if big_label and big_id:
        return f"{big_label} ({big_id})"
    if big_label:
        return big_label
    if big_id:
        return big_id
    return "<未知>"


def _extract_audio_role_display_for_shot(shot: dict[str, Any]) -> str:
    """
    功能说明：提取分镜对应的段落类型展示文本。
    参数说明：
    - shot: 分镜字典。
    返回值：
    - str: 段落类型（器乐段/人声段/<未知>）。
    异常说明：无。
    边界条件：字段缺失或非法值时返回“<未知>”。
    """
    audio_role = str(shot.get("audio_role", "")).strip().lower()
    if audio_role == "instrumental":
        return "器乐段"
    if audio_role == "vocal":
        return "人声段"
    return "<未知>"
