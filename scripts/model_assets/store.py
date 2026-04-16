"""
文件用途：提供路径解析、日志初始化与 JSON 存储操作。
核心流程：路径归一化 -> 读取或初始化配置 -> upsert 写回。
输入输出：输入项目路径与数据对象，输出标准化路径和持久化结果。
依赖说明：依赖标准库 pathlib/json/logging/sys。
维护说明：存储层不直接依赖 bypy，便于单元测试与复用。
"""

# 标准库：用于时间戳生成
from datetime import datetime
# 标准库：用于 JSON 读写
import json
# 标准库：用于日志记录
import logging
# 标准库：用于标准输出
import sys
# 标准库：用于路径处理
from pathlib import Path


# 常量：默认底模注册表路径（相对项目根目录）。
DEFAULT_BASE_REGISTRY_PATH = "configs/base_model_registry.json"
# 常量：默认 LoRA 绑定清单路径（相对项目根目录）。
DEFAULT_BINDINGS_PATH = "configs/lora_bindings.json"
# 常量：默认日志路径（相对项目根目录）。
DEFAULT_LOG_PATH = "log/model_assets.log"
# 常量：默认注册表版本号。
DEFAULT_REGISTRY_VERSION = 1
# 常量：默认绑定清单版本号。
DEFAULT_BINDINGS_VERSION = 1
# 常量：支持的系列枚举。
VALID_MODEL_SERIES = ("15", "xl", "fl")
# 常量：支持的底模格式枚举。
VALID_BASE_MODEL_FORMATS = ("single", "diffusers")


class SyncStoreError(RuntimeError):
    """
    功能说明：统一封装存储层异常。
    参数说明：继承 RuntimeError，直接传入错误消息。
    返回值：不适用。
    异常说明：不适用。
    边界条件：用于主流程统一处理。
    """


def resolve_project_root() -> Path:
    """
    功能说明：根据包目录推导项目根目录。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：假设该文件位于 <项目根>/scripts/model_assets 目录。
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(project_root: Path, raw_path: str) -> Path:
    """
    功能说明：解析参数路径为绝对路径。
    参数说明：
    - project_root: 项目根目录。
    - raw_path: 用户输入路径。
    返回值：
    - Path: 绝对路径。
    异常说明：无。
    边界条件：相对路径按项目根拼接。
    """
    candidate = Path(str(raw_path).strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def to_project_relative_path(path: Path, project_root: Path) -> str:
    """
    功能说明：将绝对路径转为项目相对路径。
    参数说明：
    - path: 目标路径。
    - project_root: 项目根目录。
    返回值：
    - str: 项目相对路径（POSIX风格）；若不在项目内则返回绝对路径。
    异常说明：无。
    边界条件：跨盘符或目录越界时回退绝对路径。
    """
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def setup_logger(log_path: Path) -> logging.Logger:
    """
    功能说明：初始化日志器（控制台+文件双写）。
    参数说明：
    - log_path: 日志文件路径。
    返回值：
    - logging.Logger: 初始化后的日志对象。
    异常说明：无。
    边界条件：重复初始化时会清理旧处理器避免重复输出。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("model_assets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def read_json(path: Path) -> dict:
    """
    功能说明：读取 JSON 文件。
    参数说明：
    - path: JSON 文件路径。
    返回值：
    - dict: 解析后的 JSON 对象。
    异常说明：
    - SyncStoreError: 文件不可读或 JSON 非法时抛出。
    边界条件：默认 UTF-8 编码。
    """
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except Exception as error:  # noqa: BLE001
        raise SyncStoreError(f"读取 JSON 失败：{path}，错误：{error}") from error

    if not isinstance(data, dict):
        raise SyncStoreError(f"JSON 顶层结构必须为对象：{path}")
    return data


def write_json(path: Path, data: dict) -> None:
    """
    功能说明：写入 JSON 文件（覆盖写）。
    参数说明：
    - path: JSON 文件路径。
    - data: 待写入对象。
    返回值：无。
    异常说明：
    - SyncStoreError: 写入失败时抛出。
    边界条件：会自动创建父目录。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)
    except Exception as error:  # noqa: BLE001
        raise SyncStoreError(f"写入 JSON 失败：{path}，错误：{error}") from error


def load_or_init_base_registry(path: Path, project_root: Path) -> dict:
    """
    功能说明：加载或初始化底模注册表。
    参数说明：
    - path: 注册表路径。
    - project_root: 项目根目录。
    返回值：
    - dict: 注册表对象。
    异常说明：
    - SyncStoreError: 结构非法时抛出。
    边界条件：缺失时初始化 15/xl/fl 三个默认 key。
    """
    if not path.exists():
        default_data = {
            "version": DEFAULT_REGISTRY_VERSION,
            "base_models": [
                {
                    "key": "sd15_base_default",
                    "series": "15",
                    "format": "diffusers",
                    "path": "models/base_model/15/diffusers/stable-diffusion-v1-5",
                    "type": "directory",
                    "enabled": True,
                    "description": "SD1.5 基础模型目录",
                },
                {
                    "key": "sdxl_base_default",
                    "series": "xl",
                    "format": "diffusers",
                    "path": "models/base_model/xl/diffusers/stable-diffusion-xl-base-1.0",
                    "type": "directory",
                    "enabled": True,
                    "description": "SDXL 基础模型目录",
                },
                {
                    "key": "fl_base_default",
                    "series": "fl",
                    "format": "diffusers",
                    "path": "models/base_model/fl/diffusers",
                    "type": "directory",
                    "enabled": True,
                    "description": "FL 基础模型目录",
                },
            ],
        }
        write_json(path=path, data=default_data)

    data = read_json(path=path)
    base_models = data.get("base_models", [])
    if not isinstance(base_models, list):
        raise SyncStoreError(f"底模注册表字段 base_models 非法（需为数组）：{path}")

    # 目录：确保系列/格式根目录存在，便于后续下载落地。
    for series in VALID_MODEL_SERIES:
        for model_format in VALID_BASE_MODEL_FORMATS:
            (project_root / "models" / "base_model" / series / model_format).mkdir(parents=True, exist_ok=True)

    return data


def load_or_init_lora_bindings(path: Path) -> dict:
    """
    功能说明：加载或初始化 LoRA 绑定清单。
    参数说明：
    - path: 绑定清单路径。
    返回值：
    - dict: 绑定清单对象。
    异常说明：
    - SyncStoreError: 结构非法时抛出。
    边界条件：缺失时初始化空 bindings 数组。
    """
    if not path.exists():
        write_json(
            path=path,
            data={
                "version": DEFAULT_BINDINGS_VERSION,
                "bindings": [],
            },
        )

    data = read_json(path=path)
    bindings = data.get("bindings", [])
    if not isinstance(bindings, list):
        raise SyncStoreError(f"绑定清单字段 bindings 非法（需为数组）：{path}")
    return data


def get_enabled_base_model_candidates(registry_data: dict, model_series: str) -> list[dict]:
    """
    功能说明：获取某系列可用底模候选项。
    参数说明：
    - registry_data: 底模注册表对象。
    - model_series: 目标系列（15/xl/fl）。
    返回值：
    - list[dict]: 候选底模记录列表。
    异常说明：无。
    边界条件：仅返回 enabled=true 且 key/path 非空的记录。
    """
    result: list[dict] = []
    for item in registry_data.get("base_models", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("series", "")).strip() != model_series:
            continue
        if not bool(item.get("enabled", False)):
            continue
        key_text = str(item.get("key", "")).strip()
        path_text = str(item.get("path", "")).strip()
        if (not key_text) or (not path_text):
            continue
        result.append(item)
    return result


def validate_and_resolve_base_model(
    registry_data: dict,
    base_model_key: str,
    model_series: str,
    project_root: Path,
) -> tuple[dict, Path]:
    """
    功能说明：校验底模 key 并解析本地底模路径。
    参数说明：
    - registry_data: 底模注册表对象。
    - base_model_key: 底模 key。
    - model_series: 目标系列。
    - project_root: 项目根目录。
    返回值：
    - tuple[dict, Path]: (匹配记录, 绝对路径)。
    异常说明：
    - SyncStoreError: key 不存在、禁用、系列不一致或路径缺失时抛出。
    边界条件：path 支持相对项目根写法。
    """
    matched: dict | None = None
    for item in registry_data.get("base_models", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("key", "")).strip() == base_model_key:
            matched = item
            break

    if matched is None:
        raise SyncStoreError(f"未找到底模 key：{base_model_key}")

    record_series = str(matched.get("series", "")).strip()
    if record_series != model_series:
        raise SyncStoreError(
            f"底模系列不匹配：key={base_model_key} 的 series={record_series}，当前系列={model_series}"
        )

    if not bool(matched.get("enabled", False)):
        raise SyncStoreError(f"底模 key 已禁用：{base_model_key}")

    record_format = str(matched.get("format", "")).strip()
    if record_format and (record_format not in VALID_BASE_MODEL_FORMATS):
        raise SyncStoreError(
            f"底模 key 的 format 非法：key={base_model_key}，format={record_format}"
        )

    raw_path = str(matched.get("path", "")).strip()
    if not raw_path:
        raise SyncStoreError(f"底模 key 缺少 path 字段：{base_model_key}")

    resolved = resolve_path(project_root=project_root, raw_path=raw_path)
    if not resolved.exists():
        raise SyncStoreError(f"底模路径不存在：key={base_model_key}，path={resolved}")

    return matched, resolved


def upsert_lora_binding(bindings_data: dict, new_record: dict) -> str:
    """
    功能说明：按 binding_name + model_series 唯一键 upsert 绑定记录。
    参数说明：
    - bindings_data: 绑定清单对象。
    - new_record: 新记录对象。
    返回值：
    - str: inserted 或 updated。
    异常说明：无。
    边界条件：若存在匹配项则覆盖整条记录。
    """
    bindings = bindings_data.get("bindings", [])
    target_name = str(new_record.get("binding_name", "")).strip()
    target_series = str(new_record.get("model_series", "")).strip()

    for index, old_item in enumerate(bindings):
        if not isinstance(old_item, dict):
            continue
        old_name = str(old_item.get("binding_name", "")).strip()
        old_series = str(old_item.get("model_series", "")).strip()
        if (old_name == target_name) and (old_series == target_series):
            bindings[index] = new_record
            return "updated"

    bindings.append(new_record)
    return "inserted"


def upsert_base_model(registry_data: dict, new_record: dict) -> str:
    """
    功能说明：按 key 唯一键 upsert 底模记录。
    参数说明：
    - registry_data: 注册表对象。
    - new_record: 新记录对象。
    返回值：
    - str: inserted 或 updated。
    异常说明：无。
    边界条件：若 key 相同则覆盖整条记录。
    """
    base_models = registry_data.get("base_models", [])
    target_key = str(new_record.get("key", "")).strip()

    for index, old_item in enumerate(base_models):
        if not isinstance(old_item, dict):
            continue
        old_key = str(old_item.get("key", "")).strip()
        if old_key == target_key:
            base_models[index] = new_record
            return "updated"

    base_models.append(new_record)
    return "inserted"


def now_iso_seconds() -> str:
    """
    功能说明：返回秒级 ISO 时间字符串。
    参数说明：无。
    返回值：
    - str: 例如 2026-04-16T19:00:00。
    异常说明：无。
    边界条件：使用本地时区时间。
    """
    return datetime.now().isoformat(timespec="seconds")
