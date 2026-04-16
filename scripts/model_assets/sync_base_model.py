""" 
文件用途：执行 BaseModel 目录下载并更新底模注册表。
核心流程：下载远端目录 -> 校验目录非空 -> upsert 注册表。
输入输出：输入远端 base_model 目录，输出下载结果与注册表写入动作。
依赖说明：依赖标准库 pathlib/re/hashlib/shutil/unicodedata 与本包 bypy/store。
维护说明：该模块只处理基础模型，不处理 LoRA 绑定。
"""

# 标准库：用于哈希生成
import hashlib
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则匹配
import re
# 标准库：用于目录删除
import shutil
# 标准库：用于 Unicode 归一化
import unicodedata

try:
    # 包内导入：bypy 客户端
    from .bypy_client import BypyClient
    # 包内导入：存储工具
    from .store import (
        load_or_init_base_registry,
        to_project_relative_path,
        upsert_base_model,
        write_json,
    )
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from bypy_client import BypyClient
    from store import (
        load_or_init_base_registry,
        to_project_relative_path,
        upsert_base_model,
        write_json,
    )


VALID_BASE_MODEL_FORMATS = ("single", "diffusers")


def slugify_model_name(model_name: str) -> str:
    """
    功能说明：将模型目录名转换为 key 可用 slug。
    参数说明：
    - model_name: 原始模型目录名。
    返回值：
    - str: 仅含 a-z0-9_ 的 slug；若无法生成则返回空字符串。
    异常说明：无。
    边界条件：中文或特殊字符可能导致空 slug。
    """
    normalized = unicodedata.normalize("NFKD", model_name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return slug


def build_base_model_key(model_series: str, model_format: str, model_name: str) -> str:
    """
    功能说明：按规则生成基础模型 key。
    参数说明：
    - model_series: 系列（15/xl/fl）。
    - model_format: 模型格式（single/diffusers）。
    - model_name: 模型目录名。
    返回值：
    - str: key 字符串。
    异常说明：无。
    边界条件：slug 为空时使用哈希后缀兜底。
    """
    slug = slugify_model_name(model_name)
    if slug:
        return f"base_{model_series}_{model_format}_{slug}"
    hash8 = hashlib.sha1(f"{model_format}:{model_name}".encode("utf-8")).hexdigest()[:8]
    return f"base_{model_series}_{model_format}_{hash8}"


def has_any_file(target_dir: Path) -> bool:
    """
    功能说明：判断目录中是否存在任意文件。
    参数说明：
    - target_dir: 目标目录。
    返回值：
    - bool: 存在文件返回 True。
    异常说明：无。
    边界条件：仅统计文件，不含纯空目录。
    """
    if not target_dir.exists():
        return False
    for path in target_dir.rglob("*"):
        if path.is_file():
            return True
    return False


def sync_base_model_item(
    project_root: Path,
    logger,
    client: BypyClient,
    option: dict,
    base_registry_path: Path,
) -> dict[str, str]:
    """
    功能说明：同步单个 BaseModel 条目并更新底模注册表。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - client: bypy 客户端。
    - option: 菜单选项（含 series/format/name/remote_dir）。
    - base_registry_path: 底模注册表路径。
    返回值：
    - dict[str, str]: 执行摘要。
    异常说明：
    - RuntimeError: 下载失败或目录为空时抛出。
    边界条件：目录存在时先删除再覆盖下载。
    """
    model_series = str(option.get("series", "")).strip()
    model_format = str(option.get("format", "")).strip()
    model_name = str(option.get("name", "")).strip()
    remote_dir = str(option.get("remote_dir", "")).strip()

    if (not model_series) or (not model_format) or (not model_name) or (not remote_dir):
        raise RuntimeError(f"BaseModel 菜单项字段不完整：{option}")
    if model_format not in VALID_BASE_MODEL_FORMATS:
        raise RuntimeError(f"BaseModel 格式非法：{model_format}，仅支持 single/diffusers")

    local_dir = (project_root / "models" / "base_model" / model_series / model_format / model_name).resolve()
    if local_dir.exists():
        logger.warning("BaseModel 目标目录已存在，执行覆盖删除：%s", local_dir)
        shutil.rmtree(local_dir)

    client.downdir(remote_dir=remote_dir, local_dir=local_dir)

    if not has_any_file(local_dir):
        raise RuntimeError(f"下载完成但目录内无文件，已拒绝写入注册表：{local_dir}")

    key_text = build_base_model_key(model_series=model_series, model_format=model_format, model_name=model_name)
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    record = {
        "key": key_text,
        "series": model_series,
        "format": model_format,
        "path": to_project_relative_path(path=local_dir, project_root=project_root),
        "type": "directory",
        "enabled": True,
        "description": f"Bypy 同步基础模型目录：{model_format}/{model_name}",
    }

    action = upsert_base_model(registry_data=registry_data, new_record=record)
    write_json(path=base_registry_path, data=registry_data)

    logger.info("BaseModel 注册表写入完成，action=%s，key=%s", action, key_text)
    return {
        "resource": "base_model",
        "model_series": model_series,
        "model_format": model_format,
        "name": model_name,
        "remote_dir": remote_dir,
        "local_dir": str(local_dir),
        "key": key_text,
        "action": action,
    }
