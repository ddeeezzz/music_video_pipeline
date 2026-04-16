"""
文件用途：执行 LoRA 目录下载与绑定关系写入。
核心流程：读取远端文件 -> 校验 LoRA 文件 -> 下载 -> 更新绑定清单。
输入输出：输入远端目录与底模 key，输出下载与绑定结果摘要。
依赖说明：依赖标准库 pathlib/re/shutil 与本包 bypy/index/store 模块。
维护说明：该模块只处理 LoRA，同步 BaseModel 由专门模块负责。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则解析元信息
import re
# 标准库：用于目录删除
import shutil

try:
    # 包内导入：bypy 客户端
    from .bypy_client import BypyClient
    # 包内导入：文件解析工具
    from .indexer import parse_remote_files
    # 包内导入：存储与校验工具
    from .store import (
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        now_iso_seconds,
        to_project_relative_path,
        upsert_lora_binding,
        validate_and_resolve_base_model,
        write_json,
    )
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from bypy_client import BypyClient
    from indexer import parse_remote_files
    from store import (
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        now_iso_seconds,
        to_project_relative_path,
        upsert_lora_binding,
        validate_and_resolve_base_model,
        write_json,
    )


# 常量：元信息里基础模型字段提取规则。
BASE_MODEL_TEXT_PATTERNS = (
    re.compile(r"基础模型\s*[：:]\s*(.+)", re.IGNORECASE),
    re.compile(r"Base\s*Model\s*[：:]\s*(.+)", re.IGNORECASE),
)


def extract_base_model_text(meta_path: Path) -> str:
    """
    功能说明：从 txt 元信息中提取“基础模型”描述。
    参数说明：
    - meta_path: 本地 txt 文件路径。
    返回值：
    - str: 提取到的文本，未命中时返回空字符串。
    异常说明：无。
    边界条件：读取失败时静默返回空字符串。
    """
    if (not meta_path.exists()) or (meta_path.suffix.lower() != ".txt"):
        return ""

    try:
        content = meta_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""

    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        for pattern in BASE_MODEL_TEXT_PATTERNS:
            matched = pattern.search(text)
            if matched:
                return str(matched.group(1)).strip()
    return ""


def select_lora_and_meta(file_items: list[dict[str, str | int]]) -> tuple[dict, dict | None]:
    """
    功能说明：从远端文件列表中选定 LoRA 与可选 txt。
    参数说明：
    - file_items: 远端文件数组。
    返回值：
    - tuple[dict, dict | None]: (LoRA 文件项, txt 文件项或None)。
    异常说明：
    - RuntimeError: 缺失或存在多个 .safetensors 时抛出。
    边界条件：txt 优先同 stem，否则取字典序第一项。
    """
    lora_candidates = [item for item in file_items if str(item.get("name", "")).lower().endswith(".safetensors")]
    if not lora_candidates:
        raise RuntimeError("远端目录中未找到 .safetensors 文件")
    if len(lora_candidates) > 1:
        names = [str(item.get("name", "")) for item in lora_candidates]
        raise RuntimeError(f"远端目录中存在多个 .safetensors 文件：{names}")

    lora_item = lora_candidates[0]
    txt_candidates = [item for item in file_items if str(item.get("name", "")).lower().endswith(".txt")]
    if not txt_candidates:
        return lora_item, None

    lora_stem = Path(str(lora_item.get("name", ""))).stem
    for candidate in txt_candidates:
        if Path(str(candidate.get("name", ""))).stem == lora_stem:
            return lora_item, candidate

    txt_candidates_sorted = sorted(txt_candidates, key=lambda item: str(item.get("name", "")))
    return lora_item, txt_candidates_sorted[0]


def sync_lora_item(
    project_root: Path,
    logger,
    client: BypyClient,
    option: dict,
    base_model_key: str,
    base_registry_path: Path,
    bindings_path: Path,
) -> dict[str, str]:
    """
    功能说明：同步单个 LoRA 条目并写入绑定记录。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - client: bypy 客户端。
    - option: 菜单选项（含 series/name/remote_dir）。
    - base_model_key: 绑定使用的底模 key。
    - base_registry_path: 底模注册表路径。
    - bindings_path: LoRA 绑定清单路径。
    返回值：
    - dict[str, str]: 执行摘要。
    异常说明：
    - RuntimeError/SyncStoreError: 任意校验或下载失败时抛出。
    边界条件：目标目录存在时会先删除再覆盖下载。
    """
    model_series = str(option.get("series", "")).strip()
    binding_name = str(option.get("name", "")).strip()
    remote_dir = str(option.get("remote_dir", "")).strip()

    if (not model_series) or (not binding_name) or (not remote_dir):
        raise RuntimeError(f"LoRA 菜单项字段不完整：{option}")

    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    _, base_model_path = validate_and_resolve_base_model(
        registry_data=registry_data,
        base_model_key=base_model_key,
        model_series=model_series,
        project_root=project_root,
    )

    list_output = client.list_remote(remote_dir)
    file_items = parse_remote_files(list_output)
    if not file_items:
        raise RuntimeError(f"远端目录中没有可识别文件：{remote_dir}")

    lora_item, meta_item = select_lora_and_meta(file_items)
    lora_name = str(lora_item.get("name", "")).strip()
    meta_name = str(meta_item.get("name", "")).strip() if isinstance(meta_item, dict) else ""

    target_dir = (project_root / "models" / "lora" / model_series / binding_name).resolve()
    if target_dir.exists():
        logger.warning("LoRA 目标目录已存在，执行覆盖删除：%s", target_dir)
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    lora_local_path = (target_dir / lora_name).resolve()
    client.downfile(remote_file=f"{remote_dir}/{lora_name}", local_path=lora_local_path)

    if lora_local_path.stat().st_size <= 0:
        raise RuntimeError(f"下载后的 LoRA 文件为空：{lora_local_path}")

    meta_local_path: Path | None = None
    if meta_name:
        meta_local_path = (target_dir / meta_name).resolve()
        client.downfile(remote_file=f"{remote_dir}/{meta_name}", local_path=meta_local_path)

    bindings_data = load_or_init_lora_bindings(path=bindings_path)
    meta_text = extract_base_model_text(meta_local_path) if isinstance(meta_local_path, Path) else ""

    record = {
        "binding_name": binding_name,
        "model_series": model_series,
        "remote_dir": remote_dir,
        "lora_file": to_project_relative_path(path=lora_local_path, project_root=project_root),
        "meta_file": (
            to_project_relative_path(path=meta_local_path, project_root=project_root)
            if isinstance(meta_local_path, Path)
            else ""
        ),
        "base_model_key": base_model_key,
        "base_model_path": to_project_relative_path(path=base_model_path, project_root=project_root),
        "meta_base_model_text": meta_text,
        "updated_at": now_iso_seconds(),
    }

    action = upsert_lora_binding(bindings_data=bindings_data, new_record=record)
    write_json(path=bindings_path, data=bindings_data)

    logger.info("LoRA 绑定写入完成，action=%s，binding=%s", action, binding_name)
    return {
        "resource": "lora",
        "model_series": model_series,
        "name": binding_name,
        "remote_dir": remote_dir,
        "local_dir": str(target_dir),
        "action": action,
    }
