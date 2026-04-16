"""
文件用途：提供 LoRA 绑定记录浏览与重绑 key 的核心逻辑。
核心流程：读取绑定清单 -> 构建分段菜单项 -> 校验新 key -> 更新绑定字段。
输入输出：输入绑定数据与目标 key，输出重绑后的摘要信息。
依赖说明：依赖标准库与本包 store 模块。
维护说明：本模块仅修改绑定关系，不下载/删除任何模型文件。
"""

# 标准库：用于路径处理
from pathlib import Path

try:
    # 包内导入：存储与校验能力
    from .store import (
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        now_iso_seconds,
        to_project_relative_path,
        validate_and_resolve_base_model,
    )
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from store import (  # type: ignore
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        now_iso_seconds,
        to_project_relative_path,
        validate_and_resolve_base_model,
    )


# 常量：系列显示顺序。
SERIES_DISPLAY_ORDER = ("15", "xl", "fl")


def build_lora_binding_options(bindings_data: dict) -> list[dict]:
    """
    功能说明：将绑定清单转换为可交互菜单项。
    参数说明：
    - bindings_data: lora_bindings.json 对象。
    返回值：
    - list[dict]: 菜单项数组，包含 index/series/binding_name/current_key/record_index。
    异常说明：无。
    边界条件：仅收录 series 为 15/xl/fl 且 binding_name 非空的记录。
    """
    bindings = bindings_data.get("bindings", [])
    if not isinstance(bindings, list):
        return []

    collected: list[dict] = []
    series_rank = {series: idx for idx, series in enumerate(SERIES_DISPLAY_ORDER)}
    for record_index, item in enumerate(bindings):
        if not isinstance(item, dict):
            continue
        series = str(item.get("model_series", "")).strip()
        binding_name = str(item.get("binding_name", "")).strip()
        if (series not in series_rank) or (not binding_name):
            continue

        collected.append(
            {
                "series": series,
                "binding_name": binding_name,
                "current_key": str(item.get("base_model_key", "")).strip(),
                "record_index": record_index,
            }
        )

    collected.sort(key=lambda item: (series_rank[item["series"]], item["binding_name"].lower()))

    options: list[dict] = []
    for idx, item in enumerate(collected, start=1):
        options.append(
            {
                "index": idx,
                "series": item["series"],
                "binding_name": item["binding_name"],
                "current_key": item["current_key"],
                "record_index": item["record_index"],
            }
        )
    return options


def get_rebind_candidates(project_root: Path, base_registry_path: Path, model_series: str) -> list[dict]:
    """
    功能说明：获取指定系列可用的底模 key 候选列表。
    参数说明：
    - project_root: 项目根目录。
    - base_registry_path: 底模注册表路径。
    - model_series: 目标系列（15/xl/fl）。
    返回值：
    - list[dict]: 候选底模记录数组。
    异常说明：无。
    边界条件：仅返回 enabled=true 且 key/path 完整的项。
    """
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    return get_enabled_base_model_candidates(registry_data=registry_data, model_series=model_series)


def apply_lora_rebind(
    project_root: Path,
    bindings_data: dict,
    option: dict,
    base_registry_path: Path,
    new_base_model_key: str,
) -> dict[str, str]:
    """
    功能说明：对单条 LoRA 绑定执行重绑 key。
    参数说明：
    - project_root: 项目根目录。
    - bindings_data: lora_bindings.json 对象（将被原地更新）。
    - option: 菜单项对象（需包含 record_index/series/binding_name）。
    - base_registry_path: 底模注册表路径。
    - new_base_model_key: 新目标底模 key。
    返回值：
    - dict[str, str]: 重绑摘要信息。
    异常说明：
    - RuntimeError: 菜单项无效或记录索引越界时抛出。
    边界条件：仅更新 base_model_key/base_model_path/updated_at 字段。
    """
    bindings = bindings_data.get("bindings", [])
    if not isinstance(bindings, list):
        raise RuntimeError("绑定清单结构非法：缺少 bindings 数组")

    record_index = int(option.get("record_index", -1))
    model_series = str(option.get("series", "")).strip()
    binding_name = str(option.get("binding_name", "")).strip()
    if record_index < 0 or record_index >= len(bindings):
        raise RuntimeError(f"绑定记录索引越界：{record_index}")

    record = bindings[record_index]
    if not isinstance(record, dict):
        raise RuntimeError(f"绑定记录结构非法：index={record_index}")

    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    _, resolved_base_path = validate_and_resolve_base_model(
        registry_data=registry_data,
        base_model_key=new_base_model_key,
        model_series=model_series,
        project_root=project_root,
    )

    old_key = str(record.get("base_model_key", "")).strip()
    record["base_model_key"] = new_base_model_key
    record["base_model_path"] = to_project_relative_path(path=resolved_base_path, project_root=project_root)
    record["updated_at"] = now_iso_seconds()

    return {
        "binding_name": binding_name,
        "model_series": model_series,
        "old_base_model_key": old_key,
        "new_base_model_key": new_base_model_key,
        "base_model_path": str(resolved_base_path),
    }
