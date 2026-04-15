"""
文件用途：构建任务监督页面所需的统一快照结构。
核心流程：聚合任务级、模块级与B/C/D链路级状态并计算进度。
输入输出：输入 StateStore 与 task_id，输出可直接渲染的字典快照。
依赖说明：依赖项目内状态库与模块顺序常量。
维护说明：字段命名需保持稳定，避免前端渲染契约漂移。
"""

# 标准库：用于类型提示
from typing import Any

# 项目内模块：模块执行顺序常量
from music_video_pipeline.constants import MODULE_ORDER
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore

# 常量：监督页面固定展示的模块顺序
MONITORED_MODULES = list(MODULE_ORDER)
# 常量：链路状态统计固定枚举
CHAIN_STATES = ["pending", "running", "done", "failed"]


def build_task_monitor_snapshot(state_store: StateStore, task_id: str) -> dict[str, Any]:
    """
    功能说明：构建单任务监督页面快照。
    参数说明：
    - state_store: 任务状态存储对象。
    - task_id: 任务唯一标识。
    返回值：
    - dict[str, Any]: 可直接用于WebSocket推送与前端渲染的快照结构。
    异常说明：无（状态不存在时返回 not_found 快照）。
    边界条件：任务初始化早期允许单元表为空，进度按模块状态兜底。
    """
    task_record = state_store.get_task(task_id=task_id)
    if not task_record:
        return _build_missing_task_snapshot(task_id=task_id)

    module_status_map = state_store.get_module_status_map(task_id=task_id)
    module_overview: dict[str, dict[str, Any]] = {}
    for module_name in MONITORED_MODULES:
        module_overview[module_name] = _build_module_progress_item(
            state_store=state_store,
            task_id=task_id,
            module_name=module_name,
            module_status_map=module_status_map,
        )

    chain_rows = state_store.list_bcd_chain_status(task_id=task_id)
    chain_counts = {state_name: 0 for state_name in CHAIN_STATES}
    for chain_item in chain_rows:
        chain_status = str(chain_item.get("chain_status", "pending"))
        if chain_status in chain_counts:
            chain_counts[chain_status] += 1

    return {
        "task_id": task_id,
        "task_status": str(task_record.get("status", "unknown")),
        "updated_at": str(task_record.get("updated_at", "")),
        "module_overview": module_overview,
        "bcd_chains": chain_rows,
        "chain_counts": chain_counts,
        "output_video_path": str(task_record.get("output_video_path", "")),
    }


def _build_missing_task_snapshot(task_id: str) -> dict[str, Any]:
    """
    功能说明：构建任务不存在时的监督页面快照。
    参数说明：
    - task_id: 任务唯一标识。
    返回值：
    - dict[str, Any]: not_found 状态快照。
    异常说明：无。
    边界条件：模块与链路字段均返回空统计，便于前端统一处理。
    """
    module_overview = {
        module_name: {
            "status": "not_found",
            "progress": 0.0,
            "done": 0,
            "total": 0,
            "error_message": "",
        }
        for module_name in MONITORED_MODULES
    }
    return {
        "task_id": task_id,
        "task_status": "not_found",
        "updated_at": "",
        "module_overview": module_overview,
        "bcd_chains": [],
        "chain_counts": {state_name: 0 for state_name in CHAIN_STATES},
        "output_video_path": "",
    }


def _build_module_progress_item(
    state_store: StateStore,
    task_id: str,
    module_name: str,
    module_status_map: dict[str, str],
) -> dict[str, Any]:
    """
    功能说明：构建单个模块的状态与进度摘要。
    参数说明：
    - state_store: 状态存储对象。
    - task_id: 任务唯一标识。
    - module_name: 模块名（A/B/C/D）。
    - module_status_map: 模块状态映射。
    返回值：
    - dict[str, Any]: 模块进度项。
    异常说明：无。
    边界条件：A 模块按单实例任务计算，B/C/D 按单元计数计算。
    """
    module_status = str(module_status_map.get(module_name, "pending"))
    module_record = state_store.get_module_record(task_id=task_id, module_name=module_name) or {}
    error_message = str(module_record.get("error_message", ""))

    if module_name == "A":
        total_units = 1
        done_units = 1 if module_status == "done" else 0
    else:
        module_summary = state_store.get_module_unit_status_summary(task_id=task_id, module_name=module_name)
        total_units = int(module_summary.get("total_units", 0))
        status_counts = module_summary.get("status_counts", {})
        done_units = int(status_counts.get("done", 0)) if isinstance(status_counts, dict) else 0

    progress = _calc_progress(module_status=module_status, done_units=done_units, total_units=total_units)
    return {
        "status": module_status,
        "progress": progress,
        "done": done_units,
        "total": total_units,
        "error_message": error_message,
    }


def _calc_progress(module_status: str, done_units: int, total_units: int) -> float:
    """
    功能说明：按模块状态与单元计数计算百分比进度。
    参数说明：
    - module_status: 模块状态文本。
    - done_units: 已完成单元数量。
    - total_units: 总单元数量。
    返回值：
    - float: 0~100 的进度百分比。
    异常说明：无。
    边界条件：无单元时若模块已done则进度视为100，否则0。
    """
    if total_units > 0:
        progress = (done_units / total_units) * 100.0
        return round(max(0.0, min(100.0, progress)), 2)
    if module_status == "done":
        return 100.0
    return 0.0
