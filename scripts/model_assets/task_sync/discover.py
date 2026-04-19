"""
文件用途：从配置与状态库发现可同步任务。
核心流程：扫描 configs -> 解析 runs_dir -> 读取 tasks 表。
输入输出：输入项目根目录，输出 DiscoveredTask 列表。
依赖说明：依赖标准库 json/sqlite3/pathlib 与本子包 types。
维护说明：发现逻辑仅负责“读”，不包含远端探测与同步执行。
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .types import DiscoveredTask


def _resolve_runs_dir(*, project_root: Path, config_path: Path, runs_dir_text: str) -> Path:
    candidate = Path(str(runs_dir_text).strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def discover_state_dbs(*, project_root: Path) -> list[tuple[Path, Path]]:
    """扫描配置文件并返回可用状态库路径列表（db_path, runs_dir）。"""
    configs_dir = (project_root / "configs").resolve()
    if not configs_dir.exists():
        return []

    discovered: dict[Path, Path] = {}
    for config_path in sorted(configs_dir.rglob("*.json")):
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(raw, dict):
            continue
        paths_obj = raw.get("paths", {})
        if not isinstance(paths_obj, dict):
            continue
        runs_dir_text = str(paths_obj.get("runs_dir", "")).strip()
        if not runs_dir_text:
            continue
        runs_dir = _resolve_runs_dir(project_root=project_root, config_path=config_path, runs_dir_text=runs_dir_text)
        db_path = (runs_dir / "pipeline_state.sqlite3").resolve()
        if not db_path.exists() or not db_path.is_file():
            continue
        discovered[db_path] = runs_dir

    return sorted(discovered.items(), key=lambda item: str(item[0]))


def _resolve_task_config_path(*, config_path_text: str, project_root: Path, db_path: Path) -> Path:
    candidate = Path(str(config_path_text).strip())
    if candidate.is_absolute():
        return candidate.resolve()
    # 兼容：历史数据可能写入相对路径，优先按项目根解析。
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (db_path.parent / candidate).resolve()


def discover_tasks(*, project_root: Path) -> list[DiscoveredTask]:
    """读取所有状态库 tasks 表并组装任务列表。"""
    discovered_tasks: list[DiscoveredTask] = []

    for db_path, runs_dir in discover_state_dbs(project_root=project_root):
        try:
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
        except Exception:  # noqa: BLE001
            continue
        try:
            rows = connection.execute(
                """
                SELECT task_id, config_path, updated_at
                FROM tasks
                ORDER BY updated_at DESC, task_id ASC
                """
            ).fetchall()
        except Exception:  # noqa: BLE001
            connection.close()
            continue
        connection.close()

        for row in rows:
            task_id = str(row["task_id"] or "").strip()
            config_path_text = str(row["config_path"] or "").strip()
            if not task_id or not config_path_text:
                continue
            config_path = _resolve_task_config_path(
                config_path_text=config_path_text,
                project_root=project_root,
                db_path=db_path,
            )
            discovered_tasks.append(
                DiscoveredTask(
                    task_id=task_id,
                    config_path=config_path,
                    updated_at=str(row["updated_at"] or "").strip(),
                    db_path=db_path,
                    runs_dir=runs_dir,
                    local_task_dir=(runs_dir / task_id).resolve(),
                )
            )

    return discovered_tasks
