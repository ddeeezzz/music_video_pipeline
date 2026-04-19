"""
文件用途：根据本地目录和远端标记对任务进行分组。
核心流程：本地存在性检查 -> 远端目录与 marker/完整性探测 -> 产出四类任务清单。
输入输出：输入发现到的任务与远端探测函数，输出 CatalogResult。
依赖说明：依赖本子包 types。
维护说明：分组语义需与交互菜单文案保持一致。
"""

from __future__ import annotations

from typing import Callable

from .types import CatalogResult, CatalogTask, DiscoveredTask, InspectResult, SyncMode


RemoteInspector = Callable[[DiscoveredTask, SyncMode], InspectResult]


def build_catalog(
    *,
    tasks: list[DiscoveredTask],
    mode: SyncMode,
    inspector: RemoteInspector,
) -> CatalogResult:
    unsynced: list[CatalogTask] = []
    synced: list[CatalogTask] = []
    missing_local: list[CatalogTask] = []
    config_error: list[CatalogTask] = []

    for task in tasks:
        if not task.local_task_dir.exists() or not task.local_task_dir.is_dir():
            missing_local.append(
                CatalogTask(
                    task=task,
                    group="missing_local",
                    remote_dir_exists=False,
                    remote_marker_exists=False,
                    reason="本地任务目录缺失",
                )
            )
            continue

        inspect_result = inspector(task, mode)
        if inspect_result.error_reason:
            config_error.append(
                CatalogTask(
                    task=task,
                    group="config_error",
                    remote_dir_exists=inspect_result.remote_dir_exists,
                    remote_marker_exists=inspect_result.remote_marker_exists,
                    reason=inspect_result.error_reason,
                )
            )
            continue

        if inspect_result.is_synced:
            synced.append(
                CatalogTask(
                    task=task,
                    group="synced",
                    remote_dir_exists=inspect_result.remote_dir_exists,
                    remote_marker_exists=inspect_result.remote_marker_exists,
                )
            )
            continue

        unsynced.append(
            CatalogTask(
                task=task,
                group="unsynced",
                remote_dir_exists=inspect_result.remote_dir_exists,
                remote_marker_exists=inspect_result.remote_marker_exists,
                reason=inspect_result.unsynced_reason or "远端目录不存在或缺少同步标记",
            )
        )

    def _sort_items(items: list[CatalogTask]) -> list[CatalogTask]:
        return sorted(items, key=lambda item: (item.task.updated_at, item.task.task_id), reverse=True)

    return CatalogResult(
        unsynced=_sort_items(unsynced),
        synced=_sort_items(synced),
        missing_local=_sort_items(missing_local),
        config_error=_sort_items(config_error),
    )
