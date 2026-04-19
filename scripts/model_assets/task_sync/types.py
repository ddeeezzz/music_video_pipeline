"""
文件用途：定义 task_sync 子包共享数据结构。
核心流程：由 discover/catalog/executor/flow 共同引用，避免字典字段漂移。
输入输出：输入任务元信息，输出结构化类型对象。
依赖说明：仅依赖标准库 dataclasses/pathlib。
维护说明：新增字段时需同步 catalog 渲染与执行摘要逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SyncMode = Literal["whitelist", "full"]
TaskGroup = Literal["unsynced", "synced", "missing_local", "config_error"]


@dataclass(frozen=True, slots=True)
class DiscoveredTask:
    """状态库发现的任务记录。"""

    task_id: str
    config_path: Path
    updated_at: str
    db_path: Path
    runs_dir: Path
    local_task_dir: Path


@dataclass(frozen=True, slots=True)
class CatalogTask:
    """带分组与远端状态标记的任务记录。"""

    task: DiscoveredTask
    group: TaskGroup
    remote_dir_exists: bool
    remote_marker_exists: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CatalogResult:
    """任务分组结果。"""

    unsynced: list[CatalogTask]
    synced: list[CatalogTask]
    missing_local: list[CatalogTask]
    config_error: list[CatalogTask]


@dataclass(frozen=True, slots=True)
class SourceFile:
    """单个待同步源文件。"""

    relative_path: str
    local_path: Path


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """任务+模式对应的期望文件清单。"""

    task_id: str
    mode: SyncMode
    selection_profile: str
    source_config: str
    files: list[SourceFile]


@dataclass(frozen=True, slots=True)
class InspectResult:
    """目录探测与标记判定结果。"""

    remote_dir_exists: bool
    remote_marker_exists: bool
    is_synced: bool
    unsynced_reason: str = ""
    error_reason: str = ""


@dataclass(frozen=True, slots=True)
class CompareSummary:
    """compare 结果摘要。"""

    same: int
    different: int
    local_only: int
    remote_only: int
    exit_code: int
    different_paths: list[str] = field(default_factory=list)
    local_only_paths: list[str] = field(default_factory=list)
    remote_only_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncExecutionResult:
    """单次任务同步执行结果。"""

    task_id: str
    mode: SyncMode
    remote_dir: str
    success: bool
    compare: CompareSummary
    message: str
    uploaded_count: int
    skipped_same_count: int
    total_count: int
