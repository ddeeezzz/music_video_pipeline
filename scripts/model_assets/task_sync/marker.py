"""
文件用途：维护任务同步增量标记 _st_sync_done.json（schema v2）。
核心流程：初始化/规范化标记 -> 写入模式期望文件集 -> 逐文件回写处理状态。
输入输出：输入任务与文件处理结果，输出可序列化的标记字典。
依赖说明：仅依赖标准库 datetime/json。
维护说明：catalog 与 executor 必须复用本模块，避免判定口径不一致。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any

from .types import SyncMode


MARKER_FILE_NAME = "_st_sync_done.json"
MARKER_SCHEMA_VERSION = 2


def now_iso() -> str:
    """返回东八区 ISO 时间戳（秒级）。"""
    china_tz = timezone(timedelta(hours=8))
    return datetime.now(china_tz).isoformat(timespec="seconds")


def create_marker_state(*, task_id: str, source_config: str) -> dict[str, Any]:
    """创建空标记。"""
    ts = now_iso()
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "task_id": str(task_id),
        "source_config": str(source_config),
        "updated_at": ts,
        "modes": {},
    }


def normalize_marker_state(
    *,
    raw_marker: Any,
    task_id: str,
    source_config: str,
) -> dict[str, Any]:
    """
    规范化标记结构。

    若远端标记缺失或格式不合法，返回新的空标记。
    """
    if not isinstance(raw_marker, dict):
        return create_marker_state(task_id=task_id, source_config=source_config)

    marker = dict(raw_marker)
    if int(marker.get("schema_version", 0) or 0) != MARKER_SCHEMA_VERSION:
        return create_marker_state(task_id=task_id, source_config=source_config)

    marker["schema_version"] = MARKER_SCHEMA_VERSION
    marker["task_id"] = str(marker.get("task_id") or task_id)
    marker["source_config"] = str(marker.get("source_config") or source_config)
    marker["updated_at"] = str(marker.get("updated_at") or now_iso())

    modes_obj = marker.get("modes")
    if not isinstance(modes_obj, dict):
        modes_obj = {}
    normalized_modes: dict[str, Any] = {}
    for mode_key, mode_value in modes_obj.items():
        if not isinstance(mode_value, dict):
            continue
        normalized_modes[str(mode_key)] = dict(mode_value)
    marker["modes"] = normalized_modes
    return marker


def ensure_mode_state(
    *,
    marker: dict[str, Any],
    mode: SyncMode,
    expected_files: list[str],
    selection_profile: str,
) -> dict[str, Any]:
    """确保 mode 分支存在，并同步期望文件集与 profile。"""
    modes_obj = marker.setdefault("modes", {})
    if not isinstance(modes_obj, dict):
        modes_obj = {}
        marker["modes"] = modes_obj

    mode_key = str(mode)
    mode_obj = modes_obj.get(mode_key)
    if not isinstance(mode_obj, dict):
        mode_obj = {}
        modes_obj[mode_key] = mode_obj

    normalized_expected = list(dict.fromkeys(str(item).strip() for item in expected_files if str(item).strip()))
    mode_obj["selection_profile"] = str(selection_profile).strip()
    mode_obj["expected_files"] = normalized_expected
    files_obj = mode_obj.get("files")
    if not isinstance(files_obj, dict):
        files_obj = {}
        mode_obj["files"] = files_obj
    mode_obj["updated_at"] = now_iso()
    marker["updated_at"] = mode_obj["updated_at"]
    return mode_obj


def mark_file_processed(
    *,
    marker: dict[str, Any],
    mode: SyncMode,
    relative_path: str,
    local_md5: str,
    action: str,
) -> None:
    """记录单文件处理结果（上传或跳过）。"""
    modes_obj = marker.setdefault("modes", {})
    if not isinstance(modes_obj, dict):
        modes_obj = {}
        marker["modes"] = modes_obj

    mode_obj = modes_obj.setdefault(str(mode), {})
    if not isinstance(mode_obj, dict):
        mode_obj = {}
        modes_obj[str(mode)] = mode_obj

    files_obj = mode_obj.setdefault("files", {})
    if not isinstance(files_obj, dict):
        files_obj = {}
        mode_obj["files"] = files_obj

    timestamp = now_iso()
    files_obj[str(relative_path)] = {
        "local_md5": str(local_md5).lower(),
        "status": str(action),
        "updated_at": timestamp,
    }
    mode_obj["updated_at"] = timestamp
    marker["updated_at"] = timestamp


def evaluate_mode_sync(
    *,
    marker: Any,
    mode: SyncMode,
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    判断标记是否覆盖“当前模式”的本地期望文件集。

    返回 (is_synced, reason_when_unsynced)。
    """
    if not isinstance(marker, dict):
        return False, "标记文件内容非法"
    if int(marker.get("schema_version", 0) or 0) != MARKER_SCHEMA_VERSION:
        return False, "标记 schema_version 不匹配（需为 2）"

    modes_obj = marker.get("modes")
    if not isinstance(modes_obj, dict):
        return False, "标记缺少 modes 字段"

    mode_obj = modes_obj.get(str(mode))
    if not isinstance(mode_obj, dict):
        return False, "标记缺少当前模式记录"

    marker_expected = mode_obj.get("expected_files")
    if not isinstance(marker_expected, list):
        return False, "标记缺少 expected_files"
    marker_expected_set = {
        str(item).strip() for item in marker_expected if isinstance(item, str) and str(item).strip()
    }
    expected_set = {str(item).strip() for item in expected_files if str(item).strip()}
    if marker_expected_set != expected_set:
        return (
            False,
            "标记 expected_files 与当前本地文件集不一致",
        )

    files_obj = mode_obj.get("files")
    if not isinstance(files_obj, dict):
        return False, "标记缺少 files 状态"
    done_set = {str(key).strip() for key, value in files_obj.items() if str(key).strip() and isinstance(value, dict)}

    missing = sorted(expected_set - done_set)
    if missing:
        return False, f"标记未覆盖全部文件（缺少 {len(missing)} 个）"
    return True, ""


def dumps_marker(marker: dict[str, Any]) -> str:
    """序列化标记 JSON。"""
    return json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True)
