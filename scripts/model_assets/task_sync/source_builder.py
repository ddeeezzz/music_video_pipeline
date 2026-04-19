"""
文件用途：构建任务同步期望文件清单（白名单/全量）。
核心流程：按模式收集文件 -> 生成相对路径与绝对路径映射。
输入输出：输入 task_id/本地目录/模式，输出 SourceManifest。
依赖说明：依赖标准库 pathlib 与 upload.staging 的白名单规则。
维护说明：catalog 与 executor 必须共用此清单构建口径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from music_video_pipeline.upload import staging as upload_staging

from .marker import MARKER_FILE_NAME
from .types import SourceFile, SourceManifest, SyncMode


def _to_relative_path(*, file_path: Path, root_dir: Path) -> str:
    return str(file_path.relative_to(root_dir)).replace("\\", "/")


def _iter_all_files(*, root_dir: Path) -> list[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    return sorted(
        [path.resolve() for path in root_dir.rglob("*") if path.is_file()],
        key=lambda path: _to_relative_path(file_path=path, root_dir=root_dir),
    )


def _collect_whitelist_files(
    *,
    task_dir: Path,
    selection_profile: str,
) -> list[Path]:
    normalized_profile = (
        str(selection_profile).strip() or upload_staging.UPLOAD_SELECTION_PROFILE_WHITELIST_V1
    )
    collector_map = {
        upload_staging.UPLOAD_SELECTION_PROFILE_WHITELIST_V1: "_collect_whitelist_files_v1",
        upload_staging.UPLOAD_SELECTION_PROFILE_MODULE_A_V1: "_collect_module_a_whitelist_files_v1",
        upload_staging.UPLOAD_SELECTION_PROFILE_MODULE_B_V1: "_collect_module_b_whitelist_files_v1",
        upload_staging.UPLOAD_SELECTION_PROFILE_MODULE_C_V1: "_collect_module_c_whitelist_files_v1",
        upload_staging.UPLOAD_SELECTION_PROFILE_MODULE_D_V1: "_collect_module_d_whitelist_files_v1",
    }
    collector_name = collector_map.get(normalized_profile)
    if not collector_name:
        raise RuntimeError(f"不支持的 selection_profile：{normalized_profile}")
    collector = getattr(upload_staging, collector_name, None)
    if collector is None:
        raise RuntimeError(f"白名单收集器不存在：{collector_name}")
    files = collector(task_dir=task_dir)
    return sorted(
        [path.resolve() for path in files if path.is_file()],
        key=lambda path: _to_relative_path(file_path=path, root_dir=task_dir),
    )


def build_source_manifest(
    *,
    task_id: str,
    local_task_dir: Path,
    mode: SyncMode,
    selection_profile: str,
    source_config: str,
    logger: Any,
) -> SourceManifest:
    """构建同步源清单。"""
    if not local_task_dir.exists() or not local_task_dir.is_dir():
        raise RuntimeError(f"本地任务目录不存在：{local_task_dir}")

    if mode == "whitelist":
        selected_files = _collect_whitelist_files(
            task_dir=local_task_dir,
            selection_profile=selection_profile,
        )
    else:
        selected_files = _iter_all_files(root_dir=local_task_dir)

    source_files: list[SourceFile] = []
    for file_path in selected_files:
        relative_path = _to_relative_path(file_path=file_path, root_dir=local_task_dir)
        if relative_path == MARKER_FILE_NAME:
            continue
        source_files.append(SourceFile(relative_path=relative_path, local_path=file_path))

    logger.info(
        "任务同步源清单已生成，task_id=%s，mode=%s，selection_profile=%s，file_count=%s",
        task_id,
        mode,
        str(selection_profile).strip() or upload_staging.UPLOAD_SELECTION_PROFILE_WHITELIST_V1,
        len(source_files),
    )
    return SourceManifest(
        task_id=task_id,
        mode=mode,
        selection_profile=str(selection_profile).strip() or upload_staging.UPLOAD_SELECTION_PROFILE_WHITELIST_V1,
        source_config=source_config,
        files=source_files,
    )
