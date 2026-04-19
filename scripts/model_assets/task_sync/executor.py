"""
文件用途：执行单个任务的手动同步（逐文件 MD5 跳过/覆盖 + 增量 marker）。
核心流程：加载任务配置 -> 构建源清单 -> 按文件比较并上传 -> compare 展示。
输入输出：输入 CatalogTask 与模式，输出 SyncExecutionResult。
依赖说明：依赖配置加载器、marker/source_builder/remote 与标准库 hashlib。
维护说明：执行顺序必须与产品约定一致（每处理 1 文件即上传 marker）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from music_video_pipeline.config import BypyUploadConfig, load_config

from .marker import (
    MARKER_FILE_NAME,
    dumps_marker,
    ensure_mode_state,
    mark_file_processed,
    normalize_marker_state,
)
from .remote import BypyRemote, BypyRuntime, RemoteOperationError
from .source_builder import build_source_manifest
from .types import CatalogTask, CompareSummary, SyncExecutionResult, SyncMode


REMOTE_ROOT = "/runs"


def _load_bypy_upload_config(*, task_config_path: Path) -> BypyUploadConfig:
    app_config = load_config(config_path=task_config_path)
    return app_config.bypy_upload


def _build_runtime(*, upload_config: BypyUploadConfig) -> BypyRuntime:
    return BypyRuntime(
        bypy_bin=str(upload_config.bypy_bin),
        retry_times=int(upload_config.retry_times),
        timeout_seconds=float(upload_config.timeout_seconds),
        config_dir=Path(str(upload_config.config_dir)).expanduser().resolve(),
        require_auth_file=bool(upload_config.require_auth_file),
    )


def _compute_file_md5(local_file: Path) -> str:
    md5_obj = hashlib.md5()  # noqa: S324
    with local_file.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            md5_obj.update(chunk)
    return md5_obj.hexdigest().lower()


def _build_compare_summary(
    *,
    local_md5_map: dict[str, str],
    remote_md5_map: dict[str, str],
) -> CompareSummary:
    local_paths = set(local_md5_map.keys())
    remote_paths = {path for path in remote_md5_map.keys() if path != MARKER_FILE_NAME}

    same_paths: list[str] = []
    different_paths: list[str] = []
    local_only_paths: list[str] = []
    remote_only_paths: list[str] = []

    for path in sorted(local_paths):
        remote_md5 = str(remote_md5_map.get(path, "")).strip().lower()
        if not remote_md5:
            local_only_paths.append(path)
            continue
        if remote_md5 == str(local_md5_map[path]).strip().lower():
            same_paths.append(path)
            continue
        different_paths.append(path)

    for path in sorted(remote_paths - local_paths):
        remote_only_paths.append(path)

    exit_code = 0 if not different_paths and not local_only_paths and not remote_only_paths else 1
    return CompareSummary(
        same=len(same_paths),
        different=len(different_paths),
        local_only=len(local_only_paths),
        remote_only=len(remote_only_paths),
        exit_code=exit_code,
        different_paths=different_paths,
        local_only_paths=local_only_paths,
        remote_only_paths=remote_only_paths,
    )


def execute_task_sync(
    *,
    task: CatalogTask,
    mode: SyncMode,
    logger: Any,
    remote_factory: Callable[[BypyRuntime, Any], BypyRemote] = BypyRemote,
) -> SyncExecutionResult:
    """执行单任务同步并返回结构化结果。"""
    if task.group == "missing_local":
        raise RuntimeError(f"任务本地目录缺失，无法同步：{task.task.task_id}")
    if task.group == "config_error":
        raise RuntimeError(f"任务配置异常，无法同步：{task.task.task_id}，原因={task.reason}")

    upload_config = _load_bypy_upload_config(task_config_path=task.task.config_path)
    runtime = _build_runtime(upload_config=upload_config)
    remote_client = remote_factory(runtime, logger)
    remote_dir = f"{REMOTE_ROOT}/{task.task.task_id}"

    try:
        manifest = build_source_manifest(
            task_id=task.task.task_id,
            local_task_dir=task.task.local_task_dir,
            mode=mode,
            selection_profile=str(upload_config.selection_profile),
            source_config=str(task.task.config_path),
            logger=logger,
        )
        expected_files = [item.relative_path for item in manifest.files]

        remote_client.mkdir_p(remote_dir)
        _exists, remote_md5_map = remote_client.list_remote_file_md5s(remote_dir=remote_dir)
        remote_marker, _marker_reason = remote_client.load_marker_json(remote_dir=remote_dir)

        marker = normalize_marker_state(
            raw_marker=remote_marker,
            task_id=task.task.task_id,
            source_config=str(task.task.config_path),
        )
        ensure_mode_state(
            marker=marker,
            mode=mode,
            expected_files=expected_files,
            selection_profile=manifest.selection_profile,
        )

        uploaded_count = 0
        skipped_same_count = 0
        local_md5_map: dict[str, str] = {}

        if not manifest.files:
            remote_client.upload_marker_json(
                remote_dir=remote_dir,
                marker_json=dumps_marker(marker),
            )

        for source_file in manifest.files:
            local_md5 = _compute_file_md5(source_file.local_path)
            local_md5_map[source_file.relative_path] = local_md5
            remote_md5 = str(remote_md5_map.get(source_file.relative_path, "")).strip().lower()

            if remote_md5 and remote_md5 == local_md5:
                action = "skipped_same"
                skipped_same_count += 1
            else:
                action = "uploaded"
                remote_file = f"{remote_dir}/{source_file.relative_path}"
                remote_client.mkdir_parents_for_file(remote_file=remote_file)
                remote_client.upload_file(
                    local_file=source_file.local_path,
                    remote_file=remote_file,
                    ondup="overwrite",
                )
                remote_md5_map[source_file.relative_path] = local_md5
                uploaded_count += 1

            mark_file_processed(
                marker=marker,
                mode=mode,
                relative_path=source_file.relative_path,
                local_md5=local_md5,
                action=action,
            )
            remote_client.upload_marker_json(
                remote_dir=remote_dir,
                marker_json=dumps_marker(marker),
            )

        remote_exists_after, remote_md5_map_after = remote_client.list_remote_file_md5s(remote_dir=remote_dir)
        if not remote_exists_after:
            remote_md5_map_after = dict(remote_md5_map)

        compare_summary = _build_compare_summary(
            local_md5_map=local_md5_map,
            remote_md5_map=remote_md5_map_after,
        )
        message = (
            "同步完成：按逐文件 MD5 规则执行上传/跳过；compare 结果仅展示，不做失败门禁。"
        )
        return SyncExecutionResult(
            task_id=task.task.task_id,
            mode=mode,
            remote_dir=remote_dir,
            success=True,
            compare=compare_summary,
            message=message,
            uploaded_count=uploaded_count,
            skipped_same_count=skipped_same_count,
            total_count=len(manifest.files),
        )
    except RemoteOperationError as error:
        raise RuntimeError(str(error)) from error
