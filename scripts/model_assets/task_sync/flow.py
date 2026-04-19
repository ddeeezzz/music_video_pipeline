"""
文件用途：提供 task_sync 交互式主流程。
核心流程：发现任务 -> 按模式分组展示 -> 数字选择执行同步 -> 循环返回主界面。
输入输出：输入用户交互，输出同步执行结果与 compare 展示。
依赖说明：依赖 discover/catalog/executor/source_builder/marker/remote。
维护说明：此模块仅做交互编排，不承载 bypy 命令细节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from music_video_pipeline.config import load_config

from .catalog import build_catalog
from .discover import discover_tasks
from .executor import execute_task_sync
from .marker import evaluate_mode_sync
from .remote import BypyRemote, BypyRuntime
from .source_builder import build_source_manifest
from .types import CatalogTask, InspectResult, SyncMode


def _build_runtime_from_config_path(config_path: Path) -> BypyRuntime:
    app_config = load_config(config_path=config_path)
    upload_config = app_config.bypy_upload
    return BypyRuntime(
        bypy_bin=str(upload_config.bypy_bin),
        retry_times=int(upload_config.retry_times),
        timeout_seconds=float(upload_config.timeout_seconds),
        config_dir=Path(str(upload_config.config_dir)).expanduser().resolve(),
        require_auth_file=bool(upload_config.require_auth_file),
    )


def _build_inspector(*, logger: Any):
    client_cache: dict[Path, BypyRemote] = {}

    def _inspect(task, mode: SyncMode) -> InspectResult:
        try:
            app_config = load_config(config_path=task.config_path)
            upload_config = app_config.bypy_upload
            runtime = _build_runtime_from_config_path(task.config_path)
        except Exception as error:  # noqa: BLE001
            return InspectResult(
                remote_dir_exists=False,
                remote_marker_exists=False,
                is_synced=False,
                error_reason=f"配置加载失败：{error}",
            )

        try:
            manifest = build_source_manifest(
                task_id=task.task_id,
                local_task_dir=task.local_task_dir,
                mode=mode,
                selection_profile=str(upload_config.selection_profile),
                source_config=str(task.config_path),
                logger=logger,
            )
            expected_files = [item.relative_path for item in manifest.files]
        except Exception as error:  # noqa: BLE001
            return InspectResult(
                remote_dir_exists=False,
                remote_marker_exists=False,
                is_synced=False,
                error_reason=f"源清单构建失败：{error}",
            )

        cache_key = task.config_path.resolve()
        client = client_cache.get(cache_key)
        if client is None:
            try:
                client = BypyRemote(runtime=runtime, logger=logger)
            except Exception as error:  # noqa: BLE001
                return InspectResult(
                    remote_dir_exists=False,
                    remote_marker_exists=False,
                    is_synced=False,
                    error_reason=f"bypy 运行参数无效：{error}",
                )
            client_cache[cache_key] = client

        try:
            remote_dir_exists, remote_marker_exists = client.inspect_task_remote(
                task_id=task.task_id,
                remote_root="/runs",
            )
            if not remote_dir_exists:
                return InspectResult(
                    remote_dir_exists=False,
                    remote_marker_exists=False,
                    is_synced=False,
                    unsynced_reason="远端目录不存在",
                )
            if not remote_marker_exists:
                return InspectResult(
                    remote_dir_exists=True,
                    remote_marker_exists=False,
                    is_synced=False,
                    unsynced_reason="远端目录存在，但缺少标记",
                )
            marker_obj, marker_reason = client.load_marker_json(remote_dir=f"/runs/{task.task_id}")
            if marker_obj is None:
                return InspectResult(
                    remote_dir_exists=True,
                    remote_marker_exists=True,
                    is_synced=False,
                    unsynced_reason=marker_reason or "远端标记不可用",
                )
            is_synced, reason = evaluate_mode_sync(
                marker=marker_obj,
                mode=mode,
                expected_files=expected_files,
            )
            return InspectResult(
                remote_dir_exists=True,
                remote_marker_exists=True,
                is_synced=is_synced,
                unsynced_reason=("" if is_synced else (reason or "标记未覆盖当前模式全部文件")),
            )
        except Exception as error:  # noqa: BLE001
            return InspectResult(
                remote_dir_exists=False,
                remote_marker_exists=False,
                is_synced=False,
                error_reason=f"远端探测失败：{error}",
            )

    return _inspect


def _render_catalog(*, catalog, mode: SyncMode) -> dict[int, CatalogTask]:
    print("")
    print("任务同步（手动）")
    print(f"当前模式：{'白名单' if mode == 'whitelist' else '全量'}")
    print("远端目录固定：/runs/<task-id>/")
    print("")

    index_map: dict[int, CatalogTask] = {}
    index_counter = 1

    def _render_group(title: str, items: list[CatalogTask], selectable: bool) -> None:
        nonlocal index_counter
        print(title)
        if not items:
            print("  (空)")
            return
        for item in items:
            task = item.task
            suffix = f"（{item.reason}）" if item.reason else ""
            line = (
                f"  [{index_counter}] {task.task_id}"
                f" | updated={task.updated_at or '-'}"
                f" | db={task.db_path}"
                f"{suffix}"
            )
            print(line)
            if selectable:
                index_map[index_counter] = item
            index_counter += 1

    _render_group("[未同步（可同步）]", catalog.unsynced, selectable=True)
    _render_group("[已同步（可重同步）]", catalog.synced, selectable=True)
    _render_group("[本地缺失（不可同步）]", catalog.missing_local, selectable=False)
    _render_group("[配置异常（不可同步）]", catalog.config_error, selectable=False)

    print("")
    print("输入说明：")
    print("  - 输入数字：执行对应任务同步")
    print("  - 输入 m：切换同步模式（白名单/全量）")
    print("  - 输入 r：刷新列表")
    print("  - 输入 q：返回上级菜单")
    return index_map


def _render_compare_paths(*, title: str, paths: list[str], total_count: int, max_items: int = 200) -> None:
    print(f"{title}（{total_count}）:")
    if total_count <= 0:
        print("  (空)")
        return
    display_items = paths[:max_items]
    for path in display_items:
        print(f"  - {path}")
    omitted = total_count - len(display_items)
    if omitted > 0:
        print(f"  ... 其余 {omitted} 条已省略")


def run_task_sync_flow(*, project_root: Path, logger: Any) -> int:
    """task_sync 交互入口。"""
    mode: SyncMode = "whitelist"

    while True:
        tasks = discover_tasks(project_root=project_root)
        inspector = _build_inspector(logger=logger)
        catalog = build_catalog(tasks=tasks, mode=mode, inspector=inspector)
        index_map = _render_catalog(catalog=catalog, mode=mode)

        answer = input("输入序号继续，m/r/q：").strip().lower()
        if answer in {"q", "quit", "exit"}:
            print("已返回上级菜单。")
            return 0
        if answer == "m":
            mode = "full" if mode == "whitelist" else "whitelist"
            continue
        if answer == "r":
            continue
        if not answer.isdigit():
            print(f"输入无效：{answer}")
            continue

        selected = index_map.get(int(answer))
        if selected is None:
            print(f"序号不可用：{answer}")
            continue

        try:
            result = execute_task_sync(
                task=selected,
                mode=mode,
                logger=logger,
            )
            print("")
            print(f"任务 {result.task_id} 同步结束：{'成功' if result.success else '失败'}")
            print(f"远端目录：{result.remote_dir}")
            print(
                "执行摘要："
                f"uploaded={result.uploaded_count}, "
                f"skipped_same={result.skipped_same_count}, "
                f"total={result.total_count}"
            )
            print(
                "compare 摘要（展示）："
                f"same={result.compare.same}, "
                f"different={result.compare.different}, "
                f"local_only={result.compare.local_only}, "
                f"remote_only={result.compare.remote_only}"
            )
            _render_compare_paths(
                title="different 路径",
                paths=result.compare.different_paths,
                total_count=result.compare.different,
            )
            _render_compare_paths(
                title="local_only 路径",
                paths=result.compare.local_only_paths,
                total_count=result.compare.local_only,
            )
            _render_compare_paths(
                title="remote_only 路径",
                paths=result.compare.remote_only_paths,
                total_count=result.compare.remote_only,
            )
            print(result.message)
        except Exception as error:  # noqa: BLE001
            print(f"同步失败：{error}")

        input("按回车返回列表...")
