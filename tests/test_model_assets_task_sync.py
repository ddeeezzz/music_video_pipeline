"""
文件用途：验证 model_assets.task_sync 的分组与逐文件同步行为。
核心流程：构造伪任务与伪远端客户端，断言 catalog 分组和 executor 行为。
输入输出：输入临时路径与 monkeypatch，输出断言结果。
依赖说明：依赖 pytest 与 scripts.model_assets.task_sync 子包。
维护说明：同步口径（marker 判定、逐文件上传）变化时需同步更新本测试。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

from music_video_pipeline.config import BypyUploadConfig

# 兼容 pytest 默认 pythonpath=src：补齐项目根目录，确保可导入 scripts.* 包。
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.model_assets.task_sync import catalog as sync_catalog
from scripts.model_assets.task_sync import executor as sync_executor
from scripts.model_assets.task_sync.types import (
    CatalogTask,
    DiscoveredTask,
    InspectResult,
    SourceFile,
    SourceManifest,
)


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324


def _build_discovered_task(tmp_path: Path, *, task_id: str) -> DiscoveredTask:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    local_task_dir = runs_dir / task_id
    local_task_dir.mkdir(parents=True, exist_ok=True)
    return DiscoveredTask(
        task_id=task_id,
        config_path=(tmp_path / "configs" / "task.json").resolve(),
        updated_at="2026-04-18T12:00:00+08:00",
        db_path=(runs_dir / "pipeline_state.sqlite3").resolve(),
        runs_dir=runs_dir.resolve(),
        local_task_dir=local_task_dir.resolve(),
    )


def test_build_catalog_should_split_into_expected_groups(tmp_path: Path) -> None:
    task_unsynced = _build_discovered_task(tmp_path, task_id="task_unsynced")
    task_synced = _build_discovered_task(tmp_path, task_id="task_synced")
    task_missing_local = _build_discovered_task(tmp_path, task_id="task_missing_local")
    task_config_error = _build_discovered_task(tmp_path, task_id="task_config_error")
    task_missing_local.local_task_dir.rmdir()

    def _inspector(task: DiscoveredTask, mode: str) -> InspectResult:
        _ = mode
        if task.task_id == "task_synced":
            return InspectResult(
                remote_dir_exists=True,
                remote_marker_exists=True,
                is_synced=True,
            )
        if task.task_id == "task_unsynced":
            return InspectResult(
                remote_dir_exists=True,
                remote_marker_exists=True,
                is_synced=False,
                unsynced_reason="标记未覆盖全部文件（缺少 1 个）",
            )
        if task.task_id == "task_config_error":
            return InspectResult(
                remote_dir_exists=False,
                remote_marker_exists=False,
                is_synced=False,
                error_reason="配置加载失败",
            )
        return InspectResult(
            remote_dir_exists=False,
            remote_marker_exists=False,
            is_synced=False,
            unsynced_reason="远端目录不存在",
        )

    result = sync_catalog.build_catalog(
        tasks=[task_unsynced, task_synced, task_missing_local, task_config_error],
        mode="whitelist",
        inspector=_inspector,
    )
    assert [item.task.task_id for item in result.unsynced] == ["task_unsynced"]
    assert [item.task.task_id for item in result.synced] == ["task_synced"]
    assert [item.task.task_id for item in result.missing_local] == ["task_missing_local"]
    assert [item.task.task_id for item in result.config_error] == ["task_config_error"]


class _FakeRemote:
    def __init__(self, *, initial_md5_map: dict[str, str], final_md5_map: dict[str, str]) -> None:
        self.calls: list[str] = []
        self.uploaded_files: list[str] = []
        self.marker_upload_count = 0
        self._initial_md5_map = dict(initial_md5_map)
        self._final_md5_map = dict(final_md5_map)
        self._list_count = 0

    def mkdir_p(self, remote_dir: str) -> None:
        _ = remote_dir
        self.calls.append("mkdir_p")

    def list_remote_file_md5s(self, *, remote_dir: str) -> tuple[bool, dict[str, str]]:
        _ = remote_dir
        self._list_count += 1
        self.calls.append("list_remote_file_md5s")
        if self._list_count == 1:
            return True, dict(self._initial_md5_map)
        return True, dict(self._final_md5_map)

    def load_marker_json(self, *, remote_dir: str):  # noqa: ANN001
        _ = remote_dir
        self.calls.append("load_marker_json")
        return None, "远端目录存在，但缺少标记"

    def mkdir_parents_for_file(self, *, remote_file: str) -> None:
        _ = remote_file
        self.calls.append("mkdir_parents_for_file")

    def upload_file(self, *, local_file: Path, remote_file: str, ondup: str) -> None:
        _ = local_file
        _ = ondup
        self.calls.append("upload_file")
        self.uploaded_files.append(remote_file)

    def upload_marker_json(self, *, remote_dir: str, marker_json: str) -> None:
        _ = remote_dir
        _ = marker_json
        self.calls.append("upload_marker_json")
        self.marker_upload_count += 1


def _build_catalog_task(tmp_path: Path, *, group: str, task_id: str) -> CatalogTask:
    discovered = _build_discovered_task(tmp_path, task_id=task_id)
    return CatalogTask(
        task=discovered,
        group=group,  # type: ignore[arg-type]
        remote_dir_exists=(group == "synced"),
        remote_marker_exists=(group == "synced"),
        reason="",
    )


def test_execute_task_sync_should_use_per_file_skip_and_overwrite(tmp_path: Path, monkeypatch) -> None:
    task = _build_catalog_task(tmp_path, group="unsynced", task_id="task_001")
    file_same = task.task.local_task_dir / "same.txt"
    file_diff = task.task.local_task_dir / "diff.txt"
    file_same.write_text("same-content", encoding="utf-8")
    file_diff.write_text("new-content", encoding="utf-8")

    source_manifest = SourceManifest(
        task_id=task.task.task_id,
        mode="whitelist",
        selection_profile="whitelist_v1",
        source_config=str(task.task.config_path),
        files=[
            SourceFile(relative_path="same.txt", local_path=file_same),
            SourceFile(relative_path="diff.txt", local_path=file_diff),
        ],
    )

    monkeypatch.setattr(
        sync_executor,
        "_load_bypy_upload_config",
        lambda task_config_path: BypyUploadConfig(  # noqa: ARG005
            enabled=True,
            bypy_bin="bypy",
            remote_runs_dir="/runs",
            retry_times=1,
            timeout_seconds=30.0,
            config_dir=str(tmp_path / "bypy_cfg"),
            require_auth_file=False,
            selection_profile="whitelist_v1",
        ),
    )
    monkeypatch.setattr(sync_executor, "build_source_manifest", lambda **kwargs: source_manifest)

    local_same_md5 = _md5_text("same-content")
    local_diff_md5 = _md5_text("new-content")
    fake_remote = _FakeRemote(
        initial_md5_map={
            "same.txt": local_same_md5,
            "diff.txt": _md5_text("old-content"),
            "remote_extra.txt": _md5_text("extra"),
        },
        final_md5_map={
            "same.txt": local_same_md5,
            "diff.txt": local_diff_md5,
            "remote_extra.txt": _md5_text("extra"),
            "_st_sync_done.json": _md5_text("marker"),
        },
    )

    result = sync_executor.execute_task_sync(
        task=task,
        mode="whitelist",
        logger=object(),
        remote_factory=lambda runtime, logger: fake_remote,  # noqa: ARG005
    )

    assert result.success is True
    assert result.uploaded_count == 1
    assert result.skipped_same_count == 1
    assert result.total_count == 2
    assert result.compare.same == 2
    assert result.compare.different == 0
    assert result.compare.local_only == 0
    assert result.compare.remote_only == 1
    assert result.compare.remote_only_paths == ["remote_extra.txt"]
    assert fake_remote.marker_upload_count == 2
    assert fake_remote.uploaded_files == ["/runs/task_001/diff.txt"]
    assert "delete" not in fake_remote.calls


def test_execute_task_sync_should_allow_remote_only_without_fail(tmp_path: Path, monkeypatch) -> None:
    task = _build_catalog_task(tmp_path, group="synced", task_id="task_002")
    file_one = task.task.local_task_dir / "only.txt"
    file_one.write_text("content", encoding="utf-8")
    local_md5 = _md5_text("content")

    source_manifest = SourceManifest(
        task_id=task.task.task_id,
        mode="full",
        selection_profile="whitelist_v1",
        source_config=str(task.task.config_path),
        files=[SourceFile(relative_path="only.txt", local_path=file_one)],
    )
    monkeypatch.setattr(
        sync_executor,
        "_load_bypy_upload_config",
        lambda task_config_path: BypyUploadConfig(  # noqa: ARG005
            enabled=True,
            bypy_bin="bypy",
            remote_runs_dir="/runs",
            retry_times=1,
            timeout_seconds=30.0,
            config_dir=str(tmp_path / "bypy_cfg"),
            require_auth_file=False,
            selection_profile="whitelist_v1",
        ),
    )
    monkeypatch.setattr(sync_executor, "build_source_manifest", lambda **kwargs: source_manifest)

    fake_remote = _FakeRemote(
        initial_md5_map={"only.txt": local_md5, "old.bin": _md5_text("old")},
        final_md5_map={"only.txt": local_md5, "old.bin": _md5_text("old"), "_st_sync_done.json": _md5_text("m")},
    )
    result = sync_executor.execute_task_sync(
        task=task,
        mode="full",
        logger=object(),
        remote_factory=lambda runtime, logger: fake_remote,  # noqa: ARG005
    )
    assert result.success is True
    assert result.compare.remote_only == 1
    assert result.compare.remote_only_paths == ["old.bin"]


def test_execute_task_sync_should_reject_missing_local_task(tmp_path: Path) -> None:
    task = _build_catalog_task(tmp_path, group="missing_local", task_id="task_003")
    with pytest.raises(RuntimeError, match="本地目录缺失"):
        sync_executor.execute_task_sync(
            task=task,
            mode="full",
            logger=object(),
            remote_factory=lambda runtime, logger: _FakeRemote(initial_md5_map={}, final_md5_map={}),  # noqa: ARG005
        )
