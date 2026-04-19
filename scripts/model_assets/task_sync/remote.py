"""
文件用途：封装 task_sync 所需 bypy 远端操作。
核心流程：统一命令构建与错误处理，提供远端探测、递归 MD5 列表、文件上传与标记读写。
输入输出：输入远端路径与本地文件，输出结构化结果。
依赖说明：依赖标准库 subprocess/selectors/tempfile/json/pathlib。
维护说明：远端根目录由上层固定为 /runs，本模块不读取历史 remote_runs_dir。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import tempfile
import time
from typing import Any

try:
    from ..indexer import parse_remote_dirs
except ImportError:
    from indexer import parse_remote_dirs  # type: ignore

from .marker import MARKER_FILE_NAME


_MISSING_HINTS = (
    "not found",
    "not exists",
    "no such",
    "doesn't exist",
    "path does not exist",
    "nofile",
)
_HEARTBEAT_INTERVAL_SECONDS = 15.0
_SELECT_POLL_SECONDS = 1.0
_MIN_COMMAND_TIMEOUT_SECONDS = 120.0
_TIMEOUT_GRACE_SECONDS = 300.0
_OUTER_RETRY_MAX = 2
_OUTER_RETRY_WAIT_SECONDS = 8.0
_FILE_LINE_PATTERN = re.compile(
    r"^F\s+(.+?)\s+\d+\s+\d{4}-\d{2}-\d{2},\s+\d{2}:\d{2}:\d{2}\s+([0-9a-fA-F]+)\s*$"
)
_COMPARE_STATS_PATTERN = re.compile(r"^(Same|Different|Local only|Remote only):\s*(\d+)\s*$")


@dataclass(frozen=True, slots=True)
class BypyRuntime:
    """bypy 运行时参数。"""

    bypy_bin: str
    retry_times: int
    timeout_seconds: float
    config_dir: Path
    require_auth_file: bool


@dataclass(frozen=True, slots=True)
class CompareResult:
    """compare 输出摘要。"""

    exit_code: int
    same: int
    different: int
    local_only: int
    remote_only: int
    stdout: str
    stderr: str


class RemoteOperationError(RuntimeError):
    """远端操作失败异常。"""


def _normalize_remote_path(path_text: str) -> str:
    text = str(path_text).strip().replace("\\", "/")
    if not text:
        return "/"
    if not text.startswith("/"):
        text = f"/{text}"
    while "//" in text:
        text = text.replace("//", "/")
    if len(text) > 1 and text.endswith("/"):
        text = text.rstrip("/")
    return text


def _normalize_rel_path(path_text: str) -> str:
    text = str(path_text).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.strip("/")


def _parse_remote_files_with_md5(list_output: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for line in str(list_output).splitlines():
        text = line.strip()
        if not text.startswith("F "):
            continue
        matched = _FILE_LINE_PATTERN.match(text)
        if not matched:
            continue
        name = str(matched.group(1)).strip()
        md5_text = str(matched.group(2)).strip().lower()
        if name and md5_text:
            result.append((name, md5_text))
    return result


class BypyRemote:
    """task_sync bypy 远端操作客户端。"""

    def __init__(self, runtime: BypyRuntime, logger: Any) -> None:
        self.runtime = runtime
        self._logger = logger
        self._mkdir_cache: set[str] = set()
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        bypy_bin_text = str(self.runtime.bypy_bin).strip()
        if not bypy_bin_text:
            raise RemoteOperationError("bypy_bin 不能为空。")

        if "/" in bypy_bin_text:
            bypy_path = Path(bypy_bin_text).expanduser().resolve()
            if not bypy_path.exists():
                raise RemoteOperationError(f"未找到 bypy 可执行文件：{bypy_path}")
        else:
            if shutil.which(bypy_bin_text) is None:
                raise RemoteOperationError(f"未找到 bypy 可执行命令：{bypy_bin_text}")

        if self.runtime.require_auth_file:
            auth_file = self.runtime.config_dir.expanduser().resolve() / "bypy.json"
            if not auth_file.exists() or not auth_file.is_file():
                raise RemoteOperationError(f"未找到 bypy 鉴权文件：{auth_file}")

    def _base_command(self) -> list[str]:
        return [
            str(self.runtime.bypy_bin),
            "--retry",
            str(max(0, int(self.runtime.retry_times))),
            "--timeout",
            str(int(max(1.0, float(self.runtime.timeout_seconds)))),
            "--config-dir",
            str(self.runtime.config_dir.expanduser().resolve()),
        ]

    def _run(self, args: list[str], *, allow_missing: bool = False) -> tuple[bool, str, str, int]:
        command = [*self._base_command(), *args]
        self._logger.info("执行 bypy 命令：%s", " ".join(command))
        action = str(args[0]).strip().lower() if args else ""
        should_echo_line = action not in {"list", "downfile"}

        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RemoteOperationError(f"bypy 命令执行失败：无法捕获输出管道：{' '.join(command)}")

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, data="stdout")
        selector.register(process.stderr, selectors.EVENT_READ, data="stderr")

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        start_time = time.monotonic()
        last_heartbeat = start_time
        hard_timeout_seconds = max(
            _MIN_COMMAND_TIMEOUT_SECONDS,
            float(self.runtime.timeout_seconds) + _TIMEOUT_GRACE_SECONDS,
        )

        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed > hard_timeout_seconds:
                process.kill()
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_chunks.append(str(remaining_stdout))
                if remaining_stderr:
                    stderr_chunks.append(str(remaining_stderr))
                raise RemoteOperationError(
                    f"bypy 命令超时（>{int(hard_timeout_seconds)}s）：{' '.join(command)}"
                )

            events = selector.select(timeout=_SELECT_POLL_SECONDS)
            if not events:
                if process.poll() is None:
                    if now - last_heartbeat >= _HEARTBEAT_INTERVAL_SECONDS:
                        self._logger.info("bypy 命令仍在执行，已耗时 %ss：%s", int(elapsed), " ".join(command))
                        last_heartbeat = now
                    continue
                break

            for key, _ in events:
                stream = key.fileobj
                line = stream.readline()
                if line == "":
                    try:
                        selector.unregister(stream)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                text_line = str(line)
                if key.data == "stdout":
                    stdout_chunks.append(text_line)
                    if should_echo_line and text_line.strip():
                        self._logger.info("[bypy] %s", text_line.rstrip())
                else:
                    stderr_chunks.append(text_line)
                    if should_echo_line and text_line.strip():
                        self._logger.info("[bypy][stderr] %s", text_line.rstrip())

        remaining_stdout = process.stdout.read()
        remaining_stderr = process.stderr.read()
        if remaining_stdout:
            stdout_chunks.append(str(remaining_stdout))
        if remaining_stderr:
            stderr_chunks.append(str(remaining_stderr))

        return_code = int(process.wait())
        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        if return_code == 0:
            return True, stdout_text, stderr_text, return_code

        merged_lower = f"{stdout_text}\n{stderr_text}".lower()
        if allow_missing and any(hint in merged_lower for hint in _MISSING_HINTS):
            return False, stdout_text, stderr_text, return_code

        raise RemoteOperationError(
            f"bypy 命令失败（exit_code={return_code}）：{' '.join(command)}\n"
            f"stdout={stdout_text.strip() or '<empty>'}\n"
            f"stderr={stderr_text.strip() or '<empty>'}"
        )

    def _run_with_slice_retry(self, args: list[str], *, allow_missing: bool = False) -> tuple[bool, str, str, int]:
        total_attempts = _OUTER_RETRY_MAX + 1
        for attempt_index in range(1, total_attempts + 1):
            try:
                return self._run(args, allow_missing=allow_missing)
            except RemoteOperationError as error:
                message_lower = str(error).lower()
                is_slice_md5_mismatch = "slice md5 mismatch" in message_lower
                if (not is_slice_md5_mismatch) or attempt_index >= total_attempts:
                    raise
                self._logger.warning(
                    "检测到 Slice MD5 mismatch，准备执行外层重试 %s/%s，等待 %ss 后重试。",
                    attempt_index,
                    _OUTER_RETRY_MAX,
                    int(_OUTER_RETRY_WAIT_SECONDS),
                )
                time.sleep(_OUTER_RETRY_WAIT_SECONDS)
        raise RemoteOperationError("bypy 命令外层重试失败。")

    def inspect_task_remote(self, *, task_id: str, remote_root: str = "/runs") -> tuple[bool, bool]:
        remote_task_dir = _normalize_remote_path(f"{remote_root}/{task_id}")
        exists, stdout_text, _stderr_text, _ = self._run(["list", remote_task_dir], allow_missing=True)
        if not exists:
            return False, False
        marker_exists = False
        for file_name, _md5 in _parse_remote_files_with_md5(stdout_text):
            if str(file_name).strip() == MARKER_FILE_NAME:
                marker_exists = True
                break
        if not marker_exists and MARKER_FILE_NAME in stdout_text:
            marker_exists = True
        return True, marker_exists

    def mkdir(self, remote_dir: str) -> None:
        normalized_dir = _normalize_remote_path(remote_dir)
        try:
            self._run(["mkdir", normalized_dir], allow_missing=False)
        except RemoteOperationError as error:
            message = str(error).lower()
            if "exist" in message or "already" in message:
                self._logger.info("远端目录已存在，跳过 mkdir：%s", normalized_dir)
            else:
                raise
        self._mkdir_cache.add(normalized_dir)

    def mkdir_p(self, remote_dir: str) -> None:
        normalized_dir = _normalize_remote_path(remote_dir)
        if normalized_dir in {"/", ""}:
            return
        segments = [segment for segment in normalized_dir.strip("/").split("/") if segment]
        current = ""
        for segment in segments:
            current = f"{current}/{segment}" if current else f"/{segment}"
            if current in self._mkdir_cache:
                continue
            self.mkdir(current)

    def mkdir_parents_for_file(self, remote_file: str) -> None:
        normalized_file = _normalize_remote_path(remote_file)
        if "/" not in normalized_file.strip("/"):
            return
        parent_dir = normalized_file.rsplit("/", 1)[0]
        self.mkdir_p(parent_dir)

    def delete(self, remote_dir: str) -> None:
        normalized_dir = _normalize_remote_path(remote_dir)
        self._run(["delete", normalized_dir], allow_missing=True)

    def upload_file(self, *, local_file: Path, remote_file: str, ondup: str = "overwrite") -> None:
        normalized_remote_file = _normalize_remote_path(remote_file)
        self._run_with_slice_retry(
            ["upload", str(local_file), normalized_remote_file, str(ondup)],
            allow_missing=False,
        )

    def upload_marker_json(self, *, remote_dir: str, marker_json: str) -> None:
        normalized_remote_dir = _normalize_remote_path(remote_dir)
        marker_remote_path = f"{normalized_remote_dir}/{MARKER_FILE_NAME}"
        with tempfile.TemporaryDirectory(prefix="task_sync_marker_upload_") as tmp_dir:
            local_marker_path = Path(tmp_dir) / MARKER_FILE_NAME
            local_marker_path.write_text(marker_json, encoding="utf-8")
            self.upload_file(
                local_file=local_marker_path,
                remote_file=marker_remote_path,
                ondup="overwrite",
            )

    def load_marker_json(self, *, remote_dir: str) -> tuple[dict[str, Any] | None, str]:
        normalized_remote_dir = _normalize_remote_path(remote_dir)
        marker_remote_path = f"{normalized_remote_dir}/{MARKER_FILE_NAME}"
        with tempfile.TemporaryDirectory(prefix="task_sync_marker_download_") as tmp_dir:
            local_marker_path = Path(tmp_dir) / MARKER_FILE_NAME
            exists, _stdout_text, _stderr_text, _ = self._run(
                ["downfile", marker_remote_path, str(local_marker_path)],
                allow_missing=True,
            )
            if not exists:
                return None, "远端目录存在，但缺少标记"
            try:
                marker_text = local_marker_path.read_text(encoding="utf-8")
            except Exception as error:  # noqa: BLE001
                return None, f"读取标记失败：{error}"
            try:
                marker_obj = json.loads(marker_text)
            except Exception as error:  # noqa: BLE001
                return None, f"解析标记失败：{error}"
            if not isinstance(marker_obj, dict):
                return None, "标记内容非法（需为 JSON 对象）"
            return marker_obj, ""

    def list_remote_file_md5s(self, *, remote_dir: str) -> tuple[bool, dict[str, str]]:
        normalized_remote_dir = _normalize_remote_path(remote_dir)
        exists, root_stdout_text, _stderr_text, _ = self._run(["list", normalized_remote_dir], allow_missing=True)
        if not exists:
            return False, {}

        remote_md5_map: dict[str, str] = {}
        stack: list[tuple[str, str, str]] = [("", normalized_remote_dir, root_stdout_text)]
        while stack:
            rel_prefix, current_remote_dir, current_stdout = stack.pop()
            for name, md5_text in _parse_remote_files_with_md5(current_stdout):
                rel_path = _normalize_rel_path(f"{rel_prefix}/{name}" if rel_prefix else name)
                if rel_path:
                    remote_md5_map[rel_path] = md5_text.lower()

            for child_dir_name in parse_remote_dirs(current_stdout):
                child_remote_dir = _normalize_remote_path(f"{current_remote_dir}/{child_dir_name}")
                child_rel_prefix = _normalize_rel_path(
                    f"{rel_prefix}/{child_dir_name}" if rel_prefix else child_dir_name
                )
                child_exists, child_stdout, _child_stderr, _ = self._run(
                    ["list", child_remote_dir],
                    allow_missing=True,
                )
                if not child_exists:
                    continue
                stack.append((child_rel_prefix, child_remote_dir, child_stdout))

        return True, remote_md5_map

    def compare(self, *, remote_dir: str, local_source_dir: Path) -> CompareResult:
        normalized_remote_dir = _normalize_remote_path(remote_dir)
        _ok, stdout_text, stderr_text, exit_code = self._run(
            ["compare", normalized_remote_dir, str(local_source_dir)],
            allow_missing=False,
        )
        same, different, local_only, remote_only = _parse_compare_stats(stdout_text=stdout_text)
        return CompareResult(
            exit_code=exit_code,
            same=same,
            different=different,
            local_only=local_only,
            remote_only=remote_only,
            stdout=stdout_text,
            stderr=stderr_text,
        )


def _parse_compare_stats(*, stdout_text: str) -> tuple[int, int, int, int]:
    stats_map = {
        "same": 0,
        "different": 0,
        "local_only": 0,
        "remote_only": 0,
    }
    for line in str(stdout_text).splitlines():
        text = line.strip()
        if not text:
            continue
        matched = _COMPARE_STATS_PATTERN.match(text)
        if not matched:
            continue
        key = matched.group(1).strip().lower().replace(" ", "_")
        stats_map[key] = int(matched.group(2))

    if all(value == 0 for value in stats_map.values()):
        section = ""
        for line in str(stdout_text).splitlines():
            text = line.strip()
            if text.startswith("===="):
                normalized = text.lower()
                if "same files" in normalized:
                    section = "same"
                elif "different files" in normalized:
                    section = "different"
                elif "local only" in normalized:
                    section = "local_only"
                elif "remote only" in normalized:
                    section = "remote_only"
                else:
                    section = ""
                continue
            if section and text.startswith(("F - ", "D - ")):
                stats_map[section] += 1

    return (
        int(stats_map["same"]),
        int(stats_map["different"]),
        int(stats_map["local_only"]),
        int(stats_map["remote_only"]),
    )
