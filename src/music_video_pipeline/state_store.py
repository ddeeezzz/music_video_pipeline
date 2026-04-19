"""
文件用途：实现任务状态存储与断点续传控制。
核心流程：通过 SQLite 管理 task 与 module 状态，支持恢复与重跑控制。
输入输出：输入任务标识和模块状态，输出状态查询结果与持久化副作用。
依赖说明：依赖标准库 sqlite3/pathlib/datetime。
维护说明：状态机仅允许 pending/running/done/failed，新增状态需同步测试。
"""

# 标准库：用于时间戳生成
from datetime import datetime, timedelta, timezone
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于 SQLite 持久化
import sqlite3
# 标准库：用于类型提示
from typing import Any

# 项目内模块：提供模块顺序与状态常量
from music_video_pipeline.constants import MODULE_ORDER, TASK_STATES


def _local_now_text() -> str:
    """
    功能说明：返回当前北京时间（UTC+08:00）的 ISO 字符串。
    参数说明：无。
    返回值：
    - str: 形如 2026-03-20T15:03:08+08:00 的时间文本。
    异常说明：无。
    边界条件：统一使用固定时区 UTC+08:00，避免依赖系统时区数据库。
    """
    china_tz = timezone(timedelta(hours=8))
    return datetime.now(china_tz).isoformat(timespec="seconds")


class StateStore:
    """
    功能说明：封装任务状态数据库操作。
    参数说明：
    - db_path: SQLite 文件路径。
    返回值：不适用。
    异常说明：数据库不可写时会抛出 sqlite3.Error。
    边界条件：同一 task_id 在模块表中的每个模块仅有一行记录。
    """

    def __init__(self, db_path: Path) -> None:
        """
        功能说明：初始化状态存储对象并创建数据库目录。
        参数说明：
        - db_path: SQLite 数据库文件路径。
        返回值：无。
        异常说明：目录不可创建时抛出 OSError。
        边界条件：目录存在时保持幂等。
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """
        功能说明：建立 SQLite 连接并启用行字典访问。
        参数说明：无。
        返回值：
        - sqlite3.Connection: 可执行 SQL 的连接对象。
        异常说明：连接失败时抛出 sqlite3.Error。
        边界条件：每次调用返回新连接，调用方负责关闭。
        """
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        """
        功能说明：初始化任务与模块状态表结构。
        参数说明：无。
        返回值：无。
        异常说明：建表 SQL 失败时抛出 sqlite3.Error。
        边界条件：重复执行不会破坏已有数据。
        """
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT NOT NULL DEFAULT '',
                    output_video_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS module_runs (
                    task_id TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, module_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS module_unit_runs (
                    task_id TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    unit_index INTEGER NOT NULL DEFAULT 0,
                    start_time REAL NOT NULL DEFAULT 0.0,
                    end_time REAL NOT NULL DEFAULT 0.0,
                    duration REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    artifact_path TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, module_name, unit_id)
                )
                """
            )
            connection.commit()

    def init_task(self, task_id: str, audio_path: str, config_path: str) -> None:
        """
        功能说明：初始化任务与模块状态记录，不存在则创建。
        参数说明：
        - task_id: 任务唯一标识。
        - audio_path: 输入音频路径。
        - config_path: 配置文件路径。
        返回值：无。
        异常说明：数据库写入失败时抛出 sqlite3.Error。
        边界条件：任务已存在时仅补齐缺失模块行，不覆盖已有状态。
        """
        now_text = _local_now_text()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(task_id, audio_path, config_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    audio_path=excluded.audio_path,
                    config_path=excluded.config_path,
                    updated_at=excluded.updated_at
                """,
                (task_id, audio_path, config_path, now_text, now_text),
            )
            for module_name in MODULE_ORDER:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO module_runs(task_id, module_name, status, updated_at)
                    VALUES (?, ?, 'pending', ?)
                    """,
                    (task_id, module_name, now_text),
                )
            connection.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """
        功能说明：查询任务记录。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict | None: 查到则返回字典，否则返回 None。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：无。
        """
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            return dict(row) if row else None

    def list_tasks(self) -> list[dict[str, Any]]:
        """
        功能说明：按更新时间倒序读取任务列表（用于交互式任务选择）。
        参数说明：无。
        返回值：
        - list[dict[str, Any]]: 任务概览数组，包含 task_id/status/config_path/audio_path/updated_at。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：无任务时返回空数组。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, status, config_path, audio_path, updated_at
                FROM tasks
                ORDER BY updated_at DESC, task_id ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def list_task_module_status_map(self, task_ids: list[str]) -> dict[str, dict[str, str]]:
        """
        功能说明：批量读取任务的 A/B/C/D 模块状态映射。
        参数说明：
        - task_ids: 任务ID数组。
        返回值：
        - dict[str, dict[str, str]]: task_id -> {A/B/C/D: status}。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：入参为空时返回空字典；缺失模块默认补为 pending。
        """
        normalized_task_ids = [str(task_id).strip() for task_id in task_ids if str(task_id).strip()]
        if not normalized_task_ids:
            return {}

        summary_map: dict[str, dict[str, str]] = {
            task_id: {module_name: "pending" for module_name in MODULE_ORDER}
            for task_id in normalized_task_ids
        }
        placeholders = ",".join("?" for _ in normalized_task_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT task_id, module_name, status
                FROM module_runs
                WHERE task_id IN ({placeholders})
                """,
                tuple(normalized_task_ids),
            ).fetchall()
        for row in rows:
            task_id = str(row["task_id"])
            module_name = str(row["module_name"])
            status_text = str(row["status"])
            if task_id not in summary_map:
                continue
            if module_name not in MODULE_ORDER:
                continue
            summary_map[task_id][module_name] = status_text
        return summary_map

    def get_audio_path(self, task_id: str) -> str | None:
        """
        功能说明：读取任务绑定的音频路径。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - str | None: 音频路径，若不存在任务则返回 None。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：无。
        """
        task_record = self.get_task(task_id)
        if not task_record:
            return None
        return str(task_record["audio_path"])

    def update_task_status(self, task_id: str, status: str, error_message: str = "", output_video_path: str = "") -> None:
        """
        功能说明：更新任务总体状态。
        参数说明：
        - task_id: 任务唯一标识。
        - status: 任务状态。
        - error_message: 失败信息。
        - output_video_path: 最终视频路径。
        返回值：无。
        异常说明：
        - ValueError: 状态非法时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：done 状态建议同时写入 output_video_path。
        """
        if status not in TASK_STATES:
            raise ValueError(f"非法任务状态: {status}")
        now_text = _local_now_text()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, error_message = ?, output_video_path = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status, error_message, output_video_path, now_text, task_id),
            )
            connection.commit()

    def get_module_status_map(self, task_id: str) -> dict[str, str]:
        """
        功能说明：读取任务的全部模块状态映射。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, str]: 模块名到状态的映射。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：若任务不存在，返回空映射。
        """
        with self._connect() as connection:
            rows = connection.execute("SELECT module_name, status FROM module_runs WHERE task_id = ?", (task_id,)).fetchall()
            return {str(row["module_name"]): str(row["status"]) for row in rows}

    def reconcile_bcd_module_statuses_by_units(self, task_id: str) -> dict[str, str]:
        """
        功能说明：基于 B/C/D 单元状态自愈模块级状态，避免中断后模块状态滞留在 running。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, str]: 自愈后的模块状态映射。
        异常说明：数据库读写失败时抛出 sqlite3.Error。
        边界条件：无单元记录的模块保持原状态不变。
        """
        for module_name in ("B", "C", "D"):
            module_record = self.get_module_record(task_id=task_id, module_name=module_name) or {}
            current_status = str(module_record.get("status", "pending"))
            summary = self.get_module_unit_status_summary(task_id=task_id, module_name=module_name)
            total_units = int(summary.get("total_units", 0))
            if total_units <= 0:
                continue

            status_counts = summary.get("status_counts", {})
            done_count = int(status_counts.get("done", 0))
            failed_count = int(status_counts.get("failed", 0))
            running_count = int(status_counts.get("running", 0))
            pending_count = int(status_counts.get("pending", 0))

            expected_status = current_status
            if done_count == total_units:
                expected_status = "done"
            elif failed_count > 0:
                expected_status = "failed"
            elif running_count > 0:
                expected_status = "running"
            elif pending_count > 0:
                expected_status = "pending"

            if expected_status == current_status:
                continue

            artifact_path = str(module_record.get("artifact_path", ""))
            error_message = ""
            if expected_status == "failed":
                problem_unit_ids = list(summary.get("problem_unit_ids", []))
                error_message = f"模块{module_name}存在未完成链路，problem_unit_ids={problem_unit_ids}"
            self.set_module_status(
                task_id=task_id,
                module_name=module_name,
                status=expected_status,
                artifact_path=artifact_path,
                error_message=error_message,
            )

        status_map = self.get_module_status_map(task_id=task_id)
        if all(status_map.get(module_name) == "done" for module_name in MODULE_ORDER):
            task_record = self.get_task(task_id=task_id) or {}
            if str(task_record.get("status", "")) != "done":
                module_d_record = self.get_module_record(task_id=task_id, module_name="D") or {}
                output_video_path = str(module_d_record.get("artifact_path", ""))
                self.update_task_status(task_id=task_id, status="done", output_video_path=output_video_path)
                status_map = self.get_module_status_map(task_id=task_id)
        return status_map

    def get_module_record(self, task_id: str, module_name: str) -> dict[str, Any] | None:
        """
        功能说明：查询单个模块的完整状态记录。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名（A/B/C/D）。
        返回值：
        - dict | None: 查到则返回记录字典，否则返回 None。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：无。
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM module_runs WHERE task_id = ? AND module_name = ?",
                (task_id, module_name),
            ).fetchone()
            return dict(row) if row else None

    def set_module_status(self, task_id: str, module_name: str, status: str, artifact_path: str = "", error_message: str = "") -> None:
        """
        功能说明：更新指定模块状态并记录产物路径或错误信息。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名（A/B/C/D）。
        - status: 模块状态。
        - artifact_path: 成功产物路径。
        - error_message: 失败信息。
        返回值：无。
        异常说明：
        - ValueError: 状态非法时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：running 会刷新 started_at，done/failed 会刷新 finished_at。
        """
        if status not in TASK_STATES:
            raise ValueError(f"非法模块状态: {status}")
        now_text = _local_now_text()
        started_at = now_text if status == "running" else ""
        finished_at = now_text if status in {"done", "failed"} else ""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE module_runs
                SET status = ?,
                    artifact_path = ?,
                    error_message = ?,
                    started_at = CASE WHEN ? = '' THEN started_at ELSE ? END,
                    finished_at = CASE WHEN ? = '' THEN finished_at ELSE ? END,
                    updated_at = ?
                WHERE task_id = ? AND module_name = ?
                """,
                (
                    status,
                    artifact_path,
                    error_message,
                    started_at,
                    started_at,
                    finished_at,
                    finished_at,
                    now_text,
                    task_id,
                    module_name,
                ),
            )
            connection.commit()

    def sync_module_units(self, task_id: str, module_name: str, units: list[dict[str, Any]]) -> None:
        """
        功能说明：同步模块最小单元集合（新增、更新索引与时间信息、清理已移除单元）。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名（当前用于模块 B/C/D）。
        - units: 单元数组，至少包含 unit_id/unit_index/start_time/end_time/duration。
        返回值：无。
        异常说明：
        - ValueError: 模块名非法或单元缺失关键字段时抛出。
        - sqlite3.Error: 数据库写入失败时抛出。
        边界条件：已完成单元默认保持状态，仅同步元信息。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        now_text = _local_now_text()

        normalized_units: list[tuple[str, int, float, float, float]] = []
        for default_index, unit in enumerate(units):
            unit_id = str(unit.get("unit_id", "")).strip()
            if not unit_id:
                raise ValueError("模块单元同步失败：存在空 unit_id")
            unit_index = int(unit.get("unit_index", default_index))
            start_time = float(unit.get("start_time", 0.0))
            end_time = max(start_time, float(unit.get("end_time", start_time)))
            duration = max(0.0, float(unit.get("duration", max(0.5, end_time - start_time))))
            normalized_units.append((unit_id, unit_index, start_time, end_time, duration))

        with self._connect() as connection:
            for unit_id, unit_index, start_time, end_time, duration in normalized_units:
                connection.execute(
                    """
                    INSERT INTO module_unit_runs(
                        task_id, module_name, unit_id, unit_index, start_time, end_time, duration, status, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    ON CONFLICT(task_id, module_name, unit_id) DO UPDATE SET
                        unit_index=excluded.unit_index,
                        start_time=excluded.start_time,
                        end_time=excluded.end_time,
                        duration=excluded.duration,
                        updated_at=excluded.updated_at
                    """,
                    (task_id, module_name, unit_id, unit_index, start_time, end_time, duration, now_text),
                )

            unit_ids = [unit_id for unit_id, _, _, _, _ in normalized_units]
            if unit_ids:
                placeholders = ",".join("?" for _ in unit_ids)
                connection.execute(
                    f"""
                    DELETE FROM module_unit_runs
                    WHERE task_id = ? AND module_name = ? AND unit_id NOT IN ({placeholders})
                    """,
                    (task_id, module_name, *unit_ids),
                )
            else:
                connection.execute(
                    "DELETE FROM module_unit_runs WHERE task_id = ? AND module_name = ?",
                    (task_id, module_name),
                )
            connection.commit()

    def list_module_units_by_status(self, task_id: str, module_name: str, statuses: list[str]) -> list[dict[str, Any]]:
        """
        功能说明：按状态筛选并读取模块单元记录。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        - statuses: 需要筛选的状态集合。
        返回值：
        - list[dict[str, Any]]: 单元记录数组（按 unit_index 升序）。
        异常说明：
        - ValueError: 状态非法时抛出。
        - sqlite3.Error: 查询失败时抛出。
        边界条件：statuses 为空时返回空数组。
        """
        if not statuses:
            return []
        for status in statuses:
            if status not in TASK_STATES:
                raise ValueError(f"非法模块单元状态: {status}")
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ? AND status IN ({placeholders})
                ORDER BY unit_index ASC
                """,
                (task_id, module_name, *statuses),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_module_units(self, task_id: str, module_name: str) -> list[dict[str, Any]]:
        """
        功能说明：读取指定模块的全部单元记录。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        返回值：
        - list[dict[str, Any]]: 单元记录数组（按 unit_index 升序）。
        异常说明：
        - ValueError: 模块名非法时抛出。
        - sqlite3.Error: 查询失败时抛出。
        边界条件：模块无单元时返回空数组。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ?
                ORDER BY unit_index ASC
                """,
                (task_id, module_name),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_module_unit_status(
        self,
        task_id: str,
        module_name: str,
        unit_id: str,
        status: str,
        artifact_path: str = "",
        error_message: str = "",
    ) -> None:
        """
        功能说明：更新模块单元状态并记录产物路径或错误信息。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        - unit_id: 单元唯一标识（模块C中对应 shot_id）。
        - status: 单元状态。
        - artifact_path: 成功产物路径。
        - error_message: 失败信息。
        返回值：无。
        异常说明：
        - ValueError: 状态非法时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：running 会刷新 started_at，done/failed 会刷新 finished_at。
        """
        if status not in TASK_STATES:
            raise ValueError(f"非法模块单元状态: {status}")
        now_text = _local_now_text()
        started_at = now_text if status == "running" else ""
        finished_at = now_text if status in {"done", "failed"} else ""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE module_unit_runs
                SET status = ?,
                    artifact_path = ?,
                    error_message = ?,
                    started_at = CASE WHEN ? = '' THEN started_at ELSE ? END,
                    finished_at = CASE WHEN ? = '' THEN finished_at ELSE ? END,
                    updated_at = ?
                WHERE task_id = ? AND module_name = ? AND unit_id = ?
                """,
                (
                    status,
                    artifact_path,
                    error_message,
                    started_at,
                    started_at,
                    finished_at,
                    finished_at,
                    now_text,
                    task_id,
                    module_name,
                    unit_id,
                ),
            )
            connection.commit()

    def get_module_unit_record(self, task_id: str, module_name: str, unit_id: str) -> dict[str, Any] | None:
        """
        功能说明：查询单个模块单元的完整状态记录。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        - unit_id: 单元唯一标识。
        返回值：
        - dict[str, Any] | None: 查到则返回记录字典，否则返回 None。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：无。
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ? AND unit_id = ?
                """,
                (task_id, module_name, unit_id),
            ).fetchone()
            return dict(row) if row else None

    def get_module_unit_record_by_index(self, task_id: str, module_name: str, unit_index: int) -> dict[str, Any] | None:
        """
        功能说明：按 unit_index 查询单个模块单元状态记录。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        - unit_index: 单元顺序索引（0 基）。
        返回值：
        - dict[str, Any] | None: 查到则返回记录字典，否则返回 None。
        异常说明：
        - ValueError: 模块名非法时抛出。
        - sqlite3.Error: 查询失败时抛出。
        边界条件：同一模块同一 task 下 unit_index 允许查询到 0 或 1 条记录。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ? AND unit_index = ?
                LIMIT 1
                """,
                (task_id, module_name, int(unit_index)),
            ).fetchone()
            return dict(row) if row else None

    def get_module_unit_status_summary(self, task_id: str, module_name: str) -> dict[str, Any]:
        """
        功能说明：汇总模块单元状态计数与问题单元列表，用于CLI可观测输出。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        返回值：
        - dict[str, Any]: 含总数、状态计数与各状态单元ID列表的摘要。
        异常说明：
        - ValueError: 模块名非法时抛出。
        - sqlite3.Error: 查询失败时抛出。
        边界条件：当模块无单元记录时计数均为0、列表为空。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        with self._connect() as connection:
            total_row = connection.execute(
                """
                SELECT COUNT(1) AS total_count
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ?
                """,
                (task_id, module_name),
            ).fetchone()
            total_units = int(total_row["total_count"]) if total_row else 0

            rows = connection.execute(
                """
                SELECT status, COUNT(1) AS count_value
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ?
                GROUP BY status
                """,
                (task_id, module_name),
            ).fetchall()
            status_counts = {status: 0 for status in TASK_STATES}
            for row in rows:
                status_text = str(row["status"])
                if status_text in status_counts:
                    status_counts[status_text] = int(row["count_value"])

            detailed_rows = connection.execute(
                """
                SELECT unit_id, status
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = ?
                ORDER BY unit_index ASC
                """,
                (task_id, module_name),
            ).fetchall()

        pending_unit_ids: list[str] = []
        running_unit_ids: list[str] = []
        failed_unit_ids: list[str] = []
        done_unit_ids: list[str] = []
        for row in detailed_rows:
            unit_id = str(row["unit_id"])
            status_text = str(row["status"])
            if status_text == "pending":
                pending_unit_ids.append(unit_id)
            elif status_text == "running":
                running_unit_ids.append(unit_id)
            elif status_text == "failed":
                failed_unit_ids.append(unit_id)
            elif status_text == "done":
                done_unit_ids.append(unit_id)

        return {
            "module_name": module_name,
            "total_units": total_units,
            "status_counts": status_counts,
            "pending_unit_ids": pending_unit_ids,
            "running_unit_ids": running_unit_ids,
            "failed_unit_ids": failed_unit_ids,
            "done_unit_ids": done_unit_ids,
            "problem_unit_ids": failed_unit_ids + running_unit_ids + pending_unit_ids,
        }

    def list_bcd_chain_status(self, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：按 unit_index 汇总 B/C/D 链路单元状态，输出链路级排障摘要。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: 链路数组（按 unit_index 升序）。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：任一模块缺失单元时，链路仍按并集索引输出。
        """
        b_rows = self.list_module_units(task_id=task_id, module_name="B")
        c_rows = self.list_module_units(task_id=task_id, module_name="C")
        d_rows = self.list_module_units(task_id=task_id, module_name="D")
        b_by_index = {int(item["unit_index"]): item for item in b_rows}
        c_by_index = {int(item["unit_index"]): item for item in c_rows}
        d_by_index = {int(item["unit_index"]): item for item in d_rows}
        all_indexes = sorted(set(b_by_index.keys()) | set(c_by_index.keys()) | set(d_by_index.keys()))

        chain_rows: list[dict[str, Any]] = []
        for unit_index in all_indexes:
            b_row = b_by_index.get(unit_index, {})
            c_row = c_by_index.get(unit_index, {})
            d_row = d_by_index.get(unit_index, {})
            segment_id = str(b_row.get("unit_id", "")).strip()
            c_shot_id = str(c_row.get("unit_id", "")).strip()
            d_shot_id = str(d_row.get("unit_id", "")).strip()
            shot_id = c_shot_id or d_shot_id or f"shot_{unit_index + 1:03d}"
            b_status = str(b_row.get("status", "pending"))
            c_status = str(c_row.get("status", "pending"))
            d_status = str(d_row.get("status", "pending"))

            chain_status = "pending"
            if "failed" in {b_status, c_status, d_status}:
                chain_status = "failed"
            elif d_status == "done":
                chain_status = "done"
            elif "running" in {b_status, c_status, d_status}:
                chain_status = "running"

            chain_rows.append(
                {
                    "unit_index": unit_index,
                    "segment_id": segment_id,
                    "shot_id": shot_id,
                    "b_status": b_status,
                    "c_status": c_status,
                    "d_status": d_status,
                    "chain_status": chain_status,
                    "b_error_message": str(b_row.get("error_message", "")),
                    "c_error_message": str(c_row.get("error_message", "")),
                    "d_error_message": str(d_row.get("error_message", "")),
                }
            )
        return chain_rows

    def reset_bcd_chain_units(self, task_id: str, segment_id: str) -> dict[str, Any]:
        """
        功能说明：按 segment_id 重置一条 B/C/D 链路单元为 pending。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: 模块 B 单元标识。
        返回值：
        - dict[str, Any]: 重置结果摘要（unit_index/segment_id/shot_id）。
        异常说明：
        - RuntimeError: 目标 segment 不存在时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：仅影响目标链路单元，不改动其他链路。
        """
        normalized_segment_id = str(segment_id).strip()
        if not normalized_segment_id:
            raise RuntimeError("segment_id 不能为空。")

        b_unit = self.get_module_unit_record(task_id=task_id, module_name="B", unit_id=normalized_segment_id)
        if not b_unit:
            raise RuntimeError(
                f"链路重置失败：segment_id 不存在或尚未建立 B 单元状态，task_id={task_id}，segment_id={normalized_segment_id}"
            )

        unit_index = int(b_unit["unit_index"])
        c_unit = self.get_module_unit_record_by_index(task_id=task_id, module_name="C", unit_index=unit_index)
        d_unit = self.get_module_unit_record_by_index(task_id=task_id, module_name="D", unit_index=unit_index)
        shot_id = str(c_unit["unit_id"]) if c_unit else (str(d_unit["unit_id"]) if d_unit else f"shot_{unit_index + 1:03d}")

        now_text = _local_now_text()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE module_unit_runs
                SET status = 'pending',
                    artifact_path = '',
                    error_message = '',
                    started_at = '',
                    finished_at = '',
                    updated_at = ?
                WHERE task_id = ? AND module_name = 'B' AND unit_id = ?
                """,
                (now_text, task_id, normalized_segment_id),
            )
            connection.execute(
                """
                UPDATE module_unit_runs
                SET status = 'pending',
                    artifact_path = '',
                    error_message = '',
                    started_at = '',
                    finished_at = '',
                    updated_at = ?
                WHERE task_id = ? AND module_name = 'C' AND unit_index = ?
                """,
                (now_text, task_id, unit_index),
            )
            connection.execute(
                """
                UPDATE module_unit_runs
                SET status = 'pending',
                    artifact_path = '',
                    error_message = '',
                    started_at = '',
                    finished_at = '',
                    updated_at = ?
                WHERE task_id = ? AND module_name = 'D' AND unit_index = ?
                """,
                (now_text, task_id, unit_index),
            )
            for module_name in ("B", "C", "D"):
                connection.execute(
                    """
                    UPDATE module_runs
                    SET status = 'pending',
                        artifact_path = '',
                        error_message = '',
                        started_at = '',
                        finished_at = '',
                        updated_at = ?
                    WHERE task_id = ? AND module_name = ?
                    """,
                    (now_text, task_id, module_name),
                )
            connection.execute(
                """
                UPDATE tasks
                SET status = 'pending',
                    error_message = '',
                    output_video_path = '',
                    updated_at = ?
                WHERE task_id = ?
                """,
                (now_text, task_id),
            )
            connection.commit()

        return {
            "task_id": task_id,
            "unit_index": unit_index,
            "segment_id": normalized_segment_id,
            "shot_id": shot_id,
        }

    def mark_bcd_downstream_blocked(self, task_id: str, unit_index: int, from_module: str, reason: str) -> None:
        """
        功能说明：将指定链路的下游单元标记为 failed（upstream_blocked）。
        参数说明：
        - task_id: 任务唯一标识。
        - unit_index: 链路顺序索引（0 基）。
        - from_module: 上游失败模块（B 或 C）。
        - reason: 失败原因文本。
        返回值：无。
        异常说明：
        - ValueError: from_module 非法时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：不会覆盖下游已 done 单元。
        """
        normalized_from = str(from_module).strip().upper()
        if normalized_from not in {"B", "C"}:
            raise ValueError(f"非法 from_module: {from_module}")
        target_modules = ["C", "D"] if normalized_from == "B" else ["D"]
        now_text = _local_now_text()
        reason_text = str(reason).strip() or "upstream_blocked"
        with self._connect() as connection:
            for module_name in target_modules:
                connection.execute(
                    """
                    UPDATE module_unit_runs
                    SET status = 'failed',
                        artifact_path = '',
                        error_message = ?,
                        finished_at = ?,
                        updated_at = ?
                    WHERE task_id = ? AND module_name = ? AND unit_index = ? AND status != 'done'
                    """,
                    (reason_text, now_text, now_text, task_id, module_name, int(unit_index)),
                )
            connection.commit()

    def reset_module_unit(self, task_id: str, module_name: str, unit_id: str) -> None:
        """
        功能说明：将指定模块单元重置为 pending，并清空产物与错误信息。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 模块名。
        - unit_id: 单元唯一标识。
        返回值：无。
        异常说明：
        - ValueError: 模块名非法时抛出。
        - RuntimeError: 单元不存在时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：仅影响目标单元，不改动同模块其他单元。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        now_text = _local_now_text()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE module_unit_runs
                SET status = 'pending',
                    artifact_path = '',
                    error_message = '',
                    started_at = '',
                    finished_at = '',
                    updated_at = ?
                WHERE task_id = ? AND module_name = ? AND unit_id = ?
                """,
                (now_text, task_id, module_name, unit_id),
            )
            if int(cursor.rowcount) == 0:
                raise RuntimeError(f"模块单元不存在，无法重置：task_id={task_id}，module={module_name}，unit_id={unit_id}")
            connection.commit()

    def list_module_c_done_frame_items(self, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：读取模块 C 已完成单元并转换为 frame_items 结构。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: 按 unit_index 升序排列的 frame_items。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：仅返回状态为 done 且有 artifact_path 的单元。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT unit_id, unit_index, artifact_path, start_time, end_time, duration
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = 'C' AND status = 'done' AND artifact_path != ''
                ORDER BY unit_index ASC
                """,
                (task_id,),
            ).fetchall()
            return [
                {
                    "shot_id": str(row["unit_id"]),
                    "frame_path": str(row["artifact_path"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "duration": float(row["duration"]),
                }
                for row in rows
            ]

    def list_module_b_done_shot_items(self, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：读取模块 B 已完成单元记录并返回按顺序可聚合的分镜清单。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: 包含 unit_id/unit_index/artifact_path/time 字段的数组。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：仅返回状态为 done 且有 artifact_path 的单元。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT unit_id, unit_index, artifact_path, start_time, end_time, duration
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = 'B' AND status = 'done' AND artifact_path != ''
                ORDER BY unit_index ASC
                """,
                (task_id,),
            ).fetchall()
            return [
                {
                    "unit_id": str(row["unit_id"]),
                    "unit_index": int(row["unit_index"]),
                    "artifact_path": str(row["artifact_path"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "duration": float(row["duration"]),
                }
                for row in rows
            ]

    def list_module_d_done_segment_items(self, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：读取模块 D 已完成单元记录并返回按顺序可终拼的片段清单。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: 包含 unit_id/unit_index/artifact_path/time 字段的数组。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：仅返回状态为 done 且有 artifact_path 的单元。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT unit_id, unit_index, artifact_path, start_time, end_time, duration
                FROM module_unit_runs
                WHERE task_id = ? AND module_name = 'D' AND status = 'done' AND artifact_path != ''
                ORDER BY unit_index ASC
                """,
                (task_id,),
            ).fetchall()
            return [
                {
                    "unit_id": str(row["unit_id"]),
                    "unit_index": int(row["unit_index"]),
                    "artifact_path": str(row["artifact_path"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "duration": float(row["duration"]),
                }
                for row in rows
            ]

    def first_non_done_module(self, task_id: str) -> str | None:
        """
        功能说明：按模块顺序定位第一个非 done 模块。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - str | None: 模块名，若全 done 则返回 None。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：仅以 MODULE_ORDER 顺序判断。
        """
        status_map = self.get_module_status_map(task_id)
        for module_name in MODULE_ORDER:
            if status_map.get(module_name) != "done":
                return module_name
        return None

    def reset_from_module(self, task_id: str, module_name: str) -> None:
        """
        功能说明：将指定模块及其下游模块重置为 pending。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 重置起点模块。
        返回值：无。
        异常说明：
        - ValueError: 模块名非法时抛出。
        - sqlite3.Error: 更新失败时抛出。
        边界条件：上游模块状态保持不变。
        """
        if module_name not in MODULE_ORDER:
            raise ValueError(f"非法模块名: {module_name}")
        start_index = MODULE_ORDER.index(module_name)
        now_text = _local_now_text()
        with self._connect() as connection:
            for name in MODULE_ORDER[start_index:]:
                connection.execute(
                    """
                    UPDATE module_runs
                    SET status = 'pending',
                        artifact_path = '',
                        error_message = '',
                        started_at = '',
                        finished_at = '',
                        updated_at = ?
                    WHERE task_id = ? AND module_name = ?
                    """,
                    (now_text, task_id, name),
                )
            connection.execute(
                """
                UPDATE tasks
                SET status = 'pending',
                    error_message = '',
                    output_video_path = '',
                    updated_at = ?
                WHERE task_id = ?
                """,
                (now_text, task_id),
            )
            if module_name in {"A", "B"}:
                connection.execute(
                    """
                    DELETE FROM module_unit_runs
                    WHERE task_id = ? AND module_name = 'B'
                    """,
                    (task_id,),
                )
            if module_name in {"A", "B", "C"}:
                connection.execute(
                    """
                    DELETE FROM module_unit_runs
                    WHERE task_id = ? AND module_name = 'C'
                    """,
                    (task_id,),
                )
            if module_name in {"A", "B", "C", "D"}:
                connection.execute(
                    """
                    DELETE FROM module_unit_runs
                    WHERE task_id = ? AND module_name = 'D'
                    """,
                    (task_id,),
                )
            connection.commit()

    def can_run_module(self, task_id: str, module_name: str) -> tuple[bool, str]:
        """
        功能说明：检查模块是否满足启动条件（上游均为 done）。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 待执行模块名。
        返回值：
        - tuple[bool, str]: (是否可运行, 说明文本)。
        异常说明：查询失败时抛出 sqlite3.Error。
        边界条件：A 模块没有上游依赖，默认可运行。
        """
        if module_name not in MODULE_ORDER:
            return False, f"模块名非法: {module_name}"

        status_map = self.get_module_status_map(task_id)
        target_index = MODULE_ORDER.index(module_name)
        for upstream_module in MODULE_ORDER[:target_index]:
            if status_map.get(upstream_module) != "done":
                return False, f"上游模块 {upstream_module} 未完成，当前状态={status_map.get(upstream_module)}"
        return True, "允许执行"

    def mark_task_done_if_possible(self, task_id: str, output_video_path: str = "") -> None:
        """
        功能说明：当全部模块完成时自动将任务状态标记为 done。
        参数说明：
        - task_id: 任务唯一标识。
        - output_video_path: 最终视频路径。
        返回值：无。
        异常说明：数据库写入失败时抛出 sqlite3.Error。
        边界条件：若存在非 done 模块则不做状态变更。
        """
        status_map = self.get_module_status_map(task_id)
        if all(status_map.get(module_name) == "done" for module_name in MODULE_ORDER):
            self.update_task_status(task_id=task_id, status="done", output_video_path=output_video_path)
