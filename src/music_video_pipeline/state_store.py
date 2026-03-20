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
