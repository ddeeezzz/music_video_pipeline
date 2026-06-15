"""
文件用途：配置 handler mixin —— 公共默认配置、任务配置读取与保存。
输入输出：通过 mixin 混入 TaskMonitorService，所有 self.xxx 由 MRO 解析。
依赖说明：依赖 config.py 的 load_config、_read_json_config、_merge_defaults。
维护说明：新增配置字段时需同步前端 schema 与 web UI。
"""

import json
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from music_video_pipeline.config import COMMON_CONFIG_NAME, _merge_defaults, _read_json_config
from music_video_pipeline.monitoring.routes import TASK_CONFIG_OVERRIDES_FILE_NAME


class ConfigHandlers:
    """Mixin —— 配置相关方法。"""

    def _read_common_config_raw(self) -> dict[str, Any]:
        """读取 common.json 的原始配置字典。"""
        common_path = Path("configs") / COMMON_CONFIG_NAME
        if common_path.exists() and common_path.is_file():
            return _read_json_config(common_path) or {}
        return {}

    def _handle_config_default_request(self) -> tuple[dict[str, Any], HTTPStatus]:
        """
        GET /api/config/default
        返回合并了 Python 默认值的完整公共配置，供前端表单预填。
        """
        try:
            common_raw = self._read_common_config_raw()
            merged = _merge_defaults(common_raw)
            return {"ok": True, "config": merged}, HTTPStatus.OK
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"加载默认配置失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR

    def _load_task_config_overrides(self, task_id: str, task_dir: Path | None = None) -> dict[str, Any]:
        """加载任务配置覆盖文件。"""
        if task_dir is None:
            task_dir = self._resolve_task_dir(task_id=task_id)
        overrides_path = task_dir / TASK_CONFIG_OVERRIDES_FILE_NAME
        if overrides_path.exists() and overrides_path.is_file():
            try:
                data = _read_json_config(overrides_path)
                return data if isinstance(data, dict) else {}
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _save_task_config_overrides(self, task_id: str, overrides: dict[str, Any]) -> None:
        """保存任务配置覆盖到任务目录。"""
        task_dir = self._resolve_task_dir(task_id=task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        overrides_path = task_dir / TASK_CONFIG_OVERRIDES_FILE_NAME
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2)

    def _merge_task_config(self, task_id: str, task_record: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        合并 common.json + per-task config + 运行时覆盖，返回完整配置字典。
        """
        # 1) common.json
        common_raw = self._read_common_config_raw()

        # 2) per-task config
        if task_record is None:
            task_record = self.state_store.get_task(task_id=task_id)
        config_path_text = str(task_record.get("config_path", "")).strip() if task_record else ""
        task_raw: dict = {}
        if config_path_text:
            config_path = Path(config_path_text)
            if config_path.exists() and config_path.is_file():
                task_raw = _read_json_config(config_path) or {}

        # 合并 common + task
        merged_from_files: dict = {}
        all_keys = set(common_raw.keys()) | set(task_raw.keys())
        for key in all_keys:
            common_val = common_raw.get(key)
            task_val = task_raw.get(key)
            if isinstance(common_val, dict) and isinstance(task_val, dict):
                merged_from_files[key] = {**common_val, **task_val}
            elif key in task_raw:
                merged_from_files[key] = task_val
            elif key in common_raw:
                merged_from_files[key] = common_val

        # 3) 运行时覆盖
        overrides = self._load_task_config_overrides(task_id=task_id)
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(merged_from_files.get(key), dict):
                merged_from_files[key] = {**merged_from_files[key], **val}
            else:
                merged_from_files[key] = val

        # 4) Python 默认值
        return _merge_defaults(merged_from_files)

    def _handle_task_config_get_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        GET /api/task/config?task_id=xxx
        返回指定任务的合并后完整配置。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [""])[0]).strip()
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数"}, HTTPStatus.BAD_REQUEST
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND
        try:
            merged = self._merge_task_config(task_id=task_id, task_record=task_record)
            overrides = self._load_task_config_overrides(task_id=task_id)
            return {
                "ok": True,
                "task_id": task_id,
                "config": merged,
                "overrides": overrides,
            }, HTTPStatus.OK
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"加载任务配置失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR

    def _handle_task_config_save_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        GET /api/task/config?task_id=xxx&overrides=<json>
        保存任务配置覆盖值。overrides 为 URL 编码的 JSON 对象。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [""])[0]).strip()
        if not task_id:
            return {"ok": False, "error": "缺少 task_id 参数"}, HTTPStatus.BAD_REQUEST
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{task_id}"}, HTTPStatus.NOT_FOUND
        overrides_text = str(query.get("overrides", [""])[0]).strip()
        if not overrides_text:
            return {"ok": False, "error": "缺少 overrides 参数"}, HTTPStatus.BAD_REQUEST
        try:
            overrides = json.loads(overrides_text)
            if not isinstance(overrides, dict):
                return {"ok": False, "error": "overrides 必须是 JSON 对象"}, HTTPStatus.BAD_REQUEST
            self._save_task_config_overrides(task_id=task_id, overrides=overrides)
            return {"ok": True, "task_id": task_id, "message": "配置已保存"}, HTTPStatus.OK
        except json.JSONDecodeError:
            return {"ok": False, "error": "overrides 不是有效 JSON"}, HTTPStatus.BAD_REQUEST
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"保存配置失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
