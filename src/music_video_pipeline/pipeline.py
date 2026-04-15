"""
文件用途：实现 MVP 流水线调度器。
核心流程：按 A->B->C->D 顺序执行模块，并由状态库统一管理恢复逻辑。
输入输出：输入任务参数，输出执行摘要（字典）。
依赖说明：依赖项目内模块实现、状态存储、上下文对象。
维护说明：调度层不应包含模型细节，只负责流程与状态控制。
"""

# 标准库：用于日志记录
import logging
# 标准库：用于上下文管理器
from contextlib import contextmanager
# 标准库：用于日志文件名时间戳
from datetime import datetime
# 标准库：用于HTML转义
from html import escape
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于可调用类型提示
from typing import Callable, Iterator

# 项目内模块：配置类型
from music_video_pipeline.config import AppConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：常量定义
from music_video_pipeline.constants import MODULE_ORDER, VALID_MODULES
# 项目内模块：目录工具
from music_video_pipeline.io_utils import ensure_dir
# 项目内模块：模块执行函数
from music_video_pipeline.modules import run_module_a_v2, run_module_b, run_module_c, run_module_d
# 项目内模块：跨模块 B/C/D 并行入口
from music_video_pipeline.modules.cross_bcd import run_cross_module_bcd
# 项目内模块：任务监督服务
from music_video_pipeline.monitoring import TaskMonitorService
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore

ModuleRunner = Callable[[RuntimeContext], Path]

# 常量：任务监督入口页文件名（由 monitor 命令写入任务目录）
TASK_MONITOR_LAUNCH_PAGE_FILE_NAME = "task_monitor.html"


class PipelineRunner:
    """
    功能说明：封装 run/resume/run-module 的核心调度逻辑。
    参数说明：
    - workspace_root: 项目根目录（固定为 t1）。
    - config: 应用配置对象。
    - logger: 日志对象。
    返回值：不适用。
    异常说明：执行失败时抛 RuntimeError。
    边界条件：仅在 t1 内写入运行产物。
    """

    def __init__(self, workspace_root: Path, config: AppConfig, logger: logging.Logger) -> None:
        """
        功能说明：初始化调度器与状态数据库。
        参数说明：
        - workspace_root: t1 根目录路径。
        - config: 应用配置。
        - logger: 日志对象。
        返回值：无。
        异常说明：状态库初始化失败时抛异常。
        边界条件：runs_dir 不存在时自动创建。
        """
        self.workspace_root = workspace_root
        self.config = config
        self.logger = logger
        self.runs_dir = (workspace_root / config.paths.runs_dir).resolve()
        ensure_dir(self.runs_dir)
        self.state_store = StateStore(db_path=self.runs_dir / "pipeline_state.sqlite3")
        self.module_runners: dict[str, ModuleRunner] = {
            "A": run_module_a_v2,
            "B": run_module_b,
            "C": run_module_c,
            "D": run_module_d,
        }

    @contextmanager
    def _bind_task_log_file(self, task_dir: Path, command_name: str) -> Iterator[Path]:
        """
        功能说明：在任务执行期间挂载任务级日志文件处理器。
        参数说明：
        - task_dir: 任务目录路径。
        - command_name: 命令名标识（run/resume/run_module_xxx）。
        返回值：
        - Iterator[Path]: 日志文件路径上下文迭代器。
        异常说明：异常由调用方或上层流程统一处理。
        边界条件：退出上下文时必须移除并关闭 handler，避免串写。
        """
        log_dir = task_dir / "log"
        ensure_dir(log_dir)
        safe_command = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in command_name.lower()).strip("_")
        if not safe_command:
            safe_command = "task"
        timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"{safe_command}_{timestamp_text}.log"
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        self.logger.addHandler(file_handler)
        monitor_service: TaskMonitorService | None = None
        try:
            self.logger.info("任务日志文件已挂载，task_id目录=%s，日志路径=%s", task_dir, log_path)
            monitor_service = self._start_task_monitor_service(task_id=task_dir.name)
            if monitor_service:
                monitor_page_path = self._write_task_monitor_launch_page(
                    task_dir=task_dir,
                    task_id=task_dir.name,
                    monitor_url=monitor_service.monitor_url,
                )
                monitor_page_link = self._build_clickable_file_link(file_path=monitor_page_path)
                self.logger.info("任务监督入口页链接=%s", monitor_page_link)
                self.logger.info("任务监督实时页面链接=%s", monitor_service.monitor_url)
            yield log_path
        finally:
            if monitor_service is not None:
                monitor_service.stop()
            self.logger.removeHandler(file_handler)
            file_handler.close()

    def _start_task_monitor_service(self, task_id: str) -> TaskMonitorService | None:
        """
        功能说明：为当前任务自动启动监督服务，失败时降级为仅日志提示。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - TaskMonitorService | None: 启动成功返回服务对象，失败返回 None。
        异常说明：内部吞掉启动异常并记录 warning，避免阻塞主流水线。
        边界条件：服务生命周期由调用方上下文统一 stop。
        """
        monitor_service = TaskMonitorService(
            state_store=self.state_store,
            task_id=task_id,
            logger=self.logger,
            auto_stop_on_terminal=True,
        )
        try:
            monitor_service.start()
            return monitor_service
        except Exception as error:  # noqa: BLE001
            monitor_service.stop()
            self.logger.warning(
                "任务监督服务自动启动失败，已降级为无监督执行，task_id=%s，错误=%s",
                task_id,
                error,
            )
            return None

    def _write_task_monitor_launch_page(self, task_dir: Path, task_id: str, monitor_url: str) -> Path:
        """
        功能说明：在任务目录写入监督入口页，打开后自动跳转到本次监督服务地址。
        参数说明：
        - task_dir: 任务目录路径（runs/<task_id>）。
        - task_id: 任务唯一标识。
        - monitor_url: 本次监督服务URL。
        返回值：
        - Path: 写入后的入口页路径。
        异常说明：无。
        边界条件：每次执行均覆盖写入，确保入口页指向当前端口。
        """
        task_dir.mkdir(parents=True, exist_ok=True)
        launch_page_path = task_dir / TASK_MONITOR_LAUNCH_PAGE_FILE_NAME
        raw_monitor_url = str(monitor_url)
        safe_task_id = escape(str(task_id), quote=True)
        safe_monitor_url = escape(raw_monitor_url, quote=True)
        html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>任务监督入口 - {safe_task_id}</title>
  <meta http-equiv="refresh" content="0;url={safe_monitor_url}">
</head>
<body>
  <p>任务监督服务正在跳转中：<a href="{safe_monitor_url}">{safe_monitor_url}</a></p>
  <script>
    (function () {{
      var targetUrl = {raw_monitor_url!r};
      if (window.location.href !== targetUrl) {{
        window.location.replace(targetUrl);
      }}
    }})();
  </script>
</body>
</html>
"""
        launch_page_path.write_text(html_text, encoding="utf-8")
        return launch_page_path

    def _build_clickable_file_link(self, file_path: Path) -> str:
        """
        功能说明：将本地文件路径转换为终端可点击的 file:// 链接。
        参数说明：
        - file_path: 目标文件路径。
        返回值：
        - str: 优先返回 file:// URL，失败时回退绝对路径字符串。
        异常说明：无。
        边界条件：仅用于日志展示，不校验路径是否可被系统浏览器打开。
        """
        resolved_path = file_path.resolve()
        try:
            return resolved_path.as_uri()
        except ValueError:
            return str(resolved_path)

    def run(self, task_id: str, audio_path: Path, config_path: Path, force_module: str | None = None) -> dict:
        """
        功能说明：执行全链路任务（支持从指定模块强制重跑）。
        参数说明：
        - task_id: 任务唯一标识。
        - audio_path: 输入音频路径。
        - config_path: 配置文件路径。
        - force_module: 可选，指定模块及其下游重跑起点。
        返回值：
        - dict: 任务执行摘要。
        异常说明：模块失败时抛 RuntimeError。
        边界条件：已完成模块默认跳过，除非 force_module 覆盖。
        """
        normalized_audio_path = self._resolve_audio_path(audio_path=audio_path)
        context = self._prepare_context(task_id=task_id, audio_path=normalized_audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="run"):
            self.state_store.init_task(task_id=task_id, audio_path=str(normalized_audio_path), config_path=str(config_path))

            if force_module:
                normalized_module = self._normalize_module_name(force_module)
                self.logger.warning("任务触发强制重跑，task_id=%s，from_module=%s", task_id, normalized_module)
                self.state_store.reset_from_module(task_id=task_id, module_name=normalized_module)

            self.state_store.update_task_status(task_id=task_id, status="running")
            output_video_path = self._execute_modules(context=context, start_module="A")
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            return summary

    def resume(self, task_id: str, config_path: Path, force_module: str | None = None) -> dict:
        """
        功能说明：从断点恢复任务执行。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path: 配置文件路径。
        - force_module: 可选，强制从指定模块重跑。
        返回值：
        - dict: 任务执行摘要。
        异常说明：任务不存在或模块失败时抛 RuntimeError。
        边界条件：若所有模块均 done，则直接返回当前摘要。
        """
        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法 resume: task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="resume"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))

            start_module = self.state_store.first_non_done_module(task_id=task_id)
            if force_module:
                start_module = self._normalize_module_name(force_module)
                self.logger.warning("任务触发强制恢复，task_id=%s，from_module=%s", task_id, start_module)
                self.state_store.reset_from_module(task_id=task_id, module_name=start_module)

            if not start_module:
                self.logger.info("任务已全部完成，无需恢复，task_id=%s", task_id)
                return self._build_summary(task_id=task_id, output_video_path=Path(str(task_record.get("output_video_path", ""))))

            self.state_store.update_task_status(task_id=task_id, status="running")
            output_video_path = self._execute_modules(context=context, start_module=start_module)
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            return summary

    def run_single_module(
        self,
        task_id: str,
        module_name: str,
        config_path: Path,
        audio_path: Path | None = None,
        force: bool = False,
    ) -> dict:
        """
        功能说明：执行单模块调试任务。
        参数说明：
        - task_id: 任务唯一标识。
        - module_name: 目标模块名（A/B/C/D）。
        - config_path: 配置文件路径。
        - audio_path: 可选音频路径，任务首次创建时必需。
        - force: 是否重置该模块及下游状态。
        返回值：
        - dict: 任务执行摘要。
        异常说明：上游未完成或模块失败时抛 RuntimeError。
        边界条件：不自动执行下游模块。
        """
        normalized_module = self._normalize_module_name(module_name)
        task_record = self.state_store.get_task(task_id=task_id)

        if task_record:
            resolved_audio_path = Path(str(task_record["audio_path"]))
        elif audio_path is not None:
            resolved_audio_path = self._resolve_audio_path(audio_path=audio_path)
        else:
            raise RuntimeError("run-module 首次执行需要提供 --audio-path。")

        context = self._prepare_context(task_id=task_id, audio_path=resolved_audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name=f"run_module_{normalized_module.lower()}"):
            self.state_store.init_task(task_id=task_id, audio_path=str(resolved_audio_path), config_path=str(config_path))

            if force:
                self.state_store.reset_from_module(task_id=task_id, module_name=normalized_module)

            can_run, reason = self.state_store.can_run_module(task_id=task_id, module_name=normalized_module)
            if not can_run:
                raise RuntimeError(f"模块 {normalized_module} 无法执行：{reason}")

            self.state_store.update_task_status(task_id=task_id, status="running")
            output_video_path = self._execute_single_module(context=context, module_name=normalized_module)
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            return summary

    def get_module_c_status_summary(self, task_id: str, config_path: Path) -> dict:
        """
        功能说明：读取指定任务的模块 C 单元状态摘要，用于 CLI 可观测排障。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务状态、模块状态与模块 C 单元摘要。
        异常说明：任务不存在时抛 RuntimeError。
        边界条件：仅查询状态，不触发模块执行。
        """
        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法查询模块C状态：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="c_task_status"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            latest_task = self.state_store.get_task(task_id=task_id) or {}
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            module_c_unit_summary = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="C")
            summary = {
                "task_id": task_id,
                "task_status": latest_task.get("status", "unknown"),
                "module_status": module_status_map,
                "module_c_status": module_status_map.get("C", "unknown"),
                "module_c_unit_summary": module_c_unit_summary,
                "output_video_path": str(latest_task.get("output_video_path", "")),
            }
            self.logger.info(
                "模块C单元状态摘要查询完成，task_id=%s，module_c_status=%s，total_units=%s",
                task_id,
                summary["module_c_status"],
                module_c_unit_summary["total_units"],
            )
            return summary

    def get_module_b_status_summary(self, task_id: str, config_path: Path) -> dict:
        """
        功能说明：读取指定任务的模块 B 单元状态摘要，用于 CLI 可观测排障。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务状态、模块状态与模块 B 单元摘要。
        异常说明：任务不存在时抛 RuntimeError。
        边界条件：仅查询状态，不触发模块执行。
        """
        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法查询模块B状态：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="b_task_status"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            latest_task = self.state_store.get_task(task_id=task_id) or {}
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            module_b_unit_summary = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="B")
            summary = {
                "task_id": task_id,
                "task_status": latest_task.get("status", "unknown"),
                "module_status": module_status_map,
                "module_b_status": module_status_map.get("B", "unknown"),
                "module_b_unit_summary": module_b_unit_summary,
                "output_video_path": str(latest_task.get("output_video_path", "")),
            }
            self.logger.info(
                "模块B单元状态摘要查询完成，task_id=%s，module_b_status=%s，total_units=%s",
                task_id,
                summary["module_b_status"],
                module_b_unit_summary["total_units"],
            )
            return summary

    def retry_module_b_segment(self, task_id: str, segment_id: str, config_path: Path) -> dict:
        """
        功能说明：按 segment 粒度重试模块 B 单元，成功后仅占位重置 C/D 待后续重建。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: 模块 B 单元标识（等价模块 A 的 segment_id）。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务执行摘要，并附带重试 segment 与下游重建占位信息。
        异常说明：任务不存在、上游未完成、单元不存在或模块执行失败时抛 RuntimeError。
        边界条件：仅允许定向重试一个单元，不自动执行 C/D。
        """
        normalized_segment_id = str(segment_id).strip()
        if not normalized_segment_id:
            raise RuntimeError("segment_id 不能为空。")

        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法重试模块B单元：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="b_retry_segment"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            if module_status_map.get("A") != "done":
                raise RuntimeError(
                    f"模块B定向重试被拒绝：上游模块A未完成，task_id={task_id}，status={module_status_map.get('A')}"
                )

            unit_record = self.state_store.get_module_unit_record(
                task_id=task_id,
                module_name="B",
                unit_id=normalized_segment_id,
            )
            if not unit_record:
                raise RuntimeError(
                    f"模块B定向重试失败：segment_id 不存在或尚未建立单元状态，task_id={task_id}，segment_id={normalized_segment_id}"
                )

            non_done_units = self.state_store.list_module_units_by_status(
                task_id=task_id,
                module_name="B",
                statuses=["pending", "running", "failed"],
            )
            blocking_unit_ids = [str(item["unit_id"]) for item in non_done_units if str(item["unit_id"]) != normalized_segment_id]
            if blocking_unit_ids:
                raise RuntimeError(
                    "模块B定向重试被拒绝：存在其他非done单元，请先清理后再重试。"
                    f"task_id={task_id}，blocking_unit_ids={blocking_unit_ids}"
                )

            self.state_store.reset_module_unit(
                task_id=task_id,
                module_name="B",
                unit_id=normalized_segment_id,
            )
            self.logger.info("模块B定向重试已重置目标单元，task_id=%s，segment_id=%s", task_id, normalized_segment_id)

            self.state_store.update_task_status(task_id=task_id, status="running")
            self._execute_one_module(context=context, module_name="B")

            # 本轮仅做下游占位，不自动重建 C/D。
            self.state_store.reset_from_module(task_id=task_id, module_name="C")

            module_d_record = self.state_store.get_module_record(task_id=task_id, module_name="D")
            output_video_path = Path(module_d_record["artifact_path"]) if module_d_record and module_d_record["artifact_path"] else Path("")
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            summary["retry_segment_id"] = normalized_segment_id
            summary["module_b_unit_summary"] = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="B")
            summary["downstream_rebuild_required"] = True
            summary["rebuild_from_module"] = "C"
            return summary

    def retry_module_c_shot(self, task_id: str, shot_id: str, config_path: Path) -> dict:
        """
        功能说明：按 shot 粒度重试模块 C 单元，并在成功后执行模块 D 生成最新成片。
        参数说明：
        - task_id: 任务唯一标识。
        - shot_id: 模块 C 单元标识（等价模块 B 的 shot_id）。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务执行摘要，并附带本次重试 shot 与模块 C 单元摘要。
        异常说明：任务不存在、上游未完成、单元不存在或模块执行失败时抛 RuntimeError。
        边界条件：仅允许定向重试一个单元，不重跑 A/B。
        """
        normalized_shot_id = str(shot_id).strip()
        if not normalized_shot_id:
            raise RuntimeError("shot_id 不能为空。")

        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法重试模块C单元：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="c_retry_shot"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            if module_status_map.get("B") != "done":
                raise RuntimeError(
                    f"模块C定向重试被拒绝：上游模块B未完成，task_id={task_id}，status={module_status_map.get('B')}"
                )

            unit_record = self.state_store.get_module_unit_record(
                task_id=task_id,
                module_name="C",
                unit_id=normalized_shot_id,
            )
            if not unit_record:
                raise RuntimeError(
                    f"模块C定向重试失败：shot_id 不存在或尚未建立单元状态，task_id={task_id}，shot_id={normalized_shot_id}"
                )

            non_done_units = self.state_store.list_module_units_by_status(
                task_id=task_id,
                module_name="C",
                statuses=["pending", "running", "failed"],
            )
            blocking_unit_ids = [str(item["unit_id"]) for item in non_done_units if str(item["unit_id"]) != normalized_shot_id]
            if blocking_unit_ids:
                raise RuntimeError(
                    "模块C定向重试被拒绝：存在其他非done单元，请先清理后再重试。"
                    f"task_id={task_id}，blocking_unit_ids={blocking_unit_ids}"
                )

            self.state_store.reset_module_unit(
                task_id=task_id,
                module_name="C",
                unit_id=normalized_shot_id,
            )
            self.logger.info("模块C定向重试已重置目标单元，task_id=%s，shot_id=%s", task_id, normalized_shot_id)

            self.state_store.update_task_status(task_id=task_id, status="running")
            self._execute_one_module(context=context, module_name="C")
            self._execute_one_module(context=context, module_name="D")

            module_d_record = self.state_store.get_module_record(task_id=task_id, module_name="D")
            output_video_path = Path(module_d_record["artifact_path"]) if module_d_record and module_d_record["artifact_path"] else Path("")
            self.state_store.mark_task_done_if_possible(task_id=task_id, output_video_path=str(output_video_path))
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            summary["retry_shot_id"] = normalized_shot_id
            summary["module_c_unit_summary"] = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="C")
            return summary

    def get_module_d_status_summary(self, task_id: str, config_path: Path) -> dict:
        """
        功能说明：读取指定任务的模块 D 单元状态摘要，用于 CLI 可观测排障。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务状态、模块状态与模块 D 单元摘要。
        异常说明：任务不存在时抛 RuntimeError。
        边界条件：仅查询状态，不触发模块执行。
        """
        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法查询模块D状态：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="d_task_status"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            latest_task = self.state_store.get_task(task_id=task_id) or {}
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            module_d_unit_summary = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="D")
            summary = {
                "task_id": task_id,
                "task_status": latest_task.get("status", "unknown"),
                "module_status": module_status_map,
                "module_d_status": module_status_map.get("D", "unknown"),
                "module_d_unit_summary": module_d_unit_summary,
                "output_video_path": str(latest_task.get("output_video_path", "")),
            }
            self.logger.info(
                "模块D单元状态摘要查询完成，task_id=%s，module_d_status=%s，total_units=%s",
                task_id,
                summary["module_d_status"],
                module_d_unit_summary["total_units"],
            )
            return summary

    def retry_module_d_shot(self, task_id: str, shot_id: str, config_path: Path) -> dict:
        """
        功能说明：按 shot 粒度重试模块 D 单元，并在 D 内重建最新成片。
        参数说明：
        - task_id: 任务唯一标识。
        - shot_id: 模块 D 单元标识（等价模块 C 的 shot_id）。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务执行摘要，并附带本次重试 shot 与模块 D 单元摘要。
        异常说明：任务不存在、上游未完成、单元不存在或模块执行失败时抛 RuntimeError。
        边界条件：仅允许定向重试一个单元，不重跑 A/B/C。
        """
        normalized_shot_id = str(shot_id).strip()
        if not normalized_shot_id:
            raise RuntimeError("shot_id 不能为空。")

        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法重试模块D单元：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="d_retry_shot"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            if module_status_map.get("C") != "done":
                raise RuntimeError(
                    f"模块D定向重试被拒绝：上游模块C未完成，task_id={task_id}，status={module_status_map.get('C')}"
                )

            unit_record = self.state_store.get_module_unit_record(
                task_id=task_id,
                module_name="D",
                unit_id=normalized_shot_id,
            )
            if not unit_record:
                raise RuntimeError(
                    f"模块D定向重试失败：shot_id 不存在或尚未建立单元状态，task_id={task_id}，shot_id={normalized_shot_id}"
                )

            non_done_units = self.state_store.list_module_units_by_status(
                task_id=task_id,
                module_name="D",
                statuses=["pending", "running", "failed"],
            )
            blocking_unit_ids = [str(item["unit_id"]) for item in non_done_units if str(item["unit_id"]) != normalized_shot_id]
            if blocking_unit_ids:
                raise RuntimeError(
                    "模块D定向重试被拒绝：存在其他非done单元，请先清理后再重试。"
                    f"task_id={task_id}，blocking_unit_ids={blocking_unit_ids}"
                )

            self.state_store.reset_module_unit(
                task_id=task_id,
                module_name="D",
                unit_id=normalized_shot_id,
            )
            self.logger.info("模块D定向重试已重置目标单元，task_id=%s，shot_id=%s", task_id, normalized_shot_id)

            self.state_store.update_task_status(task_id=task_id, status="running")
            self._execute_one_module(context=context, module_name="D")

            module_d_record = self.state_store.get_module_record(task_id=task_id, module_name="D")
            output_video_path = Path(module_d_record["artifact_path"]) if module_d_record and module_d_record["artifact_path"] else Path("")
            self.state_store.mark_task_done_if_possible(task_id=task_id, output_video_path=str(output_video_path))
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            summary["retry_shot_id"] = normalized_shot_id
            summary["module_d_unit_summary"] = self.state_store.get_module_unit_status_summary(task_id=task_id, module_name="D")
            return summary

    def get_bcd_status_summary(self, task_id: str, config_path: Path) -> dict:
        """
        功能说明：读取指定任务的跨模块 B/C/D 链路状态摘要，用于并行链路排障。
        参数说明：
        - task_id: 任务唯一标识。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务状态、模块状态与链路级状态摘要。
        异常说明：任务不存在时抛 RuntimeError。
        边界条件：仅查询状态，不触发模块执行。
        """
        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法查询跨模块链路状态：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="bcd_task_status"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            latest_task = self.state_store.get_task(task_id=task_id) or {}
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            chain_rows = self.state_store.list_bcd_chain_status(task_id=task_id)
            chain_status_counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for item in chain_rows:
                chain_status = str(item.get("chain_status", "pending"))
                if chain_status in chain_status_counts:
                    chain_status_counts[chain_status] += 1
            summary = {
                "task_id": task_id,
                "task_status": latest_task.get("status", "unknown"),
                "module_status": module_status_map,
                "module_b_status": module_status_map.get("B", "unknown"),
                "module_c_status": module_status_map.get("C", "unknown"),
                "module_d_status": module_status_map.get("D", "unknown"),
                "bcd_chain_count": len(chain_rows),
                "bcd_chain_status_counts": chain_status_counts,
                "bcd_problem_chains": [item for item in chain_rows if str(item.get("chain_status", "")) != "done"],
                "bcd_chains": chain_rows,
                "output_video_path": str(latest_task.get("output_video_path", "")),
            }
            self.logger.info(
                "跨模块链路状态摘要查询完成，task_id=%s，chain_count=%s，failed_chain_count=%s",
                task_id,
                summary["bcd_chain_count"],
                chain_status_counts["failed"],
            )
            return summary

    def retry_bcd_segment(self, task_id: str, segment_id: str, config_path: Path) -> dict:
        """
        功能说明：按 segment 粒度重置并补跑 B->C->D 链路，不影响其他链路。
        参数说明：
        - task_id: 任务唯一标识。
        - segment_id: 模块 B 单元标识（segment_id）。
        - config_path: 配置文件路径。
        返回值：
        - dict: 任务执行摘要，并附带重试链路信息。
        异常说明：任务不存在、上游未完成、链路不存在或链路补跑失败时抛 RuntimeError。
        边界条件：仅重置目标链路单元，不重置其他链路状态。
        """
        normalized_segment_id = str(segment_id).strip()
        if not normalized_segment_id:
            raise RuntimeError("segment_id 不能为空。")

        task_record = self.state_store.get_task(task_id=task_id)
        if not task_record:
            raise RuntimeError(f"任务不存在，无法重试跨模块链路：task_id={task_id}")

        audio_path = Path(str(task_record["audio_path"]))
        context = self._prepare_context(task_id=task_id, audio_path=audio_path)
        with self._bind_task_log_file(task_dir=context.task_dir, command_name="bcd_retry_segment"):
            self.state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
            module_status_map = self.state_store.get_module_status_map(task_id=task_id)
            if module_status_map.get("A") != "done":
                raise RuntimeError(
                    f"跨模块链路定向重试被拒绝：上游模块A未完成，task_id={task_id}，status={module_status_map.get('A')}"
                )

            reset_result = self.state_store.reset_bcd_chain_units(task_id=task_id, segment_id=normalized_segment_id)
            self.logger.info(
                "跨模块链路定向重试已重置目标链路，task_id=%s，segment_id=%s，unit_index=%s，shot_id=%s",
                task_id,
                reset_result["segment_id"],
                reset_result["unit_index"],
                reset_result["shot_id"],
            )
            self.state_store.update_task_status(task_id=task_id, status="running")
            output_video_path = self._execute_cross_module_bcd(context=context, target_segment_id=normalized_segment_id)
            summary = self._build_summary(task_id=task_id, output_video_path=output_video_path)
            summary["retry_segment_id"] = normalized_segment_id
            summary["retry_unit_index"] = int(reset_result["unit_index"])
            summary["retry_shot_id"] = str(reset_result["shot_id"])
            chain_rows = self.state_store.list_bcd_chain_status(task_id=task_id)
            chain_status_counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for item in chain_rows:
                chain_status = str(item.get("chain_status", "pending"))
                if chain_status in chain_status_counts:
                    chain_status_counts[chain_status] += 1
            summary["bcd_chain_summary"] = {
                "bcd_chain_count": len(chain_rows),
                "bcd_chain_status_counts": chain_status_counts,
                "bcd_problem_chains": [item for item in chain_rows if str(item.get("chain_status", "")) != "done"],
                "bcd_chains": chain_rows,
            }
            return summary

    def _prepare_context(self, task_id: str, audio_path: Path) -> RuntimeContext:
        """
        功能说明：构建并返回模块执行上下文。
        参数说明：
        - task_id: 任务唯一标识。
        - audio_path: 输入音频绝对路径。
        返回值：
        - RuntimeContext: 任务上下文对象。
        异常说明：目录创建失败时抛 OSError。
        边界条件：目录存在时保持幂等。
        """
        task_dir = self.runs_dir / task_id
        artifacts_dir = task_dir / "artifacts"
        ensure_dir(task_dir)
        ensure_dir(artifacts_dir)
        return RuntimeContext(
            task_id=task_id,
            audio_path=audio_path,
            task_dir=task_dir,
            artifacts_dir=artifacts_dir,
            config=self.config,
            logger=self.logger,
            state_store=self.state_store,
        )

    def _execute_modules(self, context: RuntimeContext, start_module: str) -> Path:
        """
        功能说明：从指定模块开始执行到 D，自动跳过已完成模块。
        参数说明：
        - context: 任务上下文。
        - start_module: 起始模块。
        返回值：
        - Path: 当前任务最终视频路径（若尚未生成则返回空路径对象）。
        异常说明：任一模块失败时抛 RuntimeError。
        边界条件：起始模块必须在 MODULE_ORDER 中。
        """
        normalized_start = self._normalize_module_name(start_module)
        if self._should_use_legacy_sequential_path():
            self.logger.warning("检测到自定义模块执行器，run/resume 回退到串行路径，task_id=%s", context.task_id)
            return self._execute_modules_sequential(context=context, start_module=normalized_start)

        if normalized_start == "A":
            status_map = self.state_store.get_module_status_map(task_id=context.task_id)
            if status_map.get("A") != "done":
                self._execute_one_module(context=context, module_name="A")
            else:
                self.logger.info("模块A已完成，自动跳过，task_id=%s", context.task_id)
            return self._execute_cross_module_bcd(context=context, target_segment_id=None)

        if normalized_start in {"B", "C", "D"}:
            return self._execute_cross_module_bcd(context=context, target_segment_id=None)

        raise RuntimeError(f"非法起始模块: {start_module}")

    def _execute_modules_sequential(self, context: RuntimeContext, start_module: str) -> Path:
        """
        功能说明：执行旧版串行模块调度路径（仅用于兼容自定义测试桩）。
        参数说明：
        - context: 任务上下文。
        - start_module: 起始模块。
        返回值：
        - Path: 当前任务最终视频路径（若尚未生成则返回空路径对象）。
        异常说明：任一模块失败时抛 RuntimeError。
        边界条件：默认正式路径不使用该函数。
        """
        start_index = MODULE_ORDER.index(start_module)
        for module_name in MODULE_ORDER[start_index:]:
            status_map = self.state_store.get_module_status_map(task_id=context.task_id)
            if status_map.get(module_name) == "done":
                self.logger.info("模块%s已完成，自动跳过，task_id=%s", module_name, context.task_id)
                continue
            self._execute_one_module(context=context, module_name=module_name)
        module_d_record = self.state_store.get_module_record(task_id=context.task_id, module_name="D")
        output_video_path = Path(module_d_record["artifact_path"]) if module_d_record and module_d_record["artifact_path"] else Path("")
        self.state_store.mark_task_done_if_possible(task_id=context.task_id, output_video_path=str(output_video_path))
        return output_video_path

    def _execute_cross_module_bcd(self, context: RuntimeContext, target_segment_id: str | None) -> Path:
        """
        功能说明：执行跨模块 B/C/D 波前并行调度并返回最新输出视频路径。
        参数说明：
        - context: 任务上下文。
        - target_segment_id: 可选链路筛选条件。
        返回值：
        - Path: 当前任务输出视频路径（若尚未产出则返回空路径对象）。
        异常说明：跨模块调度失败时抛 RuntimeError。
        边界条件：会在内部刷新 B/C/D 模块状态与产物摘要。
        """
        result = run_cross_module_bcd(context=context, target_segment_id=target_segment_id)
        output_video_path_text = str(result.get("output_video_path", "")).strip()
        output_video_path = Path(output_video_path_text) if output_video_path_text else Path("")
        self.state_store.mark_task_done_if_possible(task_id=context.task_id, output_video_path=str(output_video_path))
        return output_video_path

    def _should_use_legacy_sequential_path(self) -> bool:
        """
        功能说明：判断是否启用旧串行路径以兼容自定义模块执行器。
        参数说明：无。
        返回值：
        - bool: True 表示应回退串行执行。
        异常说明：无。
        边界条件：仅当 B/C/D 任一执行器被外部替换时返回 True。
        """
        return any(
            [
                self.module_runners.get("B") is not run_module_b,
                self.module_runners.get("C") is not run_module_c,
                self.module_runners.get("D") is not run_module_d,
            ]
        )

    def _execute_single_module(self, context: RuntimeContext, module_name: str) -> Path:
        """
        功能说明：执行单个模块并刷新任务状态。
        参数说明：
        - context: 任务上下文。
        - module_name: 目标模块名。
        返回值：
        - Path: 若执行 D 则返回视频路径，否则返回空路径对象。
        异常说明：模块执行失败时抛 RuntimeError。
        边界条件：单模块执行不自动触发下游模块。
        """
        self._execute_one_module(context=context, module_name=module_name)
        module_d_record = self.state_store.get_module_record(task_id=context.task_id, module_name="D")
        output_video_path = Path(module_d_record["artifact_path"]) if module_d_record and module_d_record["artifact_path"] else Path("")
        self.state_store.mark_task_done_if_possible(task_id=context.task_id, output_video_path=str(output_video_path))

        task_record = self.state_store.get_task(task_id=context.task_id)
        if task_record and task_record["status"] != "done":
            self.state_store.update_task_status(task_id=context.task_id, status="running")
        return output_video_path

    def _execute_one_module(self, context: RuntimeContext, module_name: str) -> None:
        """
        功能说明：执行单个模块并写入 running/done/failed 状态。
        参数说明：
        - context: 任务上下文。
        - module_name: 模块名。
        返回值：无。
        异常说明：模块异常时封装为 RuntimeError 抛出。
        边界条件：上游未 done 时拒绝执行。
        """
        can_run, reason = self.state_store.can_run_module(task_id=context.task_id, module_name=module_name)
        if not can_run:
            raise RuntimeError(f"模块 {module_name} 无法执行：{reason}")

        self.logger.info("模块%s准备执行，task_id=%s", module_name, context.task_id)
        self.state_store.set_module_status(task_id=context.task_id, module_name=module_name, status="running")

        module_runner = self.module_runners[module_name]
        try:
            artifact_path = module_runner(context)
        except Exception as error:  # noqa: BLE001
            self.state_store.set_module_status(
                task_id=context.task_id,
                module_name=module_name,
                status="failed",
                artifact_path="",
                error_message=str(error),
            )
            self.state_store.update_task_status(task_id=context.task_id, status="failed", error_message=str(error))
            self.logger.error("模块%s执行失败，task_id=%s，错误=%s", module_name, context.task_id, error)
            raise RuntimeError(f"模块 {module_name} 执行失败: {error}") from error

        self.state_store.set_module_status(
            task_id=context.task_id,
            module_name=module_name,
            status="done",
            artifact_path=str(artifact_path),
            error_message="",
        )
        is_module_a_v2 = module_name == "A" and str(context.config.module_a.implementation).lower() == "v2"
        if is_module_a_v2:
            self.logger.debug("模块%s执行完成，task_id=%s，产物=%s", module_name, context.task_id, artifact_path)
        else:
            self.logger.info("模块%s执行完成，task_id=%s，产物=%s", module_name, context.task_id, artifact_path)

    def _resolve_audio_path(self, audio_path: Path) -> Path:
        """
        功能说明：将输入音频路径解析为绝对路径并校验存在性。
        参数说明：
        - audio_path: 原始音频路径。
        返回值：
        - Path: 绝对路径。
        异常说明：文件不存在时抛 FileNotFoundError。
        边界条件：相对路径默认相对于 workspace_root。
        """
        resolved_path = audio_path if audio_path.is_absolute() else (self.workspace_root / audio_path)
        resolved_path = resolved_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {resolved_path}")
        return resolved_path

    def _normalize_module_name(self, module_name: str) -> str:
        """
        功能说明：标准化模块名并校验合法性。
        参数说明：
        - module_name: 原始模块名字符串。
        返回值：
        - str: 大写模块名。
        异常说明：非法模块名时抛 ValueError。
        边界条件：仅允许 A/B/C/D。
        """
        normalized = module_name.upper().strip()
        if normalized not in VALID_MODULES:
            raise ValueError(f"非法模块名: {module_name}，合法值={sorted(VALID_MODULES)}")
        return normalized

    def _build_summary(self, task_id: str, output_video_path: Path) -> dict:
        """
        功能说明：构建任务执行摘要。
        参数说明：
        - task_id: 任务唯一标识。
        - output_video_path: 输出视频路径。
        返回值：
        - dict: 包含任务状态、模块状态与输出路径的摘要字典。
        异常说明：查询失败时抛 sqlite3.Error。
        边界条件：视频路径不存在时返回空字符串。
        """
        task_record = self.state_store.get_task(task_id=task_id) or {}
        module_status_map = self.state_store.get_module_status_map(task_id=task_id)
        return {
            "task_id": task_id,
            "task_status": task_record.get("status", "unknown"),
            "module_status": module_status_map,
            "output_video_path": str(output_video_path) if str(output_video_path) else "",
        }
