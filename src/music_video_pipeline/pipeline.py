"""
文件用途：实现 MVP 流水线调度器。
核心流程：按 A->B->C->D 顺序执行模块，并由状态库统一管理恢复逻辑。
输入输出：输入任务参数，输出执行摘要（字典）。
依赖说明：依赖项目内模块实现、状态存储、上下文对象。
维护说明：调度层不应包含模型细节，只负责流程与状态控制。
"""

# 标准库：用于日志记录
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于可调用类型提示
from typing import Callable

# 项目内模块：配置类型
from music_video_pipeline.config import AppConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：常量定义
from music_video_pipeline.constants import MODULE_ORDER, VALID_MODULES
# 项目内模块：目录工具
from music_video_pipeline.io_utils import ensure_dir
# 项目内模块：模块执行函数
from music_video_pipeline.modules import run_module_a, run_module_b, run_module_c, run_module_d
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore

ModuleRunner = Callable[[RuntimeContext], Path]


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
            "A": run_module_a,
            "B": run_module_b,
            "C": run_module_c,
            "D": run_module_d,
        }

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
        start_index = MODULE_ORDER.index(normalized_start)
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
