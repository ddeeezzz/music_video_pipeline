"""
文件用途：实现模块 B 最小视觉单元并行执行与重试逻辑。
核心流程：按待执行单元并行生成分镜，失败单元按配置重试并写入单元状态。
输入输出：输入运行上下文、单元数组与生成器，输出执行副作用（状态与分镜文件）。
依赖说明：依赖标准库并发工具与项目内 RuntimeContext/ScriptGenerator。
维护说明：本层只负责 B 内部并行，不改变 A->B->C->D 的模块顺序。
"""

# 标准库：用于线程池并发
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则清洗
import re
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON写入工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：分镜生成器抽象
from music_video_pipeline.generators import ScriptGenerator
# 项目内模块：模块B单元数据模型
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit


def execute_units_with_retry(
    context: RuntimeContext,
    units_to_run: list[ModuleBUnit],
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Path,
) -> None:
    """
    功能说明：执行模块 B 待处理单元，并在失败时按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 单元分镜输出目录。
    返回值：无。
    异常说明：
    - RuntimeError: 单元重试耗尽仍失败时抛出。
    边界条件：已完成单元由上层过滤，不在本函数内重跑。
    """
    if not units_to_run:
        context.logger.info("模块B无待执行单元，task_id=%s", context.task_id)
        return

    worker_count = _normalize_module_b_workers(context.config.module_b.script_workers)
    retry_times = _normalize_module_b_retry_times(context.config.module_b.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    hard_fail_messages: list[str] = []

    for attempt_index in range(retry_times + 1):
        if not pending_units:
            break
        attempt_no = attempt_index + 1
        context.logger.info(
            "模块B单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s",
            context.task_id,
            attempt_no,
            retry_times + 1,
            len(pending_units),
            worker_count,
        )

        for unit in pending_units:
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="B",
                unit_id=unit.unit_id,
                status="running",
                artifact_path="",
                error_message="",
            )

        if worker_count == 1:
            failed_units = _execute_units_serial(
                context=context,
                pending_units=pending_units,
                generator=generator,
                module_a_output=module_a_output,
                unit_outputs_dir=unit_outputs_dir,
            )
        else:
            failed_units = _execute_units_parallel(
                context=context,
                pending_units=pending_units,
                generator=generator,
                module_a_output=module_a_output,
                unit_outputs_dir=unit_outputs_dir,
                worker_count=worker_count,
            )

        if not failed_units:
            pending_units = []
            continue

        if attempt_index < retry_times:
            context.logger.warning(
                "模块B单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s",
                context.task_id,
                attempt_no,
                retry_times + 1,
                len(failed_units),
            )
            pending_units = [unit for unit, _ in failed_units]
            continue

        for failed_unit, failed_error in failed_units:
            hard_fail_messages.append(f"{failed_unit.unit_id}: {failed_error}")
        pending_units = []

    if hard_fail_messages:
        error_text = "\n".join(hard_fail_messages)
        raise RuntimeError(f"模块B单元生成失败，共{len(hard_fail_messages)}个单元失败：\n{error_text}")


def execute_one_unit_with_retry(
    context: RuntimeContext,
    unit: ModuleBUnit,
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Path,
    retry_times: int | None = None,
) -> Path:
    """
    功能说明：执行单个模块 B 单元并按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 单元分镜输出目录。
    - retry_times: 可选重试次数，传空时读取模块配置。
    返回值：
    - Path: 单元分镜 JSON 路径。
    异常说明：
    - RuntimeError: 重试耗尽后抛出。
    边界条件：每次尝试前都会写入 running 状态。
    """
    normalized_retry_times = (
        _normalize_module_b_retry_times(context.config.module_b.unit_retry_times)
        if retry_times is None
        else _normalize_module_b_retry_times(retry_times)
    )
    last_error: Exception | None = None
    for attempt_index in range(normalized_retry_times + 1):
        attempt_no = attempt_index + 1
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="running",
            artifact_path="",
            error_message="",
        )
        try:
            shot_path = _generate_and_dump_one_shot(
                generator=generator,
                module_a_output=module_a_output,
                unit=unit,
                unit_outputs_dir=unit_outputs_dir,
            )
            _mark_unit_done(context=context, unit=unit, shot_path=shot_path)
            return shot_path
        except Exception as error:  # noqa: BLE001
            last_error = error
            _mark_unit_failed(context=context, unit=unit, error=error)
            if attempt_index < normalized_retry_times:
                context.logger.warning(
                    "模块B单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s",
                    context.task_id,
                    unit.unit_id,
                    attempt_no,
                    normalized_retry_times + 1,
                    error,
                )
                continue
            break
    raise RuntimeError(f"模块B单元执行失败，unit_id={unit.unit_id}，错误={last_error}")


def _execute_units_serial(
    context: RuntimeContext,
    pending_units: list[ModuleBUnit],
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Path,
) -> list[tuple[ModuleBUnit, Exception]]:
    """
    功能说明：串行执行模块 B 单元生成任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 单元分镜输出目录。
    返回值：
    - list[tuple[ModuleBUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：执行顺序固定按 unit_index 升序。
    """
    failed_units: list[tuple[ModuleBUnit, Exception]] = []
    for unit in pending_units:
        try:
            shot_path = _generate_and_dump_one_shot(
                generator=generator,
                module_a_output=module_a_output,
                unit=unit,
                unit_outputs_dir=unit_outputs_dir,
            )
            _mark_unit_done(context=context, unit=unit, shot_path=shot_path)
        except Exception as error:  # noqa: BLE001
            _mark_unit_failed(context=context, unit=unit, error=error)
            failed_units.append((unit, error))
    return failed_units


def _execute_units_parallel(
    context: RuntimeContext,
    pending_units: list[ModuleBUnit],
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Path,
    worker_count: int,
) -> list[tuple[ModuleBUnit, Exception]]:
    """
    功能说明：并行执行模块 B 单元生成任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 单元分镜输出目录。
    - worker_count: 并行 worker 数量。
    返回值：
    - list[tuple[ModuleBUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：状态写入统一在主线程完成，避免并发写库冲突。
    """
    failed_units: list[tuple[ModuleBUnit, Exception]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_unit = {
            executor.submit(
                _generate_and_dump_one_shot,
                generator,
                module_a_output,
                unit,
                unit_outputs_dir,
            ): unit
            for unit in pending_units
        }
        for future in as_completed(future_to_unit):
            unit = future_to_unit[future]
            try:
                shot_path = future.result()
                _mark_unit_done(context=context, unit=unit, shot_path=shot_path)
            except Exception as error:  # noqa: BLE001
                _mark_unit_failed(context=context, unit=unit, error=error)
                failed_units.append((unit, error))
    return failed_units


def _generate_and_dump_one_shot(
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit: ModuleBUnit,
    unit_outputs_dir: Path,
) -> Path:
    """
    功能说明：调用生成器执行单元分镜生成并落盘到单元JSON文件。
    参数说明：
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit: 模块 B 单元对象。
    - unit_outputs_dir: 单元分镜输出目录。
    返回值：
    - Path: 单元分镜JSON路径。
    异常说明：由生成器实现或JSON写入抛出异常。
    边界条件：单元输出文件名包含 unit_index 与 unit_id，保证唯一与可追溯。
    """
    shot = generator.generate_one(
        module_a_output=module_a_output,
        segment=unit.segment,
        segment_index=unit.unit_index,
    )
    if not isinstance(shot, dict):
        raise RuntimeError(f"模块B单元执行失败：返回值不是dict，unit_id={unit.unit_id}")

    unit_outputs_dir.mkdir(parents=True, exist_ok=True)
    safe_unit_id = _safe_unit_id(unit.unit_id)
    shot_path = unit_outputs_dir / f"segment_{unit.unit_index + 1:03d}_{safe_unit_id}.json"
    write_json(shot_path, shot)
    return shot_path


def _mark_unit_done(context: RuntimeContext, unit: ModuleBUnit, shot_path: Path) -> None:
    """
    功能说明：将单元状态写入 done 并记录产物路径。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 B 单元对象。
    - shot_path: 单元分镜JSON路径。
    返回值：无。
    异常说明：数据库写入失败时抛出 sqlite3.Error。
    边界条件：shot_path 必须存在。
    """
    if not shot_path.exists():
        raise RuntimeError(f"模块B单元执行失败：单元分镜文件不存在，unit_id={unit.unit_id}")
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="B",
        unit_id=unit.unit_id,
        status="done",
        artifact_path=str(shot_path),
        error_message="",
    )
    context.logger.info("模块B单元执行完成，task_id=%s，unit_id=%s，shot=%s", context.task_id, unit.unit_id, shot_path)


def _mark_unit_failed(context: RuntimeContext, unit: ModuleBUnit, error: Exception) -> None:
    """
    功能说明：将单元状态写入 failed 并记录错误。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 B 单元对象。
    - error: 执行异常。
    返回值：无。
    异常说明：数据库写入失败时抛出 sqlite3.Error。
    边界条件：错误文本会被直接写入状态库用于恢复排障。
    """
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="B",
        unit_id=unit.unit_id,
        status="failed",
        artifact_path="",
        error_message=str(error),
    )
    context.logger.error("模块B单元执行失败，task_id=%s，unit_id=%s，错误=%s", context.task_id, unit.unit_id, error)


def _normalize_module_b_workers(script_workers: int) -> int:
    """
    功能说明：归一化模块 B 并行 worker 数量。
    参数说明：
    - script_workers: 原始 worker 配置值。
    返回值：
    - int: 合法 worker 数量（范围 1~8）。
    异常说明：无。
    边界条件：非法值统一回退为 3。
    """
    try:
        normalized = int(script_workers)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 8:
        return 8
    return normalized


def _normalize_module_b_retry_times(unit_retry_times: int) -> int:
    """
    功能说明：归一化模块 B 单元重试次数。
    参数说明：
    - unit_retry_times: 原始重试次数配置值。
    返回值：
    - int: 合法重试次数（范围 0~5）。
    异常说明：无。
    边界条件：非法值统一回退为 1。
    """
    try:
        normalized = int(unit_retry_times)
    except (TypeError, ValueError):
        return 1
    if normalized < 0:
        return 1
    if normalized > 5:
        return 5
    return normalized


def _safe_unit_id(unit_id: str) -> str:
    """
    功能说明：将单元ID转换为文件名安全文本。
    参数说明：
    - unit_id: 原始单元ID。
    返回值：
    - str: 仅含字母数字与下划线的安全字符串。
    异常说明：无。
    边界条件：空文本回退为 unknown。
    """
    safe_text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(unit_id)).strip("_")
    return safe_text or "unknown"
