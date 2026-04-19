"""
文件用途：实现模块 C 最小视觉单元并行执行与重试逻辑。
核心流程：按待执行单元批量并行生成关键帧，失败单元按配置重试并写入单元状态。
输入输出：输入运行上下文、单元数组与生成器，输出执行副作用（状态与图像文件）。
依赖说明：依赖标准库并发工具与项目内 RuntimeContext/FrameGenerator。
维护说明：本层只负责 C 内部并行，不改变 A->B->C->D 的模块顺序。
"""

# 标准库：用于线程池并发
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：关键帧生成器抽象
from music_video_pipeline.generators import FrameGenerator
# 项目内模块：模块C单元数据模型
from music_video_pipeline.modules.module_c.unit_models import ModuleCUnit


def execute_units_with_retry(
    context: RuntimeContext,
    units_to_run: list[ModuleCUnit],
    generator: FrameGenerator,
    frames_dir: Path,
) -> None:
    """
    功能说明：执行模块 C 待处理单元，并在失败时按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    - generator: 关键帧生成器实例。
    - frames_dir: 帧输出目录。
    返回值：无。
    异常说明：
    - RuntimeError: 单元重试耗尽仍失败时抛出。
    边界条件：已完成单元由上层过滤，不在本函数内重跑。
    """
    if not units_to_run:
        context.logger.info("模块C无待执行单元，task_id=%s", context.task_id)
        return

    worker_count = _normalize_module_c_workers(context.config.module_c.render_workers)
    retry_times = _normalize_module_c_retry_times(context.config.module_c.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    hard_fail_messages: list[str] = []

    for attempt_index in range(retry_times + 1):
        if not pending_units:
            break
        attempt_no = attempt_index + 1
        context.logger.info(
            "模块C单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s",
            context.task_id,
            attempt_no,
            retry_times + 1,
            len(pending_units),
            worker_count,
        )

        for unit in pending_units:
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="C",
                unit_id=unit.unit_id,
                status="running",
                artifact_path="",
                error_message="",
            )

        failed_units: list[tuple[ModuleCUnit, Exception]] = []
        if worker_count == 1:
            failed_units = _execute_units_serial(
                context=context,
                pending_units=pending_units,
                generator=generator,
                frames_dir=frames_dir,
            )
        else:
            failed_units = _execute_units_parallel(
                context=context,
                pending_units=pending_units,
                generator=generator,
                frames_dir=frames_dir,
                worker_count=worker_count,
            )

        if not failed_units:
            pending_units = []
            continue

        if attempt_index < retry_times:
            context.logger.warning(
                "模块C单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s",
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
        raise RuntimeError(f"模块C单元渲染失败，共{len(hard_fail_messages)}个单元失败：\n{error_text}")


def execute_one_unit_with_retry(
    context: RuntimeContext,
    unit: ModuleCUnit,
    generator: FrameGenerator,
    frames_dir: Path,
    retry_times: int | None = None,
) -> dict[str, Any]:
    """
    功能说明：执行单个模块 C 单元并按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    - generator: 关键帧生成器实例。
    - frames_dir: 帧输出目录。
    - retry_times: 可选重试次数，传空时读取模块配置。
    返回值：
    - dict[str, Any]: 单元 frame_item。
    异常说明：
    - RuntimeError: 重试耗尽后抛出。
    边界条件：每次尝试前都会写入 running 状态。
    """
    normalized_retry_times = (
        _normalize_module_c_retry_times(context.config.module_c.unit_retry_times)
        if retry_times is None
        else _normalize_module_c_retry_times(retry_times)
    )
    last_error: Exception | None = None
    for attempt_index in range(normalized_retry_times + 1):
        attempt_no = attempt_index + 1
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="C",
            unit_id=unit.unit_id,
            status="running",
            artifact_path="",
            error_message="",
        )
        try:
            frame_item = _generate_one_frame_item(
                generator=generator,
                unit=unit,
                frames_dir=frames_dir,
                width=context.config.render.video_width,
                height=context.config.render.video_height,
            )
            _mark_unit_done(context=context, unit=unit, frame_item=frame_item)
            return frame_item
        except Exception as error:  # noqa: BLE001
            last_error = error
            _mark_unit_failed(context=context, unit=unit, error=error)
            if attempt_index < normalized_retry_times:
                context.logger.warning(
                    "模块C单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s",
                    context.task_id,
                    unit.unit_id,
                    attempt_no,
                    normalized_retry_times + 1,
                    error,
                )
                continue
            break
    raise RuntimeError(f"模块C单元执行失败，unit_id={unit.unit_id}，错误={last_error}")


def _execute_units_serial(
    context: RuntimeContext,
    pending_units: list[ModuleCUnit],
    generator: FrameGenerator,
    frames_dir: Path,
) -> list[tuple[ModuleCUnit, Exception]]:
    """
    功能说明：串行执行模块 C 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - generator: 关键帧生成器实例。
    - frames_dir: 帧输出目录。
    返回值：
    - list[tuple[ModuleCUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：执行顺序固定按 unit_index 升序。
    """
    failed_units: list[tuple[ModuleCUnit, Exception]] = []
    for unit in pending_units:
        try:
            frame_item = _generate_one_frame_item(
                generator=generator,
                unit=unit,
                frames_dir=frames_dir,
                width=context.config.render.video_width,
                height=context.config.render.video_height,
            )
            _mark_unit_done(context=context, unit=unit, frame_item=frame_item)
        except Exception as error:  # noqa: BLE001
            _mark_unit_failed(context=context, unit=unit, error=error)
            failed_units.append((unit, error))
    return failed_units


def _execute_units_parallel(
    context: RuntimeContext,
    pending_units: list[ModuleCUnit],
    generator: FrameGenerator,
    frames_dir: Path,
    worker_count: int,
) -> list[tuple[ModuleCUnit, Exception]]:
    """
    功能说明：并行执行模块 C 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - generator: 关键帧生成器实例。
    - frames_dir: 帧输出目录。
    - worker_count: 并行 worker 数量。
    返回值：
    - list[tuple[ModuleCUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：状态写入统一在主线程完成，避免并发写库冲突。
    """
    failed_units: list[tuple[ModuleCUnit, Exception]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_unit = {
            executor.submit(
                _generate_one_frame_item,
                generator,
                unit,
                frames_dir,
                context.config.render.video_width,
                context.config.render.video_height,
            ): unit
            for unit in pending_units
        }
        for future in as_completed(future_to_unit):
            unit = future_to_unit[future]
            try:
                frame_item = future.result()
                _mark_unit_done(context=context, unit=unit, frame_item=frame_item)
            except Exception as error:  # noqa: BLE001
                _mark_unit_failed(context=context, unit=unit, error=error)
                failed_units.append((unit, error))
    return failed_units


def _generate_one_frame_item(
    generator: FrameGenerator,
    unit: ModuleCUnit,
    frames_dir: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    功能说明：调用生成器执行单元渲染并返回 frame_item。
    参数说明：
    - generator: 关键帧生成器实例。
    - unit: 模块 C 单元对象。
    - frames_dir: 帧输出目录。
    - width: 图像宽度。
    - height: 图像高度。
    返回值：
    - dict[str, Any]: 单元渲染结果 frame_item。
    异常说明：由生成器实现抛出异常。
    边界条件：frame_item 结构需兼容模块 D 消费字段。
    """
    return generator.generate_one(
        shot=unit.shot,
        output_dir=frames_dir,
        width=width,
        height=height,
        shot_index=unit.unit_index,
    )


def _mark_unit_done(context: RuntimeContext, unit: ModuleCUnit, frame_item: dict[str, Any]) -> None:
    """
    功能说明：将单元状态写入 done 并记录产物路径。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 C 单元对象。
    - frame_item: 渲染返回结构。
    返回值：无。
    异常说明：数据库写入失败时抛出 sqlite3.Error。
    边界条件：frame_item 缺失 frame_path 时视为失败。
    """
    frame_path = str(frame_item.get("frame_path", "")).strip()
    if not frame_path:
        raise RuntimeError(f"模块C单元执行失败：未返回有效 frame_path，unit_id={unit.unit_id}")
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="C",
        unit_id=unit.unit_id,
        status="done",
        artifact_path=frame_path,
        error_message="",
    )
    context.logger.info("模块C单元执行完成，task_id=%s，unit_id=%s，frame=%s", context.task_id, unit.unit_id, frame_path)


def _mark_unit_failed(context: RuntimeContext, unit: ModuleCUnit, error: Exception) -> None:
    """
    功能说明：将单元状态写入 failed 并记录错误。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 C 单元对象。
    - error: 执行异常。
    返回值：无。
    异常说明：数据库写入失败时抛出 sqlite3.Error。
    边界条件：错误文本会被直接写入状态库用于恢复排障。
    """
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="C",
        unit_id=unit.unit_id,
        status="failed",
        artifact_path="",
        error_message=str(error),
    )
    context.logger.error("模块C单元执行失败，task_id=%s，unit_id=%s，错误=%s", context.task_id, unit.unit_id, error)


def _normalize_module_c_workers(render_workers: int) -> int:
    """
    功能说明：归一化模块 C 并行 worker 数量。
    参数说明：
    - render_workers: 原始 worker 配置值。
    返回值：
    - int: 合法 worker 数量（范围 1~8）。
    异常说明：无。
    边界条件：非法值统一回退为 3。
    """
    try:
        normalized = int(render_workers)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 8:
        return 8
    return normalized


def _normalize_module_c_retry_times(unit_retry_times: int) -> int:
    """
    功能说明：归一化模块 C 单元重试次数。
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
