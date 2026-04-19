"""
文件用途：实现模块 B 最小视觉单元串行执行、重试与滚动记忆写入逻辑。
核心流程：按 unit_index 严格串行生成分镜，每段生成前重建 memory_context，成功后更新 rolling_memory。
输入输出：输入运行上下文、单元数组与生成器，输出执行副作用（状态、分镜文件与 rolling_memory 文件）。
依赖说明：依赖项目内 RuntimeContext/ScriptGenerator 与 JSON 工具。
维护说明：本层仅负责模块 B 主链路串行执行，不改写 A->B->C->D 的模块顺序。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则清洗
import re
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON读写工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：分镜生成器抽象
from music_video_pipeline.generators import ScriptGenerator
# 项目内模块：模块B单元数据模型
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit


# 常量：滚动记忆文件名。
ROLLING_MEMORY_FILENAME = "rolling_memory.json"
# 常量：recent_history 默认滑动窗口大小。
ROLLING_MEMORY_WINDOW_SIZE = 5


def execute_units_with_retry(
    context: RuntimeContext,
    units_to_run: list[ModuleBUnit],
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Path,
) -> None:
    """
    功能说明：按 unit_index 严格串行执行模块 B 单元，并在失败时仅重试当前单元。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit_outputs_dir: 单元分镜输出目录。
    返回值：无。
    异常说明：
    - RuntimeError: 任一单元重试耗尽仍失败时抛出（fail-fast，后续单元不再执行）。
    边界条件：
    - 每个单元执行前都会基于“前序 done 单元”重建 memory_context，并写入 rolling_memory.json。
    """
    if not units_to_run:
        context.logger.info("模块B无待执行单元，task_id=%s", context.task_id)
        return

    retry_times = _normalize_module_b_retry_times(context.config.module_b.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    segment_meta_index = _build_segment_meta_index(module_a_output=module_a_output)
    rolling_memory_path = unit_outputs_dir / ROLLING_MEMORY_FILENAME

    context.logger.info(
        "模块B串行执行开始，task_id=%s，pending_count=%s，retry_times=%s",
        context.task_id,
        len(pending_units),
        retry_times,
    )

    for unit in pending_units:
        success = False
        last_error: Exception | None = None
        for attempt_index in range(retry_times + 1):
            attempt_no = attempt_index + 1
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="B",
                unit_id=unit.unit_id,
                status="running",
                artifact_path="",
                error_message="",
            )

            # 每次尝试前都基于“前序 done 单元”重建记忆，避免未来分镜信息泄漏。
            memory_context = _build_memory_context_from_done_units(
                context=context,
                current_unit_index=unit.unit_index,
                segment_meta_index=segment_meta_index,
            )
            _write_rolling_memory(memory_path=rolling_memory_path, memory_context=memory_context)

            try:
                shot_path = _generate_and_dump_one_shot(
                    generator=generator,
                    module_a_output=module_a_output,
                    unit=unit,
                    unit_outputs_dir=unit_outputs_dir,
                    memory_context=memory_context,
                )
                _mark_unit_done(context=context, unit=unit, shot_path=shot_path)

                shot_obj = read_json(shot_path)
                if not isinstance(shot_obj, dict):
                    raise RuntimeError(f"模块B单元执行失败：单元分镜内容不是dict，unit_id={unit.unit_id}")
                memory_item = _build_memory_item(
                    unit=unit,
                    shot_obj=shot_obj,
                    segment_meta_index=segment_meta_index,
                )
                updated_memory_context = _append_memory_item(
                    memory_context=memory_context,
                    memory_item=memory_item,
                )
                _write_rolling_memory(memory_path=rolling_memory_path, memory_context=updated_memory_context)
                success = True
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                _mark_unit_failed(context=context, unit=unit, error=error)
                if attempt_index < retry_times:
                    context.logger.warning(
                        "模块B串行单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s",
                        context.task_id,
                        unit.unit_id,
                        attempt_no,
                        retry_times + 1,
                        error,
                    )
                    continue

        if not success:
            raise RuntimeError(
                f"模块B单元执行失败（已停止后续单元），unit_id={unit.unit_id}，错误={last_error}"
            )


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


def _build_memory_context_from_done_units(
    context: RuntimeContext,
    current_unit_index: int,
    segment_meta_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    功能说明：基于当前单元之前的 done 单元重建 memory_context。
    参数说明：
    - context: 运行上下文对象。
    - current_unit_index: 当前目标单元索引（仅收集更小索引的 done 单元）。
    - segment_meta_index: segment_id 到段落元信息映射。
    返回值：
    - dict[str, Any]: memory_context 对象。
    异常说明：
    - RuntimeError: done 单元产物不存在或内容非法时抛出。
    边界条件：recent_history 仅保留最近 5 条。
    """
    done_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
    previous_records = [
        item for item in done_records if int(item.get("unit_index", 0)) < int(current_unit_index)
    ]
    ordered_records = sorted(previous_records, key=lambda item: int(item.get("unit_index", 0)))

    history_items: list[dict[str, Any]] = []
    for record in ordered_records:
        artifact_path_text = str(record.get("artifact_path", "")).strip()
        if not artifact_path_text:
            raise RuntimeError(f"模块B滚动记忆重建失败：空 artifact_path，record={record}")
        artifact_path = Path(artifact_path_text)
        if not artifact_path.exists():
            raise RuntimeError(f"模块B滚动记忆重建失败：产物文件不存在，path={artifact_path}")

        shot_obj = read_json(artifact_path)
        if not isinstance(shot_obj, dict):
            raise RuntimeError(f"模块B滚动记忆重建失败：产物内容不是dict，path={artifact_path}")

        segment_id = str(record.get("unit_id", "")).strip()
        segment_meta = segment_meta_index.get(segment_id, {})
        history_items.append(
            {
                "segment_id": segment_id,
                "lyric_text": str(shot_obj.get("lyric_text", "")),
                "generated_scene": str(shot_obj.get("scene_desc", "")),
                "keyframe_prompt": str(shot_obj.get("keyframe_prompt", "")),
                "start_time": float(segment_meta.get("start_time", record.get("start_time", 0.0))),
                "end_time": float(segment_meta.get("end_time", record.get("end_time", 0.0))),
                "segment_label": str(segment_meta.get("segment_label", "")),
                "big_segment_label": str(segment_meta.get("big_segment_label", "")),
            }
        )

    recent_history = history_items[-ROLLING_MEMORY_WINDOW_SIZE:]
    current_state = str(recent_history[-1].get("generated_scene", "")) if recent_history else ""
    return {
        "global_setting": "",
        "current_state": current_state,
        "recent_history": recent_history,
    }


def _build_segment_meta_index(module_a_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    功能说明：构建 segment_id 到段落元信息映射，用于 rolling memory 辅助字段。
    参数说明：
    - module_a_output: 模块A输出字典。
    返回值：
    - dict[str, dict[str, Any]]: 元信息映射。
    异常说明：无。
    边界条件：缺失字段时回退空字符串或 0。
    """
    segments = module_a_output.get("segments", [])
    big_segments = module_a_output.get("big_segments", [])
    big_label_by_id = {
        str(item.get("segment_id", "")).strip(): str(item.get("label", "")).strip()
        for item in big_segments
        if isinstance(item, dict)
    }

    index: dict[str, dict[str, Any]] = {}
    if not isinstance(segments, list):
        return index
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            continue
        big_segment_id = str(segment.get("big_segment_id", "")).strip()
        start_time = float(segment.get("start_time", 0.0))
        end_time = max(start_time, float(segment.get("end_time", start_time)))
        index[segment_id] = {
            "start_time": start_time,
            "end_time": end_time,
            "segment_label": str(segment.get("label", "")).strip(),
            "big_segment_label": big_label_by_id.get(big_segment_id, ""),
        }
    return index


def _build_memory_item(
    unit: ModuleBUnit,
    shot_obj: dict[str, Any],
    segment_meta_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    功能说明：基于当前单元输出构建一条 recent_history 记录。
    参数说明：
    - unit: 当前模块B单元。
    - shot_obj: 当前单元产物 JSON 对象。
    - segment_meta_index: segment_id 到段落元信息映射。
    返回值：
    - dict[str, Any]: 可写入 recent_history 的记录。
    异常说明：无。
    边界条件：关键字段缺失时回退为空字符串。
    """
    segment_meta = segment_meta_index.get(unit.unit_id, {})
    return {
        "segment_id": unit.unit_id,
        "lyric_text": str(shot_obj.get("lyric_text", "")),
        "generated_scene": str(shot_obj.get("scene_desc", "")),
        "keyframe_prompt": str(shot_obj.get("keyframe_prompt", "")),
        "start_time": float(segment_meta.get("start_time", unit.start_time)),
        "end_time": float(segment_meta.get("end_time", unit.end_time)),
        "segment_label": str(segment_meta.get("segment_label", str(unit.segment.get("label", "")))),
        "big_segment_label": str(segment_meta.get("big_segment_label", "")),
    }


def _append_memory_item(memory_context: dict[str, Any], memory_item: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：将新分镜写入 recent_history 并维护滑动窗口。
    参数说明：
    - memory_context: 旧 memory_context。
    - memory_item: 新增历史记录。
    返回值：
    - dict[str, Any]: 更新后的 memory_context。
    异常说明：无。
    边界条件：recent_history 最多保留 5 条。
    """
    old_history = memory_context.get("recent_history", [])
    history = [dict(item) for item in old_history if isinstance(item, dict)]
    history.append(dict(memory_item))
    recent_history = history[-ROLLING_MEMORY_WINDOW_SIZE:]
    current_state = str(memory_item.get("generated_scene", "")).strip()
    return {
        "global_setting": str(memory_context.get("global_setting", "")),
        "current_state": current_state,
        "recent_history": recent_history,
    }


def _write_rolling_memory(memory_path: Path, memory_context: dict[str, Any]) -> None:
    """
    功能说明：覆盖写入 rolling_memory.json。
    参数说明：
    - memory_path: 记忆文件路径。
    - memory_context: 记忆对象。
    返回值：无。
    异常说明：文件写入失败时抛 OSError。
    边界条件：会自动创建父目录。
    """
    write_json(memory_path, memory_context)


def _generate_and_dump_one_shot(
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit: ModuleBUnit,
    unit_outputs_dir: Path,
    memory_context: dict[str, Any] | None = None,
) -> Path:
    """
    功能说明：调用生成器执行单元分镜生成并落盘到单元JSON文件。
    参数说明：
    - generator: 分镜生成器实例。
    - module_a_output: 模块 A 输出字典。
    - unit: 模块 B 单元对象。
    - unit_outputs_dir: 单元分镜输出目录。
    - memory_context: 可选，滚动记忆上下文。
    返回值：
    - Path: 单元分镜JSON路径。
    异常说明：由生成器实现或JSON写入抛出异常。
    边界条件：单元输出文件名包含 unit_index 与 unit_id，保证唯一与可追溯。
    """
    shot = _invoke_generate_one_with_optional_memory(
        generator=generator,
        module_a_output=module_a_output,
        unit=unit,
        memory_context=memory_context,
    )
    if not isinstance(shot, dict):
        raise RuntimeError(f"模块B单元执行失败：返回值不是dict，unit_id={unit.unit_id}")

    unit_outputs_dir.mkdir(parents=True, exist_ok=True)
    safe_unit_id = _safe_unit_id(unit.unit_id)
    shot_path = unit_outputs_dir / f"segment_{unit.unit_index + 1:03d}_{safe_unit_id}.json"
    write_json(shot_path, shot)
    return shot_path


def _invoke_generate_one_with_optional_memory(
    generator: ScriptGenerator,
    module_a_output: dict[str, Any],
    unit: ModuleBUnit,
    memory_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    功能说明：兼容调用 generate_one（支持/不支持 memory_context 参数的实现）。
    参数说明：
    - generator: 分镜生成器实例。
    - module_a_output: 模块A输出字典。
    - unit: 当前模块B单元。
    - memory_context: 可选滚动记忆上下文。
    返回值：
    - dict[str, Any]: 生成的分镜对象。
    异常说明：透传 generate_one 抛出的异常。
    边界条件：不支持 memory_context 的历史实现会自动回退为旧调用方式。
    """
    if memory_context is None:
        return generator.generate_one(
            module_a_output=module_a_output,
            segment=unit.segment,
            segment_index=unit.unit_index,
        )

    try:
        return generator.generate_one(
            module_a_output=module_a_output,
            segment=unit.segment,
            segment_index=unit.unit_index,
            memory_context=memory_context,
        )
    except TypeError as error:
        if "memory_context" not in str(error):
            raise
        return generator.generate_one(
            module_a_output=module_a_output,
            segment=unit.segment,
            segment_index=unit.unit_index,
        )


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
