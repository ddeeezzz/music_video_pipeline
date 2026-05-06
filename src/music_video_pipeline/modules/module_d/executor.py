"""
文件用途：实现模块 D 的纯 ComfyUI 单元执行与重试逻辑。
核心流程：预热 ComfyUI 服务 -> 解析模块 D 单元视频提示词 -> 执行单元渲染 -> 写入状态与产物路径。
输入输出：输入运行上下文与模块 D 单元，输出片段路径或执行副作用。
依赖说明：依赖标准库并发工具与项目内 RuntimeContext/ModuleDUnit/ComfyUI 渲染后端。
维护说明：模块 D 已彻底收口为 ComfyUI 路径，执行器只负责任务调度、重试与状态写回。
"""

# 标准库：用于线程池并发。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：运行上下文定义。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具（用于补取模块 B 的 video_prompt_en）。
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块 D ComfyUI 渲染后端。
from music_video_pipeline.modules.module_d.backends import (
    prewarm_comfyui_runtime as prewarm_comfyui_runtime_backend,
    render_one_unit_comfyui,
)
# 项目内模块：模块 D FFmpeg 后处理工具。
from music_video_pipeline.modules.module_d.finalizer import apply_camera_plan_to_segment
# 项目内模块：模块 D 单元模型。
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit


def prewarm_comfyui_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, str]:
    """
    功能说明：预热模块 D 的 ComfyUI runtime。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 预留字段；当前 HTTP ComfyUI 路径不直接消费该值。
    返回值：
    - dict[str, str]: 预热摘要。
    异常说明：
    - RuntimeError: ComfyUI 服务探活或契约加载失败时抛出。
    边界条件：仅校验服务与工作流，不执行真实推理。
    """
    return prewarm_comfyui_runtime_backend(context=context, device_override=device_override)


def resolve_render_profile(context: RuntimeContext) -> dict[str, Any]:
    """
    功能说明：返回模块 D 当前渲染 profile 摘要。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - dict[str, Any]: 供调度层观测的 profile 字典。
    异常说明：无。
    边界条件：模块 D 真实执行路径固定为 comfyui；本 profile 不再承载编码参数。
    """
    _ = context
    return {
        "render_backend": "comfyui",
        "name": "comfyui",
        "command_args": [],
        "fallback_cpu_profile": None,
    }


def execute_units_with_retry(context: RuntimeContext, units_to_run: list[ModuleDUnit]) -> None:
    """
    功能说明：执行模块 D 待处理单元，并在失败时按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    返回值：无。
    异常说明：单元重试耗尽仍失败时抛 RuntimeError。
    边界条件：已完成单元由上层过滤，不在本函数内重跑。
    """
    if not units_to_run:
        context.logger.info("模块D无待执行单元，task_id=%s", context.task_id)
        return

    worker_count = _normalize_module_d_workers(context.config.module_d.segment_workers)
    retry_times = _normalize_module_d_retry_times(context.config.module_d.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    hard_fail_messages: list[str] = []

    for attempt_index in range(retry_times + 1):
        if not pending_units:
            break
        attempt_no = attempt_index + 1
        context.logger.info(
            "模块D单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s，backend=comfyui",
            context.task_id,
            attempt_no,
            retry_times + 1,
            len(pending_units),
            worker_count,
        )
        for unit in pending_units:
            context.state_store.set_module_unit_status(
                task_id=context.task_id,
                module_name="D",
                unit_id=unit.unit_id,
                status="running",
                artifact_path="",
                error_message="",
            )

        if worker_count == 1:
            failed_units: list[tuple[ModuleDUnit, Exception]] = []
            for unit in pending_units:
                error = _render_one_unit_comfyui(context=context, unit=unit)
                if error is not None:
                    failed_units.append((unit, error))
        else:
            failed_units = _execute_units_parallel_comfyui(
                context=context,
                pending_units=pending_units,
                worker_count=worker_count,
            )

        if not failed_units:
            pending_units = []
            continue
        if attempt_index < retry_times:
            context.logger.warning(
                "模块D单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s，backend=comfyui",
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
        raise RuntimeError(f"模块D单元渲染失败，共{len(hard_fail_messages)}个单元失败：\n{error_text}")


def execute_one_unit_with_retry(
    context: RuntimeContext,
    unit: ModuleDUnit,
    profile: dict[str, Any] | None = None,
    retry_times: int | None = None,
    device_override: str | None = None,
) -> Path:
    """
    功能说明：执行单个模块 D 单元并按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    - profile: 历史遗留参数；当前 ComfyUI 路径忽略。
    - retry_times: 可选重试次数，传空时读取模块配置。
    - device_override: 历史遗留参数；当前 ComfyUI 路径忽略。
    返回值：
    - Path: 单元片段路径。
    异常说明：
    - RuntimeError: 重试耗尽后抛出。
    边界条件：每次尝试前都会写入 running 状态。
    """
    _ = (profile, device_override)
    normalized_retry_times = (
        _normalize_module_d_retry_times(context.config.module_d.unit_retry_times)
        if retry_times is None
        else _normalize_module_d_retry_times(retry_times)
    )
    last_error: Exception | None = None
    for attempt_index in range(normalized_retry_times + 1):
        attempt_no = attempt_index + 1
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="D",
            unit_id=unit.unit_id,
            status="running",
            artifact_path="",
            error_message="",
        )
        error = _render_one_unit_comfyui(context=context, unit=unit)
        if error is None:
            return unit.segment_path
        last_error = error
        if attempt_index < normalized_retry_times:
            context.logger.warning(
                "模块D单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s，backend=comfyui",
                context.task_id,
                unit.unit_id,
                attempt_no,
                normalized_retry_times + 1,
                error,
            )
            continue
        break
    raise RuntimeError(f"模块D单元执行失败，unit_id={unit.unit_id}，错误={last_error}")


def _execute_units_parallel_comfyui(
    context: RuntimeContext,
    pending_units: list[ModuleDUnit],
    worker_count: int,
) -> list[tuple[ModuleDUnit, Exception]]:
    """
    功能说明：并行执行模块 D 的 ComfyUI 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - worker_count: 并发 worker 数量。
    返回值：
    - list[tuple[ModuleDUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常统一转换为失败列表返回）。
    边界条件：ComfyUI 服务端自身负责模型常驻与执行队列。
    """
    failed_units: list[tuple[ModuleDUnit, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
        future_to_unit = {
            executor.submit(_render_one_unit_comfyui, context, unit): unit
            for unit in pending_units
        }
        for future in as_completed(future_to_unit):
            unit = future_to_unit[future]
            try:
                error = future.result()
            except Exception as unexpected_error:  # noqa: BLE001
                error = unexpected_error
            if error is not None:
                if not isinstance(error, Exception):
                    error = RuntimeError(str(error))
                failed_units.append((unit, error))
    failed_units.sort(key=lambda item: int(item[0].unit_index))
    return failed_units


def _render_one_unit_comfyui(context: RuntimeContext, unit: ModuleDUnit) -> Exception | None:
    """
    功能说明：执行一次模块 D 的 ComfyUI 渲染并写入状态。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - Exception | None: 成功返回 None，失败返回异常对象。
    异常说明：无（异常转为返回值，交给上层重试逻辑处理）。
    边界条件：模块常驻由 ComfyUI 服务自身负责。
    """
    try:
        if not str(unit.shot.get("video_prompt_en", "")).strip():
            unit.shot["video_prompt_en"] = _resolve_unit_video_prompt_en(context=context, unit=unit)
        result = render_one_unit_comfyui(context=context, unit=unit)
        _apply_camera_plan_if_needed(context=context, unit=unit)
        _mark_unit_done(
            context=context,
            unit=unit,
            segment_path=Path(str(result["segment_path"])),
            render_summary=result if isinstance(result, dict) else None,
        )
        return None
    except Exception as error:  # noqa: BLE001
        _mark_unit_failed(context=context, unit=unit, error=error)
        return error


def _resolve_unit_video_prompt_en(context: RuntimeContext, unit: ModuleDUnit) -> str:
    """
    功能说明：解析模块 D 单元的英文视频提示词（单轨）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - str: video_prompt_en。
    异常说明：
    - RuntimeError: 无法解析有效提示词时抛出。
    边界条件：优先使用 unit.shot 内字段，缺失时回退读取模块 B 产物。
    """
    prompt_text = _extract_prompt_from_shot_payload(shot_payload=unit.shot)
    if prompt_text:
        return prompt_text

    prompt_text = _read_prompt_from_module_b_output(context=context, shot_id=unit.unit_id)
    if prompt_text:
        return prompt_text

    prompt_text = _read_prompt_from_module_b_unit_artifact(context=context, unit_index=unit.unit_index)
    if prompt_text:
        return prompt_text

    raise RuntimeError(
        "模块D ComfyUI 渲染失败：未找到可用英文视频提示词（video_prompt_en），"
        f"shot_id={unit.unit_id}，unit_index={unit.unit_index}"
    )


def _extract_prompt_from_shot_payload(shot_payload: dict[str, Any]) -> str:
    """
    功能说明：从 shot 载荷中提取 video_prompt_en。
    参数说明：
    - shot_payload: shot 字典。
    返回值：
    - str: 命中的 video_prompt_en；缺失返回空字符串。
    异常说明：无。
    边界条件：严格只读取 video_prompt_en，不做旧字段兼容回退。
    """
    return str(shot_payload.get("video_prompt_en", "")).strip()


def _read_prompt_from_module_b_output(context: RuntimeContext, shot_id: str) -> str:
    """
    功能说明：从 module_b_output.json 中读取目标 shot 的 video_prompt_en。
    参数说明：
    - context: 运行上下文对象。
    - shot_id: 目标 shot_id。
    返回值：
    - str: 解析到的提示词，未命中返回空字符串。
    异常说明：无（读取失败时仅告警并返回空字符串）。
    边界条件：仅在 module_b_output.json 存在且为数组时生效。
    """
    module_b_path = context.artifacts_dir / "module_b_output.json"
    if not module_b_path.exists():
        return ""
    try:
        module_b_output = read_json(module_b_path)
    except Exception as error:  # noqa: BLE001
        context.logger.warning("读取 module_b_output.json 失败，已跳过 prompt 回退，错误=%s", error)
        return ""
    if not isinstance(module_b_output, list):
        return ""
    for item in module_b_output:
        if not isinstance(item, dict):
            continue
        if str(item.get("shot_id", "")).strip() != str(shot_id).strip():
            continue
        prompt_text = _extract_prompt_from_shot_payload(shot_payload=item)
        if prompt_text:
            return prompt_text
    return ""


def _read_prompt_from_module_b_unit_artifact(context: RuntimeContext, unit_index: int) -> str:
    """
    功能说明：从模块 B 单元产物文件读取 video_prompt_en（作为最后回退）。
    参数说明：
    - context: 运行上下文对象。
    - unit_index: 目标 unit_index。
    返回值：
    - str: 解析到的提示词，未命中返回空字符串。
    异常说明：无（读取失败时仅告警并返回空字符串）。
    边界条件：依赖 module_unit_runs(B) 的 artifact_path 有效。
    """
    b_rows = context.state_store.list_module_units(task_id=context.task_id, module_name="B")
    target_row = next((row for row in b_rows if int(row.get("unit_index", -1)) == int(unit_index)), None)
    if target_row is None:
        return ""
    artifact_path_text = str(target_row.get("artifact_path", "")).strip()
    if not artifact_path_text:
        return ""
    artifact_path = Path(artifact_path_text)
    if not artifact_path.exists():
        return ""
    try:
        shot_payload = read_json(artifact_path)
    except Exception as error:  # noqa: BLE001
        context.logger.warning("读取模块B单元产物失败，已跳过 prompt 回退，path=%s，错误=%s", artifact_path, error)
        return ""
    if not isinstance(shot_payload, dict):
        return ""
    return _extract_prompt_from_shot_payload(shot_payload=shot_payload)


def _mark_unit_done(
    context: RuntimeContext,
    unit: ModuleDUnit,
    segment_path: Path,
    render_summary: dict[str, Any] | None = None,
) -> None:
    """
    功能说明：将单元状态写入 done 并记录产物路径。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    - segment_path: 渲染完成的片段路径。
    - render_summary: 可选渲染摘要（用于附加日志观测字段）。
    返回值：无。
    异常说明：数据库写入失败时抛 sqlite3.Error。
    边界条件：segment_path 必须存在。
    """
    if not segment_path.exists():
        raise RuntimeError(f"模块D单元执行失败：片段文件不存在，unit_id={unit.unit_id}")
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="D",
        unit_id=unit.unit_id,
        status="done",
        artifact_path=str(segment_path),
        error_message="",
    )
    target_effective_fps = None
    if isinstance(render_summary, dict):
        target_effective_fps = render_summary.get("target_effective_fps")
    if target_effective_fps is not None:
        context.logger.info(
            "模块D单元执行完成，task_id=%s，unit_id=%s，segment=%s，target_effective_fps=%s",
            context.task_id,
            unit.unit_id,
            segment_path,
            target_effective_fps,
        )
    else:
        context.logger.info("模块D单元执行完成，task_id=%s，unit_id=%s，segment=%s", context.task_id, unit.unit_id, segment_path)


def _apply_camera_plan_if_needed(context: RuntimeContext, unit: ModuleDUnit) -> None:
    """
    功能说明：若 shot 携带非 none 的 camera_plan，则对单段视频执行 FFmpeg 运镜后处理。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块D单元对象。
    返回值：无。
    异常说明：后处理失败时抛 RuntimeError。
    边界条件：处理完成后会原子替换回原 segment_path。
    """
    camera_plan = unit.shot.get("camera_plan", {})
    if not isinstance(camera_plan, dict):
        return
    applied = apply_camera_plan_to_segment(
        segment_path=unit.segment_path,
        output_path=unit.temp_segment_path,
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        fps=context.config.ffmpeg.fps,
        video_codec=context.config.ffmpeg.video_codec,
        video_preset=context.config.ffmpeg.video_preset,
        video_crf=context.config.ffmpeg.video_crf,
        camera_plan=camera_plan,
    )
    if not applied:
        return
    unit.temp_segment_path.replace(unit.segment_path)
    context.logger.info(
        "模块D单段运镜后处理完成，task_id=%s，unit_id=%s，preset_id=%s",
        context.task_id,
        unit.unit_id,
        str(camera_plan.get("preset_id", "")),
    )


def _mark_unit_failed(context: RuntimeContext, unit: ModuleDUnit, error: Exception) -> None:
    """
    功能说明：将单元状态写入 failed 并记录错误。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    - error: 执行异常。
    返回值：无。
    异常说明：数据库写入失败时抛 sqlite3.Error。
    边界条件：错误文本会被直接写入状态库用于恢复排障。
    """
    context.state_store.set_module_unit_status(
        task_id=context.task_id,
        module_name="D",
        unit_id=unit.unit_id,
        status="failed",
        artifact_path="",
        error_message=str(error),
    )
    context.logger.error("模块D单元执行失败，task_id=%s，unit_id=%s，错误=%s", context.task_id, unit.unit_id, error)


def _normalize_module_d_workers(segment_workers: int) -> int:
    """
    功能说明：归一化模块 D 并行 worker 数量。
    参数说明：
    - segment_workers: 原始 worker 配置值。
    返回值：
    - int: 合法 worker 数量（范围 1~4）。
    异常说明：无。
    边界条件：非法值统一回退为 3。
    """
    try:
        normalized = int(segment_workers)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 4:
        return 4
    return normalized


def _normalize_module_d_retry_times(unit_retry_times: int) -> int:
    """
    功能说明：归一化模块 D 单元重试次数。
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
