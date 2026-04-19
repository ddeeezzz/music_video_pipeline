"""
文件用途：实现模块 D 最小单元并行执行与重试逻辑。
核心流程：按待执行单元并行渲染片段，失败单元按配置重试并写入单元状态。
输入输出：输入运行上下文与单元数组，输出执行副作用（状态与片段文件）。
依赖说明：依赖标准库并发工具与项目内 RuntimeContext/ModuleDUnit。
维护说明：本层只负责 D 内部并行，不改变 A->B->C->D 的模块顺序。
"""

# 标准库：用于进程池与线程池并发
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# 标准库：用于多进程上下文
import multiprocessing
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于线程互斥锁
import threading
# 标准库：用于阶段计时
import time
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具（用于补取模块B视频提示词）
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块D AnimateDiff 渲染后端
from music_video_pipeline.modules.module_d.backends import (
    prewarm_animatediff_runtime as prewarm_animatediff_runtime_backend,
    run_one_unit_animatediff_denoise_stage,
    run_one_unit_animatediff_post_stage,
)
# 项目内模块：模块D终拼与FFmpeg工具
from music_video_pipeline.modules.module_d.finalizer import _resolve_video_encoder_profile, _run_ffmpeg_command
# 项目内模块：模块D单元模型
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit

# 全局锁表：约束同设备内 AnimateDiff 串行推理，允许异设备并行。
_ANIMATEDIFF_RENDER_LOCKS: dict[str, threading.Lock] = {}
_ANIMATEDIFF_RENDER_LOCKS_GUARD = threading.Lock()


def prewarm_animatediff_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, str]:
    """
    功能说明：预热指定设备上的 AnimateDiff runtime（仅加载缓存，不执行推理）。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 可选，强制绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - dict[str, str]: 预热摘要（device/cache_key）。
    异常说明：
    - RuntimeError: runtime 初始化失败时抛出。
    边界条件：仅在 animatediff 后端有意义；其他后端调用也不会改变任务状态。
    """
    return prewarm_animatediff_runtime_backend(context=context, device_override=device_override)


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

    render_backend = _normalize_module_d_render_backend(context.config.module_d.render_backend)
    if render_backend == "animatediff":
        _execute_units_with_retry_animatediff(context=context, units_to_run=units_to_run)
        return
    _execute_units_with_retry_ffmpeg(context=context, units_to_run=units_to_run)


def _execute_units_with_retry_ffmpeg(context: RuntimeContext, units_to_run: list[ModuleDUnit]) -> None:
    """
    功能说明：执行模块 D ffmpeg 后端待处理单元，并在失败时按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    返回值：无。
    异常说明：单元重试耗尽仍失败时抛 RuntimeError。
    边界条件：已完成单元由上层过滤，不在本函数内重跑。
    """
    worker_count = _normalize_module_d_workers(context.config.module_d.segment_workers)
    retry_times = _normalize_module_d_retry_times(context.config.module_d.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    hard_fail_messages: list[str] = []

    profile = _resolve_ffmpeg_render_profile(context=context)

    for attempt_index in range(retry_times + 1):
        if not pending_units:
            break
        attempt_no = attempt_index + 1
        context.logger.info(
            "模块D单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s，backend=ffmpeg",
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
            failed_units = _execute_units_serial(
                context=context,
                pending_units=pending_units,
                profile=profile,
            )
        else:
            failed_units = _execute_units_parallel(
                context=context,
                pending_units=pending_units,
                profile=profile,
                worker_count=worker_count,
            )

        if not failed_units:
            pending_units = []
            continue

        if attempt_index < retry_times:
            context.logger.warning(
                "模块D单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s，backend=ffmpeg",
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


def _execute_units_with_retry_animatediff(context: RuntimeContext, units_to_run: list[ModuleDUnit]) -> None:
    """
    功能说明：执行模块 D AnimateDiff 后端待处理单元，并在失败时按配置重试。
    参数说明：
    - context: 运行上下文对象。
    - units_to_run: 需要执行的单元数组。
    返回值：无。
    异常说明：单元重试耗尽仍失败时抛 RuntimeError。
    边界条件：同设备去噪阶段串行，后处理阶段允许并行推进。
    """
    worker_count = _normalize_module_d_workers(context.config.module_d.segment_workers)
    retry_times = _normalize_module_d_retry_times(context.config.module_d.unit_retry_times)
    pending_units = sorted(units_to_run, key=lambda item: item.unit_index)
    hard_fail_messages: list[str] = []

    for attempt_index in range(retry_times + 1):
        if not pending_units:
            break
        attempt_no = attempt_index + 1
        context.logger.info(
            "模块D单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s，backend=animatediff",
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
                error = _render_one_unit_animatediff_serialized(context=context, unit=unit)
                if error is not None:
                    failed_units.append((unit, error))
        else:
            failed_units = _execute_units_parallel_animatediff(
                context=context,
                pending_units=pending_units,
                worker_count=worker_count,
            )

        if not failed_units:
            pending_units = []
            continue

        if attempt_index < retry_times:
            context.logger.warning(
                "模块D单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s，backend=animatediff",
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


def resolve_render_profile(context: RuntimeContext) -> dict[str, Any]:
    """
    功能说明：解析并返回模块 D 当前任务的渲染 profile。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - dict[str, Any]: 渲染 profile 字典。
    异常说明：gpu_only 模式且编码器不可用时抛 RuntimeError。
    边界条件：返回值可复用于多个单元，避免重复探测开销。
    """
    render_backend = _normalize_module_d_render_backend(context.config.module_d.render_backend)
    if render_backend == "animatediff":
        return {
            "render_backend": "animatediff",
            "name": "animatediff",
            "command_args": [],
            "fallback_cpu_profile": None,
        }
    return _resolve_ffmpeg_render_profile(context=context)


def _resolve_ffmpeg_render_profile(context: RuntimeContext) -> dict[str, Any]:
    """
    功能说明：解析并返回模块 D 的 ffmpeg 编码 profile。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - dict[str, Any]: 编码 profile 字典。
    异常说明：gpu_only 模式且编码器不可用时抛 RuntimeError。
    边界条件：返回值可复用于多个单元，避免重复探测开销。
    """
    return _resolve_video_encoder_profile(
        ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
        video_accel_mode=context.config.ffmpeg.video_accel_mode,
        cpu_video_codec=context.config.ffmpeg.video_codec,
        cpu_video_preset=context.config.ffmpeg.video_preset,
        cpu_video_crf=context.config.ffmpeg.video_crf,
        gpu_video_codec=context.config.ffmpeg.gpu_video_codec,
        gpu_preset=context.config.ffmpeg.gpu_preset,
        gpu_rc_mode=context.config.ffmpeg.gpu_rc_mode,
        gpu_cq=context.config.ffmpeg.gpu_cq,
        gpu_bitrate=context.config.ffmpeg.gpu_bitrate,
    )


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
    - profile: 可选编码 profile，传空时实时解析。
    - retry_times: 可选重试次数，传空时读取模块配置。
    - device_override: 可选，强制 AnimateDiff 绑定设备（如 cuda:0/cuda:1）。
    返回值：
    - Path: 单元片段路径。
    异常说明：
    - RuntimeError: 重试耗尽后抛出。
    边界条件：每次尝试前都会写入 running 状态。
    """
    render_backend = _normalize_module_d_render_backend(context.config.module_d.render_backend)
    if render_backend == "animatediff":
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
            error = _render_one_unit_animatediff_serialized(
                context=context,
                unit=unit,
                device_override=device_override,
            )
            if error is None:
                return unit.segment_path
            last_error = error
            if attempt_index < normalized_retry_times:
                context.logger.warning(
                    "模块D单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s，backend=animatediff",
                    context.task_id,
                    unit.unit_id,
                    attempt_no,
                    normalized_retry_times + 1,
                    error,
                )
                continue
            break
        raise RuntimeError(f"模块D单元执行失败，unit_id={unit.unit_id}，错误={last_error}")

    active_profile = profile or resolve_render_profile(context=context)
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
        error = _render_one_unit_with_fallback(context=context, unit=unit, profile=active_profile)
        if error is None:
            return unit.segment_path
        last_error = error
        if attempt_index < normalized_retry_times:
            context.logger.warning(
                "模块D单元重试中，task_id=%s，unit_id=%s，attempt=%s/%s，错误=%s，backend=ffmpeg",
                context.task_id,
                unit.unit_id,
                attempt_no,
                normalized_retry_times + 1,
                error,
            )
            continue
        break
    raise RuntimeError(f"模块D单元执行失败，unit_id={unit.unit_id}，错误={last_error}")


def _execute_units_serial(
    context: RuntimeContext,
    pending_units: list[ModuleDUnit],
    profile: dict[str, Any],
) -> list[tuple[ModuleDUnit, Exception]]:
    """
    功能说明：串行执行模块 D 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - profile: 编码配置。
    返回值：
    - list[tuple[ModuleDUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：执行顺序固定按 unit_index 升序。
    """
    failed_units: list[tuple[ModuleDUnit, Exception]] = []
    for unit in pending_units:
        error = _render_one_unit_with_fallback(context=context, unit=unit, profile=profile)
        if error is not None:
            failed_units.append((unit, error))
    return failed_units


def _execute_units_parallel(
    context: RuntimeContext,
    pending_units: list[ModuleDUnit],
    profile: dict[str, Any],
    worker_count: int,
) -> list[tuple[ModuleDUnit, Exception]]:
    """
    功能说明：并行执行模块 D 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - profile: 编码配置。
    - worker_count: 并发 worker 数量。
    返回值：
    - list[tuple[ModuleDUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常转为失败列表返回）。
    边界条件：状态写入统一在主线程完成，避免并发写库冲突。
    """
    failed_units: list[tuple[ModuleDUnit, Exception]] = []

    fallback_profile = profile.get("fallback_cpu_profile")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future_to_unit = {
            executor.submit(
                _render_single_segment_worker,
                context.config.ffmpeg.ffmpeg_bin,
                str(unit.shot.get("frame_path", "")),
                int(unit.exact_frames),
                context.config.ffmpeg.fps,
                list(profile["command_args"]),
                unit.unit_index + 1,
                str(unit.temp_segment_path),
                str(unit.segment_path),
                str(profile.get("name", "unknown")),
            ): unit
            for unit in pending_units
        }

        for future in as_completed(future_to_unit):
            unit = future_to_unit[future]
            try:
                result = future.result()
                _mark_unit_done(context=context, unit=unit, segment_path=Path(str(result["segment_path"])))
            except Exception as error:  # noqa: BLE001
                if fallback_profile is not None:
                    context.logger.warning(
                        "模块D单元GPU渲染失败，准备CPU回退重试，task_id=%s，unit_id=%s，错误=%s",
                        context.task_id,
                        unit.unit_id,
                        error,
                    )
                    cpu_error = _render_one_unit_once(context=context, unit=unit, profile=fallback_profile)
                    if cpu_error is None:
                        continue
                    _mark_unit_failed(context=context, unit=unit, error=cpu_error)
                    failed_units.append((unit, cpu_error))
                    continue
                _mark_unit_failed(context=context, unit=unit, error=error)
                failed_units.append((unit, error))
    return failed_units


def _execute_units_parallel_animatediff(
    context: RuntimeContext,
    pending_units: list[ModuleDUnit],
    worker_count: int,
) -> list[tuple[ModuleDUnit, Exception]]:
    """
    功能说明：并行执行模块 D AnimateDiff 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - worker_count: 并发 worker 数量。
    返回值：
    - list[tuple[ModuleDUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常统一转换为失败列表返回）。
    边界条件：同设备去噪阶段仍受互斥锁保护，后处理阶段可并发。
    """
    failed_units: list[tuple[ModuleDUnit, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
        future_to_unit = {
            executor.submit(_render_one_unit_animatediff_serialized, context, unit): unit
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


def _render_one_unit_with_fallback(context: RuntimeContext, unit: ModuleDUnit, profile: dict[str, Any]) -> Exception | None:
    """
    功能说明：渲染单元并在需要时执行一次 CPU 回退。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    - profile: 主编码配置。
    返回值：
    - Exception | None: 成功返回 None，失败返回最终异常。
    异常说明：无（异常对象作为返回值上抛给调用方）。
    边界条件：只执行一次 CPU 回退，不做多次回退。
    """
    primary_error = _render_one_unit_once(context=context, unit=unit, profile=profile)
    if primary_error is None:
        return None

    fallback_profile = profile.get("fallback_cpu_profile")
    if fallback_profile is None:
        _mark_unit_failed(context=context, unit=unit, error=primary_error)
        return primary_error

    context.logger.warning(
        "模块D单元GPU渲染失败，准备CPU回退重试，task_id=%s，unit_id=%s，错误=%s",
        context.task_id,
        unit.unit_id,
        primary_error,
    )
    fallback_error = _render_one_unit_once(context=context, unit=unit, profile=fallback_profile)
    if fallback_error is None:
        return None

    _mark_unit_failed(context=context, unit=unit, error=fallback_error)
    return fallback_error


def _render_one_unit_once(context: RuntimeContext, unit: ModuleDUnit, profile: dict[str, Any]) -> Exception | None:
    """
    功能说明：按指定编码配置执行一次单元渲染。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    - profile: 编码配置。
    返回值：
    - Exception | None: 成功返回 None，失败返回异常。
    异常说明：无（异常转为返回值）。
    边界条件：成功后会立即写入 done 状态。
    """
    try:
        result = _render_single_segment_worker(
            ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
            frame_path=str(unit.shot.get("frame_path", "")),
            exact_frames=int(unit.exact_frames),
            fps=context.config.ffmpeg.fps,
            encoder_command_args=list(profile["command_args"]),
            segment_index=unit.unit_index + 1,
            temp_output_path=str(unit.temp_segment_path),
            final_output_path=str(unit.segment_path),
            profile_name=str(profile.get("name", "unknown")),
        )
        _mark_unit_done(context=context, unit=unit, segment_path=Path(str(result["segment_path"])))
        return None
    except Exception as error:  # noqa: BLE001
        return error


def _render_one_unit_animatediff_with_optional_fallback(
    context: RuntimeContext,
    unit: ModuleDUnit,
    device_override: str | None = None,
    denoise_summary: dict[str, Any] | None = None,
    prompt: str | None = None,
) -> Exception | None:
    """
    功能说明：按 AnimateDiff 渲染单元（严格模式，不回退 ffmpeg）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    返回值：
    - Exception | None: 成功返回 None，失败返回最终异常。
    异常说明：无（异常对象作为返回值上抛给调用方）。
    边界条件：任何 AnimateDiff 异常都会直接记为失败并上抛到重试层。
    """
    resolved_prompt = str(prompt or "").strip()
    if not resolved_prompt:
        resolved_prompt = _resolve_unit_video_prompt(context=context, unit=unit)
    if not str(unit.shot.get("video_prompt_en", "")).strip():
        unit.shot["video_prompt_en"] = resolved_prompt
    if not str(unit.shot.get("video_prompt", "")).strip():
        unit.shot["video_prompt"] = resolved_prompt
    encoder_args = _build_animatediff_encoder_args(context=context)
    try:
        active_denoise_summary = denoise_summary
        if active_denoise_summary is None:
            active_denoise_summary = run_one_unit_animatediff_denoise_stage(
                context=context,
                unit=unit,
                prompt=resolved_prompt,
                device_override=device_override,
            )
        result = run_one_unit_animatediff_post_stage(
            context=context,
            unit=unit,
            denoise_summary=active_denoise_summary,
            encoder_command_args=encoder_args,
            profile_name="animatediff",
        )
        _mark_unit_done(
            context=context,
            unit=unit,
            segment_path=Path(str(result["segment_path"])),
            render_summary=result if isinstance(result, dict) else None,
        )
        return None
    except Exception as primary_error:  # noqa: BLE001
        _mark_unit_failed(context=context, unit=unit, error=primary_error)
        context.logger.error(
            "模块D AnimateDiff 渲染失败，已按严格模式终止该单元且不回退 ffmpeg，task_id=%s，unit_id=%s，错误=%s",
            context.task_id,
            unit.unit_id,
            primary_error,
        )
        return primary_error


def _render_one_unit_animatediff_serialized(
    context: RuntimeContext,
    unit: ModuleDUnit,
    device_override: str | None = None,
) -> Exception | None:
    """
    功能说明：串行执行单个 AnimateDiff 单元渲染。
    参数说明：
    - context: 运行上下文对象。
    - unit: 目标单元。
    返回值：
    - Exception | None: 成功返回 None，失败返回最终异常。
    异常说明：无（异常对象作为返回值上抛给调用方）。
    边界条件：同一进程内按 device 维度互斥（同卡串行，异卡可并行）。
    """
    resolved_prompt = _resolve_unit_video_prompt(context=context, unit=unit)
    if not str(unit.shot.get("video_prompt_en", "")).strip():
        unit.shot["video_prompt_en"] = resolved_prompt
    if not str(unit.shot.get("video_prompt", "")).strip():
        unit.shot["video_prompt"] = resolved_prompt
    lock = _get_animatediff_render_lock(device_override=device_override)
    with lock:
        try:
            denoise_summary = run_one_unit_animatediff_denoise_stage(
                context=context,
                unit=unit,
                prompt=resolved_prompt,
                device_override=device_override,
            )
        except Exception as primary_error:  # noqa: BLE001
            _mark_unit_failed(context=context, unit=unit, error=primary_error)
            context.logger.error(
                "模块D AnimateDiff 渲染失败，已按严格模式终止该单元且不回退 ffmpeg，task_id=%s，unit_id=%s，错误=%s",
                context.task_id,
                unit.unit_id,
                primary_error,
            )
            return primary_error
    return _render_one_unit_animatediff_with_optional_fallback(
        context=context,
        unit=unit,
        device_override=device_override,
        denoise_summary=denoise_summary,
        prompt=resolved_prompt,
    )


def _get_animatediff_render_lock(device_override: str | None) -> threading.Lock:
    """
    功能说明：获取指定设备对应的 AnimateDiff 渲染互斥锁。
    参数说明：
    - device_override: 指定设备字符串（可空）。
    返回值：
    - threading.Lock: 设备级互斥锁对象。
    异常说明：无。
    边界条件：device 为空时统一归入 default 锁，保持向后兼容。
    """
    key = str(device_override or "default").strip().lower() or "default"
    with _ANIMATEDIFF_RENDER_LOCKS_GUARD:
        lock = _ANIMATEDIFF_RENDER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ANIMATEDIFF_RENDER_LOCKS[key] = lock
        return lock


def _resolve_unit_video_prompt(context: RuntimeContext, unit: ModuleDUnit) -> str:
    """
    功能说明：解析模块 D 单元的视频提示词。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - str: 可用于视频生成的 prompt。
    异常说明：
    - RuntimeError: 无法解析有效 prompt 时抛出。
    边界条件：优先使用 unit.shot 内字段，缺失时回退读取模块B产物。
    """
    prompt = _extract_prompt_from_shot_payload(shot_payload=unit.shot)
    if prompt:
        return prompt

    prompt = _read_prompt_from_module_b_output(context=context, shot_id=unit.unit_id)
    if prompt:
        return prompt

    prompt = _read_prompt_from_module_b_unit_artifact(context=context, unit_index=unit.unit_index)
    if prompt:
        return prompt

    raise RuntimeError(
        "模块D AnimateDiff 渲染失败：未找到可用英文视频提示词（video_prompt_en/keyframe_prompt_en），"
        f"shot_id={unit.unit_id}，unit_index={unit.unit_index}"
    )


def _extract_prompt_from_shot_payload(shot_payload: dict[str, Any]) -> str:
    """
    功能说明：从 shot 载荷中提取最优视频提示词。
    参数说明：
    - shot_payload: shot 字典。
    返回值：
    - str: prompt 文本；不存在时返回空字符串。
    异常说明：无。
    边界条件：字段优先级 video_prompt_en > keyframe_prompt_en（严格英文，不回退中文字段）。
    """
    for prompt_key in ("video_prompt_en", "keyframe_prompt_en"):
        prompt_text = str(shot_payload.get(prompt_key, "")).strip()
        if prompt_text:
            return prompt_text
    return ""


def _read_prompt_from_module_b_output(context: RuntimeContext, shot_id: str) -> str:
    """
    功能说明：从 module_b_output.json 中读取目标 shot 的 prompt。
    参数说明：
    - context: 运行上下文对象。
    - shot_id: 目标 shot_id。
    返回值：
    - str: 解析到的 prompt，未命中返回空字符串。
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
    功能说明：从模块 B 单元产物文件读取 prompt（作为最后回退）。
    参数说明：
    - context: 运行上下文对象。
    - unit_index: 目标 unit_index。
    返回值：
    - str: 解析到的 prompt，未命中返回空字符串。
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


def _build_animatediff_encoder_args(context: RuntimeContext) -> list[str]:
    """
    功能说明：构建 AnimateDiff 输出片段编码参数。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - list[str]: ffmpeg 编码参数数组。
    异常说明：无。
    边界条件：当前优先使用 CPU codec 参数，避免额外 NVENC 会话依赖。
    """
    codec = str(context.config.ffmpeg.video_codec).strip() or "libx264"
    args = ["-c:v", codec]
    if codec in {"libx264", "libx265"}:
        args.extend(["-preset", str(context.config.ffmpeg.video_preset)])
        args.extend(["-crf", str(int(context.config.ffmpeg.video_crf))])
    return args


def _build_single_segment_command(
    ffmpeg_bin: str,
    frame_path: str,
    exact_frames: int,
    fps: int,
    encoder_command_args: list[str],
    output_path: str,
) -> list[str]:
    """
    功能说明：构建单段渲染 ffmpeg 命令（单输入单输出）。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - frame_path: 输入帧路径。
    - exact_frames: 输出帧数。
    - fps: 输出帧率。
    - encoder_command_args: 编码参数数组。
    - output_path: 输出视频路径。
    返回值：
    - list[str]: 可执行的 ffmpeg 命令数组。
    异常说明：无。
    边界条件：固定禁用音频流并输出 yuv420p。
    """
    return [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-i",
        str(frame_path),
        "-frames:v",
        str(exact_frames),
        "-r",
        str(fps),
        *list(encoder_command_args),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]


def _render_single_segment_worker(
    ffmpeg_bin: str,
    frame_path: str,
    exact_frames: int,
    fps: int,
    encoder_command_args: list[str],
    segment_index: int,
    temp_output_path: str,
    final_output_path: str,
    profile_name: str,
) -> dict[str, Any]:
    """
    功能说明：执行单段渲染并以原子替换方式提交最终产物。
    参数说明：
    - ffmpeg_bin/frame_path/exact_frames/fps/encoder_command_args: 渲染命令参数。
    - segment_index: 片段序号（用于日志上下文）。
    - temp_output_path: 临时输出路径。
    - final_output_path: 最终输出路径。
    - profile_name: 渲染 profile 名称（gpu/cpu）。
    返回值：
    - dict[str, Any]: 渲染结果摘要（segment_index/segment_path/elapsed/profile_name）。
    异常说明：渲染失败或原子替换失败时抛 RuntimeError。
    边界条件：失败时会清理临时文件，避免残留半成品。
    """
    stage_start = time.perf_counter()
    temp_path = Path(temp_output_path)
    final_path = Path(final_output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass

    command = _build_single_segment_command(
        ffmpeg_bin=ffmpeg_bin,
        frame_path=frame_path,
        exact_frames=exact_frames,
        fps=fps,
        encoder_command_args=encoder_command_args,
        output_path=str(temp_path),
    )

    try:
        _run_ffmpeg_command(
            command=command,
            command_name=f"渲染小片段 segment_{segment_index:03d}（{profile_name}）",
        )
        temp_path.replace(final_path)
    except Exception:  # noqa: BLE001
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise

    elapsed = time.perf_counter() - stage_start
    return {
        "segment_index": int(segment_index),
        "segment_path": str(final_path),
        "elapsed": float(elapsed),
        "profile_name": str(profile_name),
    }


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
    _sync_clip_after_render_if_needed(context=context, segment_path=segment_path)


def _sync_clip_after_render_if_needed(context: RuntimeContext, segment_path: Path) -> None:
    """
    功能说明：兼容函数，当前固定禁用 clip 级上传触发。
    参数说明：
    - context: 运行上下文对象。
    - segment_path: 当前已完成 clip 路径。
    返回值：无。
    异常说明：无。
    边界条件：模块级上传改由 Pipeline 在模块完成后异步入队。
    """
    _ = (context, segment_path)
    # 设计约束：禁用 clip 级实时上传，避免 D 渲染主线被上传 I/O 抢占。
    return


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


def _normalize_module_d_render_backend(render_backend: str) -> str:
    """
    功能说明：归一化模块 D 渲染后端。
    参数说明：
    - render_backend: 原始后端配置值。
    返回值：
    - str: 合法后端（ffmpeg/animatediff）。
    异常说明：无。
    边界条件：非法值统一回退为 ffmpeg。
    """
    normalized = str(render_backend).strip().lower()
    if normalized in {"ffmpeg", "animatediff"}:
        return normalized
    return "ffmpeg"
