"""
文件用途：提供跨模块调度中的设备/实例池分配与 in-flight 统计工具。
核心流程：维护 C 生成器池与 D 设备池的容量、轮询与占用计数。
输入输出：输入调度活跃任务与配置，输出可派发目标与占用快照。
依赖说明：仅依赖标准库与 RuntimeContext，不依赖执行器实现。
维护说明：本模块只负责分配策略，不处理任务执行与状态写库。
"""

from concurrent.futures import Future
import os
from typing import Any

from music_video_pipeline.context import RuntimeContext


def _build_d_device_pool(c_gpu_index: int, d_gpu_index: int) -> list[str]:
    """
    功能说明：基于 C/D GPU 索引构建 D 阶段可用设备池（去重后按顺序）。
    参数说明：
    - c_gpu_index: C 绑定 GPU 索引。
    - d_gpu_index: D 绑定 GPU 索引。
    返回值：
    - list[str]: 设备字符串数组（如 ["cuda:0", "cuda:1"]）。
    异常说明：无。
    边界条件：至少返回 1 个设备，保证 D 阶段可调度。
    """
    ordered_indexes = [int(c_gpu_index), int(d_gpu_index)]
    unique_indexes: list[int] = []
    for gpu_index in ordered_indexes:
        if gpu_index in unique_indexes:
            continue
        unique_indexes.append(gpu_index)
    if not unique_indexes:
        unique_indexes = [0]
    return [f"cuda:{max(0, int(gpu_index))}" for gpu_index in unique_indexes]


def _build_d_device_inflight(
    active_tasks: dict[Future, tuple[str, int, str | None]],
    d_device_pool: list[str],
) -> dict[str, int]:
    """
    功能说明：统计当前 D 任务在各设备上的 in-flight 数。
    参数说明：
    - active_tasks: 活跃任务映射。
    - d_device_pool: D 阶段设备池。
    返回值：
    - dict[str, int]: device -> in-flight 数量。
    异常说明：无。
    边界条件：确保设备池内每个设备都有显式计数键。
    """
    inflight = {device: 0 for device in d_device_pool}
    for stage, _, device in active_tasks.values():
        if stage != "D":
            continue
        if not device:
            continue
        inflight[device] = int(inflight.get(device, 0)) + 1
    return inflight


def _resolve_d_device_pool_for_phase_d(
    context: RuntimeContext,
    adaptive_window_runtime: dict[str, Any],
    fallback_device_pool: list[str],
) -> list[str]:
    """
    功能说明：在进入 D 阶段时解析最终设备池，并在可观测 GPU 列表存在时做可用卡过滤。
    参数说明：
    - context: 运行上下文对象。
    - adaptive_window_runtime: 自适应窗口运行态（用于读取最近 probe 数据）。
    - fallback_device_pool: 基于配置构建的兜底设备池。
    返回值：
    - list[str]: D 阶段设备池。
    异常说明：无（探测失败自动回退 fallback_device_pool）。
    边界条件：如果仅有单卡可用，会自动退化为单设备池。
    """
    if bool(adaptive_window_runtime.get("single_gpu_mode", False)):
        return ["cuda:0"]
    probe_rows = adaptive_window_runtime.get("last_probe_rows", [])
    if not isinstance(probe_rows, list) or not probe_rows:
        from music_video_pipeline.modules.cross_bcd import scheduler_adaptive  # 避免模块级循环导入

        probe_rows, probe_error = scheduler_adaptive._run_gpu_probe_script(context=context, timeout_seconds=1.5)
        adaptive_window_runtime["last_probe_rows"] = probe_rows
        adaptive_window_runtime["last_probe_error"] = str(probe_error)
    available_gpu_indexes: set[int] = set()
    for row in probe_rows:
        if not isinstance(row, dict):
            continue
        try:
            available_gpu_indexes.add(int(row.get("index")))
        except (TypeError, ValueError):
            continue
    if not available_gpu_indexes:
        return list(fallback_device_pool)
    filtered_pool: list[str] = []
    for device in fallback_device_pool:
        try:
            device_index = int(str(device).split(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if device_index in available_gpu_indexes:
            filtered_pool.append(device)
    if filtered_pool:
        return filtered_pool
    return [fallback_device_pool[0]] if fallback_device_pool else ["cuda:0"]


def _resolve_c_generator_pool_size(
    context: RuntimeContext,
    adaptive_window_runtime: dict[str, Any],
) -> int:
    """
    功能说明：确定模块 C 生成器实例池大小（同卡多实例并发）。
    参数说明：
    - context: 运行上下文对象。
    - adaptive_window_runtime: 自适应窗口运行态。
    返回值：
    - int: 生成器实例数。
    异常说明：无。
    边界条件：仅对 diffusion 模式放大实例池，默认上限 3，避免显存峰值过高。
    """
    if bool(adaptive_window_runtime.get("single_gpu_mode", False)):
        return 1
    frame_mode = str(context.config.mode.frame_generator).strip().lower()
    if frame_mode != "diffusion":
        return 1
    try:
        configured_workers = int(context.config.module_c.render_workers)
    except (TypeError, ValueError):
        configured_workers = 1
    try:
        c_limit_max = int(adaptive_window_runtime.get("c_limit_max", 1))
    except (TypeError, ValueError):
        c_limit_max = 1
    env_override_text = str(os.environ.get("MVPL_C_DIFFUSION_POOL_SIZE", "")).strip()
    env_override: int | None = None
    if env_override_text:
        try:
            env_override = int(env_override_text)
        except ValueError:
            env_override = None
    default_pool_cap = 3
    if env_override is not None:
        default_pool_cap = max(1, min(4, env_override))
    normalized_size = max(1, min(default_pool_cap, max(1, configured_workers), max(1, c_limit_max)))
    return normalized_size


def _build_c_generator_inflight(
    active_tasks: dict[Future, tuple[str, int, str | None]],
    generator_pool_size: int,
) -> dict[int, int]:
    """
    功能说明：统计当前 C 任务在各生成器实例上的 in-flight 数。
    参数说明：
    - active_tasks: 活跃任务映射。
    - generator_pool_size: 生成器池大小。
    返回值：
    - dict[int, int]: generator_index -> in-flight 数量。
    异常说明：无。
    边界条件：仅识别标记为 cgen:<index> 的任务元数据。
    """
    inflight = {index: 0 for index in range(max(1, int(generator_pool_size)))}
    for stage, _, metadata in active_tasks.values():
        if stage != "C":
            continue
        metadata_text = str(metadata or "")
        if not metadata_text.startswith("cgen:"):
            continue
        try:
            generator_index = int(metadata_text.split(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if generator_index not in inflight:
            continue
        inflight[generator_index] = int(inflight.get(generator_index, 0)) + 1
    return inflight


def _pick_next_available_c_generator(
    c_generator_inflight: dict[int, int],
    generator_pool_size: int,
    start_cursor: int,
) -> tuple[int | None, int]:
    """
    功能说明：按轮询策略选择下一个可用的 C 生成器实例（同实例最多 1 in-flight）。
    参数说明：
    - c_generator_inflight: 各实例 in-flight 数。
    - generator_pool_size: 生成器池大小。
    - start_cursor: 起始游标。
    返回值：
    - tuple[int | None, int]: (实例索引或 None, 更新后的游标)。
    异常说明：无。
    边界条件：若所有实例都忙，返回 (None, 原游标)。
    """
    normalized_pool_size = max(1, int(generator_pool_size))
    cursor = max(0, int(start_cursor)) % normalized_pool_size
    for offset in range(normalized_pool_size):
        candidate_index = (cursor + offset) % normalized_pool_size
        active_count = int(c_generator_inflight.get(candidate_index, 0))
        if active_count < 1:
            next_cursor = (candidate_index + 1) % normalized_pool_size
            return candidate_index, next_cursor
    return None, cursor


def _pick_next_available_d_device(
    d_device_pool: list[str],
    d_device_inflight: dict[str, int],
    start_cursor: int,
    per_device_limit: int,
) -> tuple[str | None, int]:
    """
    功能说明：按轮询策略选择下一个可派发 D 任务的设备。
    参数说明：
    - d_device_pool: D 阶段设备池。
    - d_device_inflight: 当前各设备 in-flight 计数。
    - start_cursor: 轮询起始游标。
    - per_device_limit: 单设备并发上限。
    返回值：
    - tuple[str | None, int]: (选中的设备或 None, 更新后的游标)。
    异常说明：无。
    边界条件：若所有设备已满，返回 (None, 原游标)。
    """
    if not d_device_pool:
        return None, start_cursor
    pool_size = len(d_device_pool)
    cursor = max(0, int(start_cursor)) % pool_size
    for offset in range(pool_size):
        candidate_pos = (cursor + offset) % pool_size
        candidate_device = d_device_pool[candidate_pos]
        active_count = int(d_device_inflight.get(candidate_device, 0))
        if active_count < int(per_device_limit):
            next_cursor = (candidate_pos + 1) % pool_size
            return candidate_device, next_cursor
    return None, cursor
