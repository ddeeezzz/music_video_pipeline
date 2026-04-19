"""
文件用途：封装跨模块调度中的自适应窗口策略与 GPU 采样逻辑。
核心流程：构建自适应运行态、执行 probe、调整窗口、输出快照。
输入输出：输入运行上下文与采样结果，输出窗口值与状态摘要。
依赖说明：依赖标准库 subprocess/json 与 allocators 的设备池工具。
维护说明：本模块只处理窗口策略，不负责任务派发执行。
"""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.modules.cross_bcd import scheduler_allocators

# 常量：轻量 GPU 采样脚本相对路径（相对项目根目录）。
GPU_PROBE_SCRIPT_RELATIVE_PATH = Path("scripts/gpu_probe.py")


def collect_adaptive_window_status_snapshot(context: RuntimeContext) -> dict[str, Any]:
    """
    功能说明：采集当前配置下的自适应并发窗口状态摘要（供状态查询命令展示）。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - dict[str, Any]: 自适应窗口快照。
    异常说明：无（内部失败会降级为静态窗口快照）。
    边界条件：采样失败不抛异常，保持状态查询可用。
    """
    render_limit = _normalize_global_render_limit(context.config.cross_module.global_render_limit)
    render_backend = _normalize_module_d_render_backend(context.config.module_d.render_backend)
    adaptive_window_runtime = _build_adaptive_window_runtime(
        context=context,
        global_render_limit=render_limit,
        render_backend=render_backend,
    )
    if not bool(adaptive_window_runtime["enabled"]):
        d_device_pool = scheduler_allocators._build_d_device_pool(
            c_gpu_index=int(adaptive_window_runtime["c_gpu_index"]),
            d_gpu_index=int(adaptive_window_runtime["d_gpu_index"]),
        )
        return _build_adaptive_window_snapshot(
            adaptive_window_runtime=adaptive_window_runtime,
            render_limit=render_limit,
            render_backend=render_backend,
            current_phase="idle",
            d_dispatch_enabled=False,
            d_device_pool=d_device_pool,
            d_device_inflight={device: 0 for device in d_device_pool},
        )

    probe_rows, probe_error = _run_gpu_probe_script(context=context, timeout_seconds=2.0)
    adaptive_window_runtime["last_probe_rows"] = probe_rows
    adaptive_window_runtime["last_probe_error"] = str(probe_error)
    c_dynamic_limit = int(adaptive_window_runtime["c_dynamic_limit"])
    d_dynamic_limit = int(adaptive_window_runtime["d_dynamic_limit"])
    if probe_error:
        c_dynamic_limit = int(adaptive_window_runtime["fallback_c_limit"])
        d_dynamic_limit = int(adaptive_window_runtime["fallback_d_limit"])
    else:
        c_ratio = _extract_gpu_used_ratio(
            probe_rows=probe_rows,
            gpu_index=int(adaptive_window_runtime["c_gpu_index"]),
        )
        d_ratio = _extract_gpu_used_ratio(
            probe_rows=probe_rows,
            gpu_index=int(adaptive_window_runtime["d_gpu_index"]),
        )
        c_dynamic_limit = _adjust_dynamic_limit(
            current_limit=c_dynamic_limit,
            used_ratio=c_ratio,
            low_watermark=float(adaptive_window_runtime["low_watermark"]),
            high_watermark=float(adaptive_window_runtime["high_watermark"]),
            limit_min=int(adaptive_window_runtime["c_limit_min"]),
            limit_max=int(adaptive_window_runtime["c_limit_max"]),
        )
        d_dynamic_limit = _adjust_dynamic_limit(
            current_limit=d_dynamic_limit,
            used_ratio=d_ratio,
            low_watermark=float(adaptive_window_runtime["low_watermark"]),
            high_watermark=float(adaptive_window_runtime["high_watermark"]),
            limit_min=int(adaptive_window_runtime["d_limit_min"]),
            limit_max=int(adaptive_window_runtime["d_limit_max"]),
        )
    adaptive_window_runtime["c_dynamic_limit"] = c_dynamic_limit
    adaptive_window_runtime["d_dynamic_limit"] = d_dynamic_limit
    d_device_pool = scheduler_allocators._build_d_device_pool(
        c_gpu_index=int(adaptive_window_runtime["c_gpu_index"]),
        d_gpu_index=int(adaptive_window_runtime["d_gpu_index"]),
    )
    return _build_adaptive_window_snapshot(
        adaptive_window_runtime=adaptive_window_runtime,
        render_limit=render_limit,
        render_backend=render_backend,
        current_phase="idle",
        d_dispatch_enabled=False,
        d_device_pool=d_device_pool,
        d_device_inflight={device: 0 for device in d_device_pool},
    )


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


def _build_adaptive_window_runtime(
    context: RuntimeContext,
    global_render_limit: int,
    render_backend: str,
) -> dict[str, Any]:
    """
    功能说明：构建跨模块 C/D 自适应并发窗口运行态配置。
    """
    adaptive_cfg = context.config.cross_module.adaptive_window
    low_watermark, high_watermark = _normalize_gpu_watermarks(
        low_watermark=adaptive_cfg.low_watermark,
        high_watermark=adaptive_cfg.high_watermark,
    )
    c_limit_min, c_limit_max = _normalize_limit_range(
        limit_min=adaptive_cfg.c_limit_min,
        limit_max=adaptive_cfg.c_limit_max,
        fallback_min=1,
        fallback_max=max(1, global_render_limit),
    )
    d_limit_min, d_limit_max = _normalize_limit_range(
        limit_min=adaptive_cfg.d_limit_min,
        limit_max=adaptive_cfg.d_limit_max,
        fallback_min=1,
        fallback_max=max(1, min(2, global_render_limit)),
    )
    detected_gpu_count, gpu_count_source = _detect_available_gpu_count(context=context)
    single_gpu_mode = int(detected_gpu_count) <= 1
    c_gpu_index = _normalize_gpu_index(adaptive_cfg.c_gpu_index, fallback=0)
    d_gpu_index = _normalize_gpu_index(adaptive_cfg.d_gpu_index, fallback=1)
    if int(detected_gpu_count) > 0:
        c_gpu_index = c_gpu_index if c_gpu_index < int(detected_gpu_count) else 0
        d_gpu_index = d_gpu_index if d_gpu_index < int(detected_gpu_count) else 0
    if single_gpu_mode:
        c_gpu_index = 0
        d_gpu_index = 0
        c_limit_min, c_limit_max = 1, 1
        d_limit_min, d_limit_max = 1, 1

    fallback_c_limit = min(c_limit_max, max(c_limit_min, global_render_limit))
    fallback_d_limit = min(d_limit_max, max(d_limit_min, min(global_render_limit, d_limit_max)))

    enabled = bool(adaptive_cfg.enabled)
    if not enabled:
        return {
            "enabled": False,
            "probe_interval_seconds": _normalize_probe_interval_seconds(adaptive_cfg.probe_interval_ms),
            "low_watermark": low_watermark,
            "high_watermark": high_watermark,
            "c_gpu_index": c_gpu_index,
            "d_gpu_index": d_gpu_index,
            "c_limit_min": c_limit_min,
            "c_limit_max": c_limit_max,
            "d_limit_min": d_limit_min,
            "d_limit_max": d_limit_max,
            "fallback_c_limit": 1 if single_gpu_mode else global_render_limit,
            "fallback_d_limit": 1 if single_gpu_mode else global_render_limit,
            "c_dynamic_limit": 1 if single_gpu_mode else global_render_limit,
            "d_dynamic_limit": 1 if single_gpu_mode else global_render_limit,
            "max_render_workers": 2 if single_gpu_mode else global_render_limit,
            "last_probe_rows": [],
            "last_probe_error": "disabled",
            "single_gpu_mode": single_gpu_mode,
            "detected_gpu_count": int(detected_gpu_count),
            "gpu_count_source": str(gpu_count_source),
            "oom_fallback_locked_c_then_d": False,
        }

    c_dynamic_limit = fallback_c_limit
    d_dynamic_limit = fallback_d_limit
    max_render_workers = 2 if single_gpu_mode else max(1, c_limit_max + d_limit_max)
    return {
        "enabled": True,
        "probe_interval_seconds": _normalize_probe_interval_seconds(adaptive_cfg.probe_interval_ms),
        "low_watermark": low_watermark,
        "high_watermark": high_watermark,
        "c_gpu_index": c_gpu_index,
        "d_gpu_index": d_gpu_index,
        "c_limit_min": c_limit_min,
        "c_limit_max": c_limit_max,
        "d_limit_min": d_limit_min,
        "d_limit_max": d_limit_max,
        "fallback_c_limit": c_dynamic_limit,
        "fallback_d_limit": d_dynamic_limit,
        "c_dynamic_limit": c_dynamic_limit,
        "d_dynamic_limit": d_dynamic_limit,
        "max_render_workers": max_render_workers,
        "last_probe_rows": [],
        "last_probe_error": "",
        "single_gpu_mode": single_gpu_mode,
        "detected_gpu_count": int(detected_gpu_count),
        "gpu_count_source": str(gpu_count_source),
        "oom_fallback_locked_c_then_d": False,
    }


def _detect_available_gpu_count(context: RuntimeContext) -> tuple[int, str]:
    """
    功能说明：检测可用 GPU 数量（优先 probe，失败后回退 torch）。
    """
    probe_rows, probe_error = _run_gpu_probe_script(context=context, timeout_seconds=1.5)
    if not probe_error and probe_rows:
        probe_indexes: set[int] = set()
        for row in probe_rows:
            if not isinstance(row, dict):
                continue
            try:
                probe_indexes.add(int(row.get("index")))
            except (TypeError, ValueError):
                continue
        if probe_indexes:
            return len(probe_indexes), "probe"
    torch_count = _detect_gpu_count_from_torch()
    if torch_count is not None:
        return int(torch_count), "torch"
    return 0, "unknown"


def _detect_gpu_count_from_torch() -> int | None:
    """
    功能说明：尝试通过 torch 检测 CUDA 设备数量。
    """
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        if not bool(torch.cuda.is_available()):
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return None


def _run_gpu_probe_script(context: RuntimeContext, timeout_seconds: float) -> tuple[list[dict[str, Any]], str]:
    """
    功能说明：调用轻量 GPU 采样脚本并返回结构化显存占用。
    """
    _ = context
    script_path = (Path(__file__).resolve().parents[4] / GPU_PROBE_SCRIPT_RELATIVE_PATH).resolve()
    if not script_path.exists():
        return [], f"gpu_probe_missing: {script_path}"

    command = [
        sys.executable,
        str(script_path),
        "--json",
        "--timeout",
        str(max(0.5, float(timeout_seconds))),
    ]
    try:
        process_result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_seconds)),
        )
    except Exception as error:  # noqa: BLE001
        return [], f"gpu_probe_exec_failed: {error}"

    stdout_text = str(process_result.stdout or "").strip()
    stderr_text = str(process_result.stderr or "").strip()
    if not stdout_text:
        return [], f"gpu_probe_no_stdout: exit_code={process_result.returncode}, stderr={stderr_text}"
    payload_text = stdout_text.splitlines()[-1].strip()
    try:
        payload = json.loads(payload_text)
    except Exception as error:  # noqa: BLE001
        return [], f"gpu_probe_json_failed: {error}"

    if not isinstance(payload, dict):
        return [], "gpu_probe_payload_invalid: expected object"
    payload_ok = bool(payload.get("ok", False))
    payload_error = str(payload.get("error", "")).strip()
    if process_result.returncode != 0 or not payload_ok:
        if payload_error:
            return [], payload_error
        return [], f"gpu_probe_failed: exit_code={process_result.returncode}, stderr={stderr_text}"

    raw_rows = payload.get("gpus", [])
    if not isinstance(raw_rows, list):
        return [], "gpu_probe_payload_invalid: gpus must be list"

    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        try:
            gpu_index = int(item.get("index"))
            total_mb = int(item.get("total_mb"))
            used_mb = int(item.get("used_mb"))
            if total_mb <= 0:
                continue
            used_ratio = float(item.get("used_ratio", float(used_mb) / float(total_mb)))
            used_ratio = max(0.0, min(1.0, used_ratio))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "index": gpu_index,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "used_ratio": round(used_ratio, 6),
            }
        )
    if not rows:
        return [], "gpu_probe_payload_invalid: empty gpus"
    rows.sort(key=lambda row: int(row.get("index", 0)))
    return rows, ""


def _extract_gpu_used_ratio(probe_rows: list[dict[str, Any]], gpu_index: int) -> float | None:
    """
    功能说明：从采样结果中提取指定 GPU 显存占用比例。
    """
    for row in probe_rows:
        try:
            row_index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if row_index != int(gpu_index):
            continue
        try:
            used_ratio = float(row.get("used_ratio"))
            return max(0.0, min(1.0, used_ratio))
        except (TypeError, ValueError):
            pass
        try:
            used_mb = float(row.get("used_mb"))
            total_mb = float(row.get("total_mb"))
            if total_mb <= 0:
                return None
            used_ratio = used_mb / total_mb
            return max(0.0, min(1.0, used_ratio))
        except (TypeError, ValueError):
            return None
    return None


def _adjust_dynamic_limit(
    current_limit: int,
    used_ratio: float | None,
    low_watermark: float,
    high_watermark: float,
    limit_min: int,
    limit_max: int,
) -> int:
    """
    功能说明：根据显存占用比例动态调整并发窗口（单步增减 1）。
    """
    try:
        fallback_limit_max = int(limit_max)
    except (TypeError, ValueError):
        fallback_limit_max = 1
    normalized_limit_min, normalized_limit_max = _normalize_limit_range(
        limit_min=limit_min,
        limit_max=limit_max,
        fallback_min=1,
        fallback_max=max(1, fallback_limit_max),
    )
    try:
        normalized_current = int(current_limit)
    except (TypeError, ValueError):
        normalized_current = normalized_limit_min
    normalized_current = max(normalized_limit_min, min(normalized_limit_max, normalized_current))
    if used_ratio is None:
        return normalized_current
    if used_ratio <= low_watermark and normalized_current < normalized_limit_max:
        return normalized_current + 1
    if used_ratio >= high_watermark and normalized_current > normalized_limit_min:
        return normalized_current - 1
    return normalized_current


def _append_window_direction(history: list[int], old_limit: int, new_limit: int) -> list[int]:
    """
    功能说明：向窗口变化历史追加方向标记（+1 上调，-1 下调）。
    """
    if int(new_limit) == int(old_limit):
        return list(history[-4:])
    direction = 1 if int(new_limit) > int(old_limit) else -1
    merged = list(history)
    merged.append(direction)
    return merged[-4:]


def _is_two_round_trip_flap(history: list[int]) -> bool:
    """
    功能说明：判断窗口是否出现“来回往复两轮”抖动。
    """
    if len(history) < 4:
        return False
    tail = history[-4:]
    return tail in ([1, -1, 1, -1], [-1, 1, -1, 1])


def _build_adaptive_window_snapshot(
    adaptive_window_runtime: dict[str, Any],
    render_limit: int,
    render_backend: str,
    current_phase: str = "unknown",
    d_dispatch_enabled: bool = False,
    d_device_pool: list[str] | None = None,
    d_device_inflight: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    功能说明：构建可直接对外展示的自适应并发窗口快照。
    """
    probe_rows = adaptive_window_runtime.get("last_probe_rows", [])
    if not isinstance(probe_rows, list):
        probe_rows = []
    c_gpu_index = int(adaptive_window_runtime.get("c_gpu_index", 0))
    d_gpu_index = int(adaptive_window_runtime.get("d_gpu_index", 1))
    last_probe_error = str(adaptive_window_runtime.get("last_probe_error", ""))
    if d_device_pool is None:
        d_device_pool = scheduler_allocators._build_d_device_pool(c_gpu_index=c_gpu_index, d_gpu_index=d_gpu_index)
    if d_device_inflight is None:
        d_device_inflight = {device: 0 for device in d_device_pool}
    return {
        "enabled": bool(adaptive_window_runtime.get("enabled", False)),
        "render_backend": render_backend,
        "global_render_limit": int(render_limit),
        "current_phase": str(current_phase),
        "d_dispatch_enabled": bool(d_dispatch_enabled),
        "d_device_pool": list(d_device_pool),
        "d_device_inflight": {str(key): int(value) for key, value in dict(d_device_inflight).items()},
        "probe_interval_ms": int(round(float(adaptive_window_runtime.get("probe_interval_seconds", 1.0)) * 1000)),
        "low_watermark": float(adaptive_window_runtime.get("low_watermark", 0.65)),
        "high_watermark": float(adaptive_window_runtime.get("high_watermark", 0.96)),
        "c_gpu_index": c_gpu_index,
        "d_gpu_index": d_gpu_index,
        "c_limit_min": int(adaptive_window_runtime.get("c_limit_min", 1)),
        "c_limit_max": int(adaptive_window_runtime.get("c_limit_max", render_limit)),
        "d_limit_min": int(adaptive_window_runtime.get("d_limit_min", 1)),
        "d_limit_max": int(adaptive_window_runtime.get("d_limit_max", render_limit)),
        "c_dynamic_limit": int(adaptive_window_runtime.get("c_dynamic_limit", render_limit)),
        "d_dynamic_limit": int(adaptive_window_runtime.get("d_dynamic_limit", render_limit)),
        "fallback_c_limit": int(adaptive_window_runtime.get("fallback_c_limit", render_limit)),
        "fallback_d_limit": int(adaptive_window_runtime.get("fallback_d_limit", render_limit)),
        "single_gpu_mode": bool(adaptive_window_runtime.get("single_gpu_mode", False)),
        "detected_gpu_count": int(adaptive_window_runtime.get("detected_gpu_count", 0)),
        "gpu_count_source": str(adaptive_window_runtime.get("gpu_count_source", "")),
        "oom_fallback_locked_c_then_d": bool(adaptive_window_runtime.get("oom_fallback_locked_c_then_d", False)),
        "last_probe_ok": bool(not last_probe_error and len(probe_rows) > 0),
        "last_probe_error": last_probe_error,
        "c_gpu_used_ratio": _extract_gpu_used_ratio(probe_rows=probe_rows, gpu_index=c_gpu_index),
        "d_gpu_used_ratio": _extract_gpu_used_ratio(probe_rows=probe_rows, gpu_index=d_gpu_index),
    }


def _normalize_probe_interval_seconds(probe_interval_ms: int) -> float:
    """
    功能说明：归一化 GPU 采样间隔并转换为秒。
    """
    try:
        normalized = int(probe_interval_ms)
    except (TypeError, ValueError):
        return 1.0
    if normalized < 200:
        return 0.2
    if normalized > 10000:
        return 10.0
    return normalized / 1000.0


def _normalize_gpu_watermarks(low_watermark: float, high_watermark: float) -> tuple[float, float]:
    """
    功能说明：归一化显存占用阈值。
    """
    try:
        low = float(low_watermark)
    except (TypeError, ValueError):
        low = 0.65
    try:
        high = float(high_watermark)
    except (TypeError, ValueError):
        high = 0.96
    low = max(0.05, min(0.95, low))
    high = max(0.10, min(0.99, high))
    if low >= high:
        low = 0.65
        high = 0.96
    return low, high


def _normalize_gpu_index(gpu_index: int, fallback: int) -> int:
    """
    功能说明：归一化 GPU 索引配置。
    """
    try:
        normalized = int(gpu_index)
    except (TypeError, ValueError):
        return max(0, int(fallback))
    if normalized < 0:
        return max(0, int(fallback))
    return normalized


def _normalize_limit_range(limit_min: int, limit_max: int, fallback_min: int, fallback_max: int) -> tuple[int, int]:
    """
    功能说明：归一化并发窗口上下限。
    """
    try:
        normalized_min = int(limit_min)
    except (TypeError, ValueError):
        normalized_min = int(fallback_min)
    try:
        normalized_max = int(limit_max)
    except (TypeError, ValueError):
        normalized_max = int(fallback_max)
    normalized_min = max(1, min(16, normalized_min))
    normalized_max = max(1, min(16, normalized_max))
    if normalized_min > normalized_max:
        normalized_min = normalized_max
    return normalized_min, normalized_max


def _normalize_b_worker_limit(script_workers: int) -> int:
    """
    功能说明：归一化模块 B 并发上限。
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


def _normalize_global_render_limit(global_render_limit: int) -> int:
    """
    功能说明：归一化跨模块 C/D 共享并发上限。
    """
    try:
        normalized = int(global_render_limit)
    except (TypeError, ValueError):
        return 3
    if normalized < 1:
        return 3
    if normalized > 16:
        return 16
    return normalized


def _normalize_scheduler_tick_seconds(scheduler_tick_ms: int) -> float:
    """
    功能说明：归一化调度轮询间隔并转换为秒。
    """
    try:
        normalized = int(scheduler_tick_ms)
    except (TypeError, ValueError):
        return 0.05
    if normalized < 10:
        return 0.05
    if normalized > 1000:
        return 1.0
    return normalized / 1000.0
