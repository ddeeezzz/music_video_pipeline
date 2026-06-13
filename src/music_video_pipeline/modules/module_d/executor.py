"""
文件用途：实现模块 D 的 Remotion 单元执行与重试逻辑。
核心流程：预热 ComfyUI 服务 -> 解析模块 D 单元模板参数 -> 全量执行 Remotion 渲染 -> 写入状态与产物路径。
输入输出：输入运行上下文与模块 D 单元，输出片段路径或执行副作用。
依赖说明：依赖标准库并发工具与项目内 RuntimeContext/ModuleDUnit/Remotion 渲染后端。
维护说明：所有 shot 统一走 Remotion 渲染，不再走 ToonCrafter 路径。
"""

# 标准库：用于线程池并发。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于 JSON 序列化模板请求。
import json
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：运行上下文定义。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 读取工具（供旧辅助函数保留使用）。
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块 D ComfyUI 渲染后端（仅预热用）。
from music_video_pipeline.modules.module_d.backends import (
    prewarm_comfyui_runtime as prewarm_comfyui_runtime_backend,
)
# 项目内模块：模块 D FFmpeg 后处理工具。
from music_video_pipeline.modules.module_d.finalizer import apply_camera_plan_to_segment
# 项目内模块：模块 D 单元模型。
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnit

# 第三方库：用于处理图片白底变透明。
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# 常量：模板画布固定宽高（1920×1200 16:10），与 Remotion 模板工程对齐。
_TEMPLATE_WIDTH = 1920
_TEMPLATE_HEIGHT = 1200
# 常量：符号占画布比例（多主体模板，基于单格宽度 visible_cell_count=3，格子比例 3:4）。
# 每个格子宽度 = 1920/3 = 640，取 0.28 ≈ 538px 留边距；高度按 3:4 比例 = 538/(3/4) ≈ 717px
_GRID_WIDTH_RATIO = 0.28   # 每格约 1920*0.28 ≈ 538px
_GRID_HEIGHT_RATIO = round(0.28 / (3.0 / 4.0), 2)  # 538/717 ≈ 0.37，按 3:4 比例
# 常量：检测白底的亮度阈值（0-255），高于此值的像素视为白底。
_WHITE_THRESHOLD = 200


def prewarm_comfyui_runtime(context: RuntimeContext, device_override: str | None = None) -> dict[str, str]:
    """
    功能说明：预热模块 D runtime（Remotion 模式）。
    参数说明：
    - context: 运行上下文对象。
    - device_override: 预留字段。
    返回值：
    - dict[str, str]: 预热摘要。
    异常说明：
    - RuntimeError: ComfyUI 服务探活失败时抛出。
    边界条件：Remotion 模式不要求 ToonCrafter 模型存在；ToonCrafter 相关校验失败仅告警。
    """
    try:
        return prewarm_comfyui_runtime_backend(context=context, device_override=device_override)
    except RuntimeError as error:
        error_text = str(error)
        # Remotion 模式：ToonCrafter 模型缺失不阻断预热
        if "主模型不存在" in error_text or "sketch encoder 不存在" in error_text:
            context.logger.warning(
                "模块D 预热跳过 ToonCrafter 模型检查（当前 Remotion 模式），警告=%s",
                error_text,
            )
            return {
                "backend": "remotion",
                "device": str(device_override or "remotion-local"),
                "note": "ToonCrafter models not checked (Remotion mode)",
            }
        raise


def resolve_render_profile(context: RuntimeContext) -> dict[str, Any]:
    """
    功能说明：返回模块 D 当前渲染 profile 摘要。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - dict[str, Any]: 供调度层观测的 profile 字典。
    异常说明：无。
    边界条件：模块 D 统一走 Remotion 渲染；ToonCrafter 路径保留为待启用扩展。
    """
    _ = context
    return {
        "render_backend": "remotion",
        "name": "remotion",
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
            "模块D单元执行轮次开始，task_id=%s，attempt=%s/%s，pending_count=%s，workers=%s，backend=remotion",
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
                "模块D单元执行有失败，准备重试，task_id=%s，attempt=%s/%s，failed_count=%s，backend=remotion",
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
    功能说明：并行执行模块 D 的 Remotion 单元渲染任务。
    参数说明：
    - context: 运行上下文对象。
    - pending_units: 待执行单元数组。
    - worker_count: 并发 worker 数量。
    返回值：
    - list[tuple[ModuleDUnit, Exception]]: 失败单元与异常信息数组。
    异常说明：无（异常统一转换为失败列表返回）。
    边界条件：Remotion 渲染为本地子进程，并发数受 CPU/内存限制。
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
    功能说明：执行一次模块 D 的 Remotion 渲染并写入状态。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - Exception | None: 成功返回 None，失败返回异常对象。
    异常说明：无（异常转为返回值，交给上层重试逻辑处理）。
    边界条件：所有 shot 统一走 Remotion 路径，不再走 ToonCrafter。
    """
    try:
        result = _render_unit_via_remotion(context=context, unit=unit)
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


def _render_unit_via_remotion(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, Any]:
    """
    功能说明：将任意模块 D 单元通过 Remotion 渲染为视频片段（首尾帧合成）。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    返回值：
    - dict[str, Any]: 渲染摘要（backend/segment_path/composition_id/frame_count_used）。
    异常说明：
    - RuntimeError: 模板参数、素材路径或 Remotion 渲染失败时抛出。
    边界条件：根据 remotion_id 自动选择多主体模板或单主体模板；
    管道执行固定首尾帧模式；前端重跑按键通过独立 API 控制不同模式。
    """
    remotion_id = str(unit.shot.get("remotion_id", "")).strip() or "CenterTemplate"

    if remotion_id in {"GridTemplate", "ScrollTemplate"}:
        return _render_one_unit_remotion_template(context=context, unit=unit)

    if remotion_id in {"TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"}:
        return _render_one_unit_transition_template(context=context, unit=unit, remotion_id=remotion_id)

    # --- 单主体模板（CenterTemplate 等）：frames 数组包含首尾帧 ---
    start_path = str(unit.shot.get("frame_path_start", "")).strip()
    end_path = str(unit.shot.get("frame_path_end", "")).strip()
    if not start_path:
        raise RuntimeError(
            f"模块D Remotion 渲染失败：缺失首帧素材，unit_id={unit.unit_id}"
        )
    if not Path(start_path).exists():
        raise RuntimeError(
            f"模块D Remotion 渲染失败：首帧素材不存在，unit_id={unit.unit_id}，path={start_path}"
        )

    start_sym = _build_symbol_payload_single(path=start_path, unit_id=unit.unit_id)
    end_sym = (
        _build_symbol_payload_single(path=end_path, unit_id=unit.unit_id)
        if end_path and Path(end_path).exists()
        else start_sym
    )

    props: dict[str, Any] = {
        "template": remotion_id,
        "fps": int(context.config.ffmpeg.fps),
        "duration_in_frames": int(unit.exact_frames),
        "bpm": 120,
        "background": {"kind": "solid", "color": "white"},
        "frames": [start_sym, end_sym],
        "motion": {"breathe": True},
    }

    # 传递 subject_kind 给 Remotion（object 类型触发缓慢旋转效果）
    sk = str(unit.shot.get("subject_kind", "") or "").strip().lower()
    if sk:
        props["subject_kind"] = sk

    props_dir = context.artifacts_dir / "template_requests"
    props_dir.mkdir(parents=True, exist_ok=True)
    props_path = props_dir / f"{unit.unit_id}.{remotion_id}.json"
    props_path.write_text(json.dumps(props, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[4]
    remotion_project_dir = (project_root / "remotion_templates").resolve()
    from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment

    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id=remotion_id,
        props_json_path=props_path,
        output_path=unit.segment_path,
    )
    return {
        "backend": f"remotion-{remotion_id}",
        "segment_path": str(unit.segment_path),
        "composition_id": remotion_id,
        "frame_count_used": int(unit.exact_frames),
    }


def _build_symbol_payload_single(path: str, unit_id: str) -> dict[str, Any]:
    """
    功能说明：为单主体模板构建全屏 symbol payload。
    参数说明：
    - path: 帧文件路径。
    - unit_id: 单元标识。
    返回值：
    - dict[str, Any]: Remotion symbol 参数（全屏尺寸）。
    异常说明：
    - RuntimeError: 路径为空或文件不存在时抛出。
    """
    if not path:
        raise RuntimeError(f"模块D Remotion 渲染失败：素材路径为空，unit_id={unit_id}")
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise RuntimeError(f"模块D Remotion 渲染失败：素材不存在，unit_id={unit_id}，path={path}")
    return {
        "src": resolved.as_uri(),
        "width_ratio": 1.0,
        "height_ratio": 1.0,
    }


def _is_multi_subject_template_unit(unit: ModuleDUnit) -> bool:
    """
    功能说明：判断当前模块 D 单元是否应走 Remotion 多主体模板合成。
    参数说明：
    - unit: 模块 D 单元对象。
    返回值：
    - bool: GridTemplate/ScrollTemplate 且包含 template_slots 时返回 True。
    异常说明：无。
    边界条件：无 slot 时继续走 ToonCrafter，避免误吞单主体旧产物。
    """
    remotion_id = str(unit.shot.get("remotion_id", "")).strip()
    template_slots = unit.shot.get("template_slots")
    return remotion_id in {"GridTemplate", "ScrollTemplate"} and isinstance(template_slots, list) and bool(template_slots)


def _render_one_unit_remotion_template(context: RuntimeContext, unit: ModuleDUnit) -> dict[str, Any]:
    """
    功能说明：用 Remotion 多主体模板把多个格子素材合成为一个视频片段。
    参数说明：
    - context: 运行上下文对象。
    - unit: 已聚合的模块 D 单元对象。
    返回值：
    - dict[str, Any]: 渲染摘要。
    异常说明：
    - RuntimeError: 模板参数、素材或 Remotion 渲染失败时抛出。
    边界条件：当前只处理 GridTemplate/ScrollTemplate，多余素材截断为前三个，不足三个复制最后一个。
    """
    remotion_id = str(unit.shot.get("remotion_id", "")).strip()
    template_slots = unit.shot.get("template_slots")
    if remotion_id not in {"GridTemplate", "ScrollTemplate"}:
        raise RuntimeError(f"模块D Remotion 渲染失败：不支持的多主体模板，unit_id={unit.unit_id}，remotion_id={remotion_id}")
    if not isinstance(template_slots, list) or not template_slots:
        raise RuntimeError(f"模块D Remotion 渲染失败：缺失 template_slots，unit_id={unit.unit_id}")

    normalized_slots = [slot for slot in template_slots if isinstance(slot, dict)]
    if not normalized_slots:
        raise RuntimeError(f"模块D Remotion 渲染失败：template_slots 为空，unit_id={unit.unit_id}")
    while len(normalized_slots) < 3:
        normalized_slots.append(dict(normalized_slots[-1]))
    normalized_slots = normalized_slots[:3]

    symbol_trim_dir = context.artifacts_dir / "symbols_trimmed"
    slots: list[dict[str, Any]] = []
    for slot in normalized_slots:
        start_sym = _build_symbol_payload(slot=slot, frame_key="frame_path_start", unit_id=unit.unit_id,
                                          trim_dir=symbol_trim_dir)
        end_sym = _build_symbol_payload(slot=slot, frame_key="frame_path_end", unit_id=unit.unit_id,
                                        trim_dir=symbol_trim_dir)
        slots.append({"frames": [start_sym, end_sym]})

    props: dict[str, Any] = {
        "template": remotion_id,
        "fps": int(context.config.ffmpeg.fps),
        "duration_in_frames": int(unit.exact_frames),
        "bpm": 120,
        "background": {"kind": "solid", "color": "white"},
        "slots": slots,
        "layout": {"visible_cell_count": 3},
    }
    if remotion_id == "GridTemplate":
        props["motion"] = {"active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 72}
    else:
        props["motion"] = {"loop": False}

    props_dir = context.artifacts_dir / "template_requests"
    props_dir.mkdir(parents=True, exist_ok=True)
    props_path = props_dir / f"{unit.unit_id}.{remotion_id}.json"
    props_path.write_text(json.dumps(props, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[4]
    remotion_project_dir = (project_root / "remotion_templates").resolve()
    from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment

    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id=remotion_id,
        props_json_path=props_path,
        output_path=unit.segment_path,
    )
    return {
        "backend": "remotion-template",
        "segment_path": str(unit.segment_path),
        "composition_id": remotion_id,
        "slot_count": len(template_slots),
        "frame_count_used": int(unit.exact_frames),
    }


def _render_one_unit_transition_template(
    context: RuntimeContext,
    unit: ModuleDUnit,
    remotion_id: str,
) -> dict[str, Any]:
    """
    功能说明：用 Remotion 转场模板合成前后双场景过渡视频片段。
    参数说明：
    - context: 运行上下文对象。
    - unit: 模块 D 单元对象。
    - remotion_id: 转场模板标识（TiltUpTemplate/TiltDownTemplate/PanRightTemplate）。
    返回值：
    - dict[str, Any]: 渲染摘要。
    异常说明：
    - RuntimeError: 模板参数、素材或 Remotion 渲染失败时抛出。
    边界条件：首帧作为 scene_before（旧场景），尾帧作为 scene_after（新场景）。
    """
    start_path = str(unit.shot.get("frame_path_start", "")).strip()
    end_path = str(unit.shot.get("frame_path_end", "")).strip()
    if not start_path:
        raise RuntimeError(
            f"模块D Remotion 转场渲染失败：缺失首帧素材，unit_id={unit.unit_id}"
        )
    if not Path(start_path).exists():
        raise RuntimeError(
            f"模块D Remotion 转场渲染失败：首帧素材不存在，unit_id={unit.unit_id}，path={start_path}"
        )

    before_sym = _build_symbol_payload_single(path=start_path, unit_id=unit.unit_id)
    after_sym = (
        _build_symbol_payload_single(path=end_path, unit_id=unit.unit_id)
        if end_path and Path(end_path).exists()
        else before_sym
    )

    # 查找 ToonCrafter 插值帧序列作为 frames 输入
    unit_id_text = str(unit.unit_id)
    shot_id = str(unit.shot.get("shot_id", "")).strip()
    tc_base = context.artifacts_dir / "tooncrafter_frames" / unit_id_text / shot_id
    extra_frames: list[dict[str, Any]] = []
    if tc_base.is_dir():
        tc_files = sorted(tc_base.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        if tc_files:
            for fp in tc_files:
                uri = fp.resolve().as_uri()
                extra_frames.append({"src": uri, "width_ratio": 1.0, "height_ratio": 1.0})

    # travel_px：TiltUp/TiltDown 使用高度 1200，PanRight 使用宽度 1920
    travel_px = 1920 if remotion_id == "PanRightTemplate" else 1200

    props: dict[str, Any] = {
        "template": remotion_id,
        "fps": int(context.config.ffmpeg.fps),
        "duration_in_frames": int(unit.exact_frames),
        "bpm": 120,
        "scene_before": {
            "background": {"kind": "solid", "color": "white"},
            "symbol": before_sym,
        },
        "scene_after": {
            "background": {"kind": "solid", "color": "white"},
            "symbol": after_sym,
        },
        "motion": {"travel_px": travel_px, "easing": "ease_in_out"},
    }
    if extra_frames:
        props["frames"] = extra_frames

    props_dir = context.artifacts_dir / "template_requests"
    props_dir.mkdir(parents=True, exist_ok=True)
    props_path = props_dir / f"{unit.unit_id}.{remotion_id}.json"
    props_path.write_text(json.dumps(props, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[4]
    remotion_project_dir = (project_root / "remotion_templates").resolve()
    from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment

    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id=remotion_id,
        props_json_path=props_path,
        output_path=unit.segment_path,
    )
    return {
        "backend": f"remotion-transition-{remotion_id}",
        "segment_path": str(unit.segment_path),
        "composition_id": remotion_id,
        "frame_count_used": int(unit.exact_frames),
    }


def _trim_white_to_transparent(src_path: Path, output_dir: Path, target_w: int, target_h: int) -> Path:
    """
    功能说明：从原图中心按 3:4 比例裁切并缩放至目标尺寸。
    不走白底变透明，直接取中心区域——GridTemplate 的格子比例固定为 3:4，
    ToonCrafter 输出的放大重绘图取中心 3:4 区域即可得到正确的格子画面。
    参数说明：
    - src_path: 原始 PNG 路径。
    - output_dir: 处理后图片输出目录。
    - target_w: 目标宽度（像素）。
    - target_h: 目标高度（像素）。
    返回值：
    - Path: 处理后的图片路径。
    边界条件：不含 PIL 时跳过处理直接返回原路径。
    """
    if Image is None:
        return src_path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{src_path.stem}_trimmed.png"
    if out_path.exists():
        return out_path
    try:
        with Image.open(src_path) as img:
            rgba = img.convert("RGBA")
            w, h = rgba.size
            # 取中心 3:4 区域：宽 3 : 高 4
            crop_ratio = 3.0 / 4.0
            img_ratio = w / h
            if img_ratio > crop_ratio:
                # 图更宽 → 以高度为基准，取中心 3:4 宽的区域
                crop_w = int(round(h * crop_ratio))
                crop_h = h
            else:
                # 图更高或等比例 → 以宽度为基准，取中心 3:4 高的区域
                crop_w = w
                crop_h = int(round(w / crop_ratio))
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            rgba = rgba.crop((left, top, left + crop_w, top + crop_h))
            rgba = rgba.resize((int(target_w), int(target_h)), Image.LANCZOS)
            rgba.save(out_path, "PNG")
        return out_path
    except Exception:
        return src_path


def _build_symbol_payload(slot: dict[str, Any], frame_key: str, unit_id: str,
                          trim_dir: Path | None = None) -> dict[str, Any]:
    """
    功能说明：将 template_slot 中的帧路径转换为 Remotion symbol payload。
    参数说明：
    - slot: 单个格子素材载荷。
    - frame_key: 帧路径字段名。
    - unit_id: 当前模块 D 单元标识。
    - trim_dir: 白底裁剪后图片输出目录；None 时不处理直接使用原图。
    返回值：
    - dict[str, Any]: Remotion symbol 参数。
    异常说明：
    - RuntimeError: 素材路径为空或文件不存在时抛出。
    边界条件：当 trim_dir 不为 None 时，白底变透明并裁剪到 _GRID_WIDTH_RATIO/_GRID_HEIGHT_RATIO 对应尺寸。
    """
    frame_path = str(slot.get(frame_key, "")).strip()
    if not frame_path:
        fallback_key = "frame_path_start" if frame_key == "frame_path_end" else "frame_path_end"
        frame_path = str(slot.get(fallback_key, "")).strip()
    if not frame_path:
        raise RuntimeError(f"模块D Remotion 渲染失败：格子素材缺失，unit_id={unit_id}，frame_key={frame_key}")
    src = Path(frame_path)
    if not src.exists():
        raise RuntimeError(f"模块D Remotion 渲染失败：格子素材不存在，unit_id={unit_id}，path={src}")
    if trim_dir is not None:
        target_w = int(round(_TEMPLATE_WIDTH * _GRID_WIDTH_RATIO))
        target_h = int(round(_TEMPLATE_HEIGHT * _GRID_HEIGHT_RATIO))
        processed = _trim_white_to_transparent(src_path=src, output_dir=trim_dir,
                                                target_w=target_w, target_h=target_h)
        return {
            "src": processed.resolve().as_uri(),
            "width_ratio": _GRID_WIDTH_RATIO,
            "height_ratio": _GRID_HEIGHT_RATIO,
        }
    return {
        "src": src.resolve().as_uri(),
        "width_ratio": 0.26,
        "height_ratio": 0.52,
    }


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
