"""
文件用途：封装跨模块调度中的任务执行与完成回收逻辑。
核心流程：执行 B/C/D 单元、回收 Future、写入失败阻断状态。
输入输出：输入运行上下文与单元对象，输出执行结果或失败记录。
依赖说明：依赖模块 B/C/D 执行器与状态库。
维护说明：本模块不负责调度策略与并发窗口调参。
"""

from concurrent.futures import Future
import logging
from pathlib import Path
from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.io_utils import read_json
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
from music_video_pipeline.modules.module_b_v2 import run_module_b_v2_incremental
from music_video_pipeline.modules.module_c.executor import execute_one_unit_with_retry as execute_one_c_unit
from music_video_pipeline.modules.module_c.unit_models import ModuleCUnit
from music_video_pipeline.modules.module_d.executor import execute_one_unit_with_retry as execute_one_d_unit
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint, materialize_module_d_unit


def _run_b_chain_unit(
    context: RuntimeContext,
    unit: ModuleBUnit,
    generator: Any,
    module_a_output: dict[str, Any],
    unit_outputs_dir: Any,
) -> str:
    """
    功能说明：旧模块B单元执行测试钩子占位；正式运行已改走模块B v2 批量生产器。
    参数说明：保留旧签名，仅供测试 monkeypatch 复用。
    返回值：
    - str: 单元产物路径字符串。
    异常说明：
    - RuntimeError: 未被测试桩替换时调用即报错。
    边界条件：生产代码不应再直接调用本函数。
    """
    raise RuntimeError("旧模块B单元执行入口已删除；请改走模块B v2 批量生产器。")


_DEFAULT_LEGACY_B_CHAIN_UNIT = _run_b_chain_unit


def _drain_finished_tasks(
    context: RuntimeContext,
    active_tasks: dict[Future, tuple[str, int, Any]],
    failed_chain_indexes: set[int],
    failed_errors: dict[int, str],
) -> None:
    """
    功能说明：处理已完成 future 的结果，并写入链路失败隔离。
    参数说明：
    - context: 运行上下文对象。
    - active_tasks: 活跃任务映射。
    - failed_chain_indexes: 失败链路集合（会被原位更新）。
    - failed_errors: 失败错误映射（会被原位更新）。
    返回值：无。
    异常说明：无。
    边界条件：仅消费已完成任务，不阻塞等待。
    """
    finished_futures = [future for future in active_tasks if future.done()]
    for future in finished_futures:
        stage, unit_index, metadata = active_tasks.pop(future)
        try:
            result = future.result()
            if stage == "B_BATCH":
                failed_batch_indexes = [
                    int(item)
                    for item in ((result or {}).get("failed_indexes", []) if isinstance(result, dict) else [])
                ]
                error_text = str((result or {}).get("error", "")).strip() if isinstance(result, dict) else ""
                for failed_index in failed_batch_indexes:
                    failed_chain_indexes.add(failed_index)
                    failed_errors[failed_index] = f"B:{error_text or '模块B v2 批量执行失败'}"
                    context.state_store.mark_bcd_downstream_blocked(
                        task_id=context.task_id,
                        unit_index=failed_index,
                        from_module="B",
                        reason=f"upstream_blocked:B:{error_text or '模块B v2 批量执行失败'}",
                    )
                if failed_batch_indexes:
                    logging.getLogger("B").error(
                        "跨模块链路模块B批量执行存在失败，task_id=%s，failed_indexes=%s，错误=%s",
                        context.task_id,
                        failed_batch_indexes,
                        error_text or "<unknown>",
                    )
        except Exception as error:  # noqa: BLE001
            failed_chain_indexes.add(unit_index)
            failed_errors[unit_index] = f"{stage}:{error}"
            if stage == "B":
                context.state_store.mark_bcd_downstream_blocked(
                    task_id=context.task_id,
                    unit_index=unit_index,
                    from_module="B",
                    reason=f"upstream_blocked:B:{error}",
                )
            elif stage == "C":
                context.state_store.mark_bcd_downstream_blocked(
                    task_id=context.task_id,
                    unit_index=unit_index,
                    from_module="C",
                    reason=f"upstream_blocked:C:{error}",
                )
            stage_logger = logging.getLogger(stage)
            stage_logger.error(
                "跨模块链路单元失败，task_id=%s，stage=%s，unit_index=%s，错误=%s",
                context.task_id,
                stage,
                unit_index,
                error,
            )


def _run_b_chain_batch(
    context: RuntimeContext,
    target_segment_ids: set[str],
    target_units: list[ModuleBUnit],
) -> dict[str, Any]:
    """
    功能说明：以模块B v2 增量方式执行一批跨模块链路的模块 B 单元。
    """
    legacy_b_chain_unit = globals().get("_run_b_chain_unit")
    if legacy_b_chain_unit is not _DEFAULT_LEGACY_B_CHAIN_UNIT:
        failed_indexes: list[int] = []
        failed_errors: list[str] = []
        unit_outputs_dir = context.artifacts_dir / "module_b_units"
        unit_outputs_dir.mkdir(parents=True, exist_ok=True)
        for target_unit in sorted(target_units, key=lambda item: int(item.unit_index)):
            try:
                legacy_b_chain_unit(
                    context,
                    target_unit,
                    None,
                    {},
                    unit_outputs_dir,
                )
            except Exception as error:  # noqa: BLE001
                failed_indexes.append(int(target_unit.unit_index))
                failed_errors.append(str(error))
        return {
            "output_path": "",
            "failed_indexes": failed_indexes,
            "error": "; ".join(failed_errors),
        }
    try:
        output_path = run_module_b_v2_incremental(
            context=context,
            target_segment_ids=target_segment_ids,
        )
        return {
            "output_path": str(output_path),
            "failed_indexes": [],
            "error": "",
        }
    except Exception as error:  # noqa: BLE001
        failed_indexes: list[int] = []
        for target_unit in sorted(target_units, key=lambda item: int(item.unit_index)):
            b_row = context.state_store.get_module_unit_record(
                task_id=context.task_id,
                module_name="B",
                unit_id=str(target_unit.unit_id),
            )
            b_status = str((b_row or {}).get("status", "pending"))
            if b_status == "done":
                continue
            failed_indexes.append(int(target_unit.unit_index))
        return {
            "output_path": "",
            "failed_indexes": failed_indexes,
            "error": str(error),
        }


def _run_c_chain_unit(
    context: RuntimeContext,
    chain_unit: CrossChainUnit,
    c_row: dict[str, Any],
    generator: Any,
    frames_dir: Any,
) -> dict[str, Any]:
    """
    功能说明：执行单条链路的模块 C 单元。
    """
    b_row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="B", unit_id=chain_unit.segment_id)
    if not b_row:
        raise RuntimeError(f"跨模块调度失败：模块B单元不存在，segment_id={chain_unit.segment_id}")
    shot_path = str(b_row.get("artifact_path", "")).strip()
    if not shot_path:
        raise RuntimeError(f"跨模块调度失败：模块B单元产物缺失，segment_id={chain_unit.segment_id}")

    shot = read_json(Path(shot_path))
    if not isinstance(shot, dict):
        raise RuntimeError(f"跨模块调度失败：模块B单元产物非法，segment_id={chain_unit.segment_id}")
    shot_obj = dict(shot)
    shot_obj["shot_id"] = chain_unit.shot_id
    if "start_time" not in shot_obj:
        shot_obj["start_time"] = float(c_row.get("start_time", chain_unit.start_time))
    if "end_time" not in shot_obj:
        shot_obj["end_time"] = float(c_row.get("end_time", chain_unit.end_time))

    unit = ModuleCUnit(
        unit_id=chain_unit.shot_id,
        unit_index=chain_unit.unit_index,
        shot=shot_obj,
        start_time=float(c_row.get("start_time", chain_unit.start_time)),
        end_time=float(c_row.get("end_time", chain_unit.end_time)),
        duration=float(c_row.get("duration", chain_unit.duration)),
    )
    return execute_one_c_unit(
        context=context,
        unit=unit,
        generator=generator,
        frames_dir=frames_dir,
    )


def _resolve_module_c_sidecar_path(artifact_path: str, unit_id: str) -> Path:
    """
    功能说明：根据模块C单元 artifact_path 解析对应 sidecar JSON 路径。
    参数说明：
    - artifact_path: module_unit_runs 中记录的模块C产物路径。
    - unit_id: 模块C单元ID（shot_id）。
    返回值：
    - Path: sidecar JSON 路径。
    异常说明：
    - RuntimeError: 路径非法或无法定位 artifacts 根目录时抛出。
    边界条件：兼容相对/绝对路径，但必须包含 artifacts 目录层级。
    """
    normalized_artifact_path = str(artifact_path).strip()
    normalized_unit_id = str(unit_id).strip()
    if not normalized_artifact_path:
        raise RuntimeError("跨模块调度失败：模块C单元产物路径为空。")
    if not normalized_unit_id:
        raise RuntimeError("跨模块调度失败：模块C单元ID为空，无法解析双关键帧 sidecar。")
    artifact_obj = Path(normalized_artifact_path)
    path_parts = artifact_obj.parts
    if "artifacts" not in path_parts:
        raise RuntimeError(
            "跨模块调度失败：模块C产物路径不在 artifacts 目录下，"
            f"artifact_path={normalized_artifact_path}。"
        )
    artifacts_index = max(index for index, part_text in enumerate(path_parts) if part_text == "artifacts")
    artifacts_dir = Path(*path_parts[: artifacts_index + 1])
    return artifacts_dir / "module_c_units" / f"{normalized_unit_id}.json"


def _load_strict_dual_frame_item_for_d(
    *,
    blueprint: ModuleDUnitBlueprint,
    c_row: dict[str, Any],
) -> dict[str, Any]:
    """
    功能说明：从模块C sidecar 读取并校验双关键帧契约，构建模块D输入 frame_item。
    参数说明：
    - blueprint: 模块D单元蓝图。
    - c_row: 状态库中的模块C单元记录。
    返回值：
    - dict[str, Any]: 可直接用于 materialize_module_d_unit 的 frame_item。
    异常说明：
    - RuntimeError: sidecar 缺失、结构非法或双关键帧字段不完整时抛出。
    边界条件：不再接受仅 frame_path 的单帧输入。
    """
    unit_id = str(c_row.get("unit_id", "")).strip() or str(blueprint.unit_id)
    artifact_path = str(c_row.get("artifact_path", "")).strip()
    sidecar_path = _resolve_module_c_sidecar_path(artifact_path=artifact_path, unit_id=unit_id)
    if not sidecar_path.exists():
        raise RuntimeError(
            "跨模块调度失败：缺失模块C双关键帧 sidecar，"
            f"unit_id={unit_id}，sidecar={sidecar_path}。"
        )
    payload = read_json(sidecar_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"跨模块调度失败：模块C sidecar 结构非法，sidecar={sidecar_path}")

    frame_path_start = str(payload.get("frame_path_start", "")).strip()
    frame_path_end = str(payload.get("frame_path_end", "")).strip()
    if (not frame_path_start) or (not frame_path_end):
        raise RuntimeError(
            "跨模块调度失败：模块C sidecar 缺失双关键帧字段，"
            f"unit_id={unit_id}，sidecar={sidecar_path}。"
        )
    control_frame_paths_payload = payload.get("control_frame_paths")
    if isinstance(control_frame_paths_payload, list):
        normalized_control_frame_paths = [str(item).strip() for item in control_frame_paths_payload if str(item).strip()]
    else:
        normalized_control_frame_paths = []
    if len(normalized_control_frame_paths) < 2:
        raise RuntimeError(
            "跨模块调度失败：模块C sidecar 缺失 control_frame_paths 双锚点，"
            f"unit_id={unit_id}，sidecar={sidecar_path}。"
        )
    if (
        str(normalized_control_frame_paths[0]) != frame_path_start
        or str(normalized_control_frame_paths[-1]) != frame_path_end
    ):
        raise RuntimeError(
            "跨模块调度失败：模块C sidecar 双关键帧字段不一致，"
            f"unit_id={unit_id}，frame_path_start={frame_path_start}，"
            f"frame_path_end={frame_path_end}，control_frame_paths={normalized_control_frame_paths}。"
        )

    return {
        **payload,
        "shot_id": str(blueprint.unit_id),
        "frame_path": str(payload.get("frame_path", "")).strip() or frame_path_start,
        "frame_path_start": frame_path_start,
        "frame_path_end": frame_path_end,
        "control_frame_paths": [str(normalized_control_frame_paths[0]), str(normalized_control_frame_paths[-1])],
        "start_time": float(payload.get("start_time", c_row.get("start_time", blueprint.start_time))),
        "end_time": float(payload.get("end_time", c_row.get("end_time", blueprint.end_time))),
        "duration": float(payload.get("duration", c_row.get("duration", blueprint.duration))),
    }


def _run_d_chain_unit(
    context: RuntimeContext,
    blueprint: ModuleDUnitBlueprint,
    c_row: dict[str, Any],
    profile: dict[str, Any],
    device_override: str | None = None,
) -> str:
    """
    功能说明：执行单条链路的模块 D 单元。
    """
    frame_item = _load_strict_dual_frame_item_for_d(blueprint=blueprint, c_row=c_row)
    unit = materialize_module_d_unit(blueprint=blueprint, frame_item=frame_item)
    segment_path = execute_one_d_unit(
        context=context,
        unit=unit,
        profile=profile,
        device_override=device_override,
    )
    return str(segment_path)
def _split_failed_stage_and_message(error_text: str) -> tuple[str, str]:
    """
    功能说明：解析失败文本中的阶段前缀与错误正文。
    """
    normalized = str(error_text)
    if ":" not in normalized:
        return "", normalized
    stage_name, message = normalized.split(":", 1)
    return str(stage_name).strip(), str(message).strip()


def _contains_cuda_oom(error_text: str) -> bool:
    """
    功能说明：判断错误文本是否包含 CUDA OOM 信号。
    """
    normalized = str(error_text).strip().lower()
    if not normalized:
        return False
    return ("out of memory" in normalized) or ("cuda out of memory" in normalized)
