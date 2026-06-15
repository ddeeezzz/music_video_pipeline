"""
文件用途：封装跨模块调度中的任务执行与完成回收逻辑。
核心流程：执行 B/C/D 单元、回收 Future、写入失败阻断状态。
输入输出：输入运行上下文与单元对象，输出执行结果或失败记录。
依赖说明：依赖模块 B/C/D 执行器与状态库。
维护说明：本模块不负责调度策略与并发窗口调参。
"""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import re
import time
from typing import Any

from music_video_pipeline.context import RuntimeContext
from music_video_pipeline.io_utils import read_json
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
from music_video_pipeline.modules.module_b.orchestrator import (
    run_module_b_role1,
    run_module_b_role2,
    run_module_b_role3,
    run_module_b_role4,
)
from music_video_pipeline.modules.module_c.executor import execute_one_unit_with_retry as execute_one_c_unit
from music_video_pipeline.modules.module_c.unit_models import ModuleCUnit
from music_video_pipeline.modules.module_d.executor import execute_one_unit_with_retry as execute_one_d_unit
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint, materialize_module_d_unit



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
                    failed_errors[failed_index] = f"B:{error_text or '模块B 批量执行失败'}"
                    context.state_store.mark_bcd_downstream_blocked(
                        task_id=context.task_id,
                        unit_index=failed_index,
                        from_module="B",
                        reason=f"upstream_blocked:B:{error_text or '模块B 批量执行失败'}",
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
    功能说明：执行跨模块链路模块 B 的 pipeline（role1→role4）。
    执行策略：
    1. 检查 role1/2/3 产物是否已存在，有则跳过
    2. role2 执行完后立即执行 role3（不管 role1）
    3. role1 + role3 都完成后执行 role4
    4. role4 在 as_completed 中按 segment 粒度为每个已完成 shot 写 artifact 并标记 SQL done
    5. 调度器在 tick 间读取 SQL 状态，自动派发下游 C 单元（流式）
    参数说明：
    - context: 运行上下文对象。
    - target_segment_ids: 待处理 segment ID 集合。
    - target_units: 模块 B 单元数组。
    返回值：
    - dict[str, Any]: 包含 failed_indexes 的执行摘要。
    """
    # ──────────────────────────────────────────────
    task_id = context.task_id
    logger = context.logger
    unit_outputs_dir = context.artifacts_dir / "module_b_units"
    unit_outputs_dir.mkdir(parents=True, exist_ok=True)

    _segment_shot_count_map: dict[str, int] = {}

    from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_role_result_path

    # 预加载 big_xxx ↔ seg_xxxx 映射（收尾逻辑需要）
    seg_to_big: dict[str, str] = {}
    big_to_segs: dict[str, list[str]] = {}
    try:
        ma = read_json(context.artifacts_dir / "module_a_output.json")
        for s in (ma.get("segments") or []):
            bid = str(s.get("big_segment_id", "")).strip()
            sid = str(s.get("segment_id", "")).strip()
            if bid and sid:
                seg_to_big[sid] = bid
                big_to_segs.setdefault(bid, []).append(sid)
    except Exception:
        pass

    # ── 检查 role1 是否已有产物 ──
    role1_path = get_module_b_role_result_path(context.artifacts_dir, "role1")
    role1_already_done = role1_path.exists() and role1_path.stat().st_size > 0
    if role1_already_done:
        logger.info("跨模块B role1 已有产出，跳过，task_id=%s", task_id)
        context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role1", status="done")

    # ── 检查 role2 是否已有产物 ──
    role2_path = get_module_b_role_result_path(context.artifacts_dir, "role2")
    role2_already_done = role2_path.exists() and role2_path.stat().st_size > 0
    if role2_already_done:
        logger.info("跨模块B role2 已有产出，跳过，task_id=%s", task_id)
        context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role2", status="done")

    role1_done = role1_already_done
    role2_done = role2_already_done
    role3_done = False
    role1_exc: Exception | None = None
    role2_exc: Exception | None = None
    role3_exc: Exception | None = None

    # ── 阶段 1：role1 + role2 并行执行（只跑需要跑的） ──
    futures_to_submit: dict[Future, str] = {}
    with ThreadPoolExecutor(max_workers=2) as r12_executor:
        if not role1_already_done:
            context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role1", status="running")
            f_role1 = r12_executor.submit(run_module_b_role1, context)
            futures_to_submit[f_role1] = "role1"
        if not role2_already_done:
            context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role2", status="running")
            f_role2 = r12_executor.submit(run_module_b_role2, context)
            futures_to_submit[f_role2] = "role2"

        if futures_to_submit:
            for fut in as_completed(futures_to_submit):
                role_name = futures_to_submit[fut]
                try:
                    fut.result()
                    if role_name == "role1":
                        role1_done = True
                        context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role1", status="done")
                        logger.info("跨模块B role1 完成，task_id=%s", task_id)
                    else:
                        role2_done = True
                        context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role2", status="done")
                        logger.info("跨模块B role2 完成，task_id=%s", task_id)
                except Exception as exc:
                    if role_name == "role1":
                        role1_exc = exc
                        context.state_store.set_module_unit_status(
                            task_id=task_id, module_name="B", unit_id="role1", status="failed", error_message=str(exc),
                        )
                        logger.error("跨模块B role1 失败，task_id=%s，错误=%s", task_id, exc)
                    else:
                        role2_exc = exc
                        context.state_store.set_module_unit_status(
                            task_id=task_id, module_name="B", unit_id="role2", status="failed", error_message=str(exc),
                        )
                        logger.error("跨模块B role2 失败，task_id=%s，错误=%s", task_id, exc)

    # role1 失败 → 阻断
    if role1_exc is not None:
        logger.error("跨模块B role1 失败，跳过后续，task_id=%s", task_id)
        _mark_batch_failed(context, target_units, f"role1 失败：{role1_exc}")
        return {"failed_indexes": [u.unit_index for u in target_units], "error": str(role1_exc)}

    # role2 完成（或已有产物）→ 启动 role3；role3 每完成一个 big_segment streaming
    # 文件，立即触发 role4 处理该大段下的 seg_xxxx
    if role2_done:
        role3_path = get_module_b_role_result_path(context.artifacts_dir, "role3")
        role3_already_done = role3_path.exists() and role3_path.stat().st_size > 0
        role3_streaming_dir = context.artifacts_dir / "module_b_work" / "role3" / "streaming"
        role3_streaming_dir.mkdir(parents=True, exist_ok=True)
        seen_big_files: set[str] = set()

        if role3_already_done:
            logger.info("跨模块B role3 已有产出，跳过，task_id=%s", task_id)
            context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role3", status="done")
            role3_done = True
            # role3 已有产物 → 检查已有 streaming 文件并触发 role4
            role3_streaming_dir = context.artifacts_dir / "module_b_work" / "role3" / "streaming"
            seen_big_files: set[str] = set()
            for fp in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
                bid = fp.stem.replace("role3_segment_output.streaming.", "")
                if bid not in seen_big_files and fp.stat().st_size > 0:
                    seen_big_files.add(bid)
                    _run_role4_for_big_segment_shots(
                        context=context, big_segment_id=bid,
                        unit_outputs_dir=unit_outputs_dir,
                        role3_streaming_dir=role3_streaming_dir,
                    )
        else:
            logger.info("跨模块B role2 完成，启动 role3（流式触发 role4），task_id=%s", task_id)
            context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role3", status="running")

            # 提交 role3 到后台线程
            role3_thread_pool = ThreadPoolExecutor(max_workers=2)
            f_role3 = role3_thread_pool.submit(run_module_b_role3, context)
            role3_thread_pool.shutdown(wait=False)

            # 轮询：role3 streaming 新文件 → 提交 role4
            try:
                while True:
                    if f_role3.done():
                        f_role3.result()
                        role3_done = True
                        context.state_store.set_module_unit_status(task_id=task_id, module_name="B", unit_id="role3", status="done")
                        logger.info("跨模块B role3 完成，task_id=%s", task_id)
                        # 处理可能遗漏的 big_segment
                        for fp in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
                            bid = fp.stem.replace("role3_segment_output.streaming.", "")
                            if bid not in seen_big_files:
                                seen_big_files.add(bid)
                                _run_role4_for_big_segment_shots(
                                    context=context, big_segment_id=bid,
                                    unit_outputs_dir=unit_outputs_dir,
                                    role3_streaming_dir=role3_streaming_dir,
                                )
                        break

                    # 检查新 streaming 文件
                    for fp in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
                        bid = fp.stem.replace("role3_segment_output.streaming.", "")
                        if bid not in seen_big_files and fp.stat().st_size > 0:
                            seen_big_files.add(bid)
                            logger.info("跨模块B role3 流式文件就绪，触发 role4：%s，task_id=%s", bid, task_id)
                            _run_role4_for_big_segment_shots(
                                context=context, big_segment_id=bid,
                                unit_outputs_dir=unit_outputs_dir,
                                role3_streaming_dir=role3_streaming_dir,
                            )

                    time.sleep(0.5)
            except Exception as exc:
                role3_exc = exc
                context.state_store.set_module_unit_status(
                    task_id=task_id, module_name="B", unit_id="role3", status="failed", error_message=str(exc),
                )
                logger.error("跨模块B role3 失败，task_id=%s，错误=%s", task_id, exc)

    if role2_exc is not None or role3_exc is not None:
        reason = role3_exc or role2_exc
        logger.error("跨模块B role2/role3 失败，跳过后续，task_id=%s", task_id)
        _mark_batch_failed(context, target_units, f"role2/role3 失败：{reason}")
        return {"failed_indexes": [u.unit_index for u in target_units], "error": str(reason)}

    if not role1_done or not role3_done:
        logger.error("跨模块B role1 或 role3 未完成，跳过后续，task_id=%s", task_id)
        _mark_batch_failed(context, target_units, "role1 或 role3 未完成")
        return {"failed_indexes": [u.unit_index for u in target_units], "error": "role1 或 role3 未完成"}

    # ── 收尾：按 big_segment 聚合 shot 完成状态 ──
    # 从 module_a_output 构建 big_xxx → [seg_xxxx] 和 seg_xxxx → big_xxx 映射
    big_to_segs: dict[str, list[str]] = {}
    seg_to_big: dict[str, str] = {}
    try:
        module_a_path = context.artifacts_dir / "module_a_output.json"
        if module_a_path.exists():
            module_a_output = read_json(module_a_path)
            for seg in (module_a_output.get("segments") or []):
                big_id = str(seg.get("big_segment_id", "")).strip()
                seg_id = str(seg.get("segment_id", "")).strip()
                if big_id and seg_id:
                    big_to_segs.setdefault(big_id, []).append(seg_id)
                    seg_to_big[seg_id] = big_id
    except Exception:  # noqa: BLE001
        logger.warning("读取 module_a_output.json 失败，按单个 seg 聚合")

    # 查询 DB 中所有 seg_xxxx 的状态
    all_b_rows = context.state_store.list_module_units(task_id=task_id, module_name="B") or []
    seg_status: dict[str, str] = {}
    for row in all_b_rows:
        uid = str(row.get("unit_id", "")).strip()
        if uid.startswith("seg_"):
            seg_status[uid] = str(row.get("status", "pending")).lower()

    # 检查每个 big_xxx 下是否有 seg 完成。
    failed_indexes: list[int] = []
    for unit in target_units:
        big_id = seg_to_big.get(unit.unit_id) or unit.unit_id
        child_segs = big_to_segs.get(big_id, [])
        if not child_segs:
            # 没有 module_a_output 映射时，直接用 unit.unit_id 查 DB
            if seg_status.get(unit.unit_id) != "done":
                logger.error("跨模块B %s 无 seg 信息，标记失败，task_id=%s", big_id, task_id)
                failed_indexes.append(int(unit.unit_index))
            continue
        any_done = any(seg_status.get(sid) == "done" for sid in child_segs)
        if not any_done:
            logger.error("跨模块B %s 无任何 seg 完成，标记失败，task_id=%s", big_id, task_id)
            seg_indexes: list[int] = []
            for sid in child_segs:
                for row in all_b_rows:
                    if str(row.get("unit_id", "")).strip() == sid:
                        seg_indexes.append(int(row.get("unit_index", -1)))
                        break
            failed_indexes.extend(seg_indexes or [int(unit.unit_index)])

    result: dict[str, Any] = {"failed_indexes": failed_indexes}
    if failed_indexes:
        result["error"] = f"模块B {len(failed_indexes)} 个 big_segment 失败"
    logger.info("跨模块B batch 完成，task_id=%s，total=%s，failed=%s", task_id, len(target_units), len(failed_indexes))
    return result


def _run_role4_for_big_segment_shots(
    context: RuntimeContext,
    big_segment_id: str,
    unit_outputs_dir: Path,
    role3_streaming_dir: Path,
) -> None:
    """读取 role3 流式文件，为指定 big_segment 的 seg_xxxx 执行 role4 LLM。"""
    from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_role_result_path, get_module_b_streaming_dir
    from music_video_pipeline.modules.module_b.markdown_contracts import parse_shot_plans, ShotPlan
    from music_video_pipeline.modules.module_b.orchestrator import (
        _parse_visual_registry,
        _parse_remotion_catalog,
        _parse_subject_descriptions,
        _run_role4_llm_shot,
        _build_shot_id,
        _build_segment_b_artifact_json,
        _resolve_project_root,
        _resolve_storyboard_template_path,
    )

    task_id = context.task_id
    logger = context.logger

    # 读 role1
    role1_output_path = get_module_b_role_result_path(context.artifacts_dir, "role1")
    role1_streaming_path = role1_output_path.parent / "streaming" / f"{role1_output_path.stem}.streaming.md"
    role1_source = role1_streaming_path if role1_streaming_path.exists() else role1_output_path
    if not role1_source.exists():
        logger.warning("role4 per-big 跳过 %s：缺少 role1 产物", big_segment_id)
        return
    visual_registry = _parse_visual_registry(role1_source.read_text(encoding="utf-8"))

    # 读 remotion catalog
    project_root = _resolve_project_root()
    template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    remotion_catalog = _parse_remotion_catalog(template_path.read_text(encoding="utf-8"))

    # 读 role3 streaming 文件 → shot_plans
    fp = role3_streaming_dir / f"role3_segment_output.streaming.{big_segment_id}.md"
    if not fp.exists() or fp.stat().st_size == 0:
        logger.warning("role4 per-big 跳过 %s：无 role3 streaming 文件", big_segment_id)
        return
    shot_plans: list[ShotPlan] = []
    try:
        content = fp.read_text(encoding="utf-8").strip()
        if content:
            shot_plans = parse_shot_plans(content)
    except Exception as exc:
        logger.warning("role4 per-big 解析 %s 失败：%s", big_segment_id, exc)
        return
    if not shot_plans:
        logger.warning("role4 per-big %s 无 shot_plans", big_segment_id)
        return

    # 构建 shot_tasks
    comfyui_cfg = context.config.module_c.comfyui if hasattr(context.config, "module_c") else None
    prompt_prefix = str(getattr(comfyui_cfg, "prompt_prefix", "")).strip() if comfyui_cfg else ""
    prompt_suffix = str(getattr(comfyui_cfg, "prompt_suffix", "")).strip() if comfyui_cfg else ""

    shot_tasks: list[dict] = []
    for sp in shot_plans:
        seg_id = str(sp.segment_id).strip()
        if not seg_id:
            continue
        remotion_id = str(sp.remotion_id).strip()
        scene_desc = str(sp.scene_desc_zh).strip()
        subjects = _parse_subject_descriptions(scene_desc, remotion_id)
        for subj_idx, subj_desc in enumerate(subjects, start=1):
            shot_tasks.append({"sp": sp, "subj_idx": subj_idx, "subject_desc": subj_desc})

    if not shot_tasks:
        logger.warning("role4 per-big %s 无 shot tasks", big_segment_id)
        return

    # role4 streaming 路径
    role4_streaming_dir = get_module_b_streaming_dir(context.artifacts_dir, "role4")
    role4_streaming_dir.mkdir(parents=True, exist_ok=True)

    # 统计每个 seg 的 shot 计数
    seg_shot_counts: dict[str, int] = {}
    for t in shot_tasks:
        sid = str(t["sp"].segment_id).strip()
        if sid:
            seg_shot_counts[sid] = seg_shot_counts.get(sid, 0) + 1
    seg_done_counts: dict[str, int] = {}
    output_parts_map: dict[int, str] = {}

    # 检测已有 role4 产物：已有 streaming 的 shot 直接跳 LLM
    filtered_tasks: list[dict] = []
    for _ti, task in enumerate(shot_tasks):
        seg_id = str(task["sp"].segment_id).strip()
        shot_id = _build_shot_id(seg_id, task["subj_idx"])
        stream_path = role4_streaming_dir / f"role4_prompt_output.streaming.{shot_id}.md"
        if stream_path.exists() and stream_path.stat().st_size > 0:
            logger.info("跨模块B role4 跳过已有 shot：%s，task_id=%s", shot_id, task_id)
            output_parts_map[_ti] = stream_path.read_text(encoding="utf-8")
            seg_done_counts[seg_id] = seg_done_counts.get(seg_id, 0) + 1
            if seg_done_counts[seg_id] >= seg_shot_counts.get(seg_id, 0):
                if unit_outputs_dir is not None:
                    _build_segment_b_artifact_json(
                        unit_outputs_dir=unit_outputs_dir,
                        seg_id=seg_id,
                        shot_tasks=shot_tasks,
                        output_parts_map=output_parts_map,
                        shot_plans=shot_plans,
                        segment_shot_count_map={},
                    )
                _ap = str(unit_outputs_dir / f"{seg_id}.json") if unit_outputs_dir is not None else ""
                context.state_store.set_module_unit_status(
                    task_id=task_id, module_name="B", unit_id=seg_id, status="done",
                    artifact_path=_ap,
                )
                logger.info("跨模块B role4 seg 已有全部产物：%s（ap=%s），task_id=%s", seg_id, _ap, task_id)
                _try_heal_b_module_with_rebuild(context=context, task_id=task_id, seg_id=seg_id, unit_outputs_dir=unit_outputs_dir)
        else:
            filtered_tasks.append(task)

    if not filtered_tasks:
        logger.info("跨模块B role4 %s 所有 shot 已有产物，跳过 LLM，task_id=%s", big_segment_id, task_id)
        # 从 streaming 文件填充 output_parts_map，确保 artifact JSON 可构建
        for _sid, _total in seg_shot_counts.items():
            for _ti, _tk in enumerate(shot_tasks):
                if str(_tk["sp"].segment_id).strip() != _sid:
                    continue
                _shot_id = _build_shot_id(_sid, _tk["subj_idx"])
                _sp = role4_streaming_dir / f"role4_prompt_output.streaming.{_shot_id}.md"
                if _sp.exists() and _sp.stat().st_size > 0:
                    output_parts_map[_ti] = _sp.read_text(encoding="utf-8")
        for seg_id, total in seg_shot_counts.items():
            if seg_done_counts.get(seg_id, 0) >= total:
                if unit_outputs_dir is not None:
                    _build_segment_b_artifact_json(
                        unit_outputs_dir=unit_outputs_dir, seg_id=seg_id,
                        shot_tasks=shot_tasks,
                        output_parts_map=output_parts_map,
                        shot_plans=shot_plans,
                        segment_shot_count_map={},
                    )
                _ap = str(unit_outputs_dir / f"{seg_id}.json") if unit_outputs_dir is not None else ""
                context.state_store.set_module_unit_status(
                    task_id=task_id, module_name="B", unit_id=seg_id, status="done",
                    artifact_path=_ap,
                )
                logger.info("跨模块B role4 seg 写 DB 确认：%s = done（ap=%s），task_id=%s", seg_id, _ap, task_id)
                _try_heal_b_module_with_rebuild(context=context, task_id=task_id, seg_id=seg_id, unit_outputs_dir=unit_outputs_dir)
        return

    # 执行 LLM（只跑没有产出的 shot）
    failed_count = 0
    with ThreadPoolExecutor(max_workers=min(len(filtered_tasks), 4)) as executor:
        future_to_idx: dict[Future, int] = {}
        for task in filtered_tasks:
            _st_idx = shot_tasks.index(task)  # 找到在 shot_tasks 中的原始索引
            future = executor.submit(
                _run_role4_llm_shot,
                context=context,
                project_root=project_root,
                remotion_catalog=remotion_catalog,
                visual_registry=visual_registry,
                sp=task["sp"],
                subj_idx=task["subj_idx"],
                subject_desc=task["subject_desc"],
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
            )
            future_to_idx[future] = _st_idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                output_parts_map[idx] = result
                seg_id = str(shot_tasks[idx]["sp"].segment_id).strip()
                if seg_id:
                    seg_done_counts[seg_id] = seg_done_counts.get(seg_id, 0) + 1
                    if seg_done_counts[seg_id] >= seg_shot_counts.get(seg_id, 0):
                        ctx = context
                        # 写 artifact JSON
                        if unit_outputs_dir is not None:
                            _build_segment_b_artifact_json(
                                unit_outputs_dir=unit_outputs_dir,
                                seg_id=seg_id,
                                shot_tasks=shot_tasks,
                                output_parts_map=output_parts_map,
                                shot_plans=shot_plans,
                                segment_shot_count_map={},
                            )
                        _art_path = str(unit_outputs_dir / f"{seg_id}.json") if unit_outputs_dir is not None else ""
                        ctx.state_store.set_module_unit_status(
                            task_id=task_id, module_name="B", unit_id=seg_id, status="done",
                            artifact_path=_art_path,
                        )
                        logger.info("跨模块B role4 seg 完成：%s，task_id=%s", seg_id, task_id)
                        _try_heal_b_module_with_rebuild(context=ctx, task_id=task_id, seg_id=seg_id, unit_outputs_dir=unit_outputs_dir)
            except Exception as exc:
                seg_id = str(shot_tasks[idx]["sp"].segment_id).strip() if idx < len(shot_tasks) else "?"
                logger.error("跨模块B role4 shot 失败 shot=%s：%s", seg_id, exc)
                failed_count += 1

    if failed_count:
        logger.warning("跨模块B role4 per-big %s 有 %s shot 失败", big_segment_id, failed_count)


def _mark_batch_failed(context: RuntimeContext, target_units: list[ModuleBUnit], reason: str) -> None:
    """将 target_units 全部标记为 failed。"""
    for unit in target_units:
        context.state_store.set_module_unit_status(
            task_id=context.task_id, module_name="B", unit_id=unit.unit_id,
            status="failed", error_message=reason,
        )


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
    # 从 B 产物中读取正确 shot_id，而非使用链路占位符
    artifact_shot_id = str(shot_obj.get("shot_id", "")).strip()
    if not artifact_shot_id:
        raise RuntimeError(
            f"跨模块调度失败：模块B单元产物缺少 shot_id，segment_id={chain_unit.segment_id}"
        )
    shot_obj["shot_id"] = artifact_shot_id
    if "start_time" not in shot_obj:
        shot_obj["start_time"] = float(c_row.get("start_time", chain_unit.start_time))
    if "end_time" not in shot_obj:
        shot_obj["end_time"] = float(c_row.get("end_time", chain_unit.end_time))

    unit = ModuleCUnit(
        unit_id=artifact_shot_id,
        unit_index=chain_unit.unit_index,
        segment_id=chain_unit.segment_id,
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


def _load_strict_dual_frame_item_for_d(
    *,
    blueprint: ModuleDUnitBlueprint,
    c_row: dict[str, Any],
) -> dict[str, Any]:
    """
    功能说明：从模块C SQL 状态 + shot_id 命名规范推导双关键帧，构建模块D输入 frame_item。
    参数说明：
    - blueprint: 模块D单元蓝图。
    - c_row: 状态库中的模块C单元记录。
    返回值：
    - dict[str, Any]: 可直接用于 materialize_module_d_unit 的 frame_item。
    异常说明：
    - RuntimeError: 帧文件缺失或关键帧字段不完整时抛出。
    边界条件：不再接受仅 frame_path 的单帧输入。
    """
    unit_id = str(c_row.get("unit_id", "")).strip() or str(blueprint.unit_id)
    artifact_path = str(c_row.get("artifact_path", "")).strip()
    if not artifact_path:
        raise RuntimeError(
            "跨模块调度失败：模块C单元产物路径为空，"
            f"unit_id={unit_id}。"
        )
    frames_dir = Path(artifact_path).parent
    frame_path_start = str(frames_dir / f"{unit_id}_start.png")
    frame_path_end = str(frames_dir / f"{unit_id}_end.png")

    if not Path(frame_path_start).exists():
        raise RuntimeError(
            "跨模块调度失败：模块C首帧文件缺失，"
            f"unit_id={unit_id}，frame_path_start={frame_path_start}。"
        )
    if not Path(frame_path_end).exists():
        raise RuntimeError(
            "跨模块调度失败：模块C尾帧文件缺失，"
            f"unit_id={unit_id}，frame_path_end={frame_path_end}。"
        )

    return {
        "shot_id": str(blueprint.unit_id),
        "frame_path": frame_path_start,
        "frame_path_start": frame_path_start,
        "frame_path_end": frame_path_end,
        "control_frame_paths": [frame_path_start, frame_path_end],
        "start_time": float(c_row.get("start_time", blueprint.start_time)),
        "end_time": float(c_row.get("end_time", blueprint.end_time)),
        "duration": float(c_row.get("duration", blueprint.duration)),
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

    # 从模块 B artifact 加载 lyric_units，用于 Remotion 字幕渲染
    seg_id = _extract_segment_id_from_unit_id(blueprint.unit_id)
    if seg_id:
        b_artifact_path = context.artifacts_dir / "module_b_units" / f"{seg_id}.json"
        try:
            if b_artifact_path.exists():
                b_data = read_json(b_artifact_path)
                if isinstance(b_data, dict):
                    lyrics = b_data.get("lyric_units")
                    if lyrics:
                        frame_item["lyric_units"] = lyrics
        except Exception:
            pass

    unit = materialize_module_d_unit(blueprint=blueprint, frame_item=frame_item)
    segment_path = execute_one_d_unit(
        context=context,
        unit=unit,
        profile=profile,
        device_override=device_override,
    )
    return str(segment_path)


def _extract_segment_id_from_unit_id(unit_id: str) -> str:
    """从 unit_id（shot_0019_1 或 seg_0019）中提取 segment_id（seg_0019）。"""
    unit_id = str(unit_id).strip()
    if unit_id.startswith("seg_"):
        return unit_id
    m = re.search(r"shot_(\d+)_", unit_id)
    if m:
        return f"seg_{m.group(1)}"
    return ""
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


def _try_heal_b_module_with_rebuild(
    context: RuntimeContext,
    task_id: str,
    seg_id: str,
    unit_outputs_dir: Path | None,
) -> None:
    """检查全部 seg 是否 done → 自愈 B module + 重建 B 输出。"""
    all_b_rows = context.state_store.list_module_units(task_id=task_id, module_name="B") or []
    seg_rows = [r for r in all_b_rows if str(r.get("unit_id", "")).startswith("seg_")]
    if not seg_rows:
        return
    all_done = all(str(r.get("status", "")).lower() == "done" for r in seg_rows)
    if all_done:
        from music_video_pipeline.modules.module_b.output_builder import build_module_b_output
        from music_video_pipeline.io_utils import read_json, write_json
        module_a_path = context.artifacts_dir / "module_a_output.json"
        module_a_output = read_json(module_a_path) if module_a_path.exists() else {}
        done_unit_records = context.state_store.list_module_units_by_status(
            task_id=task_id, module_name="B", statuses=["done"],
        )
        output = build_module_b_output(
            done_unit_records=list(done_unit_records or []),
            module_a_output=module_a_output, instrumental_labels=[],
            artifacts_dir=context.artifacts_dir,
        )
        write_json(context.artifacts_dir / "module_b_output.json", output)
        context.state_store.set_module_status(
            task_id=task_id, module_name="B", status="done",
            artifact_path=str(context.artifacts_dir / "module_b_output.json"),
        )
        logger = logging.getLogger(__name__)
        logger.info("模块B 自愈为 done + 重建输出完成（%s 最后完成），task_id=%s", seg_id, task_id)
