"""
文件用途：验证跨模块 B/C/D 依赖链核心语义。
核心流程：构造链路单元与状态库，打桩单元执行函数，断言依赖链顺序。
输入输出：输入临时任务环境，输出调度结果断言。
依赖说明：依赖 pytest 与项目内 cross_bcd.scheduler_tasks / scheduler_engine。
维护说明：本测试聚焦 B→C→D 依赖链，不测试 GPU/自适应窗口等策略逻辑。
"""

# 标准库：用于日志构建
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于 JSON 写 module_a_output
import json

# 项目内模块：配置数据类
from music_video_pipeline.config import (
    AppConfig,
    CrossModuleAdaptiveWindowConfig,
    CrossModuleConfig,
    FfmpegConfig,
    LoggingConfig,
    ModuleAConfig,
    ModuleBConfig,
    ModuleCConfig,
    ModuleDConfig,
    PathsConfig,
)
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：跨模块链路模型
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
# 项目内模块：跨模块调度器
from music_video_pipeline.modules.cross_bcd import scheduler
from music_video_pipeline.modules.cross_bcd import scheduler_adaptive, scheduler_allocators, scheduler_tasks
from music_video_pipeline.modules.cross_bcd import scheduler_engine
# 项目内模块：模块 B 单元模型
from music_video_pipeline.modules.module_b.unit_models import ModuleBUnit
# 项目内模块：模块 D 单元蓝图
from music_video_pipeline.modules.module_d.unit_models import ModuleDUnitBlueprint
# 项目内模块：状态库
from music_video_pipeline.state_store import StateStore


def test_b_batch_should_run_role1_thru_role4_in_order(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 B 批量执行按 role1→role2→role3→role4 顺序推进，
    role4 标记每个 seg_xxxx 为 done，收尾按 big_xxx 正确聚合。
    """
    context, chain_units, b_units_map, _, _ = _build_fixture(tmp_path=tmp_path, task_id="chain_b_order")
    call_log: list[str] = []

    def _fake_role1(ctx):
        call_log.append("role1")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role1", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role1" / "role1_visual_output.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
        return path

    def _fake_role2(ctx):
        call_log.append("role2")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role2", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role2" / "role2_story_output.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
        return path

    _fake_role3 = _make_fake_role3_with_streaming(call_log, chain_units)

    def _fake_role4(ctx, *, unit_outputs_dir=None, segment_shot_count_map=None):
        call_log.append("role4")
        for cu in chain_units:
            ctx.state_store.set_module_unit_status(
                task_id=ctx.task_id, module_name="B", unit_id=cu.segment_id, status="done",
            )
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role4" / "role4_prompt_output.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
        return path

    monkeypatch.setattr(scheduler_tasks, "run_module_b_role1", _fake_role1)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role2", _fake_role2)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role3", _fake_role3)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role4", _fake_role4)
    monkeypatch.setattr(scheduler_tasks, "_run_role4_for_big_segment_shots", _make_stub_role4_for_big(chain_units))

    target_units = list(b_units_map.values())
    result = _run_b_chain_batch_for_test(context=context, target_units=target_units)

    assert call_log.index("role1") < call_log.index("role3")
    assert call_log.index("role2") < call_log.index("role3")
    assert result["failed_indexes"] == [], f"应有 0 失败，实际={result['failed_indexes']}"
    for role in ("role1", "role2", "role3", "role4"):
        row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="B", unit_id=role)
        assert row is not None and row["status"] == "done", f"B {role} 应为 done"


def test_b_batch_should_skip_existing_role_products(tmp_path: Path, monkeypatch) -> None:
    """验证 B 批量执行时 role1/2/3 已有产物可跳过。"""
    context, chain_units, b_units_map, _, _ = _build_fixture(tmp_path=tmp_path, task_id="chain_skip_existing")
    call_log: list[str] = []
    from music_video_pipeline.modules.module_b.artifact_paths import get_module_b_role_result_path
    for role in ("role1", "role2", "role3"):
        path = get_module_b_role_result_path(context.artifacts_dir, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
    # 预写 role3 streaming 文件（否则 role4 无法获取 seg 信息）
    sdir = context.artifacts_dir / "module_b_work" / "role3" / "streaming"
    sdir.mkdir(parents=True, exist_ok=True)
    big_ids_seen: set[str] = set()
    for idx, cu in enumerate(chain_units):
        bid = f"big_{idx + 1:03d}"
        if bid not in big_ids_seen:
            big_ids_seen.add(bid)
            (sdir / f"role3_segment_output.streaming.{bid}.md").write_text(
                f"## {bid}\n- segment_ids: {cu.segment_id}\n", encoding="utf-8",
            )

    def _fake_role1(ctx):
        call_log.append("role1")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role1", status="done")
        return ctx.artifacts_dir / "module_b_work" / "role1" / "role1_visual_output.md"

    def _fake_role2(ctx):
        call_log.append("role2")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role2", status="done")
        return ctx.artifacts_dir / "module_b_work" / "role2" / "role2_story_output.md"

    def _fake_role3(ctx):
        call_log.append("role3")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role3", status="done")
        return ctx.artifacts_dir / "module_b_work" / "role3" / "role3_shot_output.md"

    def _fake_role4(ctx, *, unit_outputs_dir=None, segment_shot_count_map=None):
        call_log.append("role4")
        for cu in chain_units:
            ctx.state_store.set_module_unit_status(
                task_id=ctx.task_id, module_name="B", unit_id=cu.segment_id, status="done",
            )
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role4" / "role4_prompt_output.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
        return path

    monkeypatch.setattr(scheduler_tasks, "run_module_b_role1", _fake_role1)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role2", _fake_role2)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role3", _fake_role3)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role4", _fake_role4)
    monkeypatch.setattr(scheduler_tasks, "_run_role4_for_big_segment_shots", _make_stub_role4_for_big(chain_units))

    target_units = list(b_units_map.values())
    result = _run_b_chain_batch_for_test(context=context, target_units=target_units)

    assert result["failed_indexes"] == [], f"应有 0 失败，实际={result['failed_indexes']}"
    assert "role1" not in call_log
    assert "role2" not in call_log
    assert "role3" not in call_log
    # role4 通过 _run_role4_for_big_segment_shots 触发，不在 call_log 中


def test_b_batch_should_report_big_segment_failure_by_child_segments(tmp_path: Path, monkeypatch) -> None:
    """验证 big_segment 失败判定根据子 seg 的实际完成情况。"""
    context, chain_units, b_units_map, _, _ = _build_fixture(tmp_path=tmp_path, task_id="chain_big_seg_fail")

    def _fake_role1(ctx):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role1", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role1" / "role1_visual_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    def _fake_role2(ctx):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role2", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role2" / "role2_story_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    _fake_role3 = _make_fake_role3_with_streaming(None, chain_units)

    def _fake_role4(ctx, *, unit_outputs_dir=None, segment_shot_count_map=None):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0001", status="done")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0002", status="failed", error_message="shot_0002 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="shot_0003 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0004", status="failed", error_message="shot_0004 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role4" / "role4_prompt_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    monkeypatch.setattr(scheduler_tasks, "run_module_b_role1", _fake_role1)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role2", _fake_role2)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role3", _fake_role3)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role4", _fake_role4)
    def _partial_role4_stub(context, *, big_segment_id, unit_outputs_dir, role3_streaming_dir):
        del big_segment_id, unit_outputs_dir, role3_streaming_dir
        # 模拟 test 期望：seg_0001 done, 其余 failed
        ctx = context
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0001", status="done")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0002", status="failed", error_message="shot_0002 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="shot_0003 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0004", status="failed", error_message="shot_0004 failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
    monkeypatch.setattr(scheduler_tasks, "_run_role4_for_big_segment_shots", _partial_role4_stub)

    target_units = [b_units_map["seg_0001"], b_units_map["seg_0002"],
                    b_units_map["seg_0003"], b_units_map["seg_0004"]]
    result = _run_b_chain_batch_for_test(context=context, target_units=target_units)

    fi = result.get("failed_indexes", [])
    assert not any(idx in (0, 1) for idx in fi), f"big_001 不应标记失败，failed_indexes={fi}"
    assert any(idx in (2, 3) for idx in fi), f"big_002 应标记失败，failed_indexes={fi}"


def test_scheduler_should_dispatch_c_and_d_in_wavefront(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证波前调度器能在 role4 逐段完成后触发 C，C 完成后触发 D。
    B_BATCH 提交后，调度器在 tick 中读到 b 状态变化，自动派发 C/D。
    """
    context, chain_units, b_units_map, d_blueprints_map, _ = _build_fixture(
        tmp_path=tmp_path, task_id="chain_wavefront", chain_count=4,
    )
    events: list[tuple[str, int, str]] = []  # (stage, unit_index, unit_id)
    stub_role1_called = False

    monkeypatch.setattr(scheduler_engine, "build_keyframe_generator", lambda mode, logger, app_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    # ── B_BATCH 打桩：role1~4，内部逐步标记 seg done ──
    def _fake_role1(ctx):
        nonlocal stub_role1_called
        stub_role1_called = True
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role1", status="done")
        events.append(("B_role1", -1, "role1"))
        path = ctx.artifacts_dir / "module_b_work" / "role1" / "role1_visual_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    def _fake_role2(ctx):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role2", status="done")
        events.append(("B_role2", -1, "role2"))
        path = ctx.artifacts_dir / "module_b_work" / "role2" / "role2_story_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    _fake_role3 = _make_fake_role3_with_streaming(None, chain_units)
    events.append(("B_role3", -1, "role3"))

    def _fake_role4(ctx, *, unit_outputs_dir=None, segment_shot_count_map=None):
        events.append(("B_role4", -1, "role4"))
        import time
        # 逐个标记 seg 完成，间隔时间让调度器 tick 能读到状态变化
        for cu in chain_units:
            time.sleep(0.03)
            ctx.state_store.set_module_unit_status(
                task_id=ctx.task_id, module_name="B", unit_id=cu.segment_id, status="done",
            )
            events.append(("B_seg_done", int(cu.unit_index), cu.segment_id))
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role4" / "role4_prompt_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    monkeypatch.setattr(scheduler_tasks, "run_module_b_role1", _fake_role1)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role2", _fake_role2)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role3", _fake_role3)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role4", _fake_role4)
    monkeypatch.setattr(scheduler_tasks, "_run_role4_for_big_segment_shots", _make_stub_role4_for_big(chain_units))

    # ── C 打桩 ──
    def _fake_run_c(ctx, chain_unit, c_row, generator, frames_dir):
        b_row = ctx.state_store.get_module_unit_record(
            task_id=ctx.task_id, module_name="B", unit_id=chain_unit.segment_id,
        )
        b_status = str((b_row or {}).get("status", ""))
        assert b_status == "done", f"C 启动时 B({chain_unit.segment_id}) 应为 done，实际={b_status}"
        import time
        time.sleep(0.02)
        ctx.state_store.set_module_unit_status(
            task_id=ctx.task_id, module_name="C", unit_id=chain_unit.shot_id, status="done",
        )
        events.append(("C_done", int(chain_unit.unit_index), chain_unit.shot_id))
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    # ── D 打桩 ──
    def _fake_run_d(ctx, blueprint, c_row, profile, device_override=None):
        c_row_check = ctx.state_store.get_module_unit_record(
            task_id=ctx.task_id, module_name="C", unit_id=blueprint.unit_id,
        )
        c_status = str((c_row_check or {}).get("status", ""))
        assert c_status == "done", f"D 启动时 C({blueprint.unit_id}) 应为 done，实际={c_status}"
        ctx.state_store.set_module_unit_status(
            task_id=ctx.task_id, module_name="D", unit_id=blueprint.unit_id, status="done",
        )
        events.append(("D_done", int(blueprint.unit_index), blueprint.unit_id))
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output=_build_module_a_output(),
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    assert result["failed_chain_indexes"] == [], f"应有 0 失败，实际={result['failed_chain_indexes']}"

    # 验证 C/D 全部完成
    c_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="C")
    assert c_summary["status_counts"]["done"] == len(chain_units), "C 应全部 done"
    d_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")
    assert d_summary["status_counts"]["done"] == len(chain_units), "D 应全部 done"

    # 验证 B_seg_done 先于 C_done 先于 D_done
    idx_c = {e[2] for e in events if e[0] == "C_done"}
    idx_d = {e[2] for e in events if e[0] == "D_done"}
    assert idx_c == idx_d, "C 和 D 应覆盖相同 shot_id"

    for evt in events:
        if evt[0] == "C_done":
            seg_id = f"seg_{evt[1] + 1:04d}"
            b_idx = next((i for i, e in enumerate(events) if e[2] == seg_id and e[0] == "B_seg_done"), -1)
            c_idx = next((i for i, e in enumerate(events) if e[2] == evt[2] and e[0] == "C_done"), -1)
            d_idx = next((i for i, e in enumerate(events) if e[2] == evt[2] and e[0] == "D_done"), -1)
            assert b_idx < c_idx, f"C_done({evt[2]}) 应晚于 B_seg_done"
            assert c_idx < d_idx, f"D_done({evt[2]}) 应晚于 C_done"


def test_scheduler_should_handle_partial_b_failure_in_wavefront(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证某 big_segment 的 seg 全部失败后，其余正常链路仍能完成 C/D。
    big_001 的 seg_0001/0002 完成 → C/D 正常
    big_002 的 seg_0003/0004 全部失败 → C/D 跳过
    """
    context, chain_units, b_units_map, d_blueprints_map, _ = _build_fixture(
        tmp_path=tmp_path, task_id="chain_partial_fail", chain_count=4,
    )
    monkeypatch.setattr(scheduler_engine, "build_keyframe_generator", lambda mode, logger, app_config=None: object())
    monkeypatch.setattr(scheduler_engine, "resolve_render_profile", lambda context: {"name": "cpu"})

    def _fake_role1(ctx):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role1", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role1" / "role1_visual_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    def _fake_role2(ctx):
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role2", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role2" / "role2_story_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    _fake_role3 = _make_fake_role3_with_streaming(None, chain_units)

    def _fake_role4(ctx, *, unit_outputs_dir=None, segment_shot_count_map=None):
        # big_001: seg_0001 done, seg_0002 done
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0001", status="done")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0002", status="done")
        # big_002: seg_0003 failed, seg_0004 failed
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="shot failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0004", status="failed", error_message="shot failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
        path = ctx.artifacts_dir / "module_b_work" / "role4" / "role4_prompt_output.md"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake\n", encoding="utf-8")
        return path

    monkeypatch.setattr(scheduler_tasks, "run_module_b_role1", _fake_role1)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role2", _fake_role2)
    monkeypatch.setattr(scheduler_tasks, "run_module_b_role3", _fake_role3)
    def _partial_stub(context, *, big_segment_id, unit_outputs_dir, role3_streaming_dir):
        import logging as _lg
        _lg.warning("_partial_stub CALLED big_segment_id=%s", big_segment_id)
        ctx = context
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0001", status="done")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0002", status="done")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0003", status="failed", error_message="shot failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="seg_0004", status="failed", error_message="shot failed")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role4", status="done")
    monkeypatch.setattr(scheduler_tasks, "_run_role4_for_big_segment_shots", _partial_stub)

    def _fake_run_c(ctx, chain_unit, c_row, generator, frames_dir):
        ctx.state_store.set_module_unit_status(
            task_id=ctx.task_id, module_name="C", unit_id=chain_unit.shot_id, status="done",
        )
        return {"frame_path": str(frames_dir / f"{chain_unit.shot_id}.png")}

    def _fake_run_d(ctx, blueprint, c_row, profile, device_override=None):
        ctx.state_store.set_module_unit_status(
            task_id=ctx.task_id, module_name="D", unit_id=blueprint.unit_id, status="done",
        )
        return str(blueprint.segment_path)

    monkeypatch.setattr(scheduler_tasks, "_run_c_chain_unit", _fake_run_c)
    monkeypatch.setattr(scheduler_tasks, "_run_d_chain_unit", _fake_run_d)

    result = scheduler.execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_map,
        d_blueprints_by_index=d_blueprints_map,
        module_a_output=_build_module_a_output(),
        unit_outputs_dir=context.artifacts_dir / "module_b_units",
        frames_dir=context.artifacts_dir / "frames",
        target_segment_id=None,
    )

    # big_001 (seg_0001/0002) 完成 → C/D 应完成
    # big_002 (seg_0003/0004) 失败 → 应在 failed_chain_indexes
    fi = result["failed_chain_indexes"]
    # seg_0003 unit_index=2, seg_0004 unit_index=3
    assert 2 in fi or 3 in fi, f"big_002 应在失败列表 {fi}"
    # seg_0001 unit_index=0, seg_0002 unit_index=1
    assert 0 not in fi, f"seg_0001 不应失败 {fi}"
    assert 1 not in fi, f"seg_0002 不应失败 {fi}"

    # big_001 的 C/D 应完成
    for seg_id in ("seg_0001", "seg_0002"):
        c_unit = f"shot_{int(seg_id.split('_')[1]):03d}"
        c_row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="C", unit_id=c_unit)
        assert c_row and c_row["status"] == "done", f"{c_unit} 应为 done"
        d_row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="D", unit_id=c_unit)
        assert d_row and d_row["status"] == "done", f"{c_unit} D 应为 done"

    # big_002 的 C/D 应 blocked
    for seg_id in ("seg_0003", "seg_0004"):
        c_unit = f"shot_{int(seg_id.split('_')[1]):03d}"
        c_row = context.state_store.get_module_unit_record(task_id=context.task_id, module_name="C", unit_id=c_unit)
        assert c_row and c_row["status"] == "failed", f"{c_unit} 应为 failed"


# ── 辅助函数 ──

def _build_module_a_output() -> dict:
    return {
        "duration": 20.0,
        "segments": [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.5, "label": "start", "role": "lyric"},
            {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 2.5, "end_time": 5.0, "label": "start", "role": "lyric"},
            {"segment_id": "seg_0003", "big_segment_id": "big_002", "start_time": 5.0, "end_time": 7.5, "label": "verse", "role": "lyric"},
            {"segment_id": "seg_0004", "big_segment_id": "big_002", "start_time": 7.5, "end_time": 10.0, "label": "verse", "role": "lyric"},
            {"segment_id": "seg_0005", "big_segment_id": "big_003", "start_time": 10.0, "end_time": 12.5, "label": "chorus", "role": "lyric"},
        ],
        "big_segments": [
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 5.0, "label": "start"},
            {"segment_id": "big_002", "start_time": 5.0, "end_time": 10.0, "label": "verse"},
            {"segment_id": "big_003", "start_time": 10.0, "end_time": 12.5, "label": "chorus"},
        ],
        "energy_features": [],
        "lyric_units": [],
    }


def _make_fake_role3_with_streaming(call_log, chain_units):
    """构造 role3 打桩：写 streaming 文件 + 标记 done。"""
    def _fn(ctx):
        if call_log is not None:
            call_log.append("role3")
        ctx.state_store.set_module_unit_status(task_id=ctx.task_id, module_name="B", unit_id="role3", status="done")
        # 写 streaming 文件供轮询循环检测
        sdir = ctx.artifacts_dir / "module_b_work" / "role3" / "streaming"
        sdir.mkdir(parents=True, exist_ok=True)
        big_ids_seen: set[str] = set()
        from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit
        for cu in chain_units:
            if isinstance(cu, CrossChainUnit):
                bid = f"big_{cu.unit_index + 1:03d}"
                if bid not in big_ids_seen:
                    big_ids_seen.add(bid)
                    (sdir / f"role3_segment_output.streaming.{bid}.md").write_text(
                        f"## {bid}\n- segment_ids: {cu.segment_id}\n", encoding="utf-8",
                    )
        # 写主产物
        path = ctx.artifacts_dir / "module_b_work" / "role3" / "role3_shot_output.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake output\n", encoding="utf-8")
        return path
    return _fn


def _make_stub_role4_for_big(chain_units):
    """构造 _run_role4_for_big_segment_shots 的桩函数：直接标记 seg done + role4 done。"""
    def _stub(context, *, big_segment_id, unit_outputs_dir, role3_streaming_dir):
        del big_segment_id, unit_outputs_dir, role3_streaming_dir
        context.state_store.set_module_unit_status(
            task_id=context.task_id, module_name="B", unit_id="role4", status="done",
        )
        for cu in chain_units:
            context.state_store.set_module_unit_status(
                task_id=context.task_id, module_name="B", unit_id=cu.segment_id, status="done",
            )
    return _stub


def _run_b_chain_batch_for_test(context: RuntimeContext, target_units: list[ModuleBUnit]) -> dict:
    return scheduler_tasks._run_b_chain_batch(
        context=context,
        target_segment_ids={u.unit_id for u in target_units},
        target_units=target_units,
    )


def _build_fixture(
    tmp_path: Path,
    task_id: str,
    chain_count: int = 5,
    global_render_limit: int = 6,
) -> tuple[RuntimeContext, list[CrossChainUnit], dict[str, ModuleBUnit], dict[int, ModuleDUnitBlueprint], Path]:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    task_dir = workspace_root / task_id
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = workspace_root / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")

    config = AppConfig(
        paths=PathsConfig(runs_dir="runs", default_audio_path="resources/demo.mp3"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe",
            video_codec="libx264", audio_codec="aac",
            fps=24, video_preset="veryfast", video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        module_b=ModuleBConfig(),
        module_c=ModuleCConfig(render_workers=3, unit_retry_times=1),
        module_d=ModuleDConfig(render_backend="cpu"),
        cross_module=CrossModuleConfig(
            global_render_limit=global_render_limit,
            scheduler_tick_ms=10,
            adaptive_window=CrossModuleAdaptiveWindowConfig(enabled=False),
        ),
        module_a=ModuleAConfig(funasr_language="auto"),
    )
    logger = logging.getLogger(f"bcd_chain_test_{task_id}")
    logger.setLevel(logging.INFO)
    state_store = StateStore(db_path=workspace_root / "state.sqlite3")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path="config.json")

    seg_pairs = [
        ("seg_0001", "shot_001"),
        ("seg_0002", "shot_002"),
        ("seg_0003", "shot_003"),
        ("seg_0004", "shot_004"),
        ("seg_0005", "shot_005"),
    ][:chain_count]

    chain_units = [
        CrossChainUnit(
            unit_index=idx,
            segment_id=seg_id,
            shot_id=shot_id,
            start_time=float(idx * 2.5),
            end_time=float(idx * 2.5 + 2.5),
            duration=2.5,
        )
        for idx, (seg_id, shot_id) in enumerate(seg_pairs)
    ]

    b_units = {
        item.segment_id: ModuleBUnit(
            unit_id=item.segment_id, unit_index=item.unit_index,
            segment={"segment_id": item.segment_id},
            start_time=item.start_time, end_time=item.end_time, duration=item.duration,
        )
        for item in chain_units
    }

    d_blueprints = {
        item.unit_index: ModuleDUnitBlueprint(
            unit_id=item.shot_id, unit_index=item.unit_index,
            start_time=item.start_time, end_time=item.end_time, duration=item.duration,
            exact_frames=24,
            segment_path=artifacts_dir / "segments" / f"segment_{item.unit_index + 1:03d}.mp4",
            temp_segment_path=artifacts_dir / "segments" / f"segment_{item.unit_index + 1:03d}.tmp.mp4",
        )
        for item in chain_units
    }

    # module_a_output.json
    (artifacts_dir / "module_a_output.json").write_text(
        json.dumps(_build_module_a_output(), ensure_ascii=False), encoding="utf-8",
    )

    # B 单元（含 role1~4）
    state_store.sync_module_units(
        task_id=task_id, module_name="B",
        units=[
            {"unit_id": "role1", "unit_index": -4, "start_time": 0.0, "end_time": 0.0, "duration": 0.0},
            {"unit_id": "role2", "unit_index": -3, "start_time": 0.0, "end_time": 0.0, "duration": 0.0},
            {"unit_id": "role3", "unit_index": -2, "start_time": 0.0, "end_time": 0.0, "duration": 0.0},
            {"unit_id": "role4", "unit_index": -1, "start_time": 0.0, "end_time": 0.0, "duration": 0.0},
            *({"unit_id": item.segment_id, "unit_index": item.unit_index,
               "start_time": item.start_time, "end_time": item.end_time, "duration": item.duration}
              for item in chain_units),
        ],
    )
    state_store.sync_module_units(
        task_id=task_id, module_name="C",
        units=[{"unit_id": item.shot_id, "unit_index": item.unit_index,
                "start_time": item.start_time, "end_time": item.end_time, "duration": item.duration}
               for item in chain_units],
    )
    state_store.sync_module_units(
        task_id=task_id, module_name="D",
        units=[{"unit_id": item.shot_id, "unit_index": item.unit_index,
                "start_time": item.start_time, "end_time": item.end_time, "duration": item.duration}
               for item in chain_units],
    )

    context = RuntimeContext(
        task_id=task_id, audio_path=audio_path,
        task_dir=task_dir, artifacts_dir=artifacts_dir,
        config=config, logger=logger, state_store=state_store,
    )
    return context, chain_units, b_units, d_blueprints, workspace_root
