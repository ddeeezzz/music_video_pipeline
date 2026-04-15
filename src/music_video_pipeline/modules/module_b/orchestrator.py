"""
文件用途：实现模块 B（视觉脚本）的单元级断点重试编排入口。
核心流程：读取模块 A 输出，按 segment 同步单元状态，并行执行待处理单元，汇总输出模块 B 清单。
输入输出：输入 RuntimeContext，输出模块 B 清单 JSON 路径。
依赖说明：依赖模块 B 子组件、生成器工厂与 JSON 工具。
维护说明：保持模块级状态机不变，单元级状态仅作为 B 内部恢复能力。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：分镜生成器工厂
from music_video_pipeline.generators import build_script_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块B单元执行器
from music_video_pipeline.modules.module_b.executor import execute_units_with_retry
# 项目内模块：模块B输出对象构建器
from music_video_pipeline.modules.module_b.output_builder import build_module_b_output
# 项目内模块：模块B单元模型工具
from music_video_pipeline.modules.module_b.unit_models import build_module_b_units, build_unit_map, build_unit_sync_payload
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def run_module_b(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 B，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 B 输出清单 JSON 路径。
    异常说明：输入脚本不存在、单元重试耗尽或输出不完整时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块B开始执行，task_id=%s", context.task_id)

    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    units = build_module_b_units(module_a_output=module_a_output)
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="B",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)

    pending_records = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["pending", "failed", "running"],
    )
    units_to_run = [
        units_by_id[str(record["unit_id"])]
        for record in pending_records
        if str(record["unit_id"]) in units_by_id
    ]
    context.logger.info(
        "模块B单元调度计划，task_id=%s，unit_total=%s，unit_to_run=%s",
        context.task_id,
        len(units),
        len(units_to_run),
    )

    unit_outputs_dir = context.artifacts_dir / "module_b_units"
    generator = build_script_generator(
        mode=context.config.mode.script_generator,
        logger=context.logger,
        module_b_config=context.config.module_b,
    )
    execute_units_with_retry(
        context=context,
        units_to_run=units_to_run,
        generator=generator,
        module_a_output=module_a_output,
        unit_outputs_dir=unit_outputs_dir,
    )

    done_unit_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
    if len(done_unit_records) != len(units):
        done_unit_ids = {str(item["unit_id"]) for item in done_unit_records}
        missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_unit_ids]
        raise RuntimeError(f"模块B执行失败：存在未完成单元，missing_unit_ids={missing_unit_ids}")

    module_b_output = build_module_b_output(
        done_unit_records=done_unit_records,
        module_a_output=module_a_output,
        instrumental_labels=context.config.module_a.instrumental_labels,
    )
    validate_module_b_output(module_b_output)

    output_path = context.artifacts_dir / "module_b_output.json"
    write_json(output_path, module_b_output)
    context.logger.info("模块B执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
