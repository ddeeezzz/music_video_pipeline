"""
文件用途：实现模块 C（图像生成）的单元级断点重试编排入口。
核心流程：读取模块 B 分镜，按 shot 同步单元状态，并行执行待处理单元，汇总输出模块 C 清单。
输入输出：输入 RuntimeContext，输出模块 C 清单 JSON 路径。
依赖说明：依赖模块 C 子组件、生成器工厂与 JSON 工具。
维护说明：保持模块级状态机不变，单元级状态仅作为 C 内部恢复能力。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：关键帧生成器工厂
from music_video_pipeline.generators import build_frame_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块C单元执行器
from music_video_pipeline.modules.module_c.executor import execute_units_with_retry
# 项目内模块：模块C输出对象构建器
from music_video_pipeline.modules.module_c.output_builder import build_module_c_output
# 项目内模块：模块C单元模型工具
from music_video_pipeline.modules.module_c.unit_models import build_module_c_units, build_unit_map, build_unit_sync_payload
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_b_output


def run_module_c(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 C，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：输入脚本不存在、单元重试耗尽或输出不完整时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块C开始执行，task_id=%s", context.task_id)

    module_b_path = context.artifacts_dir / "module_b_output.json"
    module_b_output = read_json(module_b_path)
    try:
        validate_module_b_output(module_b_output)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "模块C输入契约校验失败：检测到旧版或不兼容的 module_b_output。"
            "请从模块B重跑，确保产物包含 keyframe_prompt/video_prompt 字段。"
            f"原始错误：{error}"
        ) from error

    units = build_module_c_units(shots=module_b_output)
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="C",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)

    pending_records = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="C",
        statuses=["pending", "failed", "running"],
    )
    units_to_run = [
        units_by_id[str(record["unit_id"])]
        for record in pending_records
        if str(record["unit_id"]) in units_by_id
    ]
    context.logger.info(
        "模块C单元调度计划，task_id=%s，unit_total=%s，unit_to_run=%s",
        context.task_id,
        len(units),
        len(units_to_run),
    )

    frames_dir = context.artifacts_dir / "frames"
    generator = build_frame_generator(mode=context.config.mode.frame_generator, logger=context.logger)
    execute_units_with_retry(
        context=context,
        units_to_run=units_to_run,
        generator=generator,
        frames_dir=frames_dir,
    )

    frame_items = context.state_store.list_module_c_done_frame_items(task_id=context.task_id)
    if len(frame_items) != len(units):
        done_shot_ids = {str(item["shot_id"]) for item in frame_items}
        missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_shot_ids]
        raise RuntimeError(f"模块C执行失败：存在未完成单元，missing_unit_ids={missing_unit_ids}")

    output_data = build_module_c_output(
        task_id=context.task_id,
        frames_dir=frames_dir,
        frame_items=frame_items,
    )
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块C执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
