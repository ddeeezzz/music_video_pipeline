"""
文件用途：实现跨模块 B/C/D 链路波前并行的编排入口。
核心流程：读取模块 A 输出，初始化 B/C/D 单元状态，调度链路执行并在末端统一处理 D concat。
输入输出：输入 RuntimeContext，输出跨模块执行摘要。
依赖说明：依赖模块 B/C/D 子组件、状态库与 JSON 工具。
维护说明：保持模块级状态语义不变，链路并行仅改变执行编排方式。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：跨模块链路模型
from music_video_pipeline.modules.cross_bcd.models import CrossChainUnit, build_cross_chain_units
# 项目内模块：跨模块调度器
from music_video_pipeline.modules.cross_bcd.scheduler import execute_cross_bcd_wavefront
# 项目内模块：模块 B 单元模型
from music_video_pipeline.modules.module_b.unit_models import build_module_b_units
# 项目内模块：模块 B 输出构建器
from music_video_pipeline.modules.module_b.output_builder import build_module_b_output
# 项目内模块：模块 C 输出构建器
from music_video_pipeline.modules.module_c.output_builder import build_module_c_output
# 项目内模块：模块C扩散元信息解析
from music_video_pipeline.generators.frame_generator import resolve_module_c_diffusion_trace_metadata
# 项目内模块：模块 D 终拼工具
from music_video_pipeline.modules.module_d.finalizer import _concat_segment_videos, _probe_media_duration
# 项目内模块：模块 D 输出构建器
from music_video_pipeline.modules.module_d.output_builder import build_module_d_output
# 项目内模块：模块 D 单元蓝图
from music_video_pipeline.modules.module_d.unit_models import build_module_d_unit_blueprints
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def run_cross_module_bcd(context: RuntimeContext, target_segment_id: str | None = None) -> dict[str, Any]:
    """
    功能说明：执行跨模块 B/C/D 波前并行编排。
    参数说明：
    - context: 运行上下文对象。
    - target_segment_id: 可选，仅执行目标 segment 链路。
    返回值：
    - dict[str, Any]: 执行摘要（失败链路、模块状态、输出路径）。
    异常说明：
    - RuntimeError: 模块 A 输出缺失、链路执行失败或 D 终拼失败时抛出。
    边界条件：失败仅阻断对应链路，其他链路继续执行。
    """
    context.logger.info("跨模块B/C/D并行开始执行，task_id=%s，target_segment_id=%s", context.task_id, target_segment_id or "<all>")

    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    chain_units = build_cross_chain_units(module_a_output=module_a_output)
    if not chain_units:
        raise RuntimeError("跨模块调度失败：模块A segments 为空，无法构建 B/C/D 链路。")

    selected_chain_units = (
        [item for item in chain_units if item.segment_id == target_segment_id]
        if target_segment_id
        else list(chain_units)
    )
    if target_segment_id and not selected_chain_units:
        raise RuntimeError(f"跨模块调度失败：未找到目标链路，segment_id={target_segment_id}")

    unit_outputs_dir = context.artifacts_dir / "module_b_units"
    frames_dir = context.artifacts_dir / "frames"
    segments_dir = context.artifacts_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    b_units = build_module_b_units(module_a_output=module_a_output)
    b_units_by_segment_id = {item.unit_id: item for item in b_units}

    shot_stubs = _build_shot_stubs(chain_units=chain_units)
    audio_duration = _probe_media_duration(
        media_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
    )
    d_blueprints = build_module_d_unit_blueprints(
        shots=shot_stubs,
        audio_duration=audio_duration,
        fps=context.config.ffmpeg.fps,
        segments_dir=segments_dir,
    )
    d_blueprints_by_index = {item.unit_index: item for item in d_blueprints}

    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="B",
        units=[
            {
                "unit_id": item.segment_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in chain_units
        ],
    )
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="C",
        units=[
            {
                "unit_id": item.shot_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in chain_units
        ],
    )
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="D",
        units=[
            {
                "unit_id": item.unit_id,
                "unit_index": item.unit_index,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration": item.duration,
            }
            for item in d_blueprints
        ],
    )

    context.state_store.set_module_status(task_id=context.task_id, module_name="B", status="running")
    context.state_store.set_module_status(task_id=context.task_id, module_name="C", status="running")
    context.state_store.set_module_status(task_id=context.task_id, module_name="D", status="running")

    scheduler_result = execute_cross_bcd_wavefront(
        context=context,
        chain_units=chain_units,
        b_units_by_segment_id=b_units_by_segment_id,
        d_blueprints_by_index=d_blueprints_by_index,
        module_a_output=module_a_output,
        unit_outputs_dir=unit_outputs_dir,
        frames_dir=frames_dir,
        target_segment_id=target_segment_id,
    )

    module_b_output_path = _refresh_module_b_output(context=context, module_a_output=module_a_output)
    module_c_output_path = _refresh_module_c_output(context=context, frames_dir=frames_dir)
    module_d_output_path, output_video_path = _refresh_module_d_output(
        context=context,
        audio_duration=audio_duration,
        selected_indexes={item.unit_index for item in selected_chain_units},
    )

    b_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="B")
    c_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="C")
    d_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")

    _sync_module_status_by_summary(
        context=context,
        module_name="B",
        summary=b_summary,
        artifact_path=module_b_output_path,
    )
    _sync_module_status_by_summary(
        context=context,
        module_name="C",
        summary=c_summary,
        artifact_path=module_c_output_path,
    )
    _sync_module_status_by_summary(
        context=context,
        module_name="D",
        summary=d_summary,
        artifact_path=output_video_path,
    )

    failed_chain_indexes = scheduler_result.get("failed_chain_indexes", [])
    if failed_chain_indexes:
        context.state_store.update_task_status(
            task_id=context.task_id,
            status="failed",
            error_message=f"跨模块链路失败，failed_chain_indexes={failed_chain_indexes}",
            output_video_path="",
        )
        raise RuntimeError(f"跨模块链路执行失败，failed_chain_indexes={failed_chain_indexes}")

    context.state_store.mark_task_done_if_possible(task_id=context.task_id, output_video_path=str(output_video_path))
    context.logger.info(
        "跨模块B/C/D并行执行完成，task_id=%s，module_b_output=%s，module_c_output=%s，module_d_output=%s，video=%s",
        context.task_id,
        module_b_output_path,
        module_c_output_path,
        module_d_output_path,
        output_video_path,
    )
    return {
        "task_id": context.task_id,
        "module_b_output_path": str(module_b_output_path),
        "module_c_output_path": str(module_c_output_path),
        "module_d_output_path": str(module_d_output_path),
        "output_video_path": str(output_video_path),
        "failed_chain_indexes": failed_chain_indexes,
        "scheduler_result": scheduler_result,
        "module_b_unit_summary": b_summary,
        "module_c_unit_summary": c_summary,
        "module_d_unit_summary": d_summary,
    }


def _refresh_module_b_output(context: RuntimeContext, module_a_output: dict[str, Any]) -> Path:
    """
    功能说明：根据模块 B 已完成单元刷新 module_b_output.json。
    参数说明：
    - context: 运行上下文对象。
    - module_a_output: 模块 A 输出字典。
    返回值：
    - Path: module_b_output.json 路径。
    异常说明：构建失败时抛 RuntimeError。
    边界条件：允许输出部分链路分镜（用于失败后恢复排障）。
    """
    done_unit_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
    if done_unit_records:
        module_b_output = build_module_b_output(
            done_unit_records=done_unit_records,
            module_a_output=module_a_output,
            instrumental_labels=context.config.module_a.instrumental_labels,
        )
        try:
            validate_module_b_output(module_b_output)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "跨模块链路模块B产物校验失败：检测到旧版或不兼容分镜字段。"
                "请从模块B重跑并生成 keyframe_prompt/video_prompt。"
                f"原始错误：{error}"
            ) from error
    else:
        module_b_output = []
    output_path = context.artifacts_dir / "module_b_output.json"
    write_json(output_path, module_b_output)
    return output_path


def _refresh_module_c_output(context: RuntimeContext, frames_dir: Path) -> Path:
    """
    功能说明：根据模块 C 已完成单元刷新 module_c_output.json。
    参数说明：
    - context: 运行上下文对象。
    - frames_dir: 帧目录路径。
    返回值：
    - Path: module_c_output.json 路径。
    异常说明：无。
    边界条件：允许输出部分链路 frame_items（用于失败后恢复排障）。
    """
    frame_items = context.state_store.list_module_c_done_frame_items(task_id=context.task_id)
    if str(context.config.mode.frame_generator).strip().lower() == "diffusion":
        trace_metadata = resolve_module_c_diffusion_trace_metadata()
        frame_items = [
            {
                **item,
                "binding_name": str(trace_metadata["binding_name"]),
                "base_model_key": str(trace_metadata["base_model_key"]),
                "lora_file": str(trace_metadata["lora_file"]),
            }
            for item in frame_items
        ]
    module_c_output = build_module_c_output(
        task_id=context.task_id,
        frames_dir=frames_dir,
        frame_items=frame_items,
    )
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, module_c_output)
    return output_path


def _refresh_module_d_output(context: RuntimeContext, audio_duration: float, selected_indexes: set[int]) -> tuple[Path, Path]:
    """
    功能说明：刷新模块 D 输出，并在全量链路完成时执行最终 concat。
    参数说明：
    - context: 运行上下文对象。
    - audio_duration: 原音轨时长（秒）。
    - selected_indexes: 本轮调度覆盖的链路索引集合。
    返回值：
    - tuple[Path, Path]: (module_d_output.json 路径, final_output.mp4 路径)。
    异常说明：concat 失败时抛 RuntimeError。
    边界条件：仅当模块 D 全部单元 done 时执行 concat。
    """
    d_summary = context.state_store.get_module_unit_status_summary(task_id=context.task_id, module_name="D")
    done_unit_records = context.state_store.list_module_d_done_segment_items(task_id=context.task_id)
    output_video_path = context.task_dir / "final_output.mp4"

    concat_result: dict[str, Any]
    if d_summary["total_units"] > 0 and d_summary["status_counts"].get("done", 0) == d_summary["total_units"]:
        ordered_segment_paths = [
            Path(str(item.get("artifact_path", "")))
            for item in sorted(done_unit_records, key=lambda row: int(row.get("unit_index", 0)))
        ]
        concat_result = _concat_segment_videos(
            segment_paths=ordered_segment_paths,
            concat_file_path=context.artifacts_dir / "segments_concat.txt",
            ffmpeg_bin=context.config.ffmpeg.ffmpeg_bin,
            audio_path=context.audio_path,
            output_video_path=output_video_path,
            audio_duration=audio_duration,
            fps=context.config.ffmpeg.fps,
            video_codec=context.config.ffmpeg.video_codec,
            audio_codec=context.config.ffmpeg.audio_codec,
            video_preset=context.config.ffmpeg.video_preset,
            video_crf=context.config.ffmpeg.video_crf,
            video_accel_mode=context.config.ffmpeg.video_accel_mode,
            gpu_video_codec=context.config.ffmpeg.gpu_video_codec,
            gpu_preset=context.config.ffmpeg.gpu_preset,
            gpu_rc_mode=context.config.ffmpeg.gpu_rc_mode,
            gpu_cq=context.config.ffmpeg.gpu_cq,
            gpu_bitrate=context.config.ffmpeg.gpu_bitrate,
            concat_video_mode=context.config.ffmpeg.concat_video_mode,
            concat_copy_fallback_reencode=context.config.ffmpeg.concat_copy_fallback_reencode,
            logger=context.logger,
        )
    else:
        concat_result = {
            "mode": "skipped",
            "copy_fallback_triggered": False,
            "selected_indexes": sorted(selected_indexes),
        }

    shot_payload_map: dict[str, dict[str, Any]] = {}
    module_c_output_path = context.artifacts_dir / "module_c_output.json"
    if module_c_output_path.exists():
        module_c_output = read_json(module_c_output_path)
        frame_items = module_c_output.get("frame_items", []) if isinstance(module_c_output, dict) else []
        if isinstance(frame_items, list):
            for item in frame_items:
                if not isinstance(item, dict):
                    continue
                shot_id = str(item.get("shot_id", "")).strip()
                if not shot_id:
                    continue
                shot_payload_map[shot_id] = dict(item)

    module_d_output = build_module_d_output(
        task_id=context.task_id,
        output_video_path=output_video_path,
        done_unit_records=done_unit_records,
        concat_result=concat_result,
        shot_payload_map=shot_payload_map,
    )
    module_d_output_path = context.artifacts_dir / "module_d_output.json"
    write_json(module_d_output_path, module_d_output)
    return module_d_output_path, output_video_path


def _sync_module_status_by_summary(
    context: RuntimeContext,
    module_name: str,
    summary: dict[str, Any],
    artifact_path: Path,
) -> None:
    """
    功能说明：根据模块单元摘要刷新模块级状态。
    参数说明：
    - context: 运行上下文对象。
    - module_name: 模块名（B/C/D）。
    - summary: 模块单元状态摘要。
    - artifact_path: 模块产物路径。
    返回值：无。
    异常说明：无。
    边界条件：仅当 total_units>0 且全部 done 时标记模块 done。
    """
    total_units = int(summary.get("total_units", 0))
    done_count = int(summary.get("status_counts", {}).get("done", 0))
    failed_unit_ids = list(summary.get("failed_unit_ids", []))
    running_unit_ids = list(summary.get("running_unit_ids", []))
    pending_unit_ids = list(summary.get("pending_unit_ids", []))
    if total_units > 0 and done_count == total_units:
        context.state_store.set_module_status(
            task_id=context.task_id,
            module_name=module_name,
            status="done",
            artifact_path=str(artifact_path),
            error_message="",
        )
        return

    problem_unit_ids = failed_unit_ids + running_unit_ids + pending_unit_ids
    error_message = f"模块{module_name}存在未完成链路，problem_unit_ids={problem_unit_ids}"
    context.state_store.set_module_status(
        task_id=context.task_id,
        module_name=module_name,
        status="failed",
        artifact_path=str(artifact_path),
        error_message=error_message,
    )


def _build_shot_stubs(chain_units: list[CrossChainUnit]) -> list[dict[str, Any]]:
    """
    功能说明：将链路单元转换为 shot 时间桩数据，用于 D 蓝图预分配。
    参数说明：
    - chain_units: 跨模块链路单元数组。
    返回值：
    - list[dict[str, Any]]: shot 桩数组。
    异常说明：无。
    边界条件：输出顺序保持与 unit_index 一致。
    """
    return [
        {
            "shot_id": item.shot_id,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "duration": item.duration,
        }
        for item in sorted(chain_units, key=lambda row: row.unit_index)
    ]
