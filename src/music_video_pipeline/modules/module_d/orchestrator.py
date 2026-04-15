"""
文件用途：实现模块 D（视频合成）的单元级断点重试编排入口。
核心流程：读取模块 C 输出，按 shot 同步单元状态，并行执行待处理单元，最后一次性终拼输出视频。
输入输出：输入 RuntimeContext，输出最终视频路径。
依赖说明：依赖模块 D 子组件与 JSON 工具。
维护说明：保持两阶段终拼，不引入流式边拼接。
"""

# 标准库：用于阶段计时
import time
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块D执行器
from music_video_pipeline.modules.module_d.executor import execute_units_with_retry
# 项目内模块：模块D终拼器
from music_video_pipeline.modules.module_d.finalizer import _concat_segment_videos, _probe_media_duration
# 项目内模块：模块D输出对象构建器
from music_video_pipeline.modules.module_d.output_builder import build_module_d_output
# 项目内模块：模块D单元模型工具
from music_video_pipeline.modules.module_d.unit_models import build_module_d_units, build_unit_map, build_unit_sync_payload


def run_module_d(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 D，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 最终视频路径。
    异常说明：输入清单缺失、单元重试耗尽或终拼失败时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块D开始执行，task_id=%s", context.task_id)
    stage_total_start = time.perf_counter()

    module_c_path = context.artifacts_dir / "module_c_output.json"
    module_c_output = read_json(module_c_path)
    frame_items = module_c_output.get("frame_items", [])
    if not frame_items:
        raise RuntimeError("模块D无法执行：模块C输出的 frame_items 为空。")

    segments_dir = context.artifacts_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    audio_duration = _probe_media_duration(
        media_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
    )

    units = build_module_d_units(
        frame_items=frame_items,
        audio_duration=audio_duration,
        fps=context.config.ffmpeg.fps,
        segments_dir=segments_dir,
    )
    shot_payload_map = {unit.unit_id: dict(unit.shot) for unit in units}
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="D",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)

    pending_records = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="D",
        statuses=["pending", "failed", "running"],
    )
    units_to_run = [
        units_by_id[str(record["unit_id"])]
        for record in pending_records
        if str(record["unit_id"]) in units_by_id
    ]
    context.logger.info(
        "模块D单元调度计划，task_id=%s，unit_total=%s，unit_to_run=%s",
        context.task_id,
        len(units),
        len(units_to_run),
    )

    stage_render_start = time.perf_counter()
    execute_units_with_retry(context=context, units_to_run=units_to_run)
    render_elapsed = time.perf_counter() - stage_render_start

    done_unit_records = context.state_store.list_module_d_done_segment_items(task_id=context.task_id)
    if len(done_unit_records) != len(units):
        done_unit_ids = {str(item["unit_id"]) for item in done_unit_records}
        missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_unit_ids]
        raise RuntimeError(f"模块D执行失败：存在未完成单元，missing_unit_ids={missing_unit_ids}")

    ordered_segment_paths = [
        Path(str(item.get("artifact_path", "")))
        for item in sorted(done_unit_records, key=lambda row: int(row.get("unit_index", 0)))
    ]

    output_video_path = context.task_dir / "final_output.mp4"
    stage_concat_start = time.perf_counter()
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
    concat_elapsed = time.perf_counter() - stage_concat_start

    module_d_output = build_module_d_output(
        task_id=context.task_id,
        output_video_path=output_video_path,
        done_unit_records=done_unit_records,
        concat_result=concat_result,
        shot_payload_map=shot_payload_map,
    )
    module_d_output_path = context.artifacts_dir / "module_d_output.json"
    write_json(module_d_output_path, module_d_output)

    total_elapsed = time.perf_counter() - stage_total_start
    context.logger.info(
        "模块D耗时统计，render_segments_elapsed=%.3fs，concat_elapsed=%.3fs，total_elapsed=%.3fs，concat_mode=%s，copy_fallback=%s",
        render_elapsed,
        concat_elapsed,
        total_elapsed,
        concat_result.get("mode", "unknown"),
        bool(concat_result.get("copy_fallback_triggered", False)),
    )
    context.logger.info("模块D执行完成，task_id=%s，输出=%s", context.task_id, output_video_path)
    return output_video_path
