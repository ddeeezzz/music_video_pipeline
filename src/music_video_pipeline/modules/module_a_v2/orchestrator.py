"""
文件用途：实现模块A V2单一路径编排入口。
核心流程：按“感知层 -> 算法层 -> 契约校验 -> 输出落盘”执行。
输入输出：输入 RuntimeContext，输出 module_a_output.json 路径。
依赖说明：依赖 module_a_v2 子模块与公共上下文/IO能力。
维护说明：本文件仅负责调度与失败语义，不承载分段算法细节。
"""

# 标准库：用于路径类型提示
from pathlib import Path

# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON落盘工具
from music_video_pipeline.io_utils import write_json
# 项目内模块：类型契约校验
from music_video_pipeline.types import validate_module_a_output
# 项目内模块：V2别名映射
from music_video_pipeline.modules.module_a_v2.utils.alias_map import build_module_a_v2_alias_map
# 项目内模块：V2媒体时长探测
from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration
# 项目内模块：V2算法层
from music_video_pipeline.modules.module_a_v2.algorithm import AlgorithmBundle, run_algorithm_stage
# 项目内模块：V2产物路径管理
from music_video_pipeline.modules.module_a_v2.artifacts import ModuleAV2Artifacts, build_module_a_v2_artifacts
# 项目内模块：V2感知层
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle, run_perception_stage
# 项目内模块：V2可视化聚合与渲染
from music_video_pipeline.modules.module_a_v2.visualization import collect_visualization_payload, render_visualization_html


# 常量：模块A V2自动可视化默认输出文件名模板（task_id 前缀）
AUTO_VISUALIZATION_FILE_TEMPLATE = "{task_id}_module_a_v2_visualization.html"
# 常量：模块A V2自动可视化默认音频处理模式
AUTO_VISUALIZATION_AUDIO_MODE = "copy"


def _run_perception_stage(
    audio_path: Path,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    context: RuntimeContext,
) -> PerceptionBundle:
    """
    功能说明：封装感知层调用，便于测试替换与分层调度。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: 产物路径对象。
    - context: 运行上下文。
    返回值：
    - PerceptionBundle: 感知层统一输出。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return run_perception_stage(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        device=context.config.module_a.device,
        demucs_model=context.config.module_a.demucs_model,
        funasr_model=context.config.module_a.funasr_model,
        funasr_language=context.config.module_a.funasr_language,
        skip_funasr_when_vocals_silent=context.config.module_a.skip_funasr_when_vocals_silent,
        vocal_skip_peak_rms_threshold=context.config.module_a.vocal_skip_peak_rms_threshold,
        vocal_skip_active_ratio_threshold=context.config.module_a.vocal_skip_active_ratio_threshold,
        logger=context.logger,
    )


def _run_algorithm_stage(
    perception: PerceptionBundle,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    context: RuntimeContext,
) -> AlgorithmBundle:
    """
    功能说明：封装算法层调用，便于测试替换与分层调度。
    参数说明：
    - perception: 感知层产物。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: 产物路径对象。
    - context: 运行上下文。
    返回值：
    - AlgorithmBundle: 算法层统一输出。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：遵循当前实现中的兜底与裁剪策略。
    """
    return run_algorithm_stage(
        perception=perception,
        duration_seconds=duration_seconds,
        instrumental_labels=context.config.module_a.instrumental_labels,
        merge_gap_seconds=context.config.module_a.merge_gap_seconds,
        lyric_head_offset_seconds=context.config.module_a.lyric_head_offset_seconds,
        lyric_boundary_near_anchor_seconds=context.config.module_a.lyric_boundary_near_anchor_seconds,
        content_role_tiny_merge_bars=context.config.module_a.content_role_tiny_merge_bars,
        artifacts=artifacts,
        logger=context.logger,
    )


def _render_module_a_v2_visualization(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块A V2自动可视化渲染并返回HTML路径。
    参数说明：
    - context: 运行时上下文对象。
    返回值：
    - Path: 可视化HTML路径。
    异常说明：可视化聚合或渲染失败时向上抛出异常。
    边界条件：输出覆盖同名历史文件，实现“每次运行都重绘”。
    """
    output_name = AUTO_VISUALIZATION_FILE_TEMPLATE.format(task_id=context.task_id)
    output_html_path = context.task_dir / output_name
    payload = collect_visualization_payload(task_dir=context.task_dir)
    return render_visualization_html(
        payload=payload,
        output_html_path=output_html_path,
        audio_mode=AUTO_VISUALIZATION_AUDIO_MODE,
    )


def run_module_a_v2(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块A V2并产出标准JSON。
    参数说明：
    - context: 运行时上下文对象。
    返回值：
    - Path: 模块A输出JSON文件路径。
    异常说明：关键产物缺失时抛 RuntimeError。
    边界条件：lyric_units 允许为空，其他关键字段必须有效。
    """
    context.logger.info("模块A V2开始执行，task_id=%s，输入音频=%s", context.task_id, context.audio_path)
    duration_seconds = probe_audio_duration(
        audio_path=context.audio_path,
        ffprobe_bin=context.config.ffmpeg.ffprobe_bin,
        logger=context.logger,
    )
    artifacts = build_module_a_v2_artifacts(context.artifacts_dir / "module_a_work_v2")
    perception = _run_perception_stage(
        audio_path=context.audio_path,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        context=context,
    )
    analysis_bundle = _run_algorithm_stage(
        perception=perception,
        duration_seconds=duration_seconds,
        artifacts=artifacts,
        context=context,
    )

    if not analysis_bundle.big_segments:
        raise RuntimeError("模块A V2失败：big_segments 为空")
    if not analysis_bundle.segments:
        raise RuntimeError("模块A V2失败：segments 为空")
    if len(analysis_bundle.beats) < 2:
        raise RuntimeError("模块A V2失败：beats 少于2个")
    if not analysis_bundle.energy_features:
        raise RuntimeError("模块A V2失败：energy_features 为空")

    analysis_data = {
        "big_segments_stage1": analysis_bundle.big_segments_stage1,
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
    }
    alias_map = build_module_a_v2_alias_map(mode="v2_single", analysis_data=analysis_data)
    output_data = {
        "task_id": context.task_id,
        "audio_path": str(context.audio_path),
        "big_segments": analysis_bundle.big_segments,
        "segments": analysis_bundle.segments,
        "beats": analysis_bundle.beats,
        "lyric_units": analysis_bundle.lyric_units,
        "energy_features": analysis_bundle.energy_features,
        "alias_map": alias_map,
    }
    validate_module_a_output(output_data)
    output_path = context.artifacts_dir / "module_a_output.json"
    write_json(output_path, output_data)
    try:
        visualization_path = _render_module_a_v2_visualization(context=context)
        context.logger.info("模块A V2自动可视化完成，task_id=%s，输出=%s", context.task_id, visualization_path)
    except Exception as error:  # noqa: BLE001
        context.logger.warning("模块A V2自动可视化失败，已忽略，task_id=%s，错误=%s", context.task_id, error)
    context.logger.info("模块A V2执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
