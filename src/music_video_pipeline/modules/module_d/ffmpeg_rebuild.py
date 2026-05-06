"""
文件用途：基于现有模块 D 片段执行纯 FFmpeg 重后处理并重拼成片。
核心流程：读取既有模块 D 摘要，按需要增强单段运镜，再用终拼器生成新的最终视频。
输入输出：输入任务目录与配置路径，输出新的视频文件与摘要 JSON。
依赖说明：依赖项目内配置加载、JSON 读写与模块 D 终拼工具。
维护说明：本文件不触发 ToonCrafter，只复用已经落盘的 segment_*.mp4。
"""

# 标准库：用于结构化日志
import logging
# 标准库：用于路径处理
from pathlib import Path
from typing import Any
# 标准库：用于并行处理 segment 运镜后处理
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目内模块：配置加载
from music_video_pipeline.config import load_config
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块D终拼与 FFmpeg 运镜工具
from music_video_pipeline.modules.module_d.finalizer import (
    _concat_segment_videos,
    _probe_media_duration,
    apply_camera_plan_to_segment,
)


def rebuild_video_from_existing_segments(
    *,
    task_dir: Path,
    config_path: Path,
    output_video_path: Path,
    processed_segments_dir: Path,
    summary_output_path: Path,
    transition_duration_scale: float = 1.6,
    min_transition_duration_ms: int = 280,
    boost_small_camera_to_medium: bool = True,
) -> dict[str, Any]:
    """
    功能说明：基于已有 segment 视频执行纯 FFmpeg 重后处理并生成新成片。
    参数说明：
    - task_dir: 任务目录。
    - config_path: 配置文件路径。
    - output_video_path: 新成片输出路径。
    - processed_segments_dir: 新单段视频输出目录。
    - summary_output_path: 新摘要 JSON 输出路径。
    - transition_duration_scale: 非 none/hard_cut 转场时长放大倍率。
    - min_transition_duration_ms: 非 none/hard_cut 转场的最小时长。
    - boost_small_camera_to_medium: 是否将 small 运镜提升为 medium 强度。
    返回值：
    - dict[str, Any]: 新摘要对象。
    异常说明：输入摘要缺失、片段缺失、FFmpeg 失败时抛 RuntimeError。
    边界条件：仅使用现有片段文件，不访问 ToonCrafter 或 ComfyUI。
    """
    logger = logging.getLogger("music_video_pipeline.module_d.ffmpeg_rebuild")
    config = load_config(config_path=config_path)
    artifacts_dir = task_dir / "artifacts"
    module_d_output_path = artifacts_dir / "module_d_output.json"
    module_d_output = read_json(module_d_output_path)
    if not isinstance(module_d_output, dict):
        raise RuntimeError("模块D重后处理失败：module_d_output.json 结构非法。")
    segment_items = module_d_output.get("segment_items", [])
    if not isinstance(segment_items, list) or not segment_items:
        raise RuntimeError("模块D重后处理失败：module_d_output.json 缺少 segment_items。")

    ordered_segment_items = sorted(
        [item for item in segment_items if isinstance(item, dict)],
        key=lambda item: int(item.get("unit_index", 0)),
    )
    if not ordered_segment_items:
        raise RuntimeError("模块D重后处理失败：可用 segment_items 为空。")

    processed_segments_dir.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_path = _resolve_task_audio_path(task_dir=task_dir)
    audio_duration = _probe_media_duration(
        media_path=audio_path,
        ffprobe_bin=config.ffmpeg.ffprobe_bin,
    )

    def _process_single_segment(index: int, item: dict[str, Any]) -> dict[str, Any]:
        segment_path = Path(str(item.get("segment_path", ""))).resolve()
        if not segment_path.exists():
            raise RuntimeError(f"模块D重后处理失败：片段不存在，path={segment_path}")
        original_camera_plan = item.get("camera_plan", {})
        boosted_camera_plan = _boost_camera_plan(
            camera_plan=original_camera_plan,
            boost_small_to_medium=boost_small_camera_to_medium,
        )
        output_segment_path = processed_segments_dir / f"segment_{index:03d}.mp4"
        camera_applied = apply_camera_plan_to_segment(
            segment_path=segment_path,
            output_path=output_segment_path,
            ffmpeg_bin=config.ffmpeg.ffmpeg_bin,
            ffprobe_bin=config.ffmpeg.ffprobe_bin,
            fps=config.ffmpeg.fps,
            video_codec=config.ffmpeg.video_codec,
            video_preset=config.ffmpeg.video_preset,
            video_crf=config.ffmpeg.video_crf,
            camera_plan=boosted_camera_plan,
        )
        effective_segment_path = output_segment_path if camera_applied else segment_path
        boosted_transition = _boost_transition_plan(
            transition_plan=item.get("transition_plan", {}),
            duration_scale=transition_duration_scale,
            min_duration_ms=min_transition_duration_ms,
        )
        rebuilt_item = dict(item)
        rebuilt_item["source_segment_path"] = str(segment_path)
        rebuilt_item["segment_path"] = str(effective_segment_path)
        rebuilt_item["camera_plan_rebuild"] = boosted_camera_plan
        rebuilt_item["transition_plan_rebuild"] = boosted_transition
        return {
            "index": index,
            "segment_path": effective_segment_path,
            "transition_plan": boosted_transition,
            "rebuilt_item": rebuilt_item,
            "camera_applied": camera_applied,
        }

    segment_results: dict[int, dict[str, Any]] = {}
    camera_applied_count = 0
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="camera_seg") as executor:
        future_map = {
            executor.submit(_process_single_segment, index, item): index
            for index, item in enumerate(ordered_segment_items, start=1)
        }
        for future in as_completed(future_map):
            result = future.result()
            segment_results[result["index"]] = result
            if result["camera_applied"]:
                camera_applied_count += 1

    rebuilt_segment_paths: list[Path] = []
    rebuilt_transition_plans: list[dict[str, Any]] = []
    rebuilt_segment_items: list[dict[str, Any]] = []
    for index in sorted(segment_results):
        result = segment_results[index]
        rebuilt_segment_paths.append(result["segment_path"])
        rebuilt_transition_plans.append(result["transition_plan"])
        rebuilt_segment_items.append(result["rebuilt_item"])

    concat_result = _concat_segment_videos(
        segment_paths=rebuilt_segment_paths,
        concat_file_path=artifacts_dir / "segments_concat_ffmpeg_rebuild.txt",
        ffmpeg_bin=config.ffmpeg.ffmpeg_bin,
        ffprobe_bin=config.ffmpeg.ffprobe_bin,
        audio_path=audio_path,
        output_video_path=output_video_path,
        audio_duration=audio_duration,
        fps=config.ffmpeg.fps,
        video_codec=config.ffmpeg.video_codec,
        audio_codec=config.ffmpeg.audio_codec,
        video_preset=config.ffmpeg.video_preset,
        video_crf=config.ffmpeg.video_crf,
        video_accel_mode=config.ffmpeg.video_accel_mode,
        gpu_video_codec=config.ffmpeg.gpu_video_codec,
        gpu_preset=config.ffmpeg.gpu_preset,
        gpu_rc_mode=config.ffmpeg.gpu_rc_mode,
        gpu_cq=config.ffmpeg.gpu_cq,
        gpu_bitrate=config.ffmpeg.gpu_bitrate,
        concat_video_mode=config.ffmpeg.concat_video_mode,
        concat_copy_fallback_reencode=config.ffmpeg.concat_copy_fallback_reencode,
        transition_plans=rebuilt_transition_plans,
        logger=logger,
    )

    output_summary = {
        "task_id": str(module_d_output.get("task_id", task_dir.name)),
        "source_module_d_output_path": str(module_d_output_path),
        "output_video_path": str(output_video_path),
        "processed_segments_dir": str(processed_segments_dir),
        "camera_applied_count": int(camera_applied_count),
        "transition_duration_scale": float(transition_duration_scale),
        "min_transition_duration_ms": int(min_transition_duration_ms),
        "concat_mode": str(concat_result.get("mode", "unknown")),
        "copy_fallback_triggered": bool(concat_result.get("copy_fallback_triggered", False)),
        "segment_items": rebuilt_segment_items,
    }
    write_json(summary_output_path, output_summary)
    logger.info(
        "模块D纯FFmpeg重后处理完成，task_dir=%s，output=%s，camera_applied_count=%s",
        task_dir,
        output_video_path,
        camera_applied_count,
    )
    return output_summary


def _boost_camera_plan(camera_plan: Any, boost_small_to_medium: bool) -> dict[str, Any]:
    """
    功能说明：为纯 FFmpeg 重后处理构建增强版运镜计划。
    参数说明：
    - camera_plan: 原始运镜计划。
    - boost_small_to_medium: 是否将 small 提升为 medium。
    返回值：
    - dict[str, Any]: 可直接交给运镜后处理器的计划对象。
    异常说明：无。
    边界条件：非字典输入回退为空运镜。
    """
    if not isinstance(camera_plan, dict):
        return {"preset_id": "none", "mode": "none", "direction": "center", "strength": "none", "easing": "linear"}
    boosted = dict(camera_plan)
    if boost_small_to_medium and str(boosted.get("strength", "none")).strip().lower() == "small":
        boosted["strength"] = "medium"
    return boosted


def _boost_transition_plan(
    *,
    transition_plan: Any,
    duration_scale: float,
    min_duration_ms: int,
) -> dict[str, Any]:
    """
    功能说明：为纯 FFmpeg 重后处理构建增强版转场计划。
    参数说明：
    - transition_plan: 原始转场计划。
    - duration_scale: 时长放大倍率。
    - min_duration_ms: 非 none/hard_cut 转场的最小时长。
    返回值：
    - dict[str, Any]: 新转场计划。
    异常说明：无。
    边界条件：none/hard_cut 保持原样，不额外发明新转场类型。
    """
    if not isinstance(transition_plan, dict):
        return {"preset_id": "none", "kind": "none", "duration_ms": 0, "easing": "linear"}
    boosted = dict(transition_plan)
    kind = str(boosted.get("kind", "none")).strip().lower()
    if kind in {"none", "hard_cut"}:
        boosted["duration_ms"] = 0
        return boosted
    try:
        original_duration_ms = int(boosted.get("duration_ms", 0))
    except (TypeError, ValueError):
        original_duration_ms = 0
    boosted["duration_ms"] = max(int(min_duration_ms), int(round(float(original_duration_ms) * float(duration_scale))))
    return boosted


def _resolve_task_audio_path(task_dir: Path) -> Path:
    """
    功能说明：解析任务真实音频路径。
    参数说明：
    - task_dir: 任务目录。
    返回值：
    - Path: 音频文件路径。
    异常说明：无法定位时抛 RuntimeError。
    边界条件：优先相信模块A标准输出里的 audio_path，其次尝试任务目录常见文件名。
    """
    module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
    if module_a_output_path.exists():
        module_a_output = read_json(module_a_output_path)
        if isinstance(module_a_output, dict):
            audio_path_text = str(module_a_output.get("audio_path", "")).strip()
            if audio_path_text:
                audio_path = Path(audio_path_text).resolve()
                if audio_path.exists():
                    return audio_path
    for candidate_name in ["input_audio.wav", "input_audio.mp3", "audio.wav", "audio.mp3"]:
        candidate_path = (task_dir / candidate_name).resolve()
        if candidate_path.exists():
            return candidate_path
    raise RuntimeError(f"模块D重后处理失败：找不到任务音频，task_dir={task_dir}")
