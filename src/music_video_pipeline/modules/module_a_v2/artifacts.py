"""
文件用途：统一管理模块A V2中间产物路径与落盘操作。
核心流程：构建标准目录结构并提供 JSON 落盘工具。
输入输出：输入工作目录与产物内容，输出标准化路径与落盘结果。
依赖说明：依赖标准库 dataclasses/pathlib 与项目内 io_utils。
维护说明：新增中间产物时必须先在本文件补路径与落盘入口。
"""

# 标准库：用于数据结构定义
from dataclasses import dataclass
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：目录创建与JSON落盘
from music_video_pipeline.io_utils import ensure_dir, write_json


@dataclass(frozen=True)
class ModuleAV2Artifacts:
    """
    功能说明：封装模块A V2全部中间产物路径。
    参数说明：各字段均为标准化产物路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：路径对象仅描述目录，不保证文件已存在。
    """

    work_dir: Path
    perception_model_demucs_runtime_dir: Path
    perception_model_allin1_runtime_dir: Path
    perception_model_allin1_raw_response_path: Path
    perception_model_funasr_raw_response_path: Path
    perception_model_funasr_lyric_sentence_units_path: Path
    perception_model_funasr_sentence_split_stats_path: Path
    perception_signal_librosa_accompaniment_path: Path
    perception_signal_librosa_vocal_precheck_path: Path
    perception_signal_librosa_vocal_candidates_path: Path

    algorithm_window_stage_windows_raw_path: Path
    algorithm_window_stage_windows_classified_path: Path
    algorithm_window_stage_windows_merged_path: Path

    algorithm_timeline_stage_big_a0_path: Path
    algorithm_timeline_stage_big_a1_path: Path
    algorithm_timeline_stage_lyric_sentence_units_cleaned_path: Path
    algorithm_timeline_stage_small_timestamps_path: Path
    algorithm_timeline_stage_boundary_conflict_resolved_path: Path
    algorithm_timeline_stage_big_boundary_moves_path: Path

    algorithm_final_stage_segments_final_path: Path
    algorithm_final_stage_lyric_attached_path: Path
    algorithm_final_stage_energy_path: Path
    algorithm_final_analysis_data_path: Path


def build_module_a_v2_artifacts(work_dir: Path) -> ModuleAV2Artifacts:
    """
    功能说明：构建模块A V2产物目录树并返回路径对象。
    参数说明：
    - work_dir: 模块A V2工作目录（通常为 artifacts/module_a_work_v2）。
    返回值：
    - ModuleAV2Artifacts: 标准化路径对象。
    异常说明：目录无权限时抛 OSError。
    边界条件：重复调用保持幂等。
    """
    perception_model_demucs_runtime_dir = work_dir / "perception" / "model" / "demucs" / "runtime"
    perception_model_allin1_runtime_dir = work_dir / "perception" / "model" / "allin1" / "runtime"
    perception_model_allin1_dir = work_dir / "perception" / "model" / "allin1"
    perception_model_funasr_dir = work_dir / "perception" / "model" / "funasr"
    perception_signal_librosa_dir = work_dir / "perception" / "signal" / "librosa"

    algorithm_dir = work_dir / "algorithm"
    algorithm_window_dir = algorithm_dir / "window"
    algorithm_timeline_dir = algorithm_dir / "timeline"
    algorithm_final_dir = algorithm_dir / "final"

    for directory in [
        perception_model_demucs_runtime_dir,
        perception_model_allin1_runtime_dir,
        perception_model_allin1_dir,
        perception_model_funasr_dir,
        perception_signal_librosa_dir,
        algorithm_window_dir,
        algorithm_timeline_dir,
        algorithm_final_dir,
    ]:
        ensure_dir(directory)

    return ModuleAV2Artifacts(
        work_dir=work_dir,
        perception_model_demucs_runtime_dir=perception_model_demucs_runtime_dir,
        perception_model_allin1_runtime_dir=perception_model_allin1_runtime_dir,
        perception_model_allin1_raw_response_path=perception_model_allin1_dir / "allin1_raw_response.json",
        perception_model_funasr_raw_response_path=perception_model_funasr_dir / "funasr_raw_response.json",
        perception_model_funasr_lyric_sentence_units_path=perception_model_funasr_dir / "lyric_sentence_units.json",
        perception_model_funasr_sentence_split_stats_path=perception_model_funasr_dir / "sentence_split_stats.json",
        perception_signal_librosa_accompaniment_path=perception_signal_librosa_dir / "accompaniment_candidates.json",
        perception_signal_librosa_vocal_precheck_path=perception_signal_librosa_dir / "vocal_precheck_rms.json",
        perception_signal_librosa_vocal_candidates_path=perception_signal_librosa_dir / "vocal_candidates.json",

        algorithm_window_stage_windows_raw_path=algorithm_window_dir / "stage_windows_raw.json",
        algorithm_window_stage_windows_classified_path=algorithm_window_dir / "stage_windows_classified.json",
        algorithm_window_stage_windows_merged_path=algorithm_window_dir / "stage_windows_merged.json",

        algorithm_timeline_stage_big_a0_path=algorithm_timeline_dir / "stage_big_a0.json",
        algorithm_timeline_stage_big_a1_path=algorithm_timeline_dir / "stage_big_a1.json",
        algorithm_timeline_stage_lyric_sentence_units_cleaned_path=algorithm_timeline_dir
        / "stage_lyric_sentence_units_cleaned.json",
        algorithm_timeline_stage_small_timestamps_path=algorithm_timeline_dir / "stage_small_timestamps.json",
        algorithm_timeline_stage_boundary_conflict_resolved_path=algorithm_timeline_dir / "stage_boundary_conflict_resolved.json",
        algorithm_timeline_stage_big_boundary_moves_path=algorithm_timeline_dir / "stage_big_boundary_moves.json",

        algorithm_final_stage_segments_final_path=algorithm_final_dir / "stage_segments_final.json",
        algorithm_final_stage_lyric_attached_path=algorithm_final_dir / "stage_lyric_attached.json",
        algorithm_final_stage_energy_path=algorithm_final_dir / "stage_energy.json",
        algorithm_final_analysis_data_path=algorithm_final_dir / "final_analysis_data.json",
    )


def dump_json_artifact(output_path: Path, payload: Any, logger, artifact_name: str) -> None:
    """
    功能说明：以统一日志口径落盘 JSON 产物。
    参数说明：
    - output_path: 输出路径。
    - payload: 待写入对象。
    - logger: 日志记录器。
    - artifact_name: 产物名称（用于日志可读性）。
    返回值：无。
    异常说明：写入失败抛错，由上层统一处理。
    边界条件：自动创建父目录。
    """
    write_json(output_path, payload)
    logger.debug("模块A V2产物已写入，name=%s，path=%s", artifact_name, output_path)
