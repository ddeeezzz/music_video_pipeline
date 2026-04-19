"""
文件用途：按白名单规则构建上传 staging 目录。
核心流程：收集白名单文件 -> 复制到临时目录 -> 生成 manifest。
输入输出：输入 task_dir/task_id/profile，输出上传目录路径与纳入文件列表。
依赖说明：依赖标准库 fnmatch/pathlib/shutil/tempfile。
维护说明：白名单策略变更应仅修改本文件，避免 worker/runner 膨胀。
"""

# 标准库：用于通配匹配任务目录根文件
import fnmatch
# 标准库：用于文件复制
import shutil
# 标准库：用于临时目录创建
import tempfile
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于日志类型
import logging

# 常量：白名单策略名（当前仅实现 v1）。
UPLOAD_SELECTION_PROFILE_WHITELIST_V1 = "whitelist_v1"
# 常量：模块级白名单策略名（按模块完成后上传）。
UPLOAD_SELECTION_PROFILE_MODULE_A_V1 = "module_a_whitelist_v1"
UPLOAD_SELECTION_PROFILE_MODULE_B_V1 = "module_b_whitelist_v1"
UPLOAD_SELECTION_PROFILE_MODULE_C_V1 = "module_c_whitelist_v1"
UPLOAD_SELECTION_PROFILE_MODULE_D_V1 = "module_d_whitelist_v1"


def _iter_all_files(directory_path: Path) -> list[Path]:
    """
    功能说明：递归收集目录下全部文件。
    参数说明：
    - directory_path: 目录路径。
    返回值：
    - list[Path]: 文件路径数组。
    异常说明：无。
    边界条件：目录不存在时返回空数组。
    """
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    return [file_path for file_path in directory_path.rglob("*") if file_path.is_file()]


def _collect_whitelist_files_v1(task_dir: Path) -> list[Path]:
    """
    功能说明：按 whitelist_v1 规则收集任务目录内需上传文件。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """
    exact_rel_files = [
        "final_output.mp4",
        "task_monitor.html",
        "artifacts/module_a_output.json",
        "artifacts/module_b_output.json",
        "artifacts/module_c_output.json",
        "artifacts/module_d_output.json",
        "artifacts/module_a_work_v2/perception/model/allin1/allin1_raw_response.json",
        "artifacts/module_a_work_v2/perception/model/funasr/funasr_raw_response.json",
        "artifacts/module_a_work_v2/perception/model/funasr/lyric_sentence_units.json",
        "artifacts/module_a_work_v2/perception/model/funasr/sentence_split_stats.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/vocal_precheck_rms.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/accompaniment_candidates.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/vocal_candidates.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_segments_final.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_energy.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_lyric_attached.json",
        "artifacts/module_a_work_v2/algorithm/final/final_analysis_data.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_big_a0.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_big_a1.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_lyric_sentence_units_cleaned.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_lyric_sentence_units_head_refined.json",
        "artifacts/module_a_work_v2/algorithm/window/stage_windows_classified.json",
        "artifacts/module_a_work_v2/algorithm/window/stage_windows_merged.json",
    ]
    include_dirs = [
        "artifacts/frames",
        "artifacts/segments",
        "artifacts/module_b_units",
        "log",
    ]
    root_glob_patterns = [
        "*_module_a_v2_visualization.html",
        "*_module_a_v2_visualization_audio.*",
    ]
    return _collect_by_rules(
        task_dir=task_dir,
        exact_rel_files=exact_rel_files,
        include_dirs=include_dirs,
        root_glob_patterns=root_glob_patterns,
        include_demucs_tracks=True,
    )


def _collect_module_a_whitelist_files_v1(task_dir: Path) -> list[Path]:
    """
    功能说明：按 module_a_whitelist_v1 收集模块 A 完成后的关键产物。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """
    exact_rel_files = [
        "task_monitor.html",
        "artifacts/module_a_output.json",
        "artifacts/module_a_work_v2/perception/model/allin1/allin1_raw_response.json",
        "artifacts/module_a_work_v2/perception/model/funasr/funasr_raw_response.json",
        "artifacts/module_a_work_v2/perception/model/funasr/lyric_sentence_units.json",
        "artifacts/module_a_work_v2/perception/model/funasr/sentence_split_stats.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/vocal_precheck_rms.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/accompaniment_candidates.json",
        "artifacts/module_a_work_v2/perception/signal/librosa/vocal_candidates.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_segments_final.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_energy.json",
        "artifacts/module_a_work_v2/algorithm/final/stage_lyric_attached.json",
        "artifacts/module_a_work_v2/algorithm/final/final_analysis_data.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_big_a0.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_big_a1.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_lyric_sentence_units_cleaned.json",
        "artifacts/module_a_work_v2/algorithm/timeline/stage_lyric_sentence_units_head_refined.json",
        "artifacts/module_a_work_v2/algorithm/window/stage_windows_classified.json",
        "artifacts/module_a_work_v2/algorithm/window/stage_windows_merged.json",
    ]
    include_dirs = ["log"]
    root_glob_patterns = [
        "*_module_a_v2_visualization.html",
        "*_module_a_v2_visualization_audio.*",
    ]
    return _collect_by_rules(
        task_dir=task_dir,
        exact_rel_files=exact_rel_files,
        include_dirs=include_dirs,
        root_glob_patterns=root_glob_patterns,
        include_demucs_tracks=True,
    )


def _collect_module_b_whitelist_files_v1(task_dir: Path) -> list[Path]:
    """
    功能说明：按 module_b_whitelist_v1 收集模块 B 完成后的关键产物。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """
    exact_rel_files = [
        "task_monitor.html",
        "artifacts/module_a_output.json",
        "artifacts/module_b_output.json",
    ]
    include_dirs = [
        "artifacts/module_b_units",
        "log",
    ]
    return _collect_by_rules(
        task_dir=task_dir,
        exact_rel_files=exact_rel_files,
        include_dirs=include_dirs,
        root_glob_patterns=[],
        include_demucs_tracks=False,
    )


def _collect_module_c_whitelist_files_v1(task_dir: Path) -> list[Path]:
    """
    功能说明：按 module_c_whitelist_v1 收集模块 C 完成后的关键产物。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """
    exact_rel_files = [
        "task_monitor.html",
        "artifacts/module_c_output.json",
    ]
    include_dirs = [
        "artifacts/frames",
        "log",
    ]
    return _collect_by_rules(
        task_dir=task_dir,
        exact_rel_files=exact_rel_files,
        include_dirs=include_dirs,
        root_glob_patterns=[],
        include_demucs_tracks=False,
    )


def _collect_module_d_whitelist_files_v1(task_dir: Path) -> list[Path]:
    """
    功能说明：按 module_d_whitelist_v1 收集模块 D 完成后的关键产物。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """
    exact_rel_files = [
        "task_monitor.html",
        "final_output.mp4",
        "artifacts/module_d_output.json",
        "artifacts/segments_concat.txt",
    ]
    include_dirs = [
        "artifacts/segments",
        "log",
    ]
    return _collect_by_rules(
        task_dir=task_dir,
        exact_rel_files=exact_rel_files,
        include_dirs=include_dirs,
        root_glob_patterns=[],
        include_demucs_tracks=False,
    )


def _collect_by_rules(
    *,
    task_dir: Path,
    exact_rel_files: list[str],
    include_dirs: list[str],
    root_glob_patterns: list[str],
    include_demucs_tracks: bool,
) -> list[Path]:
    """
    功能说明：按通用规则收集白名单文件并去重排序。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    - exact_rel_files: 需精确纳入的相对路径列表。
    - include_dirs: 需递归纳入的目录相对路径列表。
    - root_glob_patterns: 任务根目录文件匹配模式。
    - include_demucs_tracks: 是否纳入 demucs 关键二轨结果。
    返回值：
    - list[Path]: 需上传文件绝对路径数组（已去重并排序）。
    异常说明：无。
    边界条件：缺失文件按“存在即纳入”策略处理，不报错。
    """

    selected_files: set[Path] = set()
    for rel_path_text in exact_rel_files:
        candidate = task_dir / rel_path_text
        if candidate.exists() and candidate.is_file():
            selected_files.add(candidate.resolve())

    for rel_dir_text in include_dirs:
        selected_files.update(path.resolve() for path in _iter_all_files(task_dir / rel_dir_text))

    # 仅纳入 Demucs 关键二轨结果，避免上传 runtime 噪声目录。
    if include_demucs_tracks:
        demucs_runtime_dir = task_dir / "artifacts/module_a_work_v2/perception/model/demucs/runtime"
        if demucs_runtime_dir.exists():
            for wave_name in ("vocals.wav", "no_vocals.wav"):
                for path in demucs_runtime_dir.rglob(wave_name):
                    if path.is_file():
                        selected_files.add(path.resolve())

    for path in task_dir.iterdir() if task_dir.exists() else []:
        if not path.is_file():
            continue
        for pattern in root_glob_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                selected_files.add(path.resolve())
                break

    return sorted(selected_files, key=lambda path: str(path.relative_to(task_dir)))


def _copy_file_to_staging(*, source_file: Path, task_dir: Path, staging_dir: Path) -> Path:
    """
    功能说明：将单文件复制到 staging 目录并保持 task_dir 相对路径结构。
    参数说明：
    - source_file: 源文件绝对路径。
    - task_dir: 任务目录根路径。
    - staging_dir: staging 目录根路径。
    返回值：
    - Path: 复制后的目标文件路径。
    异常说明：
    - RuntimeError: 源文件不在 task_dir 下时抛错。
    边界条件：父目录不存在时自动创建。
    """
    try:
        relative_path = source_file.relative_to(task_dir)
    except ValueError as error:
        raise RuntimeError(f"上传白名单复制失败：文件不在任务目录内，path={source_file}") from error
    target_path = staging_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_path)
    return target_path


def build_whitelist_staging_dir(
    *,
    task_dir: Path,
    task_id: str,
    selection_profile: str,
    logger: logging.Logger,
) -> tuple[Path, list[Path]]:
    """
    功能说明：按白名单策略构建本次上传 staging 目录。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    - task_id: 任务唯一标识。
    - selection_profile: 白名单策略名。
    - logger: 日志对象。
    返回值：
    - tuple[Path, list[Path]]: (staging目录路径, 纳入文件绝对路径列表)。
    异常说明：
    - RuntimeError: 选择未知 profile 或任务目录不存在时抛错。
    边界条件：无匹配文件时会返回空 staging（仍可上传，便于远端对齐）。
    """
    if not task_dir.exists() or not task_dir.is_dir():
        raise RuntimeError(f"上传白名单构建失败：任务目录不存在，task_id={task_id}，task_dir={task_dir}")
    normalized_profile = str(selection_profile).strip() or UPLOAD_SELECTION_PROFILE_WHITELIST_V1
    collector_map = {
        UPLOAD_SELECTION_PROFILE_WHITELIST_V1: _collect_whitelist_files_v1,
        UPLOAD_SELECTION_PROFILE_MODULE_A_V1: _collect_module_a_whitelist_files_v1,
        UPLOAD_SELECTION_PROFILE_MODULE_B_V1: _collect_module_b_whitelist_files_v1,
        UPLOAD_SELECTION_PROFILE_MODULE_C_V1: _collect_module_c_whitelist_files_v1,
        UPLOAD_SELECTION_PROFILE_MODULE_D_V1: _collect_module_d_whitelist_files_v1,
    }
    collector = collector_map.get(normalized_profile)
    if collector is None:
        raise RuntimeError(f"上传白名单构建失败：不支持的 selection_profile={normalized_profile}")

    selected_files = collector(task_dir=task_dir)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"bypy_stage_{task_id}_"))
    for source_file in selected_files:
        _copy_file_to_staging(source_file=source_file, task_dir=task_dir, staging_dir=staging_dir)

    manifest_path = staging_dir / "_upload_manifest.txt"
    manifest_lines = [
        f"task_id={task_id}",
        f"selection_profile={normalized_profile}",
        f"file_count={len(selected_files)}",
        "",
    ]
    manifest_lines.extend(str(path.relative_to(task_dir)) for path in selected_files)
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    logger.info(
        "上传白名单 staging 已生成，task_id=%s，selection_profile=%s，file_count=%s，staging=%s",
        task_id,
        normalized_profile,
        len(selected_files),
        staging_dir,
    )
    return staging_dir, selected_files
