"""
文件用途：执行模块A真实音频对比跑数，并输出可复现的统计结果。
核心流程：加载配置并构造跑数场景，逐音频执行 run-module A，汇总耗时与关键产物计数。
输入输出：输入配置与音频列表，输出控制台对比表及 JSON/CSV 明细文件。
依赖说明：依赖项目内 PipelineRunner/config/io_utils 与 module_a 命名空间函数。
维护说明：本脚本仅做跑数与对比，不修改生产链路实现与契约字段。
"""

from __future__ import annotations

# 标准库：命令行参数解析
import argparse
# 标准库：上下文管理器
from contextlib import contextmanager, nullcontext
# 标准库：CSV 写入
import csv
# 标准库：时间戳与计时
from datetime import datetime
# 标准库：数据类声明
from dataclasses import asdict, dataclass
# 标准库：JSON 处理
import json
# 标准库：路径处理
from pathlib import Path
# 标准库：统计计算
from statistics import mean
# 标准库：休眠控制
import time
# 标准库：类型标注
from typing import Callable, Iterator

# 项目内模块：配置加载
from music_video_pipeline.config import load_config
# 项目内模块：JSON 读写工具
from music_video_pipeline.io_utils import ensure_dir, read_json, write_json
# 项目内模块：日志初始化
from music_video_pipeline.logging_utils import setup_logging
# 项目内模块：模块A命名空间（用于受控串行补丁）
from music_video_pipeline.modules import module_a as module_a_impl
# 项目内模块：流水线调度器
from music_video_pipeline.pipeline import PipelineRunner


@dataclass(frozen=True)
class CompareScenario:
    """
    功能说明：定义单个对比场景及其可选补丁上下文。
    参数说明：
    - key: 场景短标识。
    - name: 场景显示名称。
    - patch_factory: 运行前挂载的上下文工厂（可为空操作）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：patch_factory 需保证退出时恢复现场。
    """

    key: str
    name: str
    patch_factory: Callable[[], Iterator[None]]


@dataclass
class RunRecord:
    """
    功能说明：保存单次跑数结果。
    参数说明：
    - task_id: 任务 ID。
    - audio_name: 音频显示名。
    - audio_path: 音频绝对路径字符串。
    - scenario_key: 场景短标识。
    - scenario_name: 场景名称。
    - repeat_index: 当前场景第几次重复。
    - success: 是否执行成功。
    - elapsed_seconds: 本次耗时（秒）。
    - big_segment_count: 大段数量。
    - segment_count: 小段数量。
    - beat_count: 节拍数量。
    - lyric_unit_count: 歌词单元数量。
    - energy_feature_count: 能量特征数量。
    - output_json_path: 模块A输出路径。
    - log_path: 任务日志路径。
    - error_message: 失败信息。
    - finished_at: 完成时间戳。
    返回值：不适用。
    异常说明：不适用。
    边界条件：失败时计数字段为 0。
    """

    task_id: str
    audio_name: str
    audio_path: str
    scenario_key: str
    scenario_name: str
    repeat_index: int
    success: bool
    elapsed_seconds: float
    big_segment_count: int
    segment_count: int
    beat_count: int
    lyric_unit_count: int
    energy_feature_count: int
    output_json_path: str
    log_path: str
    error_message: str
    finished_at: str


def _sanitize_token(text: str) -> str:
    """
    功能说明：将任意字符串清洗为文件名安全 token。
    参数说明：
    - text: 原始字符串。
    返回值：
    - str: 仅包含字母数字下划线的 token。
    异常说明：无。
    边界条件：清洗后为空时返回 fallback。
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()
    return cleaned or "item"


@contextmanager
def _force_serial_mode_patch() -> Iterator[None]:
    """
    功能说明：通过受控补丁关闭 allin1fix Demucs 复用路径，构造串行基线。
    参数说明：无。
    返回值：
    - Iterator[None]: 上下文对象。
    异常说明：异常由调用方处理。
    边界条件：退出上下文必须恢复原函数，避免污染后续场景。
    """
    original_prepare = module_a_impl._prepare_stems_with_allin1_demucs

    def _raise_force_serial(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("跑数脚受控串行基线：禁用 allin1fix Demucs 复用路径")

    module_a_impl._prepare_stems_with_allin1_demucs = _raise_force_serial
    try:
        yield
    finally:
        module_a_impl._prepare_stems_with_allin1_demucs = original_prepare


def _build_default_audio_list(workspace_root: Path) -> list[Path]:
    """
    功能说明：返回仓库内可用的默认真实音频列表。
    参数说明：
    - workspace_root: 项目根目录。
    返回值：
    - list[Path]: 已存在的默认音频路径列表。
    异常说明：无。
    边界条件：候选不存在时自动跳过。
    """
    candidates = [
        workspace_root / "resources" / "juebieshu.mp3",
        workspace_root / "resources" / "jieranduhuo.mp3",
        workspace_root / "resources" / "wuli.m4a",
    ]
    return [path.resolve() for path in candidates if path.exists()]


def _resolve_audio_list(workspace_root: Path, audio_args: list[str]) -> list[Path]:
    """
    功能说明：解析命令行音频列表，未提供时回退默认样本组。
    参数说明：
    - workspace_root: 项目根目录。
    - audio_args: 命令行传入音频路径字符串。
    返回值：
    - list[Path]: 解析后的绝对路径列表。
    异常说明：
    - FileNotFoundError: 解析后存在缺失文件。
    边界条件：默认样本为空时抛错并提示用户传参。
    """
    if audio_args:
        resolved_list: list[Path] = []
        for item in audio_args:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = (workspace_root / candidate).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"音频不存在：{candidate}")
            resolved_list.append(candidate)
        return resolved_list

    defaults = _build_default_audio_list(workspace_root)
    if not defaults:
        raise FileNotFoundError("未发现默认真实音频，请通过 --audio 显式传入至少1个音频路径")
    return defaults


def _write_compare_config(
    base_config_path: Path,
    runset_dir: Path,
    runs_dir_for_tasks: Path,
    module_a_mode: str,
) -> Path:
    """
    功能说明：基于基础配置写入本次跑数专用配置文件。
    参数说明：
    - base_config_path: 基础配置路径。
    - runset_dir: 本次跑数输出目录。
    - runs_dir_for_tasks: 跑数任务目录根路径。
    - module_a_mode: 模块A模式覆盖值。
    返回值：
    - Path: 专用配置文件路径。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅覆盖跑数相关字段，不扩展其他行为。
    """
    with base_config_path.open("r", encoding="utf-8-sig") as file_obj:
        raw_data = json.load(file_obj)

    raw_data.setdefault("paths", {})
    raw_data["paths"]["runs_dir"] = str(runs_dir_for_tasks.resolve())
    raw_data.setdefault("module_a", {})
    raw_data["module_a"]["mode"] = module_a_mode

    config_path = runset_dir / "_module_a_real_compare_config.json"
    write_json(config_path, raw_data)
    return config_path


def _collect_output_counts(module_a_output_path: Path) -> tuple[int, int, int, int, int]:
    """
    功能说明：读取模块A输出并统计关键列表字段长度。
    参数说明：
    - module_a_output_path: 模块A输出 JSON 路径。
    返回值：
    - tuple[int, int, int, int, int]: 大段/小段/节拍/歌词/能量数量。
    异常说明：读取失败时抛异常，由上层改写为失败信息。
    边界条件：字段缺失时按 0 处理。
    """
    output_data = read_json(module_a_output_path)
    return (
        len(output_data.get("big_segments", [])),
        len(output_data.get("segments", [])),
        len(output_data.get("beats", [])),
        len(output_data.get("lyric_units", [])),
        len(output_data.get("energy_features", [])),
    )


def _find_latest_task_log(task_dir: Path) -> Path | None:
    """
    功能说明：定位单任务目录下最新 run-module A 日志文件。
    参数说明：
    - task_dir: 任务目录。
    返回值：
    - Path | None: 最新日志路径；不存在时返回 None。
    异常说明：无。
    边界条件：同一任务可能存在多个日志文件，按文件名排序取最后一个。
    """
    log_dir = task_dir / "log"
    log_files = sorted(log_dir.glob("run_module_a_*.log")) if log_dir.exists() else []
    return log_files[-1] if log_files else None


def _run_single_measurement(
    runner: PipelineRunner,
    config_path: Path,
    task_id: str,
    audio_path: Path,
    scenario: CompareScenario,
    repeat_index: int,
    force: bool,
) -> RunRecord:
    """
    功能说明：执行一次模块A跑数并生成记录对象。
    参数说明：
    - runner: 流水线调度器。
    - config_path: 配置路径（用于状态记录）。
    - task_id: 任务标识。
    - audio_path: 音频路径。
    - scenario: 对比场景定义。
    - repeat_index: 重复轮次（从1开始）。
    - force: 是否强制重置模块状态后执行。
    返回值：
    - RunRecord: 单次执行记录。
    异常说明：异常会被捕获并写入 error_message。
    边界条件：失败时仍输出记录，便于汇总失败率。
    """
    start_time = time.perf_counter()
    success = False
    error_message = ""
    big_count = 0
    seg_count = 0
    beat_count = 0
    lyric_count = 0
    energy_count = 0

    with scenario.patch_factory():
        try:
            runner.run_single_module(
                task_id=task_id,
                module_name="A",
                config_path=config_path,
                audio_path=audio_path,
                force=force,
            )
            success = True
        except Exception as error:  # noqa: BLE001
            error_message = str(error)

    elapsed_seconds = time.perf_counter() - start_time
    task_dir = runner.runs_dir / task_id
    module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
    log_path = _find_latest_task_log(task_dir)

    if success:
        if module_a_output_path.exists():
            try:
                big_count, seg_count, beat_count, lyric_count, energy_count = _collect_output_counts(module_a_output_path)
            except Exception as error:  # noqa: BLE001
                success = False
                error_message = f"读取 module_a_output.json 失败：{error}"
        else:
            success = False
            error_message = "run-module 返回成功，但未发现 module_a_output.json"

    return RunRecord(
        task_id=task_id,
        audio_name=audio_path.stem,
        audio_path=str(audio_path),
        scenario_key=scenario.key,
        scenario_name=scenario.name,
        repeat_index=repeat_index,
        success=success,
        elapsed_seconds=elapsed_seconds,
        big_segment_count=big_count,
        segment_count=seg_count,
        beat_count=beat_count,
        lyric_unit_count=lyric_count,
        energy_feature_count=energy_count,
        output_json_path=str(module_a_output_path) if module_a_output_path.exists() else "",
        log_path=str(log_path) if log_path is not None else "",
        error_message=error_message,
        finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _write_records_csv(path: Path, records: list[RunRecord]) -> None:
    """
    功能说明：将跑数明细写为 CSV。
    参数说明：
    - path: CSV 输出路径。
    - records: 明细记录列表。
    返回值：无。
    异常说明：异常由调用方处理。
    边界条件：无记录时仅写表头。
    """
    ensure_dir(path.parent)
    fieldnames = list(asdict(records[0]).keys()) if records else list(RunRecord.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            writer.writerow(asdict(item))


def _build_summary(records: list[RunRecord]) -> list[dict[str, object]]:
    """
    功能说明：按“音频+场景”汇总平均耗时与产物计数。
    参数说明：
    - records: 明细记录列表。
    返回值：
    - list[dict[str, object]]: 汇总行列表。
    异常说明：无。
    边界条件：仅统计成功记录；失败组保留失败计数。
    """
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for item in records:
        key = (item.audio_name, item.scenario_key)
        grouped.setdefault(key, []).append(item)

    summary_rows: list[dict[str, object]] = []
    for (audio_name, scenario_key), group_items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        success_items = [row for row in group_items if row.success]
        elapsed_values = [row.elapsed_seconds for row in success_items]
        summary_rows.append(
            {
                "audio_name": audio_name,
                "scenario_key": scenario_key,
                "scenario_name": group_items[0].scenario_name,
                "run_count": len(group_items),
                "success_count": len(success_items),
                "failure_count": len(group_items) - len(success_items),
                "avg_elapsed_seconds": round(mean(elapsed_values), 4) if elapsed_values else None,
                "min_elapsed_seconds": round(min(elapsed_values), 4) if elapsed_values else None,
                "max_elapsed_seconds": round(max(elapsed_values), 4) if elapsed_values else None,
                "avg_big_segments": round(mean([row.big_segment_count for row in success_items]), 2) if success_items else None,
                "avg_segments": round(mean([row.segment_count for row in success_items]), 2) if success_items else None,
                "avg_beats": round(mean([row.beat_count for row in success_items]), 2) if success_items else None,
                "avg_lyric_units": round(mean([row.lyric_unit_count for row in success_items]), 2) if success_items else None,
            }
        )
    return summary_rows


def _print_records(records: list[RunRecord]) -> None:
    """
    功能说明：输出单次运行明细表。
    参数说明：
    - records: 明细记录列表。
    返回值：无。
    异常说明：无。
    边界条件：空列表时输出提示。
    """
    if not records:
        print("未产生任何跑数记录。")
        return

    print("\n[单次运行明细]")
    print("audio | scenario | repeat | success | elapsed_s | big | seg | beat | lyric | log")
    for row in records:
        print(
            f"{row.audio_name} | {row.scenario_key} | {row.repeat_index} | "
            f"{'Y' if row.success else 'N'} | {row.elapsed_seconds:.3f} | "
            f"{row.big_segment_count} | {row.segment_count} | {row.beat_count} | {row.lyric_unit_count} | {row.log_path}"
        )
        if row.error_message:
            print(f"  错误：{row.error_message}")


def _print_summary(summary_rows: list[dict[str, object]]) -> None:
    """
    功能说明：输出汇总表。
    参数说明：
    - summary_rows: 汇总行列表。
    返回值：无。
    异常说明：无。
    边界条件：空列表时输出提示。
    """
    if not summary_rows:
        print("\n[汇总] 无可展示结果。")
        return

    print("\n[汇总：音频+场景]")
    print("audio | scenario | ok/total | avg_s | min_s | max_s | avg_big | avg_seg | avg_beat | avg_lyric")
    for row in summary_rows:
        print(
            f"{row['audio_name']} | {row['scenario_key']} | {row['success_count']}/{row['run_count']} | "
            f"{row['avg_elapsed_seconds']} | {row['min_elapsed_seconds']} | {row['max_elapsed_seconds']} | "
            f"{row['avg_big_segments']} | {row['avg_segments']} | {row['avg_beats']} | {row['avg_lyric_units']}"
        )


def _build_speedup_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    功能说明：计算 parallel 相对 serial_baseline 的加速比。
    参数说明：
    - summary_rows: 汇总行列表。
    返回值：
    - list[dict[str, object]]: 加速比行列表。
    异常说明：无。
    边界条件：缺少任一场景或耗时为空时跳过该音频。
    """
    by_audio: dict[str, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        audio_name = str(row["audio_name"])
        scenario_key = str(row["scenario_key"])
        by_audio.setdefault(audio_name, {})[scenario_key] = row

    speed_rows: list[dict[str, object]] = []
    for audio_name, scenario_map in sorted(by_audio.items(), key=lambda item: item[0]):
        parallel_row = scenario_map.get("parallel")
        serial_row = scenario_map.get("serial_baseline")
        if not parallel_row or not serial_row:
            continue
        parallel_avg = parallel_row.get("avg_elapsed_seconds")
        serial_avg = serial_row.get("avg_elapsed_seconds")
        if parallel_avg is None or serial_avg is None:
            continue
        parallel_value = float(parallel_avg)
        serial_value = float(serial_avg)
        if serial_value <= 0:
            continue
        speed_rows.append(
            {
                "audio_name": audio_name,
                "parallel_avg_seconds": round(parallel_value, 4),
                "serial_avg_seconds": round(serial_value, 4),
                "parallel_vs_serial_ratio": round(parallel_value / serial_value, 4),
                "parallel_speedup_percent": round((serial_value - parallel_value) / serial_value * 100.0, 2),
            }
        )
    return speed_rows


def _print_speedup(speed_rows: list[dict[str, object]]) -> None:
    """
    功能说明：打印并行相对串行的速度对比。
    参数说明：
    - speed_rows: 加速比行列表。
    返回值：无。
    异常说明：无。
    边界条件：无可比数据时输出提示。
    """
    if not speed_rows:
        print("\n[并行相对串行] 当前无可计算的加速比数据。")
        return
    print("\n[并行相对串行]")
    print("audio | parallel_s | serial_s | ratio(parallel/serial) | speedup_%")
    for row in speed_rows:
        print(
            f"{row['audio_name']} | {row['parallel_avg_seconds']} | {row['serial_avg_seconds']} | "
            f"{row['parallel_vs_serial_ratio']} | {row['parallel_speedup_percent']}"
        )


def _build_scenarios(selected_keys: list[str]) -> list[CompareScenario]:
    """
    功能说明：构建用户选择的对比场景集合。
    参数说明：
    - selected_keys: 需要启用的场景键列表。
    返回值：
    - list[CompareScenario]: 场景对象列表。
    异常说明：无。
    边界条件：未知场景键会被忽略。
    """
    all_scenarios: dict[str, CompareScenario] = {
        "parallel": CompareScenario(
            key="parallel",
            name="当前并行主链",
            patch_factory=lambda: nullcontext(),
        ),
        "serial_baseline": CompareScenario(
            key="serial_baseline",
            name="受控串行基线（禁用allin1fix Demucs复用）",
            patch_factory=_force_serial_mode_patch,
        ),
    }
    return [all_scenarios[key] for key in selected_keys if key in all_scenarios]


def parse_args() -> argparse.Namespace:
    """
    功能说明：解析命令行参数。
    参数说明：无。
    返回值：
    - argparse.Namespace: 参数对象。
    异常说明：参数非法时由 argparse 自动退出。
    边界条件：--audio 可重复传入多个真实音频。
    """
    parser = argparse.ArgumentParser(description="模块A真实音频对比跑数脚本（文件名以下划线开头）")
    parser.add_argument("--config", default="configs/default.json", help="基础配置路径")
    parser.add_argument("--audio", action="append", default=[], help="真实音频路径，可重复传入")
    parser.add_argument("--runs-per-case", type=int, default=1, help="每个音频+场景的重复次数")
    parser.add_argument("--module-a-mode", default="real_auto", choices=["real_auto", "real_strict", "fallback_only"], help="模块A模式")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["parallel", "serial_baseline"],
        choices=["parallel", "serial_baseline"],
        help="启用的对比场景",
    )
    parser.add_argument("--task-prefix", default="real_audio_cmp", help="任务ID前缀")
    parser.add_argument("--force", action="store_true", help="run-module 时是否启用 force")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="两次执行间隔秒数")
    parser.add_argument("--output-dir", default="runs/benchmarks/module_a_real_audio_compare", help="跑数结果输出目录")
    return parser.parse_args()


def main() -> None:
    """
    功能说明：脚本主入口，执行真实音频对比跑数并落盘结果。
    参数说明：无（读取命令行参数）。
    返回值：无。
    异常说明：异常向上抛出，便于命令行直接看到失败原因。
    边界条件：仅执行模块A，不触发 B/C/D。
    """
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    base_config_path = Path(args.config)
    if not base_config_path.is_absolute():
        base_config_path = (workspace_root / base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{base_config_path}")

    audio_list = _resolve_audio_list(workspace_root=workspace_root, audio_args=args.audio)
    scenarios = _build_scenarios(selected_keys=args.scenarios)
    if not scenarios:
        raise RuntimeError("未启用任何有效场景，请检查 --scenarios 参数。")
    if args.runs_per_case < 1:
        raise RuntimeError("--runs-per-case 必须 >= 1")

    runset_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (workspace_root / output_root).resolve()
    runset_dir = output_root / runset_id
    ensure_dir(runset_dir)
    runs_dir_for_tasks = runset_dir / "tasks"
    ensure_dir(runs_dir_for_tasks)

    compare_config_path = _write_compare_config(
        base_config_path=base_config_path,
        runset_dir=runset_dir,
        runs_dir_for_tasks=runs_dir_for_tasks,
        module_a_mode=args.module_a_mode,
    )
    app_config = load_config(compare_config_path)
    logger = setup_logging(level=app_config.logging.level)
    runner = PipelineRunner(workspace_root=workspace_root, config=app_config, logger=logger)

    print("开始执行模块A真实音频对比跑数")
    print(f"基础配置：{base_config_path}")
    print(f"跑数配置：{compare_config_path}")
    print(f"任务输出目录：{runs_dir_for_tasks}")
    print(f"音频数量：{len(audio_list)}，场景数量：{len(scenarios)}，每组重复：{args.runs_per_case}")

    records: list[RunRecord] = []
    run_counter = 0
    total_runs = len(audio_list) * len(scenarios) * args.runs_per_case
    for audio_path in audio_list:
        audio_token = _sanitize_token(audio_path.stem)
        for scenario in scenarios:
            scenario_token = _sanitize_token(scenario.key)
            for repeat_index in range(1, args.runs_per_case + 1):
                run_counter += 1
                task_id = f"{_sanitize_token(args.task_prefix)}_{runset_id}_{audio_token}_{scenario_token}_r{repeat_index:02d}"
                print(f"\n[{run_counter}/{total_runs}] 执行中：audio={audio_path.name}，scenario={scenario.key}，repeat={repeat_index}")
                record = _run_single_measurement(
                    runner=runner,
                    config_path=compare_config_path,
                    task_id=task_id,
                    audio_path=audio_path,
                    scenario=scenario,
                    repeat_index=repeat_index,
                    force=bool(args.force),
                )
                records.append(record)
                status_text = "成功" if record.success else "失败"
                print(f"结果：{status_text}，耗时={record.elapsed_seconds:.3f}s，task_id={task_id}")
                if record.error_message:
                    print(f"失败原因：{record.error_message}")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    summary_rows = _build_summary(records)
    speed_rows = _build_speedup_rows(summary_rows)

    records_path = runset_dir / "records.json"
    records_csv_path = runset_dir / "records.csv"
    summary_path = runset_dir / "summary.json"
    speed_path = runset_dir / "speedup.json"
    metadata_path = runset_dir / "metadata.json"

    write_json(records_path, [asdict(row) for row in records])
    _write_records_csv(records_csv_path, records)
    write_json(summary_path, summary_rows)
    write_json(speed_path, speed_rows)
    write_json(
        metadata_path,
        {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "workspace_root": str(workspace_root),
            "base_config_path": str(base_config_path),
            "compare_config_path": str(compare_config_path),
            "runs_dir_for_tasks": str(runs_dir_for_tasks),
            "audios": [str(path) for path in audio_list],
            "scenarios": [asdict(item) | {"patch_factory": str(item.patch_factory)} for item in scenarios],
            "runs_per_case": args.runs_per_case,
            "module_a_mode": args.module_a_mode,
            "force": bool(args.force),
        },
    )

    _print_records(records)
    _print_summary(summary_rows)
    _print_speedup(speed_rows)

    print("\n跑数结果已写入：")
    print(f"- {records_path}")
    print(f"- {records_csv_path}")
    print(f"- {summary_path}")
    print(f"- {speed_path}")
    print(f"- {metadata_path}")


if __name__ == "__main__":
    main()
