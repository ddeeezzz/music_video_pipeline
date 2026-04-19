"""
文件用途：执行上传后 compare 核对并给出状态机门禁判定。
核心流程：构建 compare 命令 -> 解析输出 -> 写核对报告 -> 判断是否允许 job 标记成功。
输入输出：输入 task_dir/task_id/上传配置，输出结构化报告与门禁结果。
依赖说明：依赖标准库 json/re/subprocess/pathlib 与 runner/staging 模块。
维护说明：状态机判定规则（严格镜像门禁）集中维护于本文件。
"""

# 标准库：用于时间戳格式化
from datetime import datetime
# 标准库：用于结构化报告落盘
import json
# 标准库：用于日志对象类型
import logging
# 标准库：用于输出解析
import re
# 标准库：用于文件复制与临时目录清理
import shutil
# 标准库：用于子进程执行
import subprocess
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：bypy 上传配置结构
from music_video_pipeline.config import BypyUploadConfig
# 项目内模块：上传命令工具函数
from music_video_pipeline.upload.runner import _build_remote_task_dir, _validate_bypy_runtime
# 项目内模块：白名单 staging/压缩上传目录构建
from music_video_pipeline.upload.staging import build_whitelist_staging_dir


def _parse_bypy_compare_output(compare_stdout: str) -> dict[str, Any]:
    """
    功能说明：解析 bypy compare 输出，提取 Same/Different/Local only/Remote only 列表与统计值。
    参数说明：
    - compare_stdout: bypy compare 标准输出文本。
    返回值：
    - dict[str, Any]: 包含 items/stats/unknown_lines 的解析结果。
    异常说明：无。
    边界条件：输出格式变化时，未识别行会写入 unknown_lines 便于排查。
    """
    section_map = {
        "same": "==== Same files ===",
        "different": "==== Different files ===",
        "local_only": "==== Local only ====",
        "remote_only": "==== Remote only ====",
    }
    reverse_section_map = {value: key for key, value in section_map.items()}
    stat_pattern = re.compile(r"^(Same|Different|Local only|Remote only):\s*(\d+)\s*$")

    items: dict[str, list[str]] = {
        "same": [],
        "different": [],
        "local_only": [],
        "remote_only": [],
    }
    stats: dict[str, int] = {
        "same": 0,
        "different": 0,
        "local_only": 0,
        "remote_only": 0,
    }
    unknown_lines: list[str] = []
    current_section: str | None = None

    for raw_line in str(compare_stdout).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped in reverse_section_map:
            current_section = reverse_section_map[stripped]
            continue
        stat_match = stat_pattern.match(stripped)
        if stat_match:
            key_text = stat_match.group(1).strip().lower().replace(" ", "_")
            stats[key_text] = int(stat_match.group(2))
            continue
        if stripped.startswith(("F - ", "D - ")):
            entry_path = stripped[4:].strip()
            if current_section in items:
                items[current_section].append(entry_path)
            else:
                unknown_lines.append(stripped)
            continue
        if stripped.startswith(("Statistics:", "--------------------------------")):
            continue
        unknown_lines.append(stripped)

    for key in ("same", "different", "local_only", "remote_only"):
        if stats.get(key, 0) <= 0 and items.get(key):
            stats[key] = len(items[key])

    return {
        "items": items,
        "stats": stats,
        "unknown_lines": unknown_lines,
    }


def _build_bypy_compare_report_text(report: dict[str, Any]) -> str:
    """
    功能说明：将结构化 compare 报告渲染为可读文本。
    参数说明：
    - report: compare 结构化报告。
    返回值：
    - str: 文本报告内容。
    异常说明：无。
    边界条件：空列表会输出 <none>，便于人工快速判断。
    """
    summary = report.get("summary", {})
    parsed = report.get("parsed_compare", {})
    items = parsed.get("items", {})
    stats = parsed.get("stats", {})
    unknown_lines = parsed.get("unknown_lines", [])

    lines: list[str] = [
        "上传白名单 vs 远端目录核对报告",
        "",
        f"task_id: {report.get('task_id', '')}",
        f"remote_task_dir: {report.get('remote_task_dir', '')}",
        f"selection_profile: {report.get('selection_profile', '')}",
        f"bypy_compare_exit_code: {report.get('bypy_compare_exit_code', -1)}",
        "",
        "摘要：",
        f"- local_whitelist_count: {summary.get('local_whitelist_count', 0)}",
        f"- same: {stats.get('same', 0)}",
        f"- different: {stats.get('different', 0)}",
        f"- local_only(远端缺失): {stats.get('local_only', 0)}",
        f"- remote_only(远端额外): {stats.get('remote_only', 0)}",
        "",
        "local_only（远端缺失）：",
    ]
    if items.get("local_only"):
        lines.extend(f"- {item}" for item in items["local_only"])
    else:
        lines.append("- <none>")

    lines.extend(["", "remote_only（远端额外）："])
    if items.get("remote_only"):
        lines.extend(f"- {item}" for item in items["remote_only"])
    else:
        lines.append("- <none>")

    lines.extend(["", "different（同名不同内容）："])
    if items.get("different"):
        lines.extend(f"- {item}" for item in items["different"])
    else:
        lines.append("- <none>")

    if unknown_lines:
        lines.extend(["", "未识别输出行（用于排查输出格式变化）："])
        lines.extend(f"- {line}" for line in unknown_lines)

    return "\n".join(lines) + "\n"


def run_whitelist_remote_compare(
    *,
    task_dir: Path,
    task_id: str,
    upload_config: BypyUploadConfig,
    selection_profile: str,
    logger: logging.Logger,
    local_source_dir: Path | None = None,
) -> dict[str, Any]:
    """
    功能说明：执行“白名单清单 vs 远端目录”核对并写入任务日志目录报告。
    参数说明：
    - task_dir: 任务目录（runs/<task_id>）。
    - task_id: 任务唯一标识。
    - upload_config: bypy 上传配置。
    - selection_profile: 白名单策略名。
    - logger: 日志对象。
    - local_source_dir: 可选，直接指定本次上传源目录（worker 内部调用推荐传入）。
    返回值：
    - dict[str, Any]: 结构化核对结果（含报告路径与统计）。
    异常说明：
    - RuntimeError: bypy 不可用或 compare 执行异常时抛错。
    边界条件：该函数只核对，不修改远端内容；若未传 local_source_dir 则按当前上传策略临时构建压缩上传目录。
    """
    is_ready, resolved_bypy_bin, ready_reason, config_dir_path = _validate_bypy_runtime(
        task_id=task_id,
        upload_config=upload_config,
        logger=logger,
    )
    if not is_ready:
        raise RuntimeError(ready_reason)

    report_dir = task_dir / "log"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = report_dir / f"upload_whitelist_compare_{timestamp}.json"
    report_txt_path = report_dir / f"upload_whitelist_compare_{timestamp}.txt"
    remote_task_dir = _build_remote_task_dir(
        remote_runs_dir=upload_config.remote_runs_dir,
        task_id=task_id,
    )

    compare_local_dir: Path | None = None
    should_cleanup_compare_local_dir = False
    try:
        if local_source_dir is not None:
            compare_local_dir = Path(local_source_dir)
        else:
            compare_local_dir, _ = build_whitelist_staging_dir(
                task_dir=task_dir,
                task_id=task_id,
                selection_profile=selection_profile,
                logger=logger,
            )
            should_cleanup_compare_local_dir = True
        local_upload_rel_files = sorted(
            [
                str(path.relative_to(compare_local_dir)).replace("\\", "/")
                for path in compare_local_dir.rglob("*")
                if path.is_file()
            ]
        )
        command = [
            resolved_bypy_bin,
            "--retry",
            str(max(0, int(upload_config.retry_times))),
            "--timeout",
            str(int(max(1.0, float(upload_config.timeout_seconds)))),
            "--config-dir",
            str(config_dir_path),
            "compare",
            remote_task_dir,
            str(compare_local_dir),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(120.0, float(upload_config.timeout_seconds) + 60.0),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"bypy compare 超时：{error}") from error
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"bypy compare 异常：{error}") from error
        parsed = _parse_bypy_compare_output(result.stdout)
        summary = {
            "local_whitelist_count": len(local_upload_rel_files),
            "same": int(parsed["stats"].get("same", 0)),
            "different": int(parsed["stats"].get("different", 0)),
            "local_only": int(parsed["stats"].get("local_only", 0)),
            "remote_only": int(parsed["stats"].get("remote_only", 0)),
        }
        report = {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "selection_profile": selection_profile,
            "remote_task_dir": remote_task_dir,
            "bypy_bin": resolved_bypy_bin,
            "bypy_compare_command": command,
            "bypy_compare_exit_code": int(result.returncode),
            "bypy_compare_stdout": result.stdout,
            "bypy_compare_stderr": result.stderr,
            "local_whitelist_rel_files": local_upload_rel_files,
            "parsed_compare": parsed,
            "summary": summary,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "report_json_path": str(report_json_path),
            "report_txt_path": str(report_txt_path),
        }
        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_txt_path.write_text(_build_bypy_compare_report_text(report), encoding="utf-8")
        logger.info(
            "上传后核对完成，task_id=%s，same=%s，different=%s，local_only=%s，remote_only=%s，report=%s",
            task_id,
            summary["same"],
            summary["different"],
            summary["local_only"],
            summary["remote_only"],
            report_json_path,
        )
        return report
    finally:
        if should_cleanup_compare_local_dir and compare_local_dir is not None and compare_local_dir.exists():
            shutil.rmtree(compare_local_dir, ignore_errors=True)


def evaluate_compare_gate(report: dict[str, Any]) -> tuple[bool, str]:
    """
    功能说明：根据 compare 报告判断本次上传尝试是否允许判定成功。
    参数说明：
    - report: run_whitelist_remote_compare 返回的结构化报告。
    返回值：
    - tuple[bool, str]: (是否通过门禁, 失败原因/通过说明)。
    异常说明：
    - RuntimeError: 当关键字段缺失或格式错误时抛错。
    边界条件：严格镜像模式下，local_only/remote_only/different 任一大于 0 均判定失败。
    """
    compare_exit_code_raw = report.get("bypy_compare_exit_code")
    if compare_exit_code_raw is None:
        raise RuntimeError("compare 结果缺少 bypy_compare_exit_code")
    compare_exit_code = int(compare_exit_code_raw)
    if compare_exit_code != 0:
        return False, f"compare exit_code={compare_exit_code}"

    summary_obj = report.get("summary")
    if not isinstance(summary_obj, dict):
        raise RuntimeError("compare 结果缺少 summary")
    for required_key in ("local_only", "remote_only", "different"):
        if required_key not in summary_obj:
            raise RuntimeError(f"compare 结果缺少 summary.{required_key}")
    local_only_count = int(summary_obj.get("local_only", 0))
    if local_only_count > 0:
        return False, f"compare 检测到远端缺失文件，local_only={local_only_count}"
    remote_only_count = int(summary_obj.get("remote_only", 0))
    if remote_only_count > 0:
        return False, f"compare 检测到远端多余文件，remote_only={remote_only_count}"
    different_count = int(summary_obj.get("different", 0))
    if different_count > 0:
        return False, f"compare 检测到同名内容不一致，different={different_count}"
    return True, "compare gate passed"
