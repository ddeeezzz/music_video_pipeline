#!/usr/bin/env python3
"""
文件用途：提供轻量 GPU 显存采样入口，供跨模块调度器读取。
核心流程：调用 nvidia-smi 查询显存并输出结构化 JSON。
输入输出：输入命令行参数，输出标准 JSON 文本。
依赖说明：依赖标准库 argparse/json/subprocess。
维护说明：脚本应保持无第三方依赖，便于在最小环境中运行。
"""

# 标准库：用于参数解析。
import argparse
# 标准库：用于 JSON 编解码。
import json
# 标准库：用于外部命令执行。
import subprocess
# 标准库：用于类型提示。
from typing import Any


def _parse_nvidia_smi_csv(stdout_text: str) -> list[dict[str, Any]]:
    """
    功能说明：解析 nvidia-smi CSV 输出并提取显存统计。
    参数说明：
    - stdout_text: nvidia-smi 标准输出文本。
    返回值：
    - list[dict[str, Any]]: GPU 统计数组。
    异常说明：
    - ValueError: 输出格式非法时抛出。
    边界条件：只解析 index/memory.total/memory.used 三列。
    """
    rows: list[dict[str, Any]] = []
    for raw_line in str(stdout_text).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            raise ValueError(f"nvidia-smi 输出列数非法：{stripped}")
        gpu_index = int(parts[0])
        total_mb = int(parts[1])
        used_mb = int(parts[2])
        if total_mb <= 0:
            raise ValueError(f"nvidia-smi 输出显存总量非法：{stripped}")
        used_ratio = max(0.0, min(1.0, float(used_mb) / float(total_mb)))
        rows.append(
            {
                "index": gpu_index,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "used_ratio": round(used_ratio, 6),
            }
        )
    return rows


def _run_probe(timeout_seconds: float) -> dict[str, Any]:
    """
    功能说明：执行一次 GPU 显存探测。
    参数说明：
    - timeout_seconds: 命令超时时间（秒）。
    返回值：
    - dict[str, Any]: 结构化探测结果。
    异常说明：内部捕获异常并通过 ok=false 输出。
    边界条件：命令失败时不抛出，改为结构化错误返回。
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"probe_exec_failed: {error}",
            "gpus": [],
        }
    if result.returncode != 0:
        stderr_text = str(result.stderr or "").strip()
        return {
            "ok": False,
            "error": f"probe_failed: exit_code={result.returncode}, stderr={stderr_text}",
            "gpus": [],
        }
    try:
        gpus = _parse_nvidia_smi_csv(result.stdout)
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"probe_parse_failed: {error}",
            "gpus": [],
        }
    return {
        "ok": True,
        "error": "",
        "gpus": gpus,
    }


def main() -> int:
    """
    功能说明：脚本主入口。
    参数说明：无（由 argparse 解析）。
    返回值：
    - int: 0 表示成功，1 表示失败。
    异常说明：无。
    边界条件：默认输出 JSON，便于调用方直接解析。
    """
    parser = argparse.ArgumentParser(description="轻量 GPU 显存采样")
    parser.add_argument("--timeout", type=float, default=5.0, help="nvidia-smi 超时时间（秒）")
    parser.add_argument("--json", action="store_true", help="保持兼容：始终输出 JSON")
    args = parser.parse_args()
    _ = args.json
    payload = _run_probe(timeout_seconds=float(args.timeout))
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
