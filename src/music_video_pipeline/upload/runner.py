"""
文件用途：封装 bypy 命令执行与运行前置校验。
核心流程：解析 bypy 路径与鉴权 -> 组装命令 -> 执行 syncup 并返回结构化结果。
输入输出：输入 task_id/配置/本地目录，输出包含 exit_code 与输出尾部的结果字典。
依赖说明：依赖标准库 pathlib/shutil/subprocess 与项目内上传配置。
维护说明：该文件只处理命令执行，不负责队列状态机与 compare 判定。
"""

# 标准库：用于日志对象类型
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于命令查找
import shutil
# 标准库：用于子进程调用
import subprocess
# 标准库：用于类型提示
from typing import Any

# 项目内模块：上传配置结构
from music_video_pipeline.config import BypyUploadConfig

# 常量：bypy 默认鉴权文件名。
BYPY_AUTH_FILE_NAME = "bypy.json"


def _normalize_remote_runs_dir(remote_runs_dir: str) -> str:
    """
    功能说明：标准化网盘 runs 根目录，确保以单斜杠开头且不带末尾斜杠。
    参数说明：
    - remote_runs_dir: 原始远端目录文本。
    返回值：
    - str: 标准化后的远端 runs 根目录。
    异常说明：无。
    边界条件：空字符串时回退为 /runs。
    """
    normalized = str(remote_runs_dir).strip().replace("\\", "/")
    if not normalized:
        normalized = "/runs"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    normalized = normalized.rstrip("/")
    return normalized or "/runs"


def _build_remote_task_dir(remote_runs_dir: str, task_id: str) -> str:
    """
    功能说明：根据 runs 根目录与 task_id 拼接远端任务目录。
    参数说明：
    - remote_runs_dir: 远端 runs 根目录。
    - task_id: 任务唯一标识。
    返回值：
    - str: 远端任务目录（/runs/<task_id>）。
    异常说明：无。
    边界条件：task_id 为空时由调用方提前拦截。
    """
    safe_runs_dir = _normalize_remote_runs_dir(remote_runs_dir=remote_runs_dir)
    return f"{safe_runs_dir}/{str(task_id).strip()}"


def _tail_lines(text: str, limit: int = 12) -> str:
    """
    功能说明：截取文本末尾若干行，避免日志输出过长。
    参数说明：
    - text: 原始文本。
    - limit: 最大保留行数。
    返回值：
    - str: 截断后的文本。
    异常说明：无。
    边界条件：空文本返回空字符串。
    """
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(limit)):])


def _resolve_bypy_bin(bypy_bin: str) -> str | None:
    """
    功能说明：解析 bypy 可执行路径。
    参数说明：
    - bypy_bin: 配置中的命令名或路径。
    返回值：
    - str | None: 可执行路径，找不到时返回 None。
    异常说明：无。
    边界条件：支持命令名与绝对/相对路径。
    """
    candidate = str(bypy_bin).strip()
    if not candidate:
        return None
    if "/" in candidate:
        path_obj = Path(candidate).expanduser()
        return str(path_obj) if path_obj.exists() else None
    return shutil.which(candidate)


def _validate_bypy_runtime(
    *,
    task_id: str,
    upload_config: BypyUploadConfig,
    logger: logging.Logger,
) -> tuple[bool, str, str, Path]:
    """
    功能说明：校验 bypy 运行前置条件并返回可执行参数。
    参数说明：
    - task_id: 任务唯一标识。
    - upload_config: bypy 上传配置。
    - logger: 日志对象。
    返回值：
    - tuple[bool, str, str, Path]: (是否可执行, bypy路径, 失败原因, config_dir路径)。
    异常说明：无。
    边界条件：仅做前置校验，不执行上传。
    """
    resolved_bypy_bin = _resolve_bypy_bin(upload_config.bypy_bin)
    if not resolved_bypy_bin:
        reason = f"未找到 bypy 可执行文件，bypy_bin={upload_config.bypy_bin}"
        logger.warning("任务产物上传跳过：%s，task_id=%s", reason, task_id)
        return False, "", reason, Path("")

    config_dir_path = Path(str(upload_config.config_dir)).expanduser()
    auth_file_path = config_dir_path / BYPY_AUTH_FILE_NAME
    if upload_config.require_auth_file and not auth_file_path.exists():
        reason = f"未找到 bypy 鉴权文件，auth_file={auth_file_path}"
        logger.warning("任务产物上传跳过：%s，task_id=%s", reason, task_id)
        return False, "", reason, config_dir_path
    return True, resolved_bypy_bin, "", config_dir_path


def run_bypy_syncup(
    *,
    local_source_dir: Path,
    task_id: str,
    upload_config: BypyUploadConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    功能说明：执行 bypy syncup 并返回结构化执行结果。
    参数说明：
    - local_source_dir: 本地待上传目录（可为 staging 目录）。
    - task_id: 任务唯一标识。
    - upload_config: bypy 上传配置。
    - logger: 日志对象。
    返回值：
    - dict[str, Any]: 包含 success/message/exit_code/stdout_tail/stderr_tail/command 的结果字典。
    异常说明：无（异常内部捕获并转为失败结果）。
    边界条件：当前置校验失败时 exit_code 为空（None）。
    """
    if not local_source_dir.exists():
        return {
            "success": False,
            "message": f"本地上传目录不存在：{local_source_dir}",
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "command": [],
            "remote_task_dir": "",
        }

    is_ready, resolved_bypy_bin, ready_reason, config_dir_path = _validate_bypy_runtime(
        task_id=task_id,
        upload_config=upload_config,
        logger=logger,
    )
    if not is_ready:
        return {
            "success": False,
            "message": ready_reason,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "command": [],
            "remote_task_dir": "",
        }

    remote_task_dir = _build_remote_task_dir(
        remote_runs_dir=upload_config.remote_runs_dir,
        task_id=task_id,
    )
    retry_times = max(0, int(upload_config.retry_times))
    timeout_seconds = max(1.0, float(upload_config.timeout_seconds))
    # 约束：上传采用“严格镜像白名单”模式，显式开启 deleteremote 清理远端多余文件。
    command = [
        resolved_bypy_bin,
        "--retry",
        str(retry_times),
        "--timeout",
        str(int(timeout_seconds)),
        "--config-dir",
        str(config_dir_path),
        "syncup",
        str(local_source_dir),
        remote_task_dir,
        "true",
    ]

    logger.info(
        "任务产物上传开始，task_id=%s，本地=%s，远端=%s",
        task_id,
        local_source_dir,
        remote_task_dir,
    )
    logger.info("执行 bypy 命令：%s", " ".join(command))

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 60.0,
        )
    except subprocess.TimeoutExpired:
        timeout_message = f"bypy 上传超时，timeout_seconds={timeout_seconds + 60.0}"
        return {
            "success": False,
            "message": timeout_message,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": timeout_message,
            "command": command,
            "remote_task_dir": remote_task_dir,
        }
    except Exception as error:  # noqa: BLE001
        error_message = f"bypy 上传异常：{error}"
        return {
            "success": False,
            "message": error_message,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": error_message,
            "command": command,
            "remote_task_dir": remote_task_dir,
        }

    stdout_tail = _tail_lines(result.stdout, limit=10)
    stderr_tail = _tail_lines(result.stderr, limit=10)
    if int(result.returncode) != 0:
        fail_reason = (
            f"bypy exit_code={result.returncode}; "
            f"stdout_tail={stdout_tail or '<empty>'}; stderr_tail={stderr_tail or '<empty>'}"
        )
        logger.warning("任务产物上传失败，task_id=%s，%s", task_id, fail_reason)
        return {
            "success": False,
            "message": fail_reason,
            "exit_code": int(result.returncode),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "command": command,
            "remote_task_dir": remote_task_dir,
        }

    success_message = f"远端={remote_task_dir}，stdout_tail={stdout_tail or '<empty>'}"
    logger.info("任务产物上传完成，task_id=%s，%s", task_id, success_message)
    return {
        "success": True,
        "message": success_message,
        "exit_code": int(result.returncode),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "command": command,
        "remote_task_dir": remote_task_dir,
    }
