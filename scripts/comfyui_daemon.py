"""
文件用途：提供 ComfyUI 常驻后台进程的启动、停止与状态查询命令。
核心流程：发现现有 ComfyUI 进程 -> 按需后台拉起 main.py -> 通过 pid/log/http 探活统一管理。
输入输出：输入 start/stop/status 命令，输出标准中文状态信息。
依赖说明：依赖标准库 argparse/os/pathlib/signal/socket/subprocess/sys/time/urllib。
维护说明：本脚本只管理 ComfyUI HTTP 服务本身，不承担模块 C/D 业务调度。
"""

# 标准库：用于解析命令行参数。
import argparse
# 标准库：用于环境变量与进程组控制。
import os
# 标准库：用于文件路径处理。
from pathlib import Path
# 标准库：用于发送进程信号。
import signal
# 标准库：用于端口占用探测。
import socket
# 标准库：用于拉起后台子进程。
import subprocess
# 标准库：用于获取当前 Python 解释器路径。
import sys
# 标准库：用于轮询等待。
import time
# 标准库：用于 HTTP 探活。
from urllib.error import URLError
# 标准库：用于 HTTP 请求。
from urllib.request import urlopen

# 常量：项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 常量：ComfyUI 根目录。
COMFYUI_ROOT = PROJECT_ROOT / "ComfyUI"
# 常量：运行时目录。
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
# 常量：daemon pid 文件。
PID_FILE = RUNTIME_DIR / "comfyui_daemon.pid"
# 常量：daemon 日志文件。
LOG_FILE = RUNTIME_DIR / "comfyui_daemon.log"
# 常量：默认监听地址。
DEFAULT_HOST = "127.0.0.1"
# 常量：默认监听端口。
DEFAULT_PORT = 8188
# 常量：默认 HTTP 探活超时秒数。
DEFAULT_READY_TIMEOUT_SECONDS = 60.0


def build_argument_parser() -> argparse.ArgumentParser:
    """
    功能说明：构建命令行参数解析器。
    参数说明：无。
    返回值：
    - argparse.ArgumentParser: 参数解析器对象。
    异常说明：无。
    边界条件：当前仅提供 start/stop/status 三个命令。
    """
    parser = argparse.ArgumentParser(description="管理 ComfyUI 常驻后台服务。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="后台启动 ComfyUI 服务")
    start_parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址，默认 127.0.0.1")
    start_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口，默认 8188")
    start_parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help="启动后等待 HTTP 就绪的最长秒数，默认 60",
    )

    stop_parser = subparsers.add_parser("stop", help="停止 ComfyUI 服务")
    stop_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="目标端口，默认 8188")
    stop_parser.add_argument("--graceful-timeout", type=float, default=20.0, help="优雅退出等待秒数，默认 20")

    status_parser = subparsers.add_parser("status", help="查看 ComfyUI 服务状态")
    status_parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址，默认 127.0.0.1")
    status_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口，默认 8188")

    return parser


def _ensure_runtime_dir() -> None:
    """
    功能说明：确保运行时目录存在。
    参数说明：无。
    返回值：无。
    异常说明：目录创建失败时抛出 OSError。
    边界条件：重复执行保持幂等。
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid_file() -> int | None:
    """
    功能说明：读取 pid 文件中的进程号。
    参数说明：无。
    返回值：
    - int | None: 合法 pid；文件不存在或非法时返回 None。
    异常说明：无。
    边界条件：坏 pid 文件会自动忽略。
    """
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_pid_file(pid: int) -> None:
    """
    功能说明：写入 daemon pid 文件。
    参数说明：
    - pid: 目标进程号。
    返回值：无。
    异常说明：写入失败时抛出 OSError。
    边界条件：会覆盖旧 pid 文件。
    """
    _ensure_runtime_dir()
    PID_FILE.write_text(f"{int(pid)}\n", encoding="utf-8")


def _remove_pid_file() -> None:
    """
    功能说明：删除 pid 文件。
    参数说明：无。
    返回值：无。
    异常说明：无。
    边界条件：文件不存在时静默跳过。
    """
    PID_FILE.unlink(missing_ok=True)


def _is_process_alive(pid: int | None) -> bool:
    """
    功能说明：判断目标进程是否仍然存活。
    参数说明：
    - pid: 目标进程号。
    返回值：
    - bool: True 表示进程存在。
    异常说明：无。
    边界条件：pid<=0 直接返回 False。
    """
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _iter_proc_cmdlines() -> list[tuple[int, str]]:
    """
    功能说明：遍历当前系统进程的 cmdline。
    参数说明：无。
    返回值：
    - list[tuple[int, str]]: (pid, cmdline_text) 数组。
    异常说明：无。
    边界条件：仅适配 Linux `/proc` 进程视图。
    """
    process_rows: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    for proc_dir in proc_root.iterdir():
        if not proc_dir.is_dir() or not proc_dir.name.isdigit():
            continue
        cmdline_path = proc_dir / "cmdline"
        try:
            raw_bytes = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw_bytes:
            continue
        cmdline_text = raw_bytes.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        if cmdline_text:
            process_rows.append((int(proc_dir.name), cmdline_text))
    return process_rows


def _find_existing_comfyui_pid(host: str, port: int) -> int | None:
    """
    功能说明：扫描系统进程，查找匹配当前 ComfyUI 启动命令的进程号。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    返回值：
    - int | None: 匹配到的 pid；不存在时返回 None。
    异常说明：无。
    边界条件：仅匹配 `python main.py --listen <host> --port <port>` 形式的启动命令。
    """
    port_text = str(int(port))
    for pid, cmdline_text in _iter_proc_cmdlines():
        if "main.py" not in cmdline_text:
            continue
        if "--listen" not in cmdline_text or "--port" not in cmdline_text:
            continue
        if f"--listen {host}" not in cmdline_text:
            continue
        if f"--port {port_text}" not in cmdline_text:
            continue
        if "python" not in cmdline_text:
            continue
        return pid
    return None


def _is_port_open(host: str, port: int) -> bool:
    """
    功能说明：探测目标端口是否已被监听。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    返回值：
    - bool: True 表示端口可连接。
    异常说明：无。
    边界条件：仅用于快速预检查，不代表服务功能完整。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, int(port))) == 0


def _check_http_ready(host: str, port: int, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    """
    功能说明：通过 ComfyUI `system_stats` 接口探测服务是否就绪。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    - timeout_seconds: HTTP 请求超时秒数。
    返回值：
    - tuple[bool, str]: (是否就绪, 错误文本或简要状态)。
    异常说明：无。
    边界条件：请求失败只返回 False，不抛异常。
    """
    url = f"http://{host}:{int(port)}/system_stats"
    try:
        with urlopen(url, timeout=max(0.5, float(timeout_seconds))) as response:
            return response.status == 200, f"HTTP {response.status}"
    except URLError as error:
        return False, str(error)
    except Exception as error:  # noqa: BLE001
        return False, str(error)


def _wait_http_ready(host: str, port: int, ready_timeout: float) -> bool:
    """
    功能说明：轮询等待 ComfyUI HTTP 服务就绪。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    - ready_timeout: 最长等待秒数。
    返回值：
    - bool: True 表示在时限内就绪。
    异常说明：无。
    边界条件：最短等待窗口固定为 1 秒。
    """
    deadline = time.time() + max(1.0, float(ready_timeout))
    while time.time() < deadline:
        ready, _ = _check_http_ready(host=host, port=port, timeout_seconds=1.5)
        if ready:
            return True
        time.sleep(0.5)
    return False


def _spawn_comfyui_process(host: str, port: int) -> subprocess.Popen:
    """
    功能说明：以后台 detached 方式拉起 ComfyUI 主进程。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    返回值：
    - subprocess.Popen: 后台进程对象。
    异常说明：拉起失败时抛出 OSError/subprocess.SubprocessError。
    边界条件：stdout/stderr 统一重定向到固定日志文件。
    """
    _ensure_runtime_dir()
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    launch_env = dict(os.environ)
    launch_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, "main.py", "--listen", str(host), "--port", str(int(port))],
        cwd=str(COMFYUI_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=launch_env,
        start_new_session=True,
    )
    return process


def start_daemon(host: str, port: int, ready_timeout: float) -> int:
    """
    功能说明：后台启动 ComfyUI daemon；若服务已在运行则直接复用。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    - ready_timeout: 最长等待就绪秒数。
    返回值：
    - int: 进程退出码，0 表示成功。
    异常说明：无；错误以文本形式输出并返回非零退出码。
    边界条件：如果检测到现有匹配进程，会自动写入 pid 文件接管管理。
    """
    managed_pid = _read_pid_file()
    if _is_process_alive(managed_pid):
        ready, ready_text = _check_http_ready(host=host, port=port)
        status_text = "已就绪" if ready else f"存活但HTTP未就绪({ready_text})"
        print(
            f"ComfyUI daemon 已在运行，pid={managed_pid}，host={host}，port={port}，{status_text}，"
            f"log={LOG_FILE}"
        )
        return 0

    existing_pid = _find_existing_comfyui_pid(host=host, port=port)
    if _is_process_alive(existing_pid):
        _write_pid_file(int(existing_pid))
        ready, ready_text = _check_http_ready(host=host, port=port)
        status_text = "已就绪" if ready else f"存活但HTTP未就绪({ready_text})"
        print(
            f"检测到已存在的 ComfyUI 进程，已接管为 daemon，pid={existing_pid}，host={host}，port={port}，"
            f"{status_text}，log={LOG_FILE}"
        )
        return 0

    if _is_port_open(host=host, port=port):
        print(
            f"ComfyUI daemon 启动失败：{host}:{port} 端口已被占用，但未识别到可接管的 ComfyUI 进程。"
        )
        return 1

    try:
        process = _spawn_comfyui_process(host=host, port=port)
    except Exception as error:  # noqa: BLE001
        print(f"ComfyUI daemon 启动失败：拉起进程异常，错误={error}")
        return 1

    _write_pid_file(int(process.pid))
    ready = _wait_http_ready(host=host, port=port, ready_timeout=ready_timeout)
    if ready:
        print(
            f"ComfyUI daemon 启动成功，pid={process.pid}，host={host}，port={port}，log={LOG_FILE}"
        )
        return 0

    if _is_process_alive(int(process.pid)):
        print(
            f"ComfyUI daemon 进程已启动但超时未就绪，pid={process.pid}，host={host}，port={port}，log={LOG_FILE}"
        )
        return 1

    _remove_pid_file()
    print(f"ComfyUI daemon 启动失败：进程已退出，请查看日志 {LOG_FILE}")
    return 1


def _terminate_pid(pid: int, graceful_timeout: float) -> bool:
    """
    功能说明：优雅终止目标进程组，必要时再强杀。
    参数说明：
    - pid: 目标进程号。
    - graceful_timeout: 优雅退出等待秒数。
    返回值：
    - bool: True 表示目标进程已退出。
    异常说明：无。
    边界条件：若进程已不存在，直接视为成功。
    """
    if not _is_process_alive(pid):
        return True
    try:
        process_group_id = os.getpgid(int(pid))
    except OSError:
        return not _is_process_alive(pid)

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except OSError:
        return not _is_process_alive(pid)

    deadline = time.time() + max(1.0, float(graceful_timeout))
    while time.time() < deadline:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.5)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        return not _is_process_alive(pid)

    time.sleep(0.5)
    return not _is_process_alive(pid)


def stop_daemon(port: int, graceful_timeout: float) -> int:
    """
    功能说明：停止当前 ComfyUI daemon；若 pid 文件缺失则尝试扫描现有进程。
    参数说明：
    - port: 监听端口。
    - graceful_timeout: 优雅退出等待秒数。
    返回值：
    - int: 进程退出码，0 表示成功。
    异常说明：无；错误以文本形式输出并返回非零退出码。
    边界条件：当前默认 host 固定为 127.0.0.1。
    """
    pid = _read_pid_file()
    if not _is_process_alive(pid):
        pid = _find_existing_comfyui_pid(host=DEFAULT_HOST, port=port)
    if not _is_process_alive(pid):
        _remove_pid_file()
        print(f"ComfyUI daemon 未运行，host={DEFAULT_HOST}，port={port}")
        return 0

    stopped = _terminate_pid(int(pid), graceful_timeout=graceful_timeout)
    if stopped:
        _remove_pid_file()
        print(f"ComfyUI daemon 已停止，pid={pid}，host={DEFAULT_HOST}，port={port}")
        return 0

    print(f"ComfyUI daemon 停止失败，pid={pid}，请检查进程与日志 {LOG_FILE}")
    return 1


def status_daemon(host: str, port: int) -> int:
    """
    功能说明：输出当前 ComfyUI daemon 的运行状态。
    参数说明：
    - host: 监听地址。
    - port: 监听端口。
    返回值：
    - int: 进程退出码，0 表示命令执行成功。
    异常说明：无。
    边界条件：即便 pid 文件缺失，也会尝试扫描已有匹配进程并报告 HTTP 状态。
    """
    pid = _read_pid_file()
    managed = _is_process_alive(pid)
    if not managed:
        pid = _find_existing_comfyui_pid(host=host, port=port)
    alive = _is_process_alive(pid)
    ready, ready_text = _check_http_ready(host=host, port=port)
    if alive:
        source_text = "pid_file" if managed else "process_scan"
        print(
            f"ComfyUI daemon 状态：运行中。 pid={pid} source={source_text} host={host} port={port} "
            f"http_ready={ready} ({ready_text}) log={LOG_FILE} pid_file={PID_FILE}"
        )
    else:
        print(
            f"ComfyUI daemon 状态：未运行。 host={host} port={port} http_ready={ready} ({ready_text}) "
            f"log={LOG_FILE} pid_file={PID_FILE}"
        )
    return 0


def main() -> int:
    """
    功能说明：脚本主入口。
    参数说明：无。
    返回值：
    - int: 进程退出码。
    异常说明：无。
    边界条件：未知命令由 argparse 处理。
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.command == "start":
        return start_daemon(host=str(args.host), port=int(args.port), ready_timeout=float(args.ready_timeout))
    if args.command == "stop":
        return stop_daemon(port=int(args.port), graceful_timeout=float(args.graceful_timeout))
    if args.command == "status":
        return status_daemon(host=str(args.host), port=int(args.port))
    parser.error(f"未知命令：{args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
