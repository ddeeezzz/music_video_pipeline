"""
文件用途：封装 bypy 命令调用，提供统一的远端读取与下载能力。
核心流程：执行命令 -> 识别错误输出 -> 返回标准输出文本。
输入输出：输入 bypy 子命令参数，输出命令标准输出。
依赖说明：依赖标准库 subprocess/sys/re 与 logging。
维护说明：该模块仅负责命令调用与错误识别，不处理业务规则。
"""

# 标准库：用于日志记录
import logging
# 标准库：用于正则匹配错误文本
import re
# 标准库：用于子进程执行
import subprocess
# 标准库：用于获取当前 Python 解释器
import sys
# 标准库：用于路径处理
from pathlib import Path


class BypyClientError(RuntimeError):
    """
    功能说明：表示 bypy 命令调用失败。
    参数说明：继承 RuntimeError，直接传入中文错误信息。
    返回值：不适用。
    异常说明：不适用。
    边界条件：用于统一主流程错误处理。
    """


# 常量：错误输出中常见的 Error 关键字模式。
ERROR_TEXT_PATTERN = re.compile(r"(^|\n)\s*Error\s+\d+", re.IGNORECASE)


class BypyClient:
    """
    功能说明：bypy 命令客户端。
    参数说明：
    - logger: 日志对象，用于输出命令执行信息。
    返回值：不适用。
    异常说明：命令执行失败会抛出 BypyClientError。
    边界条件：默认使用当前解释器执行 `python -m bypy`。
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log_command(self, command: list[str], args: list[str]) -> None:
        """
        功能说明：按命令类型输出日志，避免高频目录探测刷屏。
        参数说明：
        - command: 完整命令数组（含 python -m bypy 前缀）。
        - args: bypy 子命令参数数组。
        返回值：无。
        异常说明：无。
        边界条件：`list` 子命令仅输出 DEBUG，下载命令输出 INFO。
        """
        command_text = " ".join(command)
        action = str(args[0]).strip().lower() if args else ""
        if action == "list":
            self._logger.debug("执行 bypy 命令：%s", command_text)
            return
        self._logger.info("执行 bypy 命令：%s", command_text)

    def run(self, args: list[str]) -> str:
        """
        功能说明：执行 bypy 子命令并返回标准输出。
        参数说明：
        - args: bypy 子命令参数数组（不含 `python -m bypy` 前缀）。
        返回值：
        - str: 命令标准输出。
        异常说明：
        - BypyClientError: 返回码非0或输出包含错误标记时抛出。
        边界条件：stderr 也会纳入错误判断。
        """
        command = [sys.executable, "-m", "bypy", *args]
        self._log_command(command=command, args=args)
        result = subprocess.run(command, check=False, capture_output=True, text=True)

        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        merged_text = (stdout_text + "\n" + stderr_text).strip()

        if result.returncode != 0:
            raise BypyClientError(
                f"bypy 命令返回非零退出码：{result.returncode}，输出：{merged_text}"
            )

        if ERROR_TEXT_PATTERN.search(merged_text) or "<E>" in merged_text:
            raise BypyClientError(f"bypy 返回错误信息：{merged_text}")

        return stdout_text

    def list_remote(self, remote_dir: str) -> str:
        """
        功能说明：列举远端目录内容。
        参数说明：
        - remote_dir: 远端目录路径（以 / 开头）。
        返回值：
        - str: bypy list 命令输出文本。
        异常说明：
        - BypyClientError: 调用失败时抛出。
        边界条件：无。
        """
        return self.run(["list", remote_dir])

    def downfile(self, remote_file: str, local_path: Path) -> None:
        """
        功能说明：下载单个远端文件到本地。
        参数说明：
        - remote_file: 远端文件路径。
        - local_path: 本地目标文件路径。
        返回值：无。
        异常说明：
        - BypyClientError: 下载失败或目标文件不存在时抛出。
        边界条件：目标父目录不存在时会自动创建。
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.run(["downfile", remote_file, str(local_path)])
        if not local_path.exists():
            raise BypyClientError(f"下载完成后未找到本地文件：{local_path}")

    def downdir(self, remote_dir: str, local_dir: Path) -> None:
        """
        功能说明：递归下载远端目录到本地。
        参数说明：
        - remote_dir: 远端目录路径。
        - local_dir: 本地目录路径。
        返回值：无。
        异常说明：
        - BypyClientError: 下载失败时抛出。
        边界条件：本地目录会自动创建。
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        self.run(["downdir", remote_dir, str(local_dir)])
