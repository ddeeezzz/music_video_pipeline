"""
文件用途：提供 model_assets 命令入口，转发到 scripts/model_assets/main.py。
核心流程：定位项目根目录 -> 校验脚本存在 -> 子进程转发参数执行。
输入输出：输入命令行参数，输出脚本执行结果并透传退出码。
依赖说明：依赖标准库 pathlib/subprocess/sys。
维护说明：该入口只做命令转发，核心逻辑维护在 scripts/model_assets 包内。
"""

# 标准库：用于路径解析
from pathlib import Path
# 标准库：用于子进程执行
import subprocess
# 标准库：用于读取命令行参数与退出码
import sys


def _resolve_main_script_path() -> Path:
    """
    功能说明：解析 model_assets 主脚本路径。
    参数说明：无。
    返回值：
    - Path: scripts/model_assets/main.py 绝对路径。
    异常说明：
    - RuntimeError: 未找到脚本时抛出。
    边界条件：优先尝试当前工作目录，其次尝试从当前文件反推项目根目录。
    """
    cwd_candidate = (Path.cwd().resolve() / "scripts" / "model_assets" / "main.py").resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (Path(__file__).resolve().parents[2] / "scripts" / "model_assets" / "main.py").resolve()
    if repo_candidate.exists():
        return repo_candidate

    raise RuntimeError(
        "未找到 model_assets 主脚本，请在项目根目录执行命令，或检查 scripts/model_assets/main.py 是否存在"
    )


def main() -> int:
    """
    功能说明：model_assets CLI 入口，转发参数到主脚本。
    参数说明：无（读取 sys.argv[1:]）。
    返回值：
    - int: 透传主脚本退出码。
    异常说明：
    - RuntimeError: 主脚本缺失时抛出。
    边界条件：参数原样透传，不做额外语义处理。
    """
    script_path = _resolve_main_script_path()
    command = [sys.executable, str(script_path), *sys.argv[1:]]
    result = subprocess.run(command, check=False)
    return int(result.returncode)
