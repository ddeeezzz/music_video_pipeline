"""
文件用途：提供模块 D 对 Remotion 模板工程的最小调用能力。
核心流程：解析本地 Remotion CLI 路径 -> 组装 render 命令 -> 调用子进程输出片段。
输入输出：输入模板工程目录、Composition 标识、请求 JSON 与输出路径，输出渲染完成的 mp4 文件。
依赖说明：依赖标准库 os/pathlib/subprocess 与模块 D 正式请求文件。
维护说明：当前只承担正式 Remotion 调用，不引入额外后端抽象层。
"""

# 标准库：用于系统平台判断。
import os
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于子进程执行。
import subprocess


def render_template_segment(
    *,
    remotion_project_dir: Path,
    composition_id: str,
    props_json_path: Path,
    output_path: Path,
) -> None:
    """
    功能说明：调用本地 Remotion CLI 渲染单个模板片段。
    参数说明：
    - remotion_project_dir: Remotion 模板工程目录。
    - composition_id: Composition 标识。
    - props_json_path: 正式模板请求 JSON 路径。
    - output_path: 目标 mp4 路径。
    返回值：无。
    异常说明：
    - FileNotFoundError: Remotion CLI 或 props JSON 不存在时抛出。
    - RuntimeError: 子进程执行失败或未生成输出文件时抛出。
    边界条件：当前固定调用 `src/index.ts` 作为 Remotion 工程入口。
    """
    normalized_project_dir = Path(remotion_project_dir).resolve()
    normalized_props_path = Path(props_json_path).resolve()
    normalized_output_path = Path(output_path).resolve()
    if not normalized_project_dir.exists():
        raise FileNotFoundError(f"Remotion 模板工程目录不存在：{normalized_project_dir}")
    if not normalized_props_path.exists():
        raise FileNotFoundError(f"模板请求 JSON 不存在：{normalized_props_path}")

    remotion_cli_path = _resolve_remotion_cli_path(normalized_project_dir)
    command = _build_remotion_render_command(
        remotion_cli_path=remotion_cli_path,
        remotion_project_dir=normalized_project_dir,
        composition_id=composition_id,
        props_json_path=normalized_props_path,
        output_path=normalized_output_path,
    )
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            command,
            cwd=str(normalized_project_dir),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as error:
        stderr_text = str(error.stderr or "").strip()
        stdout_text = str(error.stdout or "").strip()
        detail_text = stderr_text or stdout_text or "无额外输出"
        raise RuntimeError(
            "Remotion 模板渲染失败："
            f"composition_id={composition_id}，output_path={normalized_output_path}，detail={detail_text}"
        ) from error

    if not normalized_output_path.exists():
        raise RuntimeError(
            "Remotion 模板渲染失败：命令执行后未生成输出文件，"
            f"composition_id={composition_id}，output_path={normalized_output_path}"
        )


def _resolve_remotion_cli_path(remotion_project_dir: Path) -> Path:
    """
    功能说明：解析当前模板工程本地安装的 Remotion CLI 可执行文件路径。
    参数说明：
    - remotion_project_dir: Remotion 模板工程目录。
    返回值：
    - Path: 可执行文件绝对路径。
    异常说明：
    - FileNotFoundError: 未找到本地 Remotion CLI 时抛出。
    边界条件：Windows 优先查找 `remotion.CMD`，其他平台查找无扩展名文件。
    """
    bin_dir = remotion_project_dir / "node_modules" / ".bin"
    candidate_names = ["remotion.CMD", "remotion.cmd"] if os.name == "nt" else ["remotion"]
    for candidate_name in candidate_names:
        candidate_path = bin_dir / candidate_name
        if candidate_path.exists():
            return candidate_path.resolve()
    raise FileNotFoundError(
        "未找到 Remotion CLI 可执行文件。"
        f"请先在 {remotion_project_dir} 下安装依赖，预期路径={bin_dir}"
    )


def _build_remotion_render_command(
    *,
    remotion_cli_path: Path,
    remotion_project_dir: Path,
    composition_id: str,
    props_json_path: Path,
    output_path: Path,
) -> list[str]:
    """
    功能说明：构建 Remotion render 命令数组。
    参数说明：
    - remotion_cli_path: Remotion CLI 可执行文件路径。
    - remotion_project_dir: Remotion 模板工程目录。
    - composition_id: Composition 标识。
    - props_json_path: 正式模板请求 JSON 路径。
    - output_path: 目标 mp4 路径。
    返回值：
    - list[str]: 可直接交给 subprocess.run 的命令数组。
    异常说明：
    - ValueError: composition_id 为空时抛出。
    边界条件：当前固定入口文件为 `src/index.ts`。
    """
    normalized_composition_id = str(composition_id).strip()
    if not normalized_composition_id:
        raise ValueError("Remotion 渲染命令非法：composition_id 不得为空。")
    entry_path = remotion_project_dir / "src" / "index.ts"
    return [
        str(remotion_cli_path),
        "render",
        str(entry_path),
        normalized_composition_id,
        str(output_path),
        f"--props={props_json_path}",
    ]
