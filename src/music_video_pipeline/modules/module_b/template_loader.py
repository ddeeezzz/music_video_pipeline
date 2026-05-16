"""
文件用途：提供模块 B 编排模板原文的读取与落盘函数。
核心流程：解析模板路径，原样读取 Markdown，必要时原样写出产物。
输入输出：输入项目路径与模板路径，输出模板原文字符串。
依赖说明：依赖标准库 pathlib。
维护说明：这里不再拆解模板 section，也不把模板洗成 JSON。
"""

# 标准库：用于文件路径解析。
from pathlib import Path


class ModuleBTemplateError(RuntimeError):
    """模块 B 模板读取异常。"""


def resolve_storyboard_template_path(project_root: Path, template_file: str) -> Path:
    """
    功能说明：解析编排模板文件路径。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径。
    返回值：
    - Path: 模板文件绝对路径。
    异常说明：按具体实现定义。
    边界条件：相对路径应以项目根目录为基准解析。
    """
    resolved_path = Path(str(template_file).strip())
    if not resolved_path.is_absolute():
        resolved_path = (project_root / resolved_path).resolve()
    return resolved_path


def load_storyboard_template(project_root: Path, template_file: str) -> str:
    """
    功能说明：加载模块 B 编排模板原文。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径。
    返回值：
    - str: 编排模板原文。
    异常说明：
    - ModuleBTemplateError: 模板不存在或内容为空时抛出。
    边界条件：只做非空字符串保护，不解析内部 Markdown 结构。
    """
    template_path = resolve_storyboard_template_path(project_root=project_root, template_file=template_file)
    if not template_path.exists():
        raise ModuleBTemplateError(f"编排模板文件不存在：{template_path}")
    markdown_text = template_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    if not markdown_text:
        raise ModuleBTemplateError(f"编排模板文件内容为空：{template_path}")
    return markdown_text


def dump_storyboard_template_artifact(template_markdown: str, artifact_path: Path) -> None:
    """
    功能说明：写出模块 B 编排模板原文产物。
    参数说明：
    - template_markdown: 模板原文。
    - artifact_path: 产物写入路径。
    返回值：无。
    异常说明：
    - ModuleBTemplateError: 原文为空时抛出。
    边界条件：输出保持 Markdown 原样，只补一个结尾换行。
    """
    normalized_text = str(template_markdown or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        raise ModuleBTemplateError("编排模板原文不能为空。")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(normalized_text + "\n", encoding="utf-8")
