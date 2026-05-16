"""
文件用途：加载并编译模块B v2 的编排模板 Markdown 文档。
核心流程：读取 Markdown -> 解析 3 个二级标题文本块 -> 生成精简模板对象。
输入输出：输入模板路径，输出标准化编排模板字典。
依赖说明：依赖标准库 pathlib/json/re 与项目内 parser。
维护说明：模板作者只维护“故事 / 意象 / remotion模板”三个二级标题文本块。
"""

# 标准库：用于 JSON 序列化产物。
import json
# 标准库：用于解析 Markdown 标题与单行条目。
import re
# 标准库：用于路径处理。
from pathlib import Path

# 项目内模块：导入默认模板路径。
from music_video_pipeline.modules.module_b_v2.models import DEFAULT_STORYBOARD_TEMPLATE_FILE
# 项目内模块：导入模板校验器。
from music_video_pipeline.modules.module_b_v2.parser import ModuleBV2ParseError, validate_storyboard_template


# 常量：模板中的“故事”标题名。
SECTION_STORY = "故事"
# 常量：模板中的“意象”标题名。
SECTION_IMAGERY = "意象"
# 常量：模板中的“remotion模板”标题名。
SECTION_REMOTION_TEMPLATES = "remotion模板"
# 常量：二级标题解析正则。
LEVEL2_HEADING_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")
def load_storyboard_template(project_root: Path, template_file: str = DEFAULT_STORYBOARD_TEMPLATE_FILE) -> dict:
    """
    功能说明：加载并校验编排模板 Markdown 文件。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径，支持相对路径。
    返回值：
    - dict: 已通过校验的模板对象。
    异常说明：
    - ModuleBV2ParseError: 路径不存在或模板结构非法时抛出。
    边界条件：读取时兼容 UTF-8 与 UTF-8 BOM。
    """
    normalized_path = resolve_storyboard_template_path(project_root=project_root, template_file=template_file)
    if not normalized_path.exists():
        raise ModuleBV2ParseError(f"编排模板文件不存在：{normalized_path}")
    markdown_text = normalized_path.read_text(encoding="utf-8-sig")
    template_payload = _extract_storyboard_template_payload(markdown_text=markdown_text, template_path=normalized_path)
    return validate_storyboard_template(template_payload)


def resolve_storyboard_template_path(project_root: Path, template_file: str = DEFAULT_STORYBOARD_TEMPLATE_FILE) -> Path:
    """
    功能说明：将模板文件路径解析为绝对路径。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径，支持相对路径。
    返回值：
    - Path: 归一化后的绝对路径。
    异常说明：无。
    边界条件：只做路径解析，不校验文件存在性。
    """
    normalized_path = Path(str(template_file).strip())
    if not normalized_path.is_absolute():
        normalized_path = (project_root / normalized_path).resolve()
    return normalized_path


def dump_storyboard_template_artifact(template_payload: dict, artifact_path: Path) -> None:
    """
    功能说明：将已编译模板写入任务产物路径。
    参数说明：
    - template_payload: 已校验模板对象。
    - artifact_path: 目标写入路径。
    返回值：无。
    异常说明：文件写入失败时抛出 OSError。
    边界条件：会自动创建父目录。
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(template_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _extract_storyboard_template_payload(markdown_text: str, template_path: Path) -> dict:
    """
    功能说明：从 Markdown 文本中提取三段式模板对象。
    参数说明：
    - markdown_text: 原始 Markdown 文本。
    - template_path: 模板路径，用于报错定位。
    返回值：
    - dict: 待校验的模板对象。
    异常说明：
    - ModuleBV2ParseError: 缺失必需 section 或 section 为空时抛出。
    边界条件：只识别“故事 / 意象 / remotion模板”三个二级标题。
    """
    section_map = _parse_level2_sections(markdown_text)
    story_text = _require_section_text(section_map, SECTION_STORY, template_path)
    imagery_text = _require_section_text(section_map, SECTION_IMAGERY, template_path)
    remotion_templates_text = _require_section_text(section_map, SECTION_REMOTION_TEMPLATES, template_path)

    return {
        "template_id": "storyboard_template_v1_simple",
        "story": {"premise_zh": story_text},
        "imagery": imagery_text,
        "remotion_templates": remotion_templates_text,
    }


def _parse_level2_sections(markdown_text: str) -> dict[str, str]:
    """
    功能说明：解析 Markdown 中的二级标题文本块。
    参数说明：
    - markdown_text: 原始 Markdown 文本。
    返回值：
    - dict[str, str]: 标题名到正文文本的映射。
    异常说明：无。
    边界条件：相同标题若重复出现，后出现的内容会覆盖前值。
    """
    normalized_text = str(markdown_text or "").replace("\r\n", "\n").strip()
    matches = list(LEVEL2_HEADING_PATTERN.finditer(normalized_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start_index = match.end()
        end_index = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        sections[heading] = normalized_text[start_index:end_index].strip()
    return sections


def _require_section_text(section_map: dict[str, str], section_name: str, template_path: Path) -> str:
    """
    功能说明：读取并校验必需 section 的正文文本。
    参数说明：
    - section_map: 标题到正文的映射。
    - section_name: 必需标题名。
    - template_path: 模板路径，用于报错定位。
    返回值：
    - str: 去除首尾空白后的正文文本。
    异常说明：
    - ModuleBV2ParseError: 缺失或正文为空时抛出。
    边界条件：仅校验单个 section，不校验整体结构。
    """
    value = str(section_map.get(section_name, "")).strip()
    if not value:
        raise ModuleBV2ParseError(f"编排模板缺失 `## {section_name}` section：{template_path}")
    return value

