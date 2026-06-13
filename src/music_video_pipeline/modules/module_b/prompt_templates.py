"""
文件用途：提供模块 B 的 prompt 模板加载与渲染函数。
核心流程：读取模板文件并根据变量生成角色 prompt 资产。
输入输出：输入模板路径与变量，输出渲染后的 prompt 资产。
依赖说明：依赖标准库 pathlib、re、typing。
维护说明：模板字段名与角色 prompt 变量应保持一致。
"""

# 标准库：用于定义轻量不可变数据类。
from dataclasses import dataclass
# 标准库：用于文件路径处理。
from pathlib import Path
# 标准库：用于正则匹配固定标题。
import re
# 标准库：用于共享字典类型标注。
from typing import TypedDict


class PromptAsset(TypedDict):
    """
    功能说明：表示渲染后的 prompt 文本对。
    参数说明：TypedDict 无显式参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：system 与 user 两段都应是可直接发给模型的完整文本。
    """

    system_prompt: str
    user_prompt_markdown: str


@dataclass(frozen=True)
class PromptTemplateRef:
    """
    功能说明：表示 prompt 模板文件引用。
    参数说明：
    - template_file: 模板文件相对路径或绝对路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：路径解析由当前文件内的模板加载函数负责。
    """

    template_file: str


ROLE1_PROMPT_TEMPLATE_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role1_visual_director.md",
)

ROLE2_PROMPT_TEMPLATE_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role2_big_segment_director.md",
)

ROLE2A_PROMPT_TEMPLATE_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role2a_story_brainstorm.md",
)

ROLE3_PROMPT_TEMPLATE_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role3_segment_director.md",
)

ROLE4_PROMPT_HUMAN_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role4_human.md",
)

ROLE4_PROMPT_ANIMAL_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role4_animal.md",
)

ROLE4_PROMPT_OBJECT_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role4_object.md",
)

ROLE4_PROMPT_SCENE_REF = PromptTemplateRef(
    template_file="configs/prompts/module_b.role4_scene.md",
)

# 查询映射：subject_kind → PromptTemplateRef
ROLE4_PROMPT_MAP: dict[str, PromptTemplateRef] = {
    "human": ROLE4_PROMPT_HUMAN_REF,
    "animal": ROLE4_PROMPT_ANIMAL_REF,
    "object": ROLE4_PROMPT_OBJECT_REF,
    "scene": ROLE4_PROMPT_SCENE_REF,
}


def load_prompt_template(project_root: Path, template_file: str) -> str:
    """
    功能说明：读取模块 B 的 prompt 模板原文。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径。
    返回值：
    - str: 模板原文。
    异常说明：
    - FileNotFoundError: 模板不存在时抛出。
    边界条件：相对路径统一相对项目根目录解析。
    """
    template_path = Path(str(template_file).strip())
    if not template_path.is_absolute():
        template_path = (project_root / template_path).resolve()
    return template_path.read_text(encoding="utf-8")


def parse_prompt_sections(template_text: str) -> tuple[str, str]:
    """
    功能说明：从单个 prompt 模板中严格解析 system 与 user 两个 section。
    参数说明：
    - template_text: 模板原文。
    返回值：
    - tuple[str, str]: 依次为 system prompt 与 user prompt 模板。
    异常说明：
    - ValueError: 缺失固定 section 或顺序非法时抛出。
    边界条件：section 标题必须严格使用一级标题。
    """
    normalized_text = str(template_text or "").replace("\r\n", "\n").strip()
    system_match = re.search(r"(?m)^# System Prompt\s*$", normalized_text)
    user_match = re.search(r"(?m)^# User Prompt\s*$", normalized_text)
    if system_match is None or user_match is None:
        raise ValueError("prompt 模板缺失固定 section：必须同时包含 `# System Prompt` 与 `# User Prompt`。")
    if system_match.start() > user_match.start():
        raise ValueError("prompt 模板 section 顺序非法：`# System Prompt` 必须在 `# User Prompt` 之前。")
    system_text = normalized_text[system_match.end() : user_match.start()].strip()
    user_text = normalized_text[user_match.end() :].strip()
    if not system_text or not user_text:
        raise ValueError("prompt 模板 section 为空：`# System Prompt` 与 `# User Prompt` 都必须包含正文。")
    return system_text, user_text


def render_prompt_asset(
    *,
    project_root: Path,
    prompt_template_ref: PromptTemplateRef,
    user_variables: dict[str, str],
    system_variables: dict[str, str] | None = None,
) -> PromptAsset:
    """
    功能说明：渲染模块 B 单文件 prompt 模板。
    参数说明：
    - project_root: 项目根目录。
    - prompt_template_ref: prompt 模板引用。
    - user_variables: user prompt 占位变量。
    - system_variables: system prompt 占位变量，默认为空。
    返回值：
    - PromptAsset: 渲染后的 system/user prompt 文本。
    异常说明：
    - FileNotFoundError: 模板不存在时抛出。
    - ValueError: 模板 section 不合法时抛出。
    边界条件：未提供的占位符保持原样，便于定位漏填变量。
    """
    template_text = load_prompt_template(
        project_root=project_root,
        template_file=prompt_template_ref.template_file,
    )
    system_template, user_template = parse_prompt_sections(template_text)
    return {
        "system_prompt": _render_inline_template(system_template, system_variables or {}),
        "user_prompt_markdown": _render_inline_template(user_template, user_variables),
    }


def _render_inline_template(template_text: str, variables: dict[str, str]) -> str:
    """
    功能说明：对模板正文执行 `{{key}}` 占位替换。
    参数说明：
    - template_text: 模板正文。
    - variables: 占位变量映射。
    返回值：
    - str: 渲染后的文本。
    异常说明：无。
    边界条件：未提供的占位符保持原样。
    """
    rendered_text = str(template_text or "")
    for key, value in variables.items():
        rendered_text = rendered_text.replace(f"{{{{{key}}}}}", str(value))
    return rendered_text
