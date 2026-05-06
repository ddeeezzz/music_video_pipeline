"""
文件用途：加载模块B v2 各角色的单文件 prompt 模板，并解析其中的 system/user section。
核心流程：从 configs/prompts 读取模板文件 -> 按固定标题切分 section -> 执行字符串占位替换 -> 返回最终 prompt 消息。
输入输出：输入模板资源定义与替换字段，输出渲染后的 system prompt 与 user prompt。
依赖说明：依赖标准库 dataclasses、pathlib 与 re。
维护说明：每个角色只保留一个 Markdown prompt 文件，代码只维护变量填充与 section 解析。
"""

# 标准库：用于轻量数据结构定义。
from dataclasses import dataclass
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于 section 标题解析。
import re


@dataclass(frozen=True)
class PromptTemplateAsset:
    """
    功能说明：定义单个角色的单文件 prompt 模板资源。
    参数说明：
    - template_file: 模板路径。
    返回值：不适用。
    异常说明：不适用。
    边界条件：路径统一相对项目根目录解析。
    """

    template_file: str


@dataclass(frozen=True)
class RenderedPromptAsset:
    """
    功能说明：表示渲染后的 role prompt 消息。
    参数说明：
    - system_prompt: 渲染后的 system prompt。
    - user_prompt_markdown: 渲染后的 user Markdown prompt。
    返回值：不适用。
    异常说明：不适用。
    边界条件：两部分文本都已完成占位替换。
    """

    system_prompt: str
    user_prompt_markdown: str


# 常量：角色1 prompt 模板资源。
ROLE1_PROMPT_ASSET = PromptTemplateAsset(
    template_file="configs/prompts/module_b_v2.role1_visual_director.md",
)
# 常量：角色2 prompt 模板资源。
ROLE2_PROMPT_ASSET = PromptTemplateAsset(
    template_file="configs/prompts/module_b_v2.role2_big_segment_director.md",
)
# 常量：角色3 prompt 模板资源。
ROLE3_PROMPT_ASSET = PromptTemplateAsset(
    template_file="configs/prompts/module_b_v2.role3_segment_director.md",
)
# 常量：角色4 prompt 模板资源。
ROLE4_PROMPT_ASSET = PromptTemplateAsset(
    template_file="configs/prompts/module_b_v2.role4_prompt_builder.md",
)


def load_prompt_template(*, project_root: Path, template_file: str) -> str:
    """
    功能说明：读取单个 prompt 模板原文。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板相对路径或绝对路径。
    返回值：
    - str: 模板原文。
    异常说明：
    - FileNotFoundError: 模板不存在时抛出。
    边界条件：相对路径一律相对项目根目录解析。
    """
    template_path = Path(template_file)
    if not template_path.is_absolute():
        template_path = (project_root / template_path).resolve()
    return template_path.read_text(encoding="utf-8")


def render_prompt_template(*, project_root: Path, template_file: str, variables: dict[str, str]) -> str:
    """
    功能说明：读取模板文件并执行 `{{key}}` 占位替换。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板相对路径或绝对路径。
    - variables: 占位变量映射。
    返回值：
    - str: 渲染后的 Markdown 文本。
    异常说明：
    - FileNotFoundError: 模板不存在时抛出。
    边界条件：未提供的占位符保持原样，便于快速定位漏填变量。
    """
    template_text = load_prompt_template(project_root=project_root, template_file=template_file)
    rendered_text = template_text
    for key, value in variables.items():
        rendered_text = rendered_text.replace(f"{{{{{key}}}}}", str(value))
    return rendered_text


def parse_prompt_sections(template_text: str) -> tuple[str, str]:
    """
    功能说明：从单个 prompt 模板中解析 `# System` 与 `# User Template` 两个 section。
    参数说明：
    - template_text: 模板原文。
    返回值：
    - tuple[str, str]: 依次为 system prompt 与 user prompt 模板。
    异常说明：
    - ValueError: 缺失固定 section 或顺序非法时抛出。
    边界条件：section 标题必须严格使用一级标题。
    """
    normalized_text = str(template_text or "").replace("\r\n", "\n").strip()
    system_match = re.search(r"(?m)^# System\s*$", normalized_text)
    user_match = re.search(r"(?m)^# User Template\s*$", normalized_text)
    if system_match is None or user_match is None:
        raise ValueError("prompt 模板缺失固定 section：必须同时包含 `# System` 与 `# User Template`。")
    if system_match.start() > user_match.start():
        raise ValueError("prompt 模板 section 顺序非法：`# System` 必须在 `# User Template` 之前。")
    system_text = normalized_text[system_match.end() : user_match.start()].strip()
    user_text = normalized_text[user_match.end() :].strip()
    if not system_text or not user_text:
        raise ValueError("prompt 模板 section 为空：`# System` 与 `# User Template` 都必须包含正文。")
    return system_text, user_text


def render_prompt_asset(
    *,
    project_root: Path,
    prompt_asset: PromptTemplateAsset,
    user_variables: dict[str, str],
    system_variables: dict[str, str] | None = None,
) -> RenderedPromptAsset:
    """
    功能说明：统一渲染角色的 system/user prompt 模板。
    参数说明：
    - project_root: 项目根目录。
    - prompt_asset: 角色 prompt 资源定义。
    - user_variables: user prompt 占位变量。
    - system_variables: system prompt 占位变量，默认为空。
    返回值：
    - RenderedPromptAsset: 渲染后的 prompt 消息。
    异常说明：
    - FileNotFoundError: 任一模板不存在时抛出。
    边界条件：system/user 模板统一走相同替换规则。
    """
    template_text = load_prompt_template(project_root=project_root, template_file=prompt_asset.template_file)
    system_template, user_template = parse_prompt_sections(template_text)
    return RenderedPromptAsset(
        system_prompt=_render_inline_template(system_template, system_variables or {}),
        user_prompt_markdown=_render_inline_template(user_template, user_variables),
    )


def _render_inline_template(template_text: str, variables: dict[str, str]) -> str:
    """
    功能说明：对已截取的模板正文执行 `{{key}}` 占位替换。
    参数说明：
    - template_text: 模板正文。
    - variables: 占位变量映射。
    返回值：
    - str: 渲染后的文本。
    异常说明：无。
    边界条件：未提供的占位符保持原样。
    """
    rendered_text = template_text
    for key, value in variables.items():
        rendered_text = rendered_text.replace(f"{{{{{key}}}}}", str(value))
    return rendered_text
