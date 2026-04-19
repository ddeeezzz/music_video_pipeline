"""
文件用途：加载模块B外置Markdown Prompt模板并构建LLM请求消息。
核心流程：读取Markdown模板 -> 校验契约与占位符 -> 渲染messages数组。
输入输出：输入模板路径与分镜上下文，输出 chat completions messages 数组。
依赖说明：依赖标准库 json/pathlib/re。
维护说明：Prompt 文案必须由外置模板维护，不在代码内置默认值。
"""

# 标准库：用于JSON序列化与反序列化
import json
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则匹配
import re
# 标准库：用于类型提示
from typing import Any


# 常量：模板支持版本号。
PROMPT_TEMPLATE_VERSION = 1
# 常量：模板中输入JSON占位符。
INPUT_PAYLOAD_JSON_PLACEHOLDER = "{{input_payload_json}}"
# 常量：模板中用户自定义提示词占位符。
USER_CUSTOM_PROMPT_PLACEHOLDER = "{{user_custom_prompt}}"
# 常量：模板中补救提示占位符。
RETRY_HINT_PLACEHOLDER = "{{retry_hint}}"
# 常量：模板必填字段集合。
PROMPT_TEMPLATE_REQUIRED_KEYS = {
    "version",
    "system_prompt",
    "user_prompt_template",
    "retry_hint_template",
}
# 常量：Markdown二级标题匹配规则。
MARKDOWN_SECTION_PATTERN = re.compile(
    r"^\s*##\s*(system_prompt|user_prompt_template|retry_hint_template)\s*$",
    flags=re.IGNORECASE,
)


class ModuleBLlmPromptTemplateError(RuntimeError):
    """模块B Prompt模板加载或渲染异常。"""


class ModuleBPromptTemplate(dict[str, str]):
    """模块B Prompt模板结构。"""


def load_module_b_prompt_template(project_root: Path, prompt_template_file: str) -> ModuleBPromptTemplate:
    """
    功能说明：加载并校验模块B外置Prompt模板。
    参数说明：
    - project_root: 项目根目录，用于解析相对模板路径。
    - prompt_template_file: 模板文件路径（支持相对路径）。
    返回值：
    - ModuleBPromptTemplate: 已校验模板对象。
    异常说明：
    - ModuleBLlmPromptTemplateError: 路径、Markdown格式、字段或占位符非法时抛出。
    边界条件：模板 version 当前仅支持 1。
    """
    normalized_path_text = str(prompt_template_file).strip()
    if not normalized_path_text:
        raise ModuleBLlmPromptTemplateError("module_b.llm.prompt_template_file 不能为空。")

    template_path = Path(normalized_path_text)
    if not template_path.is_absolute():
        template_path = (project_root / template_path).resolve()

    if not template_path.exists():
        raise ModuleBLlmPromptTemplateError(f"Prompt模板文件不存在：{template_path}")
    if not template_path.is_file():
        raise ModuleBLlmPromptTemplateError(f"Prompt模板路径不是文件：{template_path}")

    try:
        template_text = template_path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ModuleBLlmPromptTemplateError(f"Prompt模板读取失败：{template_path}，错误={error}") from error

    template_ext = template_path.suffix.lower()
    if template_ext in {".md", ".markdown"}:
        parsed_template = _parse_markdown_prompt_template(template_text=template_text, template_path=template_path)
    else:
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板格式不支持：{template_path}，仅支持 .md/.markdown"
        )

    return _validate_and_build_template(parsed_template=parsed_template, template_path=template_path)


def build_module_b_prompt_messages(
    input_payload: dict[str, Any],
    prompt_template: ModuleBPromptTemplate,
    user_custom_prompt: str = "",
    retry_hint: str = "",
) -> list[dict[str, str]]:
    """
    功能说明：基于外置模板构建模块B真实LLM请求 messages。
    参数说明：
    - input_payload: 当前分镜输入上下文字典。
    - prompt_template: 已校验模板对象。
    - retry_hint: 解析失败后的补救提示（可选）。
    返回值：
    - list[dict[str, str]]: chat completions 标准消息数组。
    异常说明：
    - ModuleBLlmPromptTemplateError: 模板对象缺失关键字段时抛出。
    边界条件：仅在 retry_hint 非空时附加补救提示段。
    """
    system_prompt = str(prompt_template.get("system_prompt", "")).strip()
    user_prompt_template = str(prompt_template.get("user_prompt_template", "")).strip()
    retry_hint_template = str(prompt_template.get("retry_hint_template", "")).strip()
    if not system_prompt or not user_prompt_template or not retry_hint_template:
        raise ModuleBLlmPromptTemplateError("Prompt模板对象不完整，无法构建消息。")

    payload_text = json.dumps(input_payload, ensure_ascii=False, separators=(",", ":"))
    system_prompt = system_prompt.replace(INPUT_PAYLOAD_JSON_PLACEHOLDER, payload_text)
    user_prompt = user_prompt_template.replace(USER_CUSTOM_PROMPT_PLACEHOLDER, str(user_custom_prompt))
    normalized_retry_hint = str(retry_hint).strip()
    if normalized_retry_hint:
        retry_line = retry_hint_template.replace(RETRY_HINT_PLACEHOLDER, normalized_retry_hint)
        user_prompt = f"{user_prompt}\n{retry_line}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_markdown_prompt_template(template_text: str, template_path: Path) -> dict[str, Any]:
    """
    功能说明：解析Markdown模板文本。
    参数说明：
    - template_text: 模板原始文本。
    - template_path: 模板路径（用于错误定位）。
    返回值：
    - dict[str, Any]: 转换后的模板对象（含固定 version=1）。
    异常说明：
    - ModuleBLlmPromptTemplateError: 段落缺失或重复时抛出。
    边界条件：通过二级标题 `## <section_name>` 切分段落内容。
    """
    sections: dict[str, str] = {}
    current_section_key = ""
    current_buffer: list[str] = []

    for raw_line in str(template_text).splitlines():
        matched = MARKDOWN_SECTION_PATTERN.match(raw_line)
        if matched:
            if current_section_key:
                sections[current_section_key] = "\n".join(current_buffer).strip()
            normalized_section_key = str(matched.group(1)).strip().lower()
            if normalized_section_key in sections:
                raise ModuleBLlmPromptTemplateError(
                    f"Markdown模板段落重复：{template_path}，section={normalized_section_key}"
                )
            current_section_key = normalized_section_key
            current_buffer = []
            continue

        if current_section_key:
            current_buffer.append(raw_line)

    if current_section_key:
        sections[current_section_key] = "\n".join(current_buffer).strip()

    if not sections:
        raise ModuleBLlmPromptTemplateError(
            f"Markdown模板缺少有效段落：{template_path}，需要使用 ## system_prompt / ## user_prompt_template / ## retry_hint_template"
        )

    return {
        "version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": sections.get("system_prompt", ""),
        "user_prompt_template": sections.get("user_prompt_template", ""),
        "retry_hint_template": sections.get("retry_hint_template", ""),
    }


def _validate_and_build_template(parsed_template: dict[str, Any], template_path: Path) -> ModuleBPromptTemplate:
    """
    功能说明：校验模板结构并转换为内部对象。
    参数说明：
    - parsed_template: JSON/Markdown解析后的模板对象。
    - template_path: 模板路径（用于错误定位）。
    返回值：
    - ModuleBPromptTemplate: 已校验模板对象。
    异常说明：
    - ModuleBLlmPromptTemplateError: 字段集合、版本或占位符非法时抛出。
    边界条件：字段集合必须与契约完全一致。
    """
    actual_keys = set(parsed_template.keys())
    if actual_keys != PROMPT_TEMPLATE_REQUIRED_KEYS:
        missing_keys = sorted(PROMPT_TEMPLATE_REQUIRED_KEYS - actual_keys)
        extra_keys = sorted(actual_keys - PROMPT_TEMPLATE_REQUIRED_KEYS)
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板字段不匹配：{template_path}，missing={missing_keys}，extra={extra_keys}"
        )

    version_value = parsed_template.get("version")
    if not isinstance(version_value, int) or int(version_value) != PROMPT_TEMPLATE_VERSION:
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板 version 非法：{template_path}，期望={PROMPT_TEMPLATE_VERSION}，实际={version_value}"
        )

    system_prompt = _normalize_non_empty_template_text(
        field_name="system_prompt",
        value=parsed_template.get("system_prompt"),
        template_path=template_path,
    )
    user_prompt_template = _normalize_non_empty_template_text(
        field_name="user_prompt_template",
        value=parsed_template.get("user_prompt_template"),
        template_path=template_path,
    )
    retry_hint_template = _normalize_non_empty_template_text(
        field_name="retry_hint_template",
        value=parsed_template.get("retry_hint_template"),
        template_path=template_path,
    )

    if INPUT_PAYLOAD_JSON_PLACEHOLDER not in system_prompt:
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板缺失占位符 {INPUT_PAYLOAD_JSON_PLACEHOLDER}：{template_path}"
        )
    if USER_CUSTOM_PROMPT_PLACEHOLDER not in user_prompt_template:
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板缺失占位符 {USER_CUSTOM_PROMPT_PLACEHOLDER}：{template_path}"
        )
    if RETRY_HINT_PLACEHOLDER not in retry_hint_template:
        raise ModuleBLlmPromptTemplateError(
            f"Prompt模板缺失占位符 {RETRY_HINT_PLACEHOLDER}：{template_path}"
        )

    return ModuleBPromptTemplate(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        retry_hint_template=retry_hint_template,
        source_path=str(template_path),
    )


def _normalize_non_empty_template_text(field_name: str, value: object, template_path: Path) -> str:
    """
    功能说明：校验模板文本字段为非空字符串。
    参数说明：
    - field_name: 字段名。
    - value: 字段值。
    - template_path: 模板路径（用于错误定位）。
    返回值：
    - str: 去首尾空白后的文本。
    异常说明：
    - ModuleBLlmPromptTemplateError: 字段类型或内容非法时抛出。
    边界条件：空白字符串视为非法。
    """
    if not isinstance(value, str):
        raise ModuleBLlmPromptTemplateError(f"Prompt模板字段类型非法：{template_path}，{field_name} 必须是字符串。")
    normalized = value.strip()
    if not normalized:
        raise ModuleBLlmPromptTemplateError(f"Prompt模板字段为空：{template_path}，{field_name} 不能为空。")
    return normalized
