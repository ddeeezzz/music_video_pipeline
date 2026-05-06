"""
文件用途：提供模块B v2 统一的 Markdown 渲染与解析工具。
核心流程：将精简后的输入对象渲染为 Markdown；将 LLM 返回的 Markdown 解析为统一树结构。
输入输出：输入 Python 字典/列表，输出 Markdown 文本或解析后的文档结构。
依赖说明：依赖标准库 re 与 dataclasses。
维护说明：角色 prompt 与返回格式统一走本文件，避免各角色重复维护同类逻辑。
"""

# 标准库：用于轻量数据结构定义。
from dataclasses import dataclass, field
# 标准库：用于可调用类型提示。
from collections.abc import Callable
# 标准库：用于 Markdown 标题解析。
import re
from typing import Any


@dataclass(frozen=True)
class MarkdownNode:
    """
    功能说明：表示一个 Markdown 标题节点。
    参数说明：
    - heading: 当前标题文本。
    - body: 当前标题下的原始正文。
    - fields: 当前标题下解析出的 `- key: value` 字段。
    - subsections: 下一级标题节点数组。
    返回值：不适用。
    异常说明：不适用。
    边界条件：body 不包含下一级标题正文。
    """

    heading: str
    body: str
    fields: dict[str, str] = field(default_factory=dict)
    subsections: list["MarkdownNode"] = field(default_factory=list)


@dataclass(frozen=True)
class MarkdownDocument:
    """
    功能说明：表示解析后的 Markdown 文档。
    参数说明：
    - title: 一级标题文本。
    - sections: 二级标题节点数组。
    返回值：不适用。
    异常说明：不适用。
    边界条件：一级标题可为空字符串。
    """

    title: str
    sections: list[MarkdownNode]


@dataclass(frozen=True)
class MarkdownFieldSchema:
    """
    功能说明：定义从源对象提取一个 Markdown 字段的规则。
    参数说明：
    - key: 输出字段名。
    - path: 源对象取值路径，使用 `.` 分隔；为空时表示整对象。
    - default: 取值失败时的默认值。
    - transform: 可选，取值后执行的转换函数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：当 transform 抛错时由调用方统一感知。
    """

    key: str
    path: str | None = None
    default: Any = None
    transform: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class MarkdownSectionSchema:
    """
    功能说明：定义一个 Markdown 标题块的渲染规则。
    参数说明：
    - heading: 标题文本。
    - field_schema: 字段提取 schema 列表。
    - body: 可选，正文文本。
    - subsections: 可选，子块文本数组。
    - level: 标题级别。
    返回值：不适用。
    异常说明：不适用。
    边界条件：字段顺序与 schema 顺序一致。
    """

    heading: str
    field_schema: list[MarkdownFieldSchema] = field(default_factory=list)
    body: str = ""
    subsections: list[str] = field(default_factory=list)
    level: int = 2


@dataclass(frozen=True)
class MarkdownLineSchema:
    """
    功能说明：定义单行目录/候选项的渲染规则。
    参数说明：
    - id_path: 主键取值路径，使用 `.` 分隔。
    - detail_schema: 行内附加字段 schema 列表。
    - detail_separator: 附加字段之间的分隔符。
    - detail_with_key: 是否以 `key=value` 形式输出附加字段。
    返回值：不适用。
    异常说明：不适用。
    边界条件：detail_schema 为空时只输出主键。
    """

    id_path: str
    detail_schema: list[MarkdownFieldSchema] = field(default_factory=list)
    detail_separator: str = " | "
    detail_with_key: bool = False


def render_scalar(value: Any) -> str:
    """
    功能说明：将常见标量值渲染为 Markdown 字段值。
    参数说明：
    - value: 任意输入值。
    返回值：
    - str: 标准化后的字符串。
    异常说明：无。
    边界条件：空列表与空字符串统一返回 none。
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "none"
    normalized = str(value).strip()
    return normalized or "none"


def render_bullet_fields(field_map: dict[str, Any], order: list[str] | None = None) -> str:
    """
    功能说明：将字典渲染为 `- key: value` 列表。
    参数说明：
    - field_map: 输入字段映射。
    - order: 可选，指定字段渲染顺序。
    返回值：
    - str: 多行 Markdown 文本。
    异常说明：无。
    边界条件：未在 order 中声明但存在的字段会按原字典顺序追加。
    """
    emitted_keys: set[str] = set()
    lines: list[str] = []
    for key in order or []:
        if key not in field_map:
            continue
        emitted_keys.add(key)
        lines.append(f"- {key}: {render_scalar(field_map.get(key))}")
    for key, value in field_map.items():
        if key in emitted_keys:
            continue
        lines.append(f"- {key}: {render_scalar(value)}")
    return "\n".join(lines).strip()


def build_field_map_from_schema(source: Any, field_schema: list[MarkdownFieldSchema]) -> dict[str, Any]:
    """
    功能说明：按 schema 从源对象提取字段并构建字段映射。
    参数说明：
    - source: 源对象，通常为字典。
    - field_schema: 字段 schema 列表。
    返回值：
    - dict[str, Any]: 渲染前字段映射。
    异常说明：无。
    边界条件：path 解析失败时使用 default。
    """
    result: dict[str, Any] = {}
    for item in field_schema:
        raw_value = _resolve_path_value(source=source, path=item.path, default=item.default)
        if item.transform is not None:
            raw_value = item.transform(raw_value)
        result[item.key] = raw_value
    return result


def render_schema_fields(source: Any, field_schema: list[MarkdownFieldSchema]) -> str:
    """
    功能说明：按 schema 渲染 `- key: value` 字段列表。
    参数说明：
    - source: 源对象。
    - field_schema: 字段 schema 列表。
    返回值：
    - str: 渲染后的字段列表文本。
    异常说明：无。
    边界条件：字段顺序与 schema 顺序一致。
    """
    field_map = build_field_map_from_schema(source=source, field_schema=field_schema)
    return render_bullet_fields(field_map, order=[item.key for item in field_schema])


def render_heading_block(
    *,
    level: int,
    heading: str,
    field_map: dict[str, Any] | None = None,
    body: str = "",
    subsections: list[str] | None = None,
    order: list[str] | None = None,
) -> str:
    """
    功能说明：渲染统一的标题块。
    参数说明：
    - level: 标题级别，仅支持 2 或 3。
    - heading: 标题文本。
    - field_map: 可选，键值字段。
    - body: 可选，正文文本。
    - subsections: 可选，下一级子块文本列表。
    - order: 可选，字段顺序。
    返回值：
    - str: 当前标题块完整 Markdown。
    异常说明：
    - ValueError: 标题级别非法时抛出。
    边界条件：空字段与空正文会自动跳过。
    """
    if level not in {2, 3}:
        raise ValueError(f"不支持的 Markdown 标题级别：{level}")
    prefix = "#" * level
    parts = [f"{prefix} {str(heading).strip()}"]
    rendered_fields = render_bullet_fields(field_map or {}, order=order)
    if rendered_fields:
        parts.append(rendered_fields)
    if str(body).strip():
        parts.append(str(body).strip())
    for subsection in subsections or []:
        if str(subsection).strip():
            parts.append(str(subsection).strip())
    return "\n".join(parts).strip()


def render_section_from_schema(section_schema: MarkdownSectionSchema, source: Any) -> str:
    """
    功能说明：根据 section schema 与源对象渲染标题块。
    参数说明：
    - section_schema: 标题块 schema。
    - source: 源对象。
    返回值：
    - str: 标题块 Markdown。
    异常说明：无。
    边界条件：heading 由 schema 固定提供。
    """
    return render_heading_block(
        level=int(section_schema.level),
        heading=section_schema.heading,
        field_map=build_field_map_from_schema(source=source, field_schema=section_schema.field_schema),
        body=section_schema.body,
        subsections=section_schema.subsections,
        order=[item.key for item in section_schema.field_schema],
    )


def render_catalog_lines(
    items: list[Any],
    *,
    id_path: str,
    label_path: str,
    empty_text: str = "- 无",
) -> str:
    """
    功能说明：渲染简单的 `- id: label` 目录列表。
    参数说明：
    - items: 条目数组。
    - id_path: 主键取值路径。
    - label_path: 文本标签取值路径。
    - empty_text: 空列表时的占位文本。
    返回值：
    - str: 目录 Markdown。
    异常说明：无。
    边界条件：非字典条目会被跳过。
    """
    return render_compound_lines(
        items=items,
        line_schema=MarkdownLineSchema(
            id_path=id_path,
            detail_schema=[MarkdownFieldSchema("label", label_path, "none")],
            detail_separator="",
            detail_with_key=False,
        ),
        empty_text=empty_text,
    )


def render_compound_lines(
    items: list[Any],
    *,
    line_schema: MarkdownLineSchema,
    empty_text: str = "- 无",
) -> str:
    """
    功能说明：按统一 schema 渲染 `- id: detail` 多行列表。
    参数说明：
    - items: 条目数组。
    - line_schema: 行渲染规则。
    - empty_text: 空列表时的占位文本。
    返回值：
    - str: 多行 Markdown。
    异常说明：无。
    边界条件：detail_schema 为空时仅输出 `- id`。
    """
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = render_scalar(_resolve_path_value(source=item, path=line_schema.id_path, default="none"))
        details = build_field_map_from_schema(source=item, field_schema=line_schema.detail_schema)
        detail_parts: list[str] = []
        for detail_item in line_schema.detail_schema:
            detail_value = render_scalar(details.get(detail_item.key))
            if line_schema.detail_with_key:
                detail_parts.append(f"{detail_item.key}={detail_value}")
            else:
                detail_parts.append(detail_value)
        if detail_parts:
            lines.append(f"- {item_id}: {line_schema.detail_separator.join(detail_parts)}")
        else:
            lines.append(f"- {item_id}")
    return "\n".join(lines) if lines else empty_text


def render_repeated_sections(
    items: list[Any],
    *,
    heading_builder: Callable[[dict[str, Any], int], str],
    field_schema: list[MarkdownFieldSchema],
    level: int = 3,
    empty_text: str = "",
) -> str:
    """
    功能说明：按统一 schema 渲染重复的小标题块。
    参数说明：
    - items: 条目数组。
    - heading_builder: 根据条目与序号生成标题文本。
    - field_schema: 每个小标题块的字段 schema。
    - level: 标题级别。
    - empty_text: 无有效条目时的占位文本。
    返回值：
    - str: 小标题块拼接后的 Markdown。
    异常说明：无。
    边界条件：序号从 1 开始。
    """
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        blocks.append(
            render_heading_block(
                level=level,
                heading=heading_builder(item, index),
                field_map=build_field_map_from_schema(source=item, field_schema=field_schema),
                order=[schema.key for schema in field_schema],
            )
        )
    return "\n\n".join(blocks) if blocks else empty_text


def render_document(*, title: str, blocks: list[str]) -> str:
    """
    功能说明：渲染完整 Markdown 文档。
    参数说明：
    - title: 一级标题文本。
    - blocks: 二级块文本数组。
    返回值：
    - str: 完整 Markdown 文本。
    异常说明：无。
    边界条件：空块会被过滤。
    """
    normalized_blocks = [str(block).strip() for block in blocks if str(block).strip()]
    parts = [f"# {str(title).strip()}"] if str(title).strip() else []
    parts.extend(normalized_blocks)
    return "\n\n".join(parts).strip() + "\n"


def parse_markdown_document(text: str) -> MarkdownDocument:
    """
    功能说明：解析角色返回的 Markdown 文档。
    参数说明：
    - text: 原始 Markdown 文本。
    返回值：
    - MarkdownDocument: 解析后的文档结构。
    异常说明：
    - ValueError: 找不到二级标题时抛出。
    边界条件：一级标题可选；解析时忽略空行和普通说明行。
    """
    normalized_text = str(text or "").replace("\r\n", "\n").strip()
    title_match = re.search(r"(?m)^#\s+([^\n#]+?)\s*$", normalized_text)
    title = title_match.group(1).strip() if title_match else ""
    sections = _parse_level_nodes(text=normalized_text, level=2)
    return MarkdownDocument(title=title, sections=sections)


def _parse_level_nodes(*, text: str, level: int) -> list[MarkdownNode]:
    """
    功能说明：解析指定级别的标题节点。
    参数说明：
    - text: 当前块文本。
    - level: 目标标题级别。
    返回值：
    - list[MarkdownNode]: 节点数组。
    异常说明：无。
    边界条件：没有对应级别标题时返回空数组。
    """
    prefix = "#" * level
    next_prefix = "#" * (level + 1)
    pattern = rf"(?m)^{re.escape(prefix)}\s+([^\n#]+?)\s*$"
    matches = list(re.finditer(pattern, text))
    nodes: list[MarkdownNode] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start_index = match.end()
        end_index = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_body = text[start_index:end_index].strip()
        child_nodes = _parse_level_nodes(text=raw_body, level=level + 1) if level < 3 else []
        body_without_children = raw_body
        if child_nodes:
            child_match = re.search(rf"(?m)^{re.escape(next_prefix)}\s+[^\n#]+?\s*$", raw_body)
            if child_match is not None:
                body_without_children = raw_body[: child_match.start()].strip()
        nodes.append(
            MarkdownNode(
                heading=heading,
                body=body_without_children,
                fields=parse_bullet_fields(body_without_children),
                subsections=child_nodes,
            )
        )
    return nodes


def parse_bullet_fields(text: str) -> dict[str, str]:
    """
    功能说明：解析 `- key: value` 字段。
    参数说明：
    - text: 当前标题节点正文。
    返回值：
    - dict[str, str]: 字段映射。
    异常说明：无。
    边界条件：忽略空行和非键值行。
    """
    field_map: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        key_part, separator, value_part = line[2:].partition(":")
        if not separator:
            continue
        key = key_part.strip()
        if not key or key in field_map:
            continue
        field_map[key] = value_part.strip()
    return field_map


def _resolve_path_value(source: Any, path: str | None, default: Any) -> Any:
    """
    功能说明：从源对象中按点路径取值。
    参数说明：
    - source: 源对象。
    - path: 点路径，如 `style.color_mode`。
    - default: 默认值。
    返回值：
    - Any: 提取到的值或默认值。
    异常说明：无。
    边界条件：仅支持字典链式取值。
    """
    if path is None or path == "":
        return source if source is not None else default
    current = source
    for part in str(path).split("."):
        if isinstance(current, dict) and part in current:
            current = current.get(part)
            continue
        return default
    return current if current is not None else default
