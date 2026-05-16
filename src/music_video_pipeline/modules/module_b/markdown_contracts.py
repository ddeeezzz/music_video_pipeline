"""
文件用途：解析模块 B role1/2/3/4 的 Markdown 契约。
核心流程：用 markdown-it-py 拆出二级标题块，再按各 role 的字段集合做校验。
输入输出：输入 Markdown 字符串，输出 Pydantic 模型数组。
依赖说明：依赖 markdown-it-py 与 pydantic。
维护说明：role1 使用正式字段，role2/3/4 当前仍使用占位字段。
"""

# 标准库：用于声明结构化解析结果。
from dataclasses import dataclass

# 第三方库：用于解析 Markdown 语法树 token。
from markdown_it import MarkdownIt
# 第三方库：用于声明结构化校验模型。
from pydantic import BaseModel, ConfigDict, Field


# 常量：role1 当前使用的固定字段名。
ROLE1_FIELDS = ("pos_zh", "pos_en")
# 常量：role2/3/4 当前统一使用的占位字段名。
PLACEHOLDER_FIELDS = ("占位1", "占位2", "占位3")


class ModuleBMarkdownContractError(RuntimeError):
    """模块 B Markdown 契约解析异常。"""


class Role1VisualDescription(BaseModel):
    """
    功能说明：表示 role1 当前的单条视觉描述结构。
    参数说明：
    - imagery_name: 对应输入模板中的意象名称。
    - pos_zh: 中文视觉描述。
    - pos_en: 英文视觉描述。
    返回值：不适用。
    异常说明：字段缺失、为空或出现额外字段时由 Pydantic 抛错。
    边界条件：当前仅接受 `pos_zh` 与 `pos_en` 两个字段。
    """

    model_config = ConfigDict(extra="forbid")

    imagery_name: str = Field(min_length=1)
    pos_zh: str = Field(min_length=1)
    pos_en: str = Field(min_length=1)


class ScenePlan(BaseModel):
    """
    功能说明：表示 role2 当前的单条场景规划占位结构。
    参数说明：
    - 占位1: 业务字段未定时的第一个占位字段。
    - 占位2: 业务字段未定时的第二个占位字段。
    - 占位3: 业务字段未定时的第三个占位字段。
    返回值：不适用。
    异常说明：字段缺失、为空或出现额外字段时由 Pydantic 抛错。
    边界条件：当前仅接受这 3 个字段。
    """

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(min_length=1)
    占位2: str = Field(min_length=1)
    占位3: str = Field(min_length=1)


class ShotPlan(BaseModel):
    """
    功能说明：表示 role3 当前的单条镜头规划占位结构。
    参数说明：
    - 占位1: 业务字段未定时的第一个占位字段。
    - 占位2: 业务字段未定时的第二个占位字段。
    - 占位3: 业务字段未定时的第三个占位字段。
    返回值：不适用。
    异常说明：字段缺失、为空或出现额外字段时由 Pydantic 抛错。
    边界条件：当前仅接受这 3 个字段。
    """

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(min_length=1)
    占位2: str = Field(min_length=1)
    占位3: str = Field(min_length=1)


class PromptPlan(BaseModel):
    """
    功能说明：表示 role4 当前的单条提示词规划占位结构。
    参数说明：
    - 占位1: 业务字段未定时的第一个占位字段。
    - 占位2: 业务字段未定时的第二个占位字段。
    - 占位3: 业务字段未定时的第三个占位字段。
    返回值：不适用。
    异常说明：字段缺失、为空或出现额外字段时由 Pydantic 抛错。
    边界条件：当前仅接受这 3 个字段。
    """

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(min_length=1)
    占位2: str = Field(min_length=1)
    占位3: str = Field(min_length=1)


@dataclass(frozen=True)
class _MarkdownSection:
    """
    功能说明：表示一个以二级标题分隔的 Markdown 条目。
    参数说明：
    - heading: 条目标题文本，仅用于分块和报错定位。
    - list_items: 条目下的列表项文本。
    返回值：不适用。
    异常说明：不适用。
    边界条件：标题不是业务字段，只承担条目边界作用。
    """

    heading: str
    list_items: list[str]


def parse_role1_visual_descriptions(markdown_text: str) -> list[Role1VisualDescription]:
    """
    功能说明：解析 role1 Markdown 为视觉描述数组。
    参数说明：
    - markdown_text: role1 输出的 Markdown 原文。
    返回值：
    - list[Role1VisualDescription]: 视觉描述数组。
    异常说明：
    - ModuleBMarkdownContractError: Markdown 结构或字段不符合约定时抛出。
    边界条件：每个条目必须包含且只包含 `pos_zh` 与 `pos_en`。
    """
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role1",
        expected_fields=ROLE1_FIELDS,
    )
    return [
        Role1VisualDescription(
            imagery_name=heading,
            pos_zh=field_map["pos_zh"],
            pos_en=field_map["pos_en"],
        )
        for heading, field_map in parsed_sections
    ]


def parse_scene_plans(markdown_text: str) -> list[ScenePlan]:
    """
    功能说明：解析 role2 Markdown 为场景规划数组。
    参数说明：
    - markdown_text: role2 输出的 Markdown 原文。
    返回值：
    - list[ScenePlan]: 场景规划数组。
    异常说明：
    - ModuleBMarkdownContractError: Markdown 结构或字段不符合约定时抛出。
    边界条件：每个条目必须包含且只包含 `占位1/占位2/占位3`。
    """
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role2",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [ScenePlan.model_validate(field_map) for _, field_map in parsed_sections]


def parse_shot_plans(markdown_text: str) -> list[ShotPlan]:
    """
    功能说明：解析 role3 Markdown 为镜头规划数组。
    参数说明：
    - markdown_text: role3 输出的 Markdown 原文。
    返回值：
    - list[ShotPlan]: 镜头规划数组。
    异常说明：
    - ModuleBMarkdownContractError: Markdown 结构或字段不符合约定时抛出。
    边界条件：每个条目必须包含且只包含 `占位1/占位2/占位3`。
    """
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role3",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [ShotPlan.model_validate(field_map) for _, field_map in parsed_sections]


def parse_prompt_plans(markdown_text: str) -> list[PromptPlan]:
    """
    功能说明：解析 role4 Markdown 为提示词规划数组。
    参数说明：
    - markdown_text: role4 输出的 Markdown 原文。
    返回值：
    - list[PromptPlan]: 提示词规划数组。
    异常说明：
    - ModuleBMarkdownContractError: Markdown 结构或字段不符合约定时抛出。
    边界条件：每个条目必须包含且只包含 `占位1/占位2/占位3`。
    """
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role4",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [PromptPlan.model_validate(field_map) for _, field_map in parsed_sections]


def _parse_field_maps(
    *,
    markdown_text: str,
    contract_name: str,
    expected_fields: tuple[str, ...],
) -> list[tuple[str, dict[str, str]]]:
    """
    功能说明：把 Markdown 契约解析为 `标题 + 字段映射` 数组。
    参数说明：
    - markdown_text: 原始 Markdown 文本。
    - contract_name: 契约名称，仅用于报错信息。
    - expected_fields: 当前契约要求的字段集合。
    返回值：
    - list[tuple[str, dict[str, str]]]: 每个条目的标题和字段映射。
    异常说明：
    - ModuleBMarkdownContractError: 文本为空、缺少条目或字段不合法时抛出。
    边界条件：字段顺序不重要，但字段集合必须完全一致。
    """
    sections = _parse_sections(markdown_text=markdown_text, contract_name=contract_name)
    parsed_sections: list[tuple[str, dict[str, str]]] = []
    for section in sections:
        field_map: dict[str, str] = {}
        for list_item in section.list_items:
            field_name, field_value = _split_field_line(
                line=list_item,
                contract_name=contract_name,
                heading=section.heading,
            )
            if field_name in field_map:
                raise ModuleBMarkdownContractError(
                    f"{contract_name} 的条目 `{section.heading}` 出现重复字段：{field_name}"
                )
            field_map[field_name] = field_value
        _validate_field_names(
            field_map=field_map,
            contract_name=contract_name,
            heading=section.heading,
            expected_fields=expected_fields,
        )
        parsed_sections.append(
            (
                section.heading,
                {field_name: field_map[field_name] for field_name in expected_fields},
            )
        )
    return parsed_sections


def _parse_sections(markdown_text: str, contract_name: str) -> list[_MarkdownSection]:
    """
    功能说明：使用 markdown-it-py 解析二级标题条目。
    参数说明：
    - markdown_text: 原始 Markdown 文本。
    - contract_name: 契约名称，仅用于报错信息。
    返回值：
    - list[_MarkdownSection]: 解析出的条目数组。
    异常说明：
    - ModuleBMarkdownContractError: Markdown 为空或缺少二级标题时抛出。
    边界条件：只把 `##` 视为条目边界。
    """
    normalized_text = str(markdown_text or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        raise ModuleBMarkdownContractError(f"{contract_name} Markdown 不能为空。")

    markdown_parser = MarkdownIt()
    tokens = markdown_parser.parse(normalized_text)
    sections: list[_MarkdownSection] = []
    current_heading = ""
    current_list_items: list[str] = []
    current_item_parts: list[str] = []
    expecting_h2_text = False
    inside_list_item = False

    for token in tokens:
        if token.type == "heading_open" and token.tag == "h2":
            if current_heading:
                sections.append(_MarkdownSection(heading=current_heading, list_items=current_list_items))
            current_heading = ""
            current_list_items = []
            expecting_h2_text = True
            continue

        if expecting_h2_text and token.type == "inline":
            current_heading = str(token.content or "").strip()
            expecting_h2_text = False
            continue

        if token.type == "list_item_open":
            if not current_heading:
                raise ModuleBMarkdownContractError(f"{contract_name} 存在未归属到 `##` 条目的列表项。")
            inside_list_item = True
            current_item_parts = []
            continue

        if inside_list_item and token.type == "inline":
            content = str(token.content or "").strip()
            if content:
                current_item_parts.append(content)
            continue

        if token.type == "list_item_close":
            inside_list_item = False
            list_item_text = "\n".join(current_item_parts).strip()
            if list_item_text:
                current_list_items.append(list_item_text)
            current_item_parts = []

    if current_heading:
        sections.append(_MarkdownSection(heading=current_heading, list_items=current_list_items))

    if not sections:
        raise ModuleBMarkdownContractError(f"{contract_name} 必须至少包含一个 `##` 条目。")

    for section in sections:
        if not section.heading:
            raise ModuleBMarkdownContractError(f"{contract_name} 存在空的 `##` 条目标题。")
        if not section.list_items:
            raise ModuleBMarkdownContractError(f"{contract_name} 的条目 `{section.heading}` 缺少列表字段。")
    return sections


def _split_field_line(*, line: str, contract_name: str, heading: str) -> tuple[str, str]:
    """
    功能说明：拆分单个 `- 字段: 值` 列表项文本。
    参数说明：
    - line: 去掉列表标记后的行文本。
    - contract_name: 契约名称，仅用于报错信息。
    - heading: 当前条目标题，仅用于报错信息。
    返回值：
    - tuple[str, str]: 字段名与字段值。
    异常说明：
    - ModuleBMarkdownContractError: 不包含字段分隔符或字段值为空时抛出。
    边界条件：同时接受中文冒号与英文冒号。
    """
    field_name = ""
    field_value = ""
    for separator in ("：", ":"):
        if separator in line:
            field_name, field_value = line.split(separator, 1)
            break
    field_name = field_name.strip()
    field_value = field_value.strip()
    if not field_name:
        raise ModuleBMarkdownContractError(
            f"{contract_name} 的条目 `{heading}` 存在无法识别的字段行：{line}"
        )
    if not field_value:
        raise ModuleBMarkdownContractError(
            f"{contract_name} 的条目 `{heading}` 中字段 `{field_name}` 不能为空。"
        )
    return field_name, field_value


def _validate_field_names(
    *,
    field_map: dict[str, str],
    contract_name: str,
    heading: str,
    expected_fields: tuple[str, ...],
) -> None:
    """
    功能说明：校验当前条目的字段集合是否正好等于目标字段集合。
    参数说明：
    - field_map: 当前条目的字段映射。
    - contract_name: 契约名称，仅用于报错信息。
    - heading: 当前条目标题，仅用于报错信息。
    - expected_fields: 当前契约要求的字段集合。
    返回值：无。
    异常说明：
    - ModuleBMarkdownContractError: 缺字段或额外字段时抛出。
    边界条件：字段顺序不重要，但字段集合必须完全一致。
    """
    extra_fields = [field_name for field_name in field_map if field_name not in expected_fields]
    missing_fields = [field_name for field_name in expected_fields if field_name not in field_map]
    if extra_fields:
        raise ModuleBMarkdownContractError(
            f"{contract_name} 的条目 `{heading}` 出现未定义字段：{extra_fields}"
        )
    if missing_fields:
        raise ModuleBMarkdownContractError(
            f"{contract_name} 的条目 `{heading}` 缺失字段：{missing_fields}"
        )
