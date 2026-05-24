"""
轻量 Markdown 提取层。
- 有 ```md 时只看 fence 内部，外部完全不管
- 以 ## 和 - 字段名 作为天然校验边界
- 缺失字段给 warning，不打死整条
- 只在"完全不可提取"时失败
"""

# 标准库：用于声明结构化解析结果。
from dataclasses import dataclass
# 标准库：用于日志。
import logging
# 标准库：用于正则解析。
import re

# 第三方库：用于声明结构化校验模型。
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# 常量：role1 当前使用的固定字段名。
ROLE1_FIELDS = ("pos_zh", "pos_en")
# 常量：role2/3/4 当前统一使用的占位字段名。
PLACEHOLDER_FIELDS = ("占位1", "占位2", "占位3")


class ModuleBMarkdownContractError(RuntimeError):
    """模块 B Markdown 契约解析异常——仅在完全不可提取时抛出。"""


class Role1VisualDescription(BaseModel):
    """role1 单条视觉描述。缺字段时对应值为空字符串。"""

    model_config = ConfigDict(extra="forbid")

    imagery_name: str = Field(default="")
    pos_zh: str = Field(default="")
    pos_en: str = Field(default="")


class ScenePlan(BaseModel):
    """role2 单条场景规划占位结构。"""

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(default="")
    占位2: str = Field(default="")
    占位3: str = Field(default="")


class ShotPlan(BaseModel):
    """role3 单条镜头规划占位结构。"""

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(default="")
    占位2: str = Field(default="")
    占位3: str = Field(default="")


class PromptPlan(BaseModel):
    """role4 单条提示词规划占位结构。"""

    model_config = ConfigDict(extra="forbid")

    占位1: str = Field(default="")
    占位2: str = Field(default="")
    占位3: str = Field(default="")


@dataclass(frozen=True)
class _MarkdownSection:
    """以 ## 标题分隔的 Markdown 条目。"""

    heading: str
    list_items: list[str]


# ---------------------------------------------------------------------------
# 公开解析函数
# ---------------------------------------------------------------------------


def parse_role1_visual_descriptions(markdown_text: str) -> list[Role1VisualDescription]:
    """轻量提取 role1 视觉描述，缺字段仅告警不抛错。"""
    sections = _parse_sections(markdown_text, contract_name="role1")
    results: list[Role1VisualDescription] = []
    for section in sections:
        field_map = _build_field_map(section.list_items, contract_name="role1", heading=section.heading)
        _warn_missing_fields(field_map, expected=ROLE1_FIELDS, contract_name="role1", heading=section.heading)
        _warn_extra_fields(field_map, expected=ROLE1_FIELDS, contract_name="role1", heading=section.heading)
        results.append(Role1VisualDescription(
            imagery_name=section.heading,
            pos_zh=field_map.get("pos_zh", ""),
            pos_en=field_map.get("pos_en", ""),
        ))
    if len(results) == len(sections):
        logger.info("role1 成功提取 %d 条视觉描述。", len(results))
    else:
        imagery_names = [r.imagery_name for r in results]
        logger.info("role1 成功提取 %d 条视觉描述，分别是%s。", len(results), imagery_names)
    return results


def parse_scene_plans(markdown_text: str) -> list[ScenePlan]:
    """轻量提取 role2 场景规划。"""
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role2",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [ScenePlan.model_validate(field_map) for _, field_map in parsed_sections]


def parse_shot_plans(markdown_text: str) -> list[ShotPlan]:
    """轻量提取 role3 镜头规划。"""
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role3",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [ShotPlan.model_validate(field_map) for _, field_map in parsed_sections]


def parse_prompt_plans(markdown_text: str) -> list[PromptPlan]:
    """轻量提取 role4 提示词规划。"""
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role4",
        expected_fields=PLACEHOLDER_FIELDS,
    )
    return [PromptPlan.model_validate(field_map) for _, field_map in parsed_sections]


# ---------------------------------------------------------------------------
# 内部：fenced md 提取
# ---------------------------------------------------------------------------


def _extract_fenced_md(text: str) -> str:
    """提取 ```md ... ``` 内部内容；若无 fence 则返回原文。"""
    t = str(text or "").replace("\r\n", "\n")
    m = re.search(r'```(?:md|markdown)?[ \t]*\n(.*?)\n[ \t]*```', t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t.strip()


# ---------------------------------------------------------------------------
# 内部：## 条目解析
# ---------------------------------------------------------------------------


def _parse_sections(markdown_text: str, contract_name: str) -> list[_MarkdownSection]:
    """按 ## 标题拆分条目，提取 - 列表字段行。"""
    text = _extract_fenced_md(markdown_text)
    if not text:
        raise ModuleBMarkdownContractError(f"{contract_name} Markdown 不能为空。")

    # 按 "## " 开头的行拆分；第一个 ## 之前的文本视为前言，跳过
    raw_parts = re.split(r'\n(?=## )', text)
    sections: list[_MarkdownSection] = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split('\n')
        first_line = lines[0].strip()
        if not first_line.startswith('## '):
            # 第一个 ## 之前的前言文本，跳过
            continue
        heading = first_line[3:].strip()

        if not heading:
            logger.warning("%s 存在空的 ## 条目标题，已跳过。", contract_name)
            continue

        list_items: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith('- '):
                list_items.append(stripped[2:].strip())

        sections.append(_MarkdownSection(heading=heading, list_items=list_items))

    heading_names = [s.heading for s in sections]
    logger.info("%s 解析到 %d 个 ## 条目，分别是%s。", contract_name, len(sections), heading_names)

    if not sections:
        raise ModuleBMarkdownContractError(f"{contract_name} 必须至少包含一个 ## 条目。")

    return sections


# ---------------------------------------------------------------------------
# 内部：字段行拆分
# ---------------------------------------------------------------------------


def _split_field_line(line: str) -> tuple[str, str]:
    """拆分 '- 字段: 值' 行。无法识别时返回 ('', '')。"""
    for separator in ("：", ":"):
        if separator in line:
            field_name, field_value = line.split(separator, 1)
            return field_name.strip(), field_value.strip()
    return "", ""


# ---------------------------------------------------------------------------
# 内部：字段映射构建与告警
# ---------------------------------------------------------------------------


def _build_field_map(list_items: list[str], *, contract_name: str, heading: str) -> dict[str, str]:
    """把列表行解析为字段映射，重复字段给 warning 并使用最后一次出现的值。"""
    field_map: dict[str, str] = {}
    for item in list_items:
        field_name, field_value = _split_field_line(item)
        if not field_name:
            logger.warning("%s 条目 '%s' 存在无法识别的字段行：%s", contract_name, heading, item)
            continue
        if field_name in field_map:
            logger.warning(
                "%s 条目 '%s' 出现重复字段 '%s'，使用最后一次出现的值。",
                contract_name, heading, field_name,
            )
        field_map[field_name] = field_value
    return field_map


def _warn_missing_fields(
    field_map: dict[str, str], *, expected: tuple[str, ...], contract_name: str, heading: str
) -> None:
    """对缺失的期望字段发出 warning。"""
    for field_name in expected:
        if field_name not in field_map:
            logger.warning(
                "%s 条目 '%s' 缺失字段 '%s'，使用空字符串。",
                contract_name, heading, field_name,
            )


def _warn_extra_fields(
    field_map: dict[str, str], *, expected: tuple[str, ...], contract_name: str, heading: str
) -> None:
    """对未识别的额外字段发出 warning。"""
    for field_name in field_map:
        if field_name not in expected:
            logger.warning(
                "%s 条目 '%s' 包含未识别字段 '%s'，已忽略。",
                contract_name, heading, field_name,
            )


# ---------------------------------------------------------------------------
# 内部：通用字段映射解析（role2/3/4 共用）
# ---------------------------------------------------------------------------


def _parse_field_maps(
    *,
    markdown_text: str,
    contract_name: str,
    expected_fields: tuple[str, ...],
) -> list[tuple[str, dict[str, str]]]:
    """把 Markdown 解析为 (标题, 字段映射) 数组。轻量模式——缺字段告警不抛错。"""
    sections = _parse_sections(markdown_text=markdown_text, contract_name=contract_name)
    parsed_sections: list[tuple[str, dict[str, str]]] = []
    for section in sections:
        field_map = _build_field_map(section.list_items, contract_name=contract_name, heading=section.heading)
        _warn_missing_fields(field_map, expected=expected_fields, contract_name=contract_name, heading=section.heading)
        _warn_extra_fields(field_map, expected=expected_fields, contract_name=contract_name, heading=section.heading)
        parsed_sections.append((
            section.heading,
            {f: field_map.get(f, "") for f in expected_fields},
        ))
    return parsed_sections
