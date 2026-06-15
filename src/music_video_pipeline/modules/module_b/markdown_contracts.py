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

# 辅助字段（LLM 自用，不下发下游，解析时静默忽略）
SILENT_FIELDS = frozenset({"narrative_rationale"})

# 常量：role1 当前使用的固定字段名。
ROLE1_FIELDS = ("pos_zh", "pos_en")
# 常量：role2 当前使用的固定字段名。
ROLE2_FIELDS = ("imagery_used", "story_outline_zh")
# 常量：role3 镜头规划的固定字段名。
ROLE3_FIELDS = ("remotion_reason", "scene_desc_zh", "remotion_id", "shot_subject_kind")
# 常量：role4 当前使用的固定字段名（共 7 字段）。
ROLE4_FIELDS = (
    "subject_kind",
    "keyframe_prompt_start_zh",
    "keyframe_prompt_start_en",
    "keyframe_prompt_end_zh",
    "keyframe_prompt_end_en",
    "video_prompt_zh",
    "video_prompt_en",
)


class ModuleBMarkdownContractError(RuntimeError):
    """模块 B Markdown 契约解析异常——仅在完全不可提取时抛出。"""


class Role1VisualDescription(BaseModel):
    """role1 单条视觉描述。缺字段时对应值为空字符串。"""

    model_config = ConfigDict(extra="forbid")

    imagery_name: str = Field(default="")
    pos_zh: str = Field(default="")
    pos_en: str = Field(default="")


class ScenePlan(BaseModel):
    """role2 单条场景规划——由 parse_role2_scene_plans 解析 Markdown 产出。"""

    model_config = ConfigDict(extra="forbid")

    big_segment_id: str = Field(default="")
    imagery_used: str = Field(default="")
    story_outline_zh: str = Field(default="")


class ShotPlan(BaseModel):
    """role3 单条镜头规划——由 parse_shot_plans 解析 Markdown 产出。"""

    model_config = ConfigDict(extra="forbid")

    big_segment_id: str = Field(default="")
    segment_id: str = Field(default="")
    remotion_reason: str = Field(default="")
    scene_desc_zh: str = Field(default="")
    remotion_id: str = Field(default="")
    shot_subject_kind: str = Field(default="human")


class PromptPlan(BaseModel):
    """role4 单条提示词规划 —— 7 字段（含 subject_kind）。"""

    model_config = ConfigDict(extra="forbid")

    shot_id: str = Field(default="")
    subject_kind: str = Field(default="human")
    keyframe_prompt_start_zh: str = Field(default="")
    keyframe_prompt_start_en: str = Field(default="")
    keyframe_prompt_end_zh: str = Field(default="")
    keyframe_prompt_end_en: str = Field(default="")
    video_prompt_zh: str = Field(default="")
    video_prompt_en: str = Field(default="")


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
    """轻量提取 role2 场景规划——标题是 big_segment_id，字段来自列表行。"""
    parsed_sections = _parse_field_maps(
        markdown_text=markdown_text,
        contract_name="role2",
        expected_fields=ROLE2_FIELDS,
    )
    results: list[ScenePlan] = []
    for heading, field_map in parsed_sections:
        raw_imagery = field_map.get("imagery_used", "")
        story_text = field_map.get("story_outline_zh", "")
        # 过滤 imagery_used：只保留 story_outline_zh 中实际出现的词
        filtered_imagery = _filter_imagery_used(raw_imagery, story_text)
        results.append(ScenePlan(
            big_segment_id=heading,
            imagery_used=filtered_imagery,
            story_outline_zh=story_text,
        ))
    logger.debug("role2 成功提取 %d 条场景规划。", len(results))
    return results


def parse_shot_plans(markdown_text: str) -> list[ShotPlan]:
    """轻量提取 role3 镜头规划——## big → ### segment → - 字段行。"""
    text = _extract_fenced_md(markdown_text)
    if not text:
        raise ModuleBMarkdownContractError("role3 Markdown 不能为空。")

    results: list[ShotPlan] = []
    # 按 "## " 拆分大段
    big_parts = re.split(r'\n(?=## )', text)
    for big_part in big_parts:
        big_part = big_part.strip()
        if not big_part:
            continue
        big_lines = big_part.split("\n")
        big_heading = big_lines[0].strip()
        if not big_heading.startswith("## "):
            continue
        big_segment_id = big_heading[3:].strip()
        if not big_segment_id:
            continue

        # 按 "### " 拆分 segment
        segment_body = "\n".join(big_lines[1:])
        segment_parts = re.split(r'\n(?=### )', segment_body)
        for segment_part in segment_parts:
            segment_part = segment_part.strip()
            if not segment_part:
                continue
            segment_lines = segment_part.split("\n")
            segment_heading = segment_lines[0].strip()
            if not segment_heading.startswith("### "):
                continue
            segment_id = segment_heading[4:].strip()
            if not segment_id:
                continue

            # 提取 - 字段行
            field_map: dict[str, str] = {}
            for line in segment_lines[1:]:
                item = line.strip()
                if not item.startswith("- "):
                    continue
                field_name, field_value = _split_field_line(item[2:])
                if not field_name:
                    continue
                field_map[field_name] = field_value

            _warn_missing_fields(field_map, expected=ROLE3_FIELDS, contract_name="role3", heading=segment_id)
            results.append(ShotPlan(
                big_segment_id=big_segment_id,
                segment_id=segment_id,
                remotion_reason=field_map.get("remotion_reason", ""),
                scene_desc_zh=field_map.get("scene_desc_zh", ""),
                remotion_id=field_map.get("remotion_id", ""),
                shot_subject_kind=field_map.get("shot_subject_kind", "human"),
            ))

    logger.debug("role3 成功提取 %d 条镜头规划。", len(results))
    if not results:
        raise ModuleBMarkdownContractError("role3 必须至少包含一个 ### shot 条目。")
    return results


def parse_prompt_plans(markdown_text: str) -> list[PromptPlan]:
    """解析 role4 新的 markdown 格式为 PromptPlan 列表。

    支持两种格式：
    - 单主体：## shot_xxx 下直接 6 个 - field: value
    - 多主体：## shot_xxx → ### 主体名 → 6 个 - field: value
      多主体时同名字段拼接（逗号分隔），扁平化为一条 PromptPlan。
    """
    text = _extract_fenced_md(markdown_text)
    if not text:
        logger.warning("role4 markdown 为空，返回空列表。")
        return []

    results: list[PromptPlan] = []
    # 按 ## shot_xxx 拆分
    shot_blocks = re.split(r"\n(?=## )", text)
    for block in shot_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        first_line = lines[0].strip()
        if not first_line.startswith("## "):
            continue
        shot_id = first_line[3:].strip()
        if not shot_id:
            continue

        # 检查是否存在 ### 主体名 子标题
        body = "\n".join(lines[1:])
        subject_blocks = re.split(r"\n(?=### )", body)

        has_subjects = any(
            b.strip().startswith("### ") for b in subject_blocks if b.strip()
        )

        if not has_subjects:
            # 单主体：直接解析 7 字段
            field_map = _build_field_map_from_body(body, contract_name="role4", heading=shot_id)
            _warn_missing_fields(field_map, expected=ROLE4_FIELDS, contract_name="role4", heading=shot_id)
            results.append(PromptPlan(
                shot_id=shot_id,
                subject_kind=field_map.get("subject_kind", "human"),
                keyframe_prompt_start_zh=field_map.get("keyframe_prompt_start_zh", ""),
                keyframe_prompt_start_en=field_map.get("keyframe_prompt_start_en", ""),
                keyframe_prompt_end_zh=field_map.get("keyframe_prompt_end_zh", ""),
                keyframe_prompt_end_en=field_map.get("keyframe_prompt_end_en", ""),
                video_prompt_zh=field_map.get("video_prompt_zh", ""),
                video_prompt_en=field_map.get("video_prompt_en", ""),
            ))
        else:
            # 多主体：按 ### 主体名 拆分，合并同名字段
            merged: dict[str, list[str]] = {f: [] for f in ROLE4_FIELDS}
            for subj_block in subject_blocks:
                subj_block = subj_block.strip()
                if not subj_block:
                    continue
                subj_lines = subj_block.split("\n")
                subj_first = subj_lines[0].strip()
                if not subj_first.startswith("### "):
                    continue
                subj_body = "\n".join(subj_lines[1:])
                subj_field_map = _build_field_map_from_body(subj_body, contract_name="role4", heading=f"{shot_id}>{subj_first[4:].strip()}")
                for f in ROLE4_FIELDS:
                    val = subj_field_map.get(f, "")
                    if val:
                        merged[f].append(val)

            # 多主体优先级：human > animal > object > scene
            subject_kind = "human"
            priority = {"human": 0, "animal": 1, "object": 2, "scene": 3}
            best_priority = 999
            for val in merged.get("subject_kind", []):
                v = str(val).strip().lower()
                if v in priority and priority[v] < best_priority:
                    best_priority = priority[v]
                    subject_kind = v

            results.append(PromptPlan(
                shot_id=shot_id,
                subject_kind=subject_kind,
                keyframe_prompt_start_zh=", ".join(merged["keyframe_prompt_start_zh"]),
                keyframe_prompt_start_en=", ".join(merged["keyframe_prompt_start_en"]),
                keyframe_prompt_end_zh=", ".join(merged["keyframe_prompt_end_zh"]),
                keyframe_prompt_end_en=", ".join(merged["keyframe_prompt_end_en"]),
                video_prompt_zh=", ".join(merged["video_prompt_zh"]),
                video_prompt_en=", ".join(merged["video_prompt_en"]),
            ))

    logger.info("role4 成功提取 %d 条提示词规划。", len(results))
    return results


def _build_field_map_from_body(body: str, *, contract_name: str, heading: str) -> dict[str, str]:
    """从 Markdown 正文行中提取 - field: value 映射。"""
    field_map: dict[str, str] = {}
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        field_name, field_value = _split_field_line(stripped[2:])
        if not field_name:
            continue
        if field_name in field_map:
            logger.warning(
                "%s 条目 '%s' 出现重复字段 '%s'，使用最后一次出现的值。",
                contract_name, heading, field_name,
            )
        field_map[field_name] = field_value
    return field_map


# ---------------------------------------------------------------------------
# 内部：fenced md 提取
# ---------------------------------------------------------------------------


def _extract_fenced_md(text: str) -> str:
    """提取所有 ```md ... ``` 内部内容并合并；若无 fence 则返回原文。"""
    t = str(text or "").replace("\r\n", "\n")
    blocks = re.findall(r'```(?:md|markdown)?[ \t]*\n(.*?)\n[ \t]*```', t, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks).strip()
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
            elif list_items:
                # 非 - 开头的行视为上一个字段的续行（缩进换行值），追加到最后一条 item
                last = list_items[-1]
                if stripped:
                    list_items[-1] = last + "\n" + stripped

        sections.append(_MarkdownSection(heading=heading, list_items=list_items))

    heading_names = [s.heading for s in sections]
    logger.debug("%s 解析到 %d 个 ## 条目，分别是%s。", contract_name, len(sections), heading_names)

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


def _filter_imagery_used(raw_imagery: str, story_text: str) -> str:
    """过滤 imagery_used：只保留在 story_outline_zh 中实际出现的意象词。"""
    if not raw_imagery or not story_text:
        return ""
    raw_items = [item.strip() for item in re.split(r"[、,，]", raw_imagery) if item.strip()]
    kept = [item for item in raw_items if item in story_text]
    if len(kept) != len(raw_items):
        logger.info(
            "过滤 imagery_used：%d/%d 项保留，移除 %s",
            len(kept), len(raw_items),
            [item for item in raw_items if item not in story_text],
        )
    return "、".join(kept)


def _warn_extra_fields(
    field_map: dict[str, str], *, expected: tuple[str, ...], contract_name: str, heading: str
) -> None:
    """对未识别的额外字段发出 warning（SILENT_FIELDS 除外）。"""
    for field_name in field_map:
        if field_name not in expected and field_name not in SILENT_FIELDS:
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
