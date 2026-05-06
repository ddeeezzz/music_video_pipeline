"""
文件用途：提供模块B v2 模板与4角色输出的 Markdown 解析与校验能力。
核心流程：按约定标题结构提取字段 -> 校验字段与候选约束 -> 返回标准化字典。
输入输出：输入原始 Markdown 文本或字典，输出标准化对象。
依赖说明：依赖标准库与项目内 v2 数据结构常量。
维护说明：v2 字段契约调整时必须同步更新本文件。
"""

import re
from typing import Any

# 项目内模块：统一 Markdown 解析器。
from music_video_pipeline.modules.module_b_v2.markdown_io import parse_markdown_document
# 项目内模块：导入 v2 常量。
from music_video_pipeline.modules.module_b_v2.models import (
    VALID_CAMERA_PLAN_DIRECTIONS,
    VALID_CAMERA_PLAN_MODES,
    VALID_CAMERA_PLAN_STRENGTHS,
    VALID_EASING_VALUES,
    VALID_TRANSITION_KINDS,
)


class ModuleBV2ParseError(ValueError):
    """模块B v2 解析失败异常。"""


def normalize_non_empty_text(field_name: str, value: object) -> str:
    """
    功能说明：校验并清洗非空文本字段。
    参数说明：
    - field_name: 字段名。
    - value: 字段值。
    返回值：
    - str: 标准化后的文本。
    异常说明：
    - ModuleBV2ParseError: 类型或内容非法时抛出。
    边界条件：仅接受字符串。
    """
    if not isinstance(value, str):
        raise ModuleBV2ParseError(f"{field_name} 必须是字符串。")
    normalized = value.strip()
    if not normalized:
        raise ModuleBV2ParseError(f"{field_name} 不能为空。")
    return normalized


def parse_role1_visual_catalog_markdown(text: str) -> dict[str, Any]:
    """
    功能说明：解析角色1单类对象的 Markdown 输出。
    参数说明：
    - text: 模型返回的 Markdown 文本。
    返回值：
    - dict[str, Any]: 形如 {"assets": [...]} 的对象。
    异常说明：
    - ModuleBV2ParseError: 结构不符合约定时抛出。
    边界条件：顶层要求使用 `## item_id` 与 `### ref_id`。
    """
    document = parse_markdown_document(text)
    assets: list[dict[str, Any]] = []
    for section in document.sections:
        item_id = normalize_non_empty_text("role1.item_id", section.heading)
        refs: list[dict[str, Any]] = []
        for subsection in section.subsections:
            ref_id = normalize_non_empty_text(f"role1[{item_id}].ref_id", subsection.heading)
            field_map = dict(subsection.fields)
            refs.append(
                {
                    "ref_id": ref_id,
                    "pos_zh": _require_field(field_map, "pos_zh", f"role1[{item_id}][{ref_id}]"),
                    "pos_en": _require_field(field_map, "pos_en", f"role1[{item_id}][{ref_id}]"),
                    "neg_zh": _require_field(field_map, "neg_zh", f"role1[{item_id}][{ref_id}]"),
                    "neg_en": _require_field(field_map, "neg_en", f"role1[{item_id}][{ref_id}]"),
                }
            )
        assets.append({"item_id": item_id, "refs": refs})
    return {"assets": assets}


def parse_role2_big_segment_story_markdown(text: str) -> dict[str, Any]:
    """
    功能说明：解析角色2大段剧情 Markdown 输出。
    参数说明：
    - text: 模型返回的 Markdown 文本。
    返回值：
    - dict[str, Any]: 形如 {"big_segments": [...]} 的对象。
    异常说明：
    - ModuleBV2ParseError: 结构不符合约定时抛出。
    边界条件：每个大段使用 `## big_segment_id`。
    """
    document = parse_markdown_document(text)
    items: list[dict[str, Any]] = []
    for section in document.sections:
        big_segment_id = normalize_non_empty_text("role2.big_segment_id", section.heading)
        field_map = dict(section.fields)
        items.append(
            {
                "big_segment_id": big_segment_id,
                "title_zh": _require_field(field_map, "title_zh", f"role2[{big_segment_id}]"),
                "story_outline_zh": _require_field(field_map, "story_outline_zh", f"role2[{big_segment_id}]"),
                "selected_scene_ids": _parse_id_csv(
                    field_map.get("selected_scene_ids", ""),
                    field_name=f"role2[{big_segment_id}].selected_scene_ids",
                ),
                "selected_character_ids": _parse_id_csv(
                    field_map.get("selected_character_ids", ""),
                    field_name=f"role2[{big_segment_id}].selected_character_ids",
                ),
                "selected_prop_ids": _parse_id_csv(
                    field_map.get("selected_prop_ids", ""),
                    field_name=f"role2[{big_segment_id}].selected_prop_ids",
                ),
            }
        )
    return {"big_segments": items}


def parse_role3_segment_directing_markdown(text: str) -> dict[str, Any]:
    """
    功能说明：解析角色3单镜头 Markdown 输出。
    参数说明：
    - text: 模型返回的 Markdown 文本。
    返回值：
    - dict[str, Any]: 单镜头编排对象。
    异常说明：
    - ModuleBV2ParseError: 结构不符合约定时抛出。
    边界条件：要求存在唯一一个 `## shot_id`。
    """
    document = parse_markdown_document(text)
    if len(document.sections) != 1:
        raise ModuleBV2ParseError("role3 输出必须且只能包含 1 个 shot 段落。")
    section = document.sections[0]
    shot_id = normalize_non_empty_text("role3.shot_id", section.heading)
    subsection_map = {
        normalize_non_empty_text(f"role3[{shot_id}].subsection", item.heading): item for item in section.subsections
    }
    field_map = dict(section.fields)
    return {
        "shot_id": shot_id,
        "scene_desc_zh": _require_field(field_map, "scene_desc_zh", f"role3[{shot_id}]"),
        "selected_scene_id": _require_field(field_map, "selected_scene_id", f"role3[{shot_id}]"),
        "selected_character_ids": _parse_id_csv(
            field_map.get("selected_character_ids", ""),
            field_name=f"role3[{shot_id}].selected_character_ids",
        ),
        "selected_prop_ids": _parse_id_csv(
            field_map.get("selected_prop_ids", ""),
            field_name=f"role3[{shot_id}].selected_prop_ids",
        ),
        "composition_id": _require_field(field_map, "composition_id", f"role3[{shot_id}]"),
        "camera_plan": _parse_plan_section(
            subsection_map.get("camera_plan"),
            field_name=f"role3[{shot_id}].camera_plan",
            required_fields=["preset_id", "mode", "direction", "strength", "easing"],
        ),
        "transition_plan": _parse_plan_section(
            subsection_map.get("transition_plan"),
            field_name=f"role3[{shot_id}].transition_plan",
            required_fields=["preset_id", "kind", "duration_ms", "easing"],
        ),
    }


def parse_role4_prompt_markdown(text: str) -> dict[str, Any]:
    """
    功能说明：解析角色4单镜头提示词 Markdown 输出。
    参数说明：
    - text: 模型返回的 Markdown 文本。
    返回值：
    - dict[str, Any]: 单镜头提示词块对象。
    异常说明：
    - ModuleBV2ParseError: 结构不符合约定时抛出。
    边界条件：字段正文允许多行，但标题名必须固定。
    """
    document = parse_markdown_document(text)
    if len(document.sections) != 1:
        raise ModuleBV2ParseError("role4 输出必须且只能包含 1 个 shot 段落。")
    section = document.sections[0]
    shot_id = normalize_non_empty_text("role4.shot_id", section.heading)
    field_sections = {
        normalize_non_empty_text(f"role4[{shot_id}].field", item.heading): item.body.strip()
        for item in section.subsections
    }
    required_fields = [
        "scene_desc",
        "keyframe_prompt_start_zh",
        "keyframe_prompt_start_en",
        "keyframe_negative_prompt_start_zh",
        "keyframe_negative_prompt_start_en",
        "keyframe_prompt_end_zh",
        "keyframe_prompt_end_en",
        "keyframe_negative_prompt_end_zh",
        "keyframe_negative_prompt_end_en",
        "video_prompt_zh",
        "video_prompt_en",
    ]
    result: dict[str, Any] = {"shot_id": shot_id}
    for field_name in required_fields:
        result[field_name] = normalize_non_empty_text(
            f"role4[{shot_id}].{field_name}",
            field_sections.get(field_name, ""),
        )
    return result


def _parse_id_csv(value: str, *, field_name: str) -> list[str]:
    """
    功能说明：解析逗号分隔的 ID 字段。
    参数说明：
    - value: 原始文本。
    - field_name: 业务字段名，用于报错。
    返回值：
    - list[str]: 解析后的去重 ID 数组。
    异常说明：
    - ModuleBV2ParseError: ID 含空项时抛出。
    边界条件：`none`、`无`、`-` 与空字符串视为空数组。
    """
    normalized_value = str(value or "").strip()
    if normalized_value.lower() in {"", "none", "null", "-", "无"}:
        return []
    parts = [item.strip() for item in re.split(r"[,，]", normalized_value)]
    result: list[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(parts):
        normalized_item = normalize_non_empty_text(f"{field_name}[{index}]", item)
        if normalized_item in seen_ids:
            continue
        seen_ids.add(normalized_item)
        result.append(normalized_item)
    return result


def _parse_plan_section(node: Any, *, field_name: str, required_fields: list[str]) -> dict[str, Any]:
    """
    功能说明：解析运镜/转场子段落中的键值字段。
    参数说明：
    - node: `### camera_plan` 或 `### transition_plan` 对应的节点对象。
    - field_name: 业务字段名。
    - required_fields: 必填字段名列表。
    返回值：
    - dict[str, Any]: 解析后的字典。
    异常说明：
    - ModuleBV2ParseError: 段落缺失或字段不全时抛出。
    边界条件：所有值先按字符串读取，后续由校验器负责类型归一化。
    """
    if node is None:
        raise ModuleBV2ParseError(f"{field_name} 段落缺失。")
    field_map = getattr(node, "fields", {})
    if not isinstance(field_map, dict) or not field_map:
        raise ModuleBV2ParseError(f"{field_name} 段落缺失字段。")
    return {field: _require_field(field_map, field, field_name) for field in required_fields}


def _require_field(field_map: dict[str, str], key: str, field_name: str) -> str:
    """
    功能说明：从键值映射中读取必填字段。
    参数说明：
    - field_map: 已解析字段映射。
    - key: 字段名。
    - field_name: 业务字段名，用于报错。
    返回值：
    - str: 标准化后的字段值。
    异常说明：
    - ModuleBV2ParseError: 缺失或为空时抛出。
    边界条件：无。
    """
    return normalize_non_empty_text(f"{field_name}.{key}", field_map.get(key, ""))


def validate_storyboard_template(data: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：校验编排模板编译结果。
    参数说明：
    - data: 模板对象。
    返回值：
    - dict[str, Any]: 原样返回已通过校验的对象。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或结构非法时抛出。
    边界条件：只做当前 v1 所需最小校验。
    """
    required_keys = {
        "template_id",
        "style",
        "story",
        "scene_catalog",
        "prop_catalog",
        "character_catalog",
        "composition_catalog",
        "camera_plan_presets",
        "camera_mapping",
        "transition_presets",
        "transition_mapping",
    }
    missing_keys = required_keys.difference(data.keys())
    if missing_keys:
        raise ModuleBV2ParseError(f"编排模板缺失字段：{sorted(missing_keys)}")
    _validate_unique_ids(data.get("scene_catalog", []), key_name="item_id", field_name="scene_catalog")
    _validate_unique_ids(data.get("prop_catalog", []), key_name="item_id", field_name="prop_catalog")
    _validate_unique_ids(data.get("character_catalog", []), key_name="item_id", field_name="character_catalog")
    _validate_unique_ids(data.get("composition_catalog", []), key_name="composition_id", field_name="composition_catalog")
    _validate_unique_ids(data.get("camera_plan_presets", []), key_name="preset_id", field_name="camera_plan_presets")
    _validate_unique_ids(data.get("transition_presets", []), key_name="preset_id", field_name="transition_presets")
    camera_preset_ids = {
        validate_camera_plan(item).get("preset_id", "")
        for item in data.get("camera_plan_presets", [])
        if isinstance(item, dict)
    }
    transition_preset_ids = {
        validate_transition_plan(item).get("preset_id", "")
        for item in data.get("transition_presets", [])
        if isinstance(item, dict)
    }
    _validate_camera_mapping_rules(data=data.get("camera_mapping", []), camera_preset_ids=camera_preset_ids)
    _validate_transition_mapping_rules(data=data.get("transition_mapping", []), transition_preset_ids=transition_preset_ids)
    return data


def validate_camera_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：校验标准化运镜计划。
    参数说明：
    - plan: 运镜对象。
    返回值：
    - dict[str, Any]: 原样返回已通过校验的对象。
    异常说明：
    - ModuleBV2ParseError: 字段非法时抛出。
    边界条件：mode=none 时 direction/strength 仍要求存在。
    """
    preset_id = normalize_non_empty_text("camera_plan.preset_id", plan.get("preset_id", ""))
    mode = normalize_non_empty_text("camera_plan.mode", plan.get("mode", ""))
    direction = normalize_non_empty_text("camera_plan.direction", plan.get("direction", ""))
    strength = normalize_non_empty_text("camera_plan.strength", plan.get("strength", ""))
    easing = normalize_non_empty_text("camera_plan.easing", plan.get("easing", ""))
    if mode not in VALID_CAMERA_PLAN_MODES:
        raise ModuleBV2ParseError(f"camera_plan.mode 非法：{mode}")
    if direction not in VALID_CAMERA_PLAN_DIRECTIONS:
        raise ModuleBV2ParseError(f"camera_plan.direction 非法：{direction}")
    if strength not in VALID_CAMERA_PLAN_STRENGTHS:
        raise ModuleBV2ParseError(f"camera_plan.strength 非法：{strength}")
    if easing not in VALID_EASING_VALUES:
        raise ModuleBV2ParseError(f"camera_plan.easing 非法：{easing}")
    return {
        "preset_id": preset_id,
        "mode": mode,
        "direction": direction,
        "strength": strength,
        "easing": easing,
    }


def validate_transition_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：校验标准化转场计划。
    参数说明：
    - plan: 转场对象。
    返回值：
    - dict[str, Any]: 原样返回已通过校验的对象。
    异常说明：
    - ModuleBV2ParseError: 字段非法时抛出。
    边界条件：duration_ms 允许为 0。
    """
    preset_id = normalize_non_empty_text("transition_plan.preset_id", plan.get("preset_id", ""))
    kind = normalize_non_empty_text("transition_plan.kind", plan.get("kind", ""))
    easing = normalize_non_empty_text("transition_plan.easing", plan.get("easing", ""))
    try:
        duration_ms = int(plan.get("duration_ms", 0))
    except (TypeError, ValueError) as error:
        raise ModuleBV2ParseError("transition_plan.duration_ms 必须是整数。") from error
    if kind not in VALID_TRANSITION_KINDS:
        raise ModuleBV2ParseError(f"transition_plan.kind 非法：{kind}")
    if easing not in VALID_EASING_VALUES:
        raise ModuleBV2ParseError(f"transition_plan.easing 非法：{easing}")
    if duration_ms < 0:
        raise ModuleBV2ParseError("transition_plan.duration_ms 不能小于 0。")
    return {
        "preset_id": preset_id,
        "kind": kind,
        "duration_ms": duration_ms,
        "easing": easing,
    }


def _validate_unique_ids(items: Any, key_name: str, field_name: str) -> None:
    """
    功能说明：校验列表中的 ID 唯一性。
    参数说明：
    - items: 条目列表。
    - key_name: 目标 ID 字段名。
    - field_name: 业务字段名。
    返回值：无。
    异常说明：
    - ModuleBV2ParseError: 非法结构或重复 ID 时抛出。
    边界条件：空列表允许通过。
    """
    if not isinstance(items, list):
        raise ModuleBV2ParseError(f"{field_name} 必须是列表。")
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"{field_name}[{index}] 必须是对象。")
        item_id = normalize_non_empty_text(f"{field_name}[{index}].{key_name}", item.get(key_name, ""))
        if item_id in seen_ids:
            raise ModuleBV2ParseError(f"{field_name} 存在重复 ID：{item_id}")
        seen_ids.add(item_id)


def validate_role1_visual_catalog_output(
    data: dict[str, Any],
    *,
    scene_ids: list[str],
    prop_ids: list[str],
    character_ids: list[str],
) -> dict[str, Any]:
    """
    功能说明：校验角色1视觉词库输出。
    参数说明：
    - data: 角色1输出对象。
    - scene_ids/prop_ids/character_ids: 各目录合法 ID 集合。
    返回值：
    - dict[str, Any]: 标准化后的角色1输出。
    异常说明：
    - ModuleBV2ParseError: 字段或 ID 非法时抛出。
    边界条件：每个对象至少要返回 1 组 refs。
    """
    payload = {
        "scene_refs": _validate_visual_asset_refs(data.get("scene_refs", []), "scene_refs", set(scene_ids)),
        "prop_refs": _validate_visual_asset_refs(data.get("prop_refs", []), "prop_refs", set(prop_ids)),
        "character_refs": _validate_visual_asset_refs(data.get("character_refs", []), "character_refs", set(character_ids)),
    }
    _assert_visual_asset_full_coverage(payload["scene_refs"], expected_ids=set(scene_ids), field_name="scene_refs")
    _assert_visual_asset_full_coverage(payload["prop_refs"], expected_ids=set(prop_ids), field_name="prop_refs")
    _assert_visual_asset_full_coverage(
        payload["character_refs"],
        expected_ids=set(character_ids),
        field_name="character_refs",
    )
    return payload


def validate_role2_big_segment_story_output(
    data: dict[str, Any],
    *,
    big_segment_ids: list[str],
    scene_ids: list[str],
    prop_ids: list[str],
    character_ids: list[str],
) -> dict[str, Any]:
    """
    功能说明：校验角色2大段剧情输出。
    参数说明：
    - data: 角色2输出对象。
    - big_segment_ids: 合法大段 ID。
    - scene_ids/prop_ids/character_ids: 合法对象 ID。
    返回值：
    - dict[str, Any]: 标准化后的角色2输出。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或 ID 非法时抛出。
    边界条件：每个 big_segment 必须完整覆盖一次。
    """
    items = data.get("big_segments", [])
    if not isinstance(items, list):
        raise ModuleBV2ParseError("role2.big_segments 必须是列表。")
    known_big_segment_ids = set(big_segment_ids)
    seen_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"role2.big_segments[{index}] 必须是对象。")
        big_segment_id = normalize_non_empty_text(
            f"role2.big_segments[{index}].big_segment_id",
            item.get("big_segment_id", ""),
        )
        if big_segment_id not in known_big_segment_ids:
            raise ModuleBV2ParseError(f"role2.big_segments[{index}] 引用了未知 big_segment_id：{big_segment_id}")
        if big_segment_id in seen_ids:
            raise ModuleBV2ParseError(f"role2.big_segments 存在重复 big_segment_id：{big_segment_id}")
        seen_ids.add(big_segment_id)
        normalized_items.append(
            {
                "big_segment_id": big_segment_id,
                "title_zh": normalize_non_empty_text(
                    f"role2.big_segments[{index}].title_zh",
                    item.get("title_zh", ""),
                ),
                "story_outline_zh": normalize_non_empty_text(
                    f"role2.big_segments[{index}].story_outline_zh",
                    item.get("story_outline_zh", ""),
                ),
                "selected_scene_ids": _validate_id_list(
                    item.get("selected_scene_ids", []),
                    field_name=f"role2.big_segments[{index}].selected_scene_ids",
                    valid_ids=set(scene_ids),
                ),
                "selected_character_ids": _validate_id_list(
                    item.get("selected_character_ids", []),
                    field_name=f"role2.big_segments[{index}].selected_character_ids",
                    valid_ids=set(character_ids),
                ),
                "selected_prop_ids": _validate_id_list(
                    item.get("selected_prop_ids", []),
                    field_name=f"role2.big_segments[{index}].selected_prop_ids",
                    valid_ids=set(prop_ids),
                ),
            }
        )
    if seen_ids != known_big_segment_ids:
        missing_ids = sorted(known_big_segment_ids.difference(seen_ids))
        raise ModuleBV2ParseError(f"role2.big_segments 覆盖不完整，missing={missing_ids}")
    return {"big_segments": normalized_items}


def validate_role3_segment_directing_output(
    data: dict[str, Any],
    *,
    shot_ids: list[str],
    scene_ids: list[str],
    prop_ids: list[str],
    character_ids: list[str],
    composition_ids: list[str],
) -> dict[str, Any]:
    """
    功能说明：校验角色3镜头编排输出。
    参数说明：
    - data: 角色3输出对象。
    - shot_ids: 合法 shot_id 顺序。
    - scene_ids/prop_ids/character_ids/composition_ids: 合法对象 ID。
    返回值：
    - dict[str, Any]: 标准化后的角色3输出。
    异常说明：
    - ModuleBV2ParseError: 字段、ID 或 plan 非法时抛出。
    边界条件：每个 shot_id 必须完整覆盖一次。
    """
    items = data.get("shots", [])
    if not isinstance(items, list):
        raise ModuleBV2ParseError("role3.shots 必须是列表。")
    known_shot_ids = set(shot_ids)
    seen_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"role3.shots[{index}] 必须是对象。")
        shot_id = normalize_non_empty_text(f"role3.shots[{index}].shot_id", item.get("shot_id", ""))
        if shot_id not in known_shot_ids:
            raise ModuleBV2ParseError(f"role3.shots[{index}] 引用了未知 shot_id：{shot_id}")
        if shot_id in seen_ids:
            raise ModuleBV2ParseError(f"role3.shots 存在重复 shot_id：{shot_id}")
        seen_ids.add(shot_id)
        normalized_items.append(
            {
                "shot_id": shot_id,
                "scene_desc_zh": normalize_non_empty_text(
                    f"role3.shots[{index}].scene_desc_zh",
                    item.get("scene_desc_zh", ""),
                ),
                "selected_scene_id": _validate_single_id(
                    item.get("selected_scene_id", ""),
                    field_name=f"role3.shots[{index}].selected_scene_id",
                    valid_ids=set(scene_ids),
                ),
                "selected_character_ids": _validate_id_list(
                    item.get("selected_character_ids", []),
                    field_name=f"role3.shots[{index}].selected_character_ids",
                    valid_ids=set(character_ids),
                ),
                "selected_prop_ids": _validate_id_list(
                    item.get("selected_prop_ids", []),
                    field_name=f"role3.shots[{index}].selected_prop_ids",
                    valid_ids=set(prop_ids),
                ),
                "composition_id": _validate_single_id(
                    item.get("composition_id", ""),
                    field_name=f"role3.shots[{index}].composition_id",
                    valid_ids=set(composition_ids),
                ),
                "camera_plan": validate_camera_plan(item.get("camera_plan", {})),
                "transition_plan": validate_transition_plan(item.get("transition_plan", {})),
            }
        )
    if seen_ids != known_shot_ids:
        missing_ids = sorted(known_shot_ids.difference(seen_ids))
        raise ModuleBV2ParseError(f"role3.shots 覆盖不完整，missing={missing_ids}")
    return {"shots": normalized_items}


def validate_role4_prompt_output(data: dict[str, Any], *, shot_ids: list[str]) -> dict[str, Any]:
    """
    功能说明：校验角色4提示词块输出。
    参数说明：
    - data: 角色4输出对象。
    - shot_ids: 本轮要求生成的 shot_id 列表。
    返回值：
    - dict[str, Any]: 标准化后的角色4输出。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或文本为空时抛出。
    边界条件：只校验本轮 target_shot_ids，不要求覆盖全部项目 shot。
    """
    items = data.get("shots", [])
    if not isinstance(items, list):
        raise ModuleBV2ParseError("role4.shots 必须是列表。")
    known_shot_ids = set(shot_ids)
    seen_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    required_fields = [
        "scene_desc",
        "keyframe_prompt_start_zh",
        "keyframe_prompt_start_en",
        "keyframe_negative_prompt_start_zh",
        "keyframe_negative_prompt_start_en",
        "keyframe_prompt_end_zh",
        "keyframe_prompt_end_en",
        "keyframe_negative_prompt_end_zh",
        "keyframe_negative_prompt_end_en",
        "video_prompt_zh",
        "video_prompt_en",
    ]
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"role4.shots[{index}] 必须是对象。")
        shot_id = normalize_non_empty_text(f"role4.shots[{index}].shot_id", item.get("shot_id", ""))
        if shot_id not in known_shot_ids:
            raise ModuleBV2ParseError(f"role4.shots[{index}] 引用了未知 shot_id：{shot_id}")
        if shot_id in seen_ids:
            raise ModuleBV2ParseError(f"role4.shots 存在重复 shot_id：{shot_id}")
        seen_ids.add(shot_id)
        normalized_item = {"shot_id": shot_id}
        for field_name in required_fields:
            normalized_item[field_name] = normalize_non_empty_text(
                f"role4.shots[{index}].{field_name}",
                item.get(field_name, ""),
            )
        normalized_items.append(normalized_item)
    if seen_ids != known_shot_ids:
        missing_ids = sorted(known_shot_ids.difference(seen_ids))
        raise ModuleBV2ParseError(f"role4.shots 覆盖不完整，missing={missing_ids}")
    return {"shots": normalized_items}


def _validate_visual_asset_refs(items: Any, field_name: str, valid_ids: set[str]) -> list[dict[str, Any]]:
    """
    功能说明：校验角色1单类对象 refs。
    参数说明：
    - items: refs 列表。
    - field_name: 字段名。
    - valid_ids: 合法 item_id 集合。
    返回值：
    - list[dict[str, Any]]: 标准化后的 refs。
    异常说明：
    - ModuleBV2ParseError: 字段或 ID 非法时抛出。
    边界条件：输入目录为空时允许返回空数组。
    """
    if not isinstance(items, list):
        raise ModuleBV2ParseError(f"{field_name} 必须是列表。")
    seen_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"{field_name}[{index}] 必须是对象。")
        item_id = normalize_non_empty_text(f"{field_name}[{index}].item_id", item.get("item_id", ""))
        if item_id not in valid_ids:
            raise ModuleBV2ParseError(f"{field_name}[{index}] 引用了未知 item_id：{item_id}")
        if item_id in seen_ids:
            raise ModuleBV2ParseError(f"{field_name} 存在重复 item_id：{item_id}")
        refs = item.get("refs", [])
        if not isinstance(refs, list) or not refs:
            raise ModuleBV2ParseError(f"{field_name}[{index}].refs 必须是非空列表。")
        normalized_refs: list[dict[str, Any]] = []
        seen_ref_ids: set[str] = set()
        for ref_index, ref_item in enumerate(refs):
            if not isinstance(ref_item, dict):
                raise ModuleBV2ParseError(f"{field_name}[{index}].refs[{ref_index}] 必须是对象。")
            ref_id = normalize_non_empty_text(
                f"{field_name}[{index}].refs[{ref_index}].ref_id",
                ref_item.get("ref_id", ""),
            )
            if ref_id in seen_ref_ids:
                raise ModuleBV2ParseError(f"{field_name}[{index}] 存在重复 ref_id：{ref_id}")
            seen_ref_ids.add(ref_id)
            normalized_refs.append(
                {
                    "ref_id": ref_id,
                    "pos_zh": normalize_non_empty_text(
                        f"{field_name}[{index}].refs[{ref_index}].pos_zh",
                        ref_item.get("pos_zh", ""),
                    ),
                    "pos_en": normalize_non_empty_text(
                        f"{field_name}[{index}].refs[{ref_index}].pos_en",
                        ref_item.get("pos_en", ""),
                    ),
                    "neg_zh": normalize_non_empty_text(
                        f"{field_name}[{index}].refs[{ref_index}].neg_zh",
                        ref_item.get("neg_zh", ""),
                    ),
                    "neg_en": normalize_non_empty_text(
                        f"{field_name}[{index}].refs[{ref_index}].neg_en",
                        ref_item.get("neg_en", ""),
                    ),
                }
            )
        seen_ids.add(item_id)
        normalized_items.append({"item_id": item_id, "refs": normalized_refs})
    return normalized_items


def _validate_id_list(items: Any, *, field_name: str, valid_ids: set[str]) -> list[str]:
    """
    功能说明：校验字符串 ID 列表。
    参数说明：
    - items: 输入数组。
    - field_name: 字段名。
    - valid_ids: 合法 ID 集合。
    返回值：
    - list[str]: 标准化后的唯一 ID 列表。
    异常说明：
    - ModuleBV2ParseError: 结构或 ID 非法时抛出。
    边界条件：允许空数组。
    """
    if not isinstance(items, list):
        raise ModuleBV2ParseError(f"{field_name} 必须是列表。")
    result: list[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        item_id = normalize_non_empty_text(f"{field_name}[{index}]", item)
        if item_id not in valid_ids:
            raise ModuleBV2ParseError(f"{field_name}[{index}] 引用了未知 ID：{item_id}")
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        result.append(item_id)
    return result


def _validate_single_id(value: Any, *, field_name: str, valid_ids: set[str]) -> str:
    """
    功能说明：校验单个字符串 ID。
    参数说明：
    - value: 输入值。
    - field_name: 字段名。
    - valid_ids: 合法 ID 集合。
    返回值：
    - str: 标准化后的 ID。
    异常说明：
    - ModuleBV2ParseError: ID 非法时抛出。
    边界条件：仅接受非空字符串。
    """
    normalized_id = normalize_non_empty_text(field_name, value)
    if normalized_id not in valid_ids:
        raise ModuleBV2ParseError(f"{field_name} 引用了未知 ID：{normalized_id}")
    return normalized_id


def _validate_camera_mapping_rules(data: Any, camera_preset_ids: set[str]) -> None:
    """
    功能说明：校验模板中的运镜映射规则。
    参数说明：
    - data: camera_mapping 原始数据。
    - camera_preset_ids: 合法 preset_id 集合。
    返回值：无。
    异常说明：
    - ModuleBV2ParseError: 结构、重复键或 preset 引用非法时抛出。
    边界条件：要求覆盖 3x3 的 energy/trend 组合。
    """
    if not isinstance(data, list):
        raise ModuleBV2ParseError("camera_mapping 必须是列表。")
    valid_energy_levels = {"low", "mid", "high"}
    valid_trends = {"down", "flat", "up"}
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"camera_mapping[{index}] 必须是对象。")
        energy_level = normalize_non_empty_text(f"camera_mapping[{index}].energy_level", item.get("energy_level", ""))
        trend = normalize_non_empty_text(f"camera_mapping[{index}].trend", item.get("trend", ""))
        if energy_level not in valid_energy_levels:
            raise ModuleBV2ParseError(f"camera_mapping[{index}].energy_level 非法：{energy_level}")
        if trend not in valid_trends:
            raise ModuleBV2ParseError(f"camera_mapping[{index}].trend 非法：{trend}")
        if (energy_level, trend) in seen_keys:
            raise ModuleBV2ParseError(f"camera_mapping 存在重复规则：{energy_level}/{trend}")
        seen_keys.add((energy_level, trend))
        default_preset_id = normalize_non_empty_text(
            f"camera_mapping[{index}].default_preset_id",
            item.get("default_preset_id", ""),
        )
        if default_preset_id not in camera_preset_ids:
            raise ModuleBV2ParseError(f"camera_mapping[{index}] 引用了未知 camera preset：{default_preset_id}")
        candidate_ids = _validate_preset_id_list(
            item.get("candidate_preset_ids", []),
            field_name=f"camera_mapping[{index}].candidate_preset_ids",
            valid_ids=camera_preset_ids,
        )
        if "none" not in candidate_ids:
            raise ModuleBV2ParseError(f"camera_mapping[{index}] 候选中必须包含 none。")
    expected_keys = {(energy, trend) for energy in valid_energy_levels for trend in valid_trends}
    if seen_keys != expected_keys:
        missing_keys = sorted(expected_keys.difference(seen_keys))
        raise ModuleBV2ParseError(f"camera_mapping 覆盖不完整，missing={missing_keys}")


def _validate_transition_mapping_rules(data: Any, transition_preset_ids: set[str]) -> None:
    """
    功能说明：校验模板中的转场映射规则。
    参数说明：
    - data: transition_mapping 原始数据。
    - transition_preset_ids: 合法 preset_id 集合。
    返回值：无。
    异常说明：
    - ModuleBV2ParseError: 结构、重复键或 preset 引用非法时抛出。
    边界条件：要求覆盖 3x3 的 current/next energy 组合。
    """
    if not isinstance(data, list):
        raise ModuleBV2ParseError("transition_mapping 必须是列表。")
    valid_energy_levels = {"low", "mid", "high"}
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ModuleBV2ParseError(f"transition_mapping[{index}] 必须是对象。")
        current_energy = normalize_non_empty_text(
            f"transition_mapping[{index}].current_energy_level",
            item.get("current_energy_level", ""),
        )
        next_energy = normalize_non_empty_text(
            f"transition_mapping[{index}].next_energy_level",
            item.get("next_energy_level", ""),
        )
        if current_energy not in valid_energy_levels:
            raise ModuleBV2ParseError(f"transition_mapping[{index}].current_energy_level 非法：{current_energy}")
        if next_energy not in valid_energy_levels:
            raise ModuleBV2ParseError(f"transition_mapping[{index}].next_energy_level 非法：{next_energy}")
        if (current_energy, next_energy) in seen_keys:
            raise ModuleBV2ParseError(f"transition_mapping 存在重复规则：{current_energy}/{next_energy}")
        seen_keys.add((current_energy, next_energy))
        default_preset_id = normalize_non_empty_text(
            f"transition_mapping[{index}].default_preset_id",
            item.get("default_preset_id", ""),
        )
        if default_preset_id not in transition_preset_ids:
            raise ModuleBV2ParseError(
                f"transition_mapping[{index}] 引用了未知 transition preset：{default_preset_id}"
            )
        candidate_ids = _validate_preset_id_list(
            item.get("candidate_preset_ids", []),
            field_name=f"transition_mapping[{index}].candidate_preset_ids",
            valid_ids=transition_preset_ids,
        )
        if "none" not in candidate_ids:
            raise ModuleBV2ParseError(f"transition_mapping[{index}] 候选中必须包含 none。")
    expected_keys = {(left, right) for left in valid_energy_levels for right in valid_energy_levels}
    if seen_keys != expected_keys:
        missing_keys = sorted(expected_keys.difference(seen_keys))
        raise ModuleBV2ParseError(f"transition_mapping 覆盖不完整，missing={missing_keys}")


def _validate_preset_id_list(items: Any, *, field_name: str, valid_ids: set[str]) -> list[str]:
    """
    功能说明：校验 preset_id 列表。
    参数说明：
    - items: 输入数组。
    - field_name: 字段名。
    - valid_ids: 合法 preset 集合。
    返回值：
    - list[str]: 标准化后的 preset_id 列表。
    异常说明：
    - ModuleBV2ParseError: 结构或 preset 非法时抛出。
    边界条件：允许空数组，由上层决定是否额外限制。
    """
    if not isinstance(items, list):
        raise ModuleBV2ParseError(f"{field_name} 必须是列表。")
    result: list[str] = []
    for index, item in enumerate(items):
        preset_id = normalize_non_empty_text(f"{field_name}[{index}]", item)
        if preset_id not in valid_ids:
            raise ModuleBV2ParseError(f"{field_name}[{index}] 引用了未知 preset_id：{preset_id}")
        result.append(preset_id)
    return result


def _assert_visual_asset_full_coverage(items: list[dict[str, Any]], *, expected_ids: set[str], field_name: str) -> None:
    """
    功能说明：确保角色1输出完整覆盖输入目录中的全部对象。
    参数说明：
    - items: 已通过基础校验的 refs 列表。
    - expected_ids: 目录中的完整 item_id 集合。
    - field_name: 字段名。
    返回值：无。
    异常说明：
    - ModuleBV2ParseError: 覆盖不完整时抛出。
    边界条件：目录为空时允许返回空。
    """
    actual_ids = {str(item.get("item_id", "")).strip() for item in items if isinstance(item, dict)}
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids.difference(actual_ids))
        raise ModuleBV2ParseError(f"{field_name} 覆盖不完整，missing={missing_ids}")
