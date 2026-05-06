"""
文件用途：加载并编译模块B v2 的编排模板 Markdown 文档。
核心流程：读取 Markdown -> 解析固定 section/subsection -> 校验并输出模板对象。
输入输出：输入模板路径，输出标准化编排模板字典。
依赖说明：依赖标准库 pathlib/json 与项目内 Markdown 解析器和 parser。
维护说明：模板源文件固定放在 configs/storyboard_templates，运行时可编译为 task artifact。
"""

# 标准库：用于 JSON 解析。
import json
# 标准库：用于路径处理。
from pathlib import Path

# 项目内模块：统一 Markdown 解析器。
from music_video_pipeline.modules.module_b_v2.markdown_io import MarkdownNode, parse_markdown_document
# 项目内模块：导入 v2 常量。
from music_video_pipeline.modules.module_b_v2.models import DEFAULT_STORYBOARD_TEMPLATE_FILE
# 项目内模块：导入模板校验器。
from music_video_pipeline.modules.module_b_v2.parser import ModuleBV2ParseError, validate_storyboard_template


# 常量：模板元信息 section 标题。
SECTION_TEMPLATE_META = "template_meta"
# 常量：模板风格 section 标题。
SECTION_STYLE = "style"
# 常量：模板故事 section 标题。
SECTION_STORY = "story"
# 常量：场景目录 section 标题。
SECTION_SCENE_CATALOG = "scene_catalog"
# 常量：道具目录 section 标题。
SECTION_PROP_CATALOG = "prop_catalog"
# 常量：角色目录 section 标题。
SECTION_CHARACTER_CATALOG = "character_catalog"
# 常量：构图目录 section 标题。
SECTION_COMPOSITION_CATALOG = "composition_catalog"
# 常量：运镜预设 section 标题。
SECTION_CAMERA_PLAN_PRESETS = "camera_plan_presets"
# 常量：运镜映射 section 标题。
SECTION_CAMERA_MAPPING = "camera_mapping"
# 常量：转场预设 section 标题。
SECTION_TRANSITION_PRESETS = "transition_presets"
# 常量：转场映射 section 标题。
SECTION_TRANSITION_MAPPING = "transition_mapping"


def load_storyboard_template(project_root: Path, template_file: str = DEFAULT_STORYBOARD_TEMPLATE_FILE) -> dict:
    """
    功能说明：加载并校验编排模板 Markdown 文件。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径（支持相对路径）。
    返回值：
    - dict: 已标准化的模板对象。
    异常说明：
    - ModuleBV2ParseError: 路径、结构或字段非法时抛出。
    边界条件：模板必须使用固定 Markdown section/subsection 结构。
    """
    normalized_path = resolve_storyboard_template_path(project_root=project_root, template_file=template_file)
    if not normalized_path.exists():
        raise ModuleBV2ParseError(f"编排模板文件不存在：{normalized_path}")
    markdown_text = normalized_path.read_text(encoding="utf-8-sig")
    template_payload = _extract_storyboard_template_payload(markdown_text=markdown_text, template_path=normalized_path)
    return validate_storyboard_template(template_payload)


def resolve_storyboard_template_path(project_root: Path, template_file: str = DEFAULT_STORYBOARD_TEMPLATE_FILE) -> Path:
    """
    功能说明：将编排模板路径解析为绝对路径。
    参数说明：
    - project_root: 项目根目录。
    - template_file: 模板文件路径（支持相对路径）。
    返回值：
    - Path: 绝对模板路径。
    异常说明：无。
    边界条件：仅做路径解析，不校验文件是否存在。
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
    功能说明：从 Markdown section/subsection 编译 storyboard 模板对象。
    参数说明：
    - markdown_text: 原始 Markdown 文本。
    - template_path: 模板路径，用于错误定位。
    返回值：
    - dict: 解析后的模板对象。
    异常说明：
    - ModuleBV2ParseError: section 或字段缺失时抛出。
    边界条件：顶层只识别固定二级标题。
    """
    document = parse_markdown_document(markdown_text)
    section_map = {
        str(section.heading).strip().lower(): section
        for section in document.sections
        if str(section.heading).strip()
    }
    meta_section = _require_section(section_map=section_map, section_name=SECTION_TEMPLATE_META, template_path=template_path)
    style_section = _require_section(section_map=section_map, section_name=SECTION_STYLE, template_path=template_path)
    story_section = _require_section(section_map=section_map, section_name=SECTION_STORY, template_path=template_path)
    payload = {
        "template_id": _require_text(
            field_map=meta_section.fields,
            key="template_id",
            field_name="template_meta.template_id",
            template_path=template_path,
        ),
        "style": {
            "color_mode": _require_text(
                field_map=style_section.fields,
                key="color_mode",
                field_name="style.color_mode",
                template_path=template_path,
            ),
            "render_style": _require_text(
                field_map=style_section.fields,
                key="render_style",
                field_name="style.render_style",
                template_path=template_path,
            ),
        },
        "story": {
            "premise_zh": _require_text(
                field_map=story_section.fields,
                key="premise_zh",
                field_name="story.premise_zh",
                template_path=template_path,
            ),
        },
        "scene_catalog": _parse_catalog_section(
            section_node=_require_section(section_map=section_map, section_name=SECTION_SCENE_CATALOG, template_path=template_path),
            id_key="item_id",
            template_path=template_path,
        ),
        "prop_catalog": _parse_catalog_section(
            section_node=_require_section(section_map=section_map, section_name=SECTION_PROP_CATALOG, template_path=template_path),
            id_key="item_id",
            template_path=template_path,
        ),
        "character_catalog": _parse_catalog_section(
            section_node=_require_section(section_map=section_map, section_name=SECTION_CHARACTER_CATALOG, template_path=template_path),
            id_key="item_id",
            template_path=template_path,
        ),
        "composition_catalog": _parse_composition_section(
            section_node=_require_section(
                section_map=section_map,
                section_name=SECTION_COMPOSITION_CATALOG,
                template_path=template_path,
            ),
            template_path=template_path,
        ),
        "camera_plan_presets": _parse_camera_plan_presets_section(
            section_node=_require_section(
                section_map=section_map,
                section_name=SECTION_CAMERA_PLAN_PRESETS,
                template_path=template_path,
            ),
            template_path=template_path,
        ),
        "camera_mapping": _parse_camera_mapping_section(
            section_node=_require_section(section_map=section_map, section_name=SECTION_CAMERA_MAPPING, template_path=template_path),
            template_path=template_path,
        ),
        "transition_presets": _parse_transition_presets_section(
            section_node=_require_section(
                section_map=section_map,
                section_name=SECTION_TRANSITION_PRESETS,
                template_path=template_path,
            ),
            template_path=template_path,
        ),
        "transition_mapping": _parse_transition_mapping_section(
            section_node=_require_section(
                section_map=section_map,
                section_name=SECTION_TRANSITION_MAPPING,
                template_path=template_path,
            ),
            template_path=template_path,
        ),
    }
    return payload


def _require_section(*, section_map: dict[str, MarkdownNode], section_name: str, template_path: Path) -> MarkdownNode:
    """
    功能说明：按固定标题名获取必需 section。
    参数说明：
    - section_map: section 映射。
    - section_name: 目标 section 名。
    - template_path: 模板路径。
    返回值：
    - MarkdownNode: 命中的 section 节点。
    异常说明：
    - ModuleBV2ParseError: section 缺失时抛出。
    边界条件：section 名匹配时统一按小写比较。
    """
    section_node = section_map.get(str(section_name).strip().lower())
    if section_node is None:
        raise ModuleBV2ParseError(f"编排模板缺失 `## {section_name}` section：{template_path}")
    return section_node


def _require_text(*, field_map: dict[str, str], key: str, field_name: str, template_path: Path) -> str:
    """
    功能说明：读取必填文本字段。
    参数说明：
    - field_map: 字段映射。
    - key: 目标字段名。
    - field_name: 用于报错的业务字段名。
    - template_path: 模板路径。
    返回值：
    - str: 去空白后的字段值。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或为空时抛出。
    边界条件：仅接受非空字符串。
    """
    value = str(field_map.get(key, "")).strip()
    if not value:
        raise ModuleBV2ParseError(f"编排模板字段缺失或为空：{field_name}，template={template_path}")
    return value


def _parse_catalog_section(*, section_node: MarkdownNode, id_key: str, template_path: Path) -> list[dict]:
    """
    功能说明：解析场景/道具/角色目录 section。
    参数说明：
    - section_node: 目录 section。
    - id_key: 顶层对象 ID 字段名。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 标准化目录条目数组。
    异常说明：
    - ModuleBV2ParseError: 子项缺失时抛出。
    边界条件：每个目录条目必须使用三级标题。
    """
    items: list[dict] = []
    if not section_node.subsections:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    for item_node in section_node.subsections:
        item_id = str(item_node.heading).strip()
        if not item_id:
            raise ModuleBV2ParseError(f"编排模板存在空标题目录项：section={section_node.heading}，template={template_path}")
        items.append(
            {
                id_key: item_id,
                "name_zh": _require_text(
                    field_map=item_node.fields,
                    key="name_zh",
                    field_name=f"{section_node.heading}.{item_id}.name_zh",
                    template_path=template_path,
                ),
                "description_zh": _require_text(
                    field_map=item_node.fields,
                    key="description_zh",
                    field_name=f"{section_node.heading}.{item_id}.description_zh",
                    template_path=template_path,
                ),
            }
        )
    return items


def _parse_composition_section(*, section_node: MarkdownNode, template_path: Path) -> list[dict]:
    """
    功能说明：解析构图目录 section。
    参数说明：
    - section_node: 构图目录 section。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 构图数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失时抛出。
    边界条件：prompt_tags_en 允许用逗号分隔单行书写。
    """
    items: list[dict] = []
    if not section_node.subsections:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    for item_node in section_node.subsections:
        composition_id = str(item_node.heading).strip()
        items.append(
            {
                "composition_id": composition_id,
                "name_zh": _require_text(
                    field_map=item_node.fields,
                    key="name_zh",
                    field_name=f"{section_node.heading}.{composition_id}.name_zh",
                    template_path=template_path,
                ),
                "description_zh": _require_text(
                    field_map=item_node.fields,
                    key="description_zh",
                    field_name=f"{section_node.heading}.{composition_id}.description_zh",
                    template_path=template_path,
                ),
                "prompt_tags_en": _parse_csv_field(
                    field_map=item_node.fields,
                    key="prompt_tags_en",
                    field_name=f"{section_node.heading}.{composition_id}.prompt_tags_en",
                    template_path=template_path,
                ),
                "safe_for_closeup": _parse_bool_field(
                    field_map=item_node.fields,
                    key="safe_for_closeup",
                    field_name=f"{section_node.heading}.{composition_id}.safe_for_closeup",
                    template_path=template_path,
                ),
                "safe_for_motion": _parse_bool_field(
                    field_map=item_node.fields,
                    key="safe_for_motion",
                    field_name=f"{section_node.heading}.{composition_id}.safe_for_motion",
                    template_path=template_path,
                ),
            }
        )
    return items


def _parse_camera_plan_presets_section(*, section_node: MarkdownNode, template_path: Path) -> list[dict]:
    """
    功能说明：解析运镜预设 section。
    参数说明：
    - section_node: 运镜预设 section。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 运镜预设数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失时抛出。
    边界条件：preset_id 直接使用三级标题名。
    """
    items: list[dict] = []
    for item_node in section_node.subsections:
        preset_id = str(item_node.heading).strip()
        items.append(
            {
                "preset_id": preset_id,
                "mode": _require_text(
                    field_map=item_node.fields,
                    key="mode",
                    field_name=f"{section_node.heading}.{preset_id}.mode",
                    template_path=template_path,
                ),
                "direction": _require_text(
                    field_map=item_node.fields,
                    key="direction",
                    field_name=f"{section_node.heading}.{preset_id}.direction",
                    template_path=template_path,
                ),
                "strength": _require_text(
                    field_map=item_node.fields,
                    key="strength",
                    field_name=f"{section_node.heading}.{preset_id}.strength",
                    template_path=template_path,
                ),
                "easing": _require_text(
                    field_map=item_node.fields,
                    key="easing",
                    field_name=f"{section_node.heading}.{preset_id}.easing",
                    template_path=template_path,
                ),
            }
        )
    if not items:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    return items


def _parse_camera_mapping_section(*, section_node: MarkdownNode, template_path: Path) -> list[dict]:
    """
    功能说明：解析运镜映射 section。
    参数说明：
    - section_node: 运镜映射 section。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 运镜映射数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失时抛出。
    边界条件：三级标题仅作人读标签，不参与最终契约。
    """
    items: list[dict] = []
    for item_node in section_node.subsections:
        item_name = str(item_node.heading).strip() or "camera_mapping_item"
        items.append(
            {
                "energy_level": _require_text(
                    field_map=item_node.fields,
                    key="energy_level",
                    field_name=f"{section_node.heading}.{item_name}.energy_level",
                    template_path=template_path,
                ),
                "trend": _require_text(
                    field_map=item_node.fields,
                    key="trend",
                    field_name=f"{section_node.heading}.{item_name}.trend",
                    template_path=template_path,
                ),
                "default_preset_id": _require_text(
                    field_map=item_node.fields,
                    key="default_preset_id",
                    field_name=f"{section_node.heading}.{item_name}.default_preset_id",
                    template_path=template_path,
                ),
                "candidate_preset_ids": _parse_csv_field(
                    field_map=item_node.fields,
                    key="candidate_preset_ids",
                    field_name=f"{section_node.heading}.{item_name}.candidate_preset_ids",
                    template_path=template_path,
                ),
            }
        )
    if not items:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    return items


def _parse_transition_presets_section(*, section_node: MarkdownNode, template_path: Path) -> list[dict]:
    """
    功能说明：解析转场预设 section。
    参数说明：
    - section_node: 转场预设 section。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 转场预设数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或类型非法时抛出。
    边界条件：duration_ms 必须是非负整数文本。
    """
    items: list[dict] = []
    for item_node in section_node.subsections:
        preset_id = str(item_node.heading).strip()
        items.append(
            {
                "preset_id": preset_id,
                "kind": _require_text(
                    field_map=item_node.fields,
                    key="kind",
                    field_name=f"{section_node.heading}.{preset_id}.kind",
                    template_path=template_path,
                ),
                "duration_ms": _parse_int_field(
                    field_map=item_node.fields,
                    key="duration_ms",
                    field_name=f"{section_node.heading}.{preset_id}.duration_ms",
                    template_path=template_path,
                ),
                "easing": _require_text(
                    field_map=item_node.fields,
                    key="easing",
                    field_name=f"{section_node.heading}.{preset_id}.easing",
                    template_path=template_path,
                ),
            }
        )
    if not items:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    return items


def _parse_transition_mapping_section(*, section_node: MarkdownNode, template_path: Path) -> list[dict]:
    """
    功能说明：解析转场映射 section。
    参数说明：
    - section_node: 转场映射 section。
    - template_path: 模板路径。
    返回值：
    - list[dict]: 转场映射数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失时抛出。
    边界条件：三级标题仅作人读标签，不参与最终契约。
    """
    items: list[dict] = []
    for item_node in section_node.subsections:
        item_name = str(item_node.heading).strip() or "transition_mapping_item"
        items.append(
            {
                "current_energy_level": _require_text(
                    field_map=item_node.fields,
                    key="current_energy_level",
                    field_name=f"{section_node.heading}.{item_name}.current_energy_level",
                    template_path=template_path,
                ),
                "next_energy_level": _require_text(
                    field_map=item_node.fields,
                    key="next_energy_level",
                    field_name=f"{section_node.heading}.{item_name}.next_energy_level",
                    template_path=template_path,
                ),
                "default_preset_id": _require_text(
                    field_map=item_node.fields,
                    key="default_preset_id",
                    field_name=f"{section_node.heading}.{item_name}.default_preset_id",
                    template_path=template_path,
                ),
                "candidate_preset_ids": _parse_csv_field(
                    field_map=item_node.fields,
                    key="candidate_preset_ids",
                    field_name=f"{section_node.heading}.{item_name}.candidate_preset_ids",
                    template_path=template_path,
                ),
            }
        )
    if not items:
        raise ModuleBV2ParseError(f"编排模板 section 不能为空：{section_node.heading}，template={template_path}")
    return items


def _parse_csv_field(*, field_map: dict[str, str], key: str, field_name: str, template_path: Path) -> list[str]:
    """
    功能说明：解析逗号分隔文本数组字段。
    参数说明：
    - field_map: 字段映射。
    - key: 目标字段名。
    - field_name: 用于报错的业务字段名。
    - template_path: 模板路径。
    返回值：
    - list[str]: 非空文本数组。
    异常说明：
    - ModuleBV2ParseError: 字段缺失或为空时抛出。
    边界条件：支持英文逗号和中文逗号。
    """
    raw_value = _require_text(field_map=field_map, key=key, field_name=field_name, template_path=template_path)
    items = [part.strip() for part in raw_value.replace("，", ",").split(",")]
    result = [part for part in items if part]
    if not result:
        raise ModuleBV2ParseError(f"编排模板字段缺失或为空：{field_name}，template={template_path}")
    return result


def _parse_bool_field(*, field_map: dict[str, str], key: str, field_name: str, template_path: Path) -> bool:
    """
    功能说明：解析布尔字段。
    参数说明：
    - field_map: 字段映射。
    - key: 目标字段名。
    - field_name: 用于报错的业务字段名。
    - template_path: 模板路径。
    返回值：
    - bool: 解析后的布尔值。
    异常说明：
    - ModuleBV2ParseError: 值不是 true/false 时抛出。
    边界条件：大小写不敏感。
    """
    raw_value = _require_text(field_map=field_map, key=key, field_name=field_name, template_path=template_path).lower()
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    raise ModuleBV2ParseError(f"编排模板布尔字段非法：{field_name}={raw_value}，template={template_path}")


def _parse_int_field(*, field_map: dict[str, str], key: str, field_name: str, template_path: Path) -> int:
    """
    功能说明：解析整数文本字段。
    参数说明：
    - field_map: 字段映射。
    - key: 目标字段名。
    - field_name: 用于报错的业务字段名。
    - template_path: 模板路径。
    返回值：
    - int: 解析后的整数。
    异常说明：
    - ModuleBV2ParseError: 非整数文本时抛出。
    边界条件：允许 0。
    """
    raw_value = _require_text(field_map=field_map, key=key, field_name=field_name, template_path=template_path)
    try:
        return int(raw_value)
    except ValueError as error:
        raise ModuleBV2ParseError(f"编排模板整数字段非法：{field_name}={raw_value}，template={template_path}") from error
