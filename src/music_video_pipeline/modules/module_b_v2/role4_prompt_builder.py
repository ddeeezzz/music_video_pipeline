"""
文件用途：实现模块B v2 的角色4“关键帧提示词生成器”。
核心流程：按 shot 并发请求 LLM，将视觉词库与镜头编排融合为双关键帧与视频提示词块。
输入输出：输入视觉词库、角色3编排结果与模板风格，输出提示词块标准结构。
依赖说明：依赖标准库并发工具、v2 LLM runtime、Markdown 渲染器与 parser。
维护说明：本角色只产出提示词，不负责 camera/transition 选择。
"""

# 标准库：用于并发执行。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：v2 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：统一 Markdown 渲染。
from music_video_pipeline.modules.module_b_v2.markdown_io import (
    MarkdownFieldSchema,
    render_repeated_sections,
    render_schema_fields,
)
# 项目内模块：v2 parser。
from music_video_pipeline.modules.module_b_v2.parser import (
    parse_role4_prompt_markdown,
    validate_role4_prompt_output,
)
# 项目内模块：统一 prompt 模板加载。
from music_video_pipeline.modules.module_b_v2.prompt_templates import (
    ROLE4_PROMPT_ASSET,
    render_prompt_asset,
)


# 常量：角色4单镜头提示词输出默认 token 下限。
ROLE4_PROMPT_BUILDER_MIN_MAX_TOKENS = 2200
# 常量：角色4单镜头提示词请求超时（秒）。
ROLE4_PROMPT_BUILDER_TIMEOUT_SECONDS = 180.0
# 常量：角色4风格字段 schema。
ROLE4_STYLE_SCHEMA = [
    MarkdownFieldSchema("色彩风格", "style.color_mode", ""),
    MarkdownFieldSchema("画风", "style.render_style", ""),
]
# 常量：角色4镜头摘要字段 schema。
ROLE4_SHOT_BRIEF_SCHEMA = [
    MarkdownFieldSchema("shot_id", "shot_brief.shot_id", ""),
    MarkdownFieldSchema("scene_desc_zh", "shot_brief.scene_desc_zh", ""),
    MarkdownFieldSchema("selected_scene_id", "shot_brief.selected_scene_id", ""),
    MarkdownFieldSchema("selected_character_ids", "shot_brief.selected_character_ids", []),
    MarkdownFieldSchema("selected_prop_ids", "shot_brief.selected_prop_ids", []),
    MarkdownFieldSchema("composition_id", "shot_brief.composition.composition_id", ""),
    MarkdownFieldSchema("composition_name_zh", "shot_brief.composition.name_zh", ""),
    MarkdownFieldSchema("composition_desc_zh", "shot_brief.composition.description_zh", ""),
    MarkdownFieldSchema("composition_tags_en", "shot_brief.composition.prompt_tags_en", []),
]
# 常量：角色4视觉参考字段 schema。
ROLE4_VISUAL_REFERENCE_SCHEMA = [
    MarkdownFieldSchema("item_id", "item_id", "none"),
    MarkdownFieldSchema(
        "positive_cues_en",
        "positive_cues_en",
        [],
        lambda value: " | ".join([str(item).strip() for item in value if str(item).strip()]) or "none",
    ),
    MarkdownFieldSchema(
        "negative_cues_en",
        "negative_cues_en",
        [],
        lambda value: " | ".join([str(item).strip() for item in value if str(item).strip()]) or "none",
    ),
]


class Role4PromptBuilder:
    """
    功能说明：执行角色4提示词构建。
    参数说明：
    - llm_runtime: 通用 LLM 运行时。
    返回值：不适用。
    异常说明：具体异常由 generate 抛出。
    边界条件：shot 之间完全独立，允许并发。
    """

    def __init__(self, llm_runtime: ModuleBV2LlmRuntime) -> None:
        self._llm_runtime = llm_runtime

    def _build_composition_lookup(self, storyboard_template: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        功能说明：构建 composition_id 到构图说明的轻量索引。
        参数说明：
        - storyboard_template: 已编译模板。
        返回值：
        - dict[str, dict[str, Any]]: 构图索引。
        异常说明：无。
        边界条件：无效条目会被过滤。
        """
        lookup: dict[str, dict[str, Any]] = {}
        for item in storyboard_template.get("composition_catalog", []):
            if not isinstance(item, dict):
                continue
            composition_id = str(item.get("composition_id", "")).strip()
            if not composition_id:
                continue
            lookup[composition_id] = {
                "composition_id": composition_id,
                "name_zh": str(item.get("name_zh", "")).strip(),
                "description_zh": str(item.get("description_zh", "")).strip(),
                "prompt_tags_en": [
                    str(tag).strip() for tag in item.get("prompt_tags_en", []) if str(tag).strip()
                ],
            }
        return lookup

    def _build_shot_brief(
        self,
        *,
        shot_item: dict[str, Any],
        composition_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        功能说明：压缩角色3单镜头结果，仅保留角色4构词所需字段。
        参数说明：
        - shot_item: 角色3单镜头结果。
        - composition_lookup: 构图索引。
        返回值：
        - dict[str, Any]: 轻量镜头摘要对象。
        异常说明：无。
        边界条件：构图缺失时返回空对象兜底。
        """
        composition_id = str(shot_item.get("composition_id", "")).strip()
        return {
            "shot_id": str(shot_item.get("shot_id", "")).strip(),
            "scene_desc_zh": str(shot_item.get("scene_desc_zh", "")).strip(),
            "selected_scene_id": str(shot_item.get("selected_scene_id", "")).strip(),
            "selected_character_ids": [
                str(item).strip() for item in shot_item.get("selected_character_ids", []) if str(item).strip()
            ],
            "selected_prop_ids": [
                str(item).strip() for item in shot_item.get("selected_prop_ids", []) if str(item).strip()
            ],
            "composition": dict(composition_lookup.get(composition_id, {"composition_id": composition_id})),
        }

    def _compact_visual_reference_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：压缩角色1视觉参考，只保留角色4真正消费的英文风格线索。
        参数说明：
        - item: 角色1单对象 refs。
        返回值：
        - dict[str, Any]: 轻量视觉参考对象。
        异常说明：无。
        边界条件：缺失 refs 时返回空数组字段。
        """
        refs = item.get("refs", [])
        positive_cues_en = []
        negative_cues_en = []
        if isinstance(refs, list):
            for ref_item in refs:
                if not isinstance(ref_item, dict):
                    continue
                pos_en = str(ref_item.get("pos_en", "")).strip()
                neg_en = str(ref_item.get("neg_en", "")).strip()
                if pos_en:
                    positive_cues_en.append(pos_en)
                if neg_en:
                    negative_cues_en.append(neg_en)
        return {
            "item_id": str(item.get("item_id", "")).strip(),
            "positive_cues_en": positive_cues_en,
            "negative_cues_en": negative_cues_en,
        }

    def generate(
        self,
        *,
        storyboard_template: dict[str, Any],
        role1_output: dict[str, Any],
        role3_output: dict[str, Any],
        target_shot_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        功能说明：按 shot 并发生成双关键帧与视频提示词块。
        参数说明：
        - storyboard_template: 已编译模板。
        - role1_output: 角色1视觉词库。
        - role3_output: 角色3镜头编排。
        - target_shot_ids: 可选，仅为这些 shot 生成提示词。
        返回值：
        - dict[str, Any]: 角色4输出。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：target_shot_ids 为空时默认生成全部 shot。
        """
        selected_shots = [
            dict(item)
            for item in role3_output.get("shots", [])
            if isinstance(item, dict) and (not target_shot_ids or str(item.get("shot_id", "")).strip() in target_shot_ids)
        ]
        self._llm_runtime.logger.info("模块B v2 role4 开始执行，shot_count=%s", len(selected_shots))
        generation_context = self.build_generation_context(
            storyboard_template=storyboard_template,
            role1_output=role1_output,
        )
        result_shots: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(selected_shots) or 1))) as executor:
            future_map = {
                executor.submit(
                    self.generate_one,
                    storyboard_template=storyboard_template,
                    shot_item=shot_item,
                    generation_context=generation_context,
                ): str(shot_item.get("shot_id", "")).strip()
                for shot_item in selected_shots
            }
            for future in as_completed(future_map):
                result_shots.append(future.result())
        result_shots.sort(key=lambda item: str(item.get("shot_id", "")))
        validated_output = validate_role4_prompt_output(
            data={"shots": result_shots},
            shot_ids=[str(item.get("shot_id", "")).strip() for item in selected_shots],
        )
        self._llm_runtime.logger.info("模块B v2 role4 执行完成")
        return validated_output

    def build_generation_context(
        self,
        *,
        storyboard_template: dict[str, Any],
        role1_output: dict[str, Any],
    ) -> dict[str, Any]:
        """
        功能说明：构建角色4批量/单镜头共用的只读上下文。
        参数说明：
        - storyboard_template: 已编译模板。
        - role1_output: 角色1视觉词库。
        返回值：
        - dict[str, Any]: 角色4生成上下文。
        异常说明：无。
        边界条件：上下文只包含静态索引，便于 role3 完成后立刻串流调用。
        """
        return {
            "visual_lookup": self._build_visual_lookup(role1_output=role1_output),
            "composition_lookup": self._build_composition_lookup(storyboard_template=storyboard_template),
        }

    def generate_one(
        self,
        *,
        storyboard_template: dict[str, Any],
        shot_item: dict[str, Any],
        generation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        功能说明：为单个 shot 生成提示词块。
        参数说明：
        - storyboard_template: 已编译模板。
        - shot_item: 角色3输出的单镜头结果。
        - generation_context: 由 build_generation_context 构建的只读索引。
        返回值：
        - dict[str, Any]: 单镜头提示词块。
        异常说明：LLM 或解析失败时抛出异常。
        边界条件：缺少视觉词库时仍会把对象选择信息原样传给模型。
        """
        visual_lookup = generation_context.get("visual_lookup", {})
        composition_lookup = generation_context.get("composition_lookup", {})
        shot_brief = self._build_shot_brief(
            shot_item=shot_item,
            composition_lookup=composition_lookup,
        )
        self._llm_runtime.logger.info(
            "模块B v2 role4 开始生成 shot，shot_id=%s",
            str(shot_brief.get("shot_id", "")).strip(),
        )
        selected_scene_id = str(shot_brief.get("selected_scene_id", "")).strip()
        selected_character_ids = [
            str(item).strip() for item in shot_brief.get("selected_character_ids", []) if str(item).strip()
        ]
        selected_prop_ids = [str(item).strip() for item in shot_brief.get("selected_prop_ids", []) if str(item).strip()]
        payload = {
            "style": storyboard_template.get("style", {}),
            "shot_brief": shot_brief,
            "style_reference": {
                "scene": self._compact_visual_reference_item(visual_lookup.get(selected_scene_id, {})),
                "characters": [
                    self._compact_visual_reference_item(visual_lookup.get(item_id, {}))
                    for item_id in selected_character_ids
                ],
                "props": [
                    self._compact_visual_reference_item(visual_lookup.get(item_id, {}))
                    for item_id in selected_prop_ids
                ],
            },
        }
        prompt_asset = render_prompt_asset(
            project_root=self._llm_runtime.project_root,
            prompt_asset=ROLE4_PROMPT_ASSET,
            user_variables=self._build_prompt_variables(payload),
        )
        response_text = self._llm_runtime.call_markdown(
            role_name=f"role4_prompt_builder:{shot_item.get('shot_id', '')}",
            system_prompt=prompt_asset.system_prompt,
            user_prompt_markdown=prompt_asset.user_prompt_markdown,
            max_tokens_override=ROLE4_PROMPT_BUILDER_MIN_MAX_TOKENS,
            timeout_seconds_override=ROLE4_PROMPT_BUILDER_TIMEOUT_SECONDS,
        )
        response = parse_role4_prompt_markdown(response_text)
        response["shot_id"] = str(shot_item.get("shot_id", "")).strip()
        self._llm_runtime.logger.info(
            "模块B v2 role4 完成 shot，shot_id=%s",
            str(shot_item.get("shot_id", "")).strip(),
        )
        return response

    def _build_prompt_variables(self, payload: dict[str, Any]) -> dict[str, str]:
        """
        功能说明：构建角色4 user prompt 模板变量。
        参数说明：
        - payload: 当前 shot 的精简输入。
        返回值：
        - dict[str, str]: user prompt 模板变量映射。
        异常说明：无。
        边界条件：视觉参考按对象整合，避免拆成碎字段块。
        """
        shot_brief = payload.get("shot_brief", {}) if isinstance(payload.get("shot_brief"), dict) else {}
        style_reference = payload.get("style_reference", {}) if isinstance(payload.get("style_reference"), dict) else {}
        scene_ref = style_reference.get("scene", {}) if isinstance(style_reference.get("scene"), dict) else {}
        character_refs = style_reference.get("characters", []) if isinstance(style_reference.get("characters"), list) else []
        prop_refs = style_reference.get("props", []) if isinstance(style_reference.get("props"), list) else []
        visual_reference_items: list[dict[str, Any]] = []
        if scene_ref:
            visual_reference_items.append({"section_heading": "scene", **scene_ref})
        for index, item in enumerate(character_refs, start=1):
            if isinstance(item, dict):
                visual_reference_items.append({"section_heading": f"character_{index}", **item})
        for index, item in enumerate(prop_refs, start=1):
            if isinstance(item, dict):
                visual_reference_items.append({"section_heading": f"prop_{index}", **item})

        return {
            "style_block": render_schema_fields(payload, ROLE4_STYLE_SCHEMA),
            "shot_brief": render_schema_fields(payload, ROLE4_SHOT_BRIEF_SCHEMA),
            "visual_reference": render_repeated_sections(
                visual_reference_items,
                heading_builder=lambda item, _index: str(item.get("section_heading", "")).strip() or "reference",
                field_schema=ROLE4_VISUAL_REFERENCE_SCHEMA,
                level=3,
            ),
        }

    def _build_visual_lookup(self, role1_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        功能说明：将角色1输出整理为 item_id 到 refs 的统一索引。
        参数说明：
        - role1_output: 角色1输出。
        返回值：
        - dict[str, dict[str, Any]]: 统一视觉索引。
        异常说明：无。
        边界条件：后出现的同名 item_id 会覆盖前值，模板本身已保证唯一。
        """
        lookup: dict[str, dict[str, Any]] = {}
        for field_name in ("scene_refs", "prop_refs", "character_refs"):
            for item in role1_output.get(field_name, []):
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("item_id", "")).strip()
                if item_id:
                    lookup[item_id] = dict(item)
        return lookup
