"""
文件用途：实现模块B v2 的角色1“视觉编导”。
核心流程：按场景/道具/角色三类并发请求 LLM，生成对象级参考提示词词库。
输入输出：输入模板风格与对象目录，输出结构化视觉词库标准结构。
依赖说明：依赖标准库并发工具、v2 LLM runtime、Markdown 渲染器与 parser。
维护说明：本角色只负责“对象长什么样”，不描述剧情与镜头动作。
"""

# 标准库：用于并发执行。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：v2 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：v2 常量与数据结构。
from music_video_pipeline.modules.module_b_v2.markdown_io import render_bullet_fields, render_document, render_heading_block
from music_video_pipeline.modules.module_b_v2.models import (
    VISUAL_ASSET_KIND_CHARACTER,
    VISUAL_ASSET_KIND_PROP,
    VISUAL_ASSET_KIND_SCENE,
    Role1VisualCatalogOutput,
)
# 项目内模块：v2 parser。
from music_video_pipeline.modules.module_b_v2.parser import (
    parse_role1_visual_catalog_markdown,
    validate_role1_visual_catalog_output,
)
# 项目内模块：统一 prompt 模板加载。
from music_video_pipeline.modules.module_b_v2.prompt_templates import (
    ROLE1_PROMPT_ASSET,
    render_prompt_asset,
)


# 常量：角色1视觉词库输出默认 token 上限。
ROLE1_VISUAL_DIRECTOR_MIN_MAX_TOKENS = 1400
# 常量：角色1视觉词库请求超时（秒）。
ROLE1_VISUAL_DIRECTOR_TIMEOUT_SECONDS = 180.0


class Role1VisualDirector:
    """
    功能说明：执行角色1视觉词库生成。
    参数说明：
    - llm_runtime: 通用 LLM 运行时。
    返回值：不适用。
    异常说明：具体异常由 generate 抛出。
    边界条件：场景/道具/角色三类输入为空时返回空数组。
    """

    def __init__(self, llm_runtime: ModuleBV2LlmRuntime) -> None:
        self._llm_runtime = llm_runtime

    def generate(self, storyboard_template: dict[str, Any]) -> Role1VisualCatalogOutput:
        """
        功能说明：并发生成场景/道具/角色三类参考图提示词词库。
        参数说明：
        - storyboard_template: 已编译编排模板。
        返回值：
        - Role1VisualCatalogOutput: 视觉词库输出。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：每个对象固定要求返回 2 组 refs。
        """
        self._llm_runtime.logger.info("模块B v2 role1 开始执行")
        tasks = [
            (VISUAL_ASSET_KIND_SCENE, storyboard_template.get("scene_catalog", [])),
            (VISUAL_ASSET_KIND_PROP, storyboard_template.get("prop_catalog", [])),
            (VISUAL_ASSET_KIND_CHARACTER, storyboard_template.get("character_catalog", [])),
        ]
        result_map = {
            VISUAL_ASSET_KIND_SCENE: [],
            VISUAL_ASSET_KIND_PROP: [],
            VISUAL_ASSET_KIND_CHARACTER: [],
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_map = {
                executor.submit(
                    self._generate_asset_refs,
                    asset_kind=asset_kind,
                    style_payload=storyboard_template.get("style", {}),
                    asset_items=asset_items,
                ): asset_kind
                for asset_kind, asset_items in tasks
            }
            for future in as_completed(future_map):
                asset_kind = future_map[future]
                result_map[asset_kind] = future.result()
                self._llm_runtime.logger.info(
                    "模块B v2 role1 完成一类对象，asset_kind=%s，item_count=%s",
                    asset_kind,
                    len(result_map[asset_kind]),
                )

        payload: Role1VisualCatalogOutput = {
            "scene_refs": result_map[VISUAL_ASSET_KIND_SCENE],
            "prop_refs": result_map[VISUAL_ASSET_KIND_PROP],
            "character_refs": result_map[VISUAL_ASSET_KIND_CHARACTER],
        }
        scene_ids = [str(item.get("item_id", "")).strip() for item in storyboard_template.get("scene_catalog", []) if isinstance(item, dict)]
        prop_ids = [str(item.get("item_id", "")).strip() for item in storyboard_template.get("prop_catalog", []) if isinstance(item, dict)]
        character_ids = [
            str(item.get("item_id", "")).strip()
            for item in storyboard_template.get("character_catalog", [])
            if isinstance(item, dict)
        ]
        validated_output = validate_role1_visual_catalog_output(
            data=payload,
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
        )
        self._llm_runtime.logger.info("模块B v2 role1 执行完成")
        return validated_output

    def _generate_asset_refs(
        self,
        *,
        asset_kind: str,
        style_payload: dict[str, Any],
        asset_items: list[Any],
    ) -> list[dict[str, Any]]:
        """
        功能说明：为某一类对象生成参考图提示词集合。
        参数说明：
        - asset_kind: scene/prop/character。
        - style_payload: 模板风格字段。
        - asset_items: 当前类别目录条目数组。
        返回值：
        - list[dict[str, Any]]: 单类对象 refs 列表。
        异常说明：LLM 或解析失败时抛出异常。
        边界条件：asset_items 为空时直接返回空数组。
        """
        normalized_items = [dict(item) for item in asset_items if isinstance(item, dict)]
        if not normalized_items:
            return []
        self._llm_runtime.logger.info(
            "模块B v2 role1 准备请求，asset_kind=%s，item_count=%s",
            asset_kind,
            len(normalized_items),
        )
        prompt_asset = render_prompt_asset(
            project_root=self._llm_runtime.project_root,
            prompt_asset=ROLE1_PROMPT_ASSET,
            user_variables=self._build_prompt_variables(
                asset_kind=asset_kind,
                style_payload=style_payload,
                asset_items=normalized_items,
            ),
        )
        response_text = self._llm_runtime.call_markdown(
            role_name=f"role1_visual_director:{asset_kind}",
            system_prompt=prompt_asset.system_prompt,
            user_prompt_markdown=prompt_asset.user_prompt_markdown,
            max_tokens_override=max(
                ROLE1_VISUAL_DIRECTOR_MIN_MAX_TOKENS,
                len(normalized_items) * 320,
            ),
            timeout_seconds_override=ROLE1_VISUAL_DIRECTOR_TIMEOUT_SECONDS,
        )
        parsed_output = parse_role1_visual_catalog_markdown(response_text)
        assets = parsed_output.get("assets", [])
        return assets if isinstance(assets, list) else []

    def _build_prompt_variables(
        self,
        *,
        asset_kind: str,
        style_payload: dict[str, Any],
        asset_items: list[dict[str, Any]],
    ) -> str:
        """
        功能说明：构建角色1的 Markdown few-shot 提示。
        参数说明：
        - asset_kind: scene/prop/character。
        - style_payload: 风格字段。
        - asset_items: 当前类别对象列表。
        返回值：
        - dict[str, str]: user prompt 模板变量映射。
        异常说明：无。
        边界条件：对象属性按单对象整合输出，避免结构化字段噪声。
        """
        kind_name_map = {
            VISUAL_ASSET_KIND_SCENE: "场景",
            VISUAL_ASSET_KIND_PROP: "道具",
            VISUAL_ASSET_KIND_CHARACTER: "角色",
        }
        item_blocks: list[str] = []
        for item in asset_items:
            item_id = str(item.get("item_id", "")).strip()
            name_zh = str(item.get("name_zh", "")).strip()
            description_zh = str(item.get("description_zh", "")).strip()
            item_blocks.append(
                render_heading_block(
                    level=2,
                    heading=item_id,
                    field_map={"名称": name_zh, "描述": description_zh or "无补充描述"},
                    order=["名称", "描述"],
                )
            )
        style_block = render_bullet_fields(
            {
                "色彩风格": str(style_payload.get("color_mode", "")).strip(),
                "画风": str(style_payload.get("render_style", "")).strip(),
            },
            order=["色彩风格", "画风"],
        )
        object_catalog = render_document(title="", blocks=item_blocks).strip()
        return {
            "asset_kind_name": kind_name_map.get(asset_kind, asset_kind),
            "style_block": style_block,
            "object_catalog": object_catalog,
        }
