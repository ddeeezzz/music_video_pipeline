"""
文件用途：实现模块B v2 的角色2“大段剧情编导”。
核心流程：单次请求 LLM，根据 big_segments 与模板元素生成大段剧情骨架。
输入输出：输入大段时间结构与模板目录，输出大段剧情标准结构。
依赖说明：依赖 v2 LLM runtime、Markdown 渲染器与 parser。
维护说明：本角色偏名词/动词驱动，不做细碎镜头级描述。
"""

# 标准库：用于类型提示。
from typing import Any

# 项目内模块：v2 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：歌词挂载上下文整理。
from music_video_pipeline.modules.module_b_v2.lyric_context import build_big_segment_lyric_context
# 项目内模块：统一 Markdown 渲染。
from music_video_pipeline.modules.module_b_v2.markdown_io import (
    MarkdownFieldSchema,
    MarkdownSectionSchema,
    render_catalog_lines,
    render_schema_fields,
    render_section_from_schema,
)
# 项目内模块：v2 parser。
from music_video_pipeline.modules.module_b_v2.parser import (
    parse_role2_big_segment_story_markdown,
    validate_role2_big_segment_story_output,
)
# 项目内模块：统一 prompt 模板加载。
from music_video_pipeline.modules.module_b_v2.prompt_templates import (
    ROLE2_PROMPT_ASSET,
    render_prompt_asset,
)


# 常量：角色2大段剧情输出默认 token 下限。
ROLE2_BIG_SEGMENT_MIN_MAX_TOKENS = 2200
# 常量：角色2大段剧情请求超时（秒）。
ROLE2_BIG_SEGMENT_TIMEOUT_SECONDS = 180.0
# 常量：角色2全局上下文字段 schema。
ROLE2_GLOBAL_CONTEXT_SCHEMA = [
    MarkdownFieldSchema("色彩风格", "style.color_mode", ""),
    MarkdownFieldSchema("画风", "style.render_style", ""),
    MarkdownFieldSchema("故事前提", "story.premise_zh", ""),
    MarkdownFieldSchema("歌词使用规则", "lyric_usage_rule", ""),
]
# 常量：角色2大段输入字段 schema。
ROLE2_BIG_SEGMENT_SECTION_SCHEMA = [
    MarkdownFieldSchema("label", "label", ""),
    MarkdownFieldSchema("start_time", "start_time", 0.0),
    MarkdownFieldSchema("duration_seconds", "duration_seconds", 0.0),
    MarkdownFieldSchema("segment_count", "segment_count", 0),
    MarkdownFieldSchema("lyric_line_count", "lyric_line_count", 0),
    MarkdownFieldSchema("lyric_brief", "lyric_brief", "无"),
]


class Role2BigSegmentDirector:
    """
    功能说明：执行角色2大段剧情骨架生成。
    参数说明：
    - llm_runtime: 通用 LLM 运行时。
    返回值：不适用。
    异常说明：具体异常由 generate 抛出。
    边界条件：big_segments 为空时返回空列表。
    """

    def __init__(self, llm_runtime: ModuleBV2LlmRuntime) -> None:
        self._llm_runtime = llm_runtime

    def _compact_catalog_items(self, items: list[Any]) -> list[dict[str, str]]:
        """
        功能说明：压缩目录输入，只保留角色2选型真正需要的 ID 与名称。
        参数说明：
        - items: 模板目录原始条目列表。
        返回值：
        - list[dict[str, str]]: 轻量目录条目列表。
        异常说明：无。
        边界条件：无效条目会被过滤。
        """
        compact_items: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("item_id", "")).strip()
            name_zh = str(item.get("name_zh", "")).strip()
            if not item_id or not name_zh:
                continue
            compact_items.append(
                {
                    "item_id": item_id,
                    "name_zh": name_zh,
                }
            )
        return compact_items

    def _build_payload(
        self,
        *,
        storyboard_template: dict[str, Any],
        big_segments: list[dict[str, Any]],
        segment_counts: dict[str, int],
        big_segment_lyric_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        功能说明：构建角色2单次调用所需的中间载荷。
        参数说明：
        - storyboard_template: 已编译模板。
        - big_segments: 当前批次的大段列表。
        - segment_counts: 各大段下属小段数量映射。
        - big_segment_lyric_context: 当前批次的大段歌词挂载摘要。
        返回值：
        - dict[str, Any]: 可直接送入 Markdown prompt 渲染层的载荷。
        异常说明：无。
        边界条件：仅保留角色2真实需要的轻量字段。
        """
        return {
            "style": storyboard_template.get("style", {}),
            "story": storyboard_template.get("story", {}),
            "scene_catalog": self._compact_catalog_items(storyboard_template.get("scene_catalog", [])),
            "prop_catalog": self._compact_catalog_items(storyboard_template.get("prop_catalog", [])),
            "character_catalog": self._compact_catalog_items(storyboard_template.get("character_catalog", [])),
            "lyric_usage_rule": "歌词只作为情感、节奏、语气与叙事推进的参考，不作为具体视觉意象、场景道具或角色造型的直接来源。",
            "big_segment_lyric_briefs": big_segment_lyric_context,
            "big_segments": [
                {
                    "big_segment_id": str(item.get("segment_id", "")).strip(),
                    "label": str(item.get("label", "")).strip(),
                    "start_time": float(item.get("start_time", 0.0)),
                    "duration_seconds": max(
                        0.0,
                        float(item.get("end_time", item.get("start_time", 0.0))) - float(item.get("start_time", 0.0)),
                    ),
                    "segment_count": int(segment_counts.get(str(item.get("segment_id", "")).strip(), 0)),
                }
                for item in big_segments
            ],
        }

    def _build_prompt_variables(self, payload: dict[str, Any]) -> dict[str, str]:
        """
        功能说明：构建角色2 user prompt 模板变量。
        参数说明：
        - payload: 已裁剪的角色2输入载荷。
        返回值：
        - dict[str, str]: user prompt 模板变量映射。
        异常说明：无。
        边界条件：只保留大段级必要信息，不再分批。
        """
        lyric_map = {
            str(item.get("big_segment_id", "")).strip(): dict(item)
            for item in payload.get("big_segment_lyric_briefs", [])
            if isinstance(item, dict)
        }
        big_segment_blocks: list[str] = []
        for big_segment in payload.get("big_segments", []):
            if not isinstance(big_segment, dict):
                continue
            big_segment_id = str(big_segment.get("big_segment_id", "")).strip()
            lyric_item = lyric_map.get(big_segment_id, {})
            section_source = {
                **big_segment,
                "lyric_line_count": int(lyric_item.get("lyric_line_count", 0)),
                "lyric_brief": str(lyric_item.get("lyric_excerpt", "")).strip() or "无",
            }
            big_segment_blocks.append(
                render_section_from_schema(
                    MarkdownSectionSchema(
                        heading=big_segment_id,
                        field_schema=ROLE2_BIG_SEGMENT_SECTION_SCHEMA,
                        level=2,
                    ),
                    section_source,
                )
            )
        return {
            "global_context": render_schema_fields(payload, ROLE2_GLOBAL_CONTEXT_SCHEMA),
            "scene_catalog": render_catalog_lines(
                payload.get("scene_catalog", []),
                id_path="item_id",
                label_path="name_zh",
            ),
            "character_catalog": render_catalog_lines(
                payload.get("character_catalog", []),
                id_path="item_id",
                label_path="name_zh",
            ),
            "prop_catalog": render_catalog_lines(
                payload.get("prop_catalog", []),
                id_path="item_id",
                label_path="name_zh",
            ),
            "big_segment_catalog": "\n\n".join(big_segment_blocks),
        }

    def generate(self, module_a_output: dict[str, Any], storyboard_template: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：根据全部大段结构生成完整剧情骨架。
        参数说明：
        - module_a_output: 模块A输出。
        - storyboard_template: 已编译模板。
        返回值：
        - dict[str, Any]: 角色2输出。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：每个 big_segment 必须覆盖一次，不允许漏段。
        """
        big_segments = [dict(item) for item in module_a_output.get("big_segments", []) if isinstance(item, dict)]
        if not big_segments:
            return {"big_segments": []}
        self._llm_runtime.logger.info("模块B v2 role2 开始执行，big_segment_count=%s", len(big_segments))
        segment_counts: dict[str, int] = {}
        for segment in module_a_output.get("segments", []):
            if not isinstance(segment, dict):
                continue
            big_segment_id = str(segment.get("big_segment_id", "")).strip()
            if big_segment_id:
                segment_counts[big_segment_id] = int(segment_counts.get(big_segment_id, 0)) + 1
        full_big_segment_lyric_context = build_big_segment_lyric_context(module_a_output=module_a_output)
        lyric_context_by_big_segment_id = {
            str(item.get("big_segment_id", "")).strip(): dict(item)
            for item in full_big_segment_lyric_context
            if str(item.get("big_segment_id", "")).strip()
        }
        scene_ids = [
            str(item.get("item_id", "")).strip()
            for item in storyboard_template.get("scene_catalog", [])
            if isinstance(item, dict)
        ]
        prop_ids = [
            str(item.get("item_id", "")).strip()
            for item in storyboard_template.get("prop_catalog", [])
            if isinstance(item, dict)
        ]
        character_ids = [
            str(item.get("item_id", "")).strip()
            for item in storyboard_template.get("character_catalog", [])
            if isinstance(item, dict)
        ]
        payload = self._build_payload(
            storyboard_template=storyboard_template,
            big_segments=big_segments,
            segment_counts=segment_counts,
            big_segment_lyric_context=[
                lyric_context_by_big_segment_id[big_segment_id]
                for big_segment_id in [str(item.get("segment_id", "")).strip() for item in big_segments]
                if big_segment_id in lyric_context_by_big_segment_id
            ],
        )
        prompt_asset = render_prompt_asset(
            project_root=self._llm_runtime.project_root,
            prompt_asset=ROLE2_PROMPT_ASSET,
            user_variables=self._build_prompt_variables(payload),
        )
        response_text = self._llm_runtime.call_markdown(
            role_name="role2_big_segment_director",
            system_prompt=prompt_asset.system_prompt,
            user_prompt_markdown=prompt_asset.user_prompt_markdown,
            max_tokens_override=max(
                ROLE2_BIG_SEGMENT_MIN_MAX_TOKENS,
                len(big_segments) * 260,
            ),
            timeout_seconds_override=ROLE2_BIG_SEGMENT_TIMEOUT_SECONDS,
        )
        parsed_output = parse_role2_big_segment_story_markdown(response_text)
        validated_output = validate_role2_big_segment_story_output(
            data=parsed_output,
            big_segment_ids=[str(item.get("segment_id", "")).strip() for item in big_segments],
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
        )
        self._llm_runtime.logger.info("模块B v2 role2 执行完成")
        return validated_output
