"""
文件用途：实现模块B v2 的角色3“小段剧情编导”。
核心流程：按大段并发、段内串行地生成镜头描述、对象选择、构图与运镜/转场选择。
输入输出：输入增强音频特征、大段剧情与模板候选，输出 shot 级编排标准结构。
依赖说明：依赖标准库并发工具、v2 LLM runtime、Markdown 渲染器与 parser。
维护说明：本角色只能从规则层给出的 preset 候选中选择，不允许自由发明。
"""

# 标准库：用于并发执行。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 标准库：用于可调用类型提示。
from collections.abc import Callable
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：v2 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：歌词挂载上下文整理。
from music_video_pipeline.modules.module_b_v2.lyric_context import build_role3_big_segment_lyric_context
# 项目内模块：统一 Markdown 渲染。
from music_video_pipeline.modules.module_b_v2.markdown_io import (
    MarkdownFieldSchema,
    MarkdownLineSchema,
    render_compound_lines,
    render_schema_fields,
)
# 项目内模块：v2 数据结构。
from music_video_pipeline.modules.module_b_v2.models import SAFE_CLOSEUP_COMPOSITION_IDS, UnitHistoryItem
# 项目内模块：v2 parser。
from music_video_pipeline.modules.module_b_v2.parser import (
    ModuleBV2ParseError,
    parse_role3_segment_directing_markdown,
    validate_camera_plan,
    validate_role3_segment_directing_output,
    validate_transition_plan,
)
# 项目内模块：统一 prompt 模板加载。
from music_video_pipeline.modules.module_b_v2.prompt_templates import (
    ROLE3_PROMPT_ASSET,
    render_prompt_asset,
)


# 常量：角色3歌词使用规则。
ROLE3_LYRIC_USAGE_RULE = "歌词只作为情感、节奏、语气与叙事推进的参考，不作为具体视觉意象、场景道具或角色造型的直接来源。"
# 常量：角色3当前镜头字段 schema。
ROLE3_SHOT_CONTEXT_SCHEMA = [
    MarkdownFieldSchema("shot_id", "shot_id", ""),
    MarkdownFieldSchema("big_segment_id", "big_segment_id", ""),
    MarkdownFieldSchema("segment_id", "segment.segment_id", ""),
    MarkdownFieldSchema("label", "segment.label", ""),
    MarkdownFieldSchema("role", "segment.role", "unknown", lambda value: str(value).strip() or "unknown"),
    MarkdownFieldSchema("start_time", "segment.start_time", 0.0),
    MarkdownFieldSchema("duration_seconds", "segment.duration_seconds", 0.0),
    MarkdownFieldSchema("segment_index_in_big_segment", "segment_index_in_big_segment", 0),
    MarkdownFieldSchema("segment_count_in_big_segment", "segment_count_in_big_segment", 0),
]
# 常量：角色3大段剧情字段 schema。
ROLE3_BIG_SEGMENT_STORY_SCHEMA = [
    MarkdownFieldSchema("title_zh", "big_segment_story.title_zh", ""),
    MarkdownFieldSchema("story_outline_zh", "big_segment_story.story_outline_zh", ""),
    MarkdownFieldSchema("selected_scene_ids", "big_segment_story.selected_scene_ids", []),
    MarkdownFieldSchema("selected_character_ids", "big_segment_story.selected_character_ids", []),
    MarkdownFieldSchema("selected_prop_ids", "big_segment_story.selected_prop_ids", []),
]
# 常量：角色3歌词字段 schema。
ROLE3_LYRIC_CONTEXT_SCHEMA = [
    MarkdownFieldSchema("lyric_usage_rule", "lyric_usage_rule", ""),
    MarkdownFieldSchema("big_segment_lyric_excerpt", "big_segment_lyric_excerpt", "无", lambda value: str(value).strip() or "无"),
    MarkdownFieldSchema("current_segment_lyrics", "current_segment_lyrics", []),
]
# 常量：角色3音频语义字段 schema。
ROLE3_AUDIO_CONTEXT_SCHEMA = [
    MarkdownFieldSchema("energy_level", "audio_profile.energy_level", ""),
    MarkdownFieldSchema("trend", "audio_profile.trend", ""),
    MarkdownFieldSchema("tension_band", "audio_profile.tension_band", ""),
    MarkdownFieldSchema("tension_delta", "audio_profile.tension_delta", ""),
    MarkdownFieldSchema("is_local_peak", "audio_profile.is_local_peak", False),
    MarkdownFieldSchema("position_in_big_segment", "audio_profile.position_in_big_segment", ""),
]
# 常量：角色3选择提示字段 schema。
ROLE3_SELECTION_HINT_SCHEMA = [
    MarkdownFieldSchema("prefer_scene_ids", "selection_hint.prefer_scene_ids", []),
    MarkdownFieldSchema("safe_closeup_composition_ids", "selection_hint.safe_closeup_composition_ids", []),
]
# 常量：角色3历史镜头摘要行 schema。
ROLE3_HISTORY_LINE_SCHEMA = MarkdownLineSchema(
    id_path="shot_id",
    detail_schema=[
        MarkdownFieldSchema("selected_scene_id", "selected_scene_id", "none"),
        MarkdownFieldSchema("composition_id", "composition_id", "none"),
        MarkdownFieldSchema("camera_plan_preset_id", "camera_plan_preset_id", "none"),
        MarkdownFieldSchema("scene_desc_zh", "scene_desc_zh", "none"),
    ],
    detail_separator=" / ",
    detail_with_key=False,
)
# 常量：角色3构图库行 schema。
ROLE3_COMPOSITION_LINE_SCHEMA = MarkdownLineSchema(
    id_path="composition_id",
    detail_schema=[
        MarkdownFieldSchema("name_zh", "name_zh", "none"),
        MarkdownFieldSchema("description_zh", "description_zh", "none"),
    ],
    detail_separator=" | ",
    detail_with_key=False,
)
# 常量：角色3运镜候选行 schema。
ROLE3_CAMERA_PLAN_LINE_SCHEMA = MarkdownLineSchema(
    id_path="preset_id",
    detail_schema=[
        MarkdownFieldSchema("mode", "mode", "none"),
        MarkdownFieldSchema("direction", "direction", "none"),
        MarkdownFieldSchema("strength", "strength", "none"),
        MarkdownFieldSchema("easing", "easing", "none"),
    ],
    detail_separator=", ",
    detail_with_key=True,
)
# 常量：角色3转场候选行 schema。
ROLE3_TRANSITION_PLAN_LINE_SCHEMA = MarkdownLineSchema(
    id_path="preset_id",
    detail_schema=[
        MarkdownFieldSchema("kind", "kind", "none"),
        MarkdownFieldSchema("duration_ms", "duration_ms", 0),
        MarkdownFieldSchema("easing", "easing", "none"),
    ],
    detail_separator=", ",
    detail_with_key=True,
)


class Role3SegmentDirector:
    """
    功能说明：执行角色3小段镜头编排。
    参数说明：
    - llm_runtime: 通用 LLM 运行时。
    返回值：不适用。
    异常说明：具体异常由 generate 抛出。
    边界条件：大段之间允许并发，大段内部严格按时间串行。
    """

    def __init__(self, llm_runtime: ModuleBV2LlmRuntime) -> None:
        self._llm_runtime = llm_runtime

    def _extract_item_ids(self, items: list[dict[str, Any]], *, field_name: str) -> list[str]:
        """
        功能说明：从模板目录中抽取指定字段的合法 ID 列表。
        参数说明：
        - items: 目录条目列表。
        - field_name: ID 字段名。
        返回值：
        - list[str]: 清洗后的 ID 列表。
        异常说明：无。
        边界条件：非字典条目与空 ID 会被过滤。
        """
        return [
            str(item.get(field_name, "")).strip()
            for item in items
            if isinstance(item, dict) and str(item.get(field_name, "")).strip()
        ]

    def _group_segments_by_big_segment(self, segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """
        功能说明：按 big_segment_id 对小段分组。
        参数说明：
        - segments: 模块A小段列表。
        返回值：
        - dict[str, list[dict[str, Any]]]: 分组结果。
        异常说明：无。
        边界条件：缺失 big_segment_id 时会归入空字符串键。
        """
        grouped_segments: dict[str, list[dict[str, Any]]] = {}
        for segment in segments:
            grouped_segments.setdefault(str(segment.get("big_segment_id", "")).strip(), []).append(segment)
        return grouped_segments

    def _compact_big_story(self, big_story: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：压缩角色2返回的大段剧情骨架，只保留镜头编排所需字段。
        参数说明：
        - big_story: 角色2原始单段输出。
        返回值：
        - dict[str, Any]: 轻量大段剧情对象。
        异常说明：无。
        边界条件：缺失字段时回退为空字符串或空列表。
        """
        return {
            "big_segment_id": str(big_story.get("big_segment_id", "")).strip(),
            "title_zh": str(big_story.get("title_zh", "")).strip(),
            "story_outline_zh": str(big_story.get("story_outline_zh", "")).strip(),
            "selected_scene_ids": [
                str(item).strip() for item in big_story.get("selected_scene_ids", []) if str(item).strip()
            ],
            "selected_character_ids": [
                str(item).strip() for item in big_story.get("selected_character_ids", []) if str(item).strip()
            ],
            "selected_prop_ids": [
                str(item).strip() for item in big_story.get("selected_prop_ids", []) if str(item).strip()
            ],
        }

    def _compact_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：压缩当前小段对象，避免把模块A原始字段整包透传给 LLM。
        参数说明：
        - segment: 模块A单段对象。
        返回值：
        - dict[str, Any]: 轻量小段对象。
        异常说明：无。
        边界条件：duration_seconds 允许为 0。
        """
        start_time = float(segment.get("start_time", 0.0))
        end_time = float(segment.get("end_time", start_time))
        return {
            "segment_id": str(segment.get("segment_id", "")).strip(),
            "label": str(segment.get("label", "")).strip(),
            "role": str(segment.get("role", "")).strip(),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": max(0.0, end_time - start_time),
        }

    def _compact_composition_catalog(self, composition_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        功能说明：压缩构图库，仅保留角色3选型所需字段。
        参数说明：
        - composition_catalog: 模板构图库。
        返回值：
        - list[dict[str, Any]]: 轻量构图条目列表。
        异常说明：无。
        边界条件：无效条目会被过滤。
        """
        compact_items: list[dict[str, Any]] = []
        for item in composition_catalog:
            if not isinstance(item, dict):
                continue
            composition_id = str(item.get("composition_id", "")).strip()
            name_zh = str(item.get("name_zh", "")).strip()
            if not composition_id or not name_zh:
                continue
            compact_items.append(
                {
                    "composition_id": composition_id,
                    "name_zh": name_zh,
                    "description_zh": str(item.get("description_zh", "")).strip(),
                    "prompt_tags_en": [
                        str(tag).strip() for tag in item.get("prompt_tags_en", []) if str(tag).strip()
                    ],
                }
            )
        return compact_items

    def _compact_audio_profile(self, audio_payload: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：压缩规则增强音频特征，避免与候选 plan 字段重复透传。
        参数说明：
        - audio_payload: 原始增强音频对象。
        返回值：
        - dict[str, Any]: 轻量音频语义对象。
        异常说明：无。
        边界条件：布尔与文本字段缺失时做安全回退。
        """
        return {
            "energy_level": str(audio_payload.get("energy_level", "")).strip(),
            "trend": str(audio_payload.get("trend", "")).strip(),
            "tension_band": str(audio_payload.get("tension_band", "")).strip(),
            "tension_delta": str(audio_payload.get("tension_delta", "")).strip(),
            "is_local_peak": bool(audio_payload.get("is_local_peak", False)),
            "position_in_big_segment": str(audio_payload.get("position_in_big_segment", "")).strip(),
        }

    def _build_shot_id(self, segment: dict[str, Any]) -> str:
        """
        功能说明：根据全局索引构建标准 shot_id。
        参数说明：
        - segment: 当前小段对象。
        返回值：
        - str: 标准 shot_id。
        异常说明：无。
        边界条件：缺失 `_global_index` 时回退到 `global_index`，再回退到 0。
        """
        global_index = int(segment.get("_global_index", segment.get("global_index", 0)))
        return f"shot_{global_index + 1:03d}"

    def _resolve_preferred_scene_ids(
        self,
        *,
        compact_big_story: dict[str, Any],
        segment: dict[str, Any],
    ) -> list[str]:
        """
        功能说明：解析当前镜头优先场景候选。
        参数说明：
        - compact_big_story: 压缩后的大段剧情对象。
        - segment: 当前小段对象。
        返回值：
        - list[str]: 优先场景 ID 列表。
        异常说明：无。
        边界条件：当大段剧情未指定 scene 时，回退到当前 segment.label。
        """
        preferred_scene_ids = [
            str(item).strip() for item in compact_big_story.get("selected_scene_ids", []) if str(item).strip()
        ]
        if preferred_scene_ids:
            return preferred_scene_ids
        fallback_label = str(segment.get("label", "")).strip()
        return [fallback_label] if fallback_label else []

    def _build_history_item(self, *, shot_id: str, response: dict[str, Any]) -> UnitHistoryItem:
        """
        功能说明：从角色3单镜头结果构建历史摘要项。
        参数说明：
        - shot_id: 当前镜头 ID。
        - response: 角色3返回结果。
        返回值：
        - UnitHistoryItem: 用于后续镜头 continuity 的轻量历史项。
        异常说明：无。
        边界条件：字段缺失时回退为空字符串。
        """
        return {
            "shot_id": shot_id,
            "scene_desc_zh": str(response.get("scene_desc_zh", "")).strip(),
            "selected_scene_id": str(response.get("selected_scene_id", "")).strip(),
            "composition_id": str(response.get("composition_id", "")).strip(),
            "camera_plan_preset_id": str((response.get("camera_plan") or {}).get("preset_id", "")).strip(),
        }

    def _resolve_segment_lyrics(self, *, lyric_timeline: Any, segment_id: str) -> list[str]:
        """
        功能说明：从当前大段歌词时间线中取出当前小段挂载的歌词。
        参数说明：
        - lyric_timeline: 当前大段的歌词挂载数组。
        - segment_id: 当前 segment_id。
        返回值：
        - list[str]: 当前小段歌词行数组。
        异常说明：无。
        边界条件：未命中时返回空数组。
        """
        if not isinstance(lyric_timeline, list):
            return []
        for item in lyric_timeline:
            if not isinstance(item, dict):
                continue
            if str(item.get("segment_id", "")).strip() != segment_id:
                continue
            return [str(line).strip() for line in item.get("lyric_lines", []) if str(line).strip()]
        return []

    def _build_role3_prompt_payload(
        self,
        *,
        shot_id: str,
        big_segment_id: str,
        segment: dict[str, Any],
        segment_index: int,
        segment_count: int,
        compact_big_story: dict[str, Any],
        composition_catalog: list[dict[str, Any]],
        audio_payload: dict[str, Any],
        big_segment_lyric_context: dict[str, Any],
        history_items: list[UnitHistoryItem],
    ) -> dict[str, Any]:
        """
        功能说明：构建单镜头角色3 prompt 载荷。
        参数说明：
        - shot_id: 当前镜头 ID。
        - big_segment_id: 当前大段 ID。
        - segment: 当前小段对象。
        - segment_index: 当前小段在大段内的序号（0基）。
        - segment_count: 当前大段下小段总数。
        - compact_big_story: 压缩后的大段剧情对象。
        - composition_catalog: 压缩后的构图库。
        - audio_payload: 当前小段规则增强音频对象。
        - big_segment_lyric_context: 当前大段歌词上下文。
        - history_items: 已完成镜头历史摘要。
        返回值：
        - dict[str, Any]: 送入 prompt 模板渲染的轻量载荷。
        异常说明：无。
        边界条件：history 仅保留最近两条。
        """
        segment_id = str(segment.get("segment_id", "")).strip()
        return {
            "shot_id": shot_id,
            "big_segment_id": big_segment_id,
            "segment": self._compact_segment(segment),
            "segment_index_in_big_segment": segment_index + 1,
            "segment_count_in_big_segment": segment_count,
            "big_segment_story": compact_big_story,
            "lyric_usage_rule": ROLE3_LYRIC_USAGE_RULE,
            "current_segment_id": segment_id,
            "lyric_timeline": big_segment_lyric_context.get("segment_lyrics", []),
            "big_segment_lyric_excerpt": str(big_segment_lyric_context.get("lyric_excerpt", "")).strip(),
            "audio_profile": self._compact_audio_profile(audio_payload),
            "composition_catalog": composition_catalog,
            "camera_plan_candidates": audio_payload.get("camera_plan_candidates", []),
            "transition_plan_candidates": audio_payload.get("transition_plan_candidates", []),
            "history": history_items[-2:],
            "selection_hint": {
                "prefer_scene_ids": self._resolve_preferred_scene_ids(
                    compact_big_story=compact_big_story,
                    segment=segment,
                ),
                "safe_closeup_composition_ids": sorted(SAFE_CLOSEUP_COMPOSITION_IDS),
            },
        }

    def generate(
        self,
        *,
        module_a_output: dict[str, Any],
        storyboard_template: dict[str, Any],
        role2_output: dict[str, Any],
        segment_audio_features: dict[str, dict[str, Any]],
        target_shot_ids: set[str] | None = None,
        existing_shots: dict[str, dict[str, Any]] | None = None,
        on_shot_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        功能说明：按大段并发生成全部小段镜头编排结果。
        参数说明：
        - module_a_output: 模块A输出。
        - storyboard_template: 已编译模板。
        - role2_output: 角色2输出。
        - segment_audio_features: 规则增强后的音频特征映射。
        - target_shot_ids: 可选，仅要求这些 shot 在本轮结果中覆盖。
        - existing_shots: 可选，已完成 shot 的缓存结果，用于断点续跑与连续性恢复。
        - on_shot_completed: 可选，单个 shot 完成后立即回调。
        返回值：
        - dict[str, Any]: 角色3输出。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：每个 segment 对应唯一 shot_id；同一大段内若前序 shot 缓存缺失，会自动补生成以恢复 history。
        """
        segments = [dict(item) for item in module_a_output.get("segments", []) if isinstance(item, dict)]
        grouped_segments = self._group_segments_by_big_segment(segments)
        big_segment_lyric_context_map = build_role3_big_segment_lyric_context(module_a_output=module_a_output)
        big_story_map = {
            str(item.get("big_segment_id", "")).strip(): dict(item)
            for item in role2_output.get("big_segments", [])
            if isinstance(item, dict)
        }
        composition_catalog = [
            dict(item) for item in storyboard_template.get("composition_catalog", []) if isinstance(item, dict)
        ]
        compact_composition_catalog = self._compact_composition_catalog(composition_catalog)
        scene_ids = self._extract_item_ids(storyboard_template.get("scene_catalog", []), field_name="item_id")
        prop_ids = self._extract_item_ids(storyboard_template.get("prop_catalog", []), field_name="item_id")
        character_ids = self._extract_item_ids(storyboard_template.get("character_catalog", []), field_name="item_id")
        composition_ids = self._extract_item_ids(
            storyboard_template.get("composition_catalog", []),
            field_name="composition_id",
        )
        all_shot_ids = [self._build_shot_id(segment) for segment in segments]
        required_shot_ids = set(target_shot_ids or all_shot_ids)
        existing_shot_map = {
            str(shot_id).strip(): dict(shot_payload)
            for shot_id, shot_payload in (existing_shots or {}).items()
            if str(shot_id).strip() and isinstance(shot_payload, dict)
        }
        self._llm_runtime.logger.info(
            "模块B v2 role3 开始执行，big_segment_count=%s，shot_count=%s，required_shot_count=%s，cached_shot_count=%s",
            len(grouped_segments),
            len(segments),
            len(required_shot_ids),
            len(existing_shot_map),
        )

        shots: list[dict[str, Any]] = []
        failed_big_segments: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, len(grouped_segments) or 1)) as executor:
            future_map = {
                executor.submit(
                    self._generate_big_segment_shots,
                    big_segment_id=big_segment_id,
                    segments=sorted(items, key=lambda item: float(item.get("start_time", 0.0))),
                    big_story=big_story_map.get(big_segment_id, {}),
                    composition_catalog=compact_composition_catalog,
                    segment_audio_features=segment_audio_features,
                    big_segment_lyric_context=big_segment_lyric_context_map.get(big_segment_id, {}),
                    required_shot_ids=required_shot_ids,
                    existing_shots=existing_shot_map,
                    on_shot_completed=on_shot_completed,
                ): big_segment_id
                for big_segment_id, items in grouped_segments.items()
            }
            for future in as_completed(future_map):
                big_segment_id = future_map[future]
                try:
                    shots.extend(future.result())
                except Exception as error:  # noqa: BLE001
                    failed_big_segments.append(big_segment_id)
                    self._llm_runtime.logger.error(
                        "模块B v2 role3 大段执行失败，big_segment_id=%s，错误=%s",
                        big_segment_id,
                        error,
                    )

        combined_shot_map = {
            shot_id: dict(shot_payload)
            for shot_id, shot_payload in existing_shot_map.items()
            if shot_id in required_shot_ids
        }
        for shot_item in shots:
            if not isinstance(shot_item, dict):
                continue
            shot_id = str(shot_item.get("shot_id", "")).strip()
            if shot_id:
                combined_shot_map[shot_id] = dict(shot_item)

        ordered_shot_ids = [shot_id for shot_id in all_shot_ids if shot_id in required_shot_ids]
        validated_output = validate_role3_segment_directing_output(
            data={"shots": [combined_shot_map[shot_id] for shot_id in ordered_shot_ids if shot_id in combined_shot_map]},
            shot_ids=ordered_shot_ids,
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
            composition_ids=composition_ids,
        )
        if failed_big_segments:
            raise RuntimeError(f"模块B v2 role3 存在失败大段，failed_big_segments={sorted(failed_big_segments)}")
        self._llm_runtime.logger.info("模块B v2 role3 执行完成")
        return validated_output

    def _generate_big_segment_shots(
        self,
        *,
        big_segment_id: str,
        segments: list[dict[str, Any]],
        big_story: dict[str, Any],
        composition_catalog: list[dict[str, Any]],
        segment_audio_features: dict[str, dict[str, Any]],
        big_segment_lyric_context: dict[str, Any],
        required_shot_ids: set[str],
        existing_shots: dict[str, dict[str, Any]],
        on_shot_completed: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        """
        功能说明：串行为单个大段生成全部小段镜头编排。
        参数说明：
        - big_segment_id: 当前大段ID。
        - segments: 当前大段下的小段列表。
        - big_story: 当前大段剧情骨架。
        - composition_catalog: 构图库。
        - segment_audio_features: 全量增强音频特征映射。
        - required_shot_ids: 本轮必须覆盖的 shot_id 集合。
        - existing_shots: 已完成 shot 的缓存结果。
        - on_shot_completed: 可选，单个 shot 完成后立即回调。
        返回值：
        - list[dict[str, Any]]: 当前大段下全部 shot 编排结果。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：通过 history 摘要维持段内连续性；若目标 shot 前存在缓存缺口，会顺带补生成这些前序 shot。
        """
        ordered_shot_ids = [self._build_shot_id(segment) for segment in segments]
        if not any(shot_id in required_shot_ids for shot_id in ordered_shot_ids):
            return []
        history_items: list[UnitHistoryItem] = []
        results: list[dict[str, Any]] = []
        compact_big_story = self._compact_big_story(big_story)
        lyric_excerpt = str(big_segment_lyric_context.get("lyric_excerpt", "")).strip()
        lyric_timeline = big_segment_lyric_context.get("segment_lyrics", [])
        self._llm_runtime.logger.info(
            "模块B v2 role3 开始处理大段，big_segment_id=%s，shot_count=%s",
            big_segment_id,
            len(segments),
        )
        for segment_index, segment in enumerate(segments):
            shot_id = self._build_shot_id(segment)
            if shot_id in existing_shots:
                cached_response = dict(existing_shots[shot_id])
                history_items.append(self._build_history_item(shot_id=shot_id, response=cached_response))
                if shot_id in required_shot_ids:
                    results.append(cached_response)
                self._llm_runtime.logger.info(
                    "模块B v2 role3 复用缓存 shot，big_segment_id=%s，shot_id=%s，progress=%s/%s",
                    big_segment_id,
                    shot_id,
                    segment_index + 1,
                    len(segments),
                )
                continue
            segment_id = str(segment.get("segment_id", "")).strip()
            audio_payload = dict(segment_audio_features.get(segment_id, {}))
            payload = self._build_role3_prompt_payload(
                shot_id=shot_id,
                big_segment_id=big_segment_id,
                segment=segment,
                segment_index=segment_index,
                segment_count=len(segments),
                compact_big_story=compact_big_story,
                composition_catalog=composition_catalog,
                audio_payload=audio_payload,
                big_segment_lyric_context={
                    "lyric_excerpt": lyric_excerpt,
                    "segment_lyrics": lyric_timeline,
                },
                history_items=history_items,
            )
            prompt_asset = render_prompt_asset(
                project_root=self._llm_runtime.project_root,
                prompt_asset=ROLE3_PROMPT_ASSET,
                user_variables=self._build_prompt_variables(payload),
            )
            response_text = self._llm_runtime.call_markdown(
                role_name=f"role3_segment_director:{shot_id}",
                system_prompt=prompt_asset.system_prompt,
                user_prompt_markdown=prompt_asset.user_prompt_markdown,
            )
            response = parse_role3_segment_directing_markdown(response_text)
            response["shot_id"] = shot_id
            self._assert_plan_selected_from_candidates(
                shot_id=shot_id,
                camera_plan=response.get("camera_plan", {}),
                transition_plan=response.get("transition_plan", {}),
                camera_plan_candidates=audio_payload.get("camera_plan_candidates", []),
                transition_plan_candidates=audio_payload.get("transition_plan_candidates", []),
            )
            if shot_id in required_shot_ids:
                results.append(response)
            self._llm_runtime.logger.info(
                "模块B v2 role3 完成 shot，big_segment_id=%s，shot_id=%s，progress=%s/%s",
                big_segment_id,
                shot_id,
                segment_index + 1,
                len(segments),
            )
            if on_shot_completed is not None:
                on_shot_completed(dict(response))
            history_items.append(self._build_history_item(shot_id=shot_id, response=response))
        self._llm_runtime.logger.info("模块B v2 role3 完成大段，big_segment_id=%s", big_segment_id)
        return results

    def _build_prompt_variables(self, payload: dict[str, Any]) -> dict[str, str]:
        """
        功能说明：构建角色3 user prompt 模板变量。
        参数说明：
        - payload: 当前 shot 的精简输入。
        返回值：
        - dict[str, str]: user prompt 模板变量映射。
        异常说明：无。
        边界条件：候选 preset 只以必要字段列出，避免重复结构化字段噪声。
        """
        current_segment_id = str(payload.get("current_segment_id", "")).strip()
        prompt_source = dict(payload)
        prompt_source["current_segment_lyrics"] = self._resolve_segment_lyrics(
            lyric_timeline=payload.get("lyric_timeline", []),
            segment_id=current_segment_id,
        )
        return {
            "shot_context": render_schema_fields(prompt_source, ROLE3_SHOT_CONTEXT_SCHEMA),
            "big_segment_story": render_schema_fields(prompt_source, ROLE3_BIG_SEGMENT_STORY_SCHEMA),
            "lyric_context": render_schema_fields(prompt_source, ROLE3_LYRIC_CONTEXT_SCHEMA),
            "audio_context": render_schema_fields(prompt_source, ROLE3_AUDIO_CONTEXT_SCHEMA),
            "composition_catalog": render_compound_lines(
                payload.get("composition_catalog", []),
                line_schema=ROLE3_COMPOSITION_LINE_SCHEMA,
            ),
            "camera_candidates": render_compound_lines(
                payload.get("camera_plan_candidates", []),
                line_schema=ROLE3_CAMERA_PLAN_LINE_SCHEMA,
            ),
            "transition_candidates": render_compound_lines(
                payload.get("transition_plan_candidates", []),
                line_schema=ROLE3_TRANSITION_PLAN_LINE_SCHEMA,
            ),
            "history_context": render_compound_lines(
                payload.get("history", []),
                line_schema=ROLE3_HISTORY_LINE_SCHEMA,
            ),
            "selection_hint": render_schema_fields(prompt_source, ROLE3_SELECTION_HINT_SCHEMA),
        }

    def _assert_plan_selected_from_candidates(
        self,
        *,
        shot_id: str,
        camera_plan: Any,
        transition_plan: Any,
        camera_plan_candidates: Any,
        transition_plan_candidates: Any,
    ) -> None:
        """
        功能说明：校验角色3返回的运镜/转场是否严格来自规则候选集。
        参数说明：
        - shot_id: 当前镜头 ID。
        - camera_plan/transition_plan: LLM 返回对象。
        - camera_plan_candidates/transition_plan_candidates: 规则候选数组。
        返回值：无。
        异常说明：
        - ModuleBV2ParseError: 检测到越权选择时抛出。
        边界条件：比较逻辑按完整标准化对象匹配，而非仅比 preset_id。
        """
        normalized_camera_plan = validate_camera_plan(camera_plan if isinstance(camera_plan, dict) else {})
        normalized_transition_plan = validate_transition_plan(
            transition_plan if isinstance(transition_plan, dict) else {}
        )
        normalized_camera_candidates = [
            validate_camera_plan(item) for item in camera_plan_candidates if isinstance(item, dict)
        ]
        normalized_transition_candidates = [
            validate_transition_plan(item) for item in transition_plan_candidates if isinstance(item, dict)
        ]
        if normalized_camera_candidates and normalized_camera_plan not in normalized_camera_candidates:
            raise ModuleBV2ParseError(
                f"role3[{shot_id}] 返回的 camera_plan 不在候选集中：{normalized_camera_plan}"
            )
        if normalized_transition_candidates and normalized_transition_plan not in normalized_transition_candidates:
            raise ModuleBV2ParseError(
                f"role3[{shot_id}] 返回的 transition_plan 不在候选集中：{normalized_transition_plan}"
            )
