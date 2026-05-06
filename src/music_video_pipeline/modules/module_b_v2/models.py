"""
文件用途：定义模块B v2 多角色编排所需的数据结构与常量。
核心流程：提供模板、角色输出、运镜与转场的标准化字段约束。
输入输出：输入为上层传入字典，输出为结构化 TypedDict 与常量。
依赖说明：依赖标准库 typing。
维护说明：字段契约变更时必须同步更新 parser、orchestrator 与测试。
"""

# 标准库：用于类型声明。
from typing import NotRequired, TypedDict


# 常量：默认编排模板文件路径（相对项目根目录）。
DEFAULT_STORYBOARD_TEMPLATE_FILE = "configs/storyboard_templates/storyboard_template.v1.md"

# 常量：角色1场景资产类别名。
VISUAL_ASSET_KIND_SCENE = "scene"
# 常量：角色1道具资产类别名。
VISUAL_ASSET_KIND_PROP = "prop"
# 常量：角色1角色资产类别名。
VISUAL_ASSET_KIND_CHARACTER = "character"

# 常量：近景安全构图集合。
SAFE_CLOSEUP_COMPOSITION_IDS = {
    "comp_sym_center",
    "comp_left_third_profile",
    "comp_right_third_profile",
    "comp_frame_within_frame",
    "comp_low_angle_presence",
}

# 常量：运镜 mode 合法值集合。
VALID_CAMERA_PLAN_MODES = {"none", "pan", "zoom"}
# 常量：运镜 direction 合法值集合。
VALID_CAMERA_PLAN_DIRECTIONS = {
    "center",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
}
# 常量：运镜强度合法值集合。
VALID_CAMERA_PLAN_STRENGTHS = {"none", "small", "medium"}
# 常量：运镜 easing 合法值集合。
VALID_EASING_VALUES = {"linear", "ease_in", "ease_out", "ease_in_out"}
# 常量：转场类型合法值集合。
VALID_TRANSITION_KINDS = {
    "none",
    "hard_cut",
    "crossfade",
    "fade_black",
    "fade_white",
    "wipe_left",
    "wipe_right",
}


class StoryboardStyle(TypedDict):
    """定义编排模板的风格字段。"""

    color_mode: str
    render_style: str


class StoryboardStory(TypedDict):
    """定义编排模板的故事字段。"""

    premise_zh: str


class StoryboardCatalogItem(TypedDict):
    """定义场景/道具/角色目录条目。"""

    item_id: str
    name_zh: str
    description_zh: str


class CompositionCatalogItem(TypedDict):
    """定义构图目录条目。"""

    composition_id: str
    name_zh: str
    description_zh: str
    prompt_tags_en: list[str]
    safe_for_closeup: bool
    safe_for_motion: bool


class CameraPlanPreset(TypedDict):
    """定义运镜 preset 条目。"""

    preset_id: str
    mode: str
    direction: str
    strength: str
    easing: str


class CameraMappingRule(TypedDict):
    """定义运镜映射规则。"""

    energy_level: str
    trend: str
    default_preset_id: str
    candidate_preset_ids: list[str]


class TransitionPreset(TypedDict):
    """定义转场 preset 条目。"""

    preset_id: str
    kind: str
    duration_ms: int
    easing: str


class TransitionMappingRule(TypedDict):
    """定义转场映射规则。"""

    current_energy_level: str
    next_energy_level: str
    default_preset_id: str
    candidate_preset_ids: list[str]


class StoryboardTemplate(TypedDict):
    """定义编排模板编译后的标准结构。"""

    template_id: str
    style: StoryboardStyle
    story: StoryboardStory
    scene_catalog: list[StoryboardCatalogItem]
    prop_catalog: list[StoryboardCatalogItem]
    character_catalog: list[StoryboardCatalogItem]
    composition_catalog: list[CompositionCatalogItem]
    camera_plan_presets: list[CameraPlanPreset]
    camera_mapping: list[CameraMappingRule]
    transition_presets: list[TransitionPreset]
    transition_mapping: list[TransitionMappingRule]


class VisualRefPrompt(TypedDict):
    """定义角色1返回的单张参考图提示词。"""

    ref_id: str
    pos_zh: str
    pos_en: str
    neg_zh: str
    neg_en: str


class VisualAssetRefItem(TypedDict):
    """定义角色1返回的单个对象提示词集合。"""

    item_id: str
    refs: list[VisualRefPrompt]


class Role1VisualCatalogOutput(TypedDict):
    """定义角色1汇总输出。"""

    scene_refs: list[VisualAssetRefItem]
    prop_refs: list[VisualAssetRefItem]
    character_refs: list[VisualAssetRefItem]


class BigSegmentStoryItem(TypedDict):
    """定义角色2输出的大段剧情条目。"""

    big_segment_id: str
    title_zh: str
    story_outline_zh: str
    selected_scene_ids: list[str]
    selected_character_ids: list[str]
    selected_prop_ids: list[str]


class Role2BigSegmentStoryOutput(TypedDict):
    """定义角色2汇总输出。"""

    big_segments: list[BigSegmentStoryItem]


class CameraPlan(TypedDict):
    """定义标准化运镜计划。"""

    preset_id: str
    mode: str
    direction: str
    strength: str
    easing: str


class TransitionPlan(TypedDict):
    """定义标准化转场计划。"""

    preset_id: str
    kind: str
    duration_ms: int
    easing: str


class SegmentAudioFeaturesV2(TypedDict):
    """定义角色3使用的增强音频特征。"""

    segment_id: str
    big_segment_id: str
    energy_level: str
    trend: str
    tension_band: str
    tension_delta: str
    is_local_peak: bool
    position_in_big_segment: str
    segment_rank_in_big_segment: int
    segment_count_in_big_segment: int
    camera_plan_candidates: list[CameraPlan]
    transition_plan_candidates: list[TransitionPlan]
    default_camera_plan: CameraPlan
    default_transition_plan: TransitionPlan
    beat_positions: NotRequired[list[float]]
    onset_positions: NotRequired[list[dict[str, float]]]
    onset_density: NotRequired[float]
    spectral_centroid_mean: NotRequired[float]


class Role3SegmentDirectingItem(TypedDict):
    """定义角色3输出的小段场景编排条目。"""

    shot_id: str
    scene_desc_zh: str
    selected_scene_id: str
    selected_character_ids: list[str]
    selected_prop_ids: list[str]
    composition_id: str
    camera_plan: CameraPlan
    transition_plan: TransitionPlan


class Role3SegmentDirectingOutput(TypedDict):
    """定义角色3汇总输出。"""

    shots: list[Role3SegmentDirectingItem]


class Role4PromptBlock(TypedDict):
    """定义角色4输出的提示词块。"""

    shot_id: str
    scene_desc: str
    keyframe_prompt_start_zh: str
    keyframe_prompt_start_en: str
    keyframe_negative_prompt_start_zh: str
    keyframe_negative_prompt_start_en: str
    keyframe_prompt_end_zh: str
    keyframe_prompt_end_en: str
    keyframe_negative_prompt_end_zh: str
    keyframe_negative_prompt_end_en: str
    video_prompt_zh: str
    video_prompt_en: str


class Role4PromptOutput(TypedDict):
    """定义角色4汇总输出。"""

    shots: list[Role4PromptBlock]


class UnitHistoryItem(TypedDict):
    """定义同大段前序镜头摘要。"""

    shot_id: str
    scene_desc_zh: str
    selected_scene_id: str
    composition_id: str
    camera_plan_preset_id: str


class FinalModuleBShotV2(TypedDict):
    """定义模块B v2 最终单镜头输出。"""

    shot_id: str
    start_time: float
    end_time: float
    scene_desc: str
    keyframe_prompt_start_zh: str
    keyframe_prompt_start_en: str
    keyframe_negative_prompt_start_zh: str
    keyframe_negative_prompt_start_en: str
    keyframe_prompt_end_zh: str
    keyframe_prompt_end_en: str
    keyframe_negative_prompt_end_zh: str
    keyframe_negative_prompt_end_en: str
    video_prompt_zh: str
    video_prompt_en: str
    camera_plan: CameraPlan
    transition_plan: TransitionPlan
    constraints: dict[str, bool]
    lyric_text: NotRequired[str]
    lyric_units: NotRequired[list[dict]]
    big_segment_id: NotRequired[str]
    big_segment_label: NotRequired[str]
    segment_label: NotRequired[str]
    segment_role: NotRequired[str]
    audio_role: NotRequired[str]
