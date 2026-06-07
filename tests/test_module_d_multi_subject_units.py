"""
文件用途：验证模块 D 对多主体模板的单元聚合规则。
核心流程：构造 GridTemplate 子主体帧清单 -> 归一化为单个模板视频单元 -> 校验格子素材顺序。
输入输出：输入 pytest 临时数据，输出断言结果。
依赖说明：依赖 pytest 与模块 D 单元模型工具。
维护说明：聚合规则服务 GridTemplate/ScrollTemplate，避免同一 segment 被渲染为多个视频片段。
"""

# 项目内模块：用于验证模块 D 多主体归一化。
from music_video_pipeline.modules.module_d.unit_models import normalize_frame_items_for_module_d


def test_normalize_frame_items_for_module_d_should_group_grid_subjects() -> None:
    """
    功能说明：验证 GridTemplate 的多个子主体会聚合为一个模块 D 视频单元。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：同一 segment 下三个子 shot 应按 subject_index 顺序进入 template_slots。
    """
    frame_items = [
        {
            "shot_id": "shot_0004_2",
            "segment_id": "seg_0004",
            "subject_index": 2,
            "remotion_id": "GridTemplate",
            "frame_path_start": "cat_start.png",
            "frame_path_end": "cat_end.png",
            "start_time": 7.65,
            "end_time": 11.29,
            "duration": 3.64,
        },
        {
            "shot_id": "shot_0004_1",
            "segment_id": "seg_0004",
            "subject_index": 1,
            "remotion_id": "GridTemplate",
            "frame_path_start": "girl_start.png",
            "frame_path_end": "girl_end.png",
            "start_time": 7.65,
            "end_time": 11.29,
            "duration": 3.64,
        },
        {
            "shot_id": "shot_0004_3",
            "segment_id": "seg_0004",
            "subject_index": 3,
            "remotion_id": "GridTemplate",
            "frame_path_start": "corridor_start.png",
            "frame_path_end": "corridor_end.png",
            "start_time": 7.65,
            "end_time": 11.29,
            "duration": 3.64,
        },
    ]

    normalized_items = normalize_frame_items_for_module_d(frame_items)

    assert len(normalized_items) == 1
    assert normalized_items[0]["shot_id"] == "seg_0004"
    assert normalized_items[0]["segment_id"] == "seg_0004"
    assert normalized_items[0]["source_shot_ids"] == ["shot_0004_1", "shot_0004_2", "shot_0004_3"]
    assert [slot["frame_path_start"] for slot in normalized_items[0]["template_slots"]] == [
        "girl_start.png",
        "cat_start.png",
        "corridor_start.png",
    ]
