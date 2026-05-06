"""
文件用途：验证模块 D ComfyUI ToonCrafter 渲染器的关键辅助逻辑。
核心流程：直接调用重采样与索引映射辅助函数，断言帧序列行为稳定可预测。
输入输出：输入临时图片序列，输出目标帧序列断言。
依赖说明：依赖 pytest 与模块 D ComfyUI 渲染器实现。
维护说明：本文件只覆盖纯 ToonCrafter 路径的稳定辅助行为。
"""

# 标准库：用于路径处理。
from pathlib import Path

# 项目内模块：模块 D ComfyUI 渲染器辅助函数。
from music_video_pipeline.modules.module_d.backends import comfyui_renderer



def test_map_resample_index_should_align_head_and_tail() -> None:
    """
    功能说明：验证重采样索引映射会稳定对齐首尾帧。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：覆盖放大采样场景。
    """
    mapped_indexes = [
        comfyui_renderer._map_resample_index(
            target_index=target_index,
            target_count=5,
            source_count=3,
        )
        for target_index in range(5)
    ]
    assert mapped_indexes == [0, 0, 1, 2, 2]



def test_resample_frame_sequence_should_expand_to_target_count(tmp_path: Path) -> None:
    """
    功能说明：验证 ToonCrafter 原生帧序列可被确定性拉伸到目标帧数。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证文件数量与首尾内容映射。
    """
    source_files: list[Path] = []
    for index in range(3):
        source_file = tmp_path / f"source_{index + 1:04d}.png"
        source_file.write_bytes(f"frame-{index}".encode("utf-8"))
        source_files.append(source_file)

    sequence_dir = tmp_path / "resampled"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    resampled_files = comfyui_renderer._resample_frame_sequence(
        source_files=source_files,
        target_count=5,
        sequence_dir=sequence_dir,
    )

    assert len(resampled_files) == 5
    assert resampled_files[0].read_bytes() == b"frame-0"
    assert resampled_files[-1].read_bytes() == b"frame-2"
