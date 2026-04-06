"""
文件用途：验证模块A导出面的最小兼容约束与白名单结构。
核心流程：读取 module_a 命名空间，校验 public_api/test_compat_api 与 __all__ 一致性。
输入输出：输入模块对象，输出导出集合断言结果。
依赖说明：依赖 pytest 与模块A导出入口。
维护说明：若兼容面收缩，需先更新本测试中的最小白名单。
"""

# 项目内模块：模块A命名空间
from music_video_pipeline.modules import module_a as module_a_impl


def test_module_a_exports_should_follow_whitelist_structure() -> None:
    """
    功能说明：验证 __all__ 由 public_api + test_compat_api 组成且无重复。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：public_api 仅保留稳定入口，兼容导出放入 test_compat_api。
    """
    assert isinstance(module_a_impl.public_api, list)
    assert isinstance(module_a_impl.test_compat_api, list)
    assert module_a_impl.__all__ == [*module_a_impl.public_api, *module_a_impl.test_compat_api]
    assert len(module_a_impl.__all__) == len(set(module_a_impl.__all__))
    assert module_a_impl.public_api == ["run_module_a"]


def test_module_a_exports_should_keep_minimum_compat_symbols() -> None:
    """
    功能说明：验证测试与迁移链路依赖的最小兼容符号仍可访问。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本测试只约束最小白名单，不限制兼容面的进一步收缩节奏。
    """
    minimum_symbols = {
        "run_module_a",
        "_run_real_pipeline",
        "_separate_with_demucs",
        "_detect_big_segments_with_allin1",
        "_extract_acoustic_candidates_with_librosa",
        "_recognize_lyrics_with_funasr",
        "_build_segments_with_lyric_priority",
        "_select_small_timestamps",
        "_attach_lyrics_to_segments",
        "_build_beats_from_segments",
        "_build_energy_features",
        "_build_segmentation_tuning",
    }
    for symbol_name in minimum_symbols:
        assert hasattr(module_a_impl, symbol_name)
        assert symbol_name in module_a_impl.__all__
