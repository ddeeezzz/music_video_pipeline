"""
文件用途：验证模块A V2能量特征构建逻辑的自适应行为。
核心流程：构造可控RMS序列并断言 energy_level/trend 输出。
输入输出：输入模拟 segments 与 rms 序列，输出断言结果。
依赖说明：依赖 pytest 与模块A V2能量特征函数。
维护说明：本测试仅校验特征规则，不覆盖编排逻辑。
"""

# 项目内模块：能量特征构建函数
from music_video_pipeline.modules.module_a_v2.energy.features import build_energy_features


def test_build_energy_features_should_generate_high_and_low_levels() -> None:
    """
    功能说明：验证分布有跨度时可稳定产出 high 与 low 能量段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时由 pytest 抛出。
    边界条件：使用固定构造样本保证可重复。
    """
    segments = [
        {"segment_id": f"s_{index}", "start_time": float(index), "end_time": float(index + 1), "label": "verse"}
        for index in range(8)
    ]
    rms_times = []
    rms_values = []
    energy_pairs = [
        (0.05, 0.06),
        (0.08, 0.09),
        (0.12, 0.13),
        (0.18, 0.19),
        (0.27, 0.28),
        (0.40, 0.41),
        (0.58, 0.59),
        (0.80, 0.82),
    ]
    for index, pair in enumerate(energy_pairs):
        rms_times.extend([index + 0.2, index + 0.8])
        rms_values.extend(pair)

    output = build_energy_features(segments=segments, rms_times=rms_times, rms_values=rms_values, beat_candidates=[])
    levels = [str(item.get("energy_level", "")) for item in output]

    assert len(output) == len(segments)
    assert "high" in levels
    assert "low" in levels


def test_build_energy_features_should_detect_relative_trend_under_low_absolute_energy() -> None:
    """
    功能说明：验证低绝对能量条件下仍可识别相对趋势。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时由 pytest 抛出。
    边界条件：趋势变化量设置为小绝对值但高相对值。
    """
    segments = [
        {"segment_id": f"s_{index}", "start_time": float(index), "end_time": float(index + 1), "label": "verse"}
        for index in range(6)
    ]
    rms_times = []
    rms_values = []
    trend_pairs = [
        (0.020, 0.028),
        (0.031, 0.024),
        (0.022, 0.024),
        (0.025, 0.024),
        (0.023, 0.0235),
        (0.024, 0.0242),
    ]
    for index, pair in enumerate(trend_pairs):
        rms_times.extend([index + 0.2, index + 0.8])
        rms_values.extend(pair)

    output = build_energy_features(segments=segments, rms_times=rms_times, rms_values=rms_values, beat_candidates=[])
    trends = [str(item.get("trend", "")) for item in output]

    assert "up" in trends
    assert "down" in trends


def test_build_energy_features_should_keep_mid_when_distribution_is_flat() -> None:
    """
    功能说明：验证能量分布过平时不会误判大量 high/low。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时由 pytest 抛出。
    边界条件：所有分段能量均在窄幅区间波动。
    """
    segments = [
        {"segment_id": f"s_{index}", "start_time": float(index), "end_time": float(index + 1), "label": "verse"}
        for index in range(5)
    ]
    rms_times = []
    rms_values = []
    flat_pairs = [
        (0.100, 0.101),
        (0.101, 0.1005),
        (0.1008, 0.1012),
        (0.1007, 0.1009),
        (0.1009, 0.1010),
    ]
    for index, pair in enumerate(flat_pairs):
        rms_times.extend([index + 0.2, index + 0.8])
        rms_values.extend(pair)

    output = build_energy_features(segments=segments, rms_times=rms_times, rms_values=rms_values, beat_candidates=[])
    levels = [str(item.get("energy_level", "")) for item in output]

    assert set(levels) == {"mid"}
