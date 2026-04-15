"""
文件用途：提供模块A V2输出别名映射构建。
核心流程：基于分析结果统计构建 alias_map 字段。
输入输出：输入运行模式与分析数据，输出标准 alias_map 字典。
依赖说明：仅依赖标准类型提示。
维护说明：输出结构需与既有 module_a_alias_v1 契约保持一致。
"""

# 标准库：类型提示
from typing import Any


def build_module_a_v2_alias_map(mode: str, analysis_data: dict[str, Any]) -> dict[str, Any]:
    """
    功能说明：构建模块A关键产物与时间戳处理链的统一命名别名表。
    参数说明：
    - mode: 模块A运行模式（fallback_only / real_* / v2_single）。
    - analysis_data: 模块A内部分析结果字典。
    返回值：
    - dict[str, Any]: 顶层 alias_map 字典，用于产物可读性与排查一致性。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：不改变原有契约字段，仅补充说明性别名对象。
    """
    is_fallback_mode = str(mode).lower().strip() == "fallback_only"
    stage1_big_segments = analysis_data.get("big_segments_stage1", analysis_data.get("big_segments", []))
    final_big_segments = analysis_data.get("big_segments", [])
    final_segments = analysis_data.get("segments", [])
    beats = analysis_data.get("beats", [])

    stage1_count = len(stage1_big_segments) if isinstance(stage1_big_segments, list) else 0
    final_big_count = len(final_big_segments) if isinstance(final_big_segments, list) else 0
    final_small_count = len(final_segments) if isinstance(final_segments, list) else 0
    beat_count = len(beats) if isinstance(beats, list) else 0

    return {
        "version": "module_a_alias_v1",
        "runtime_mode": "fallback" if is_fallback_mode else "real",
        "artifacts": {
            "A0": {
                "display_name": "A0段",
                "meaning": "Allin1直出大段（stage1）",
                "binding": "analysis_data.big_segments_stage1",
            },
            "AL": {
                "display_name": "AL段",
                "meaning": "按歌词证据重算边界后的A段",
                "binding": "analysis_data.big_segments（real链）",
            },
            "B": {
                "display_name": "B段",
                "meaning": "最终对外大段",
                "binding": "output_data.big_segments",
            },
            "M": {
                "display_name": "M段",
                "meaning": "中段（内部mid segments）",
                "binding": "segmentation内部阶段产物，不直接对外",
            },
            "S": {
                "display_name": "S段",
                "meaning": "最终对外小段",
                "binding": "output_data.segments",
            },
            "B_I": {
                "display_name": "B-I段",
                "meaning": "大器乐段（B段中 label∈instrumental_set）",
                "binding": "derived_from_output_data.big_segments",
            },
            "M_I": {
                "display_name": "M-I段",
                "meaning": "中器乐段（M段中 is_vocal=False）",
                "binding": "derived_from_mid_segments",
            },
            "S_I": {
                "display_name": "S-I段",
                "meaning": "小器乐段（S段中 label∈instrumental_set）",
                "binding": "derived_from_output_data.segments",
            },
            "B_V": {
                "display_name": "B-V段",
                "meaning": "大人声段（B段中 label∉instrumental_set）",
                "binding": "derived_from_output_data.big_segments",
            },
            "M_V": {
                "display_name": "M-V段",
                "meaning": "中人声段（M段中 is_vocal=True）",
                "binding": "derived_from_mid_segments",
            },
            "S_V": {
                "display_name": "S-V段",
                "meaning": "小人声段（S段中 label∉instrumental_set）",
                "binding": "derived_from_output_data.segments",
            },
        },
        "timestamp_chain": {
            "BT0": "A0段边界时戳（stage1大段边界）",
            "BT1": "AL段边界时戳（歌词重算后大段边界）",
            "BT_OUT": "B段边界时戳（最终对外大段边界）",
            "ST0": "小时戳候选池（beat/onset/lyric起点候选）",
            "ST1": "_select_small_timestamps筛选结果",
            "ST_OUT": "S段最终边界时戳（由segments首尾推导）",
        },
        "split_chain": [
            "切分1-M切：按人声RMS切出中段(M)",
            "切分2-R选：按歌词优先与器乐单次切分生成候选区间",
            "切分3-N并：短段并合与连续性归一",
            "切分4-S落：候选区间落成最终小段(S)",
        ],
        "observed_counts": {
            "A0_count": stage1_count,
            "AL_count": final_big_count,
            "B_count": final_big_count,
            "S_count": final_small_count,
            "beat_count": beat_count,
        },
    }
