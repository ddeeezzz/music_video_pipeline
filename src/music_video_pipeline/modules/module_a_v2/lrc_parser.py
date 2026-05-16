"""
文件用途：将 LRCLIB 返回的同步歌词 LRC 解析为模块A V2统一句级歌词单元。
核心流程：按行提取时间戳与文本，补齐 end_time，并生成统一结构。
输入输出：输入 LRC 文本与音频时长，输出 lyric_sentence_units 列表。
依赖说明：依赖标准库正则表达式。
维护说明：阶段一不做 beat 吸附，只做最小可用解析。
"""

# 标准库：用于正则解析
import re
# 标准库：用于类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time


# 常量：LRC 行时间戳提取规则
LRC_TIMESTAMP_PATTERN = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]")
# 常量：LRC 元信息行识别规则
LRC_METADATA_LINE_PATTERN = re.compile(r"^\[[a-zA-Z]{2,}:[^\]]*\]\s*$")
# 常量：LRCLIB 主链默认置信度
DEFAULT_LRCLIB_CONFIDENCE = 0.95


def parse_synced_lyrics_to_sentence_units(synced_lyrics: str, audio_duration: float, logger) -> list[dict[str, Any]]:
    """
    功能说明：解析同步歌词 LRC 为统一句级歌词单元。
    参数说明：
    - synced_lyrics: LRC 原文。
    - audio_duration: 音频总时长（秒）。
    - logger: 日志记录器。
    返回值：
    - list[dict[str, Any]]: 标准化歌词句单元列表。
    异常说明：异常在函数内吞并并回退空列表。
    边界条件：只保留同时具备时间戳和非空文本的歌词行。
    """
    try:
        parsed_rows: list[dict[str, Any]] = []
        for raw_line in str(synced_lyrics).splitlines():
            line_text = raw_line.strip()
            if not line_text:
                continue
            if LRC_METADATA_LINE_PATTERN.fullmatch(line_text):
                continue
            timestamps = list(LRC_TIMESTAMP_PATTERN.finditer(line_text))
            if not timestamps:
                continue
            lyric_text = LRC_TIMESTAMP_PATTERN.sub("", line_text).strip()
            if not lyric_text:
                continue
            for match_item in timestamps:
                parsed_rows.append(
                    {
                        "start_time": round_time(_parse_lrc_timestamp(match_item)),
                        "text": lyric_text,
                    }
                )
        parsed_rows.sort(key=lambda item: float(item["start_time"]))
        sentence_units: list[dict[str, Any]] = []
        safe_audio_duration = max(0.0, float(audio_duration))
        for index, item in enumerate(parsed_rows):
            start_time = round_time(float(item["start_time"]))
            next_start = safe_audio_duration
            if index + 1 < len(parsed_rows):
                next_start = float(parsed_rows[index + 1]["start_time"])
            end_time = round_time(max(start_time, min(safe_audio_duration, next_start)))
            sentence_units.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": str(item["text"]).strip(),
                    "confidence": DEFAULT_LRCLIB_CONFIDENCE,
                    "token_units": [],
                    "source_sentence_index": index,
                    "unit_transform": "original",
                }
            )
        logger.info("模块A V2-LRC解析完成，句数=%s", len(sentence_units))
        return sentence_units
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-LRC解析失败，错误=%s", error)
        return []


def _parse_lrc_timestamp(match_item: re.Match[str]) -> float:
    """
    功能说明：将 LRC 时间戳匹配项转换为秒。
    参数说明：
    - match_item: 正则匹配对象。
    返回值：
    - float: 秒级时间。
    异常说明：无。
    边界条件：毫秒位不足三位时按常规十进制补齐。
    """
    minutes_value = int(match_item.group(1) or "0")
    seconds_value = int(match_item.group(2) or "0")
    fraction_raw = str(match_item.group(3) or "").strip()
    fraction_value = 0.0
    if fraction_raw:
        fraction_value = float(f"0.{fraction_raw}")
    return float(minutes_value * 60 + seconds_value) + fraction_value
