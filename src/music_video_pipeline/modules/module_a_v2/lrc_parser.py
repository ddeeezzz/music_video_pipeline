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
# 常量：增强 LRC 词级时间标记
ENHANCED_LRC_TOKEN_PATTERN = re.compile(
    r"<(?P<start>\d{1,2}:\d{2}(?:\.\d{1,3})?)>(?P<text>.*?)<(?P<end>\d{1,2}:\d{2}(?:\.\d{1,3})?)>"
)
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
            token_units = _parse_enhanced_lrc_token_units(line_text)
            lyric_text = _strip_lrc_tags(line_text).strip()
            if token_units:
                lyric_text = "".join(str(item.get("text", "")) for item in token_units).strip() or lyric_text
            if not lyric_text:
                continue
            for match_item in timestamps:
                parsed_rows.append(
                    {
                        "start_time": round_time(
                            float(token_units[0]["start_time"]) if token_units else _parse_lrc_timestamp(match_item)
                        ),
                        "text": lyric_text,
                        "token_units": token_units,
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
                    "token_units": list(item.get("token_units", [])) if isinstance(item.get("token_units", []), list) else [],
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


def _parse_lrc_timestamp_text(timestamp_text: str) -> float:
    """
    功能说明：将 `MM:SS.xx` 文本转换为秒。
    参数说明：
    - timestamp_text: 时间戳文本。
    返回值：
    - float: 秒级时间。
    异常说明：格式不合法时返回 0.0。
    边界条件：与 LRC 常见 2/3 位小数兼容。
    """
    normalized_text = str(timestamp_text).strip()
    timestamp_match = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?", normalized_text)
    if timestamp_match is None:
        return 0.0
    minutes_value = int(timestamp_match.group(1) or "0")
    seconds_value = int(timestamp_match.group(2) or "0")
    fraction_raw = str(timestamp_match.group(3) or "").strip()
    fraction_value = float(f"0.{fraction_raw}") if fraction_raw else 0.0
    return float(minutes_value * 60 + seconds_value) + fraction_value


def _parse_enhanced_lrc_token_units(line_text: str) -> list[dict[str, Any]]:
    """
    功能说明：解析增强 LRC 的词级 `<start>词<end>` 标记。
    参数说明：
    - line_text: 单行 LRC 文本。
    返回值：
    - list[dict[str, Any]]: 词级 token 列表。
    异常说明：无；匹配不到时返回空列表。
    边界条件：仅保留非空文本 token。
    """
    token_units: list[dict[str, Any]] = []
    for token_match in ENHANCED_LRC_TOKEN_PATTERN.finditer(str(line_text)):
        token_text = str(token_match.group("text") or "")
        if not token_text:
            continue
        start_time = round_time(_parse_lrc_timestamp_text(str(token_match.group("start") or "")))
        end_time = round_time(max(start_time, _parse_lrc_timestamp_text(str(token_match.group("end") or ""))))
        token_units.append(
            {
                "text": token_text,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return token_units


def _strip_lrc_tags(line_text: str) -> str:
    """
    功能说明：移除单行 LRC 中的行级与词级时间标记，只保留正文。
    参数说明：
    - line_text: 单行 LRC 文本。
    返回值：
    - str: 去标记后的文本。
    异常说明：无。
    边界条件：不会压缩正文内部原始空格。
    """
    without_line_tags = LRC_TIMESTAMP_PATTERN.sub("", str(line_text))
    return re.sub(r"<\d{1,2}:\d{2}(?:\.\d{1,3})?>", "", without_line_tags)
