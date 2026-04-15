"""
文件用途：提供模块A V2歌词清洗能力。
核心流程：过滤器乐段、噪声词与低可信占位词，输出可挂载句级歌词。
输入输出：输入原始歌词单元与大段信息，输出清洗后的歌词单元。
依赖说明：依赖正则与 v2 时间工具。
维护说明：本文件仅负责清洗，不承担歌词挂载逻辑。
"""

# 标准库：正则处理
import re
# 标准库：类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time


def clean_lyric_units(
    lyric_units_raw: list[dict[str, Any]],
    big_segments: list[dict[str, Any]],
    instrumental_labels: list[str],
    logger,
    min_confidence: float = 0.25,
) -> list[dict[str, Any]]:
    """
    功能说明：清洗 ASR 原始结果，输出可靠歌词/未识别歌词/吟唱三态。
    参数说明：
    - lyric_units_raw: ASR原始歌词单元列表。
    - big_segments: 大段落列表。
    - instrumental_labels: 器乐标签集合配置，用于区分器乐段与人声段。
    - logger: 日志记录器。
    - min_confidence: 最低可接受置信度阈值。
    返回值：
    - list[dict[str, Any]]: 清洗后的歌词单元列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：空输入返回空列表。
    """
    if not lyric_units_raw:
        return []

    instrumental_set = {label.lower().strip() for label in instrumental_labels}
    cleaned_units: list[dict[str, Any]] = []
    unknown_text = "[未识别歌词]"
    for item in lyric_units_raw:
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", start_time))
        start_time = round_time(start_time)
        end_time = round_time(max(start_time, end_time))
        text = str(item.get("text", "")).strip()
        confidence = round(float(item.get("confidence", 0.0)), 3)
        no_speech_prob = max(0.0, min(1.0, float(item.get("no_speech_prob", 0.35))))
        token_units = _normalize_token_units(item.get("token_units", []))
        if not text:
            continue

        mid_time = (start_time + end_time) / 2.0
        big_segment = _find_big_segment(mid_time, big_segments)
        big_label = str(big_segment.get("label", "unknown")).lower().strip()
        if big_label in instrumental_set:
            continue

        if _is_vocalise_text(text):
            cleaned_units.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": "吟唱",
                    "confidence": max(0.5, confidence),
                    "token_units": [],
                }
            )
            continue

        if _is_obvious_noise_text(text=text, confidence=confidence, no_speech_prob=no_speech_prob):
            continue

        suspected_vocal = _is_probable_vocal_presence(
            confidence=confidence,
            no_speech_prob=no_speech_prob,
            start_time=start_time,
            end_time=end_time,
        )
        if _is_placeholder_text(text):
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                        "token_units": [],
                    }
                )
            continue

        if confidence < min_confidence:
            if suspected_vocal:
                cleaned_units.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": unknown_text,
                        "confidence": max(0.2, confidence),
                        "token_units": [],
                    }
                )
            continue

        cleaned_units.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "confidence": confidence,
                "token_units": token_units,
            }
        )

    cleaned_units.sort(key=lambda item: float(item["start_time"]))
    logger.info("模块A V2-歌词清洗完成，原始=%s，保留=%s", len(lyric_units_raw), len(cleaned_units))
    return cleaned_units


def _find_big_segment(time_value: float, big_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：按时间定位所属大段落。
    参数说明：
    - time_value: 当前处理的时间戳（秒）。
    - big_segments: 大段落列表。
    返回值：
    - dict[str, Any]: 命中的大段落字典对象。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：未命中时回退最后一段；空列表时返回 unknown 占位段。
    """
    if not big_segments:
        return {"segment_id": "big_000", "start_time": 0.0, "end_time": 0.0, "label": "unknown"}
    for item in big_segments:
        if float(item["start_time"]) <= time_value <= float(item["end_time"]):
            return item
    return big_segments[-1]


def _normalize_token_units(token_units: Any) -> list[dict[str, Any]]:
    """
    功能说明：清洗 token_units，确保结构满足 text/start/end。
    参数说明：
    - token_units: token级时间单元列表。
    返回值：
    - list[dict[str, Any]]: 归一化后的 token 列表。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：输入非法或空内容时返回空列表。
    """
    if not isinstance(token_units, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in token_units:
        if not isinstance(item, dict):
            continue
        start_raw = item.get("start_time", item.get("start", 0.0))
        end_raw = item.get("end_time", item.get("end", start_raw))
        if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            continue
        start_time = round_time(float(start_raw))
        end_time = round_time(max(start_time, float(end_raw)))
        token_text = str(item.get("text", ""))
        if not token_text.strip():
            continue
        normalized.append(
            {
                "text": token_text,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return normalized


def _is_placeholder_text(text: str) -> bool:
    """
    功能说明：判断是否为歌词占位词（用于转写不可靠标记）。
    参数说明：
    - text: 文本内容。
    返回值：
    - bool: 是否占位词。
    异常说明：无。
    边界条件：空文本返回 False。
    """
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return False
    placeholder_set = {"lyrics", "lyric", "歌詞", "歌词", "歌詩", "lyrix", "music"}
    return normalized in placeholder_set


def _is_obvious_noise_text(text: str, confidence: float, no_speech_prob: float) -> bool:
    """
    功能说明：判断明显噪声文本（用于直接过滤，避免幻觉词显示）。
    参数说明：
    - text: 文本内容。
    - confidence: 识别置信度。
    - no_speech_prob: 无语音概率估计值。
    返回值：
    - bool: 是否应过滤。
    异常说明：无。
    边界条件：空文本视为噪声。
    """
    normalized = re.sub(r"[\s\-\_\.\,\!\?]+", "", text.strip().lower())
    if not normalized:
        return True
    if len(normalized) <= 2 and re.fullmatch(r"[a-z]+", normalized):
        return True
    if len(normalized) <= 3 and re.fullmatch(r"[a-z]+", normalized) and confidence < 0.35:
        return True
    if no_speech_prob >= 0.85 and confidence < 0.4:
        return True
    return False


def _is_probable_vocal_presence(confidence: float, no_speech_prob: float, start_time: float, end_time: float) -> bool:
    """
    功能说明：基于置信度+无声概率估计是否存在人声。
    参数说明：
    - confidence: 识别置信度。
    - no_speech_prob: 无语音概率估计值。
    - start_time: 区间起始时间（秒）。
    - end_time: 区间结束时间（秒）。
    返回值：
    - bool: 是否可能存在人声。
    异常说明：无。
    边界条件：遵循现有经验规则，不新增分支。
    """
    duration = max(0.0, end_time - start_time)
    if no_speech_prob <= 0.45:
        return True
    if no_speech_prob <= 0.6 and (confidence >= 0.12 or duration >= 0.35):
        return True
    return False


def _is_vocalise_text(text: str) -> bool:
    """
    功能说明：判断是否为无语义吟唱片段（lalala/dadada/啦啦啦 等）。
    参数说明：
    - text: 文本内容。
    返回值：
    - bool: 是否吟唱文本。
    异常说明：无。
    边界条件：空文本返回 False。
    """
    raw_text = text.strip().lower()
    compact_text = re.sub(r"[\s\-\_\,\.\!\?]+", "", raw_text)
    if not compact_text:
        return False

    if re.fullmatch(r"(?:la|da|na|ra){3,}", compact_text):
        return True
    if re.fullmatch(r"[らラ啦らぁラァ]{3,}", compact_text):
        return True
    if re.fullmatch(r"(?:は|ハ|啊|あ){3,}", compact_text):
        return True
    if re.fullmatch(r"(?:吟唱)+", compact_text):
        return True
    return False
