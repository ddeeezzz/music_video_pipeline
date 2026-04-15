"""
文件用途：实现模块A V2的 FunASR 歌词重建与“按空歇分句”规则。
核心流程：FunASR推理 -> token时间戳归一 -> 鲁棒阈值估计 -> gap分句。
输入输出：输入音频路径与FunASR参数，输出歌词单元与分句统计信息。
依赖说明：依赖 FunASR、正则与基础数学函数；不依赖 module_a 分句实现。
维护说明：本文件禁止使用固定“间奏阈值秒数”过滤，离群由统计方法自动识别。
"""

# 标准库：数学函数
import math
# 标准库：正则表达式
import re
# 标准库：路径处理
from pathlib import Path
# 标准库：类型提示
from typing import Any

# 项目内模块：时间取整
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time as _round_time


# 常量：有效歌词内容判定（排除空白和常见标点）
SENTENCE_CONTENT_PATTERN = re.compile(r"[^\s，、；：。！？!?,.;:]+")
# 常量：标点识别（仅用于“样本采样”，不用于直接触发分句）
PUNCTUATION_PATTERN = re.compile(r"[，、；：。！？!?,.;:]")
# 常量：无可用样本时的默认分句阈值（秒）
DEFAULT_SENTENCE_SPLIT_GAP_SECONDS = 0.35
# 常量：分句阈值下限（秒），防止噪声抖动导致过密分句
MIN_SENTENCE_SPLIT_GAP_SECONDS = 0.04
# 常量：MAD 统计法离群阈值（Modified Z-Score）
GAP_OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
# 常量：MAD 数值稳定下界
MAD_EPSILON = 1e-6
# 常量：FunASR歌词默认无语音概率
DEFAULT_NO_SPEECH_PROB = 0.35
# 常量：FunASR置信度默认值
DEFAULT_CONFIDENCE = 0.65
# 常量：ModelScope默认模型缓存根目录
MODELSCOPE_MODELS_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "models"
# 常量：FunASR使用的VAD模型别名到仓库ID映射
FUNASR_VAD_MODEL_ALIAS_MAP = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    功能说明：安全转浮点，异常时回退默认值。
    参数说明：
    - value: 待转换值。
    - default: 回退值。
    返回值：
    - float: 转换结果。
    异常说明：异常在函数内吞并。
    边界条件：NaN/inf 回退 default。
    """
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if number != number or number in {float("inf"), float("-inf")}:
        return float(default)
    return number


def _to_json_safe(value: Any) -> Any:
    """
    功能说明：递归转换为可 JSON 序列化对象。
    参数说明：
    - value: 任意输入对象。
    返回值：
    - Any: 可序列化对象。
    异常说明：异常在函数内吞并并回退字符串。
    边界条件：复杂对象优先走 tolist/__dict__，失败则字符串化。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_json_safe(
                {
                    str(key): item
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            )
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _has_effective_lyric_content(text: str) -> bool:
    """
    功能说明：判断文本是否包含有效歌词内容（非纯标点）。
    参数说明：
    - text: 待判断文本。
    返回值：
    - bool: 是否有效。
    异常说明：无。
    边界条件：空字符串返回 False。
    """
    return bool(SENTENCE_CONTENT_PATTERN.search(str(text)))


def _is_punctuation_token(text: str) -> bool:
    """
    功能说明：判断 token 是否为标点。
    参数说明：
    - text: token 文本。
    返回值：
    - bool: 是否标点。
    异常说明：无。
    边界条件：仅用于样本抽样与左归属保护。
    """
    return bool(PUNCTUATION_PATTERN.search(str(text)))


def _split_text_for_timestamp(text: str) -> list[str]:
    """
    功能说明：按空格优先切词，否则逐字符切分，便于映射 timestamp。
    参数说明：
    - text: 文本内容。
    返回值：
    - list[str]: 切分结果。
    异常说明：无。
    边界条件：空文本返回空列表。
    """
    raw_text = str(text)
    if not raw_text.strip():
        return []
    if " " in raw_text:
        # 保留词前空白语义（如 " sounds"），用于后续英文文本拼接保持可读性。
        token_items: list[str] = []
        matches = list(re.finditer(r"\S+", raw_text))
        previous_end = 0
        for match_index, match_item in enumerate(matches):
            start_index = int(match_item.start())
            end_index = int(match_item.end())
            core_text = raw_text[start_index:end_index]
            if not core_text:
                continue
            if match_index == 0:
                token_items.append(core_text)
            else:
                leading_space = raw_text[previous_end:start_index]
                token_items.append(f"{leading_space}{core_text}")
            previous_end = end_index
        return [item for item in token_items if item.strip()]
    return [item for item in raw_text.strip() if item.strip()]


def _normalize_funasr_language(funasr_language: str, logger) -> str:
    """
    功能说明：归一化 FunASR 语言配置，不合法时回退 auto。
    参数说明：
    - funasr_language: FunASR语言策略配置。
    - logger: 日志记录器。
    返回值：
    - str: 归一化语言值。
    异常说明：无。
    边界条件：非法配置会记录 warning。
    """
    normalized = str(funasr_language).strip().lower().replace("_", "-")
    if not normalized or normalized == "auto":
        return "auto"
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", normalized):
        return normalized
    logger.warning("模块A V2-FunASR语言配置非法，已回退自动检测，原始值=%s", funasr_language)
    return "auto"


def _resolve_modelscope_cached_model_path(model_name: str) -> str | None:
    """
    功能说明：解析 ModelScope 本地缓存模型目录，命中则返回本地路径。
    参数说明：
    - model_name: 模型名、仓库ID或本地路径。
    返回值：
    - str | None: 命中缓存时返回目录字符串，否则返回 None。
    异常说明：无。
    边界条件：必须包含 model.pt 才视为可用缓存目录。
    """
    normalized = str(model_name).strip()
    if not normalized:
        return None

    candidates: list[Path] = []
    input_path = Path(normalized).expanduser()
    candidates.append(input_path)
    if "/" in normalized:
        candidates.append(MODELSCOPE_MODELS_DIR.joinpath(*normalized.split("/")))
    else:
        candidates.append(MODELSCOPE_MODELS_DIR / normalized)

    for path_item in candidates:
        if path_item.is_dir() and (path_item / "model.pt").exists():
            return str(path_item)
    return None


def _resolve_funasr_model_and_vad(model_name: str, vad_model_name: str) -> tuple[str, str, bool, bool]:
    """
    功能说明：优先解析 FunASR 主模型与VAD模型的本地缓存路径。
    参数说明：
    - model_name: FunASR主模型名或路径。
    - vad_model_name: VAD模型别名或路径。
    返回值：
    - tuple[str, str, bool, bool]: `(resolved_model, resolved_vad_model, model_cache_hit, vad_cache_hit)`。
    异常说明：无。
    边界条件：缓存未命中时回退原模型名/仓库ID。
    """
    model_cache_path = _resolve_modelscope_cached_model_path(model_name)
    resolved_model = model_cache_path or str(model_name).strip()
    preferred_vad = FUNASR_VAD_MODEL_ALIAS_MAP.get(str(vad_model_name).strip().lower(), str(vad_model_name).strip())
    vad_cache_path = _resolve_modelscope_cached_model_path(preferred_vad)
    resolved_vad_model = vad_cache_path or preferred_vad
    return resolved_model, resolved_vad_model, bool(model_cache_path), bool(vad_cache_path)


def _extract_funasr_records(result: Any) -> list[dict[str, Any]]:
    """
    功能说明：统一提取 FunASR record 列表。
    参数说明：
    - result: FunASR原始输出。
    返回值：
    - list[dict[str, Any]]: record 列表。
    异常说明：无。
    边界条件：结构不匹配时返回空列表。
    """
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        result_items = result.get("result")
        if isinstance(result_items, list):
            return [item for item in result_items if isinstance(item, dict)]
        return [result]
    return []


def _collect_numeric_time_values(timestamp_items: Any) -> list[float]:
    """
    功能说明：提取 timestamp 中的数值时间样本。
    参数说明：
    - timestamp_items: timestamp 列表。
    返回值：
    - list[float]: 时间样本。
    异常说明：无。
    边界条件：输入非法返回空列表。
    """
    if not isinstance(timestamp_items, list):
        return []
    values: list[float] = []
    for item in timestamp_items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            left_value = item[0]
            right_value = item[1]
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                values.extend([float(left_value), float(right_value)])
            continue
        if isinstance(item, dict):
            for key in ["start", "end", "begin", "finish", "start_time", "end_time"]:
                numeric_value = item.get(key)
                if isinstance(numeric_value, (int, float)) and not isinstance(numeric_value, bool):
                    values.append(float(numeric_value))
    return values


def _infer_funasr_time_scale(records: list[dict[str, Any]]) -> float:
    """
    功能说明：推断 FunASR 时间单位倍率（毫秒=0.001，秒=1.0）。
    参数说明：
    - records: FunASR records。
    返回值：
    - float: 时间单位倍率。
    异常说明：无。
    边界条件：无样本时按毫秒制回退 0.001。
    """
    values: list[float] = []
    for record in records:
        values.extend(_collect_numeric_time_values(record.get("timestamps", [])))
        values.extend(_collect_numeric_time_values(record.get("timestamp", [])))
        sentence_info = record.get("sentence_info", [])
        if isinstance(sentence_info, list):
            for sentence_item in sentence_info:
                if not isinstance(sentence_item, dict):
                    continue
                for key in ["start", "end"]:
                    numeric_value = sentence_item.get(key)
                    if isinstance(numeric_value, (int, float)) and not isinstance(numeric_value, bool):
                        values.append(float(numeric_value))
                values.extend(_collect_numeric_time_values(sentence_item.get("timestamp", [])))

    if not values:
        return 0.001
    max_value = max(values)
    has_fraction = any(abs(value - round(value)) > 1e-6 for value in values)
    if max_value >= 500.0:
        return 0.001
    if has_fraction and max_value <= 600.0:
        return 1.0
    if max_value <= 60.0:
        return 1.0
    return 0.001


def _normalize_funasr_confidence(raw_value: Any) -> float:
    """
    功能说明：将 FunASR 置信信号映射到 0~1。
    参数说明：
    - raw_value: 原始置信值。
    返回值：
    - float: 归一化置信度。
    异常说明：无。
    边界条件：异常值回退默认值。
    """
    if raw_value is None or isinstance(raw_value, bool):
        return DEFAULT_CONFIDENCE
    value = _safe_float(raw_value, DEFAULT_CONFIDENCE)
    if 0.0 <= value <= 1.0:
        return round(value, 3)
    if 1.0 < value <= 100.0:
        return round(max(0.0, min(1.0, value / 100.0)), 3)
    if -20.0 <= value < 0.0:
        return round(max(0.0, min(1.0, 1.0 + value / 20.0)), 3)
    return DEFAULT_CONFIDENCE


def _build_token_units_from_timestamp(
    timestamp_items: Any,
    text: str,
    time_scale: float,
    confidence: float,
) -> list[dict[str, Any]]:
    """
    功能说明：将 FunASR timestamp 转为统一 token_units。
    参数说明：
    - timestamp_items: timestamp 原始条目。
    - text: 对应文本。
    - time_scale: 时间倍率。
    - confidence: token 置信度。
    返回值：
    - list[dict[str, Any]]: token 列表。
    异常说明：无。
    边界条件：不支持的 timestamp 结构返回空列表。
    """
    if not isinstance(timestamp_items, list) or not timestamp_items:
        return []

    if all(isinstance(item, dict) for item in timestamp_items):
        token_units: list[dict[str, Any]] = []
        for item in timestamp_items:
            token_text_raw = str(item.get("text", item.get("token", "")))
            if not token_text_raw.strip():
                continue
            start_raw = item.get("start", item.get("begin", item.get("start_time", None)))
            end_raw = item.get("end", item.get("finish", item.get("end_time", None)))
            if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
                continue
            start_time = _round_time(_safe_float(start_raw, 0.0) * time_scale)
            end_time = _round_time(max(start_time, _safe_float(end_raw, start_time) * time_scale))
            token_units.append(
                {
                    "text": token_text_raw,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": confidence,
                }
            )
        return token_units

    if all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in timestamp_items):
        raw_text = str(text)
        token_texts = _split_text_for_timestamp(raw_text)
        token_units: list[dict[str, Any]] = []
        for index, item in enumerate(timestamp_items):
            start_raw = item[0]
            end_raw = item[1]
            if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
                continue
            token_text = token_texts[index] if index < len(token_texts) else ""
            if not str(token_text).strip():
                continue
            start_time = _round_time(_safe_float(start_raw, 0.0) * time_scale)
            end_time = _round_time(max(start_time, _safe_float(end_raw, start_time) * time_scale))
            token_units.append(
                {
                    "text": token_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": confidence,
                }
            )
        return token_units
    return []


def _build_token_units_from_record(record: dict[str, Any], time_scale: float) -> list[dict[str, Any]]:
    """
    功能说明：从单条 record 提取 token 列表（优先 timestamps）。
    参数说明：
    - record: FunASR record。
    - time_scale: 时间倍率。
    返回值：
    - list[dict[str, Any]]: token 列表。
    异常说明：无。
    边界条件：timestamps 缺失时回退 sentence_info。
    """
    confidence = _normalize_funasr_confidence(record.get("confidence", record.get("score")))
    text = str(record.get("text", "")).strip()
    timestamp_items = record.get("timestamp", record.get("timestamps", []))
    token_units = _build_token_units_from_timestamp(
        timestamp_items=timestamp_items,
        text=text,
        time_scale=time_scale,
        confidence=confidence,
    )
    if token_units:
        return token_units

    sentence_info = record.get("sentence_info", [])
    if not isinstance(sentence_info, list):
        return []

    fallback_tokens: list[dict[str, Any]] = []
    for sentence in sentence_info:
        if not isinstance(sentence, dict):
            continue
        sentence_text = str(sentence.get("text", "")).strip()
        sentence_confidence = _normalize_funasr_confidence(sentence.get("confidence", sentence.get("score", confidence)))
        sentence_tokens = _build_token_units_from_timestamp(
            timestamp_items=sentence.get("timestamp", []),
            text=sentence_text,
            time_scale=time_scale,
            confidence=sentence_confidence,
        )
        if sentence_tokens:
            fallback_tokens.extend(sentence_tokens)
            continue
        start_raw = sentence.get("start", 0.0)
        end_raw = sentence.get("end", start_raw)
        if not sentence_text or not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            continue
        start_time = _round_time(_safe_float(start_raw, 0.0) * time_scale)
        end_time = _round_time(max(start_time, _safe_float(end_raw, start_time) * time_scale))
        fallback_tokens.append(
            {
                "text": sentence_text,
                "start_time": start_time,
                "end_time": end_time,
                "confidence": sentence_confidence,
            }
        )
    return sorted(fallback_tokens, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))


def _find_neighbor_content_token_index(
    token_items: list[dict[str, Any]],
    center_index: int,
    direction: int,
) -> int | None:
    """
    功能说明：从中心点向左/右搜索最近内容 token。
    参数说明：
    - token_items: token 列表。
    - center_index: 中心索引。
    - direction: 方向（-1 左，+1 右）。
    返回值：
    - int | None: 命中索引。
    异常说明：无。
    边界条件：跨不到内容 token 时返回 None。
    """
    current_index = center_index + direction
    while 0 <= current_index < len(token_items):
        token_text = str(token_items[current_index].get("text", ""))
        if _has_effective_lyric_content(token_text):
            return current_index
        current_index += direction
    return None


def _estimate_boundary_content_gap(token_items: list[dict[str, Any]], left_boundary_index: int) -> float | None:
    """
    功能说明：估计边界左 token 右边界到右侧最近内容 token 左边界的空歇。
    参数说明：
    - token_items: token 列表。
    - left_boundary_index: 边界左侧索引。
    返回值：
    - float | None: 空歇秒数。
    异常说明：无。
    边界条件：边界越界或右侧找不到内容 token 时返回 None。
    """
    if left_boundary_index < 0 or left_boundary_index >= len(token_items) - 1:
        return None

    left_end = _safe_float(token_items[left_boundary_index].get("end_time", 0.0), 0.0)
    right_index = left_boundary_index + 1
    while right_index < len(token_items):
        if _has_effective_lyric_content(str(token_items[right_index].get("text", ""))):
            break
        right_index += 1
    if right_index >= len(token_items):
        return None
    right_start = _safe_float(token_items[right_index].get("start_time", left_end), left_end)
    return max(0.0, right_start - left_end)


def _median(values: list[float]) -> float:
    """
    功能说明：计算中位数。
    参数说明：
    - values: 数值列表。
    返回值：
    - float: 中位数。
    异常说明：无。
    边界条件：空列表返回 0.0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    count = len(sorted_values)
    middle = count // 2
    if count % 2 == 1:
        return sorted_values[middle]
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


def _filter_high_outliers_by_log_mad(samples: list[float]) -> tuple[list[float], list[float]]:
    """
    功能说明：使用 log1p + MAD 自动剔除“过长空歇”离群点。
    参数说明：
    - samples: 原始空歇样本。
    返回值：
    - tuple[list[float], list[float]]: `(保留样本, 被判离群样本)`。
    异常说明：无。
    边界条件：样本太少或MAD接近0时不剔除。
    """
    safe_samples = [max(0.0, float(item)) for item in samples if float(item) > 0.0]
    if len(safe_samples) < 4:
        return safe_samples, []

    transformed = [math.log1p(item) for item in safe_samples]
    median_value = _median(transformed)
    mad_value = _median([abs(item - median_value) for item in transformed])
    if mad_value <= MAD_EPSILON:
        return safe_samples, []

    kept_samples: list[float] = []
    outlier_samples: list[float] = []
    for raw_sample, transformed_sample in zip(safe_samples, transformed):
        modified_z_score = 0.6745 * (transformed_sample - median_value) / mad_value
        if modified_z_score > GAP_OUTLIER_MODIFIED_Z_THRESHOLD:
            outlier_samples.append(raw_sample)
        else:
            kept_samples.append(raw_sample)
    return kept_samples, outlier_samples


def _collect_punctuation_neighbor_gap_items(token_items: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    功能说明：收集标点左右最近内容 token 的边界间隔样本。
    参数说明：
    - token_items: token 列表。
    返回值：
    - list[dict[str, float]]: 样本列表（含左右内容边界与gap）。
    异常说明：无。
    边界条件：仅保留 gap>0 且标点左右均命中内容 token 的样本。
    """
    gap_items: list[dict[str, float]] = []
    for token_index, token_item in enumerate(token_items):
        token_text = str(token_item.get("text", ""))
        if not _is_punctuation_token(token_text) or _has_effective_lyric_content(token_text):
            continue
        left_index = _find_neighbor_content_token_index(token_items=token_items, center_index=token_index, direction=-1)
        right_index = _find_neighbor_content_token_index(token_items=token_items, center_index=token_index, direction=1)
        if left_index is None or right_index is None:
            continue
        left_end = _safe_float(token_items[left_index].get("end_time", 0.0), 0.0)
        right_start = _safe_float(token_items[right_index].get("start_time", left_end), left_end)
        gap_value = max(0.0, right_start - left_end)
        if gap_value <= 0.0:
            continue
        gap_items.append(
            {
                "left_end": float(left_end),
                "right_start": float(right_start),
                "gap": float(gap_value),
            }
        )
    return gap_items


def _compute_dynamic_sentence_split_gap(token_items: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    功能说明：估计动态断句阈值（无固定间奏秒阈值）。
    参数说明：
    - token_items: 全量 token 列表（按时间排序）。
    返回值：
    - tuple[float, dict[str, Any]]: `(阈值秒数, 统计信息)`。
    异常说明：无。
    边界条件：无样本时回退默认值。
    """
    punctuation_gap_items = _collect_punctuation_neighbor_gap_items(token_items=token_items)
    raw_samples = [float(item.get("gap", 0.0)) for item in punctuation_gap_items]
    if not raw_samples:
        return DEFAULT_SENTENCE_SPLIT_GAP_SECONDS, {
            "sample_source": "none",
            "sample_count_raw": 0,
            "sample_count_kept": 0,
            "sample_count_outlier": 0,
            "outlier_samples": [],
            "dynamic_gap_threshold_seconds": float(DEFAULT_SENTENCE_SPLIT_GAP_SECONDS),
        }

    kept_samples, outlier_samples = _filter_high_outliers_by_log_mad(raw_samples)
    effective_samples = kept_samples if kept_samples else raw_samples
    mean_gap = sum(effective_samples) / max(1, len(effective_samples))
    dynamic_gap_threshold = max(MIN_SENTENCE_SPLIT_GAP_SECONDS, float(mean_gap))
    return dynamic_gap_threshold, {
        "sample_source": "punctuation_neighbor",
        "sample_count_raw": len(raw_samples),
        "sample_count_kept": len(effective_samples),
        "sample_count_outlier": len(outlier_samples),
        "outlier_samples": [round(float(item), 6) for item in sorted(outlier_samples, reverse=True)],
        "dynamic_gap_threshold_seconds": float(dynamic_gap_threshold),
    }


def _build_sentence_unit_from_tokens(tokens: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """
    功能说明：将 token 子序列构造为句级歌词单元。
    参数说明：
    - tokens: token 子序列。
    - index: 输出句序号（从0开始）。
    返回值：
    - dict[str, Any]: 句级单元。
    异常说明：无。
    边界条件：空输入返回空字典。
    """
    if not tokens:
        return {}
    sorted_tokens = sorted(tokens, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))
    start_time = _round_time(_safe_float(sorted_tokens[0].get("start_time", 0.0), 0.0))
    end_time = _round_time(max(start_time, _safe_float(sorted_tokens[-1].get("end_time", start_time), start_time)))
    text_parts = [str(item.get("text", "")) for item in sorted_tokens]
    merged_text = "".join(text_parts).strip()
    confidence_values = [_safe_float(item.get("confidence", DEFAULT_CONFIDENCE), DEFAULT_CONFIDENCE) for item in sorted_tokens]
    confidence = _round_time(sum(confidence_values) / max(1, len(confidence_values)))
    return {
        "start_time": start_time,
        "end_time": end_time,
        "text": merged_text,
        "confidence": _normalize_funasr_confidence(confidence),
        "no_speech_prob": DEFAULT_NO_SPEECH_PROB,
        "token_units": sorted_tokens,
        "source_sentence_index": int(index),
        "unit_transform": "gap_split_v2",
    }


def _split_tokens_by_gap(token_items: list[dict[str, Any]], gap_threshold_seconds: float) -> list[dict[str, Any]]:
    """
    功能说明：按 token 间空歇切句（标点只左归属，不直接触发切句）。
    参数说明：
    - token_items: 全量 token 列表。
    - gap_threshold_seconds: 动态阈值（秒）。
    返回值：
    - list[dict[str, Any]]: 切分后的句级单元。
    异常说明：无。
    边界条件：纯标点句会被过滤。
    """
    if not token_items:
        return []
    sorted_tokens = sorted(token_items, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))
    safe_threshold = max(MIN_SENTENCE_SPLIT_GAP_SECONDS, float(gap_threshold_seconds))
    current_tokens: list[dict[str, Any]] = []
    current_has_content = False
    sentence_units: list[dict[str, Any]] = []

    for token_index, token_item in enumerate(sorted_tokens):
        token_text = str(token_item.get("text", ""))
        current_tokens.append(token_item)
        if _has_effective_lyric_content(token_text):
            current_has_content = True

        should_split = False
        if token_index < len(sorted_tokens) - 1:
            next_text = str(sorted_tokens[token_index + 1].get("text", ""))
            # 句尾标点左归属：不在“标点前”切分，避免把句尾标点甩给后句。
            if not (_is_punctuation_token(next_text) and not _has_effective_lyric_content(next_text)):
                gap_value = _estimate_boundary_content_gap(token_items=sorted_tokens, left_boundary_index=token_index)
                if gap_value is not None and float(gap_value) >= safe_threshold:
                    should_split = True

        if should_split and current_has_content:
            sentence_unit = _build_sentence_unit_from_tokens(current_tokens, len(sentence_units))
            if sentence_unit and _has_effective_lyric_content(str(sentence_unit.get("text", ""))):
                sentence_units.append(sentence_unit)
            current_tokens = []
            current_has_content = False

    if current_tokens and current_has_content:
        sentence_unit = _build_sentence_unit_from_tokens(current_tokens, len(sentence_units))
        if sentence_unit and _has_effective_lyric_content(str(sentence_unit.get("text", ""))):
            sentence_units.append(sentence_unit)
    return sentence_units


def build_lyric_units_from_funasr_result(
    raw_result: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：从 FunASR 原始输出重建 V2 句级歌词单元与分句统计。
    参数说明：
    - raw_result: FunASR 原始输出。
    返回值：
    - tuple[list[dict[str, Any]], dict[str, Any]]: `(lyric_units_raw, split_stats)`。
    异常说明：无。
    边界条件：无 token 时返回空歌词并标记原因。
    """
    records = _extract_funasr_records(raw_result)
    if not records:
        return [], {
            "sample_source": "none",
            "sample_count_raw": 0,
            "sample_count_kept": 0,
            "sample_count_outlier": 0,
            "outlier_samples": [],
            "dynamic_gap_threshold_seconds": float(DEFAULT_SENTENCE_SPLIT_GAP_SECONDS),
            "reason": "empty_records",
        }

    time_scale = _infer_funasr_time_scale(records)
    token_items: list[dict[str, Any]] = []
    for record in records:
        token_items.extend(_build_token_units_from_record(record=record, time_scale=time_scale))
    token_items = sorted(token_items, key=lambda item: _safe_float(item.get("start_time", 0.0), 0.0))
    if not token_items:
        return [], {
            "sample_source": "none",
            "sample_count_raw": 0,
            "sample_count_kept": 0,
            "sample_count_outlier": 0,
            "outlier_samples": [],
            "dynamic_gap_threshold_seconds": float(DEFAULT_SENTENCE_SPLIT_GAP_SECONDS),
            "time_scale": float(time_scale),
            "token_count": 0,
            "reason": "empty_tokens",
        }

    dynamic_gap_threshold_seconds, split_stats = _compute_dynamic_sentence_split_gap(token_items=token_items)
    lyric_units_raw = _split_tokens_by_gap(token_items=token_items, gap_threshold_seconds=dynamic_gap_threshold_seconds)
    split_stats = {
        **split_stats,
        "time_scale": float(time_scale),
        "token_count": len(token_items),
        "sentence_count": len(lyric_units_raw),
    }
    return lyric_units_raw, split_stats


def recognize_lyrics_with_funasr_v2(
    audio_path: str,
    model_name: str,
    device: str,
    funasr_language: str,
    logger,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """
    功能说明：执行 FunASR 识别并返回 V2 句级歌词单元。
    参数说明：
    - audio_path: 输入音频路径字符串。
    - model_name: FunASR 模型名称。
    - device: 推理设备（cpu/cuda/auto）。
    - funasr_language: 语言策略。
    - logger: 日志记录器。
    返回值：
    - tuple[Any, list[dict[str, Any]], dict[str, Any]]: `(raw_result_json_safe, lyric_units_raw, split_stats)`。
    异常说明：模型初始化或推理失败时抛 RuntimeError。
    边界条件：歌词可为空；raw_result 始终返回可序列化结构。
    """
    try:
        # 第三方库：FunASR 包版本信息读取与模型注册环境
        import funasr as funasr_pkg  # type: ignore
        # 第三方库：FunASR 统一模型入口
        from funasr import AutoModel  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"funasr 导入失败: {error}") from error

    normalized_language = _normalize_funasr_language(funasr_language=funasr_language, logger=logger)
    language_policy = "auto_detect" if normalized_language == "auto" else normalized_language
    logger.info("模块A V2调用 FunASR 识别歌词，模型=%s，设备=%s，语言策略=%s", model_name, device, language_policy)

    resolved_model_name, resolved_vad_model_name, model_cache_hit, vad_cache_hit = _resolve_funasr_model_and_vad(
        model_name=model_name,
        vad_model_name="fsmn-vad",
    )
    if model_cache_hit:
        logger.info("模块A V2-FunASR命中主模型本地缓存，path=%s", resolved_model_name)
    if vad_cache_hit:
        logger.info("模块A V2-FunASR命中VAD本地缓存，path=%s", resolved_vad_model_name)

    model_kwargs: dict[str, Any] = {
        "model": resolved_model_name,
        "vad_model": resolved_vad_model_name,
        "vad_kwargs": {"max_single_segment_time": 30000},
        "check_latest": False,
        "disable_update": True,
    }
    if str(device).strip().lower() != "auto":
        model_kwargs["device"] = device

    try:
        model = AutoModel(**model_kwargs)
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if "is not registered" in message or "FunASRNano" in message:
            funasr_version = str(getattr(funasr_pkg, "__version__", "unknown"))
            raise RuntimeError(
                f"FunASR 模型注册失败（FunASRNano 未注册），当前 funasr 版本={funasr_version}。"
                "请使用 README 对齐的 Git 锁定版 FunASR 依赖。"
            ) from error
        raise RuntimeError(f"FunASR 模型初始化失败: {error}") from error

    generate_kwargs: dict[str, Any] = {
        "input": [str(audio_path)],
        "cache": {},
        "batch_size_s": 0,
    }
    if normalized_language != "auto":
        generate_kwargs["language"] = normalized_language

    raw_result = model.generate(**generate_kwargs)
    lyric_units_raw, split_stats = build_lyric_units_from_funasr_result(raw_result=raw_result)
    return _to_json_safe(raw_result), lyric_units_raw, split_stats
