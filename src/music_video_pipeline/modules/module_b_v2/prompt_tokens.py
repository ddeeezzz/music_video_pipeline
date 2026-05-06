"""
文件用途：提供模块B v2 提示词 token 的解析、补全、去重与回编译工具。
核心流程：将 LLM 输出的宽松 tag 文本解析为结构化 token，执行风格补全与禁词清洗，再编译回兼容字符串。
输入输出：输入原始提示词文本与风格上下文，输出 token 数组与编译后的提示词字符串。
依赖说明：依赖标准库 re/hashlib 进行文本解析与稳定 ID 生成。
维护说明：本文件只处理提示词标准化，不负责业务字段路由与角色级调度。
"""

# 标准库：用于稳定 token_id 生成。
import hashlib
# 标准库：用于正则解析权重语法与文本清洗。
import re


# 常量：英文正向提示词最大 token 数量。
MAX_PROMPT_EN_TOKENS = 18
# 常量：中文正向提示词最大 token 数量。
MAX_PROMPT_ZH_TOKENS = 18
# 常量：英文正向提示词中需要移除的写实/彩色禁词。
BANNED_POSITIVE_EN_TOKENS = {
    "color",
    "colored",
    "colour",
    "colorful",
    "colourful",
    "photo",
    "photography",
    "photographic",
    "photorealistic",
    "realistic",
    "hyperrealistic",
    "3d",
    "cgs",
    "rendering",
    "cinematic lighting",
    "lens flare",
    "bokeh",
    "depth of field",
}
# 常量：中文正向提示词中需要移除的写实/彩色禁词。
BANNED_POSITIVE_ZH_TOKENS = {
    "彩色",
    "彩色照片",
    "照片",
    "摄影",
    "写实",
    "超写实",
    "三维",
    "3d",
    "景深",
    "散景",
}
# 常量：角色4默认追加的视频稳定性英文 token。
DEFAULT_VIDEO_STABILITY_TOKENS_EN = (
    "anime limited animation",
    "stable composition",
    "clean line continuity",
    "no flicker",
)
# 常量：角色4默认追加的视频稳定性中文 token。
DEFAULT_VIDEO_STABILITY_TOKENS_ZH = (
    "有限动画",
    "构图稳定",
    "线条连续",
    "无闪烁",
)
# 常量：权重语法解析正则。
WEIGHTED_TOKEN_PATTERN = re.compile(r"^\(\s*(.+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\)$")
# 常量：空白压缩正则。
MULTISPACE_PATTERN = re.compile(r"\s+")


class PromptToken(dict):
    """
    功能说明：表示单个标准化提示词 token。
    参数说明：继承 dict，不额外定义初始化参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：约定包含 id/text/weight 三个键。
    """


def parse_prompt_text_to_tokens(
    text: str,
    *,
    language: str,
    max_tokens: int,
    banned_texts: set[str] | None = None,
) -> list[dict[str, str | float | None]]:
    """
    功能说明：将宽松的逗号分隔提示词文本解析为结构化 token 数组。
    参数说明：
    - text: 原始提示词文本。
    - language: 语言标记（zh/en）。
    - max_tokens: 最大 token 数量。
    - banned_texts: 可选，需过滤的禁词集合。
    返回值：
    - list[dict[str, str | float | None]]: 标准化 token 数组。
    异常说明：无。
    边界条件：超过 max_tokens 时自动截断；空文本返回空数组。
    """
    raw_text = str(text or "").replace("，", ",").replace("；", ",")
    parts = [item.strip() for item in raw_text.split(",") if item and item.strip()]
    result: list[dict[str, str | float | None]] = []
    seen_keys: set[str] = set()
    forbidden = {_normalize_token_text(item) for item in (banned_texts or set()) if str(item).strip()}
    for item in parts:
        token_text, token_weight = _parse_single_token(item)
        normalized_key = _normalize_token_text(token_text)
        if not normalized_key:
            continue
        if normalized_key in forbidden:
            continue
        if normalized_key in seen_keys:
            continue
        seen_keys.add(normalized_key)
        result.append(_build_prompt_token(text=token_text, weight=token_weight, language=language))
        if len(result) >= max(1, int(max_tokens)):
            break
    return result


def compile_tokens_to_prompt_text(
    tokens: list[dict[str, str | float | None]],
    *,
    language: str,
    separator: str | None = None,
) -> str:
    """
    功能说明：将结构化 token 数组编译回兼容旧链路的提示词字符串。
    参数说明：
    - tokens: 结构化 token 数组。
    - language: 语言标记（zh/en）。
    - separator: 可选，自定义分隔符。
    返回值：
    - str: 编译后的提示词文本。
    异常说明：无。
    边界条件：空数组返回空字符串。
    """
    safe_separator = separator if separator is not None else ("，" if language == "zh" else ", ")
    items: list[str] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        token_text = str(token.get("text", "")).strip()
        if not token_text:
            continue
        token_weight = token.get("weight")
        if isinstance(token_weight, (int, float)) and float(token_weight) > 0:
            formatted_weight = f"{float(token_weight):.2f}".rstrip("0").rstrip(".")
            items.append(f"({token_text}:{formatted_weight})")
        else:
            items.append(token_text)
    return safe_separator.join(items)


def ensure_monochrome_style_tokens(
    tokens: list[dict[str, str | float | None]],
    *,
    language: str,
    style_text: str,
) -> list[dict[str, str | float | None]]:
    """
    功能说明：为提示词补齐黑白漫画风格 token。
    参数说明：
    - tokens: 原始 token 数组。
    - language: 语言标记（zh/en）。
    - style_text: 风格文本上下文。
    返回值：
    - list[dict[str, str | float | None]]: 补齐后的 token 数组。
    异常说明：无。
    边界条件：仅在 style_text 命中黑白/漫画语义时补齐相应 token。
    """
    result = [dict(item) for item in tokens if isinstance(item, dict)]
    style_context = str(style_text or "").lower()
    if language == "en":
        defaults = []
        if "black" in style_context or "monochrome" in style_context or "黑白" in style_context:
            defaults.extend(
                [
                    _build_prompt_token(text="black and white", weight=1.3, language="en"),
                    _build_prompt_token(text="monochrome", weight=1.2, language="en"),
                ]
            )
        if "manga" in style_context or "漫画" in style_context:
            defaults.extend(
                [
                    _build_prompt_token(text="line art", weight=None, language="en"),
                    _build_prompt_token(text="manga style", weight=None, language="en"),
                ]
            )
    else:
        defaults = []
        if "black" in style_context or "monochrome" in style_context or "黑白" in style_context:
            defaults.extend(
                [
                    _build_prompt_token(text="黑白", weight=1.3, language="zh"),
                    _build_prompt_token(text="单色", weight=1.2, language="zh"),
                ]
            )
        if "manga" in style_context or "漫画" in style_context:
            defaults.extend(
                [
                    _build_prompt_token(text="线稿", weight=None, language="zh"),
                    _build_prompt_token(text="漫画风", weight=None, language="zh"),
                ]
            )
    return _merge_prompt_tokens(defaults, result)


def build_negative_tokens_with_fixed_template(
    increment_text: str,
    *,
    language: str,
    fixed_template_text: str,
) -> tuple[list[dict[str, str | float | None]], list[dict[str, str | float | None]]]:
    """
    功能说明：将固定负面模板与 LLM 输出的增量负面线索合并为标准化 token。
    参数说明：
    - increment_text: LLM 输出的增量负面文本。
    - language: 语言标记（zh/en）。
    - fixed_template_text: 固定模板文本。
    返回值：
    - tuple[list[dict], list[dict]]: (增量 token，最终合并 token)。
    异常说明：无。
    边界条件：固定模板始终位于前部，增量部分只做追加去重。
    """
    increment_tokens = parse_prompt_text_to_tokens(
        increment_text,
        language=language,
        max_tokens=12,
        banned_texts=None,
    )
    template_tokens = parse_prompt_text_to_tokens(
        fixed_template_text,
        language=language,
        max_tokens=128,
        banned_texts=None,
    )
    return increment_tokens, _merge_prompt_tokens(template_tokens, increment_tokens)


def build_video_prompt_tokens(
    text: str,
    *,
    language: str,
    style_text: str,
) -> list[dict[str, str | float | None]]:
    """
    功能说明：构建视频提示词 token，并自动补齐风格与稳定性固定项。
    参数说明：
    - text: 原始视频提示词文本。
    - language: 语言标记（zh/en）。
    - style_text: 风格文本上下文。
    返回值：
    - list[dict[str, str | float | None]]: 标准化后的 token 数组。
    异常说明：无。
    边界条件：当前默认最多保留 18 个主 token，再追加固定稳定性 token。
    """
    base_tokens = parse_prompt_text_to_tokens(
        text,
        language=language,
        max_tokens=18,
        banned_texts=BANNED_POSITIVE_EN_TOKENS if language == "en" else BANNED_POSITIVE_ZH_TOKENS,
    )
    styled_tokens = ensure_monochrome_style_tokens(base_tokens, language=language, style_text=style_text)
    default_tail = [
        _build_prompt_token(text=item, weight=None, language=language)
        for item in (DEFAULT_VIDEO_STABILITY_TOKENS_EN if language == "en" else DEFAULT_VIDEO_STABILITY_TOKENS_ZH)
    ]
    return _merge_prompt_tokens(styled_tokens, default_tail)


def build_positive_prompt_tokens(
    text: str,
    *,
    language: str,
    style_text: str,
) -> list[dict[str, str | float | None]]:
    """
    功能说明：构建正向关键帧提示词 token，并执行禁词清洗与风格补齐。
    参数说明：
    - text: 原始正向提示词文本。
    - language: 语言标记（zh/en）。
    - style_text: 风格文本上下文。
    返回值：
    - list[dict[str, str | float | None]]: 标准化后的 token 数组。
    异常说明：无。
    边界条件：英文/中文均限制在 18 个主 token 以内。
    """
    base_tokens = parse_prompt_text_to_tokens(
        text,
        language=language,
        max_tokens=MAX_PROMPT_EN_TOKENS if language == "en" else MAX_PROMPT_ZH_TOKENS,
        banned_texts=BANNED_POSITIVE_EN_TOKENS if language == "en" else BANNED_POSITIVE_ZH_TOKENS,
    )
    merged_tokens = ensure_monochrome_style_tokens(base_tokens, language=language, style_text=style_text)
    max_count = MAX_PROMPT_EN_TOKENS if language == "en" else MAX_PROMPT_ZH_TOKENS
    return merged_tokens[:max_count]


def _merge_prompt_tokens(
    primary_tokens: list[dict[str, str | float | None]],
    secondary_tokens: list[dict[str, str | float | None]],
) -> list[dict[str, str | float | None]]:
    """
    功能说明：按顺序合并两组 token，并基于归一化文本去重。
    参数说明：
    - primary_tokens: 优先保留的 token。
    - secondary_tokens: 追加 token。
    返回值：
    - list[dict[str, str | float | None]]: 合并后的 token 数组。
    异常说明：无。
    边界条件：先出现的 token 保留原权重与原顺序。
    """
    result: list[dict[str, str | float | None]] = []
    seen_keys: set[str] = set()
    for token in [*primary_tokens, *secondary_tokens]:
        if not isinstance(token, dict):
            continue
        token_text = str(token.get("text", "")).strip()
        normalized_key = _normalize_token_text(token_text)
        if not normalized_key or normalized_key in seen_keys:
            continue
        seen_keys.add(normalized_key)
        result.append(
            {
                "id": str(token.get("id", "")).strip() or _build_token_id(token_text),
                "text": token_text,
                "weight": token.get("weight"),
            }
        )
    return result


def _build_prompt_token(text: str, weight: float | None, *, language: str) -> dict[str, str | float | None]:
    """
    功能说明：构造单个标准化 token。
    参数说明：
    - text: token 文本。
    - weight: 权重值，可为空。
    - language: 语言标记（zh/en）。
    返回值：
    - dict[str, str | float | None]: 标准化 token 对象。
    异常说明：无。
    边界条件：language 当前仅用于保留接口，不参与分支逻辑。
    """
    del language
    clean_text = _clean_token_text(text)
    return {
        "id": _build_token_id(clean_text),
        "text": clean_text,
        "weight": float(weight) if isinstance(weight, (int, float)) else None,
    }


def _parse_single_token(text: str) -> tuple[str, float | None]:
    """
    功能说明：解析单个 token 的文本与可选权重。
    参数说明：
    - text: 原始 token 文本。
    返回值：
    - tuple[str, float | None]: (token 文本, 权重)。
    异常说明：无。
    边界条件：未匹配 `(text:weight)` 语法时返回原文本与空权重。
    """
    normalized_text = _clean_token_text(text)
    match = WEIGHTED_TOKEN_PATTERN.match(normalized_text)
    if match is None:
        return normalized_text, None
    token_text = _clean_token_text(match.group(1))
    try:
        token_weight = float(match.group(2))
    except (TypeError, ValueError):
        token_weight = None
    return token_text, token_weight


def _clean_token_text(text: str) -> str:
    """
    功能说明：清洗 token 文本中的多余空白与分隔符残留。
    参数说明：
    - text: 原始文本。
    返回值：
    - str: 清洗后的文本。
    异常说明：无。
    边界条件：会移除首尾括号残留与多余空白。
    """
    normalized = str(text or "").strip().strip(",").strip("，").strip()
    normalized = MULTISPACE_PATTERN.sub(" ", normalized)
    return normalized


def _normalize_token_text(text: str) -> str:
    """
    功能说明：生成用于去重与禁词判定的归一化 key。
    参数说明：
    - text: token 文本。
    返回值：
    - str: 归一化后的 key。
    异常说明：无。
    边界条件：英文统一转小写，中文保留原字面。
    """
    normalized = _clean_token_text(text).lower()
    normalized = normalized.replace("（", "(").replace("）", ")")
    return normalized


def _build_token_id(text: str) -> str:
    """
    功能说明：基于 token 文本生成稳定 ID。
    参数说明：
    - text: token 文本。
    返回值：
    - str: 稳定 token ID。
    异常说明：无。
    边界条件：使用文本哈希避免中文/符号对路径与字段造成污染。
    """
    digest = hashlib.sha1(_clean_token_text(text).encode("utf-8")).hexdigest()[:12]
    return f"tok_{digest}"
