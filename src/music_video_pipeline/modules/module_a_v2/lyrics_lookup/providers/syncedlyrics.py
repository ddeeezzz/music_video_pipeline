"""
文件用途：提供 syncedlyrics 聚合歌词搜索的最小封装。
核心流程：延迟导入第三方库，执行同步歌词搜索，并将结果标准化为项目内部结构。
输入输出：输入查询词或歌手/歌名，输出标准化同步歌词候选列表。
依赖说明：依赖第三方库 syncedlyrics 与项目内查询词规范化工具。
维护说明：本文件只负责补充歌词搜索源，不承担主链优先级编排职责。
"""

# 标准库：用于正则清洗查询词
import re
# 标准库：用于类型提示
from typing import Any


# 常量：标题尾部括注清理规则，避免中日文解释影响命中
TRAILING_BRACKET_NOTE_PATTERN = re.compile(r"\s*[\(（\[【].*?[\)）\]】]\s*$")
SYNCEDLYRICS_PROVIDER_NAMES = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "Genius"]


def search_syncedlyrics_candidates(
    *,
    query_text: str = "",
    artist: str = "",
    title: str = "",
    logger,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """
    功能说明：使用 syncedlyrics 聚合源搜索同步歌词，并标准化为项目候选结构。
    参数说明：
    - query_text: 自由文本查询词。
    - artist: 艺人名。
    - title: 曲名。
    - logger: 日志对象。
    - limit: 最多返回候选数；当前聚合库一次最多稳定返回一首，保留该参数以兼容上层接口。
    返回值：
    - list[dict[str, Any]]: 已标准化的同步歌词候选数组。
    异常说明：第三方库缺失或搜索异常时返回空数组。
    边界条件：若原始查询失败，会自动尝试去除标题尾部括注后的净化查询。
    """
    syncedlyrics_module = _import_syncedlyrics_module(logger=logger)
    if syncedlyrics_module is None:
        return []
    search_terms = _build_syncedlyrics_search_terms(
        query_text=str(query_text).strip(),
        artist=str(artist).strip(),
        title=str(title).strip(),
    )
    for search_term in search_terms:
        try:
            synced_lyrics = syncedlyrics_module.search(search_term, synced_only=True)
        except Exception as error:  # noqa: BLE001
            logger.warning("模块A V2-syncedlyrics 搜索失败，query=%s，错误=%s", search_term, error)
            continue
        normalized_lyrics = str(synced_lyrics or "").strip()
        if not normalized_lyrics:
            continue
        normalized_artist = str(artist).strip()
        normalized_title = str(title).strip()
        if not normalized_artist and not normalized_title:
            normalized_artist, normalized_title = _extract_artist_title_from_query_text(query_text=str(query_text).strip())
        if not normalized_title:
            normalized_title = _infer_title_from_search_term(search_term=search_term, artist=normalized_artist)
        logger.info("模块A V2-syncedlyrics 搜索命中，query=%s", search_term)
        return [
            {
                "status": "synced",
                "artist": normalized_artist,
                "title": normalized_title,
                "duration_seconds": 0.0,
                "plain_lyrics": "",
                "synced_lyrics": normalized_lyrics,
                "provider": "syncedlyrics",
                "provider_id": search_term,
                "instrumental": False,
                "error": "",
            }
        ][: max(1, int(limit))]
    logger.info(
        "模块A V2-syncedlyrics 未命中，artist=%s，title=%s，query=%s",
        str(artist).strip() or "<empty>",
        str(title).strip() or "<empty>",
        str(query_text).strip() or "<empty>",
    )
    return []


def search_syncedlyrics_candidates_by_provider(
    *,
    provider_name: str,
    query_text: str = "",
    artist: str = "",
    title: str = "",
    logger,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """
    功能说明：仅使用 syncedlyrics 的指定底层来源搜索同步歌词。
    参数说明：
    - provider_name: syncedlyrics 内部 provider 名称，如 Musixmatch。
    - query_text: 自由文本查询词。
    - artist: 艺人名。
    - title: 曲名。
    - logger: 日志对象。
    - limit: 最多返回候选数；当前单 provider 搜索至多返回一首。
    返回值：
    - list[dict[str, Any]]: 已标准化的同步歌词候选数组。
    异常说明：第三方库缺失或搜索异常时返回空数组。
    边界条件：若原始查询失败，会自动尝试净化尾部括注后的查询。
    """
    normalized_provider_name = str(provider_name).strip()
    if not normalized_provider_name:
        return []
    syncedlyrics_module = _import_syncedlyrics_module(logger=logger)
    if syncedlyrics_module is None:
        return []
    search_terms = _build_syncedlyrics_search_terms(
        query_text=str(query_text).strip(),
        artist=str(artist).strip(),
        title=str(title).strip(),
    )
    for search_term in search_terms:
        try:
            synced_lyrics = syncedlyrics_module.search(
                search_term,
                synced_only=True,
                providers=[normalized_provider_name],
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "模块A V2-syncedlyrics:%s 搜索失败，query=%s，错误=%s",
                normalized_provider_name,
                search_term,
                error,
            )
            continue
        normalized_lyrics = str(synced_lyrics or "").strip()
        if not normalized_lyrics:
            continue
        normalized_artist = str(artist).strip()
        normalized_title = str(title).strip()
        if not normalized_artist and not normalized_title:
            normalized_artist, normalized_title = _extract_artist_title_from_query_text(query_text=str(query_text).strip())
        if not normalized_title:
            normalized_title = _infer_title_from_search_term(search_term=search_term, artist=normalized_artist)
        provider_label = f"syncedlyrics:{normalized_provider_name}"
        logger.info("模块A V2-syncedlyrics:%s 搜索命中，query=%s", normalized_provider_name, search_term)
        return [
            {
                "status": "synced",
                "artist": normalized_artist,
                "title": normalized_title,
                "duration_seconds": 0.0,
                "plain_lyrics": "",
                "synced_lyrics": normalized_lyrics,
                "provider": provider_label,
                "provider_id": f"{normalized_provider_name}:{search_term}",
                "instrumental": False,
                "error": "",
            }
        ][: max(1, int(limit))]
    logger.info(
        "模块A V2-syncedlyrics:%s 未命中，artist=%s，title=%s，query=%s",
        normalized_provider_name,
        str(artist).strip() or "<empty>",
        str(title).strip() or "<empty>",
        str(query_text).strip() or "<empty>",
    )
    return []


def _import_syncedlyrics_module(logger) -> Any | None:
    """
    功能说明：延迟导入 syncedlyrics，避免在未安装依赖时影响主链启动。
    参数说明：
    - logger: 日志对象。
    返回值：
    - Any | None: 导入成功返回模块对象，否则返回 None。
    异常说明：无；导入失败时写日志并回退。
    边界条件：仅在实际调用聚合搜索时尝试导入。
    """
    try:
        # 第三方库：用于聚合同步歌词搜索
        import syncedlyrics
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-syncedlyrics 依赖不可用，错误=%s", error)
        return None
    return syncedlyrics


def _build_syncedlyrics_search_terms(query_text: str, artist: str, title: str) -> list[str]:
    """
    功能说明：构造 syncedlyrics 搜索词尝试序列。
    参数说明：
    - query_text: 自由文本查询词。
    - artist: 艺人名。
    - title: 曲名。
    返回值：
    - list[str]: 去重后的搜索词数组。
    异常说明：无。
    边界条件：原始词优先，净化掉尾部括注的词作为次级兜底。
    """
    normalized_query_text = str(query_text).strip()
    normalized_artist = str(artist).strip()
    normalized_title = str(title).strip()
    search_terms: list[str] = []
    if normalized_query_text:
        search_terms.append(normalized_query_text)
        stripped_query_text = _strip_trailing_bracket_note(normalized_query_text)
        if stripped_query_text and stripped_query_text != normalized_query_text:
            search_terms.append(stripped_query_text)
    if normalized_title:
        if normalized_artist:
            search_terms.append(f"{normalized_title} {normalized_artist}".strip())
        search_terms.append(normalized_title)
        stripped_title = _strip_trailing_bracket_note(normalized_title)
        if stripped_title and stripped_title != normalized_title:
            if normalized_artist:
                search_terms.append(f"{stripped_title} {normalized_artist}".strip())
            search_terms.append(stripped_title)
    return _dedupe_search_terms(search_terms)


def _strip_trailing_bracket_note(text: str) -> str:
    """
    功能说明：去除标题尾部括注说明，减少翻译备注对搜索命中的干扰。
    参数说明：
    - text: 原始文本。
    返回值：
    - str: 清理后的文本。
    异常说明：无。
    边界条件：仅移除尾部括注，不改动中间正文。
    """
    normalized_text = str(text).strip()
    if not normalized_text:
        return ""
    return TRAILING_BRACKET_NOTE_PATTERN.sub("", normalized_text).strip()


def _extract_artist_title_from_query_text(query_text: str) -> tuple[str, str]:
    """
    功能说明：从自由文本中尝试拆出歌手与歌名。
    参数说明：
    - query_text: 用户输入的原始搜歌文本。
    返回值：
    - tuple[str, str]: `(artist, title)`；无法识别时返回空字符串元组。
    异常说明：无。
    边界条件：仅在常见分隔符两侧均有内容时视为可拆分。
    """
    normalized_query_text = str(query_text).strip()
    split_separators = [" - ", " -", "- ", "-", " / ", "/", " | ", "|", "：", ":"]
    for separator in split_separators:
        if separator not in normalized_query_text:
            continue
        artist_part, title_part = normalized_query_text.split(separator, 1)
        normalized_artist = str(artist_part).strip()
        normalized_title = str(title_part).strip()
        if normalized_artist and normalized_title:
            return normalized_artist, normalized_title
    return "", ""


def _infer_title_from_search_term(search_term: str, artist: str) -> str:
    """
    功能说明：在无结构化标题时，根据最终搜索词推断用于展示的标题。
    参数说明：
    - search_term: 实际用于搜索的词。
    - artist: 已识别艺人名。
    返回值：
    - str: 供前端显示的标题文本。
    异常说明：无。
    边界条件：若搜索词尾部包含艺人名，则仅截取前半段作为标题。
    """
    normalized_search_term = str(search_term).strip()
    normalized_artist = str(artist).strip()
    if normalized_artist and normalized_search_term.endswith(normalized_artist):
        title_text = normalized_search_term[: -len(normalized_artist)].strip()
        if title_text:
            return title_text
    return normalized_search_term


def _dedupe_search_terms(items: list[str]) -> list[str]:
    """
    功能说明：对搜索词序列去重并裁剪空白项。
    参数说明：
    - items: 原始搜索词数组。
    返回值：
    - list[str]: 去重后的搜索词数组。
    异常说明：无。
    边界条件：大小写无关去重，保留首次出现顺序。
    """
    normalized_items: list[str] = []
    seen_items: set[str] = set()
    for item in items:
        normalized_item = str(item).strip()
        if not normalized_item:
            continue
        dedupe_key = normalized_item.lower()
        if dedupe_key in seen_items:
            continue
        seen_items.add(dedupe_key)
        normalized_items.append(normalized_item)
    return normalized_items
