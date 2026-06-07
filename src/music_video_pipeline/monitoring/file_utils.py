"""
文件用途：纯 HTTP/文件工具函数，不依赖实例状态。
输入输出：提供 HTTP Range 解析、HTTP 响应构建、文本文件读取等无副作用工具。
依赖说明：仅依赖标准库。
维护说明：新增纯函数工具时在此登记。
"""

from http import HTTPStatus
from pathlib import Path
from typing import Any


def _parse_http_range(range_header: str, file_size: int) -> tuple[int, int] | str | None:
    """
    功能说明：解析浏览器发来的单区间 bytes Range 请求。
    参数说明：
    - range_header: Range 请求头原文。
    - file_size: 目标文件总字节数。
    返回值：
    - tuple[int, int] | str | None: 成功返回 `(start, end)`，无 Range 返回 None，非法返回 `"invalid"`。
    异常说明：无。
    边界条件：仅支持 `bytes=start-end` / `bytes=start-` / `bytes=-suffix` 三种单区间形式。
    """
    normalized = str(range_header or "").strip()
    if not normalized:
        return None
    if (not normalized.startswith("bytes=")) or ("," in normalized):
        return "invalid"
    raw_range = normalized[len("bytes="):].strip()
    if "-" not in raw_range:
        return "invalid"
    start_text, end_text = raw_range.split("-", 1)
    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return "invalid"
            start_pos = max(0, file_size - suffix_length)
            return start_pos, max(0, file_size - 1)
        start_pos = int(start_text)
        if start_pos < 0 or start_pos >= file_size:
            return "invalid"
        if end_text == "":
            return start_pos, max(0, file_size - 1)
        end_pos = int(end_text)
        if end_pos < start_pos:
            return "invalid"
        return start_pos, min(end_pos, max(0, file_size - 1))
    except (TypeError, ValueError):
        return "invalid"


def _build_http_response(
    status: HTTPStatus,
    content_type: str,
    body_text: str,
    extra_headers: list[tuple[str, str]] | None = None,
    body_bytes: bytes | None = None,
) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
    """
    功能说明：构造 websockets process_request 需要的HTTP响应三元组。
    参数说明：
    - status: HTTP状态码。
    - content_type: Content-Type 头。
    - body_text: 响应正文文本。
    - extra_headers: 额外响应头。
    - body_bytes: 可选原始字节正文；传入时优先于 body_text。
    返回值：
    - tuple[HTTPStatus, list[tuple[str, str]], bytes]: HTTP响应对象。
    异常说明：无。
    边界条件：body统一按UTF-8编码。
    """
    if body_bytes is None:
        body_bytes = body_text.encode("utf-8")
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body_bytes))),
        ("Cache-Control", "no-store"),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    return status, headers, body_bytes


def _build_text_file_asset(file_path: Path | None) -> dict[str, Any]:
    """
    功能说明：把文本文件包装为前端可直接展示的文本资产对象。
    参数说明：
    - file_path: 文本文件路径。
    返回值：
    - dict[str, Any]: 包含 available/path/content 的对象。
    异常说明：无；读取失败时统一返回 available=false。
    边界条件：内容仅按 UTF-8 读取。
    """
    if file_path is None or (not file_path.exists()) or (not file_path.is_file()):
        return {"available": False, "path": str(file_path) if file_path else "", "content": ""}
    try:
        content_text = file_path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return {"available": False, "path": str(file_path), "content": ""}
    return {"available": True, "path": str(file_path), "content": content_text}
