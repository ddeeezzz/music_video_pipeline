"""
文件用途：定义模块 B 的 LLM 调用接口。
核心流程：接收消息、发起模型请求并返回文本或详细响应。
输入输出：输入模型调用参数，输出文本结果或响应结构。
依赖说明：仅依赖标准库类型工具。
维护说明：接口签名应与上层角色调用保持一致。
"""

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any
import json

import requests


class ModuleBLlmClientError(RuntimeError):
    """模块 B LLM 客户端异常。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str = "",
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.response_headers = dict(response_headers or {})


class ModuleBLlmRateLimitError(ModuleBLlmClientError):
    """模块 B LLM 限流异常。"""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
        response_text: str = "",
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            response_headers=response_headers,
        )
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ModuleBLlmChatResponse:
    """模块 B LLM 响应类型。"""

    content: str
    response_headers: dict[str, str]
    status_code: int
    response_json: dict[str, Any]


def call_module_b_llm_chat(
    logger: Any,
    llm_config: Any,
    messages: list[dict[str, str]],
    project_root: Path,
    on_stream_chunk: Any | None = None,
    on_retry_hint: Any | None = None,
) -> str:
    """
    功能说明：执行模块 B 的文本 LLM 调用。
    参数说明：
    - logger: 日志对象。
    - llm_config: LLM 配置对象。
    - messages: 对话消息数组。
    - project_root: 项目根目录。
    返回值：
    - str: 模型返回文本。
    异常说明：按具体实现定义。
    边界条件：消息结构应与目标模型接口保持兼容。
    """
    return call_module_b_llm_chat_detailed(
        logger=logger,
        llm_config=llm_config,
        messages=messages,
        project_root=project_root,
        on_stream_chunk=on_stream_chunk,
        on_retry_hint=on_retry_hint,
    ).content


def call_module_b_llm_chat_detailed(
    logger: Any,
    llm_config: Any,
    messages: list[dict[str, str]],
    project_root: Path,
    on_rate_limited: Any | None = None,
    on_stream_chunk: Any | None = None,
    on_retry_hint: Any | None = None,
) -> ModuleBLlmChatResponse:
    """
    功能说明：执行模块 B 的详细 LLM 调用。
    参数说明：
    - logger: 日志对象。
    - llm_config: LLM 配置对象。
    - messages: 对话消息数组。
    - project_root: 项目根目录。
    - on_rate_limited: 限流回调。
    返回值：
    - ModuleBLlmChatResponse: 模型详细响应。
    异常说明：按具体实现定义。
    边界条件：返回结构应包含文本与必要响应元信息。
    """
    retry_times = max(0, int(getattr(llm_config, "retry_times", 1)))
    timeout_seconds = float(getattr(llm_config, "timeout_seconds", 60.0))
    first_chunk_timeout_seconds = float(getattr(llm_config, "first_chunk_timeout_seconds", 10.0))
    endpoint = _build_chat_completions_url(base_url=str(getattr(llm_config, "base_url", "")).strip())
    api_key = _read_api_key(
        project_root=project_root,
        api_key_file=str(getattr(llm_config, "api_key_file", "")).strip(),
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_request_payload(llm_config=llm_config, messages=messages)
    use_stream = bool(payload.get("stream", False))
    last_error: ModuleBLlmClientError | None = None
    _original_messages = messages
    try:
        logger.info(
            "模块 B LLM 请求准备完成：provider=%s model=%s stream=%s timeout=%.1fs first_chunk_timeout=%.1fs retries=%s messages=%s response_format_json=%s",
            str(getattr(llm_config, "provider", "")).strip() or "-",
            payload.get("model", ""),
            use_stream,
            timeout_seconds,
            first_chunk_timeout_seconds,
            retry_times,
            len(messages),
            "response_format" in payload,
        )
    except Exception:  # noqa: BLE001
        pass

    for attempt_index in range(retry_times + 1):
        attempt_number = attempt_index + 1
        if attempt_index > 0 and on_retry_hint is not None and last_error is not None:
            try:
                hint = on_retry_hint(attempt_index, last_error)
            except Exception:  # noqa: BLE001
                hint = None
            if hint:
                messages = _inject_retry_hint(messages=_original_messages, hint=str(hint))
                payload = _build_request_payload(llm_config=llm_config, messages=messages)
        try:
            logger.info(
                "模块 B LLM 开始请求：attempt=%s/%s endpoint=%s stream=%s",
                attempt_number,
                retry_times + 1,
                endpoint,
                use_stream,
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            request_timeout: float | tuple[float, float] = timeout_seconds
            if use_stream:
                connect_timeout = min(10.0, first_chunk_timeout_seconds)
                request_timeout = (connect_timeout, first_chunk_timeout_seconds)
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=request_timeout,
                stream=use_stream,
            )
        except requests.RequestException as error:
            last_error = ModuleBLlmClientError(f"模块 B LLM 请求失败：{error}")
            try:
                logger.warning(
                    "模块 B LLM 请求异常：attempt=%s/%s error=%s",
                    attempt_number,
                    retry_times + 1,
                    error,
                )
            except Exception:  # noqa: BLE001
                pass
            if attempt_index >= retry_times:
                break
            try:
                logger.info("模块 B LLM 准备重试：next_attempt=%s/%s", attempt_number + 1, retry_times + 1)
            except Exception:  # noqa: BLE001
                pass
            sleep(0.5 * (attempt_index + 1))
            continue

        response_headers = {str(key): str(value) for key, value in response.headers.items()}
        response_text = ""
        try:
            logger.info(
                "模块 B LLM 已收到响应头：attempt=%s/%s status=%s content_type=%s",
                attempt_number,
                retry_times + 1,
                response.status_code,
                response_headers.get("Content-Type", ""),
            )
        except Exception:  # noqa: BLE001
            pass

        if response.status_code == 429:
            response_text = _read_response_text(response=response, use_stream=use_stream)
            retry_after_seconds = _parse_retry_after_seconds(response_headers.get("Retry-After"))
            rate_limit_error = ModuleBLlmRateLimitError(
                "模块 B LLM 命中限流。",
                retry_after_seconds=retry_after_seconds,
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            try:
                logger.warning(
                    "模块 B LLM 命中限流：attempt=%s/%s retry_after=%s response_excerpt=%s",
                    attempt_number,
                    retry_times + 1,
                    retry_after_seconds,
                    response_text[:240],
                )
            except Exception:  # noqa: BLE001
                pass
            if on_rate_limited is not None:
                on_rate_limited(rate_limit_error)
            last_error = rate_limit_error
            if attempt_index >= retry_times:
                raise rate_limit_error
            sleep(max(0.5, retry_after_seconds or (1.0 + attempt_index * 0.8)))
            continue

        if response.status_code >= 400:
            response_text = _read_response_text(response=response, use_stream=use_stream)
            last_error = ModuleBLlmClientError(
                f"模块 B LLM HTTP 请求失败：status={response.status_code}",
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            try:
                logger.warning(
                    "模块 B LLM HTTP 失败：attempt=%s/%s status=%s response_excerpt=%s",
                    attempt_number,
                    retry_times + 1,
                    response.status_code,
                    response_text[:240],
                )
            except Exception:  # noqa: BLE001
                pass
            if attempt_index >= retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue

        if use_stream:
            try:
                content, response_json = _read_streaming_chat_completion(
                    response=response,
                    logger=logger,
                    on_stream_chunk=on_stream_chunk,
                    attempt_number=attempt_number,
                    attempt_total=retry_times + 1,
                    first_chunk_timeout_seconds=first_chunk_timeout_seconds,
                )
            except ModuleBLlmClientError as error:
                last_error = ModuleBLlmClientError(
                    str(error),
                    status_code=response.status_code,
                    response_text=str(getattr(error, "response_text", "") or ""),
                    response_headers=response_headers,
                )
                try:
                    logger.warning(
                        "模块 B LLM 流式解析失败：attempt=%s/%s status=%s error=%s response_excerpt=%s",
                        attempt_number,
                        retry_times + 1,
                        response.status_code,
                        error,
                        str(getattr(error, "response_text", "") or "")[:240],
                    )
                except Exception:  # noqa: BLE001
                    pass
                if attempt_index >= retry_times:
                    break
                sleep(0.5 * (attempt_index + 1))
                continue
        else:
            response_text = _read_response_text(response=response, use_stream=use_stream)
            content = _extract_chat_content_from_text(response_text)
            response_json = {
                "object": "chat.completion.non_stream.aggregate",
                "choices": [{"message": {"content": content}}],
                "raw_text": response_text,
            }

        if not content:
            last_error = ModuleBLlmClientError(
                "模块 B LLM 返回空 content。",
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            try:
                logger.warning(
                    "模块 B LLM 返回空内容：attempt=%s/%s status=%s stream=%s",
                    attempt_number,
                    retry_times + 1,
                    response.status_code,
                    use_stream,
                )
            except Exception:  # noqa: BLE001
                pass
            if attempt_index >= retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue
        try:
            logger.info(
                "模块 B LLM 请求完成：attempt=%s/%s status=%s content_chars=%s stream=%s",
                attempt_number,
                retry_times + 1,
                response.status_code,
                len(content),
                use_stream,
            )
        except Exception:  # noqa: BLE001
            pass
        return ModuleBLlmChatResponse(
            content=content,
            response_headers=response_headers,
            status_code=int(response.status_code),
            response_json=response_json,
        )

    raise last_error or ModuleBLlmClientError("模块 B LLM 请求失败。")


def _inject_retry_hint(*, messages: list[dict[str, str]], hint: str) -> list[dict[str, str]]:
    """
    功能说明：构造重试消息，将提示拼入 user message 前面。
    参数说明：
    - messages: 原始消息列表。
    - hint: 要拼接的重试提示。
    返回值：
    - list[dict[str, str]]: 修改后的消息副本。
    异常说明：无。
    边界条件：若原始 user content 已含相同 hint 前缀则直接返回副本。
    """
    hint_prefix = f"## 重试要求\n{str(hint).strip()}\n\n"
    result: list[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "user":
            original_content = str(msg.get("content", ""))
            if original_content.startswith(hint_prefix):
                result.append(msg)
            else:
                result.append({"role": "user", "content": hint_prefix + original_content})
        else:
            result.append(dict(msg))
    return result


def _build_chat_completions_url(*, base_url: str) -> str:
    """
    功能说明：构造 OpenAI 兼容 Chat Completions 接口地址。
    参数说明：
    - base_url: 配置中的接口根地址。
    返回值：
    - str: 完整接口 URL。
    异常说明：
    - ModuleBLlmClientError: base_url 为空时抛出。
    边界条件：若已包含 `/chat/completions` 后缀则直接返回。
    """
    normalized_base_url = str(base_url or "").strip().rstrip("/")
    if not normalized_base_url:
        raise ModuleBLlmClientError("模块 B LLM base_url 不能为空。")
    if normalized_base_url.endswith("/chat/completions"):
        return normalized_base_url
    return f"{normalized_base_url}/chat/completions"


def _read_api_key(*, project_root: Path, api_key_file: str) -> str:
    """
    功能说明：读取模块 B LLM API Key。
    参数说明：
    - project_root: 项目根目录。
    - api_key_file: API Key 文件路径。
    返回值：
    - str: 非空 API Key。
    异常说明：
    - ModuleBLlmClientError: 路径为空、文件不存在或内容为空时抛出。
    边界条件：相对路径统一相对项目根目录解析。
    """
    normalized_path = Path(str(api_key_file).strip()).expanduser()
    if not str(normalized_path):
        raise ModuleBLlmClientError("模块 B LLM api_key_file 不能为空。")
    if not normalized_path.is_absolute():
        normalized_path = (project_root / normalized_path).resolve()
    if not normalized_path.exists():
        raise ModuleBLlmClientError(f"模块 B LLM API Key 文件不存在：{normalized_path}")
    api_key = normalized_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ModuleBLlmClientError(f"模块 B LLM API Key 文件为空：{normalized_path}")
    return api_key


def _build_request_payload(llm_config: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    """
    功能说明：构造 OpenAI 兼容 Chat Completions 请求载荷。
    参数说明：
    - llm_config: 模块 B LLM 配置对象。
    - messages: 标准消息数组。
    返回值：
    - dict[str, Any]: 请求载荷。
    异常说明：无。
    边界条件：模块 B v2 统一按自由文本消费，不再强制要求 `response_format=json_object`。
    """
    payload: dict[str, Any] = {
        "model": str(getattr(llm_config, "model", "")).strip(),
        "messages": messages,
        "temperature": float(getattr(llm_config, "temperature", 0.3)),
        "top_p": float(getattr(llm_config, "top_p", 0.9)),
        "stream": bool(getattr(llm_config, "stream", False)),
    }
    enable_thinking_value = getattr(llm_config, "enable_thinking", None)
    if enable_thinking_value is not None:
        payload["enable_thinking"] = bool(enable_thinking_value)
    return payload


def _extract_chat_content(response_json: dict[str, Any]) -> str:
    """
    功能说明：从 Chat Completions 响应中提取首个 message.content。
    参数说明：
    - response_json: 响应 JSON。
    返回值：
    - str: 提取到的 content 文本；若缺失则返回空字符串。
    异常说明：无。
    边界条件：仅消费第一条 choice。
    """
    choices = response_json.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    return str(content).strip()


def _extract_chat_content_from_text(response_text: str) -> str:
    """
    功能说明：从普通响应文本中尽力提取 content，提取不到时回退原文。
    参数说明：
    - response_text: HTTP 响应体文本。
    返回值：
    - str: 提取出的文本内容。
    异常说明：无。
    边界条件：不再要求响应必须是合法 JSON。
    """
    normalized_text = str(response_text or "").strip()
    if not normalized_text:
        return ""
    try:
        response_json = json.loads(normalized_text)
    except Exception:  # noqa: BLE001
        return normalized_text
    if isinstance(response_json, dict):
        content = _extract_chat_content(response_json)
        if content:
            return content
    return normalized_text


def _parse_retry_after_seconds(value: str | None) -> float | None:
    """
    功能说明：将 Retry-After 响应头解析为秒数。
    参数说明：
    - value: 响应头原始值。
    返回值：
    - float | None: 成功则返回秒数，否则返回 None。
    异常说明：无。
    边界条件：仅支持数值秒数格式。
    """
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_response_text(*, response: Any, use_stream: bool) -> str:
    """
    功能说明：读取响应文本，用于错误日志与异常信息。
    参数说明：
    - response: requests 响应对象。
    - use_stream: 当前是否为流式请求。
    返回值：
    - str: 响应文本。
    异常说明：无；读取失败时回退为空字符串。
    边界条件：流式模式优先尝试读取 text，避免提前消费正常内容流。
    """
    try:
        return str(response.text or "")
    except Exception:  # noqa: BLE001
        if use_stream:
            return ""
        return ""


def _read_streaming_chat_completion(
    *,
    response: Any,
    logger: Any,
    on_stream_chunk: Any | None = None,
    attempt_number: int = 1,
    attempt_total: int = 1,
    first_chunk_timeout_seconds: float = 10.0,
) -> tuple[str, dict[str, Any]]:
    """
    功能说明：解析流式 Chat Completions 响应，拼接 delta.content 为完整文本。
    参数说明：
    - response: requests 流式响应对象。
    - logger: 日志对象。
    - first_chunk_timeout_seconds: 首个内容 chunk 超时秒数。
    返回值：
    - tuple[str, dict[str, Any]]: (完整 content, 汇总后的调试结构)。
    异常说明：首 chunk 超时抛 ModuleBLlmClientError。
    边界条件：兼容 `data: {...}` SSE 行与 `[DONE]` 终止标记。
    """
    content_parts: list[str] = []
    chunk_payloads: list[str] = []
    raw_lines: list[str] = []
    first_data_line_logged = False
    first_chunk_logged = False
    data_line_count = 0
    content_chunk_count = 0
    reasoning_chunk_count = 0
    reasoning_parts: list[str] = []
    first_reasoning_logged = False
    stream_start = monotonic()
    deadline = stream_start + max(0.0, float(first_chunk_timeout_seconds))

    try:
        logger.info(
            "模块 B LLM 开始流式读取：attempt=%s/%s first_chunk_timeout=%.1fs",
            attempt_number,
            attempt_total,
            first_chunk_timeout_seconds,
        )
    except Exception:  # noqa: BLE001
        pass

    for raw_line in response.iter_lines(decode_unicode=False):
        line = _decode_sse_line(raw_line).strip()
        if not line:
            continue
        raw_lines.append(line)
        if line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text:
            continue
        data_line_count += 1
        if not first_data_line_logged:
            elapsed_s = monotonic() - stream_start
            try:
                logger.info(
                    "模块 B LLM 收到首个 SSE data 行：attempt=%s/%s 耗时=%.2fs",
                    attempt_number,
                    attempt_total,
                    elapsed_s,
                )
            except Exception:  # noqa: BLE001
                pass
            first_data_line_logged = True
        if data_text == "[DONE]":
            break
        chunk_payloads.append(data_text)
        delta_text = _extract_stream_delta_content_from_data_text(data_text)
        if delta_text:
            content_parts.append(delta_text)
            content_chunk_count += 1
            if on_stream_chunk is not None:
                on_stream_chunk("".join(content_parts), delta_text)
            if not first_chunk_logged:
                elapsed_s = monotonic() - stream_start
                try:
                    logger.info(
                        "模块 B LLM 已收到首个流式内容 chunk：attempt=%s/%s 耗时=%.2fs delta_chars=%s total_chars=%s",
                        attempt_number,
                        attempt_total,
                        elapsed_s,
                        len(delta_text),
                        len("".join(content_parts)),
                    )
                except Exception:  # noqa: BLE001
                    pass
                first_chunk_logged = True
        else:
            reasoning_chunk = _extract_stream_reasoning_content_from_data_text(data_text)
            if reasoning_chunk:
                reasoning_parts.append(reasoning_chunk)
                reasoning_chunk_count += 1
                if on_stream_chunk is not None:
                    on_stream_chunk(
                        "[思考] " + "".join(reasoning_parts),
                        "[思考] " + reasoning_chunk,
                    )
                if not first_reasoning_logged:
                    first_reasoning_logged = True
                    try:
                        logger.info(
                            "模块 B LLM 收到首个 reasoning chunk：attempt=%s/%s 耗时=%.2fs",
                            attempt_number,
                            attempt_total,
                            monotonic() - stream_start,
                        )
                    except Exception:  # noqa: BLE001
                        pass
            if not first_chunk_logged and monotonic() > deadline:
                elapsed_s = monotonic() - stream_start
                raise ModuleBLlmClientError(
                    f"首 chunk 超时：{elapsed_s:.1f}s 内未收到任何内容 chunk，"
                    f"已收到 {data_line_count} 条 SSE data 行，"
                    f"其中 reasoning_content {reasoning_chunk_count} 条、"
                    f"内容 chunk {content_chunk_count} 条"
                )

    content = "".join(content_parts).strip()
    try:
        logger.info(
            "模块 B LLM 流式读取结束：attempt=%s/%s data_lines=%s content_chunks=%s reasoning_chunks=%s content_chars=%s has_first_chunk=%s",
            attempt_number,
            attempt_total,
            data_line_count,
            content_chunk_count,
            reasoning_chunk_count,
            len(content),
            first_chunk_logged,
        )
    except Exception:  # noqa: BLE001
        pass
    synthetic_response_json: dict[str, Any] = {
        "object": "chat.completion.stream.aggregate",
        "choices": [{"message": {"content": content}}],
        "chunks": chunk_payloads,
    }
    return content, synthetic_response_json


def _decode_sse_line(raw_line: Any) -> str:
    """
    功能说明：将 SSE 原始行按字节显式解码为文本，优先使用 UTF-8。
    参数说明：
    - raw_line: requests.iter_lines 返回的单行内容，可能是 bytes 或 str。
    返回值：
    - str: 解码后的文本行；空值返回空字符串。
    异常说明：无。
    边界条件：UTF-8 解码失败时回退响应声明编码或受控替换，避免中文被错误双重转码。
    """
    if raw_line is None:
        return ""
    if isinstance(raw_line, bytes):
        try:
            return raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return raw_line.decode("utf-8", errors="replace")
    return str(raw_line)


def _extract_stream_delta_content(chunk_json: dict[str, Any]) -> str:
    """
    功能说明：从单个流式 chunk 中提取可拼接的文本内容。
    参数说明：
    - chunk_json: 单个 SSE `data:` 片段 JSON。
    返回值：
    - str: 当前 chunk 中的内容增量；缺失时返回空字符串。
    异常说明：无。
    边界条件：兼容 `delta.content`、`message.content` 以及数组型 content 段。
    """
    choices = chunk_json.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = first_choice.get("delta", {})
    if isinstance(delta, dict):
        delta_content = _normalize_stream_content_fragment(delta.get("content"))
        if delta_content:
            return delta_content
    message = first_choice.get("message", {})
    if isinstance(message, dict):
        message_content = _normalize_stream_content_fragment(message.get("content"))
        if message_content:
            return message_content
    return ""


def _extract_stream_delta_content_from_data_text(data_text: str) -> str:
    """
    功能说明：从 SSE data 原文中尽力提取 content，不要求整段是合法 JSON。
    参数说明：
    - data_text: 单个 SSE `data:` 原文。
    返回值：
    - str: 当前 chunk 中的文本增量；提取不到时返回空字符串。
    异常说明：无。
    边界条件：字段截断、夹带 reasoning_content 或非 JSON 文本时均允许静默跳过。
    """
    normalized_text = str(data_text or "").strip()
    if not normalized_text:
        return ""
    try:
        chunk_json = json.loads(normalized_text)
    except Exception:  # noqa: BLE001
        return _extract_first_json_string_field(normalized_text, "content")
    if isinstance(chunk_json, dict):
        return _extract_stream_delta_content(chunk_json)
    return ""


def _extract_stream_reasoning_content_from_data_text(data_text: str) -> str:
    """
    功能说明：从 SSE data 行中提取 reasoning_content（思考链），用于调试展示。
    参数说明：
    - data_text: 单个 SSE `data:` 原文。
    返回值：
    - str: reasoning_content 文本；提取不到时返回空字符串。
    异常说明：无。
    边界条件：仅用于调试，不影响主流程。
    """
    normalized_text = str(data_text or "").strip()
    if not normalized_text:
        return ""
    try:
        chunk_json = json.loads(normalized_text)
    except Exception:  # noqa: BLE001
        return _extract_first_json_string_field(normalized_text, "reasoning_content")
    if isinstance(chunk_json, dict):
        choices = chunk_json.get("choices", [])
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                delta = first_choice.get("delta", {})
                if isinstance(delta, dict):
                    rc = delta.get("reasoning_content", "")
                    if isinstance(rc, str) and rc.strip():
                        return rc
    return ""


def _extract_first_json_string_field(source_text: str, field_name: str) -> str:
    """
    功能说明：从近似 JSON 文本中提取首个字符串字段值。
    参数说明：
    - source_text: 原始文本。
    - field_name: 字段名。
    返回值：
    - str: 提取出的字符串；不存在或不完整时返回空字符串。
    异常说明：无。
    边界条件：仅处理 `"field":"value"` 形态。
    """
    pattern = f'"{field_name}"'
    field_index = source_text.find(pattern)
    if field_index < 0:
        return ""
    cursor = source_text.find(":", field_index + len(pattern))
    if cursor < 0:
        return ""
    cursor += 1
    while cursor < len(source_text) and source_text[cursor] in " \t\r\n":
        cursor += 1
    if cursor >= len(source_text) or source_text[cursor] != '"':
        return ""
    cursor += 1
    buffer: list[str] = []
    escaping = False
    while cursor < len(source_text):
        char = source_text[cursor]
        if escaping:
            buffer.append("\\" + char)
            escaping = False
            cursor += 1
            continue
        if char == "\\":
            escaping = True
            cursor += 1
            continue
        if char == '"':
            try:
                return json.loads(f'"{"".join(buffer)}"')
            except Exception:  # noqa: BLE001
                return "".join(buffer)
        buffer.append(char)
        cursor += 1
    return ""


def _normalize_stream_content_fragment(value: Any) -> str:
    """
    功能说明：把流式片段中的 content 字段归一化为纯文本。
    参数说明：
    - value: content 原始值，可能是字符串或 OpenAI 风格 content part 数组。
    返回值：
    - str: 可直接拼接的文本。
    异常说明：无。
    边界条件：未知结构回退为空字符串。
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip() == "text" and isinstance(item.get("text"), str):
            parts.append(str(item.get("text")))
    return "".join(parts)




