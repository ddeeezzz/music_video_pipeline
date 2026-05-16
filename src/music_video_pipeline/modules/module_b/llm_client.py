"""
文件用途：定义模块 B 的 LLM 调用接口。
核心流程：接收消息、发起模型请求并返回文本或详细响应。
输入输出：输入模型调用参数，输出文本结果或响应结构。
依赖说明：仅依赖标准库类型工具。
维护说明：接口签名应与上层角色调用保持一致。
"""

from dataclasses import dataclass
from pathlib import Path
from time import sleep
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
    ).content


def call_module_b_llm_chat_detailed(
    logger: Any,
    llm_config: Any,
    messages: list[dict[str, str]],
    project_root: Path,
    on_rate_limited: Any | None = None,
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
    request_retry_times = max(0, int(getattr(llm_config, "request_retry_times", 0)))
    timeout_seconds = float(getattr(llm_config, "timeout_seconds", 60.0))
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
    last_error: ModuleBLlmClientError | None = None

    for attempt_index in range(request_retry_times + 1):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
        except requests.RequestException as error:
            last_error = ModuleBLlmClientError(f"模块 B LLM 请求失败：{error}")
            if attempt_index >= request_retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue

        response_headers = {str(key): str(value) for key, value in response.headers.items()}
        response_text = str(response.text or "")

        if response.status_code == 429:
            retry_after_seconds = _parse_retry_after_seconds(response_headers.get("Retry-After"))
            rate_limit_error = ModuleBLlmRateLimitError(
                "模块 B LLM 命中限流。",
                retry_after_seconds=retry_after_seconds,
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            if on_rate_limited is not None:
                on_rate_limited(rate_limit_error)
            last_error = rate_limit_error
            if attempt_index >= request_retry_times:
                raise rate_limit_error
            sleep(max(0.5, retry_after_seconds or (1.0 + attempt_index * 0.8)))
            continue

        if response.status_code >= 400:
            last_error = ModuleBLlmClientError(
                f"模块 B LLM HTTP 请求失败：status={response.status_code}",
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            if attempt_index >= request_retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue

        try:
            response_json = response.json()
        except json.JSONDecodeError as error:
            last_error = ModuleBLlmClientError(
                f"模块 B LLM 返回的响应不是合法 JSON：{error}",
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            if attempt_index >= request_retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue

        content = _extract_chat_content(response_json)
        if not content:
            last_error = ModuleBLlmClientError(
                "模块 B LLM 返回空 content。",
                status_code=response.status_code,
                response_text=response_text,
                response_headers=response_headers,
            )
            if attempt_index >= request_retry_times:
                break
            sleep(0.5 * (attempt_index + 1))
            continue
        return ModuleBLlmChatResponse(
            content=content,
            response_headers=response_headers,
            status_code=int(response.status_code),
            response_json=response_json,
        )

    raise last_error or ModuleBLlmClientError("模块 B LLM 请求失败。")


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
    边界条件：仅在配置要求时添加 `response_format=json_object`。
    """
    payload: dict[str, Any] = {
        "model": str(getattr(llm_config, "model", "")).strip(),
        "messages": messages,
        "temperature": float(getattr(llm_config, "temperature", 0.3)),
        "top_p": float(getattr(llm_config, "top_p", 0.9)),
        "max_tokens": int(getattr(llm_config, "max_tokens", 350)),
    }
    if bool(getattr(llm_config, "use_response_format_json_object", False)):
        payload["response_format"] = {"type": "json_object"}
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




