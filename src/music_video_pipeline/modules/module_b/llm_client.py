"""
文件用途：封装模块B真实LLM的SiliconFlow Chat Completions调用。
核心流程：读取密钥 -> 构建HTTP请求 -> 执行重试 -> 返回文本内容。
输入输出：输入模型参数与messages，输出模型message.content字符串。
依赖说明：依赖标准库 urllib/json/pathlib/time。
维护说明：仅负责传输与协议转换，不负责业务字段校验。
"""

# 标准库：用于声明轻量返回结构。
from dataclasses import dataclass
# 标准库：用于回调类型提示。
from collections.abc import Callable
# 标准库：用于JSON序列化与反序列化
import json
# 标准库：用于日志输出
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于重试退避
import time
# 标准库：用于HTTP请求
from urllib import error as urllib_error
from urllib import request as urllib_request

# 项目内模块：模块B LLM配置类型
from music_video_pipeline.config import ModuleBLlmConfig


class ModuleBLlmClientError(RuntimeError):
    """模块B LLM 客户端异常。"""


class ModuleBLlmRateLimitError(ModuleBLlmClientError):
    """模块B LLM 限流异常。"""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.response_headers = dict(response_headers or {})


@dataclass(frozen=True)
class ModuleBLlmChatResponse:
    """
    功能说明：封装模块B LLM单次成功调用的文本与响应头。
    参数说明：
    - content: 模型返回文本。
    - response_headers: 响应头（统一转为小写键）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：仅承载一次最终成功请求，不含业务解析结果。
    """

    content: str
    response_headers: dict[str, str]


def call_module_b_llm_chat(
    logger: logging.Logger,
    llm_config: ModuleBLlmConfig,
    messages: list[dict[str, str]],
    project_root: Path,
) -> str:
    """
    功能说明：调用SiliconFlow Chat Completions并返回文本内容。
    参数说明：
    - logger: 日志对象。
    - llm_config: 模块B LLM配置。
    - messages: OpenAI兼容消息数组。
    - project_root: 项目根目录，用于解析相对密钥路径。
    返回值：
    - str: 模型返回的 message.content。
    异常说明：
    - ModuleBLlmClientError: 网络失败、鉴权失败、返回结构异常时抛出。
    边界条件：请求重试次数由 llm_config.request_retry_times 控制。
    """
    return call_module_b_llm_chat_detailed(
        logger=logger,
        llm_config=llm_config,
        messages=messages,
        project_root=project_root,
    ).content


def call_module_b_llm_chat_detailed(
    logger: logging.Logger,
    llm_config: ModuleBLlmConfig,
    messages: list[dict[str, str]],
    project_root: Path,
    on_rate_limited: Callable[[ModuleBLlmRateLimitError], None] | None = None,
) -> ModuleBLlmChatResponse:
    """
    功能说明：调用SiliconFlow Chat Completions并返回文本与响应头。
    参数说明：
    - logger: 日志对象。
    - llm_config: 模块B LLM配置。
    - messages: OpenAI兼容消息数组。
    - project_root: 项目根目录，用于解析相对密钥路径。
    - on_rate_limited: 可选，命中 429 时的即时回调。
    返回值：
    - ModuleBLlmChatResponse: 包含文本与响应头的结果对象。
    异常说明：
    - ModuleBLlmClientError: 网络失败、鉴权失败、返回结构异常时抛出。
    边界条件：请求重试次数由 llm_config.request_retry_times 控制。
    """
    normalized_provider = str(llm_config.provider).strip().lower()
    if normalized_provider != "siliconflow":
        raise ModuleBLlmClientError(f"模块B LLM provider 不支持：{llm_config.provider}")

    api_key = _read_api_key(
        api_key_file=str(llm_config.api_key_file),
        project_root=project_root,
    )
    endpoint = _build_chat_endpoint(base_url=str(llm_config.base_url))
    request_retry_times = max(0, int(llm_config.request_retry_times))
    timeout_seconds = max(1.0, float(llm_config.timeout_seconds))

    payload: dict[str, object] = {
        "model": str(llm_config.model),
        "messages": messages,
        "temperature": float(llm_config.temperature),
        "top_p": float(llm_config.top_p),
        "stream": False,
        "enable_thinking": False,
    }
    max_tokens = int(llm_config.max_tokens)
    # 配置约定：当 max_tokens<=0 时视为关闭上限，不向接口显式透传。
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if bool(llm_config.use_response_format_json_object):
        payload["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt_index in range(request_retry_times + 1):
        try:
            response_obj, response_headers = _post_json(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            return ModuleBLlmChatResponse(
                content=_extract_message_content(response_obj=response_obj),
                response_headers=response_headers,
            )
        except ModuleBLlmRateLimitError as error:
            last_error = error
            if on_rate_limited is not None:
                on_rate_limited(error)
                raise
            if attempt_index >= request_retry_times:
                break
            sleep_seconds = _resolve_retry_after_seconds(error=error, attempt_index=attempt_index)
            logger.warning(
                "模块B LLM命中限流，准备退避重试，attempt=%s/%s，sleep=%.1fs，错误=%s",
                attempt_index + 1,
                request_retry_times + 1,
                sleep_seconds,
                error,
            )
            time.sleep(sleep_seconds)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt_index >= request_retry_times:
                break
            sleep_seconds = 0.4 * (attempt_index + 1)
            logger.warning(
                "模块B LLM请求失败，准备重试，attempt=%s/%s，sleep=%.1fs，错误=%s",
                attempt_index + 1,
                request_retry_times + 1,
                sleep_seconds,
                error,
            )
            time.sleep(sleep_seconds)

    raise ModuleBLlmClientError(f"模块B LLM请求失败：{last_error}")


def _build_chat_endpoint(base_url: str) -> str:
    """
    功能说明：拼接Chat Completions接口地址。
    参数说明：
    - base_url: 配置中的基础URL。
    返回值：
    - str: 完整接口地址。
    异常说明：无。
    边界条件：自动去除重复斜杠。
    """
    normalized_base_url = str(base_url).strip().rstrip("/")
    if not normalized_base_url:
        normalized_base_url = "https://api.siliconflow.cn/v1"
    return f"{normalized_base_url}/chat/completions"


def _read_api_key(api_key_file: str, project_root: Path) -> str:
    """
    功能说明：读取API Key文本文件首行。
    参数说明：
    - api_key_file: 密钥文件路径（可相对）。
    - project_root: 项目根目录。
    返回值：
    - str: API Key。
    异常说明：
    - ModuleBLlmClientError: 文件不存在或内容为空时抛出。
    边界条件：相对路径默认相对于 project_root。
    """
    key_path = Path(str(api_key_file).strip())
    if not key_path.is_absolute():
        key_path = (project_root / key_path).resolve()

    if not key_path.exists():
        raise ModuleBLlmClientError(f"模块B LLM密钥文件不存在：{key_path}")

    key_text = key_path.read_text(encoding="utf-8").strip()
    if not key_text:
        raise ModuleBLlmClientError(f"模块B LLM密钥文件为空：{key_path}")
    return key_text


def _post_json(
    endpoint: str,
    api_key: str,
    payload: dict[str, object],
    timeout_seconds: float,
) -> tuple[dict, dict[str, str]]:
    """
    功能说明：向目标接口发送JSON POST请求。
    参数说明：
    - endpoint: 接口地址。
    - api_key: 鉴权密钥。
    - payload: JSON请求体。
    - timeout_seconds: 超时时间（秒）。
    返回值：
    - tuple[dict, dict[str, str]]: 响应JSON对象与响应头。
    异常说明：
    - ModuleBLlmClientError: 网络、HTTP或JSON解析失败时抛出。
    边界条件：响应体非JSON时按错误处理。
    """
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = urllib_request.Request(
        url=endpoint,
        data=body_bytes,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib_request.urlopen(http_request, timeout=float(timeout_seconds)) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            response_headers = _normalize_headers(dict(response.headers.items()))
    except urllib_error.HTTPError as error:
        error_text = error.read().decode("utf-8", errors="replace") if hasattr(error, "read") else str(error)
        error_headers = _normalize_headers(dict(getattr(error, "headers", {}).items()))
        if int(getattr(error, "code", 0) or 0) == 429:
            raise ModuleBLlmRateLimitError(
                f"模块B LLM HTTP错误：status={getattr(error, 'code', 'unknown')}，body={error_text}",
                retry_after_seconds=_parse_retry_after_seconds(error_headers),
                response_headers=error_headers,
            ) from error
        raise ModuleBLlmClientError(
            f"模块B LLM HTTP错误：status={getattr(error, 'code', 'unknown')}，body={error_text}"
        ) from error
    except urllib_error.URLError as error:
        raise ModuleBLlmClientError(f"模块B LLM网络错误：{error}") from error

    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ModuleBLlmClientError(f"模块B LLM响应非JSON：{error}") from error

    if not isinstance(response_obj, dict):
        raise ModuleBLlmClientError("模块B LLM响应结构非法：顶层不是对象。")
    return response_obj, response_headers


def _normalize_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    功能说明：将响应头标准化为小写键的普通字典。
    参数说明：
    - headers: 原始响应头字典。
    返回值：
    - dict[str, str]: 标准化后的响应头。
    异常说明：无。
    边界条件：非字符串键值会被安全转为字符串。
    """
    return {
        str(key).strip().lower(): str(value).strip()
        for key, value in headers.items()
        if str(key).strip()
    }


def _parse_retry_after_seconds(headers: dict[str, str]) -> float | None:
    """
    功能说明：从响应头解析 Retry-After 秒数。
    参数说明：
    - headers: 已标准化响应头。
    返回值：
    - float | None: 可用秒数；解析失败时返回空。
    异常说明：无。
    边界条件：当前仅支持数值秒，不解析 HTTP-date 格式。
    """
    raw_value = str(headers.get("retry-after", "")).strip()
    if not raw_value:
        return None
    try:
        parsed_value = float(raw_value)
    except ValueError:
        return None
    if parsed_value <= 0:
        return None
    return parsed_value


def _resolve_retry_after_seconds(error: ModuleBLlmRateLimitError, attempt_index: int) -> float:
    """
    功能说明：为 429 重试计算退避秒数。
    参数说明：
    - error: 限流异常对象。
    - attempt_index: 当前重试序号（0基）。
    返回值：
    - float: 退避秒数。
    异常说明：无。
    边界条件：优先遵从 Retry-After，其次使用温和线性退避。
    """
    retry_after_seconds = error.retry_after_seconds
    if isinstance(retry_after_seconds, (int, float)) and float(retry_after_seconds) > 0:
        return max(0.5, float(retry_after_seconds))
    return 1.0 + 0.8 * float(attempt_index)


def _extract_message_content(response_obj: dict) -> str:
    """
    功能说明：从OpenAI兼容响应中提取 message.content。
    参数说明：
    - response_obj: 响应JSON对象。
    返回值：
    - str: 内容文本。
    异常说明：
    - ModuleBLlmClientError: choices/message/content 缺失或类型错误时抛出。
    边界条件：仅读取第一条 choice。
    """
    choices = response_obj.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ModuleBLlmClientError("模块B LLM响应缺失 choices。")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ModuleBLlmClientError("模块B LLM响应结构非法：choices[0] 不是对象。")

    message_obj = first_choice.get("message", {})
    if not isinstance(message_obj, dict):
        raise ModuleBLlmClientError("模块B LLM响应结构非法：message 不是对象。")

    content = message_obj.get("content", "")
    if not isinstance(content, str):
        raise ModuleBLlmClientError("模块B LLM响应结构非法：content 不是字符串。")
    normalized = content.strip()
    if not normalized:
        raise ModuleBLlmClientError("模块B LLM响应内容为空。")
    return normalized
