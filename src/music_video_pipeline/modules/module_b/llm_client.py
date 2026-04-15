"""
文件用途：封装模块B真实LLM的SiliconFlow Chat Completions调用。
核心流程：读取密钥 -> 构建HTTP请求 -> 执行重试 -> 返回文本内容。
输入输出：输入模型参数与messages，输出模型message.content字符串。
依赖说明：依赖标准库 urllib/json/pathlib/time。
维护说明：仅负责传输与协议转换，不负责业务字段校验。
"""

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
        "max_tokens": int(llm_config.max_tokens),
    }
    if bool(llm_config.use_response_format_json_object):
        payload["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt_index in range(request_retry_times + 1):
        try:
            response_obj = _post_json(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            return _extract_message_content(response_obj=response_obj)
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


def _post_json(endpoint: str, api_key: str, payload: dict[str, object], timeout_seconds: float) -> dict:
    """
    功能说明：向目标接口发送JSON POST请求。
    参数说明：
    - endpoint: 接口地址。
    - api_key: 鉴权密钥。
    - payload: JSON请求体。
    - timeout_seconds: 超时时间（秒）。
    返回值：
    - dict: 响应JSON对象。
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
    except urllib_error.HTTPError as error:
        error_text = error.read().decode("utf-8", errors="replace") if hasattr(error, "read") else str(error)
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
    return response_obj


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
