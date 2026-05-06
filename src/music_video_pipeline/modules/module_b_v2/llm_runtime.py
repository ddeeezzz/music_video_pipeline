"""
文件用途：提供模块B v2 多角色链路的通用 LLM Markdown 调用封装。
核心流程：构建 system/user messages -> 调用兼容接口 -> 返回 Markdown 文本 -> 按重试策略回补提示。
输入输出：输入角色名、系统提示词与 Markdown 用户提示，输出模型返回的 Markdown 文本。
依赖说明：依赖标准库 dataclasses/pathlib/logging 与旧模块B的 LLM client。
维护说明：本文件只负责通用传输与 prompt 落盘，不承载角色级 Markdown 解析。
"""

# 标准库：用于 dataclass 局部替换。
from dataclasses import replace
# 标准库：用于线程条件变量协调并发门禁。
from threading import Condition
# 标准库：用于日志类型提示。
import logging
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于文件名安全化。
import re
# 标准库：用于单调时钟。
from time import monotonic, sleep
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：模块B LLM 配置。
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：通用目录工具。
from music_video_pipeline.io_utils import ensure_dir
# 项目内模块：复用旧模块B的 OpenAI 兼容 HTTP client。
from music_video_pipeline.modules.module_b.llm_client import (
    ModuleBLlmChatResponse,
    ModuleBLlmClientError,
    ModuleBLlmRateLimitError,
    call_module_b_llm_chat_detailed,
)


class ModuleBV2LlmRuntimeError(RuntimeError):
    """模块B v2 LLM 运行时异常。"""


class _AdaptiveLlmConcurrencyGate:
    """
    功能说明：维护模块B v2 全角色共享的自适应并发门禁。
    参数说明：
    - logger: 日志对象。
    返回值：不适用。
    异常说明：不适用。
    边界条件：默认不限流；仅在观测到 429 后收缩并发并按成功结果缓慢恢复。
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._condition = Condition()
        self._inflight = 0
        self._limit: int | None = None
        self._backoff_until = 0.0
        self._success_streak = 0

    def acquire(self, *, role_name: str) -> None:
        """
        功能说明：进入一次真实 LLM 请求前申请全局并发许可。
        参数说明：
        - role_name: 当前角色名，用于日志定位。
        返回值：无。
        异常说明：无。
        边界条件：若尚未触发过限流，默认不主动限制并发。
        """
        del role_name
        with self._condition:
            while True:
                now = monotonic()
                if now < self._backoff_until:
                    self._condition.wait(timeout=max(0.05, self._backoff_until - now))
                    continue
                if self._limit is None or self._inflight < self._limit:
                    self._inflight += 1
                    return
                self._condition.wait(timeout=0.10)

    def release_success(self, *, role_name: str, response_headers: dict[str, str]) -> None:
        """
        功能说明：在一次请求成功后释放并发许可，并尝试恢复被压缩的窗口。
        参数说明：
        - role_name: 当前角色名。
        - response_headers: 成功响应头。
        返回值：无。
        异常说明：无。
        边界条件：仅在此前因 429 收缩过窗口时才执行恢复逻辑。
        """
        rate_limit_headers = {
            key: value
            for key, value in response_headers.items()
            if "ratelimit" in str(key).lower() or str(key).lower() == "retry-after"
        }
        with self._condition:
            self._inflight = max(0, self._inflight - 1)
            if self._limit is not None:
                self._success_streak += 1
                recovery_threshold = max(1, self._limit)
                if self._success_streak >= recovery_threshold:
                    self._limit += 1
                    self._success_streak = 0
                    self._logger.info(
                        "模块B v2 LLM并发窗口恢复，role=%s，new_limit=%s，inflight=%s",
                        role_name,
                        self._limit,
                        self._inflight,
                    )
            self._condition.notify_all()
        if rate_limit_headers:
            self._logger.debug(
                "模块B v2 LLM响应包含限流相关响应头，role=%s，headers=%s",
                role_name,
                rate_limit_headers,
            )

    def release_rate_limited(self, *, role_name: str, error: ModuleBLlmRateLimitError) -> None:
        """
        功能说明：在命中 429 时收缩并发窗口并设置全局退避。
        参数说明：
        - role_name: 当前角色名。
        - error: 限流异常。
        返回值：无。
        异常说明：无。
        边界条件：优先使用 Retry-After；若无则采用温和默认退避。
        """
        retry_after_seconds = float(error.retry_after_seconds) if error.retry_after_seconds else 1.5
        with self._condition:
            observed_inflight = max(1, self._inflight)
            self._inflight = max(0, self._inflight - 1)
            if self._limit is None:
                self._limit = max(1, observed_inflight // 2)
            else:
                self._limit = max(1, min(self._limit - 1, observed_inflight // 2 or 1))
            self._success_streak = 0
            self._backoff_until = max(self._backoff_until, monotonic() + max(0.5, retry_after_seconds))
            self._logger.warning(
                "模块B v2 LLM全局限流退避，role=%s，new_limit=%s，retry_after=%.2fs，inflight=%s",
                role_name,
                self._limit,
                max(0.5, retry_after_seconds),
                observed_inflight,
            )
            self._condition.notify_all()

    def release_failure(self, *, role_name: str) -> None:
        """
        功能说明：在普通失败后释放并发许可。
        参数说明：
        - role_name: 当前角色名。
        返回值：无。
        异常说明：无。
        边界条件：普通错误不主动收缩并发窗口，只释放占用。
        """
        del role_name
        with self._condition:
            self._inflight = max(0, self._inflight - 1)
            self._condition.notify_all()


class ModuleBV2LlmRuntime:
    """
    功能说明：封装模块B v2 各角色通用的 Markdown LLM 调用能力。
    参数说明：
    - logger: 日志对象。
    - llm_config: 模块B LLM配置。
    - project_root: 项目根目录。
    返回值：不适用。
    异常说明：初始化阶段不抛业务异常。
        边界条件：默认沿用模块B共享的 provider/base_url/model/api_key 配置。
    """

    def __init__(self, logger: logging.Logger, llm_config: ModuleBLlmConfig, project_root: Path) -> None:
        self._logger = logger
        self._llm_config = llm_config
        self._project_root = project_root
        self._prompt_dump_dir: Path | None = None
        self._concurrency_gate = _AdaptiveLlmConcurrencyGate(logger=logger)

    def set_prompt_dump_dir(self, prompt_dump_dir: Path | None) -> None:
        """
        功能说明：设置角色调用最终 prompt 的落盘目录。
        参数说明：
        - prompt_dump_dir: 目标目录；传空表示关闭落盘。
        返回值：无。
        异常说明：无。
        边界条件：目录会在首次写入时自动创建。
        """
        self._prompt_dump_dir = prompt_dump_dir

    @property
    def logger(self) -> logging.Logger:
        """
        功能说明：暴露统一日志对象，供角色记录进度。
        参数说明：无。
        返回值：
        - logging.Logger: 运行时日志器。
        异常说明：无。
        边界条件：只读访问。
        """
        return self._logger

    @property
    def project_root(self) -> Path:
        """
        功能说明：暴露项目根目录，供角色加载 prompt 模板。
        参数说明：无。
        返回值：
        - Path: 项目根目录。
        异常说明：无。
        边界条件：只读访问。
        """
        return self._project_root

    def call_markdown(
        self,
        *,
        role_name: str,
        system_prompt: str,
        user_prompt_markdown: str,
        max_tokens_override: int | None = None,
        timeout_seconds_override: float | None = None,
    ) -> str:
        """
        功能说明：调用 LLM 并要求其返回 Markdown 文本。
        参数说明：
        - role_name: 角色名，用于日志与错误定位。
        - system_prompt: 角色级系统提示词。
        - user_prompt_markdown: 最终送入模型的 Markdown 用户提示。
        - max_tokens_override: 可选，按单次调用覆写输出 token 上限。
        - timeout_seconds_override: 可选，按单次调用覆写超时时间（秒）。
        返回值：
        - str: 模型返回的 Markdown 文本。
        异常说明：
        - ModuleBV2LlmRuntimeError: 请求失败或返回空文本时抛出。
        边界条件：模型必须按角色约定返回可解析 Markdown。
        """
        retry_times = int(self._llm_config.get_output_retry_times())
        last_error: Exception | None = None
        retry_hint = ""
        call_llm_config = self._llm_config
        replace_kwargs: dict[str, Any] = {"use_response_format_json_object": False}
        if isinstance(max_tokens_override, int) and max_tokens_override > 0:
            replace_kwargs["max_tokens"] = int(max_tokens_override)
        if isinstance(timeout_seconds_override, (int, float)) and float(timeout_seconds_override) > 0:
            replace_kwargs["timeout_seconds"] = float(timeout_seconds_override)
        if replace_kwargs:
            call_llm_config = replace(self._llm_config, **replace_kwargs)

        for attempt_index in range(retry_times + 1):
            try:
                messages = self._build_messages(
                    system_prompt=system_prompt,
                    user_prompt_markdown=user_prompt_markdown,
                    retry_hint=retry_hint,
                )
                self._dump_prompt_artifacts(
                    role_name=role_name,
                    attempt_no=attempt_index + 1,
                    messages=messages,
                )
                llm_response = self._call_llm_with_gate(
                    role_name=role_name,
                    llm_config=call_llm_config,
                    messages=messages,
                )
                normalized_output = str(llm_response.content).strip()
                if not normalized_output:
                    raise ModuleBV2LlmRuntimeError("模型返回空文本。")
                return normalized_output
            except (ModuleBLlmClientError, ModuleBV2LlmRuntimeError) as error:
                last_error = error
                if attempt_index >= retry_times:
                    break
                retry_hint = (
                    f"上次输出不符合要求：{error}。"
                    "这次必须严格按约定 Markdown 模板输出，不要补充解释，不要擅自改字段名、ID、preset_id。"
                )
                self._logger.warning(
                    "模块B v2 角色调用失败，准备重试，role=%s，attempt=%s/%s，错误=%s",
                    role_name,
                    attempt_index + 1,
                    retry_times + 1,
                    error,
                )

        raise ModuleBV2LlmRuntimeError(f"模块B v2 角色调用失败，role={role_name}，错误={last_error}")

    def _call_llm_with_gate(
        self,
        *,
        role_name: str,
        llm_config: ModuleBLlmConfig,
        messages: list[dict[str, str]],
    ) -> ModuleBLlmChatResponse:
        """
        功能说明：在统一门禁控制下执行一次真实 LLM 请求。
        参数说明：
        - role_name: 角色名。
        - llm_config: 本次调用使用的配置。
        - messages: 最终 messages 数组。
        返回值：
        - ModuleBLlmChatResponse: 带响应头的成功结果。
        异常说明：
        - ModuleBLlmClientError: 客户端层失败时向上抛出。
        边界条件：429 会即时反馈给统一门禁，再由运行时统一执行请求级重试。
        """
        request_retry_times = max(0, int(llm_config.request_retry_times))
        single_request_config = replace(llm_config, request_retry_times=0)
        last_error: ModuleBLlmClientError | None = None
        for attempt_index in range(request_retry_times + 1):
            self._concurrency_gate.acquire(role_name=role_name)
            try:
                response = call_module_b_llm_chat_detailed(
                    logger=self._logger,
                    llm_config=single_request_config,
                    messages=messages,
                    project_root=self._project_root,
                    on_rate_limited=lambda error: self._concurrency_gate.release_rate_limited(
                        role_name=role_name,
                        error=error,
                    ),
                )
            except ModuleBLlmRateLimitError as error:
                last_error = error
                if attempt_index >= request_retry_times:
                    break
                sleep_seconds = float(error.retry_after_seconds) if error.retry_after_seconds else 1.0 + 0.8 * attempt_index
                self._logger.warning(
                    "模块B v2 运行时命中限流，准备重试，role=%s，attempt=%s/%s，sleep=%.1fs，错误=%s",
                    role_name,
                    attempt_index + 1,
                    request_retry_times + 1,
                    max(0.5, sleep_seconds),
                    error,
                )
                sleep(max(0.5, sleep_seconds))
                continue
            except ModuleBLlmClientError as error:
                self._concurrency_gate.release_failure(role_name=role_name)
                last_error = error
                if attempt_index >= request_retry_times:
                    break
                sleep_seconds = 0.4 * (attempt_index + 1)
                self._logger.warning(
                    "模块B v2 运行时请求失败，准备重试，role=%s，attempt=%s/%s，sleep=%.1fs，错误=%s",
                    role_name,
                    attempt_index + 1,
                    request_retry_times + 1,
                    sleep_seconds,
                    error,
                )
                sleep(sleep_seconds)
                continue
            self._concurrency_gate.release_success(
                role_name=role_name,
                response_headers=response.response_headers,
            )
            return response
        raise ModuleBLlmClientError(f"模块B v2 运行时请求失败，role={role_name}，错误={last_error}")

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt_markdown: str,
        retry_hint: str,
    ) -> list[dict[str, str]]:
        """
        功能说明：构建兼容 Chat Completions 的 messages 数组。
        参数说明：
        - system_prompt: 系统提示词。
        - user_prompt_markdown: 用户 Markdown 提示词。
        - retry_hint: 重试提示。
        返回值：
        - list[dict[str, str]]: 标准 messages。
        异常说明：无。
        边界条件：用户提示整体保留 Markdown 结构，便于 few-shot 约束输出。
        """
        user_prompt = ""
        if retry_hint:
            user_prompt += f"## 重试要求\n{retry_hint}\n\n"
        user_prompt += user_prompt_markdown.strip()
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ]

    def _dump_prompt_artifacts(
        self,
        *,
        role_name: str,
        attempt_no: int,
        messages: list[dict[str, str]],
    ) -> None:
        """
        功能说明：将最终送入 LLM 的 prompt 写入 Markdown 调试目录。
        参数说明：
        - role_name: 角色名。
        - attempt_no: 第几次尝试（1基）。
        - messages: 最终 messages 数组。
        返回值：无。
        异常说明：写文件失败时仅记录 warning，不中断主流程。
        边界条件：未配置目录时直接跳过。
        """
        if self._prompt_dump_dir is None:
            return
        try:
            ensure_dir(self._prompt_dump_dir)
            safe_role_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(role_name).strip()) or "role"
            stem = f"{safe_role_name}.attempt_{int(attempt_no):02d}"
            prompt_text_lines = ["# Module B v2 LLM Prompt", ""]
            for message in messages:
                role_label = str(message.get("role", "")).strip().lower()
                prompt_text_lines.append(f"## {role_label}")
                prompt_text_lines.append(str(message.get("content", "")))
                prompt_text_lines.append("")
            (self._prompt_dump_dir / f"{stem}.prompt.md").write_text(
                "\n".join(prompt_text_lines).strip() + "\n",
                encoding="utf-8",
            )
        except Exception as error:  # noqa: BLE001
            self._logger.warning("模块B v2 prompt 落盘失败，role=%s，attempt=%s，错误=%s", role_name, attempt_no, error)
