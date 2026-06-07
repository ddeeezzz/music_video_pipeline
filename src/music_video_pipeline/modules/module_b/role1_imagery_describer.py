"""
文件用途：提供模块 B role1 的视觉描述生成器。
核心流程：将完整用户模板 Markdown 传给 LLM，再对返回结果做 Markdown 解析校验。
输入输出：输入 Markdown 字符串，输出解析后的视觉描述数组。
依赖说明：依赖模块 B prompt 模板、LLM 客户端与 Markdown 契约解析器。
维护说明：role1 当前公开接口返回解析后的标准结果，并保留失败原文供排障。
"""

# 标准库：用于复制 dataclass 配置对象。
from dataclasses import replace
# 标准库：用于路径类型标注。
from pathlib import Path
# 标准库：用于 JSON 元信息写盘。
import json
# 标准库：用于日志类型标注。
import logging
# 标准库：用于时间戳。
import time

# 项目内模块：提供模块 B LLM 配置对象。
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：提供模块 B LLM 调用函数。
from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat
# 项目内模块：提供模块 B role 工作目录路径。
from music_video_pipeline.modules.module_b.artifact_paths import (
    get_module_b_prompt_dir,
    get_module_b_role_dir,
    get_module_b_streaming_dir,
)
# 项目内模块：提供 role1 Markdown 契约解析器。
from music_video_pipeline.modules.module_b.markdown_contracts import (
    Role1VisualDescription,
    parse_role1_visual_descriptions,
)
# 项目内模块：提供 role1 prompt 模板装配能力。
from music_video_pipeline.modules.module_b.prompt_templates import (
    ROLE1_PROMPT_TEMPLATE_REF,
    render_prompt_asset,
)


class Role1ImageryDescriber:
    """执行模块 B role1 视觉描述生成。"""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm_config: ModuleBLlmConfig,
        project_root: Path,
        artifacts_dir: Path | None = None,
    ) -> None:
        self._logger = logger
        self._llm_config = llm_config
        self._project_root = project_root
        self._artifacts_dir = artifacts_dir.resolve() if isinstance(artifacts_dir, Path) else None
        self._work_dir = (
            get_module_b_role_dir(self._artifacts_dir, "role1")
            if self._artifacts_dir is not None
            else None
        )
        self._prompt_dir = (
            get_module_b_prompt_dir(self._artifacts_dir, "role1")
            if self._artifacts_dir is not None
            else None
        )
        self._streaming_dir = (
            get_module_b_streaming_dir(self._artifacts_dir, "role1")
            if self._artifacts_dir is not None
            else None
        )
        self._stream_preview_path = (
            (self._streaming_dir / "role1_visual_output.streaming.md").resolve()
            if self._streaming_dir is not None
            else None
        )
        self._stream_preview_meta_path = (
            (self._streaming_dir / "role1_visual_output.streaming.meta.json").resolve()
            if self._streaming_dir is not None
            else None
        )

    def generate(self, user_template_markdown: str) -> list[Role1VisualDescription]:
        """根据完整用户模板 Markdown 生成并校验 role1 结果。"""
        prompt_template_ref = ROLE1_PROMPT_TEMPLATE_REF
        prompt_template_file_override = str(self._llm_config.prompt_template_file).strip()
        if prompt_template_file_override:
            prompt_template_ref = replace(prompt_template_ref, template_file=prompt_template_file_override)

        prompt_asset = render_prompt_asset(
            project_root=self._project_root,
            prompt_template_ref=prompt_template_ref,
            user_variables={
                "User Template": _normalize_markdown_text("role1.user_template_markdown", user_template_markdown),
            },
        )
        if self._artifacts_dir is not None:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._prompt_dir.mkdir(parents=True, exist_ok=True)
            (self._prompt_dir / "role1_rendered_prompt.md").write_text(
                prompt_asset["user_prompt_markdown"], encoding="utf-8"
            )
        call_llm_config = self._llm_config
        if self._stream_preview_path is not None:
            self._stream_preview_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream_preview_path.write_text("", encoding="utf-8")
        if self._stream_preview_meta_path is not None:
            self._write_stream_preview_meta(
                current_attempt=1,
                first_chunk_at_ms=0,
                last_chunk_at_ms=0,
            )

        last_error: Exception | None = None
        response_text = ""
        try:
            llm_result = call_module_b_llm_chat(**self._build_llm_call_kwargs(
                call_llm_config=call_llm_config,
                prompt_asset=prompt_asset,
            ))
            if isinstance(llm_result, tuple):
                response_text, usage = llm_result
            else:
                response_text, usage = str(llm_result or ""), None
            _update_meta_with_usage(self._stream_preview_meta_path, usage, fallback_text=response_text)
            response_markdown = _normalize_markdown_text("role1.response_markdown", response_text)
            return parse_role1_visual_descriptions(response_markdown)
        except Exception as error:  # noqa: BLE001
            last_error = self._persist_failure_artifacts(response_text=response_text, error=error)
        raise RuntimeError(f"module_b: role1 执行失败：{last_error}")

    def _write_stream_preview_meta(
        self,
        *,
        current_attempt: int,
        first_chunk_at_ms: int,
        last_chunk_at_ms: int,
        completion_tokens: int | None = None,
        speed_tokens_per_sec: float | None = None,
    ) -> None:
        """
        功能说明：写入 role1 流式预览元信息，供监控页展示时间与重试次数。
        参数说明：
        - current_attempt: 当前尝试序号（从 1 开始）。
        - first_chunk_at_ms: 首个 chunk 到达时间戳（毫秒）。
        - last_chunk_at_ms: 最近 chunk 到达时间戳（毫秒）。
        - completion_tokens: LLM 输出 token 数。
        - speed_tokens_per_sec: 输出速率（tokens/s）。
        返回值：无。
        异常说明：无；写盘失败时静默忽略，不影响主流程。
        边界条件：时间戳为 0 表示尚未收到对应 chunk。
        """
        if self._stream_preview_meta_path is None:
            return
        try:
            payload = {
                "current_attempt": max(1, int(current_attempt)),
                "first_chunk_at_ms": max(0, int(first_chunk_at_ms)),
                "first_chunk_at": _format_timestamp_ms(first_chunk_at_ms),
                "last_chunk_at_ms": max(0, int(last_chunk_at_ms)),
                "last_chunk_at": _format_timestamp_ms(last_chunk_at_ms),
            }
            if completion_tokens is not None and completion_tokens > 0:
                payload["completion_tokens"] = completion_tokens
            if speed_tokens_per_sec is not None and speed_tokens_per_sec > 0:
                payload["speed_tokens_per_sec"] = round(speed_tokens_per_sec, 2)
            self._stream_preview_meta_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            return

    def _persist_failure_artifacts(self, response_text: str, error: Exception) -> Exception:
        """
        功能说明：在 role1 失败时保留原始返回与失败原因文件。
        参数说明：
        - response_text: LLM 返回的原始文本；为空时允许仅保留失败原因。
        - error: 原始异常。
        返回值：
        - Exception: 拼接落盘路径后的新异常对象。
        异常说明：无；落盘失败时仍返回包含原始错误的异常对象。
        边界条件：仅在 artifacts_dir 可用时写文件，否则直接返回原始异常。
        """
        if self._work_dir is None:
            return error
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            raw_output_path = self._work_dir / "role1_visual_output.failed.md"
            reason_path = self._work_dir / "role1_visual_output.failed.reason.txt"
            if str(response_text).strip():
                raw_output_path.write_text(str(response_text).strip() + "\n", encoding="utf-8")
            reason_path.write_text(str(error).strip() + "\n", encoding="utf-8")
            self._logger.warning(
                "模块 B role1 执行失败，已保留原始返回与失败原因，raw_path=%s，reason_path=%s",
                raw_output_path,
                reason_path,
            )
            return RuntimeError(
                "模块 B role1 执行失败："
                f"{str(error).strip()}；"
                f"原始返回路径={raw_output_path if str(response_text).strip() else '<空响应未落盘>'}；"
                f"失败原因路径={reason_path}"
            )
        except Exception as persist_error:  # noqa: BLE001
            self._logger.warning(
                "模块 B role1 失败落盘原始返回时出错，error=%s，persist_error=%s",
                error,
                persist_error,
            )
            return RuntimeError(
                "模块 B role1 执行失败："
                f"{str(error).strip()}；"
                f"原始返回落盘失败={persist_error}"
            )

    def _build_llm_call_kwargs(
        self,
        *,
        call_llm_config: ModuleBLlmConfig,
        prompt_asset: dict[str, str],
    ) -> dict[str, object]:
        """
        功能说明：构造 role1 LLM 调用参数，并在可用时注入流式预览回调。
        参数说明：
        - call_llm_config: 本次调用使用的 LLM 配置。
        - prompt_asset: 已渲染好的 prompt 资产。
        - retry_hint: 当前重试提示。
        - attempt_index: 当前第几次尝试（从 0 开始）。
        返回值：
        - dict[str, object]: 可直接展开给 call_module_b_llm_chat 的参数字典。
        异常说明：无。
        边界条件：无 artifacts_dir 时不注入回调，保持旧接口兼容。
        """
        kwargs: dict[str, object] = {
            "logger": self._logger,
            "llm_config": call_llm_config,
            "messages": _build_messages(
                system_prompt=prompt_asset["system_prompt"],
                user_prompt_markdown=prompt_asset["user_prompt_markdown"],
            ),
            "project_root": self._project_root,
        }
        stream_callback = self._build_stream_preview_callback()
        if stream_callback is not None:
            kwargs["on_stream_chunk"] = stream_callback
        retry_hint_callback = self._build_retry_hint_callback()
        if retry_hint_callback is not None:
            kwargs["on_retry_hint"] = retry_hint_callback
        return kwargs

    def _build_retry_hint_callback(self):
        """构造 role1 重试提示回调，供 llm_client 在重试时拼入 user prompt。"""
        def _on_retry(attempt_index: int, error: Exception) -> str:
            del attempt_index
            return (
                f"上次输出不符合要求：{error}。"
                "这次必须严格输出 Markdown，只保留 `## 意象名称`、`- pos_zh:`、`- pos_en:` 三层。"
            )
        return _on_retry

    def _build_stream_preview_callback(self):
        """
        功能说明：构造 role1 流式输出写盘回调。
        参数说明：无。
        返回值：
        - callable | None: 可接收 (aggregated_text, delta_text) 的回调；不可用时返回 None。
        异常说明：无。
        边界条件：llm_client 重试时会在已有内容后追加分隔线。
        """
        if self._stream_preview_path is None:
            return None
        self._write_stream_preview_meta(
            current_attempt=1,
            first_chunk_at_ms=0,
            last_chunk_at_ms=0,
        )
        first_chunk_at_ms = 0
        has_written_chunk = False
        stream_start_ms = int(time.time() * 1000)
        call_count = 0

        def _on_stream_chunk(aggregated_text: str, _delta_text: str) -> None:
            nonlocal first_chunk_at_ms, has_written_chunk, call_count
            call_count += 1
            current_time_ms = int(time.time() * 1000)
            delta_text = str(_delta_text or "")
            if not delta_text:
                return
            if first_chunk_at_ms <= 0:
                first_chunk_at_ms = current_time_ms
                elapsed_ms = first_chunk_at_ms - stream_start_ms
                self._logger.info(
                    "模块 B role1 流式写盘收到首个 chunk：耗时=%sms",
                    elapsed_ms,
                )
            if not has_written_chunk:
                if call_count > 1 and self._stream_preview_path.exists():
                    existing_text = self._stream_preview_path.read_text(encoding="utf-8").rstrip()
                    if existing_text:
                        self._stream_preview_path.write_text(
                            f"{existing_text}\n\n[Retry]\n", encoding="utf-8"
                        )
                has_written_chunk = True
            with self._stream_preview_path.open("a", encoding="utf-8") as preview_file:
                preview_file.write(delta_text)
            self._logger.debug(
                "模块 B role1 流式写盘：delta_chars=%s total_chars=%s",
                len(delta_text),
                len(aggregated_text),
            )
            self._write_stream_preview_meta(
                current_attempt=1,
                first_chunk_at_ms=first_chunk_at_ms,
                last_chunk_at_ms=current_time_ms,
            )

        return _on_stream_chunk


def _format_timestamp_ms(timestamp_ms: int) -> str:
    """
    功能说明：把毫秒时间戳格式化为本地可读时间文本。
    参数说明：
    - timestamp_ms: 毫秒时间戳。
    返回值：
    - str: 格式化后的时间文本；无效时间返回空字符串。
    异常说明：无。
    边界条件：时间戳小于等于 0 时返回空字符串。
    """
    normalized_timestamp_ms = int(timestamp_ms or 0)
    if normalized_timestamp_ms <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(normalized_timestamp_ms / 1000))


def _normalize_markdown_text(field_name: str, value: str) -> str:
    """
    功能说明：标准化并校验非空 Markdown 字符串。
    参数说明：
    - field_name: 字段名。
    - value: 原始 Markdown 文本。
    返回值：
    - str: 去除首尾空白后的 Markdown 文本。
    异常说明：
    - ValueError: 文本为空时抛出。
    边界条件：仅做字符串级校验，不解析内部结构。
    """
    normalized_text = str(value or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        raise ValueError(f"{field_name} 不能为空。")
    return normalized_text


def _build_messages(
    *,
    system_prompt: str,
    user_prompt_markdown: str,
) -> list[dict[str, str]]:
    """构建 role1 的标准 messages 数组。"""
    return [
        {"role": "system", "content": str(system_prompt or "").strip()},
        {"role": "user", "content": str(user_prompt_markdown or "").strip()},
    ]


def _update_meta_with_usage(meta_path: Path | None, usage: dict | None, fallback_text: str = "") -> None:
    """用 LLM usage 信息更新流式预览 meta 文件；无 usage 时用文本长度估算。"""
    if meta_path is None:
        return
    try:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    if usage is not None:
        completion_tokens = int(usage.get("completion_tokens", 0))
    elif fallback_text:
        completion_tokens = max(1, len(str(fallback_text)) // 2)
    else:
        return
    if completion_tokens <= 0:
        return
    existing_meta["completion_tokens"] = completion_tokens
    first_ms = int(existing_meta.get("first_chunk_at_ms", 0))
    last_ms = int(existing_meta.get("last_chunk_at_ms", 0))
    if first_ms > 0 and last_ms > first_ms:
        elapsed_s = (last_ms - first_ms) / 1000.0
        if elapsed_s > 0:
            existing_meta["speed_tokens_per_sec"] = round(completion_tokens / elapsed_s, 2)
    try:
        meta_path.write_text(
            json.dumps(existing_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        return
