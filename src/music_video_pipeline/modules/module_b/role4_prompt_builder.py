"""
文件用途：提供模块 B role4 的提示词构建器。
核心流程：按 shot 逐个调 LLM，将 remotion 模板、镜头摘要与视觉参考融合为关键帧提示词。
输入输出：输入 user_variables 与 shot_id，输出 LLM 原始 Markdown 文本。
依赖说明：依赖模块 B prompt 模板、LLM 客户端与 artifact_paths。
维护说明：role4 按 shot 独立调用，每个 shot 写入独立 streaming 预览文件。
"""

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
# 项目内模块：复用 role1 的 usage 元信息更新工具。
from music_video_pipeline.modules.module_b.role1_imagery_describer import _update_meta_with_usage
# 项目内模块：提供 role4 prompt 模板装配能力。
from music_video_pipeline.modules.module_b.prompt_templates import (
    PromptTemplateRef,
    ROLE4_PROMPT_MAP,
    render_prompt_asset,
)


class Role4PromptBuilder:
    """执行模块 B role4 提示词构建。"""

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
            get_module_b_role_dir(self._artifacts_dir, "role4")
            if self._artifacts_dir is not None
            else None
        )
        self._prompt_dir = (
            get_module_b_prompt_dir(self._artifacts_dir, "role4")
            if self._artifacts_dir is not None
            else None
        )
        self._streaming_dir = (
            get_module_b_streaming_dir(self._artifacts_dir, "role4")
            if self._artifacts_dir is not None
            else None
        )
        self._stream_preview_path: Path | None = None
        self._stream_preview_meta_path: Path | None = None

    def generate(self, user_variables: dict[str, str], shot_id: str, subject_kind: str = "") -> str:
        """为单个 shot 生成 role4 Markdown 提示词，返回 LLM 原始响应文本。
        subject_kind 用于选择对应类别的 prompt 模板（character_human/animal/object/scene）。"""
        sid = str(shot_id).strip()
        prompt_template_file_override = str(self._llm_config.prompt_template_file).strip()
        if prompt_template_file_override:
            prompt_template_ref = PromptTemplateRef(template_file=prompt_template_file_override)
        else:
            # 从 subject_kind 选择对应 prompt
            sk = str(subject_kind or "").strip().lower()
            if sk in ROLE4_PROMPT_MAP:
                prompt_template_ref = ROLE4_PROMPT_MAP[sk]
            else:
                # 从 shot_brief 回退解析
                shot_brief = str(user_variables.get("shot_brief", "") or "")
                for line in shot_brief.split("\n"):
                    if line.strip().startswith("- subject_kind:"):
                        sk_from_brief = line.split(":", 1)[1].strip().lower()
                        if sk_from_brief in ROLE4_PROMPT_MAP:
                            prompt_template_ref = ROLE4_PROMPT_MAP[sk_from_brief]
                            break
                else:
                    prompt_template_ref = ROLE4_PROMPT_MAP["human"]

        prompt_asset = render_prompt_asset(
            project_root=self._project_root,
            prompt_template_ref=prompt_template_ref,
            user_variables=user_variables,
        )

        # 设置 per-shot streaming 路径
        if self._work_dir is not None and sid:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._prompt_dir.mkdir(parents=True, exist_ok=True)
            (self._prompt_dir / f"role4_rendered_prompt.{sid}.md").write_text(
                prompt_asset["user_prompt_markdown"], encoding="utf-8"
            )
            self._stream_preview_path = (
                self._streaming_dir / f"role4_prompt_output.streaming.{sid}.md"
            ).resolve()
            self._stream_preview_meta_path = (
                self._streaming_dir / f"role4_prompt_output.streaming.{sid}.meta.json"
            ).resolve()

        if self._stream_preview_path is not None:
            self._stream_preview_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream_preview_path.write_text("", encoding="utf-8")
        if self._stream_preview_meta_path is not None:
            self._write_stream_preview_meta(
                current_attempt=1,
                first_chunk_at_ms=0,
                last_chunk_at_ms=0,
            )

        call_llm_config = self._llm_config
        last_error: Exception | None = None
        response_text = ""
        try:
            response_text, usage = call_module_b_llm_chat(**self._build_llm_call_kwargs(
                call_llm_config=call_llm_config,
                prompt_asset=prompt_asset,
                shot_id=sid,
            ))
            _update_meta_with_usage(self._stream_preview_meta_path, usage, fallback_text=response_text)
            return str(response_text or "")
        except Exception as error:
            last_error = self._persist_failure_artifacts(response_text=response_text, error=error, shot_id=sid)
        raise RuntimeError(f"module_b: role4 执行失败，shot_id={sid}：{last_error}")

    def _write_stream_preview_meta(
        self,
        *,
        current_attempt: int,
        first_chunk_at_ms: int,
        last_chunk_at_ms: int,
        completion_tokens: int | None = None,
        speed_tokens_per_sec: float | None = None,
    ) -> None:
        """写入 role4 per-shot 流式预览元信息。"""
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
        except Exception:
            return

    def _persist_failure_artifacts(self, response_text: str, error: Exception, shot_id: str) -> Exception:
        """在 role4 失败时保留原始返回与失败原因文件。"""
        if self._work_dir is None:
            return error
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            raw_output_path = self._work_dir / f"role4_prompt_output.{shot_id}.failed.md"
            reason_path = self._work_dir / f"role4_prompt_output.{shot_id}.failed.reason.txt"
            if str(response_text).strip():
                raw_output_path.write_text(str(response_text).strip() + "\n", encoding="utf-8")
            reason_path.write_text(str(error).strip() + "\n", encoding="utf-8")
            self._logger.warning(
                "模块 B role4 执行失败，shot_id=%s，raw_path=%s，reason_path=%s",
                shot_id,
                raw_output_path,
                reason_path,
            )
            return RuntimeError(
                f"模块 B role4 执行失败（shot_id={shot_id}）："
                f"{str(error).strip()}；"
                f"原始返回路径={raw_output_path if str(response_text).strip() else '<空响应未落盘>'}；"
                f"失败原因路径={reason_path}"
            )
        except Exception as persist_error:
            self._logger.warning(
                "模块 B role4 失败落盘时出错，shot_id=%s，error=%s，persist_error=%s",
                shot_id,
                error,
                persist_error,
            )
            return RuntimeError(
                f"模块 B role4 执行失败（shot_id={shot_id}）："
                f"{str(error).strip()}；"
                f"原始返回落盘失败={persist_error}"
            )

    def _build_llm_call_kwargs(
        self,
        *,
        call_llm_config: ModuleBLlmConfig,
        prompt_asset: dict[str, str],
        shot_id: str,
    ) -> dict[str, object]:
        """构造 role4 LLM 调用参数，并在可用时注入流式预览回调。"""
        kwargs: dict[str, object] = {
            "logger": self._logger,
            "llm_config": call_llm_config,
            "messages": _build_messages(
                system_prompt=prompt_asset["system_prompt"],
                user_prompt_markdown=prompt_asset["user_prompt_markdown"],
            ),
            "project_root": self._project_root,
        }
        stream_callback = self._build_stream_preview_callback(shot_id=shot_id)
        if stream_callback is not None:
            kwargs["on_stream_chunk"] = stream_callback
        retry_hint_callback = self._build_retry_hint_callback(shot_id=shot_id)
        if retry_hint_callback is not None:
            kwargs["on_retry_hint"] = retry_hint_callback
        return kwargs

    def _build_retry_hint_callback(self, shot_id: str):
        """构造 role4 重试提示回调。"""
        def _on_retry(attempt_index: int, error: Exception) -> str:
            del attempt_index
            return (
                f"上次输出不符合要求（shot_id={shot_id}）：{error}。"
                "这次必须严格输出 Markdown，按照模板要求携带 ```md 代码块。"
            )
        return _on_retry

    def _build_stream_preview_callback(self, shot_id: str):
        """构造 role4 per-shot 流式输出写盘回调。"""
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
                    "模块 B role4 流式写盘收到首个 chunk：shot_id=%s，耗时=%sms",
                    shot_id,
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
                "模块 B role4 流式写盘：shot_id=%s，delta_chars=%s total_chars=%s",
                shot_id,
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
    """把毫秒时间戳格式化为本地可读时间文本。"""
    normalized_timestamp_ms = int(timestamp_ms or 0)
    if normalized_timestamp_ms <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(normalized_timestamp_ms / 1000))


def _build_messages(
    *,
    system_prompt: str,
    user_prompt_markdown: str,
) -> list[dict[str, str]]:
    """构建 role4 的标准 messages 数组。"""
    return [
        {"role": "system", "content": str(system_prompt or "").strip()},
        {"role": "user", "content": str(user_prompt_markdown or "").strip()},
    ]
