"""
文件用途：提供模块 B role3 的镜头规划器。
核心流程：将故事模板 remotion 段与大段上下文传给 LLM，再对返回结果做 Markdown 解析校验。
输入输出：输入 storyboard_markdown 与 big_segment_context，输出校验后的镜头规划数组。
依赖说明：依赖模块 B prompt 模板、LLM 客户端与 Markdown 契约解析器。
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
# 标准库：用于正则匹配 big_segment_id。
import re

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
# 项目内模块：提供 role3 Markdown 契约解析器与数据结构。
from music_video_pipeline.modules.module_b.markdown_contracts import (
    ShotPlan,
    parse_shot_plans,
)
# 项目内模块：提供 role3 prompt 模板装配能力。
from music_video_pipeline.modules.module_b.prompt_templates import (
    ROLE3_PROMPT_TEMPLATE_REF,
    render_prompt_asset,
)


class Role3ShotPlanner:
    """执行模块 B role3 镜头规划。"""

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
            get_module_b_role_dir(self._artifacts_dir, "role3")
            if self._artifacts_dir is not None
            else None
        )
        self._prompt_dir = (
            get_module_b_prompt_dir(self._artifacts_dir, "role3")
            if self._artifacts_dir is not None
            else None
        )
        self._streaming_dir = (
            get_module_b_streaming_dir(self._artifacts_dir, "role3")
            if self._artifacts_dir is not None
            else None
        )
        self._stream_preview_path: Path | None = None
        self._stream_preview_meta_path: Path | None = None

    def generate(self, storyboard_markdown: str, big_segment_context: str) -> list[ShotPlan]:
        """为当前大段生成并校验 role3 镜头规划结果。"""
        prompt_template_ref = ROLE3_PROMPT_TEMPLATE_REF
        prompt_template_file_override = str(self._llm_config.prompt_template_file).strip()
        if prompt_template_file_override:
            prompt_template_ref = replace(prompt_template_ref, template_file=prompt_template_file_override)

        prompt_asset = render_prompt_asset(
            project_root=self._project_root,
            prompt_template_ref=prompt_template_ref,
            user_variables={
                "模板的## remotion模板": _normalize_markdown_text(
                    "role3.remotion_template", storyboard_markdown
                ),
                "当前大段剧情和镜头": str(big_segment_context or "").strip(),
            },
        )
        # 从上下文中提取 big_segment_id，设置 per-segment 路径
        seg_match = re.match(r"##\s+(big_\S+)", str(big_segment_context or "").strip())
        bid = seg_match.group(1).strip() if seg_match else ""
        if self._work_dir is not None and bid:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._prompt_dir.mkdir(parents=True, exist_ok=True)
            (self._prompt_dir / f"role3_rendered_prompt.{bid}.md").write_text(
                prompt_asset["user_prompt_markdown"], encoding="utf-8"
            )
            self._stream_preview_path = (
                self._streaming_dir / f"role3_segment_output.streaming.{bid}.md"
            ).resolve()
            self._stream_preview_meta_path = (
                self._streaming_dir / f"role3_segment_output.streaming.{bid}.meta.json"
            ).resolve()
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
            response_text, usage = call_module_b_llm_chat(**self._build_llm_call_kwargs(
                call_llm_config=call_llm_config,
                prompt_asset=prompt_asset,
            ))
            _update_meta_with_usage(self._stream_preview_meta_path, usage, fallback_text=response_text)
            response_markdown = _normalize_markdown_text("role3.response_markdown", response_text)
            return parse_shot_plans(response_markdown)
        except Exception as error:  # noqa: BLE001
            last_error = self._persist_failure_artifacts(response_text=response_text, error=error)
        raise RuntimeError(f"module_b: role3 执行失败：{last_error}")

    def _write_stream_preview_meta(
        self,
        *,
        current_attempt: int,
        first_chunk_at_ms: int,
        last_chunk_at_ms: int,
        completion_tokens: int | None = None,
        speed_tokens_per_sec: float | None = None,
    ) -> None:
        """写入 role3 流式预览元信息。"""
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
        """在 role3 失败时保留原始返回与失败原因文件。"""
        if self._work_dir is None:
            return error
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            raw_output_path = self._work_dir / "role3_segment_output.failed.md"
            reason_path = self._work_dir / "role3_segment_output.failed.reason.txt"
            if str(response_text).strip():
                raw_output_path.write_text(str(response_text).strip() + "\n", encoding="utf-8")
            reason_path.write_text(str(error).strip() + "\n", encoding="utf-8")
            self._logger.warning(
                "模块 B role3 执行失败，已保留原始返回与失败原因，raw_path=%s，reason_path=%s",
                raw_output_path,
                reason_path,
            )
            return RuntimeError(
                "模块 B role3 执行失败："
                f"{str(error).strip()}；"
                f"原始返回路径={raw_output_path if str(response_text).strip() else '<空响应未落盘>'}；"
                f"失败原因路径={reason_path}"
            )
        except Exception as persist_error:  # noqa: BLE001
            self._logger.warning(
                "模块 B role3 失败落盘原始返回时出错，error=%s，persist_error=%s",
                error,
                persist_error,
            )
            return RuntimeError(
                "模块 B role3 执行失败："
                f"{str(error).strip()}；"
                f"原始返回落盘失败={persist_error}"
            )

    def _build_llm_call_kwargs(
        self,
        *,
        call_llm_config: ModuleBLlmConfig,
        prompt_asset: dict[str, str],
    ) -> dict[str, object]:
        """构造 role3 LLM 调用参数，并在可用时注入流式预览与重试回调。"""
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
        """构造 role3 重试提示回调。"""
        def _on_retry(attempt_index: int, error: Exception) -> str:
            del attempt_index
            return (
                f"上次输出不符合要求：{error}。"
                "这次必须严格输出 Markdown，每个 `## big_segment_id` 下按 `### shot_id` 组织，"
                "每个 shot 只保留 `- scene_desc_zh:` 和 `- remotion_id:` 两个字段。"
            )
        return _on_retry

    def _build_stream_preview_callback(self):
        """构造 role3 流式输出写盘回调。"""
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
                    "模块 B role3 流式写盘收到首个 chunk：耗时=%sms",
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
                "模块 B role3 流式写盘：delta_chars=%s total_chars=%s",
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


def _normalize_markdown_text(field_name: str, value: str) -> str:
    """标准化并校验非空 Markdown 字符串。"""
    normalized_text = str(value or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        raise ValueError(f"{field_name} 不能为空。")
    return normalized_text


def _build_messages(
    *,
    system_prompt: str,
    user_prompt_markdown: str,
) -> list[dict[str, str]]:
    """构建 role3 的标准 messages 数组。"""
    return [
        {"role": "system", "content": str(system_prompt or "").strip()},
        {"role": "user", "content": str(user_prompt_markdown or "").strip()},
    ]
