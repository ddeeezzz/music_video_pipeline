"""
文件用途：提供模块 B role1 的视觉描述生成器。
核心流程：将完整用户模板 Markdown 传给 LLM，再对返回结果做 Markdown 契约校验。
输入输出：输入 Markdown 字符串，输出校验后的视觉描述数组。
依赖说明：依赖模块 B prompt 模板、LLM 客户端与 Markdown 契约解析器。
维护说明：role1 与 role2/3/4 一样，公开接口返回校验后的标准结果。
"""

# 标准库：用于复制 dataclass 配置对象。
from dataclasses import replace
# 标准库：用于路径类型标注。
from pathlib import Path
# 标准库：用于日志类型标注。
import logging

# 项目内模块：提供模块 B LLM 配置对象。
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：提供模块 B LLM 调用函数。
from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat
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
    ) -> None:
        self._logger = logger
        self._llm_config = llm_config
        self._project_root = project_root

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
        call_llm_config = replace(self._llm_config, use_response_format_json_object=False)
        retry_times = call_llm_config.get_output_retry_times()
        retry_hint = ""
        last_error: Exception | None = None

        for attempt_index in range(retry_times + 1):
            try:
                response_text = call_module_b_llm_chat(
                    logger=self._logger,
                    llm_config=call_llm_config,
                    messages=_build_messages(
                        system_prompt=prompt_asset["system_prompt"],
                        user_prompt_markdown=prompt_asset["user_prompt_markdown"],
                        retry_hint=retry_hint,
                    ),
                    project_root=self._project_root,
                )
                response_markdown = _normalize_markdown_text("role1.response_markdown", response_text)
                return parse_role1_visual_descriptions(response_markdown)
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt_index >= retry_times:
                    break
                retry_hint = (
                    f"上次输出不符合要求：{error}。"
                    "这次必须严格输出 Markdown，只保留 `## 意象名称`、`- pos_zh:`、`- pos_en:` 三层。"
                )
                self._logger.warning(
                    "模块 B role1 输出不符合契约，准备重试，attempt=%s/%s，error=%s",
                    attempt_index + 1,
                    retry_times + 1,
                    error,
                )
        raise RuntimeError(f"module_b: role1 failed after retries: {last_error}")


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
    retry_hint: str,
) -> list[dict[str, str]]:
    """
    功能说明：构建 role1 的标准 messages 数组。
    参数说明：
    - system_prompt: 系统提示词。
    - user_prompt_markdown: 用户 Markdown 提示词。
    - retry_hint: 可选重试提示。
    返回值：
    - list[dict[str, str]]: 标准 messages。
    异常说明：无。
    边界条件：重试提示会以 Markdown 小节前缀拼到 user prompt 前面。
    """
    user_prompt = str(user_prompt_markdown or "").strip()
    if retry_hint:
        user_prompt = f"## 重试要求\n{str(retry_hint).strip()}\n\n{user_prompt}"
    return [
        {"role": "system", "content": str(system_prompt or "").strip()},
        {"role": "user", "content": user_prompt},
    ]
