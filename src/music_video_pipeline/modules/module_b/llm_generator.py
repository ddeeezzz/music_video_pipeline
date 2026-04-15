"""
文件用途：提供模块B真实LLM单段分镜双语提示词生成能力。
核心流程：构建提示词 -> 调用SiliconFlow -> 解析并校验输出JSON。
输入输出：输入分镜上下文，输出 scene_desc 与 keyframe/video 中英文提示词。
依赖说明：依赖 llm_prompt/llm_client/llm_parser 子模块。
维护说明：本文件不负责模块B完整shot拼装，只负责LLM提示词字段。
"""

# 标准库：用于日志输出
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any

# 项目内模块：模块B LLM配置
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：LLM客户端调用
from music_video_pipeline.modules.module_b.llm_client import ModuleBLlmClientError, call_module_b_llm_chat
# 项目内模块：LLM输出解析
from music_video_pipeline.modules.module_b.llm_parser import ModuleBLlmParseError, parse_module_b_llm_output
# 项目内模块：LLM提示词构建
from music_video_pipeline.modules.module_b.llm_prompt import build_module_b_prompt_messages


class ModuleBLlmGenerationError(RuntimeError):
    """模块B LLM 生成失败异常。"""


def generate_module_b_prompts(
    logger: logging.Logger,
    llm_config: ModuleBLlmConfig,
    llm_input_payload: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    """
    功能说明：执行模块B真实LLM分镜生成并返回双语提示词字段。
    参数说明：
    - logger: 日志对象。
    - llm_config: 模块B LLM配置。
    - llm_input_payload: 单段分镜输入上下文字典。
    - project_root: 项目根目录，用于解析密钥相对路径。
    返回值：
    - dict[str, str]: scene_desc 与 keyframe/video 的中英文提示词。
    异常说明：
    - ModuleBLlmGenerationError: 请求或解析重试耗尽时抛出。
    边界条件：JSON解析重试次数由 llm_config.json_retry_times 控制。
    """
    parse_retry_times = max(0, int(llm_config.json_retry_times))
    last_error: Exception | None = None
    retry_hint = ""

    for attempt_index in range(parse_retry_times + 1):
        try:
            messages = build_module_b_prompt_messages(
                input_payload=llm_input_payload,
                retry_hint=retry_hint,
            )
            llm_output_text = call_module_b_llm_chat(
                logger=logger,
                llm_config=llm_config,
                messages=messages,
                project_root=project_root,
            )
            return parse_module_b_llm_output(
                llm_output_text=llm_output_text,
                scene_desc_max_chars=int(llm_config.scene_desc_max_chars),
                keyframe_prompt_max_chars=int(llm_config.keyframe_prompt_max_chars),
                video_prompt_max_chars=int(llm_config.video_prompt_max_chars),
            )
        except (ModuleBLlmClientError, ModuleBLlmParseError) as error:
            last_error = error
            if attempt_index >= parse_retry_times:
                break
            retry_hint = f"上次输出不符合要求：{error}。请严格只输出五字段JSON对象。"
            logger.warning(
                "模块B LLM双语提示词生成失败，准备重试，attempt=%s/%s，错误=%s",
                attempt_index + 1,
                parse_retry_times + 1,
                error,
            )

    raise ModuleBLlmGenerationError(f"模块B LLM双语提示词生成失败：{last_error}")
