"""
文件用途：实现模块B v2 的角色1“视觉编导”。
核心流程：原样读取故事与意象文本，请求 LLM 细化每个意象的外观描述。
输入输出：输入故事/意象原文，输出结构化对象外观描述结果。
依赖说明：依赖 v2 LLM runtime、parser 与 prompt 模板加载。
维护说明：本角色只负责“意象长什么样”，不在本地拆解意象原文。
"""

# 标准库：用于正则提取意象名称。
import re
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：v2 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：v2 常量与数据结构。
from music_video_pipeline.modules.module_b_v2.models import Role1VisualOutput
# 项目内模块：v2 parser。
from music_video_pipeline.modules.module_b_v2.parser import (
    parse_role1_visual_markdown,
    validate_role1_visual_output,
)
# 项目内模块：统一 prompt 模板加载。
from music_video_pipeline.modules.module_b_v2.prompt_templates import (
    ROLE1_PROMPT_ASSET,
    render_prompt_asset,
)


# 常量：角色1对象外观描述输出默认 token 上限。
ROLE1_VISUAL_DIRECTOR_MIN_MAX_TOKENS = 1400
# 常量：角色1请求超时（秒）。
ROLE1_VISUAL_DIRECTOR_TIMEOUT_SECONDS = 180.0
ROLE1_IMAGERY_NAME_PATTERN = re.compile(r"^\s*([^:：]+?)\s*[：:]\s*.+$")


class Role1VisualDirector:
    """
    功能说明：执行角色1对象外观描述生成。
    参数说明：
    - llm_runtime: 通用 LLM 运行时。
    返回值：不适用。
    异常说明：具体异常由 generate 抛出。
    边界条件：目录为空时返回空数组。
    """

    def __init__(self, llm_runtime: ModuleBV2LlmRuntime) -> None:
        self._llm_runtime = llm_runtime

    def generate(self, storyboard_template: dict[str, Any]) -> Role1VisualOutput:
        """
        功能说明：为全部意象一次性生成外观描述。
        参数说明：
        - storyboard_template: 已编译编排模板。
        返回值：
        - Role1VisualOutput: 对象外观描述输出。
        异常说明：LLM 或字段校验失败时抛出异常。
        边界条件：每个意象固定要求返回 2 组 refs。
        """
        self._llm_runtime.logger.info("模块B v2 role1 开始执行")
        story_payload = storyboard_template.get("story", {})
        story_text = (
            str(story_payload.get("premise_zh", "")).strip()
            if isinstance(story_payload, dict)
            else str(story_payload).strip()
        )
        imagery_text = str(storyboard_template.get("imagery", "")).strip()
        imagery_names = _extract_imagery_names(imagery_text)
        if not imagery_names:
            return {"items": []}
        self._llm_runtime.logger.info("模块B v2 role1 准备请求，item_count=%s", len(imagery_names))
        prompt_asset = render_prompt_asset(
            project_root=self._llm_runtime.project_root,
            prompt_asset=ROLE1_PROMPT_ASSET,
            user_variables={
                "User Template": _build_user_template(
                    story_text=story_text,
                    imagery_text=imagery_text,
                ),
            },
        )
        response_text = self._llm_runtime.call_markdown(
            role_name="role1_visual_director",
            system_prompt=prompt_asset.system_prompt,
            user_prompt_markdown=prompt_asset.user_prompt_markdown,
            max_tokens_override=max(
                ROLE1_VISUAL_DIRECTOR_MIN_MAX_TOKENS,
                len(imagery_names) * 320,
            ),
            timeout_seconds_override=ROLE1_VISUAL_DIRECTOR_TIMEOUT_SECONDS,
        )
        parsed_output = parse_role1_visual_markdown(response_text)
        validated_output = validate_role1_visual_output(
            data=parsed_output,
            requested_item_ids=imagery_names,
        )
        self._llm_runtime.logger.info("模块B v2 role1 执行完成")
        return validated_output


def _extract_imagery_names(imagery_text: str) -> list[str]:
    """
    功能说明：从意象原文中提取每条意象的名称。
    参数说明：
    - imagery_text: 模板中的意象原文。
    返回值：
    - list[str]: 按出现顺序返回的意象名称数组。
    异常说明：无。
    边界条件：仅识别 `名称：描述` 或 `名称:描述` 形式的行。
    """
    result: list[str] = []
    seen_names: set[str] = set()
    for raw_line in str(imagery_text or "").replace("\r\n", "\n").split("\n"):
        matched = ROLE1_IMAGERY_NAME_PATTERN.match(raw_line.strip())
        if matched is None:
            continue
        name = str(matched.group(1)).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        result.append(name)
    return result


def _build_user_template(*, story_text: str, imagery_text: str) -> str:
    """
    功能说明：构建 role1 传给 LLM 的完整用户输入块。
    参数说明：
    - story_text: 故事原文。
    - imagery_text: 意象原文。
    返回值：
    - str: 包含二级标题的完整用户模板文本。
    异常说明：无。
    边界条件：空文本统一回退为 `none`。
    """
    return "\n".join(
        [
            "## 故事",
            story_text or "none",
            "",
            "## 意象",
            imagery_text or "none",
        ]
    ).strip()
