"""
文件用途：验证模块 B role1 的 prompt 拼装与 Markdown 契约。
核心流程：校验 prompt section 解析，以及 role1 的结构化返回行为。
输入输出：输入 pytest 测试上下文，输出断言结果。
依赖说明：依赖 pytest 与模块 B role1 实现。
维护说明：当 role1 prompt 或输出契约调整时需同步更新本测试。
"""

# 标准库：用于日志构造。
import logging
# 标准库：用于项目根路径解析。
from pathlib import Path

# 第三方库：用于异常断言。
import pytest

# 项目内模块：提供模块 B LLM 配置对象。
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：提供 prompt 模板 section 解析函数。
from music_video_pipeline.modules.module_b.prompt_templates import parse_prompt_sections
# 项目内模块：提供 role1 视觉描述生成器。
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber


def test_parse_prompt_sections_should_require_system_prompt_and_user_prompt() -> None:
    """
    功能说明：验证模块 B prompt 模板只接受 `# System Prompt` 与 `# User Prompt`。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧标题必须报错。
    """
    system_text, user_text = parse_prompt_sections(
        "# System Prompt\nsystem body\n\n# User Prompt\nuser body\n"
    )
    assert system_text == "system body"
    assert user_text == "user body"

    with pytest.raises(ValueError, match="# System Prompt"):
        parse_prompt_sections("# System\nsystem body\n\n# User Template\nuser body\n")


def test_role1_generate_should_send_full_user_template_and_return_validated_items(monkeypatch) -> None:
    """
    功能说明：验证 role1 会把完整模板块传给模型，并返回校验后的标准结果。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：公开接口不再返回裸 Markdown，而是结构化结果。
    """
    captured: dict[str, object] = {}
    user_template_markdown = (
        "## 故事\n"
        "黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。\n\n"
        "## 意象\n"
        "少女：水手服少女，黑长直，细瘦身形。\n"
        "黑猫：瘦长、警觉的黑猫，细尾，尖耳。"
    )
    response_markdown = (
        "## 少女\n"
        "- pos_zh: 水手服少女，黑长直发，细瘦身形，百褶裙\n"
        "- pos_en: sailor uniform girl, long straight black hair, slim figure, pleated skirt\n\n"
        "## 黑猫\n"
        "- pos_zh: 瘦长黑猫，尖耳，细尾，短毛，紧绷背线\n"
        "- pos_en: slender black cat, pointed ears, thin tail, short fur, tense back line\n"
    )

    def _fake_call_module_b_llm_chat(*, logger, llm_config, messages, project_root):  # type: ignore[no-untyped-def]
        del logger, llm_config, project_root
        captured["messages"] = messages
        return response_markdown

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_b.role1_imagery_describer.call_module_b_llm_chat",
        _fake_call_module_b_llm_chat,
    )
    project_root = Path(__file__).resolve().parents[1]
    describer = Role1ImageryDescriber(
        logger=logging.getLogger("test_module_b_role1"),
        llm_config=ModuleBLlmConfig(),
        project_root=project_root,
    )

    result = describer.generate(user_template_markdown)

    assert len(result) == 2
    assert result[0].imagery_name == "少女"
    assert result[0].pos_zh == "水手服少女，黑长直发，细瘦身形，百褶裙"
    assert result[1].imagery_name == "黑猫"
    assert result[1].pos_en == "slender black cat, pointed ears, thin tail, short fur, tense back line"
    assert captured["messages"] == [
        {
            "role": "system",
            "content": parse_prompt_sections(
                (project_root / "configs" / "prompts" / "module_b.role1_visual_director.md").read_text(encoding="utf-8")
            )[0],
        },
        {
            "role": "user",
            "content": user_template_markdown,
        },
    ]


def test_role1_generate_should_retry_when_response_markdown_breaks_contract(monkeypatch) -> None:
    """
    功能说明：验证 role1 在模型返回的 Markdown 不符合契约时会触发重试。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：第一次返回缺字段、第二次返回合法时应成功收敛。
    """
    call_count = {"value": 0}

    def _fake_call_module_b_llm_chat(*, logger, llm_config, messages, project_root):  # type: ignore[no-untyped-def]
        del logger, llm_config, messages, project_root
        call_count["value"] += 1
        if call_count["value"] == 1:
            return "## 少女\n- pos_zh: 水手服少女，黑长直发，细瘦身形，百褶裙\n"
        return (
            "## 少女\n"
            "- pos_zh: 水手服少女，黑长直发，细瘦身形，百褶裙\n"
            "- pos_en: sailor uniform girl, long straight black hair, slim figure, pleated skirt\n"
        )

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_b.role1_imagery_describer.call_module_b_llm_chat",
        _fake_call_module_b_llm_chat,
    )
    project_root = Path(__file__).resolve().parents[1]
    describer = Role1ImageryDescriber(
        logger=logging.getLogger("test_module_b_role1"),
        llm_config=ModuleBLlmConfig(),
        project_root=project_root,
    )

    result = describer.generate("## 故事\n故事\n\n## 意象\n少女：水手服少女。")

    assert call_count["value"] == 2
    assert len(result) == 1
    assert result[0].imagery_name == "少女"


def test_role1_generate_should_reject_empty_user_template_markdown() -> None:
    """
    功能说明：验证 role1 入口要求非空 Markdown 字符串。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅做最小的字符串级保护。
    """
    project_root = Path(__file__).resolve().parents[1]
    describer = Role1ImageryDescriber(
        logger=logging.getLogger("test_module_b_role1"),
        llm_config=ModuleBLlmConfig(),
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="user_template_markdown"):
        describer.generate("   ")
