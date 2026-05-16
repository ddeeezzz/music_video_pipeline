"""
文件用途：验证模块 B v2 prompt 模板 section 解析兼容新旧标题。
核心流程：构造模板文本并调用解析器，断言 system/user section 可正确切分。
输入输出：输入测试模板文本，输出断言结果。
依赖说明：依赖项目内 prompt 模板解析器。
维护说明：当 prompt section 标题约定调整时需同步更新本测试。
"""

from music_video_pipeline.modules.module_b_v2.prompt_templates import parse_prompt_sections


def test_parse_prompt_sections_should_support_system_prompt_and_user_prompt() -> None:
    """
    功能说明：验证解析器兼容 `# System Prompt` 与 `# User Prompt` 标题。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：section 正文允许为普通多行文本。
    """
    system_text, user_text = parse_prompt_sections(
        "# System Prompt\nsystem body\n\n# User Prompt\nuser body\n"
    )
    assert system_text == "system body"
    assert user_text == "user body"


def test_parse_prompt_sections_should_reject_legacy_section_titles() -> None:
    """
    功能说明：验证解析器不再兼容旧版 section 标题。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：旧标题包括 `# System` 与 `# User Template`。
    """
    import pytest

    with pytest.raises(ValueError, match="# System Prompt"):
        parse_prompt_sections("# System\nsystem body\n\n# User Template\nuser body\n")
