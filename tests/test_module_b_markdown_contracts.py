"""
验证模块 B role1/2/3/4 的 Markdown 轻量提取层。
核心流程：构造最小 Markdown 示例，断言提取成功与完全不可提取时的失败分支。
"""

# 第三方库：用于断言异常。
import pytest

# 项目内模块：提供 role1/2/3/4 Markdown 契约提取函数。
from music_video_pipeline.modules.module_b.markdown_contracts import (
    ModuleBMarkdownContractError,
    parse_prompt_plans,
    parse_role1_visual_descriptions,
    parse_scene_plans,
    parse_shot_plans,
)


ROLE1_MARKDOWN = (
    "## 少女\n"
    "- pos_zh: 水手服少女，黑长直发，细瘦身形，百褶裙\n"
    "- pos_en: sailor uniform girl, long straight black hair, slim figure, pleated skirt\n\n"
    "## 黑猫\n"
    "- pos_zh: 瘦长黑猫，尖耳，细尾，短毛，紧绷背线\n"
    "- pos_en: slender black cat, pointed ears, thin tail, short fur, tense back line\n"
)

PLACEHOLDER_MARKDOWN = (
    "## 条目一\n"
    "- 占位1: 内容1\n"
    "- 占位2: 内容2\n"
    "- 占位3: 内容3\n\n"
    "## 条目二\n"
    "- 占位1: 内容4\n"
    "- 占位2: 内容5\n"
    "- 占位3: 内容6\n"
)


# ---------------------------------------------------------------------------
# role1 正常提取
# ---------------------------------------------------------------------------


def test_parse_role1_visual_descriptions_should_accept_visual_markdown() -> None:
    """验证 role1 轻量提取正常 Markdown 为标准结果数组。"""
    items = parse_role1_visual_descriptions(ROLE1_MARKDOWN)
    assert len(items) == 2
    assert items[0].imagery_name == "少女"
    assert items[0].pos_zh == "水手服少女，黑长直发，细瘦身形，百褶裙"
    assert items[1].imagery_name == "黑猫"
    assert items[1].pos_en == "slender black cat, pointed ears, thin tail, short fur, tense back line"


def test_parse_role1_visual_descriptions_should_accept_fenced_markdown() -> None:
    """验证 role1 支持外层 fenced code block 包裹的 Markdown——只看内部。"""
    markdown_text = f"```md\n{ROLE1_MARKDOWN}\n```"

    items = parse_role1_visual_descriptions(markdown_text)

    assert len(items) == 2
    assert items[0].imagery_name == "少女"
    assert items[1].imagery_name == "黑猫"


def test_parse_role1_visual_descriptions_should_ignore_outer_text_when_fenced() -> None:
    """验证有 ```md 时外部文本完全被忽略。"""
    markdown_text = (
        "前面多余的废话\n"
        "```md\n"
        "## 少女\n"
        "- pos_zh: 水手服少女\n"
        "- pos_en: sailor girl\n"
        "```\n"
        "后面多余的废话\n"
    )
    items = parse_role1_visual_descriptions(markdown_text)
    assert len(items) == 1
    assert items[0].imagery_name == "少女"


def test_parse_role1_visual_descriptions_should_accept_missing_field_with_warning(caplog) -> None:
    """验证 role1 缺失字段时不抛错，仅告警并用空字符串填充。"""
    markdown_text = (
        "## 少女\n"
        "- pos_zh: 水手服少女，黑长直发\n"
    )
    items = parse_role1_visual_descriptions(markdown_text)
    assert len(items) == 1
    assert items[0].imagery_name == "少女"
    assert items[0].pos_zh == "水手服少女，黑长直发"
    assert items[0].pos_en == ""
    assert "缺失字段" in caplog.text


def test_parse_role1_visual_descriptions_should_accept_extra_field_with_warning(caplog) -> None:
    """验证 role1 出现额外字段时不抛错，仅告警并忽略。"""
    markdown_text = (
        "## 少女\n"
        "- pos_zh: 水手服少女\n"
        "- pos_en: sailor girl\n"
        "- 未知字段: 多余内容\n"
    )
    items = parse_role1_visual_descriptions(markdown_text)
    assert len(items) == 1
    assert items[0].imagery_name == "少女"
    assert "未识别字段" in caplog.text


# ---------------------------------------------------------------------------
# role1 完全不可提取
# ---------------------------------------------------------------------------


def test_parse_role1_visual_descriptions_should_fail_on_empty_markdown() -> None:
    """验证 role1 对空字符串抛出异常。"""
    with pytest.raises(ModuleBMarkdownContractError, match="不能为空"):
        parse_role1_visual_descriptions("   ")


def test_parse_role1_visual_descriptions_should_fail_when_no_h2_sections() -> None:
    """验证 role1 对没有 ## 条目的文本抛出异常。"""
    with pytest.raises(ModuleBMarkdownContractError, match="至少包含一个"):
        parse_role1_visual_descriptions("只有一段普通文本，没有标题。")


def test_parse_role1_visual_descriptions_should_fail_on_empty_fenced_md() -> None:
    """验证 role1 对空的 fenced md 内部抛出异常。"""
    with pytest.raises(ModuleBMarkdownContractError, match="不能为空"):
        parse_role1_visual_descriptions("```md\n\n```")


# ---------------------------------------------------------------------------
# role2/3/4 占位提取
# ---------------------------------------------------------------------------


def test_parse_scene_plans_should_accept_placeholder_markdown() -> None:
    """验证 role2 轻量提取占位 Markdown 为 ScenePlan 数组。"""
    plans = parse_scene_plans(PLACEHOLDER_MARKDOWN)
    assert len(plans) == 2
    assert plans[0].占位1 == "内容1"
    assert plans[0].占位2 == "内容2"
    assert plans[0].占位3 == "内容3"
    assert plans[1].占位1 == "内容4"


def test_parse_shot_plans_should_accept_extra_field_with_warning(caplog) -> None:
    """验证 role3 出现额外字段时不抛错，仅告警。"""
    markdown_text = (
        "## 条目一\n"
        "- 占位1: 内容1\n"
        "- 占位2: 内容2\n"
        "- 占位3: 内容3\n"
        "- 占位4: 内容4\n"
    )

    plans = parse_shot_plans(markdown_text)
    assert len(plans) == 1
    assert plans[0].占位1 == "内容1"
    assert "未识别字段" in caplog.text


def test_parse_prompt_plans_should_accept_missing_field_with_warning(caplog) -> None:
    """验证 role4 缺失字段时不抛错，仅告警并用空字符串填充。"""
    markdown_text = (
        "## 条目一\n"
        "- 占位1: 内容1\n"
        "- 占位2: 内容2\n"
    )

    plans = parse_prompt_plans(markdown_text)
    assert len(plans) == 1
    assert plans[0].占位1 == "内容1"
    assert plans[0].占位2 == "内容2"
    assert plans[0].占位3 == ""
    assert "缺失字段" in caplog.text


def test_parse_prompt_plans_should_fail_on_empty_markdown() -> None:
    """验证 role4 对空字符串抛出异常。"""
    with pytest.raises(ModuleBMarkdownContractError, match="不能为空"):
        parse_prompt_plans("   ")
