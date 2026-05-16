"""
文件用途：验证模块 B role1/2/3/4 的 Markdown 契约解析。
核心流程：构造最小 Markdown 示例，断言解析成功与字段校验失败分支。
输入输出：输入 pytest 测试上下文，输出断言结果。
依赖说明：依赖 pytest 与模块 B Markdown 契约实现。
维护说明：当 role 契约字段调整时需同步更新本测试。
"""

# 第三方库：用于断言异常。
import pytest

# 项目内模块：提供 role1/2/3/4 Markdown 契约解析函数。
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


def test_parse_role1_visual_descriptions_should_accept_visual_markdown() -> None:
    """
    功能说明：验证 role1 契约会把视觉 Markdown 解析成标准结果数组。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：二级标题会作为 imagery_name 写入结果。
    """
    items = parse_role1_visual_descriptions(ROLE1_MARKDOWN)
    assert len(items) == 2
    assert items[0].imagery_name == "少女"
    assert items[0].pos_zh == "水手服少女，黑长直发，细瘦身形，百褶裙"
    assert items[1].imagery_name == "黑猫"
    assert items[1].pos_en == "slender black cat, pointed ears, thin tail, short fur, tense back line"


def test_parse_scene_plans_should_accept_placeholder_markdown() -> None:
    """
    功能说明：验证 role2 契约会把占位 Markdown 解析成 ScenePlan 数组。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：标题文本仅承担条目边界，不进入模型字段。
    """
    plans = parse_scene_plans(PLACEHOLDER_MARKDOWN)
    assert len(plans) == 2
    assert plans[0].占位1 == "内容1"
    assert plans[0].占位2 == "内容2"
    assert plans[0].占位3 == "内容3"
    assert plans[1].占位1 == "内容4"


def test_parse_shot_plans_should_reject_extra_placeholder_field() -> None:
    """
    功能说明：验证 role3 契约会拒绝未声明的额外字段。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前只允许 `占位1/占位2/占位3`。
    """
    markdown_text = (
        "## 条目一\n"
        "- 占位1: 内容1\n"
        "- 占位2: 内容2\n"
        "- 占位3: 内容3\n"
        "- 占位4: 内容4\n"
    )

    with pytest.raises(ModuleBMarkdownContractError, match="未定义字段"):
        parse_shot_plans(markdown_text)


def test_parse_prompt_plans_should_reject_missing_placeholder_field() -> None:
    """
    功能说明：验证 role4 契约会拒绝缺失占位字段的条目。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：缺任何一个占位字段都应失败。
    """
    markdown_text = (
        "## 条目一\n"
        "- 占位1: 内容1\n"
        "- 占位2: 内容2\n"
    )

    with pytest.raises(ModuleBMarkdownContractError, match="缺失字段"):
        parse_prompt_plans(markdown_text)
