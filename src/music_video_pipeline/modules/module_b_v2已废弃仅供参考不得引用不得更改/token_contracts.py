"""
文件用途：定义模块B v2 提示词 token 结构辅助类型。
核心流程：提供角色1、角色4 与最终模块B输出复用的 token TypedDict。
输入输出：输入为上层字典对象，输出为结构化类型约束。
依赖说明：依赖标准库 typing。
维护说明：token 字段扩展时必须同步 parser、role4 与最终聚合逻辑。
"""

# 标准库：用于类型声明。
from typing import NotRequired, TypedDict


class PromptToken(TypedDict):
    """定义结构化提示词 token。"""

    id: str
    text: str
    weight: NotRequired[float | None]
