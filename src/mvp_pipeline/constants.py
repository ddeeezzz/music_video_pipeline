"""
文件用途：集中定义流水线常量，避免魔法字符串分散在代码中。
核心流程：声明模块执行顺序与合法状态集合。
输入输出：无输入，输出可复用常量。
依赖说明：仅依赖 Python 标准库 typing。
维护说明：新增模块时需同步更新 MODULE_ORDER。
"""

# 标准库：用于类型别名声明
from typing import Final

MODULE_ORDER: Final[list[str]] = ["A", "B", "C", "D"]
VALID_MODULES: Final[set[str]] = set(MODULE_ORDER)
TASK_STATES: Final[set[str]] = {"pending", "running", "done", "failed"}
