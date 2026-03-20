"""
文件用途：提供通用文件与 JSON 读写工具。
核心流程：统一目录创建、JSON 序列化与反序列化行为。
输入输出：输入路径与数据，输出文件写入结果或读取对象。
依赖说明：依赖标准库 json/pathlib。
维护说明：所有文件读写默认使用 UTF-8 编码。
"""

# 标准库：用于 JSON 编解码
import json
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any


def ensure_dir(path: Path) -> None:
    """
    功能说明：确保目录存在，不存在则递归创建。
    参数说明：
    - path: 目标目录路径。
    返回值：无。
    异常说明：权限不足时抛出 OSError。
    边界条件：path 已存在时保持幂等。
    """
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    """
    功能说明：将对象序列化为 JSON 文件。
    参数说明：
    - path: JSON 输出路径。
    - data: 待写入的数据对象。
    返回值：无。
    异常说明：
    - TypeError: 数据不可序列化时抛出。
    - OSError: 文件写入失败时抛出。
    边界条件：会自动创建父目录。
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """
    功能说明：读取 JSON 文件并返回对象。
    参数说明：
    - path: JSON 文件路径。
    返回值：
    - Any: 反序列化后的 Python 对象。
    异常说明：
    - FileNotFoundError: 文件不存在。
    - json.JSONDecodeError: 文件内容非法。
    边界条件：文件编码必须为 UTF-8。
    """
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
