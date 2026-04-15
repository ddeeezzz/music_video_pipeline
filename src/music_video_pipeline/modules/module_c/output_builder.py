"""
文件用途：构建模块 C 输出 JSON 数据结构。
核心流程：将已完成单元 frame_items 组装为模块 C 标准产物对象。
输入输出：输入 task_id/frames_dir/frame_items，输出可写入 JSON 的字典对象。
依赖说明：依赖标准库 typing/pathlib。
维护说明：输出结构需保持与模块 D 读取逻辑兼容。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于类型提示
from typing import Any


def build_module_c_output(task_id: str, frames_dir: Path, frame_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    功能说明：组装模块 C 输出对象。
    参数说明：
    - task_id: 任务唯一标识。
    - frames_dir: 帧目录路径。
    - frame_items: 已完成单元的帧清单数组。
    返回值：
    - dict[str, Any]: 模块 C 输出数据对象。
    异常说明：无。
    边界条件：frame_items 由上层保证为稳定顺序。
    """
    return {
        "task_id": task_id,
        "frames_dir": str(frames_dir),
        "frame_items": frame_items,
    }
