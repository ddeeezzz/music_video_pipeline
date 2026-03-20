"""
文件用途：定义流水线运行上下文对象。
核心流程：在模块间传递统一的任务级参数与依赖。
输入输出：输入构造参数，输出 RuntimeContext 实例。
依赖说明：依赖标准库 dataclasses/pathlib/logging。
维护说明：新增跨模块共享资源时优先扩展该对象。
"""

# 标准库：用于数据类定义
from dataclasses import dataclass
# 标准库：用于日志类型提示
import logging
# 标准库：用于路径类型提示
from pathlib import Path

# 项目内模块：提供配置类型
from music_video_pipeline.config import AppConfig
# 项目内模块：提供状态存储类型
from music_video_pipeline.state_store import StateStore


@dataclass
class RuntimeContext:
    """
    功能说明：封装单次任务运行所需的公共上下文。
    参数说明：
    - task_id: 任务唯一标识。
    - audio_path: 输入音频路径。
    - task_dir: 当前任务输出目录。
    - artifacts_dir: 当前任务产物目录。
    - config: 应用配置对象。
    - logger: 日志对象。
    - state_store: 状态数据库操作对象。
    返回值：不适用。
    异常说明：不适用。
    边界条件：目录路径应在使用前确保存在。
    """

    task_id: str
    audio_path: Path
    task_dir: Path
    artifacts_dir: Path
    config: AppConfig
    logger: logging.Logger
    state_store: StateStore
