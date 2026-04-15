"""
文件用途：聚合任务监督页面相关组件导出。
核心流程：对外暴露快照构建函数与监控服务类。
输入输出：输入运行时状态库与任务ID，输出可视化快照与服务实例。
依赖说明：依赖 monitoring.snapshot 与 monitoring.server。
维护说明：新增监控能力时需保持导出接口语义稳定。
"""

# 项目内模块：提供任务监督快照构建函数
from music_video_pipeline.monitoring.snapshot import build_task_monitor_snapshot
# 项目内模块：提供任务监督服务
from music_video_pipeline.monitoring.server import TaskMonitorService

__all__ = ["TaskMonitorService", "build_task_monitor_snapshot"]
