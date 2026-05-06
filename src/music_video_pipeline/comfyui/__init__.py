"""
文件用途：导出 ComfyUI API 客户端与工作流契约工具。
核心流程：统一加载契约、渲染工作流、提交到 ComfyUI 服务并等待结果。
输入输出：无输入，输出可复用的客户端与契约函数符号。
依赖说明：依赖同目录下 client/contracts 子模块。
维护说明：模块 C/D 的 ComfyUI 接线应优先复用本目录能力，避免各自实现一套 HTTP 调用。
"""

# 项目内模块：导出 ComfyUI 客户端。
from music_video_pipeline.comfyui.client import ComfyUIClient, ComfyUIServiceOptions
# 项目内模块：导出工作流契约加载与绑定函数。
from music_video_pipeline.comfyui.contracts import (
    ComfyUIWorkflowContract,
    load_workflow_contract,
    render_workflow_from_contract,
)

__all__ = [
    "ComfyUIClient",
    "ComfyUIServiceOptions",
    "ComfyUIWorkflowContract",
    "load_workflow_contract",
    "render_workflow_from_contract",
]
