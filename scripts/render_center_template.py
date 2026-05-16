"""
文件用途：提供正式模板渲染命令的脚本入口包装。
核心流程：转调项目内正式 CLI 入口，便于从 scripts 路径直接执行。
输入输出：输入命令行参数，输出模板请求 JSON 与模板片段文件。
依赖说明：依赖项目内 music_video_pipeline.template_render_cli。
维护说明：脚本层不承载业务逻辑，只做入口转发。
"""

# 项目内模块：用于执行正式模板渲染命令。
from music_video_pipeline.template_render_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
