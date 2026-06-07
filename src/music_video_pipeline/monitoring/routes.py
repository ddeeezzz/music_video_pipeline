"""
文件用途：监控服务路由常量定义。
输入输出：对外提供 HTTP/WebSocket 路径常量。
依赖说明：无外部依赖。
维护说明：新增接口时在此统一登记路径。
"""

# 常量：正式前端任务列表路由
TASK_LIST_ROUTE_PATH = "/tasks"
# 常量：正式前端静态资源路由前缀
WEB_APP_STATIC_ROUTE_PREFIX = "/app/"
# 常量：正式前端构建目录名
WEB_APP_BUILD_DIR_NAME = "app"
# 常量：正式前端入口文件名
WEB_APP_INDEX_FILE_NAME = "index.html"
# 常量：主页任务列表接口路径
TASK_LIST_API_PATH = "/api/tasks"
# 常量：任务详情接口路径
TASK_DETAIL_API_PATH = "/api/task"
# 常量：任务新建接口路径
TASK_CREATE_API_PATH = "/api/task/create"
# 常量：任务改名接口路径
TASK_RENAME_API_PATH = "/api/task/rename"
# 常量：任务复制接口路径
TASK_COPY_API_PATH = "/api/task/copy"
# 常量：任务强制重跑接口路径
TASK_RERUN_API_PATH = "/api/task/rerun"
# 常量：模块 B 页面数据接口路径
TASK_MODULE_B_API_PATH = "/api/module-b"
# 常量：模块 A 页面数据接口路径
TASK_MODULE_A_API_PATH = "/api/module-a"
# 常量：模块 A 联网歌词搜索接口路径
TASK_MODULE_A_SEARCH_LYRICS_API_PATH = "/api/module-a/search-lyrics"
# 常量：模块 A 联网歌词搜索 WebSocket 路径
TASK_MODULE_A_SEARCH_LYRICS_WS_PATH = "/ws/module-a/search-lyrics"
# 常量：模块 A 联网歌词选择接口路径
TASK_MODULE_A_SELECT_LYRICS_API_PATH = "/api/module-a/select-lyrics"
# 常量：模块 A 候选歌词详情接口路径
TASK_MODULE_A_CANDIDATE_LYRICS_API_PATH = "/api/module-a/candidate-lyrics"
# 常量：模块 A 可视化数据负载接口路径
TASK_MODULE_A_VISUALIZATION_PAYLOAD_API_PATH = "/api/module-a/visualization-payload"
# 常量：模块 B role 重跑接口路径
TASK_MODULE_B_RERUN_ROLE_API_PATH = "/api/module-b/rerun-role"
# 常量：模块 B role 内 segment 重跑接口路径
TASK_MODULE_B_RERUN_ROLE_SEGMENT_API_PATH = "/api/module-b/rerun-role-segment"
TASK_MODULE_B_REBUILD_OUTPUT_API_PATH = "/api/module-b/rebuild-output"
TASK_MODULE_B_RESUME_API_PATH = "/api/module-b/resume"
# 常量：模块 C 页面数据接口路径
TASK_MODULE_C_API_PATH = "/api/module-c"
# 常量：模块 C shot 重跑接口路径
TASK_MODULE_C_RERUN_SHOT_API_PATH = "/api/module-c/rerun-shot"
# 常量：模块 C 单帧重跑接口路径
TASK_MODULE_C_RERUN_FRAME_API_PATH = "/api/module-c/rerun-frame"
# 常量：模块 C 单元重建接口路径（从 module_b_output.json 重建 Module C 单元列表）
TASK_MODULE_C_REBUILD_UNITS_API_PATH = "/api/module-c/rebuild-units"
# 常量：模块 D 页面数据接口路径
TASK_MODULE_D_API_PATH = "/api/module-d"
# 常量：模块 D segment 视频文件元数据接口路径（直接扫描 segments 目录）
TASK_MODULE_D_SEGMENT_VIDEOS_API_PATH = "/api/module-d/segment-videos"
# 常量：模块 D segment 重跑接口路径
TASK_MODULE_D_RERUN_SEGMENT_API_PATH = "/api/module-d/rerun-segment"
# 常量：模块 D segment 首尾帧重跑接口路径
TASK_MODULE_D_RERUN_BOTH_FRAMES_API_PATH = "/api/module-d/rerun-both-frames"
# 常量：模块 D 批量重跑接口路径（frame_type=start|end|both）
TASK_MODULE_D_RERUN_MODULE_API_PATH = "/api/module-d/rerun-module"
# 常量：模块 B 活跃重跑子进程状态文件名
ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME = "active_module_b_rerun_process.json"
# 常量：模块 B 已完成重跑元数据文件名（持久化 duration_ms 等）
COMPLETED_MODULE_B_RERUN_META_FILE_NAME = "completed_module_b_rerun_meta.json"
