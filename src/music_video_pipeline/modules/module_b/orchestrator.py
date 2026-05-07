"""
文件用途：实现模块 B v2 的统一编排入口。
核心流程：读取模块 A 输出并转入多角色 v2 主链执行。
输入输出：输入 RuntimeContext，输出模块 B 清单 JSON 路径。
依赖说明：依赖模块 B v2 子组件与 JSON 工具。
维护说明：模块 B 已固定为多角色 v2，不再保留旧单元生成分支。
"""

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块B v2 新编排入口
from music_video_pipeline.modules.module_b_v2 import run_module_b_v2
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output


def run_module_b(context: RuntimeContext):
    """
    功能说明：执行模块 B，并以最小视觉单元粒度支持断点重试。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 B 输出清单 JSON 路径。
    异常说明：输入脚本不存在、单元重试耗尽或输出不完整时抛出异常。
    边界条件：仅重跑 pending/failed/running 单元，done 单元直接复用。
    """
    context.logger.info("模块B开始执行，task_id=%s", context.task_id)

    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    return run_module_b_v2(context)
