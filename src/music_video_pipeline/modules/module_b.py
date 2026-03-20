"""
文件用途：实现模块 B（视觉脚本生成）的 MVP 版本。
核心流程：读取模块 A 输出，调用分镜生成器并落盘模块 B JSON。
输入输出：输入 RuntimeContext，输出 ModuleBOutput JSON 路径。
依赖说明：依赖项目内脚本生成器工厂与 JSON 工具。
维护说明：接入真实 LLM 时只替换生成器，不改模块出口契约。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：分镜生成器工厂
from music_video_pipeline.generators import build_script_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output


def run_module_b(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 B 并输出分镜脚本 JSON。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 B 输出 JSON 路径。
    异常说明：输入文件缺失或契约不合法时抛异常。
    边界条件：生成器模式未知时自动降级为 mock。
    """
    context.logger.info("模块B开始执行，task_id=%s", context.task_id)
    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    generator = build_script_generator(mode=context.config.mode.script_generator, logger=context.logger)
    module_b_output = generator.generate(module_a_output=module_a_output)
    validate_module_b_output(module_b_output)

    output_path = context.artifacts_dir / "module_b_output.json"
    write_json(output_path, module_b_output)
    context.logger.info("模块B执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
