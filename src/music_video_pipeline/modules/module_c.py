"""
文件用途：实现模块 C（图像生成）的 MVP 版本。
核心流程：读取模块 B 分镜并生成占位关键帧及帧清单。
输入输出：输入 RuntimeContext，输出模块 C 清单 JSON 路径。
依赖说明：依赖项目内关键帧生成器工厂与 JSON 工具。
维护说明：后续接入真实扩散模型时保持输出清单结构不变。
"""

# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：运行上下文定义
from music_video_pipeline.context import RuntimeContext
# 项目内模块：关键帧生成器工厂
from music_video_pipeline.generators import build_frame_generator
# 项目内模块：JSON 工具
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：契约校验
from music_video_pipeline.types import validate_module_b_output


def run_module_c(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块 C 并生成关键帧与清单文件。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: 模块 C 输出清单 JSON 路径。
    异常说明：输入脚本不存在或图像写入失败时抛异常。
    边界条件：生成器未知模式时自动降级 mock。
    """
    context.logger.info("模块C开始执行，task_id=%s", context.task_id)
    module_b_path = context.artifacts_dir / "module_b_output.json"
    module_b_output = read_json(module_b_path)
    validate_module_b_output(module_b_output)

    frames_dir = context.artifacts_dir / "frames"
    generator = build_frame_generator(mode=context.config.mode.frame_generator, logger=context.logger)
    frame_items = generator.generate(
        shots=module_b_output,
        output_dir=frames_dir,
        width=context.config.mock.video_width,
        height=context.config.mock.video_height,
    )

    output_data = {
        "task_id": context.task_id,
        "frames_dir": str(frames_dir),
        "frame_items": frame_items,
    }
    output_path = context.artifacts_dir / "module_c_output.json"
    write_json(output_path, output_data)
    context.logger.info("模块C执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path
