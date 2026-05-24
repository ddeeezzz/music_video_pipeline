"""
文件用途：提供模块 B 的统一编排入口。
核心流程：协调 role1 执行，并在完整编排未接通时显式报错。
输入输出：输入 RuntimeContext，输出 role1 产物路径或抛出未接通异常。
依赖说明：依赖运行上下文。
维护说明：编排顺序与上下游契约变更时需同步更新。
"""

# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于未定阶段的宽松参数占位。
from typing import Any

# 项目内模块：提供运行上下文对象。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：提供 role1 真实执行逻辑。
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber


class MultiRoleScriptGenerator:
    """
    功能说明：协调模块 B 多角色执行流程。
    参数说明：初始化时接收模块 B 运行依赖。
    返回值：不适用。
    异常说明：角色执行失败时向上抛出异常。
    边界条件：应保证角色执行顺序与数据流向一致。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def generate(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        功能说明：执行模块 B 生成流程。
        参数说明：接收编排所需的上下文与输入对象。
        返回值：
        - list[dict[str, Any]]: 模块 B 输出结果。
        异常说明：按具体实现定义。
        边界条件：输出结构应满足下游模块消费要求。
        """
        del args, kwargs
        raise NotImplementedError("module_b: orchestrator.generate is not implemented.")


def run_module_b(context: RuntimeContext):
    """
    功能说明：执行模块 B 顶层流程。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path | dict | object: 模块 B 主流程产物。
    异常说明：按具体实现定义。
    边界条件：当前仅允许真实执行 role1，不再允许用 role1 结果伪造 module_b_output.json。
    """
    role1_output_path = run_module_b_role1(context)
    raise NotImplementedError(
        "模块B完整编排尚未接通：role2/role3/role4 仍未实现，"
        "已禁止再使用 role1 结果占位生成 module_b_output.json。"
        f"当前仅已写出 role1 产物：{role1_output_path}"
    )


def run_module_b_role1(context: RuntimeContext) -> Path:
    """
    功能说明：仅执行模块 B 的 role1，并写出 role1 Markdown 产物。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: role1 Markdown 产物路径。
    异常说明：按 role1 真实执行逻辑定义。
    边界条件：不生成 module_b_output.json，不触碰模块 B 单元状态。
    """
    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_markdown = storyboard_template_path.read_text(encoding="utf-8")

    role1_describer = Role1ImageryDescriber(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )
    role1_items = role1_describer.generate(storyboard_template_markdown)
    output_path = _write_role1_markdown_output(context=context, role1_items=role1_items)
    context.logger.info(
        "模块B role1 执行完成，task_id=%s，role1_item_count=%s，artifact=%s",
        context.task_id,
        len(role1_items),
        output_path,
    )
    return output_path


def _resolve_project_root() -> Path:
    """
    功能说明：解析项目根目录路径。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：约定当前文件位于 src/music_video_pipeline/modules/module_b 下。
    """
    return Path(__file__).resolve().parents[4]


def _resolve_storyboard_template_path(context: RuntimeContext, project_root: Path) -> Path:
    """
    功能说明：解析模块 B 当前使用的编排模板路径。
    参数说明：
    - context: 运行上下文。
    - project_root: 项目根目录。
    返回值：
    - Path: 模板绝对路径。
    异常说明：
    - FileNotFoundError: 模板文件不存在时抛出异常。
    边界条件：相对路径默认相对于项目根目录解析。
    """
    raw_template_path = str(context.config.module_b.storyboard_template_file).strip()
    if not raw_template_path:
        raise FileNotFoundError("模块B执行失败：storyboard_template_file 不能为空。")
    template_path = Path(raw_template_path)
    resolved_template_path = template_path if template_path.is_absolute() else (project_root / template_path)
    resolved_template_path = resolved_template_path.resolve()
    if not resolved_template_path.exists():
        raise FileNotFoundError(f"模块B执行失败：未找到编排模板文件 {resolved_template_path}")
    return resolved_template_path


def _write_role1_markdown_output(context: RuntimeContext, role1_items: list[Any]) -> Path:
    """
    功能说明：把 role1 的结构化结果回写为 Markdown 主产物。
    参数说明：
    - context: 运行上下文。
    - role1_items: role1 视觉描述数组。
    返回值：
    - Path: Markdown 产物路径。
    异常说明：无。
    边界条件：当前固定写入 artifacts/module_b_role1_visual_output.md。
    """
    output_path = context.artifacts_dir / "module_b_role1_visual_output.md"
    markdown_lines: list[str] = []
    for item in role1_items:
        markdown_lines.extend(
            [
                f"## {str(getattr(item, 'name', '')).strip()}",
                f"- pos_zh: {str(getattr(item, 'pos_zh', '')).strip()}",
                f"- pos_en: {str(getattr(item, 'pos_en', '')).strip()}",
                "",
            ]
        )
    output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return output_path
