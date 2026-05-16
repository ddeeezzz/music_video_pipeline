"""
文件用途：提供正式模板渲染的命令行入口。
核心流程：解析参数 -> 构建正式模板请求 -> 写入正式 request JSON -> 调用 Remotion 输出 mp4。
输入输出：输入命令行参数，输出 request JSON 与模板片段文件。
依赖说明：依赖标准库 argparse/pathlib，以及模块 D 的模板请求与 Remotion 调用器。
维护说明：当前覆盖 center、grid 与 scroll 三条正式模板链路，不提前扩展更多模板。
"""

# 标准库：用于命令行参数解析。
import argparse
# 标准库：用于路径处理。
from pathlib import Path

# 项目内模块：用于正式模板请求定义与 JSON 落盘。
from music_video_pipeline.modules.module_d.template_request import (
    BackgroundRequest,
    CenterMotionRequest,
    CenterTemplateRequest,
    GridLayoutRequest,
    GridMotionRequest,
    GridTemplateRequest,
    ScrollLayoutRequest,
    ScrollMotionRequest,
    ScrollTemplateRequest,
    SymbolRequest,
    write_template_request_json,
)
# 项目内模块：用于调用本地 Remotion CLI 输出片段。
from music_video_pipeline.modules.module_d.remotion_renderer import render_template_segment


# 常量：默认输出任务目录名称。
DEFAULT_RUN_NAME = ""
# 常量：默认模板工程目录名称。
DEFAULT_REMOTION_DIR_NAME = "remotion_templates"
# 常量：默认内置符号素材路径。
DEFAULT_SYMBOL_SRC = "/fixtures/center-symbol.svg"
# 常量：默认 grid 模板三个内置符号素材路径。
DEFAULT_GRID_SYMBOL_SRC_LIST = [
    "/fixtures/grid-a.svg",
    "/fixtures/grid-b.svg",
    "/fixtures/grid-c.svg",
]
# 常量：默认 scroll 模板三个内置符号素材路径。
DEFAULT_SCROLL_SYMBOL_SRC_LIST = [
    "/fixtures/scroll-symbol.svg",
    "/fixtures/scroll-symbol.svg",
    "/fixtures/scroll-symbol.svg",
]
# 常量：默认帧率。
DEFAULT_FPS = 24
# 常量：默认 center 总帧数。
DEFAULT_CENTER_DURATION_IN_FRAMES = 48
# 常量：默认 grid 总帧数。
DEFAULT_GRID_DURATION_IN_FRAMES = 84
# 常量：默认 scroll 总帧数。
DEFAULT_SCROLL_DURATION_IN_FRAMES = 144
# 常量：默认 BPM。
DEFAULT_BPM = 130.0
# 常量：默认居中贴图宽度比例。
DEFAULT_CENTER_WIDTH_RATIO = 0.42
# 常量：默认居中贴图高度比例。
DEFAULT_CENTER_HEIGHT_RATIO = 0.42
# 常量：默认 center 呼吸开关。
DEFAULT_CENTER_BREATHE = True
# 常量：默认 grid 单图宽度比例。
DEFAULT_GRID_WIDTH_RATIO = 0.26
# 常量：默认 grid 单图高度比例。
DEFAULT_GRID_HEIGHT_RATIO = 0.52
# 常量：默认 grid 静态格子数。
DEFAULT_GRID_VISIBLE_CELL_COUNT = 3
# 常量：默认 grid 动画激活时长比例。
DEFAULT_GRID_ACTIVE_RATIO = 0.45
# 常量：默认 grid 入场缩放超调比例。
DEFAULT_GRID_OVERSHOOT_RATIO = 0.08
# 常量：默认 grid 入场纵向位移距离。
DEFAULT_GRID_ENTER_DISTANCE = 72
# 常量：默认 scroll 单图宽度比例。
DEFAULT_SCROLL_WIDTH_RATIO = 0.28
# 常量：默认 scroll 单图高度比例。
DEFAULT_SCROLL_HEIGHT_RATIO = 0.72
# 常量：默认 scroll 静态格子数。
DEFAULT_SCROLL_VISIBLE_CELL_COUNT = 3
# 常量：默认 scroll 是否循环。
DEFAULT_SCROLL_LOOP = False


def _default_run_name_for_template(template_name: str) -> str:
    """
    功能说明：根据模板名返回默认 runs 子目录名。
    参数说明：
    - template_name: 模板标识。
    返回值：
    - str: 默认输出目录名。
    异常说明：无。
    边界条件：未知模板名回退为统一目录名。
    """
    normalized = str(template_name).strip().lower()
    if normalized == "grid":
        return "grid_template_render"
    if normalized == "scroll":
        return "scroll_template_render"
    if normalized == "center":
        return "center_template_render"
    return "template_render"


def build_argument_parser() -> argparse.ArgumentParser:
    """
    功能说明：构建正式模板渲染命令的参数解析器。
    参数说明：无。
    返回值：
    - argparse.ArgumentParser: 参数解析器对象。
    异常说明：无。
    边界条件：默认值覆盖最小正式链路所需参数。
    """
    parser = argparse.ArgumentParser(description="生成正式模板 request 并渲染 mp4。")
    parser.add_argument(
        "--template",
        default="center",
        choices=["center", "grid", "scroll"],
        help="模板标识，默认 center。",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="输出 runs 子目录名；留空时按模板自动生成默认目录名。",
    )
    parser.add_argument(
        "--symbol-src",
        default=DEFAULT_SYMBOL_SRC,
        help="CenterTemplate 符号资源路径，可写 public 目录路径或绝对文件路径。",
    )
    parser.add_argument(
        "--background-kind",
        default="solid",
        choices=["none", "solid", "image", "video"],
        help="背景类型，默认 solid。",
    )
    parser.add_argument(
        "--background-color",
        default="#FFFFFF",
        help="当背景类型为 solid 时使用的颜色值。",
    )
    parser.add_argument(
        "--background-src",
        default="",
        help="当背景类型为 image/video 时使用的背景资源路径。",
    )
    parser.add_argument(
        "--symbol-src-list",
        nargs="*",
        default=[],
        help="GridTemplate 或 ScrollTemplate 使用的三个符号资源路径；留空时使用内置 fixtures。",
    )
    return parser


def build_background_request(*, kind: str, color: str, src: str) -> BackgroundRequest:
    """
    功能说明：根据命令行参数构建正式背景请求。
    参数说明：
    - kind: 背景类型。
    - color: 背景颜色。
    - src: 背景资源路径。
    返回值：
    - BackgroundRequest: 背景请求对象。
    异常说明：
    - ValueError: 参数组合非法时由 BackgroundRequest 抛出。
    边界条件：不同背景类型只填充其实际需要的字段。
    """
    if kind == "none":
        return BackgroundRequest(kind="none")
    if kind == "solid":
        return BackgroundRequest(kind="solid", color=str(color).strip())
    return BackgroundRequest(kind=kind, src=str(src).strip())


def render_center_template(
    *,
    project_root: Path,
    run_name: str = DEFAULT_RUN_NAME,
    symbol_src: str = DEFAULT_SYMBOL_SRC,
    background_kind: str = "solid",
    background_color: str = "#FFFFFF",
    background_src: str = "",
) -> dict[str, str]:
    """
    功能说明：生成正式 CenterTemplate 请求并调用 Remotion 输出片段。
    参数说明：
    - project_root: 项目根目录。
    - run_name: 输出 runs 子目录名。
    - symbol_src: 符号资源路径。
    - background_kind: 背景类型。
    - background_color: 纯色背景颜色。
    - background_src: 图片或视频背景资源路径。
    返回值：
    - dict[str, str]: 渲染摘要（request_path/output_path/run_dir）。
    异常说明：
    - 运行失败时向上抛出异常。
    边界条件：当前固定输出一份 shot_001.center.json 与 segment_001.mp4。
    """
    runs_dir = project_root / "runs"
    run_dir = runs_dir / str(run_name).strip()
    request_path = run_dir / "artifacts" / "template_requests" / "shot_001.center.json"
    output_path = run_dir / "segments" / "segment_001.mp4"
    remotion_project_dir = project_root / DEFAULT_REMOTION_DIR_NAME

    print(f"开始构建模板请求，run_dir={run_dir}")
    request = CenterTemplateRequest(
        template="center",
        fps=DEFAULT_FPS,
        duration_in_frames=DEFAULT_CENTER_DURATION_IN_FRAMES,
        bpm=DEFAULT_BPM,
        background=build_background_request(
            kind=str(background_kind).strip(),
            color=str(background_color).strip(),
            src=str(background_src).strip(),
        ),
        symbol=SymbolRequest(
            src=str(symbol_src).strip(),
            width_ratio=DEFAULT_CENTER_WIDTH_RATIO,
            height_ratio=DEFAULT_CENTER_HEIGHT_RATIO,
        ),
        motion=CenterMotionRequest(breathe=DEFAULT_CENTER_BREATHE),
    )

    print(f"写入模板请求 JSON，path={request_path}")
    written_request_path = write_template_request_json(request, request_path)

    print(f"开始调用 Remotion 渲染，output={output_path}")
    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id="CenterTemplate",
        props_json_path=written_request_path,
        output_path=output_path,
    )

    print(f"模板请求已写入：{written_request_path}")
    print(f"视频片段已生成：{output_path}")
    return {
        "run_dir": str(run_dir),
        "request_path": str(written_request_path),
        "output_path": str(output_path),
    }


def render_grid_template(
    *,
    project_root: Path,
    run_name: str = "",
    symbol_src_list: list[str] | tuple[str, str, str] | None = None,
    background_kind: str = "solid",
    background_color: str = "#FFFFFF",
    background_src: str = "",
) -> dict[str, str]:
    """
    功能说明：生成正式 GridTemplate 请求并调用 Remotion 输出片段。
    参数说明：
    - project_root: 项目根目录。
    - run_name: 输出 runs 子目录名。
    - symbol_src_list: 三个符号资源路径。
    - background_kind: 背景类型。
    - background_color: 纯色背景颜色。
    - background_src: 图片或视频背景资源路径。
    返回值：
    - dict[str, str]: 渲染摘要（request_path/output_path/run_dir）。
    异常说明：
    - ValueError: 符号数量不为 3 时抛出。
    边界条件：当前固定输出一份 shot_001.grid.json 与 segment_001.mp4。
    """
    normalized_run_name = str(run_name).strip() or _default_run_name_for_template("grid")
    normalized_symbol_src_list = [
        str(item).strip() for item in (symbol_src_list or DEFAULT_GRID_SYMBOL_SRC_LIST) if str(item).strip()
    ]
    if len(normalized_symbol_src_list) != 3:
        raise ValueError("GridTemplate 渲染参数非法：symbol_src_list 必须恰好包含 3 个路径。")

    runs_dir = project_root / "runs"
    run_dir = runs_dir / normalized_run_name
    request_path = run_dir / "artifacts" / "template_requests" / "shot_001.grid.json"
    output_path = run_dir / "segments" / "segment_001.mp4"
    remotion_project_dir = project_root / DEFAULT_REMOTION_DIR_NAME

    print(f"开始构建模板请求，run_dir={run_dir}")
    request = GridTemplateRequest(
        template="grid",
        fps=DEFAULT_FPS,
        duration_in_frames=DEFAULT_GRID_DURATION_IN_FRAMES,
        bpm=DEFAULT_BPM,
        background=build_background_request(
            kind=str(background_kind).strip(),
            color=str(background_color).strip(),
            src=str(background_src).strip(),
        ),
        symbols=(
            SymbolRequest(
                src=normalized_symbol_src_list[0],
                width_ratio=DEFAULT_GRID_WIDTH_RATIO,
                height_ratio=DEFAULT_GRID_HEIGHT_RATIO,
            ),
            SymbolRequest(
                src=normalized_symbol_src_list[1],
                width_ratio=DEFAULT_GRID_WIDTH_RATIO,
                height_ratio=DEFAULT_GRID_HEIGHT_RATIO,
            ),
            SymbolRequest(
                src=normalized_symbol_src_list[2],
                width_ratio=DEFAULT_GRID_WIDTH_RATIO,
                height_ratio=DEFAULT_GRID_HEIGHT_RATIO,
            ),
        ),
        layout=GridLayoutRequest(visible_cell_count=DEFAULT_GRID_VISIBLE_CELL_COUNT),
        motion=GridMotionRequest(
            active_ratio=DEFAULT_GRID_ACTIVE_RATIO,
            overshoot_ratio=DEFAULT_GRID_OVERSHOOT_RATIO,
            enter_distance=DEFAULT_GRID_ENTER_DISTANCE,
        ),
    )

    print(f"写入模板请求 JSON，path={request_path}")
    written_request_path = write_template_request_json(request, request_path)

    print(f"开始调用 Remotion 渲染，output={output_path}")
    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id="GridTemplate",
        props_json_path=written_request_path,
        output_path=output_path,
    )

    print(f"模板请求已写入：{written_request_path}")
    print(f"视频片段已生成：{output_path}")
    return {
        "run_dir": str(run_dir),
        "request_path": str(written_request_path),
        "output_path": str(output_path),
    }


def render_scroll_template(
    *,
    project_root: Path,
    run_name: str = "",
    symbol_src: str = DEFAULT_SYMBOL_SRC,
    symbol_src_list: list[str] | tuple[str, str, str] | None = None,
    background_kind: str = "solid",
    background_color: str = "#FFFFFF",
    background_src: str = "",
) -> dict[str, str]:
    """
    功能说明：生成正式 ScrollTemplate 请求并调用 Remotion 输出片段。
    参数说明：
    - project_root: 项目根目录。
    - run_name: 输出 runs 子目录名。
    - symbol_src: 单个符号资源路径；当未显式传入 symbol_src_list 时用于兜底复制。
    - symbol_src_list: 三个条带符号资源路径。
    - background_kind: 背景类型。
    - background_color: 纯色背景颜色。
    - background_src: 图片或视频背景资源路径。
    返回值：
    - dict[str, str]: 渲染摘要（request_path/output_path/run_dir）。
    异常说明：
    - ValueError: 符号数量不为 3 时抛出。
    边界条件：当前固定输出一份 shot_001.scroll.json 与 segment_001.mp4。
    """
    normalized_run_name = str(run_name).strip() or _default_run_name_for_template("scroll")
    normalized_symbol_src_list = [
        str(item).strip()
        for item in (
            symbol_src_list
            or [str(symbol_src).strip(), str(symbol_src).strip(), str(symbol_src).strip()]
            or DEFAULT_SCROLL_SYMBOL_SRC_LIST
        )
        if str(item).strip()
    ]
    if len(normalized_symbol_src_list) != 3:
        raise ValueError("ScrollTemplate 渲染参数非法：symbol_src_list 必须恰好包含 3 个路径。")
    runs_dir = project_root / "runs"
    run_dir = runs_dir / normalized_run_name
    request_path = run_dir / "artifacts" / "template_requests" / "shot_001.scroll.json"
    output_path = run_dir / "segments" / "segment_001.mp4"
    remotion_project_dir = project_root / DEFAULT_REMOTION_DIR_NAME

    print(f"开始构建模板请求，run_dir={run_dir}")
    request = ScrollTemplateRequest(
        template="scroll",
        fps=DEFAULT_FPS,
        duration_in_frames=DEFAULT_SCROLL_DURATION_IN_FRAMES,
        bpm=DEFAULT_BPM,
        background=build_background_request(
            kind=str(background_kind).strip(),
            color=str(background_color).strip(),
            src=str(background_src).strip(),
        ),
        symbols=(
            SymbolRequest(
                src=normalized_symbol_src_list[0],
                width_ratio=DEFAULT_SCROLL_WIDTH_RATIO,
                height_ratio=DEFAULT_SCROLL_HEIGHT_RATIO,
            ),
            SymbolRequest(
                src=normalized_symbol_src_list[1],
                width_ratio=DEFAULT_SCROLL_WIDTH_RATIO,
                height_ratio=DEFAULT_SCROLL_HEIGHT_RATIO,
            ),
            SymbolRequest(
                src=normalized_symbol_src_list[2],
                width_ratio=DEFAULT_SCROLL_WIDTH_RATIO,
                height_ratio=DEFAULT_SCROLL_HEIGHT_RATIO,
            ),
        ),
        layout=ScrollLayoutRequest(visible_cell_count=DEFAULT_SCROLL_VISIBLE_CELL_COUNT),
        motion=ScrollMotionRequest(loop=DEFAULT_SCROLL_LOOP),
    )

    print(f"写入模板请求 JSON，path={request_path}")
    written_request_path = write_template_request_json(request, request_path)

    print(f"开始调用 Remotion 渲染，output={output_path}")
    render_template_segment(
        remotion_project_dir=remotion_project_dir,
        composition_id="ScrollTemplate",
        props_json_path=written_request_path,
        output_path=output_path,
    )

    print(f"模板请求已写入：{written_request_path}")
    print(f"视频片段已生成：{output_path}")
    return {
        "run_dir": str(run_dir),
        "request_path": str(written_request_path),
        "output_path": str(output_path),
    }


def main() -> int:
    """
    功能说明：执行正式模板的本地 request 生成与 Remotion 渲染。
    参数说明：无。
    返回值：
    - int: 进程退出码，成功返回 0。
    异常说明：
    - 运行失败时向上抛出异常，由命令行直接显示错误。
    边界条件：当前根据 template 分支写出 center、grid 或 scroll 的正式请求与片段。
    """
    args = build_argument_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    template_name = str(args.template).strip()
    if template_name == "grid":
        render_grid_template(
            project_root=project_root,
            run_name=str(args.run_name).strip(),
            symbol_src_list=[str(item).strip() for item in args.symbol_src_list],
            background_kind=str(args.background_kind).strip(),
            background_color=str(args.background_color).strip(),
            background_src=str(args.background_src).strip(),
        )
        return 0
    if template_name == "scroll":
        render_scroll_template(
            project_root=project_root,
            run_name=str(args.run_name).strip(),
            symbol_src=str(args.symbol_src).strip(),
            symbol_src_list=[str(item).strip() for item in args.symbol_src_list],
            background_kind=str(args.background_kind).strip(),
            background_color=str(args.background_color).strip(),
            background_src=str(args.background_src).strip(),
        )
        return 0

    render_center_template(
        project_root=project_root,
        run_name=str(args.run_name).strip() or _default_run_name_for_template("center"),
        symbol_src=str(args.symbol_src).strip(),
        background_kind=str(args.background_kind).strip(),
        background_color=str(args.background_color).strip(),
        background_src=str(args.background_src).strip(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
