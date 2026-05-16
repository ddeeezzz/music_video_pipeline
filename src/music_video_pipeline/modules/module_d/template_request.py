"""
文件用途：定义模块 D 模板渲染请求的数据结构与 JSON 落盘工具。
核心流程：构建正式模板请求对象 -> 校验最小字段 -> 序列化为稳定 JSON。
输入输出：输入模板请求数据类，输出可供 Remotion 消费的 JSON 文件。
依赖说明：依赖标准库 dataclasses/json/pathlib。
维护说明：当前覆盖 CenterTemplate、GridTemplate 与 ScrollTemplate 正式请求。
"""

# 标准库：用于数据类定义与序列化。
from dataclasses import asdict, dataclass
# 标准库：用于 JSON 序列化。
import json
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于字面量类型提示。
from typing import Literal


# 常量：正式支持的背景类型。
BackgroundKind = Literal["none", "solid", "image", "video"]


@dataclass(frozen=True)
class BackgroundRequest:
    """
    功能说明：表示模板请求中的背景定义。
    参数说明：
    - kind: 背景类型。
    - color: 纯色背景颜色；仅 kind=solid 时使用。
    - src: 图片或视频资源路径；仅 kind=image/video 时使用。
    返回值：不适用。
    异常说明：
    - ValueError: 字段组合非法时抛出。
    边界条件：背景由调用方明确给定，模板层不补默认风格。
    """

    kind: BackgroundKind
    color: str | None = None
    src: str | None = None

    def __post_init__(self) -> None:
        """
        功能说明：校验背景字段组合是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段组合非法时抛出。
        边界条件：none 不允许附带 color/src；solid 必须有 color；image/video 必须有 src。
        """
        if self.kind == "none":
            if self.color is not None or self.src is not None:
                raise ValueError("模板背景请求非法：kind=none 时不允许同时提供 color 或 src。")
            return
        if self.kind == "solid":
            if not str(self.color or "").strip():
                raise ValueError("模板背景请求非法：kind=solid 时必须提供 color。")
            if self.src is not None:
                raise ValueError("模板背景请求非法：kind=solid 时不允许提供 src。")
            return
        if self.kind in {"image", "video"}:
            if not str(self.src or "").strip():
                raise ValueError("模板背景请求非法：kind=image/video 时必须提供 src。")
            if self.color is not None:
                raise ValueError("模板背景请求非法：kind=image/video 时不允许提供 color。")
            return
        raise ValueError(f"模板背景请求非法：不支持的背景类型 kind={self.kind}")

    def to_dict(self) -> dict[str, str]:
        """
        功能说明：将背景请求转换为稳定字典。
        参数说明：无。
        返回值：
        - dict[str, str]: 仅包含合法字段的背景字典。
        异常说明：无。
        边界条件：空字段不会写入结果字典。
        """
        payload: dict[str, str] = {"kind": self.kind}
        if self.color is not None:
            payload["color"] = str(self.color).strip()
        if self.src is not None:
            payload["src"] = str(self.src).strip()
        return payload


@dataclass(frozen=True)
class SymbolRequest:
    """
    功能说明：表示模板请求中的单个视觉符号资源。
    参数说明：
    - src: 符号资源路径。
    - width_ratio: 符号宽度相对画面宽度比例。
    - height_ratio: 符号高度相对画面高度比例。
    返回值：不适用。
    异常说明：
    - ValueError: 字段非法时抛出。
    边界条件：尺寸比例用于暴露最终贴图大小调节入口。
    """

    src: str
    width_ratio: float
    height_ratio: float

    def __post_init__(self) -> None:
        """
        功能说明：校验符号资源路径与尺寸比例是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段非法时抛出。
        边界条件：尺寸比例必须位于 0 到 1 之间。
        """
        if not str(self.src).strip():
            raise ValueError("模板符号请求非法：src 不得为空。")
        if not 0 < float(self.width_ratio) <= 1:
            raise ValueError("模板符号请求非法：width_ratio 必须位于 0 到 1 之间。")
        if not 0 < float(self.height_ratio) <= 1:
            raise ValueError("模板符号请求非法：height_ratio 必须位于 0 到 1 之间。")


@dataclass(frozen=True)
class CenterMotionRequest:
    """
    功能说明：表示 CenterTemplate 的运动参数。
    参数说明：
    - breathe: 是否启用呼吸动画。
    返回值：不适用。
    异常说明：无。
    边界条件：频次当前仍由音频节拍侧驱动，不在契约层暴露。
    """

    breathe: bool


@dataclass(frozen=True)
class CenterTemplateRequest:
    """
    功能说明：表示正式的 CenterTemplate 渲染请求。
    参数说明：
    - template: 模板标识，固定为 center。
    - fps: 输出帧率。
    - duration_in_frames: 输出总帧数。
    - bpm: 音乐 BPM。
    - background: 背景请求。
    - symbol: 单个视觉符号请求。
    - motion: 运动请求。
    返回值：不适用。
    异常说明：
    - ValueError: 关键数值字段非法时抛出。
    边界条件：画布宽高已在模板工程中固定为 512x320，不再由契约传入。
    """

    template: Literal["center"]
    fps: int
    duration_in_frames: int
    bpm: float
    background: BackgroundRequest
    symbol: SymbolRequest
    motion: CenterMotionRequest

    def __post_init__(self) -> None:
        """
        功能说明：校验 CenterTemplate 请求的最小数值约束。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 关键数值字段非法时抛出。
        边界条件：template 必须固定为 center。
        """
        if self.template != "center":
            raise ValueError(f"CenterTemplate 请求非法：template 必须为 center，当前为 {self.template}")
        if int(self.fps) <= 0:
            raise ValueError("CenterTemplate 请求非法：fps 必须大于 0。")
        if int(self.duration_in_frames) <= 0:
            raise ValueError("CenterTemplate 请求非法：duration_in_frames 必须大于 0。")
        if float(self.bpm) <= 0:
            raise ValueError("CenterTemplate 请求非法：bpm 必须大于 0。")

    def to_dict(self) -> dict[str, object]:
        """
        功能说明：将 CenterTemplate 请求转换为稳定字典。
        参数说明：无。
        返回值：
        - dict[str, object]: 可直接写入 JSON 的模板请求字典。
        异常说明：无。
        边界条件：嵌套数据类统一转为显式字典结构。
        """
        return {
            "template": self.template,
            "fps": int(self.fps),
            "duration_in_frames": int(self.duration_in_frames),
            "bpm": float(self.bpm),
            "background": self.background.to_dict(),
            "symbol": asdict(self.symbol),
            "motion": asdict(self.motion),
        }


@dataclass(frozen=True)
class GridLayoutRequest:
    """
    功能说明：表示 GridTemplate 的布局参数。
    参数说明：
    - visible_cell_count: 一张完整静态图应占用的等分格子数量。
    返回值：不适用。
    异常说明：
    - ValueError: 字段非法时抛出。
    边界条件：当前固定为三符号横向布局。
    """

    visible_cell_count: int

    def __post_init__(self) -> None:
        """
        功能说明：校验 GridTemplate 布局参数是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段非法时抛出。
        边界条件：visible_cell_count 必须大于 0。
        """
        if int(self.visible_cell_count) <= 0:
            raise ValueError("GridTemplate 布局请求非法：visible_cell_count 必须大于 0。")


@dataclass(frozen=True)
class GridMotionRequest:
    """
    功能说明：表示 GridTemplate 的运动参数。
    参数说明：
    - active_ratio: 三个格子完成跳出所占段落时长比例。
    - overshoot_ratio: 入场缩放超调比例。
    - enter_distance: 入场纵向位移距离。
    返回值：不适用。
    异常说明：
    - ValueError: 字段非法时抛出。
    边界条件：比例必须位于 0 到 1 之间。
    """

    active_ratio: float
    overshoot_ratio: float
    enter_distance: float

    def __post_init__(self) -> None:
        """
        功能说明：校验 GridTemplate 运动参数是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段非法时抛出。
        边界条件：active_ratio 必须位于 0 到 1 之间。
        """
        if not 0 < float(self.active_ratio) <= 1:
            raise ValueError("GridTemplate 运动请求非法：active_ratio 必须位于 0 到 1 之间。")
        if float(self.overshoot_ratio) < 0:
            raise ValueError("GridTemplate 运动请求非法：overshoot_ratio 不得小于 0。")
        if float(self.enter_distance) < 0:
            raise ValueError("GridTemplate 运动请求非法：enter_distance 不得小于 0。")


@dataclass(frozen=True)
class GridTemplateRequest:
    """
    功能说明：表示正式的 GridTemplate 渲染请求。
    参数说明：
    - template: 模板标识，固定为 grid。
    - fps: 输出帧率。
    - duration_in_frames: 输出总帧数。
    - bpm: 音乐 BPM。
    - background: 背景请求。
    - symbols: 三个视觉符号请求。
    - layout: 布局请求。
    - motion: 运动请求。
    返回值：不适用。
    异常说明：
    - ValueError: 关键数值字段非法时抛出。
    边界条件：当前正式契约要求恰好传入 3 个符号。
    """

    template: Literal["grid"]
    fps: int
    duration_in_frames: int
    bpm: float
    background: BackgroundRequest
    symbols: tuple[SymbolRequest, SymbolRequest, SymbolRequest]
    layout: GridLayoutRequest
    motion: GridMotionRequest

    def __post_init__(self) -> None:
        """
        功能说明：校验 GridTemplate 请求的最小数值约束。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 关键数值字段非法时抛出。
        边界条件：template 必须固定为 grid，symbols 固定为 3 项。
        """
        if self.template != "grid":
            raise ValueError(f"GridTemplate 请求非法：template 必须为 grid，当前为 {self.template}")
        if int(self.fps) <= 0:
            raise ValueError("GridTemplate 请求非法：fps 必须大于 0。")
        if int(self.duration_in_frames) <= 0:
            raise ValueError("GridTemplate 请求非法：duration_in_frames 必须大于 0。")
        if float(self.bpm) <= 0:
            raise ValueError("GridTemplate 请求非法：bpm 必须大于 0。")
        if len(tuple(self.symbols)) != 3:
            raise ValueError("GridTemplate 请求非法：symbols 必须恰好包含 3 个符号。")

    def to_dict(self) -> dict[str, object]:
        """
        功能说明：将 GridTemplate 请求转换为稳定字典。
        参数说明：无。
        返回值：
        - dict[str, object]: 可直接写入 JSON 的模板请求字典。
        异常说明：无。
        边界条件：symbols 按稳定顺序展开为数组。
        """
        return {
            "template": self.template,
            "fps": int(self.fps),
            "duration_in_frames": int(self.duration_in_frames),
            "bpm": float(self.bpm),
            "background": self.background.to_dict(),
            "symbols": [asdict(item) for item in self.symbols],
            "layout": asdict(self.layout),
            "motion": asdict(self.motion),
        }


@dataclass(frozen=True)
class ScrollLayoutRequest:
    """
    功能说明：表示 ScrollTemplate 的布局参数。
    参数说明：
    - visible_cell_count: 一张完整静态图应占用的等分格子数量。
    返回值：不适用。
    异常说明：
    - ValueError: 字段非法时抛出。
    边界条件：用于统一条带静态排布密度。
    """

    visible_cell_count: int

    def __post_init__(self) -> None:
        """
        功能说明：校验 ScrollTemplate 布局参数是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段非法时抛出。
        边界条件：visible_cell_count 必须大于 0。
        """
        if int(self.visible_cell_count) <= 0:
            raise ValueError("ScrollTemplate 布局请求非法：visible_cell_count 必须大于 0。")


@dataclass(frozen=True)
class ScrollMotionRequest:
    """
    功能说明：表示 ScrollTemplate 的运动参数。
    参数说明：
    - loop: 是否启用循环滚动。
    - loop_beats: 当 loop=true 时单轮循环跨越的拍数。
    返回值：不适用。
    异常说明：
    - ValueError: 字段非法时抛出。
    边界条件：loop=false 时不允许传入 loop_beats。
    """

    loop: bool
    loop_beats: float | None = None

    def __post_init__(self) -> None:
        """
        功能说明：校验 ScrollTemplate 运动参数是否合法。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 字段非法时抛出。
        边界条件：loop_beats 仅在循环模式下允许传入。
        """
        if not self.loop and self.loop_beats is not None:
            raise ValueError("ScrollTemplate 运动请求非法：loop=false 时不允许提供 loop_beats。")
        if self.loop_beats is not None and float(self.loop_beats) <= 0:
            raise ValueError("ScrollTemplate 运动请求非法：loop_beats 必须大于 0。")

    def to_dict(self) -> dict[str, object]:
        """
        功能说明：将 ScrollTemplate 运动请求转换为稳定字典。
        参数说明：无。
        返回值：
        - dict[str, object]: 仅包含合法字段的运动参数字典。
        异常说明：无。
        边界条件：loop_beats 缺省时不写入结果字典。
        """
        payload: dict[str, object] = {"loop": bool(self.loop)}
        if self.loop_beats is not None:
            payload["loop_beats"] = float(self.loop_beats)
        return payload


@dataclass(frozen=True)
class ScrollTemplateRequest:
    """
    功能说明：表示正式的 ScrollTemplate 渲染请求。
    参数说明：
    - template: 模板标识，固定为 scroll。
    - fps: 输出帧率。
    - duration_in_frames: 输出总帧数。
    - bpm: 音乐 BPM。
    - background: 背景请求。
    - symbols: 三个视觉符号请求。
    - layout: 布局请求。
    - motion: 运动请求。
    返回值：不适用。
    异常说明：
    - ValueError: 关键数值字段非法时抛出。
    边界条件：当前为三元素连续条带模板。
    """

    template: Literal["scroll"]
    fps: int
    duration_in_frames: int
    bpm: float
    background: BackgroundRequest
    symbols: tuple[SymbolRequest, SymbolRequest, SymbolRequest]
    layout: ScrollLayoutRequest
    motion: ScrollMotionRequest

    def __post_init__(self) -> None:
        """
        功能说明：校验 ScrollTemplate 请求的最小数值约束。
        参数说明：无。
        返回值：无。
        异常说明：
        - ValueError: 关键数值字段非法时抛出。
        边界条件：template 必须固定为 scroll，symbols 固定为 3 项。
        """
        if self.template != "scroll":
            raise ValueError(f"ScrollTemplate 请求非法：template 必须为 scroll，当前为 {self.template}")
        if int(self.fps) <= 0:
            raise ValueError("ScrollTemplate 请求非法：fps 必须大于 0。")
        if int(self.duration_in_frames) <= 0:
            raise ValueError("ScrollTemplate 请求非法：duration_in_frames 必须大于 0。")
        if float(self.bpm) <= 0:
            raise ValueError("ScrollTemplate 请求非法：bpm 必须大于 0。")
        if len(tuple(self.symbols)) != 3:
            raise ValueError("ScrollTemplate 请求非法：symbols 必须恰好包含 3 个符号。")

    def to_dict(self) -> dict[str, object]:
        """
        功能说明：将 ScrollTemplate 请求转换为稳定字典。
        参数说明：无。
        返回值：
        - dict[str, object]: 可直接写入 JSON 的模板请求字典。
        异常说明：无。
        边界条件：嵌套数据类统一转为显式字典结构。
        """
        return {
            "template": self.template,
            "fps": int(self.fps),
            "duration_in_frames": int(self.duration_in_frames),
            "bpm": float(self.bpm),
            "background": self.background.to_dict(),
            "symbols": [asdict(item) for item in self.symbols],
            "layout": asdict(self.layout),
            "motion": self.motion.to_dict(),
        }


def write_template_request_json(
    request: CenterTemplateRequest | GridTemplateRequest | ScrollTemplateRequest,
    output_path: Path,
) -> Path:
    """
    功能说明：将正式模板请求写入 JSON 文件。
    参数说明：
    - request: 正式模板请求对象。
    - output_path: 输出 JSON 路径。
    返回值：
    - Path: 实际写入路径。
    异常说明：
    - OSError: 目录创建或文件写入失败时抛出。
    边界条件：父目录不存在时自动创建。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
