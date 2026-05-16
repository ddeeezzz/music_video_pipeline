"""
文件用途：验证模块 D 的正式模板请求与 Remotion 调用器的最小行为。
核心流程：构造 CenterTemplate 正式请求 -> 落盘 JSON -> 构建并执行渲染命令。
输入输出：输入临时目录与 monkeypatch，输出断言结果。
依赖说明：依赖 pytest 与模块 D 模板请求/Remotion 调用器实现。
维护说明：本文件只覆盖最小正式链路，不涉及真实 Remotion 渲染。
"""

# 标准库：用于 JSON 解析。
import json
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于子进程结果模拟。
import subprocess

# 项目内模块：模块 D Remotion 调用器。
from music_video_pipeline.modules.module_d import remotion_renderer
# 项目内模块：模块 D 正式模板请求定义。
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


def test_write_template_request_json_should_emit_formal_center_schema(tmp_path: Path) -> None:
    """
    功能说明：验证正式 CenterTemplate 请求可稳定写成 JSON 文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前只验证 center 正式请求，不扩展其他模板。
    """
    request = CenterTemplateRequest(
        template="center",
        fps=24,
        duration_in_frames=48,
        bpm=130,
        background=BackgroundRequest(kind="solid", color="#FFFFFF"),
        symbol=SymbolRequest(
            src="/fixtures/center-symbol.svg",
            width_ratio=0.42,
            height_ratio=0.42,
        ),
        motion=CenterMotionRequest(breathe=True),
    )
    output_path = tmp_path / "template_requests" / "shot_001.center.json"
    written_path = write_template_request_json(request, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["template"] == "center"
    assert payload["background"] == {"kind": "solid", "color": "#FFFFFF"}
    assert payload["symbol"] == {
        "src": "/fixtures/center-symbol.svg",
        "width_ratio": 0.42,
        "height_ratio": 0.42,
    }
    assert payload["motion"] == {"breathe": True}


def test_write_template_request_json_should_emit_formal_grid_schema(tmp_path: Path) -> None:
    """
    功能说明：验证正式 GridTemplate 请求可稳定写成 JSON 文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前要求固定输出三个符号。
    """
    request = GridTemplateRequest(
        template="grid",
        fps=24,
        duration_in_frames=48,
        bpm=130,
        background=BackgroundRequest(kind="solid", color="#FFFFFF"),
        symbols=(
            SymbolRequest(src="/fixtures/grid-a.svg", width_ratio=0.26, height_ratio=0.52),
            SymbolRequest(src="/fixtures/grid-b.svg", width_ratio=0.26, height_ratio=0.52),
            SymbolRequest(src="/fixtures/grid-c.svg", width_ratio=0.26, height_ratio=0.52),
        ),
        layout=GridLayoutRequest(visible_cell_count=3),
        motion=GridMotionRequest(active_ratio=0.45, overshoot_ratio=0.08, enter_distance=72),
    )
    output_path = tmp_path / "template_requests" / "shot_001.grid.json"
    written_path = write_template_request_json(request, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["template"] == "grid"
    assert payload["background"] == {"kind": "solid", "color": "#FFFFFF"}
    assert payload["symbols"] == [
        {"src": "/fixtures/grid-a.svg", "width_ratio": 0.26, "height_ratio": 0.52},
        {"src": "/fixtures/grid-b.svg", "width_ratio": 0.26, "height_ratio": 0.52},
        {"src": "/fixtures/grid-c.svg", "width_ratio": 0.26, "height_ratio": 0.52},
    ]
    assert payload["layout"] == {"visible_cell_count": 3}
    assert payload["motion"] == {"active_ratio": 0.45, "overshoot_ratio": 0.08, "enter_distance": 72}


def test_write_template_request_json_should_emit_formal_scroll_schema(tmp_path: Path) -> None:
    """
    功能说明：验证正式 ScrollTemplate 请求可稳定写成 JSON 文件。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：当前只验证单图横向滚动的最小正式请求。
    """
    request = ScrollTemplateRequest(
        template="scroll",
        fps=24,
        duration_in_frames=48,
        bpm=130,
        background=BackgroundRequest(kind="solid", color="#FFFFFF"),
        symbols=(
            SymbolRequest(src="/fixtures/scroll-symbol.svg", width_ratio=0.28, height_ratio=0.72),
            SymbolRequest(src="/fixtures/scroll-symbol.svg", width_ratio=0.28, height_ratio=0.72),
            SymbolRequest(src="/fixtures/scroll-symbol.svg", width_ratio=0.28, height_ratio=0.72),
        ),
        layout=ScrollLayoutRequest(visible_cell_count=3),
        motion=ScrollMotionRequest(loop=False),
    )
    output_path = tmp_path / "template_requests" / "shot_001.scroll.json"
    written_path = write_template_request_json(request, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["template"] == "scroll"
    assert payload["background"] == {"kind": "solid", "color": "#FFFFFF"}
    assert payload["symbols"] == [
        {"src": "/fixtures/scroll-symbol.svg", "width_ratio": 0.28, "height_ratio": 0.72},
        {"src": "/fixtures/scroll-symbol.svg", "width_ratio": 0.28, "height_ratio": 0.72},
        {"src": "/fixtures/scroll-symbol.svg", "width_ratio": 0.28, "height_ratio": 0.72},
    ]
    assert payload["layout"] == {"visible_cell_count": 3}
    assert payload["motion"] == {"loop": False}


def test_render_template_segment_should_call_local_remotion_cli(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 Python 侧会调用模板工程本地安装的 Remotion CLI。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的运行时打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过伪 subprocess.run 避免真实渲染。
    """
    project_dir = tmp_path / "remotion_templates"
    cli_dir = project_dir / "node_modules" / ".bin"
    cli_dir.mkdir(parents=True, exist_ok=True)
    cli_path = cli_dir / "remotion.CMD"
    cli_path.write_text("@echo off\n", encoding="utf-8")
    props_json_path = project_dir / "request.json"
    props_json_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "runs" / "segment_001.mp4"

    captured: dict[str, object] = {}

    def _fake_subprocess_run(command, cwd=None, check=None, capture_output=None, text=None, encoding=None, errors=None):
        captured["command"] = command
        captured["cwd"] = cwd
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(remotion_renderer.subprocess, "run", _fake_subprocess_run)

    remotion_renderer.render_template_segment(
        remotion_project_dir=project_dir,
        composition_id="CenterTemplate",
        props_json_path=props_json_path,
        output_path=output_path,
    )

    assert Path(str(captured["command"][0])).name.lower() == "remotion.cmd"
    assert captured["command"][1:] == [
        "render",
        str((project_dir / "src" / "index.ts")),
        "CenterTemplate",
        str(output_path.resolve()),
        f"--props={props_json_path.resolve()}",
    ]
    assert captured["cwd"] == str(project_dir.resolve())
    assert output_path.exists()
