"""
文件用途：提供模块 D 对 Remotion 模板工程的最小调用能力。
核心流程：解析本地 Remotion CLI 路径 -> 启动本地图片 HTTP 服务 ->
          替换 props JSON 中的 file:// URL 为 http:// -> 调用子进程输出片段。
输入输出：输入模板工程目录、Composition 标识、请求 JSON 与输出路径，输出渲染完成的 mp4 文件。
依赖说明：依赖标准库 os/pathlib/subprocess/http.server 与模块 D 正式请求文件。
"""

import os
from pathlib import Path
import subprocess
import http.server
import socketserver
import threading
import json
import time
import re


def render_template_segment(
    *,
    remotion_project_dir: Path,
    composition_id: str,
    props_json_path: Path,
    output_path: Path,
) -> None:
    normalized_project_dir = Path(remotion_project_dir).resolve()
    normalized_props_path = Path(props_json_path).resolve()
    normalized_output_path = Path(output_path).resolve()
    if not normalized_project_dir.exists():
        raise FileNotFoundError(f"Remotion 模板工程目录不存在：{normalized_project_dir}")
    if not normalized_props_path.exists():
        raise FileNotFoundError(f"模板请求 JSON 不存在：{normalized_props_path}")

    # 启动临时 HTTP 服务器服务图片文件，避免 Chrome 拦截 file:// URL
    httpd, http_port, http_root = _start_image_http_server(normalized_props_path)
    try:
        patched_props_path = _patch_props_with_http_urls(
            normalized_props_path, http_port=http_port, http_root=http_root
        )

        remotion_cli_path = _resolve_remotion_cli_path(normalized_project_dir)
        command = _build_remotion_render_command(
            remotion_cli_path=remotion_cli_path,
            remotion_project_dir=normalized_project_dir,
            composition_id=composition_id,
            props_json_path=patched_props_path,
            output_path=normalized_output_path,
        )
        normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        try:
            subprocess.run(
                command,
                cwd=str(normalized_project_dir),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200.0,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - t_start) * 1000
            raise RuntimeError(
                "Remotion 模板渲染超时（7200秒）："
                f"composition_id={composition_id}，output_path={normalized_output_path}，elapsed_ms={elapsed:.0f}"
            ) from None
        except subprocess.CalledProcessError as error:
            elapsed = (time.perf_counter() - t_start) * 1000
            stderr_text = str(error.stderr or "").strip()
            stdout_text = str(error.stdout or "").strip()
            detail_text = stderr_text or stdout_text or "无额外输出"
            raise RuntimeError(
                "Remotion 模板渲染失败："
                f"composition_id={composition_id}，output_path={normalized_output_path}，elapsed_ms={elapsed:.0f}，detail={detail_text}"
            ) from error

        elapsed = (time.perf_counter() - t_start) * 1000
        if not normalized_output_path.exists():
            raise RuntimeError(
                "Remotion 模板渲染失败：命令执行后未生成输出文件，"
                f"composition_id={composition_id}，output_path={normalized_output_path}，elapsed_ms={elapsed:.0f}"
            )
    finally:
        httpd.shutdown()
        httpd.server_close()
        _cleanup_patched_props(normalized_props_path)


def _resolve_remotion_cli_path(remotion_project_dir: Path) -> Path:
    bin_dir = remotion_project_dir / "node_modules" / ".bin"
    candidate_names = ["remotion.CMD", "remotion.cmd"] if os.name == "nt" else ["remotion"]
    for candidate_name in candidate_names:
        candidate_path = bin_dir / candidate_name
        if candidate_path.exists():
            return candidate_path.resolve()
    raise FileNotFoundError(
        "未找到 Remotion CLI 可执行文件。"
        f"请先在 {remotion_project_dir} 下安装依赖，预期路径={bin_dir}"
    )


def _build_remotion_render_command(
    *,
    remotion_cli_path: Path,
    remotion_project_dir: Path,
    composition_id: str,
    props_json_path: Path,
    output_path: Path,
) -> list[str]:
    normalized_composition_id = str(composition_id).strip()
    if not normalized_composition_id:
        raise ValueError("Remotion 渲染命令非法：composition_id 不得为空。")
    entry_path = remotion_project_dir / "src" / "index.ts"
    return [
        str(remotion_cli_path),
        "render",
        str(entry_path),
        normalized_composition_id,
        str(output_path),
        f"--props={props_json_path}",
    ]


# ---------------------------------------------------------------------------
# 图片 HTTP 服务（解决 Chrome 阻拦 file:// URL 问题）
# ---------------------------------------------------------------------------

def _find_http_root_from_props(props_json_path: Path) -> Path:
    """
    从 props JSON 中的 src 路径推导 HTTP 服务根目录。
    规则：所有 file:// URL 的最近公共目录。
    """
    props = json.loads(props_json_path.read_text(encoding="utf-8"))
    all_srcs = _collect_src_values(props)
    if not all_srcs:
        return props_json_path.parent.resolve()
    # 去重并找到公共前缀
    paths = []
    for s in all_srcs:
        p = s
        if p.startswith("file:///"):
            p = p[8:]  # 去掉 file:///
        paths.append(Path(p).resolve())
    common = _common_path_prefix(paths)
    if common:
        return common
    return props_json_path.parent.resolve()


def _collect_src_values(obj) -> list[str]:
    srcs = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "src" and isinstance(value, str) and value.strip():
                srcs.append(value.strip())
            else:
                srcs.extend(_collect_src_values(value))
    elif isinstance(obj, list):
        for item in obj:
            srcs.extend(_collect_src_values(item))
    return srcs


def _common_path_prefix(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    try:
        parts_list = [p.parts for p in paths]
        common = []
        for i in range(min(len(p) for p in parts_list)):
            if len(set(p[i] for p in parts_list)) == 1:
                common.append(parts_list[0][i])
            else:
                break
        if len(common) >= 1:
            return Path(*common)
        return None
    except Exception:
        return None


def _start_image_http_server(props_json_path: Path) -> tuple[socketserver.TCPServer, int, Path]:
    """启动临时 HTTP 服务器服务于 props 中的图片路径。"""
    http_root = _find_http_root_from_props(props_json_path)
    for port_attempt in range(10):
        base_port = 19100 + (abs(hash(str(props_json_path))) % 1000)
        port = base_port + port_attempt
        try:
            class _Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(http_root), **kwargs)
                def log_message(self, fmt, *args):
                    pass  # 不输出 HTTP 日志
            httpd = socketserver.TCPServer(("127.0.0.1", port), _Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            return httpd, port, http_root
        except OSError:
            continue
    raise RuntimeError("无法启动图片 HTTP 服务（端口已被占用）。")


def _patch_props_with_http_urls(props_json_path: Path, *, http_port: int, http_root: Path) -> Path:
    """将 props JSON 中所有 file:// URL 替换为 http://127.0.0.1:PORT 路径。"""
    raw = props_json_path.read_text(encoding="utf-8")
    root_str = str(http_root.resolve())

    def _replacer(m):
        quote_before = m.group(1) if m.lastindex >= 1 else ""
        raw_value = m.group(2) if m.lastindex >= 2 else ""
        if raw_value.startswith("file:///"):
            file_path = raw_value[7:]
            try:
                abs_p = Path(file_path).resolve()
                if root_str in str(abs_p):
                    relative = str(abs_p)[len(root_str):].lstrip("/")
                    return f'{quote_before}"http://127.0.0.1:{http_port}/{relative}"'
            except Exception:
                pass
        return m.group(0)

    patched = re.sub(r'("src"\s*:\s*)"([^"]*)"', _replacer, raw)
    patched_path = props_json_path.with_suffix(".patched.json")
    patched_path.write_text(patched, encoding="utf-8")
    return patched_path


def _cleanup_patched_props(original_props_path: Path) -> None:
    """清理渲染过程中生成的 .patched.json 文件。"""
    patched_path = original_props_path.with_suffix(".patched.json")
    try:
        if patched_path.exists():
            patched_path.unlink()
    except Exception:
        pass
