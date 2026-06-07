"""
文件用途：提供任务 Web 页面的轻量HTTP+WebSocket服务。
核心流程：后台线程启动异步服务，提供任务列表与详情接口，并为单任务页面推送实时快照。
输入输出：输入 StateStore 与可选默认 task_id，输出可访问URL与实时状态流。
依赖说明：依赖标准库 asyncio/threading 与第三方 websockets。
维护说明：支持“任务列表总览 + 指定任务预选”两种入口，保持最小可维护实现。
"""

# 标准库：用于异步协程与事件循环
import asyncio
# 标准库：用于识别端口占用错误码
import errno
# 标准库：用于状态快照JSON序列化
import json
# 标准库：用于日志记录
import logging
# 标准库：用于MIME类型推断
import mimetypes
# 标准库：用于HTTP状态码常量
from http import HTTPStatus
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于从文件名中提取 segment_id
import re
# 标准库：用于子进程状态探测与终止
import subprocess
# 标准库：用于后台线程
import threading
# 标准库：用于时间戳
import time
# 标准库：用于URL解析与编码
from urllib.parse import parse_qs, quote, unquote, urlparse
# 标准库：用于类型提示
from typing import Any, Callable

# 项目内模块：任务监督快照构建
from music_video_pipeline.monitoring.snapshot import build_task_monitor_snapshot
# 项目内模块：模块A 手动联网歌词查找
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.pipeline import (
    search_synced_lrc_candidates,
    stream_synced_lrc_candidates,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.kugou_music import (
    fetch_kugou_music_lyrics_bundle,
    fetch_kugou_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.netease_music import (
    fetch_netease_music_lyrics_bundle,
    fetch_netease_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.qq_music import (
    fetch_qq_music_lyrics_bundle,
    fetch_qq_music_synced_lyrics,
)
from music_video_pipeline.modules.module_a_v2.lrclib_client import query_lrclib_lyrics
# 项目内模块：模块A 联网歌词状态
from music_video_pipeline.modules.module_a_v2.network_lyrics_state import (
    current_time_text,
    load_module_a_network_lyrics_state,
    write_module_a_network_lyrics_state,
)
# 项目内模块：模块A 音频时长探测
from music_video_pipeline.modules.module_a_v2.utils.media_probe import probe_audio_duration
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore
# 项目内模块：任务音频路径回映射
from music_video_pipeline.task_audio_path import resolve_task_audio_path

try:
    # 第三方库：提供WebSocket与HTTP复用服务
    import websockets
    # 第三方库：WebSocket连接关闭异常
    from websockets.exceptions import ConnectionClosed
except Exception:  # noqa: BLE001
    websockets = None
    ConnectionClosed = Exception

from music_video_pipeline.monitoring.routes import (
    TASK_COPY_API_PATH,
    TASK_CREATE_API_PATH,
    TASK_DETAIL_API_PATH,
    TASK_LIST_API_PATH,
    TASK_LIST_ROUTE_PATH,
    TASK_MODULE_A_API_PATH,
    TASK_MODULE_A_CANDIDATE_LYRICS_API_PATH,
    TASK_MODULE_A_SEARCH_LYRICS_API_PATH,
    TASK_MODULE_A_SEARCH_LYRICS_WS_PATH,
    TASK_MODULE_A_SELECT_LYRICS_API_PATH,
    TASK_MODULE_B_API_PATH,
    TASK_MODULE_B_REBUILD_OUTPUT_API_PATH,
    TASK_MODULE_B_RERUN_ROLE_API_PATH,
    TASK_MODULE_B_RERUN_ROLE_SEGMENT_API_PATH,
    TASK_MODULE_B_RESUME_API_PATH,
    TASK_MODULE_C_API_PATH,
    TASK_MODULE_C_RERUN_FRAME_API_PATH,
    TASK_MODULE_C_REBUILD_UNITS_API_PATH,
    TASK_MODULE_C_RERUN_SHOT_API_PATH,
    TASK_MODULE_D_API_PATH,
    TASK_MODULE_D_SEGMENT_VIDEOS_API_PATH,
    TASK_MODULE_D_RERUN_SEGMENT_API_PATH,
    TASK_MODULE_D_RERUN_BOTH_FRAMES_API_PATH,
    TASK_MODULE_D_RERUN_MODULE_API_PATH,
    TASK_RENAME_API_PATH,
    TASK_RERUN_API_PATH,
    WEB_APP_BUILD_DIR_NAME,
    WEB_APP_INDEX_FILE_NAME,
    WEB_APP_STATIC_ROUTE_PREFIX,
)


# 类型别名：用于触发任务强制重跑的回调函数。

# handler mixin
from music_video_pipeline.monitoring.handlers.module_a import ModuleAHandlers
from music_video_pipeline.monitoring.handlers.module_b import ModuleBHandlers
from music_video_pipeline.monitoring.handlers.module_c import ModuleCHandlers
from music_video_pipeline.monitoring.handlers.module_d import ModuleDHandlers
from music_video_pipeline.monitoring.handlers.tasks import TaskHandlers
from music_video_pipeline.monitoring.handlers.review import ReviewHandlers
TaskRerunHandler = Callable[[str], dict[str, Any]]
# 类型别名：用于触发模块 B role 级重跑的回调函数。
ModuleBRoleRerunHandler = Callable[[str, str], dict[str, Any]]
# 类型别名：用于触发模块 B role 内 segment 级重跑的回调函数。
ModuleBRoleSegmentRerunHandler = Callable[[str, str, str], dict[str, Any]]
# 类型别名：用于触发模块 C shot 级重跑的回调函数。
ModuleCShotRerunHandler = Callable[[str, str], dict[str, Any]]
# 类型别名：用于触发模块 C 单帧重跑的回调函数。
ModuleCFrameRerunHandler = Callable[[str, str, str], dict[str, Any]]
# 类型别名：用于触发模块 D segment 重跑的回调函数。
ModuleDSegmentRerunHandler = Callable[[str, str, str], dict[str, Any]]


class TaskMonitorService(
    ModuleAHandlers,
    ModuleBHandlers,
    ModuleCHandlers,
    ModuleDHandlers,
    TaskHandlers,
    ReviewHandlers,
):
    """
    功能说明：封装任务 Web 服务的生命周期。
    参数说明：
    - state_store: 任务状态存储对象。
    - task_id: 当前默认任务唯一标识；为空时进入任务列表页。
    - logger: 日志对象。
    - host: HTTP/WS 监听地址。
    - port: HTTP/WS 起始监听端口（0表示自动分配；正整数被占用时自动顺延）。
    - tick_seconds: 快照推送与终态轮询间隔（秒）。
    返回值：不适用。
    异常说明：启动失败时抛 RuntimeError。
    边界条件：服务仅绑定本地地址，默认不对外网暴露。
    """

    def __init__(
        self,
        state_store: StateStore,
        task_id: str,
        logger: logging.Logger,
        rerun_handler: TaskRerunHandler | None = None,
        module_b_role_rerun_handler: ModuleBRoleRerunHandler | None = None,
        module_b_role_segment_rerun_handler: ModuleBRoleSegmentRerunHandler | None = None,
        module_c_shot_rerun_handler: ModuleCShotRerunHandler | None = None,
        module_c_frame_rerun_handler: ModuleCFrameRerunHandler | None = None,
        module_d_segment_rerun_handler: ModuleDSegmentRerunHandler | None = None,
        app_config: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        tick_seconds: float = 1.0,
        auto_stop_on_terminal: bool = True,
        frontend_build_dir: Path | None = None,
    ) -> None:
        """
        功能说明：初始化监督服务对象。
        参数说明：
        - state_store: 状态存储对象。
        - task_id: 任务唯一标识。
        - logger: 日志对象。
        - rerun_handler: 可选的任务强制重跑回调。
        - module_b_role_rerun_handler: 可选的模块 B role 级重跑回调。
        - module_b_role_segment_rerun_handler: 可选的模块 B role 内 segment 级重跑回调。
        - app_config: 可选的运行配置对象（模块 A 联网歌词检索使用）。
        - host: 监听地址。
        - port: 起始监听端口。
        - tick_seconds: 推送间隔秒数。
        - auto_stop_on_terminal: 任务进入终态且无页面连接时是否自动停止服务。
        返回值：无。
        异常说明：无。
        边界条件：不在构造阶段启动网络监听。
        """
        self.state_store = state_store
        self.task_id = task_id
        self.logger = logger
        self.rerun_handler = rerun_handler
        self.module_b_role_rerun_handler = module_b_role_rerun_handler
        self.module_b_role_segment_rerun_handler = module_b_role_segment_rerun_handler
        self.module_c_shot_rerun_handler = module_c_shot_rerun_handler
        self.module_c_frame_rerun_handler = module_c_frame_rerun_handler
        self.module_d_segment_rerun_handler = module_d_segment_rerun_handler
        self.app_config = app_config
        self.host = host
        self.port = int(port)
        self.tick_seconds = max(0.2, float(tick_seconds))
        self.auto_stop_on_terminal = bool(auto_stop_on_terminal)
        self.frontend_build_dir = (
            frontend_build_dir.resolve()
            if isinstance(frontend_build_dir, Path)
            else Path(__file__).resolve().parent / "static" / WEB_APP_BUILD_DIR_NAME
        )
        self._bound_port = 0

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._async_stop_event: asyncio.Event | None = None
        self._connections: set[Any] = set()
        self._task_terminal = False
        self._rerun_threads: dict[str, threading.Thread] = {}
        self._rerun_thread_meta: dict[str, dict[str, Any]] = {}

        self._started_event = threading.Event()
        self._startup_error: Exception | None = None

    def _log_task_scope(self) -> str:
        """
        功能说明：返回用于日志展示的任务作用域文本。
        参数说明：无。
        返回值：
        - str: 具体 task_id，或通用列表模式标识。
        异常说明：无。
        边界条件：空 task_id 统一展示为 all-tasks。
        """
        normalized_task_id = str(self.task_id).strip()
        return normalized_task_id or "all-tasks"

    @staticmethod
    def _is_address_in_use_error(error: OSError) -> bool:
        """
        功能说明：判断启动失败是否属于端口占用。
        参数说明：
        - error: 启动监听时抛出的 OSError。
        返回值：
        - bool: 端口占用返回 True，否则 False。
        异常说明：无。
        边界条件：同时兼容常见 Unix errno 与 Windows winerror。
        """
        if int(getattr(error, "errno", -1)) == int(errno.EADDRINUSE):
            return True
        if int(getattr(error, "winerror", -1)) == 10048:
            return True
        error_text = str(error).lower()
        return ("address already in use" in error_text) or ("only one usage of each socket address" in error_text)

    async def _start_server_with_port_fallback(self) -> None:
        """
        功能说明：按起始端口启动 HTTP/WS 服务，端口被占用时自动顺延。
        参数说明：无。
        返回值：无。
        异常说明：
        - OSError: 绑定过程中遇到非端口占用错误时透传。
        - RuntimeError: 起始端口之后已无可用端口时抛出。
        边界条件：port<=0 时保持交给系统自动分配，不参与顺延。
        """
        start_port = int(self.port)
        if start_port <= 0:
            self._server = await websockets.serve(
                self._handle_websocket,
                self.host,
                self.port,
                process_request=self._process_request,
                ping_interval=20,
                ping_timeout=None,
            )
            sockets = list(self._server.sockets or [])
            if sockets:
                self._bound_port = int(sockets[0].getsockname()[1])
            return

        for candidate_port in range(start_port, 65536):
            try:
                self._server = await websockets.serve(
                    self._handle_websocket,
                    self.host,
                    candidate_port,
                    process_request=self._process_request,
                    ping_interval=20,
                    ping_timeout=None,
                )
                sockets = list(self._server.sockets or [])
                self._bound_port = int(sockets[0].getsockname()[1]) if sockets else int(candidate_port)
                if candidate_port != start_port:
                    self.logger.warning(
                        "[监督服务] 监听端口 %s 已被占用，已自动顺延到 %s",
                        start_port,
                        self._bound_port,
                    )
                return
            except OSError as error:
                if not self._is_address_in_use_error(error):
                    raise
                continue

        raise RuntimeError(f"任务 Web 服务启动失败：从端口 {start_port} 开始直到 65535 均不可用。")

    @property
    def monitor_url(self) -> str:
        """
        功能说明：返回任务前端页面URL。
        参数说明：无。
        返回值：
        - str: 前端页面地址。
        异常说明：无。
        边界条件：服务未启动时端口为初始化值。
        """
        port = self._bound_port or self.port
        return f"http://{self.host}:{port}{self._build_task_monitor_route(task_id=self.task_id)}"

    @property
    def is_running(self) -> bool:
        """
        功能说明：判断监督服务线程是否仍在运行。
        参数说明：无。
        返回值：
        - bool: 运行中返回 True，否则 False。
        异常说明：无。
        边界条件：线程存在且存活才视为运行中。
        """
        return self._thread is not None and self._thread.is_alive()

    def websocket_url_for(self, task_id: str | None = None) -> str:
        """
        功能说明：返回指定任务的WebSocket连接URL。
        参数说明：
        - task_id: 可选任务ID，未传时默认当前任务。
        返回值：
        - str: WebSocket URL。
        异常说明：无。
        边界条件：服务未启动时端口为初始化值。
        """
        target_task_id = str(task_id or self.task_id).strip() or self.task_id
        port = self._bound_port or self.port
        return f"ws://{self.host}:{port}/ws?task_id={quote(target_task_id)}"

    def start(self) -> None:
        """
        功能说明：启动监督服务。
        参数说明：无。
        返回值：无。
        异常说明：
        - RuntimeError: 启动失败或超时时抛出。
        边界条件：重复调用保持幂等，不重复启动线程。
        """
        if self.is_running:
            return
        if websockets is None:
            raise RuntimeError("任务监督服务启动失败：缺少 websockets 依赖。")

        self._startup_error = None
        self._started_event.clear()
        thread_scope = self._log_task_scope()
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"task-monitor-{thread_scope}",
            daemon=True,
        )
        self._thread.start()
        if not self._started_event.wait(timeout=5.0):
            raise RuntimeError("任务监督服务启动超时。")
        if self._startup_error:
            raise RuntimeError(f"任务监督服务启动失败：{self._startup_error}")

    def stop(self) -> None:
        """
        功能说明：停止监督服务并回收后台线程。
        参数说明：无。
        返回值：无。
        异常说明：无。
        边界条件：重复调用保持幂等。
        """
        if not self._thread:
            return
        if self._loop and self._async_stop_event:
            self._loop.call_soon_threadsafe(self._async_stop_event.set)
        self.wait_until_stopped(timeout_seconds=5.0)

    def wait_until_stopped(self, timeout_seconds: float | None = None) -> bool:
        """
        功能说明：阻塞等待监督服务线程退出。
        参数说明：
        - timeout_seconds: 最长等待秒数，None 表示一直等待。
        返回值：
        - bool: True 表示服务已停止，False 表示超时仍在运行。
        异常说明：无。
        边界条件：若线程不存在，直接返回 True。
        """
        if not self._thread:
            return True
        self._thread.join(timeout=timeout_seconds)
        stopped = not self._thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def _thread_main(self) -> None:
        """
        功能说明：后台线程入口，运行异步监督服务。
        参数说明：无。
        返回值：无。
        异常说明：异常会记录到 _startup_error 并触发启动完成事件。
        边界条件：退出时总会关闭事件循环。
        """
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async_server())
        except Exception as error:  # noqa: BLE001
            self._startup_error = error
            self._started_event.set()
        finally:
            pending = asyncio.all_tasks(loop=loop)
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None

    async def _run_async_server(self) -> None:
        """
        功能说明：启动HTTP/WS复用服务并等待停止信号。
        参数说明：无。
        返回值：无。
        异常说明：服务绑定失败时抛异常到线程入口。
        边界条件：任务进入done/failed时自动触发停止。
        """
        self._async_stop_event = asyncio.Event()
        self._task_terminal = False
        await self._start_server_with_port_fallback()
        self._started_event.set()
        self.logger.info("[监督服务] 任务 Web 服务已启动，task_id=%s，地址=%s", self._log_task_scope(), self.monitor_url)

        watcher_task = asyncio.create_task(self._watch_task_status_until_terminal())
        await self._async_stop_event.wait()
        watcher_task.cancel()
        await asyncio.gather(watcher_task, return_exceptions=True)
        await self._close_all_connections()

        self._server.close()
        await self._server.wait_closed()
        self.logger.info("[监督服务] 任务 Web 服务已停止，task_id=%s", self._log_task_scope())

    async def _watch_task_status_until_terminal(self) -> None:
        """
        功能说明：轮询任务状态，命中终态后停止监督服务。
        参数说明：无。
        返回值：无。
        异常说明：无。
        边界条件：未预选任务时仅空转等待停止信号。
        """
        if not self._async_stop_event:
            return
        while not self._async_stop_event.is_set():
            if not str(self.task_id).strip():
                await asyncio.sleep(self.tick_seconds)
                continue
            snapshot = build_task_monitor_snapshot(state_store=self.state_store, task_id=self.task_id)
            task_status = str(snapshot.get("task_status", "unknown"))
            if task_status in {"done", "failed"}:
                if not self._task_terminal:
                    self._task_terminal = True
                    if self.auto_stop_on_terminal:
                        self.logger.info(
                            "任务Web服务检测到任务终态，等待页面连接关闭后停止，task_id=%s，status=%s，active_connections=%s",
                            self.task_id,
                            task_status,
                            len(self._connections),
                        )
                    else:
                        self.logger.info(
                            "任务Web服务检测到任务终态（手动模式不自动停止），task_id=%s，status=%s",
                            self.task_id,
                            task_status,
                        )
                if self.auto_stop_on_terminal and not self._connections:
                    self._async_stop_event.set()
                    return
            await asyncio.sleep(self.tick_seconds)

    async def _close_all_connections(self) -> None:
        """
        功能说明：关闭当前全部WebSocket连接。
        参数说明：无。
        返回值：无。
        异常说明：无。
        边界条件：关闭失败会被吞掉，避免阻塞服务退出。
        """
        if not self._connections:
            return
        for connection in list(self._connections):
            try:
                await connection.close(code=1001, reason="task-monitor-stop")
            except Exception:  # noqa: BLE001
                continue

    async def _handle_websocket(self, websocket: Any, path: str) -> None:
        """
        功能说明：处理WebSocket连接并周期推送任务快照。
        参数说明：
        - websocket: 当前连接对象。
        - path: 请求路径（含查询串）。
        返回值：无。
        异常说明：连接异常中断时自动退出循环。
        边界条件：默认使用URL中的 task_id；缺失时回退当前任务ID。
        """
        parsed = urlparse(path)
        if parsed.path == TASK_MODULE_A_SEARCH_LYRICS_WS_PATH:
            await self._handle_module_a_search_lyrics_socket(websocket=websocket, parsed=parsed)
            return
        if parsed.path != "/ws":
            await websocket.close(code=1008, reason="unsupported_path")
            return
        query = parse_qs(parsed.query)
        target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        self._connections.add(websocket)
        try:
            while self._async_stop_event and not self._async_stop_event.is_set():
                snapshot = build_task_monitor_snapshot(state_store=self.state_store, task_id=target_task_id)
                await websocket.send(json.dumps(snapshot, ensure_ascii=False))
                await asyncio.sleep(self.tick_seconds)
        except ConnectionClosed:
            return
        finally:
            self._connections.discard(websocket)
            if (
                self.auto_stop_on_terminal
                and self._task_terminal
                and not self._connections
                and self._async_stop_event
                and not self._async_stop_event.is_set()
            ):
                self._async_stop_event.set()




    async def _process_request(self, path: str, _request_headers: Any) -> Any:
        """
        功能说明：在同端口处理简易HTTP请求（监督页与健康检查）。
        参数说明：
        - path: 请求路径（含查询串）。
        - _request_headers: 请求头对象（当前无需使用）。
        返回值：
        - Any: websockets 约定的HTTP响应三元组或 None。
        异常说明：无。
        边界条件：返回 None 时交由WebSocket握手流程继续处理。
        """
        parsed = urlparse(path)
        if parsed.path in {"/ws", TASK_MODULE_A_SEARCH_LYRICS_WS_PATH}:
            return None
        if parsed.path == "/" or parsed.path == "":
            location = TASK_LIST_ROUTE_PATH
            return self._build_http_response(
                status=HTTPStatus.FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="redirect",
                extra_headers=[("Location", location)],
            )
        if parsed.path == "/healthz":
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text='{"ok": true}',
            )
        if parsed.path == "/snapshot":
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            snapshot = build_task_monitor_snapshot(state_store=self.state_store, task_id=target_task_id)
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(snapshot, ensure_ascii=False),
            )
        if parsed.path == TASK_LIST_API_PATH:
            payload = self._build_task_list_payload()
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_DETAIL_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_task_detail_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_CREATE_API_PATH:
            payload, status = self._handle_create_task_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_RENAME_API_PATH:
            payload, status = self._handle_rename_task_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_COPY_API_PATH:
            payload, status = self._handle_copy_task_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_RERUN_API_PATH:
            payload, status = self._handle_rerun_task_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_A_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_module_a_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_A_SEARCH_LYRICS_API_PATH:
            payload, status = self._handle_module_a_search_lyrics_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_A_SELECT_LYRICS_API_PATH:
            payload, status = self._handle_module_a_select_lyrics_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_A_CANDIDATE_LYRICS_API_PATH:
            payload, status = self._handle_module_a_candidate_lyrics_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_B_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_module_b_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_B_RERUN_ROLE_API_PATH:
            payload, status = self._handle_module_b_role_rerun_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_B_RERUN_ROLE_SEGMENT_API_PATH:
            payload, status = self._handle_module_b_role_segment_rerun_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_B_REBUILD_OUTPUT_API_PATH:
            payload, status = self._handle_module_b_rebuild_output_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_B_RESUME_API_PATH:
            payload, status = self._handle_module_b_resume_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_C_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_module_c_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_C_RERUN_SHOT_API_PATH:
            payload, status = self._handle_module_c_shot_rerun_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_C_RERUN_FRAME_API_PATH:
            payload, status = self._handle_module_c_frame_rerun_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_C_REBUILD_UNITS_API_PATH:
            payload, status = self._handle_module_c_rebuild_units_request(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_D_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_module_d_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_D_SEGMENT_VIDEOS_API_PATH:
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_module_d_segment_videos_payload(task_id=target_task_id)
            if not payload.get("ok", False):
                return self._build_http_response(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    body_text=json.dumps(payload, ensure_ascii=False),
                )
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_D_RERUN_SEGMENT_API_PATH:
            payload, status = self._handle_module_d_rerun_segment(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_D_RERUN_BOTH_FRAMES_API_PATH:
            payload, status = self._handle_module_d_rerun_both_frames(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_MODULE_D_RERUN_MODULE_API_PATH:
            payload, status = self._handle_module_d_rerun_module(parsed=parsed)
            return self._build_http_response(
                status=status,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == "/web-data":
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_web_payload(task_id=target_task_id)
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == TASK_LIST_ROUTE_PATH or parsed.path.startswith(f"{TASK_LIST_ROUTE_PATH}/"):
            return self._build_frontend_app_entry_response()
        if parsed.path.startswith(WEB_APP_STATIC_ROUTE_PREFIX):
            return self._build_frontend_static_response(path=parsed.path, request_headers=_request_headers)
        if parsed.path.startswith("/task/"):
            return self._build_task_file_response(path=parsed.path, request_headers=_request_headers)
        return self._build_http_response(
            status=HTTPStatus.NOT_FOUND,
            content_type="text/plain; charset=utf-8",
            body_text="not found",
        )

    def _build_task_route(self, task_id: str) -> str:
        """
        功能说明：构建当前任务的新前端详情根路由。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - str: `/tasks/<task_id>` 形式的新前端路由。
        异常说明：无。
        边界条件：task_id 为空时回退任务列表页。
        """
        normalized_task_id = str(task_id).strip()
        if not normalized_task_id:
            return TASK_LIST_ROUTE_PATH
        return f"{TASK_LIST_ROUTE_PATH}/{quote(normalized_task_id)}"

    def _build_task_monitor_route(self, task_id: str) -> str:
        """
        功能说明：构建当前任务的新前端监督页路由。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - str: `/tasks/<task_id>/monitor` 形式的新前端路由。
        异常说明：无。
        边界条件：task_id 为空时回退任务列表页。
        """
        task_route = self._build_task_route(task_id=task_id)
        if task_route == TASK_LIST_ROUTE_PATH:
            return task_route
        return f"{task_route}/monitor"

    def _load_frontend_app_html(self) -> str:
        """
        功能说明：读取正式前端构建后的 HTML 入口文件。
        参数说明：无。
        返回值：
        - str: 前端入口 HTML 文本。
        异常说明：
        - FileNotFoundError: 构建产物缺失时抛出。
        边界条件：不再回退旧 task_monitor.html。
        """
        index_path = (self.frontend_build_dir / WEB_APP_INDEX_FILE_NAME).resolve()
        if (not index_path.exists()) or (not index_path.is_file()):
            raise FileNotFoundError(index_path)
        return index_path.read_text(encoding="utf-8")

    def _build_frontend_app_entry_response(self) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：返回正式前端 SPA 的 HTML 入口。
        参数说明：无。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: 前端入口响应。
        异常说明：无；缺失构建产物时转为 503 提示页。
        边界条件：不再回退旧监督页模板。
        """
        try:
            html_text = self._load_frontend_app_html()
        except FileNotFoundError as error:
            missing_path = str(error.args[0]) if error.args else str(self.frontend_build_dir / WEB_APP_INDEX_FILE_NAME)
            self.logger.error("[监督服务] 正式前端构建产物缺失，path=%s", missing_path)
            return self._build_http_response(
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                content_type="text/html; charset=utf-8",
                body_text=(
                    "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
                    "<title>前端未构建</title></head><body>"
                    "<h1>正式前端尚未构建</h1>"
                    f"<p>未找到构建产物：{missing_path}</p>"
                    "<p>请先在 <code>src/music_video_pipeline/web_frontend</code> 目录执行 "
                    "<code>npm install</code> 与 <code>npm run build</code>。</p>"
                    "</body></html>"
                ),
            )
        return self._build_http_response(
            status=HTTPStatus.OK,
            content_type="text/html; charset=utf-8",
            body_text=html_text,
        )

    def _build_frontend_static_response(
        self,
        path: str,
        request_headers: Any,
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：从正式前端构建目录中返回静态资源文件。
        参数说明：
        - path: 请求路径。
        - request_headers: HTTP 请求头对象。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: 静态资源响应。
        异常说明：无；非法路径统一转为 404。
        边界条件：仅允许访问 frontend_build_dir 内文件。
        """
        raw_parts = [unquote(part) for part in str(path).split("/") if part]
        if len(raw_parts) < 2:
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        relative_parts = raw_parts[1:]
        target_path = self.frontend_build_dir.joinpath(*relative_parts).resolve()
        try:
            target_path.relative_to(self.frontend_build_dir)
        except ValueError:
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        if (not target_path.exists()) or (not target_path.is_file()):
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        content_type = mimetypes.guess_type(str(target_path))[0] or "application/octet-stream"
        return self._build_file_http_response(
            file_path=target_path,
            content_type=content_type,
            request_headers=request_headers,
        )














    @staticmethod
    def _build_text_file_asset(file_path: Path | None) -> dict[str, Any]:
        """
        功能说明：把文本文件包装为前端可直接展示的文本资产对象。
        参数说明：
        - file_path: 文本文件路径。
        返回值：
        - dict[str, Any]: 包含 available/path/content 的对象。
        异常说明：无；读取失败时统一返回 available=false。
        边界条件：内容仅按 UTF-8 读取。
        """
        if file_path is None or (not file_path.exists()) or (not file_path.is_file()):
            return {"available": False, "path": str(file_path) if file_path else "", "content": ""}
        try:
            content_text = file_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return {"available": False, "path": str(file_path), "content": ""}
        return {"available": True, "path": str(file_path), "content": content_text}

    def _build_task_text_file_asset(self, task_id: str, file_path: Path | None) -> dict[str, Any]:
        """
        功能说明：把任务目录内文本文件包装为前端可展示的文本资产对象。
        参数说明：
        - task_id: 任务唯一标识。
        - file_path: 文本文件绝对路径。
        返回值：
        - dict[str, Any]: 包含 available/path/content/updated_at 的对象。
        异常说明：无；越界或读取失败时统一返回 unavailable。
        边界条件：仅允许访问 runs/<task_id> 目录下的文本文件。
        """
        if file_path is None or (not file_path.exists()) or (not file_path.is_file()):
            return {
                "available": False,
                "path": str(file_path) if file_path else "",
                "content": "",
                "updated_at": "",
                "updated_at_ms": 0,
            }
        try:
            self._build_task_file_url(task_id=task_id, file_path=file_path)
            content_text = file_path.read_text(encoding="utf-8")
            stat_result = file_path.stat()
        except Exception:  # noqa: BLE001
            return {
                "available": False,
                "path": str(file_path),
                "content": "",
                "updated_at": "",
                "updated_at_ms": 0,
            }
        return {
            "available": True,
            "path": str(file_path),
            "content": content_text,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat_result.st_mtime)),
            "updated_at_ms": int(stat_result.st_mtime * 1000),
        }

    def _build_task_json_asset(self, task_id: str, file_path: Path | None) -> dict[str, Any]:
        """
        功能说明：把任务目录内 JSON 文件包装为前端可消费的对象。
        参数说明：
        - task_id: 任务唯一标识。
        - file_path: JSON 文件绝对路径。
        返回值：
        - dict[str, Any]: 读取成功时直接返回 JSON 对象，并补充 available；失败时返回稳定空结构。
        异常说明：无；越界、缺失、损坏时统一回退。
        边界条件：仅允许访问 runs/<task_id> 目录下的 JSON 文件。
        """
        if file_path is None or (not file_path.exists()) or (not file_path.is_file()):
            return {
                "available": False,
                "current_attempt": 0,
                "first_chunk_at": "",
                "first_chunk_at_ms": 0,
                "last_chunk_at": "",
                "last_chunk_at_ms": 0,
            }
        try:
            self._build_task_file_url(task_id=task_id, file_path=file_path)
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {
                "available": False,
                "current_attempt": 0,
                "first_chunk_at": "",
                "first_chunk_at_ms": 0,
                "last_chunk_at": "",
                "last_chunk_at_ms": 0,
            }
        return {
            "available": True,
            "current_attempt": int(payload.get("current_attempt", 0) or 0),
            "first_chunk_at": str(payload.get("first_chunk_at", "")).strip(),
            "first_chunk_at_ms": int(payload.get("first_chunk_at_ms", 0) or 0),
            "last_chunk_at": str(payload.get("last_chunk_at", "")).strip(),
            "last_chunk_at_ms": int(payload.get("last_chunk_at_ms", 0) or 0),
        }

    def _build_task_file_asset(self, task_id: str, file_path: Path | None) -> dict[str, Any]:
        """
        功能说明：把任务目录内文件包装为前端可跳转访问的资产对象。
        参数说明：
        - task_id: 任务唯一标识。
        - file_path: 文件绝对路径。
        返回值：
        - dict[str, Any]: 包含 available/url/path 的对象。
        异常说明：无；路径不在任务目录内时回退为 unavailable。
        边界条件：仅允许构建 runs/<task_id> 目录下的文件 URL。
        """
        if file_path is None or (not file_path.exists()) or (not file_path.is_file()):
            return {
                "available": False,
                "url": "",
                "path": str(file_path) if file_path else "",
                "updated_at": "",
                "updated_at_ms": 0,
            }
        try:
            file_url = self._build_task_file_url(task_id=task_id, file_path=file_path)
        except Exception:  # noqa: BLE001
            return {
                "available": False,
                "url": "",
                "path": str(file_path),
                "updated_at": "",
                "updated_at_ms": 0,
            }
        try:
            stat_result = file_path.stat()
            updated_at_ms = int(stat_result.st_mtime * 1000)
            updated_at_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat_result.st_mtime))
        except Exception:  # noqa: BLE001
            updated_at_ms = 0
            updated_at_text = ""
        return {
            "available": True,
            "url": file_url,
            "path": str(file_path),
            "updated_at": updated_at_text,
            "updated_at_ms": updated_at_ms,
        }

    @staticmethod
    def _build_empty_module_unit_summary(module_name: str) -> dict[str, Any]:
        """
        功能说明：构建模块单元摘要的空占位对象。
        参数说明：
        - module_name: 模块名。
        返回值：
        - dict[str, Any]: 空摘要对象。
        异常说明：无。
        边界条件：用于状态库尚未建立模块单元时的前端稳定渲染。
        """
        return {
            "module_name": str(module_name).strip(),
            "total_units": 0,
            "status_counts": {},
            "pending_unit_ids": [],
            "running_unit_ids": [],
            "failed_unit_ids": [],
            "done_unit_ids": [],
            "problem_unit_ids": [],
        }

    @staticmethod
    def _resolve_project_root() -> Path:
        """
        功能说明：解析项目工作根目录。
        参数说明：无。
        返回值：
        - Path: 项目根目录绝对路径。
        异常说明：无。
        边界条件：约定当前文件位于 `src/music_video_pipeline/monitoring` 下。
        """
        return Path(__file__).resolve().parents[3]

    def _build_task_workspace_roots(self, task_record: dict[str, Any]) -> list[Path]:
        """
        功能说明：为任务音频回映射构建工作区根目录候选。
        参数说明：
        - task_record: 状态库中的任务记录。
        返回值：
        - list[Path]: 已去重的工作区候选根目录数组。
        异常说明：无。
        边界条件：优先从任务配置路径反推工作区；失败时回退状态库路径与当前项目根目录。
        """
        candidate_roots: list[Path] = []
        config_path_text = str(task_record.get("config_path", "")).strip()
        if config_path_text:
            try:
                config_path = Path(config_path_text).resolve()
                lowered_parts = [str(part).lower() for part in config_path.parts]
                if "configs" in lowered_parts:
                    configs_index = lowered_parts.index("configs")
                    if configs_index > 0:
                        candidate_roots.append(Path(*config_path.parts[:configs_index]).resolve())
                else:
                    candidate_roots.append(config_path.parent.resolve())
            except Exception:  # noqa: BLE001
                pass
        try:
            candidate_roots.append(self.state_store.db_path.parent.resolve())
            candidate_roots.append(self.state_store.db_path.parent.parent.resolve())
        except Exception:  # noqa: BLE001
            pass
        candidate_roots.append(self._resolve_project_root())

        normalized_roots: list[Path] = []
        seen_keys: set[str] = set()
        for root in candidate_roots:
            root_key = str(root).casefold()
            if root_key in seen_keys:
                continue
            seen_keys.add(root_key)
            normalized_roots.append(root)
        return normalized_roots

    def _resolve_task_audio_path_from_record(
        self,
        *,
        task_id: str,
        task_record: dict[str, Any],
        persist: bool = False,
    ) -> Path:
        """
        功能说明：解析任务记录中的输入音频路径，并按需把旧外机路径自愈为本机真实路径。
        参数说明：
        - task_id: 任务唯一标识。
        - task_record: 状态库中的任务记录。
        - persist: 是否把成功回映射后的本机路径写回状态库。
        返回值：
        - Path: 当前机器可访问的音频绝对路径。
        异常说明：
        - FileNotFoundError: 原始路径与回退路径均不可用时抛出。
        边界条件：仅在 `persist=True` 且存在 config_path 时回写状态库。
        """
        audio_path_text = str(task_record.get("audio_path", "")).strip()
        config_path_text = str(task_record.get("config_path", "")).strip()
        fallback_default_audio_path = ""
        if self.app_config is not None:
            fallback_default_audio_path = str(getattr(getattr(self.app_config, "paths", None), "default_audio_path", "") or "")
        resolved_audio_path = resolve_task_audio_path(
            raw_audio_path=audio_path_text,
            config_path=config_path_text,
            workspace_roots=self._build_task_workspace_roots(task_record=task_record),
            fallback_default_audio_path=fallback_default_audio_path,
        )
        if persist and config_path_text and str(resolved_audio_path) != audio_path_text:
            self.state_store.init_task(task_id=task_id, audio_path=str(resolved_audio_path), config_path=config_path_text)
        return resolved_audio_path














        return None








    def _resolve_task_dir(self, task_id: str) -> Path:
        """
        功能说明：根据任务ID解析 runs 目录下的任务根目录。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - Path: 任务目录绝对路径。
        异常说明：无。
        边界条件：默认按状态库同级 runs 目录组织。
        """
        return (self.state_store.db_path.parent / str(task_id).strip()).resolve()

    def _resolve_output_video_path(self, task_dir: Path, task_record: dict[str, Any]) -> Path | None:
        """
        功能说明：定位任务最终成片路径。
        参数说明：
        - task_dir: 任务目录。
        - task_record: tasks 表记录。
        返回值：
        - Path | None: 找到则返回视频路径，否则返回 None。
        异常说明：无。
        边界条件：优先使用状态表 output_video_path，缺失时回退任务目录标准文件名。
        """
        candidate_paths: list[Path] = []
        output_video_path_text = str(task_record.get("output_video_path", "")).strip()
        if output_video_path_text:
            candidate_paths.append(Path(output_video_path_text).resolve())
        candidate_paths.append((task_dir / "final_output.mp4").resolve())
        for candidate_path in candidate_paths:
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path
        return None

    def _resolve_module_a_visualization_path(self, task_dir: Path, task_id: str) -> Path | None:
        """
        功能说明：定位模块A V2 自动可视化页面路径。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        返回值：
        - Path | None: 找到则返回页面路径，否则返回 None。
        异常说明：无。
        边界条件：优先命中标准文件名，缺失时回退 glob 搜索。
        """
        standard_path = (task_dir / f"{str(task_id).strip()}_module_a_v2_visualization.html").resolve()
        if standard_path.exists() and standard_path.is_file():
            return standard_path
        candidates = sorted(task_dir.glob("*_module_a_v2_visualization.html"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None













    def _load_json_file(self, file_path: Path) -> Any:
        """
        功能说明：以 UTF-8 读取并解析 JSON 文件。
        参数说明：
        - file_path: JSON 文件路径。
        返回值：
        - Any: 解析成功返回 JSON 对象，失败返回 None。
        异常说明：无；内部吞并异常并返回 None。
        边界条件：文件缺失、编码错误、JSON 非法均统一返回 None。
        """
        if (not file_path.exists()) or (not file_path.is_file()):
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None


    def _build_task_file_url(self, task_id: str, file_path: Path) -> str:
        """
        功能说明：为任务目录内文件构建同源访问URL。
        参数说明：
        - task_id: 任务唯一标识。
        - file_path: 任务内文件绝对路径。
        返回值：
        - str: `/task/<task_id>/<relative_path>` 形式的URL。
        异常说明：
        - ValueError: 文件不在对应任务目录下时抛出。
        边界条件：路径按 POSIX 形式编码，兼容浏览器直接访问。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        relative_path = file_path.resolve().relative_to(task_dir)
        encoded_parts = [quote(str(part)) for part in relative_path.parts]
        return f"/task/{quote(str(task_id).strip())}/{'/'.join(encoded_parts)}"

    def _build_task_file_response(
        self,
        path: str,
        request_headers: Any,
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：从 runs 任务目录中读取并返回静态文件。
        参数说明：
        - path: 请求路径。
        - request_headers: HTTP请求头对象。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: 文件响应三元组。
        异常说明：无；异常统一转为 404/416。
        边界条件：支持单一 bytes Range，供 mp4/mp3 在浏览器内顺畅拖动播放。
        """
        raw_parts = [unquote(part) for part in str(path).split("/") if part]
        if len(raw_parts) < 3:
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        task_id = str(raw_parts[1]).strip()
        relative_parts = raw_parts[2:]
        if (not task_id) or (not relative_parts):
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        task_dir = self._resolve_task_dir(task_id=task_id)
        target_path = task_dir.joinpath(*relative_parts).resolve()
        try:
            target_path.relative_to(task_dir)
        except ValueError:
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        if (not target_path.exists()) or (not target_path.is_file()):
            return self._build_http_response(
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="not found",
            )
        content_type = mimetypes.guess_type(str(target_path))[0] or "application/octet-stream"
        return self._build_file_http_response(
            file_path=target_path,
            content_type=content_type,
            request_headers=request_headers,
        )

    def _build_file_http_response(
        self,
        file_path: Path,
        content_type: str,
        request_headers: Any,
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：构造支持 Range 的文件响应。
        参数说明：
        - file_path: 目标文件路径。
        - content_type: 响应 MIME 类型。
        - request_headers: HTTP 请求头对象。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: 文件响应三元组。
        异常说明：无；非法 Range 时返回 416。
        边界条件：仅支持单区间 bytes Range；无 Range 时返回整文件。
        """
        file_size = int(file_path.stat().st_size)
        range_header = ""
        if request_headers is not None and hasattr(request_headers, "get"):
            range_header = str(request_headers.get("Range", "") or "").strip()
        range_spec = self._parse_http_range(range_header=range_header, file_size=file_size)
        if range_spec == "invalid":
            return self._build_http_response(
                status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                content_type="text/plain; charset=utf-8",
                body_text="invalid range",
                extra_headers=[("Content-Range", f"bytes */{file_size}")],
            )
        start_pos = 0
        end_pos = max(0, file_size - 1)
        status = HTTPStatus.OK
        extra_headers = [("Accept-Ranges", "bytes")]
        if isinstance(range_spec, tuple):
            start_pos, end_pos = range_spec
            status = HTTPStatus.PARTIAL_CONTENT
            extra_headers.append(("Content-Range", f"bytes {start_pos}-{end_pos}/{file_size}"))
        read_length = max(0, end_pos - start_pos + 1)
        with file_path.open("rb") as file_obj:
            file_obj.seek(start_pos)
            body_bytes = file_obj.read(read_length)
        return self._build_http_response(
            status=status,
            content_type=content_type,
            body_text="",
            extra_headers=extra_headers,
            body_bytes=body_bytes,
        )

    def _parse_http_range(self, range_header: str, file_size: int) -> tuple[int, int] | str | None:
        """
        功能说明：解析浏览器发来的单区间 bytes Range 请求。
        参数说明：
        - range_header: Range 请求头原文。
        - file_size: 目标文件总字节数。
        返回值：
        - tuple[int, int] | str | None: 成功返回 `(start, end)`，无 Range 返回 None，非法返回 `"invalid"`。
        异常说明：无。
        边界条件：仅支持 `bytes=start-end` / `bytes=start-` / `bytes=-suffix` 三种单区间形式。
        """
        normalized = str(range_header or "").strip()
        if not normalized:
            return None
        if (not normalized.startswith("bytes=")) or ("," in normalized):
            return "invalid"
        raw_range = normalized[len("bytes=") :].strip()
        if "-" not in raw_range:
            return "invalid"
        start_text, end_text = raw_range.split("-", 1)
        try:
            if start_text == "":
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    return "invalid"
                start_pos = max(0, file_size - suffix_length)
                return start_pos, max(0, file_size - 1)
            start_pos = int(start_text)
            if start_pos < 0 or start_pos >= file_size:
                return "invalid"
            if end_text == "":
                return start_pos, max(0, file_size - 1)
            end_pos = int(end_text)
            if end_pos < start_pos:
                return "invalid"
            return start_pos, min(end_pos, max(0, file_size - 1))
        except (TypeError, ValueError):
            return "invalid"

    def _build_http_response(
        self,
        status: HTTPStatus,
        content_type: str,
        body_text: str,
        extra_headers: list[tuple[str, str]] | None = None,
        body_bytes: bytes | None = None,
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：构造 websockets process_request 需要的HTTP响应三元组。
        参数说明：
        - status: HTTP状态码。
        - content_type: Content-Type 头。
        - body_text: 响应正文文本。
        - extra_headers: 额外响应头。
        - body_bytes: 可选原始字节正文；传入时优先于 body_text。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: HTTP响应对象。
        异常说明：无。
        边界条件：body统一按UTF-8编码。
        """
        if body_bytes is None:
            body_bytes = body_text.encode("utf-8")
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body_bytes))),
            ("Cache-Control", "no-store"),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        return status, headers, body_bytes
