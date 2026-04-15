"""
文件用途：提供任务监督页面的轻量HTTP+WebSocket服务。
核心流程：后台线程启动异步服务，推送任务快照并在任务终态自动停止。
输入输出：输入 StateStore 与 task_id，输出可访问URL与实时状态流。
依赖说明：依赖标准库 asyncio/threading 与第三方 websockets。
维护说明：该服务仅用于单任务监督，保持最小可维护实现。
"""

# 标准库：用于异步协程与事件循环
import asyncio
# 标准库：用于状态快照JSON序列化
import json
# 标准库：用于日志记录
import logging
# 标准库：用于HTTP状态码常量
from http import HTTPStatus
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于后台线程
import threading
# 标准库：用于URL解析与编码
from urllib.parse import parse_qs, quote, urlparse
# 标准库：用于类型提示
from typing import Any

# 项目内模块：任务监督快照构建
from music_video_pipeline.monitoring.snapshot import build_task_monitor_snapshot
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore

try:
    # 第三方库：提供WebSocket与HTTP复用服务
    import websockets
    # 第三方库：WebSocket连接关闭异常
    from websockets.exceptions import ConnectionClosed
except Exception:  # noqa: BLE001
    websockets = None
    ConnectionClosed = Exception

# 常量：监督页静态文件名
MONITOR_HTML_FILE_NAME = "task_monitor.html"


class TaskMonitorService:
    """
    功能说明：封装单任务监督服务的生命周期。
    参数说明：
    - state_store: 任务状态存储对象。
    - task_id: 当前任务唯一标识。
    - logger: 日志对象。
    - host: HTTP/WS 监听地址。
    - port: HTTP/WS 监听端口（0表示自动分配）。
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
        host: str = "127.0.0.1",
        port: int = 0,
        tick_seconds: float = 1.0,
        auto_stop_on_terminal: bool = True,
    ) -> None:
        """
        功能说明：初始化监督服务对象。
        参数说明：
        - state_store: 状态存储对象。
        - task_id: 任务唯一标识。
        - logger: 日志对象。
        - host: 监听地址。
        - port: 监听端口。
        - tick_seconds: 推送间隔秒数。
        - auto_stop_on_terminal: 任务进入终态且无页面连接时是否自动停止服务。
        返回值：无。
        异常说明：无。
        边界条件：不在构造阶段启动网络监听。
        """
        self.state_store = state_store
        self.task_id = task_id
        self.logger = logger
        self.host = host
        self.port = int(port)
        self.tick_seconds = max(0.2, float(tick_seconds))
        self.auto_stop_on_terminal = bool(auto_stop_on_terminal)
        self._bound_port = 0

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._async_stop_event: asyncio.Event | None = None
        self._connections: set[Any] = set()
        self._task_terminal = False

        self._started_event = threading.Event()
        self._startup_error: Exception | None = None

    @property
    def monitor_url(self) -> str:
        """
        功能说明：返回任务监督页面URL。
        参数说明：无。
        返回值：
        - str: 监督页面地址。
        异常说明：无。
        边界条件：服务未启动时端口为初始化值。
        """
        port = self._bound_port or self.port
        return f"http://{self.host}:{port}/task-monitor?task_id={quote(self.task_id)}"

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
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"task-monitor-{self.task_id}",
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
        self._started_event.set()
        self.logger.info("任务监督服务已启动，task_id=%s，地址=%s", self.task_id, self.monitor_url)

        watcher_task = asyncio.create_task(self._watch_task_status_until_terminal())
        await self._async_stop_event.wait()
        watcher_task.cancel()
        await asyncio.gather(watcher_task, return_exceptions=True)
        await self._close_all_connections()

        self._server.close()
        await self._server.wait_closed()
        self.logger.info("任务监督服务已停止，task_id=%s", self.task_id)

    async def _watch_task_status_until_terminal(self) -> None:
        """
        功能说明：轮询任务状态，命中终态后停止监督服务。
        参数说明：无。
        返回值：无。
        异常说明：无。
        边界条件：仅在 done/failed 时自动停止。
        """
        if not self._async_stop_event:
            return
        while not self._async_stop_event.is_set():
            snapshot = build_task_monitor_snapshot(state_store=self.state_store, task_id=self.task_id)
            task_status = str(snapshot.get("task_status", "unknown"))
            if task_status in {"done", "failed"}:
                if not self._task_terminal:
                    self._task_terminal = True
                    if self.auto_stop_on_terminal:
                        self.logger.info(
                            "任务监督服务检测到任务终态，等待页面连接关闭后停止，task_id=%s，status=%s，active_connections=%s",
                            self.task_id,
                            task_status,
                            len(self._connections),
                        )
                    else:
                        self.logger.info(
                            "任务监督服务检测到任务终态（手动模式不自动停止），task_id=%s，status=%s",
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
        if parsed.path == "/ws":
            return None
        if parsed.path == "/" or parsed.path == "":
            location = f"/task-monitor?task_id={quote(self.task_id)}"
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
        if parsed.path == "/task-monitor":
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="text/html; charset=utf-8",
                body_text=self._load_monitor_html(),
            )
        return self._build_http_response(
            status=HTTPStatus.NOT_FOUND,
            content_type="text/plain; charset=utf-8",
            body_text="not found",
        )

    def _load_monitor_html(self) -> str:
        """
        功能说明：读取监督页HTML模板文本。
        参数说明：无。
        返回值：
        - str: 页面HTML内容。
        异常说明：文件不存在时抛 FileNotFoundError。
        边界条件：模板位于 monitoring/static/task_monitor.html。
        """
        html_path = Path(__file__).resolve().parent / "static" / MONITOR_HTML_FILE_NAME
        return html_path.read_text(encoding="utf-8")

    def _build_http_response(
        self,
        status: HTTPStatus,
        content_type: str,
        body_text: str,
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> tuple[HTTPStatus, list[tuple[str, str]], bytes]:
        """
        功能说明：构造 websockets process_request 需要的HTTP响应三元组。
        参数说明：
        - status: HTTP状态码。
        - content_type: Content-Type 头。
        - body_text: 响应正文文本。
        - extra_headers: 额外响应头。
        返回值：
        - tuple[HTTPStatus, list[tuple[str, str]], bytes]: HTTP响应对象。
        异常说明：无。
        边界条件：body统一按UTF-8编码。
        """
        body_bytes = body_text.encode("utf-8")
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body_bytes))),
            ("Cache-Control", "no-store"),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        return status, headers, body_bytes
