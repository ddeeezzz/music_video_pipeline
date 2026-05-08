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
# 标准库：用于MIME类型推断
import mimetypes
# 标准库：用于HTTP状态码常量
from http import HTTPStatus
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于后台线程
import threading
# 标准库：用于URL解析与编码
from urllib.parse import parse_qs, quote, unquote, urlparse
# 标准库：用于类型提示
from typing import Any, Callable

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
# 常量：前端主页面路由路径
WEB_ROUTE_PATH = "/web"
# 常量：主页任务列表接口路径
TASK_LIST_API_PATH = "/api/tasks"
# 常量：任务详情接口路径
TASK_DETAIL_API_PATH = "/api/task"
# 常量：任务新建接口路径
TASK_CREATE_API_PATH = "/api/task/create"
# 常量：任务改名接口路径
TASK_RENAME_API_PATH = "/api/task/rename"
# 常量：任务复制接口路径
TASK_COPY_API_PATH = "/api/task/copy"
# 常量：任务强制重跑接口路径
TASK_RERUN_API_PATH = "/api/task/rerun"


# 类型别名：用于触发任务强制重跑的回调函数。
TaskRerunHandler = Callable[[str], dict[str, Any]]


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
        rerun_handler: TaskRerunHandler | None = None,
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
        - rerun_handler: 可选的任务强制重跑回调。
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
        self.rerun_handler = rerun_handler
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
        self._rerun_threads: dict[str, threading.Thread] = {}

        self._started_event = threading.Event()
        self._startup_error: Exception | None = None

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
        return f"http://{self.host}:{port}{WEB_ROUTE_PATH}?task_id={quote(self.task_id)}"

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
        self.logger.info("任务Web服务已启动，task_id=%s，地址=%s", self.task_id, self.monitor_url)

        watcher_task = asyncio.create_task(self._watch_task_status_until_terminal())
        await self._async_stop_event.wait()
        watcher_task.cancel()
        await asyncio.gather(watcher_task, return_exceptions=True)
        await self._close_all_connections()

        self._server.close()
        await self._server.wait_closed()
        self.logger.info("任务Web服务已停止，task_id=%s", self.task_id)

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
            location = f"{WEB_ROUTE_PATH}?task_id={quote(self.task_id)}"
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
        if parsed.path == "/web-data":
            query = parse_qs(parsed.query)
            target_task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
            payload = self._build_web_payload(task_id=target_task_id)
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                body_text=json.dumps(payload, ensure_ascii=False),
            )
        if parsed.path == WEB_ROUTE_PATH:
            return self._build_http_response(
                status=HTTPStatus.OK,
                content_type="text/html; charset=utf-8",
                body_text=self._load_monitor_html(),
            )
        if parsed.path == "/task-monitor":
            location = f"{WEB_ROUTE_PATH}?task_id={quote(self.task_id)}"
            return self._build_http_response(
                status=HTTPStatus.FOUND,
                content_type="text/plain; charset=utf-8",
                body_text="redirect",
                extra_headers=[("Location", location)],
            )
        if parsed.path.startswith("/task/"):
            return self._build_task_file_response(path=parsed.path, request_headers=_request_headers)
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

    def _build_web_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建 Web 前端主页面所需的数据负载。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含视频地址、模块A可视化地址与歌词时间戳的数据对象。
        异常说明：无；缺失文件时返回 available=false。
        边界条件：歌词时间戳直接复用模块A输出中的 FunASR 对齐结果。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id) or {}
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        video_path = self._resolve_output_video_path(task_dir=task_dir, task_record=task_record)
        visualization_path = self._resolve_module_a_visualization_path(task_dir=task_dir, task_id=normalized_task_id)
        lyric_units = self._load_lyric_units(task_dir=task_dir)
        segment_units = self._load_segment_units(task_dir=task_dir, task_id=normalized_task_id)
        return {
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "video": {
                "available": video_path is not None and video_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=video_path) if video_path else "",
                "path": str(video_path) if video_path else "",
            },
            "module_a_visualization": {
                "available": visualization_path is not None and visualization_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=visualization_path)
                if visualization_path
                else "",
                "path": str(visualization_path) if visualization_path else "",
            },
            "lyric_units": lyric_units,
            "segment_units": segment_units,
        }

    def _build_task_list_payload(self) -> dict[str, Any]:
        """
        功能说明：构建主页任务列表所需的任务概览与模块状态摘要。
        参数说明：无。
        返回值：
        - dict[str, Any]: 包含 tasks 数组与 current_task_id 的页面数据。
        异常说明：无。
        边界条件：无任务时返回空数组。
        """
        task_rows = self.state_store.list_tasks()
        task_ids = [str(item.get("task_id", "")).strip() for item in task_rows if str(item.get("task_id", "")).strip()]
        module_status_map = self.state_store.list_task_module_status_map(task_ids=task_ids)
        normalized_tasks: list[dict[str, Any]] = []
        for item in task_rows:
            task_id = str(item.get("task_id", "")).strip()
            normalized_tasks.append(
                {
                    "task_id": task_id,
                    "status": str(item.get("status", "unknown")),
                    "audio_path": str(item.get("audio_path", "")),
                    "config_path": str(item.get("config_path", "")),
                    "output_video_path": str(item.get("output_video_path", "")),
                    "updated_at": str(item.get("updated_at", "")),
                    "module_status": module_status_map.get(task_id, {}),
                }
            )
        return {"ok": True, "current_task_id": self.task_id, "tasks": normalized_tasks}

    def _build_task_detail_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建单任务详情面板所需的数据对象。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 成功时返回任务详情，失败时返回错误说明。
        异常说明：无。
        边界条件：任务不存在时返回 ok=false。
        """
        normalized_task_id = str(task_id).strip()
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {"ok": False, "error": f"任务不存在：{normalized_task_id}", "task": None}
        module_status_map = self.state_store.list_task_module_status_map([normalized_task_id]).get(normalized_task_id, {})
        return {
            "ok": True,
            "task": {
                "task_id": normalized_task_id,
                "status": str(task_record.get("status", "unknown")),
                "audio_path": str(task_record.get("audio_path", "")),
                "config_path": str(task_record.get("config_path", "")),
                "output_video_path": str(task_record.get("output_video_path", "")),
                "updated_at": str(task_record.get("updated_at", "")),
                "created_at": str(task_record.get("created_at", "")),
                "error_message": str(task_record.get("error_message", "")),
                "module_status": module_status_map,
            },
        }

    def _handle_create_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理主页新建任务请求，仅写入状态记录，不触发实际运行。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：task_id 已存在时拒绝创建。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [""])[0]).strip()
        audio_path = str(query.get("audio_path", [""])[0]).strip()
        config_path = str(query.get("config_path", [""])[0]).strip()
        if not task_id or not audio_path or not config_path:
            return {"ok": False, "error": "新建任务失败：task_id、audio_path、config_path 不能为空。"}, HTTPStatus.BAD_REQUEST
        if self.state_store.task_exists(task_id=task_id):
            return {"ok": False, "error": f"新建任务失败：task_id 已存在，task_id={task_id}"}, HTTPStatus.CONFLICT
        self.state_store.init_task(task_id=task_id, audio_path=audio_path, config_path=config_path)
        return {
            "ok": True,
            "task_id": task_id,
            "task": self._build_task_detail_payload(task_id=task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_rename_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理主页任务改名请求，并同步重命名任务目录。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：任务目录不存在时仅改库；目录冲突时拒绝改名。
        """
        query = parse_qs(parsed.query)
        old_task_id = str(query.get("old_task_id", [""])[0]).strip()
        new_task_id = str(query.get("new_task_id", [""])[0]).strip()
        if not old_task_id or not new_task_id:
            return {"ok": False, "error": "任务改名失败：old_task_id 与 new_task_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        try:
            self._rename_task_with_artifacts(old_task_id=old_task_id, new_task_id=new_task_id)
        except ValueError as error:
            return {"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST
        except RuntimeError as error:
            return {"ok": False, "error": str(error)}, HTTPStatus.CONFLICT
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"任务改名失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
        return {
            "ok": True,
            "task_id": new_task_id,
            "task": self._build_task_detail_payload(task_id=new_task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_copy_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理基于现有任务复制为新任务的请求，仅创建新记录，不自动运行。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：新任务默认继承原任务音频与配置路径，可被显式覆盖。
        """
        query = parse_qs(parsed.query)
        source_task_id = str(query.get("source_task_id", [""])[0]).strip()
        new_task_id = str(query.get("new_task_id", [""])[0]).strip()
        if not source_task_id or not new_task_id:
            return {"ok": False, "error": "复制任务失败：source_task_id 与 new_task_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        source_task = self.state_store.get_task(task_id=source_task_id)
        if source_task is None:
            return {"ok": False, "error": f"复制任务失败：源任务不存在，task_id={source_task_id}"}, HTTPStatus.NOT_FOUND
        if self.state_store.task_exists(task_id=new_task_id):
            return {"ok": False, "error": f"复制任务失败：目标 task_id 已存在，task_id={new_task_id}"}, HTTPStatus.CONFLICT
        audio_path = str(query.get("audio_path", [str(source_task.get("audio_path", ""))])[0]).strip()
        config_path = str(query.get("config_path", [str(source_task.get("config_path", ""))])[0]).strip()
        if not audio_path or not config_path:
            return {"ok": False, "error": "复制任务失败：audio_path 与 config_path 不能为空。"}, HTTPStatus.BAD_REQUEST
        self.state_store.init_task(task_id=new_task_id, audio_path=audio_path, config_path=config_path)
        return {
            "ok": True,
            "task_id": new_task_id,
            "task": self._build_task_detail_payload(task_id=new_task_id).get("task"),
        }, HTTPStatus.OK

    def _handle_rerun_task_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理主页“生成”按钮触发的强制全链路重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：仅接受已存在任务，且同一任务不允许并发重复触发。
        """
        if self.rerun_handler is None:
            return {"ok": False, "error": "当前监督服务未配置生成能力。"}, HTTPStatus.NOT_IMPLEMENTED
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"生成失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"生成失败：任务已在后台启动中，task_id={task_id}"}, HTTPStatus.CONFLICT
        task_status = str(task_record.get("status", "")).strip().lower()
        if task_status == "running":
            return {"ok": False, "error": f"生成失败：任务当前正在运行，task_id={task_id}"}, HTTPStatus.CONFLICT

        rerun_thread = threading.Thread(
            target=self._run_rerun_task_in_background,
            name=f"task-rerun-{task_id}",
            args=(task_id,),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        rerun_thread.start()
        self.logger.info("任务强制重跑已提交，task_id=%s，from_module=A", task_id)
        return {
            "ok": True,
            "task_id": task_id,
            "message": f"任务已开始生成，task_id={task_id}，模式=强制从A模块开始覆盖式重跑",
        }, HTTPStatus.OK

    def _run_rerun_task_in_background(self, task_id: str) -> None:
        """
        功能说明：在后台线程中执行任务强制重跑。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        try:
            self.logger.info("后台开始执行任务强制重跑，task_id=%s，from_module=A", task_id)
            self.rerun_handler(task_id)
            self.logger.info("后台任务强制重跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            self.logger.error("后台任务强制重跑失败，task_id=%s，错误信息=%s", task_id, error)
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)

    def _rename_task_with_artifacts(self, old_task_id: str, new_task_id: str) -> None:
        """
        功能说明：协调状态库改名与 runs 目录改名，确保任务上下文一致。
        参数说明：
        - old_task_id: 原任务ID。
        - new_task_id: 新任务ID。
        返回值：无。
        异常说明：
        - RuntimeError: 任务不存在、目标冲突或回滚失败时抛出。
        - ValueError: 任务ID非法时抛出。
        边界条件：若旧目录不存在，仅执行数据库改名。
        """
        normalized_old_task_id = str(old_task_id).strip()
        normalized_new_task_id = str(new_task_id).strip()
        old_task_dir = self._resolve_task_dir(task_id=normalized_old_task_id)
        new_task_dir = self._resolve_task_dir(task_id=normalized_new_task_id)
        if old_task_dir.exists() and new_task_dir.exists():
            raise RuntimeError(f"任务改名失败：目标任务目录已存在，path={new_task_dir}")

        self.state_store.rename_task(old_task_id=normalized_old_task_id, new_task_id=normalized_new_task_id)
        try:
            if old_task_dir.exists():
                old_task_dir.rename(new_task_dir)
        except Exception as error:  # noqa: BLE001
            try:
                self.state_store.rename_task(old_task_id=normalized_new_task_id, new_task_id=normalized_old_task_id)
            except Exception as rollback_error:  # noqa: BLE001
                raise RuntimeError(
                    f"任务改名失败：目录改名出错且数据库回滚失败，dir_error={error}，rollback_error={rollback_error}"
                ) from rollback_error
            raise RuntimeError(f"任务改名失败：目录改名出错，已回滚数据库，error={error}") from error

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

    def _load_lyric_units(self, task_dir: Path) -> list[dict[str, Any]]:
        """
        功能说明：从模块A标准输出中读取歌词时间戳数组。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: lyric_units 数组，供 Web 前端按时间滚动显示。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：只透传页面所需字段，避免把无关大对象塞给前端。
        """
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        try:
            payload = json.loads(module_a_output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        raw_lyric_units = payload.get("lyric_units", [])
        if not isinstance(raw_lyric_units, list):
            return []
        normalized_items: list[dict[str, Any]] = []
        for item in raw_lyric_units:
            if not isinstance(item, dict):
                continue
            token_units_payload = item.get("token_units", [])
            normalized_token_units: list[dict[str, Any]] = []
            if isinstance(token_units_payload, list):
                for token_item in token_units_payload:
                    if not isinstance(token_item, dict):
                        continue
                    normalized_token_units.append(
                        {
                            "text": str(token_item.get("text", "")),
                            "start_time": float(token_item.get("start_time", 0.0)),
                            "end_time": float(token_item.get("end_time", float(token_item.get("start_time", 0.0)))),
                        }
                    )
            normalized_items.append(
                {
                    "segment_id": str(item.get("segment_id", "")),
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "text": str(item.get("text", "")),
                    "confidence": float(item.get("confidence", 0.0)),
                    "token_units": normalized_token_units,
                }
            )
        return normalized_items

    def _load_segment_units(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：从模块A标准输出中读取最终小段时间轴数组。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: segments 数组，供 Web 前端按播放时间滚动高亮。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：仅透传主页面检查音画同步需要的字段。
        """
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        try:
            payload = json.loads(module_a_output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list):
            return []
        scene_desc_items = self._load_segment_scene_desc_items(task_dir=task_dir, task_id=task_id)
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                continue
            scene_desc_item: dict[str, Any] = {}
            if index < len(scene_desc_items):
                scene_desc_item = scene_desc_items[index]
            scene_desc = str(scene_desc_item.get("scene_desc", "")).strip()
            normalized_items.append(
                {
                    "segment_id": str(item.get("segment_id", "")),
                    "big_segment_id": str(item.get("big_segment_id", "")),
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "label": str(item.get("label", "")),
                    "role": str(item.get("role", "")),
                    "scene_desc": scene_desc,
                    "shot_id": str(scene_desc_item.get("shot_id", "")).strip(),
                    "camera_plan": scene_desc_item.get("camera_plan", {}),
                    "keyframe_prompt_start_zh": str(scene_desc_item.get("keyframe_prompt_start_zh", "")).strip(),
                    "keyframe_prompt_start_en": str(scene_desc_item.get("keyframe_prompt_start_en", "")).strip(),
                    "keyframe_prompt_end_zh": str(scene_desc_item.get("keyframe_prompt_end_zh", "")).strip(),
                    "keyframe_prompt_end_en": str(scene_desc_item.get("keyframe_prompt_end_en", "")).strip(),
                    "video_prompt_zh": str(scene_desc_item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(scene_desc_item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": str(scene_desc_item.get("frame_path_start", "")).strip(),
                    "frame_path_end": str(scene_desc_item.get("frame_path_end", "")).strip(),
                    "frame_url_start": str(scene_desc_item.get("frame_url_start", "")).strip(),
                    "frame_url_end": str(scene_desc_item.get("frame_url_end", "")).strip(),
                }
            )
        return normalized_items

    def _load_segment_scene_desc_items(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：从模块D摘要中读取按顺序落盘的场景描述数组。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: 含 scene_desc 的 segment_items 数组。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：仅透传页面展示当前段描述所需字段。
        """
        module_d_output_path = task_dir / "artifacts" / "module_d_output.json"
        if not module_d_output_path.exists():
            return []
        try:
            payload = json.loads(module_d_output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        raw_segment_items = payload.get("segment_items", [])
        if not isinstance(raw_segment_items, list):
            return []
        frame_item_map = self._load_frame_item_map_by_shot_id(task_dir=task_dir)
        normalized_items: list[dict[str, Any]] = []
        for item in raw_segment_items:
            if not isinstance(item, dict):
                continue
            shot_id = str(item.get("shot_id", "")).strip()
            frame_item = frame_item_map.get(shot_id, {}) if shot_id else {}
            frame_path_start = str(frame_item.get("frame_path_start", "")).strip()
            frame_path_end = str(frame_item.get("frame_path_end", "")).strip()
            frame_url_start = self._build_task_file_url(task_id=task_id, file_path=Path(frame_path_start)) if frame_path_start else ""
            frame_url_end = self._build_task_file_url(task_id=task_id, file_path=Path(frame_path_end)) if frame_path_end else ""
            normalized_items.append(
                {
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "scene_desc": str(item.get("scene_desc", "")).strip(),
                    "shot_id": shot_id,
                    "camera_plan": item.get("camera_plan", {}),
                    "keyframe_prompt_start_zh": str(item.get("keyframe_prompt_start_zh", "")).strip(),
                    "keyframe_prompt_start_en": str(item.get("keyframe_prompt_start_en", "")).strip(),
                    "keyframe_prompt_end_zh": str(item.get("keyframe_prompt_end_zh", "")).strip(),
                    "keyframe_prompt_end_en": str(item.get("keyframe_prompt_end_en", "")).strip(),
                    "video_prompt_zh": str(item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": frame_path_start,
                    "frame_path_end": frame_path_end,
                    "frame_url_start": frame_url_start,
                    "frame_url_end": frame_url_end,
                }
            )
        return normalized_items

    def _load_frame_item_map_by_shot_id(self, task_dir: Path) -> dict[str, dict[str, Any]]:
        """
        功能说明：从模块C输出中读取 shot_id 到关键帧产物的映射。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, dict[str, Any]]: shot_id -> frame_item 映射。
        异常说明：读取失败时返回空映射，不中断页面。
        边界条件：仅使用当前页面需要的关键帧路径字段。
        """
        module_c_output_path = task_dir / "artifacts" / "module_c_output.json"
        if not module_c_output_path.exists():
            return {}
        try:
            payload = json.loads(module_c_output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        raw_frame_items = payload.get("frame_items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_frame_items, list):
            return {}
        item_map: dict[str, dict[str, Any]] = {}
        for item in raw_frame_items:
            if not isinstance(item, dict):
                continue
            shot_id = str(item.get("shot_id", "")).strip()
            if not shot_id:
                continue
            item_map[shot_id] = item
        return item_map

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
