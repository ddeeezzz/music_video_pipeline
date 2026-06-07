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

# 常量：正式前端任务列表路由
TASK_LIST_ROUTE_PATH = "/tasks"
# 常量：正式前端静态资源路由前缀
WEB_APP_STATIC_ROUTE_PREFIX = "/app/"
# 常量：正式前端构建目录名
WEB_APP_BUILD_DIR_NAME = "app"
# 常量：正式前端入口文件名
WEB_APP_INDEX_FILE_NAME = "index.html"
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
# 常量：模块 B 页面数据接口路径
TASK_MODULE_B_API_PATH = "/api/module-b"
# 常量：模块 A 页面数据接口路径
TASK_MODULE_A_API_PATH = "/api/module-a"
# 常量：模块 A 联网歌词搜索接口路径
TASK_MODULE_A_SEARCH_LYRICS_API_PATH = "/api/module-a/search-lyrics"
# 常量：模块 A 联网歌词搜索 WebSocket 路径
TASK_MODULE_A_SEARCH_LYRICS_WS_PATH = "/ws/module-a/search-lyrics"
# 常量：模块 A 联网歌词选择接口路径
TASK_MODULE_A_SELECT_LYRICS_API_PATH = "/api/module-a/select-lyrics"
# 常量：模块 A 候选歌词详情接口路径
TASK_MODULE_A_CANDIDATE_LYRICS_API_PATH = "/api/module-a/candidate-lyrics"
# 常量：模块 B role 重跑接口路径
TASK_MODULE_B_RERUN_ROLE_API_PATH = "/api/module-b/rerun-role"
# 常量：模块 B role 内 segment 重跑接口路径
TASK_MODULE_B_RERUN_ROLE_SEGMENT_API_PATH = "/api/module-b/rerun-role-segment"
# 常量：模块 B 活跃重跑子进程状态文件名
ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME = "active_module_b_rerun_process.json"


# 类型别名：用于触发任务强制重跑的回调函数。
TaskRerunHandler = Callable[[str], dict[str, Any]]
# 类型别名：用于触发模块 B role 级重跑的回调函数。
ModuleBRoleRerunHandler = Callable[[str, str], dict[str, Any]]
# 类型别名：用于触发模块 B role 内 segment 级重跑的回调函数。
ModuleBRoleSegmentRerunHandler = Callable[[str, str, str], dict[str, Any]]


class TaskMonitorService:
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

    async def _handle_module_a_search_lyrics_socket(self, websocket: Any, parsed: Any) -> None:
        """
        功能说明：通过 WebSocket 实时推送模块A联网歌词搜索进度与来源结果。
        参数说明：
        - websocket: 当前连接对象。
        - parsed: 已解析的请求 URL。
        返回值：无。
        异常说明：连接中断时自动退出；业务异常会以 error/complete 事件返回。
        边界条件：每次连接仅执行一次搜索，完成后主动关闭。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        manual_query = str(query.get("manual_query", [""])[0]).strip()
        self._connections.add(websocket)
        try:
            task_record = self.state_store.get_task(task_id=task_id)
            if task_record is None:
                await websocket.send(
                    json.dumps(
                        {
                            "event": "error",
                            "data": {"message": f"任务不存在：{task_id}"},
                        },
                        ensure_ascii=False,
                    )
                )
                return
            if self.app_config is None:
                await websocket.send(
                    json.dumps(
                        {
                            "event": "error",
                            "data": {"message": "当前监督服务缺少运行配置，无法联网查找歌词。"},
                        },
                        ensure_ascii=False,
                    )
                )
                return
            audio_path = self._resolve_task_audio_path_from_record(task_id=task_id, task_record=task_record, persist=False)
            duration_seconds = probe_audio_duration(
                audio_path=audio_path,
                ffprobe_bin=str(getattr(getattr(self.app_config, "ffmpeg", None), "ffprobe_bin", "ffprobe")),
                logger=self.logger,
            )
            event_loop = asyncio.get_running_loop()

            async def _send_event(event_name: str, payload: dict[str, Any]) -> None:
                await websocket.send(json.dumps({"event": event_name, "data": payload}, ensure_ascii=False))

            def _emit_event(event_name: str, payload: dict[str, Any]) -> None:
                stream_payload = payload
                if event_name == "provider_group":
                    stream_payload = self._build_module_a_stream_preview_provider_group(payload)
                elif event_name == "complete":
                    self._persist_module_a_search_result(task_id=task_id, search_result=payload)
                    stream_payload = self._build_module_a_stream_preview_result(payload)
                asyncio.run_coroutine_threadsafe(_send_event(event_name, stream_payload), event_loop)

            result = await asyncio.to_thread(
                stream_synced_lrc_candidates,
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                fpcalc_bin=str(getattr(getattr(self.app_config, "module_a", None), "fpcalc_bin", "fpcalc")),
                acoustid_api_key_file=str(
                    getattr(getattr(self.app_config, "module_a", None), "acoustid_api_key_file", "")
                ),
                logger=self.logger,
                manual_query=manual_query,
                emit_event=_emit_event,
                split_syncedlyrics_providers=True,
            )
            await asyncio.to_thread(self._persist_module_a_search_result, task_id, result)
        except ConnectionClosed:
            return
        except Exception as error:  # noqa: BLE001
            await websocket.send(
                json.dumps(
                    {
                        "event": "error",
                        "data": {"message": str(error).strip() or "module_a_search_stream_failed"},
                    },
                    ensure_ascii=False,
                )
            )
        finally:
            self._connections.discard(websocket)
            try:
                await websocket.close(code=1000, reason="module-a-search-complete")
            except Exception:  # noqa: BLE001
                pass
            if (
                self.auto_stop_on_terminal
                and self._task_terminal
                and not self._connections
                and self._async_stop_event
                and not self._async_stop_event.is_set()
            ):
                self._async_stop_event.set()

    def _build_module_a_stream_preview_provider_group(self, provider_group: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：为 WebSocket 实时流裁剪来源预览页，只推送当前页首屏所需内容。
        参数说明：
        - provider_group: 完整来源分组对象。
        返回值：
        - dict[str, Any]: 裁剪后的来源分组对象。
        异常说明：无。
        边界条件：保留 total_count/has_more，方便前端决定是否继续按需加载。
        """
        preview_group = dict(provider_group) if isinstance(provider_group, dict) else {}
        candidates = provider_group.get("candidates", []) if isinstance(provider_group, dict) else []
        normalized_candidates = [dict(item) for item in candidates[:10] if isinstance(item, dict)]
        preview_group["candidates"] = normalized_candidates
        preview_group["page_size"] = 10
        preview_group["total_count"] = int(provider_group.get("total_count", len(candidates)) or len(candidates))
        preview_group["has_more"] = preview_group["total_count"] > 10
        return preview_group

    def _build_module_a_stream_preview_result(self, search_result: dict[str, Any]) -> dict[str, Any]:
        """
        功能说明：为 WebSocket 完成事件裁剪来源预览页，避免前端在未翻页前接收全部候选。
        参数说明：
        - search_result: 完整搜索结果。
        返回值：
        - dict[str, Any]: 裁剪后的搜索结果。
        异常说明：无。
        边界条件：持久化仍使用完整结果，本函数仅影响实时推送。
        """
        preview_result = dict(search_result) if isinstance(search_result, dict) else {}
        provider_groups = search_result.get("provider_groups", []) if isinstance(search_result, dict) else []
        preview_groups = [
            self._build_module_a_stream_preview_provider_group(item)
            for item in provider_groups
            if isinstance(item, dict)
        ]
        preview_result["provider_groups"] = preview_groups
        preview_result["candidates"] = [
            dict(candidate)
            for provider_group in preview_groups
            for candidate in provider_group.get("candidates", [])
            if isinstance(candidate, dict)
        ]
        return preview_result

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
        segment_units = self._load_segment_units(task_dir=task_dir, task_id=normalized_task_id)
        lyric_units = self._load_lyric_units(task_dir=task_dir, review_segment_units=segment_units)
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

    def _build_module_b_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 B 页面所需的数据负载，仅围绕当前 module_b 源码方案。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含 role 模板、当前实现状态与 segment 入口的数据对象。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：仅展示当前 module_b 源码方案与当前任务上下文。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
                "roles": [],
                "segment_items": [],
            }
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:  # noqa: BLE001
            module_status_map = {}
        try:
            module_b_unit_summary = self.state_store.get_module_unit_status_summary(
                task_id=normalized_task_id,
                module_name="B",
            )
        except Exception:  # noqa: BLE001
            module_b_unit_summary = self._build_empty_module_unit_summary(module_name="B")
        return {
            "ok": True,
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "module_status": module_status_map,
            "module_b_status": module_status_map.get("B", "unknown"),
            "module_b_unit_summary": module_b_unit_summary,
            "active_rerun": self._build_module_b_active_rerun_payload(task_id=normalized_task_id),
            "aggregate_output": self._build_task_file_asset(
                task_id=normalized_task_id,
                file_path=(task_dir / "artifacts" / "module_b_output.json").resolve(),
            ),
            "roles": self._build_module_b_role_payloads(task_id=normalized_task_id, task_dir=task_dir),
            "segment_items": self._load_module_b_segment_selector_items(task_dir=task_dir, task_id=normalized_task_id),
        }

    def _build_module_b_role_payloads(self, task_id: str, task_dir: Path) -> list[dict[str, Any]]:
        """
        功能说明：构建当前 module_b 四个 role 的展示数据。
        参数说明：
        - task_id: 任务唯一标识。
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: role 页面卡片数据数组。
        异常说明：无。
        边界条件：模板缺失或角色尚未实现时也返回稳定结构，供前端呈现占位态。
        """
        project_root = self._resolve_project_root()
        role_specs = [
            {
                "role_name": "role1",
                "title": "Role 1",
                "description": "视觉意象细化",
                "template_relpath": Path("configs/prompts/module_b.role1_visual_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role1_imagery_describer.py"),
                "contract_fields": ["pos_zh", "pos_en"],
                "supports_segment_retry": False,
            },
            {
                "role_name": "role2",
                "title": "Role 2",
                "description": "大段剧情规划",
                "template_relpath": Path("configs/prompts/module_b.role2_big_segment_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role2_story_planner.py"),
                "contract_fields": ["占位1", "占位2", "占位3"],
                "supports_segment_retry": False,
            },
            {
                "role_name": "role3",
                "title": "Role 3",
                "description": "镜头规划",
                "template_relpath": Path("configs/prompts/module_b.role3_segment_director.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role3_shot_planner.py"),
                "contract_fields": ["占位1", "占位2", "占位3"],
                "supports_segment_retry": True,
            },
            {
                "role_name": "role4",
                "title": "Role 4",
                "description": "提示词规划",
                "template_relpath": Path("configs/prompts/module_b.role4_prompt_builder.md"),
                "source_relpath": Path("src/music_video_pipeline/modules/module_b/role4_prompt_builder.py"),
                "contract_fields": ["占位1", "占位2", "占位3"],
                "supports_segment_retry": True,
            },
        ]
        role_payloads: list[dict[str, Any]] = []
        for role_spec in role_specs:
            role_name = str(role_spec["role_name"]).strip()
            source_path = (project_root / Path(role_spec["source_relpath"])).resolve()
            template_path = (project_root / Path(role_spec["template_relpath"])).resolve()
            implementation_status, implementation_detail = self._describe_module_b_role_implementation(
                role_name=role_name,
                source_path=source_path,
                contract_fields=[str(item) for item in role_spec["contract_fields"]],
            )
            role_payloads.append(
                {
                    "role_name": role_name,
                    "title": str(role_spec["title"]).strip(),
                    "description": str(role_spec["description"]).strip(),
                    "source_path": str(source_path),
                    "contract_fields": [str(item) for item in role_spec["contract_fields"]],
                    "implementation_status": implementation_status,
                    "implementation_detail": implementation_detail,
                    "supports_role_rerun": role_name == "role1" and implementation_status == "implemented",
                    "supports_segment_retry": bool(role_spec["supports_segment_retry"]) and implementation_status == "implemented",
                    "prompt_template": self._build_text_file_asset(file_path=template_path),
                    "stream_preview": self._build_task_text_file_asset(
                        task_id=task_id,
                        file_path=(task_dir / "artifacts" / "module_b_role1_visual_output.streaming.md").resolve()
                        if role_name == "role1"
                        else None,
                    ),
                    "stream_preview_meta": self._build_task_json_asset(
                        task_id=task_id,
                        file_path=(task_dir / "artifacts" / "module_b_role1_visual_output.streaming.meta.json").resolve()
                        if role_name == "role1"
                        else None,
                    ),
                    "result": self._build_task_file_asset(
                        task_id=task_id,
                        file_path=self._find_task_local_module_b_role_artifact(task_dir=task_dir, role_name=role_name),
                    ),
                    "result_text": self._build_task_text_file_asset(
                        task_id=task_id,
                        file_path=self._find_task_local_module_b_role_artifact(task_dir=task_dir, role_name=role_name)
                        if role_name == "role1"
                        else None,
                    ),
                }
            )
        return role_payloads

    def _build_module_b_active_rerun_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 B 当前活跃重跑动作摘要，供前端显示进度态。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, Any]: 活跃重跑摘要。
        异常说明：无。
        边界条件：无活跃动作时返回 active=false 的稳定结构。
        """
        meta = self._rerun_thread_meta.get(str(task_id).strip(), {})
        if not meta:
            persisted_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
            if persisted_meta:
                return persisted_meta
        if not meta:
            return {
                "active": False,
                "status": "",
                "mode": "",
                "role_name": "",
                "segment_id": "",
                "shot_id": "",
                "submitted_at": "",
                "submitted_at_ms": 0,
                "started_at": "",
                "started_at_ms": 0,
                "finished_at": "",
                "finished_at_ms": 0,
                "duration_ms": 0,
                "last_error": "",
                "failure_reason": "",
            }
        return {
            "active": bool(meta.get("active", False)),
            "status": str(meta.get("status", "")).strip(),
            "mode": str(meta.get("mode", "")).strip(),
            "role_name": str(meta.get("role_name", "")).strip(),
            "segment_id": str(meta.get("segment_id", "")).strip(),
            "shot_id": str(meta.get("shot_id", "")).strip(),
            "submitted_at": str(meta.get("submitted_at", "")).strip(),
            "submitted_at_ms": int(meta.get("submitted_at_ms", 0) or 0),
            "started_at": str(meta.get("started_at", "")).strip(),
            "started_at_ms": int(meta.get("started_at_ms", 0) or 0),
            "finished_at": str(meta.get("finished_at", "")).strip(),
            "finished_at_ms": int(meta.get("finished_at_ms", 0) or 0),
            "duration_ms": int(meta.get("duration_ms", 0) or 0),
            "last_error": str(meta.get("last_error", "")).strip(),
            "failure_reason": str(meta.get("failure_reason", "")).strip(),
        }

    def _build_active_module_b_rerun_process_path(self, task_id: str) -> Path:
        """
        功能说明：构建模块 B 活跃重跑子进程状态文件路径。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - Path: 状态文件绝对路径。
        异常说明：无。
        边界条件：文件固定写在 runs/<task_id>/ 目录下。
        """
        return self._resolve_task_dir(task_id=task_id) / ACTIVE_MODULE_B_RERUN_PROCESS_FILE_NAME

    @staticmethod
    def _parse_truthy_flag(value: str) -> bool:
        """
        功能说明：把查询字符串中的布尔开关解析为真值。
        参数说明：
        - value: 原始字符串。
        返回值：
        - bool: 常见 true/1/yes/on 视为真。
        异常说明：无。
        边界条件：空字符串返回 False。
        """
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _load_active_module_b_rerun_process_meta(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：从子进程状态文件恢复模块 B 活跃重跑摘要。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - dict[str, Any]: 若存在活跃子进程则返回前端可用摘要，否则返回空字典。
        异常说明：无；文件损坏时自动清理并回退为空。
        边界条件：仅在 PID 仍存活时视为 active。
        """
        process_file_path = self._build_active_module_b_rerun_process_path(task_id=task_id)
        if not process_file_path.exists():
            return {}
        try:
            payload = json.loads(process_file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            process_file_path.unlink(missing_ok=True)
            return {}
        pid = int(payload.get("pid", 0) or 0)
        if pid <= 0 or not self._is_process_alive(pid):
            process_file_path.unlink(missing_ok=True)
            return {}
        submitted_at_ms = int(payload.get("submitted_at_ms", 0) or 0)
        submitted_at = str(payload.get("submitted_at", "")).strip()
        if submitted_at_ms > 0 and not submitted_at:
            submitted_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_ms / 1000))
        return {
            "active": True,
            "status": "running",
            "mode": str(payload.get("mode", "")).strip(),
            "role_name": str(payload.get("role_name", "")).strip(),
            "segment_id": str(payload.get("segment_id", "")).strip(),
            "shot_id": str(payload.get("shot_id", "")).strip(),
            "submitted_at": submitted_at,
            "submitted_at_ms": submitted_at_ms,
            "started_at": submitted_at,
            "started_at_ms": submitted_at_ms,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": max(0, int(time.time() * 1000) - submitted_at_ms) if submitted_at_ms > 0 else 0,
            "last_error": "",
            "failure_reason": "",
        }

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """
        功能说明：在 Windows 上检查指定 PID 是否仍然存活。
        参数说明：
        - pid: 目标进程 PID。
        返回值：
        - bool: 进程存在且仍运行返回 True，否则返回 False。
        异常说明：无；底层探测异常时回退为 False。
        边界条件：依赖 tasklist 输出，适用于当前本地 Windows 环境。
        """
        normalized_pid = int(pid or 0)
        if normalized_pid <= 0:
            return False
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {normalized_pid}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
        except Exception:  # noqa: BLE001
            return False
        output_text = f"{completed.stdout}\n{completed.stderr}".lower()
        if "no tasks are running" in output_text or "没有运行的任务" in output_text:
            return False
        return str(normalized_pid) in output_text

    def _terminate_active_module_b_rerun_process(self, task_id: str) -> bool:
        """
        功能说明：终止当前任务的模块 B 活跃重跑子进程树。
        参数说明：
        - task_id: 任务唯一标识。
        返回值：
        - bool: 成功发起终止返回 True；无活跃进程或终止失败返回 False。
        异常说明：无；错误由调用方转成冲突提示。
        边界条件：当前实现依赖 Windows `taskkill /T /F`。
        """
        process_file_path = self._build_active_module_b_rerun_process_path(task_id=task_id)
        if not process_file_path.exists():
            return False
        try:
            payload = json.loads(process_file_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            process_file_path.unlink(missing_ok=True)
            return False
        pid = int(payload.get("pid", 0) or 0)
        if pid <= 0:
            process_file_path.unlink(missing_ok=True)
            return False
        if not self._is_process_alive(pid):
            process_file_path.unlink(missing_ok=True)
            return False
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        return completed.returncode == 0

    def _load_module_b_segment_selector_items(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：为 role3/role4 的按 segment 操作构建选择列表。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: segment 入口数组。
        异常说明：无；缺失上游产物时回退为空数组。
        边界条件：优先复用当前审阅页 segment 数据，确保 segment_id/shot_id 对齐页面观察上下文。
        """
        try:
            segment_units = self._load_segment_units(task_dir=task_dir, task_id=task_id)
        except Exception:  # noqa: BLE001
            segment_units = []
        selector_items: list[dict[str, Any]] = []
        for index, item in enumerate(segment_units, start=1):
            segment_id = str(item.get("segment_id", "")).strip()
            if not segment_id:
                continue
            selector_items.append(
                {
                    "segment_id": segment_id,
                    "shot_id": str(item.get("shot_id", "")).strip() or f"shot_{index:03d}",
                    "start_time": float(item.get("start_time", 0.0) or 0.0),
                    "end_time": float(item.get("end_time", 0.0) or 0.0),
                    "label": str(item.get("label", "")).strip(),
                    "role": str(item.get("role", "")).strip(),
                    "scene_desc": str(item.get("scene_desc", "")).strip(),
                }
            )
        return selector_items

    @staticmethod
    def _classify_module_b_rerun_failure_reason(error: Exception) -> str:
        """
        功能说明：把模块 B 重跑异常归类为更易读的失败原因。
        参数说明：
        - error: 后台线程捕获到的异常对象。
        返回值：
        - str: 适合前端展示的失败原因摘要。
        异常说明：无。
        边界条件：无法归类时回退为原始异常文本。
        """
        error_text = str(error).strip()
        lowered_text = error_text.lower()
        if (
            "必须至少包含一个 `##` 条目" in error_text
            or "缺失字段" in error_text
            or "未定义字段" in error_text
            or "重复字段" in error_text
            or "模块 b role1 输出不符合契约" in lowered_text
        ):
            return "未通过契约校验"
        if "timeout" in lowered_text or "timed out" in lowered_text or "超时" in error_text:
            return "LLM 超时"
        return error_text or "模块B重跑失败"

    def _describe_module_b_role_implementation(
        self,
        role_name: str,
        source_path: Path,
        contract_fields: list[str],
    ) -> tuple[str, str]:
        """
        功能说明：根据当前源码粗略判断模块 B role 的实现状态。
        参数说明：
        - role_name: 角色名。
        - source_path: 角色源码路径。
        - contract_fields: 当前角色对外暴露的契约字段集合。
        返回值：
        - tuple[str, str]: `(状态标识, 说明文本)`。
        异常说明：无；读取失败统一降级为 unknown。
        边界条件：仅做源码字符串级判断，不执行导入或运行角色逻辑。
        """
        if (not source_path.exists()) or (not source_path.is_file()):
            return "missing", f"未找到角色源码文件；当前契约字段为 {', '.join(contract_fields)}。"
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return "unknown", f"角色源码读取失败；当前契约字段为 {', '.join(contract_fields)}。"
        if f'module_b: {str(role_name).strip()} is not implemented.' in source_text:
            return "placeholder", f"当前源码仍为占位实现；当前契约字段为 {', '.join(contract_fields)}。"
        return "implemented", f"当前源码已接入具体执行逻辑；当前契约字段为 {', '.join(contract_fields)}。"

    def _find_task_local_module_b_role_artifact(self, task_dir: Path, role_name: str) -> Path | None:
        """
        功能说明：在当前任务 artifacts 根目录下查找当前 module_b role 对应的直系产物。
        参数说明：
        - task_dir: 任务目录。
        - role_name: 角色名。
        返回值：
        - Path | None: 找到则返回文件路径，否则返回空。
        异常说明：无。
        边界条件：仅扫描 `runs/<task_id>/artifacts` 直属文件，避免误读弃用目录。
        """
        artifacts_dir = (task_dir / "artifacts").resolve()
        if (not artifacts_dir.exists()) or (not artifacts_dir.is_dir()):
            return None
        role_pattern_map = {
            "role1": "module_b_role1*",
            "role2": "module_b_role2*",
            "role3": "module_b_role3*",
            "role4": "module_b_role4*",
        }
        pattern = str(role_pattern_map.get(str(role_name).strip(), "")).strip()
        if not pattern:
            return None
        candidate_files = sorted([item for item in artifacts_dir.glob(pattern) if item.is_file()], key=lambda item: item.name)
        if not candidate_files:
            return None
        return candidate_files[-1].resolve()

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

    def _build_display_audio_path(self, task_id: str, task_record: dict[str, Any]) -> str:
        """
        功能说明：为任务列表/详情优先展示当前机器可访问的音频路径。
        参数说明：
        - task_id: 任务唯一标识。
        - task_record: 状态库中的任务记录。
        返回值：
        - str: 可解析则返回本机绝对路径，否则保留原始记录文本。
        异常说明：无；解析失败时回退原始路径。
        边界条件：仅用于展示，不触发状态库回写。
        """
        try:
            return str(self._resolve_task_audio_path_from_record(task_id=task_id, task_record=task_record, persist=False))
        except Exception:  # noqa: BLE001
            return str(task_record.get("audio_path", ""))

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
                    "audio_path": self._build_display_audio_path(task_id=task_id, task_record=item),
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
                "audio_path": self._build_display_audio_path(task_id=normalized_task_id, task_record=task_record),
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
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        return self._submit_task_rerun_request(
            task_id=task_id,
            success_message=f"任务已开始生成，task_id={task_id}，模式=强制从A模块开始覆盖式重跑",
            log_reason="manual_rerun",
        )

    def _submit_task_rerun_request(
        self,
        *,
        task_id: str,
        success_message: str,
        log_reason: str,
    ) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：提交一次“从模块A开始”的后台重跑请求。
        参数说明：
        - task_id: 任务唯一标识。
        - success_message: 成功响应文案。
        - log_reason: 日志中记录的触发原因。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：同一任务在后台触发线程未结束前不允许重复提交。
        """
        if self.rerun_handler is None:
            return {"ok": False, "error": "当前监督服务未配置生成能力。"}, HTTPStatus.NOT_IMPLEMENTED
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
        self.logger.info(
            "[监督服务] 任务强制重跑已提交，task_id=%s，from_module=A，reason=%s",
            task_id,
            log_reason,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": success_message,
        }, HTTPStatus.OK

    def _build_module_a_payload(self, task_id: str) -> dict[str, Any]:
        """
        功能说明：构建模块 A 页面所需的数据负载。
        参数说明：
        - task_id: 目标任务ID。
        返回值：
        - dict[str, Any]: 包含模块A可视化与联网歌词状态的数据对象。
        异常说明：无；任务不存在时返回 ok=false。
        边界条件：不在此接口返回重型审阅时间线数据，避免页面重复拉取。
        """
        normalized_task_id = str(task_id).strip() or self.task_id
        task_record = self.state_store.get_task(task_id=normalized_task_id)
        if task_record is None:
            return {
                "ok": False,
                "error": f"任务不存在：{normalized_task_id}",
                "task_id": normalized_task_id,
            }
        task_dir = self._resolve_task_dir(task_id=normalized_task_id)
        visualization_path = self._resolve_module_a_visualization_path(task_dir=task_dir, task_id=normalized_task_id)
        try:
            module_status_map = self.state_store.get_module_status_map(task_id=normalized_task_id)
        except Exception:  # noqa: BLE001
            module_status_map = {}
        return {
            "ok": True,
            "task_id": normalized_task_id,
            "task_status": str(task_record.get("status", "unknown")),
            "module_a_status": str(module_status_map.get("A", "unknown")),
            "module_a_visualization": {
                "available": visualization_path is not None and visualization_path.exists(),
                "url": self._build_task_file_url(task_id=normalized_task_id, file_path=visualization_path)
                if visualization_path
                else "",
                "path": str(visualization_path) if visualization_path else "",
            },
            "network_lrc_state": self._build_module_a_network_lyrics_summary(task_dir=task_dir),
        }

    def _handle_module_a_search_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 A 的联网同步歌词搜索请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：仅返回最多10个具备同步歌词的候选。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        manual_query = str(query.get("manual_query", [""])[0]).strip()
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"联网查找歌词失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        if self.app_config is None:
            return {"ok": False, "error": "当前监督服务缺少运行配置，无法联网查找歌词。"}, HTTPStatus.SERVICE_UNAVAILABLE

        if not str(task_record.get("audio_path", "")).strip():
            return {"ok": False, "error": f"联网查找歌词失败：任务缺少 audio_path，task_id={task_id}"}, HTTPStatus.BAD_REQUEST
        try:
            audio_path = self._resolve_task_audio_path_from_record(task_id=task_id, task_record=task_record, persist=True)
        except FileNotFoundError as error:
            return {"ok": False, "error": f"联网查找歌词失败：{error}"}, HTTPStatus.NOT_FOUND

        try:
            duration_seconds = probe_audio_duration(
                audio_path=audio_path,
                ffprobe_bin=str(getattr(getattr(self.app_config, "ffmpeg", None), "ffprobe_bin", "ffprobe")),
                logger=self.logger,
            )
            search_result = search_synced_lrc_candidates(
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                fpcalc_bin=str(getattr(getattr(self.app_config, "module_a", None), "fpcalc_bin", "fpcalc")),
                acoustid_api_key_file=str(
                    getattr(getattr(self.app_config, "module_a", None), "acoustid_api_key_file", "")
                ),
                logger=self.logger,
                manual_query=manual_query,
                max_candidates=10,
                raw_candidate_limit=30,
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning("[监督服务] 模块A联网歌词搜索失败，task_id=%s，错误=%s", task_id, error)
            return {"ok": False, "error": f"联网查找歌词失败：{error}", "task_id": task_id}, HTTPStatus.BAD_GATEWAY

        self._persist_module_a_search_result(task_id=task_id, search_result=search_result)
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        metadata_trace = self._build_module_a_metadata_trace_summary(raw_state.get("metadata_trace", {}))

        status_text = str(search_result.get("status", "")).strip().lower()
        if status_text == "failed":
            error_text = str(search_result.get("error", "")).strip() or "联网查找歌词失败"
            if bool(search_result.get("suggest_manual_query", False)):
                return {
                    "ok": True,
                    "task_id": task_id,
                    "search_status": "failed",
                    "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
                    "message": str(search_result.get("message", "")).strip() or error_text,
                    "error": error_text,
                    "suggest_manual_query": True,
                    "metadata_trace": metadata_trace,
                    "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
                    "candidates": [],
                }, HTTPStatus.OK
            return {"ok": False, "error": error_text, "task_id": task_id}, HTTPStatus.BAD_GATEWAY

        candidates = [
            self._build_module_a_candidate_summary(item)
            for item in raw_state.get("candidates", [])
            if isinstance(item, dict)
        ]
        if status_text == "not_found":
            return {
                "ok": True,
                "task_id": task_id,
                "search_status": "not_found",
                "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
                "message": str(search_result.get("message", "")).strip()
                or str(search_result.get("error", "")).strip()
                or "未找到可用的同步lrc歌词候选",
                "suggest_manual_query": bool(search_result.get("suggest_manual_query", False)),
                "metadata_trace": metadata_trace,
                "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
                "candidates": [],
            }, HTTPStatus.OK
        return {
            "ok": True,
            "task_id": task_id,
            "search_status": "ok",
            "search_mode": str(search_result.get("search_mode", "automatic")).strip(),
            "message": str(search_result.get("message", "")).strip() or f"已找到 {len(candidates)} 个同步lrc歌词候选",
            "suggest_manual_query": False,
            "metadata_trace": metadata_trace,
            "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
            "candidates": candidates,
        }, HTTPStatus.OK

    def _persist_module_a_search_result(self, task_id: str, search_result: dict[str, Any]) -> None:
        """
        功能说明：把模块A联网歌词搜索结果写入任务状态文件。
        参数说明：
        - task_id: 任务唯一标识。
        - search_result: 搜索结果对象。
        返回值：无。
        异常说明：无；调用方负责外围异常处理。
        边界条件：仅持久化前端需要的最小字段。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        raw_state["updated_at"] = current_time_text()
        raw_state["last_search_at"] = raw_state["updated_at"]
        raw_state["search_status"] = str(search_result.get("status", "")).strip()
        raw_state["lookup_error"] = str(search_result.get("error", "")).strip()
        raw_state["fingerprint_status"] = str(
            search_result.get("fingerprint_result", {}).get("status", "")
            if isinstance(search_result.get("fingerprint_result", {}), dict)
            else ""
        ).strip()
        raw_state["acoustid_status"] = str(
            search_result.get("acoustid_result", {}).get("status", "")
            if isinstance(search_result.get("acoustid_result", {}), dict)
            else ""
        ).strip()
        raw_state["metadata_trace"] = (
            dict(search_result.get("metadata_trace", {}))
            if isinstance(search_result.get("metadata_trace", {}), dict)
            else {}
        )
        raw_state["candidates"] = [
            dict(item)
            for item in search_result.get("candidates", [])
            if isinstance(item, dict)
        ]
        raw_state["provider_groups"] = [
            dict(item)
            for item in search_result.get("provider_groups", [])
            if isinstance(item, dict)
        ]
        write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)

    def _handle_module_a_select_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 A 页面“选中候选歌词并决定是否启用”的请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：启用时会立即触发从模块A开始的后台重跑。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        candidate_id = str(query.get("candidate_id", [""])[0]).strip()
        enable_text = str(query.get("enable", ["0"])[0]).strip().lower()
        enable_lookup = enable_text in {"1", "true", "yes", "enabled"}
        if not candidate_id:
            return {"ok": False, "error": "候选歌词选择失败：candidate_id 不能为空。"}, HTTPStatus.BAD_REQUEST

        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"候选歌词选择失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND

        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        selected_candidate = self._find_module_a_candidate_by_id(
            candidates=raw_state.get("candidates", []),
            candidate_id=candidate_id,
        )
        if selected_candidate is None:
            return {
                "ok": False,
                "error": f"候选歌词选择失败：未找到 candidate_id={candidate_id}，请重新联网查找。",
            }, HTTPStatus.NOT_FOUND
        if enable_lookup and not str(selected_candidate.get("synced_lyrics", "")).strip():
            return {"ok": False, "error": "候选歌词选择失败：当前候选不包含可用的同步lrc歌词。"}, HTTPStatus.BAD_REQUEST

        raw_state["selected_candidate_id"] = candidate_id
        raw_state["selected_candidate"] = dict(selected_candidate)
        raw_state["enabled"] = bool(enable_lookup)
        raw_state["display_status"] = "enabled" if enable_lookup else "searched_not_enabled"
        raw_state["updated_at"] = current_time_text()
        raw_state["lookup_error"] = ""
        write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)

        if not enable_lookup:
            return {
                "ok": True,
                "task_id": task_id,
                "message": "已联网查找lrc但未启用",
            }, HTTPStatus.OK

        payload, status = self._submit_task_rerun_request(
            task_id=task_id,
            success_message="已经启用联网查找的lrc，并开始重跑模块A",
            log_reason=f"module_a_network_lrc:{candidate_id}",
        )
        return payload, status

    def _handle_module_a_candidate_lyrics_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：按候选ID返回模块 A 联网歌词候选的完整同步歌词内容。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：仅从已缓存候选中读取，不触发新的联网搜索。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        candidate_id = str(query.get("candidate_id", [""])[0]).strip()
        if not candidate_id:
            return {"ok": False, "error": "候选歌词详情读取失败：candidate_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"候选歌词详情读取失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        task_dir = self._resolve_task_dir(task_id=task_id)
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        candidate = self._find_module_a_candidate_by_id(
            candidates=raw_state.get("candidates", []),
            candidate_id=candidate_id,
        )
        if candidate is None:
            return {
                "ok": False,
                "error": f"候选歌词详情读取失败：未找到 candidate_id={candidate_id}。",
            }, HTTPStatus.NOT_FOUND
        candidate = self._hydrate_module_a_candidate_detail(task_id=task_id, artifacts_dir=artifacts_dir, raw_state=raw_state, candidate=candidate)
        return {
            "ok": True,
            "task_id": task_id,
            "candidate": self._build_module_a_candidate_summary(candidate),
            "synced_lyrics": str(candidate.get("synced_lyrics", "")).strip(),
            "word_timed_lyrics": str(candidate.get("word_timed_lyrics", "")).strip(),
            "translated_lyrics": str(candidate.get("translated_lyrics", "")).strip(),
            "romanized_lyrics": str(candidate.get("romanized_lyrics", "")).strip(),
        }, HTTPStatus.OK

    def _build_module_a_network_lyrics_summary(self, task_dir: Path) -> dict[str, Any]:
        """
        功能说明：构建模块 A 页面所需的联网歌词状态摘要。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, Any]: 轻量状态摘要。
        异常说明：无。
        边界条件：不返回完整歌词正文，避免页面重复拉取大字段。
        """
        artifacts_dir = task_dir / "artifacts"
        raw_state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
        return {
            "display_status": str(raw_state.get("display_status", "idle")).strip() or "idle",
            "enabled": bool(raw_state.get("enabled", False)),
            "updated_at": str(raw_state.get("updated_at", "")).strip(),
            "last_search_at": str(raw_state.get("last_search_at", "")).strip(),
            "search_status": str(raw_state.get("search_status", "")).strip(),
            "lookup_error": str(raw_state.get("lookup_error", "")).strip(),
            "cached_candidates_count": len(raw_state.get("candidates", []))
            if isinstance(raw_state.get("candidates", []), list)
            else 0,
            "metadata_trace": self._build_module_a_metadata_trace_summary(raw_state.get("metadata_trace", {})),
            "provider_groups": self._build_module_a_provider_group_summaries(raw_state.get("provider_groups", [])),
            "selected_candidate": self._build_module_a_candidate_summary(raw_state.get("selected_candidate", {})),
        }

    def _fetch_module_a_candidate_synced_lyrics(self, candidate: dict[str, Any]) -> str:
        """
        功能说明：当缓存候选未带正文时，按来源即时补拉同步歌词。
        参数说明：
        - candidate: 候选对象。
        返回值：
        - str: 补拉到的同步歌词正文；失败时返回空字符串。
        异常说明：无；内部异常统一记录日志并回退空字符串。
        边界条件：优先复用 provider/provider_id，必要时回退 artist/title 检索。
        """
        if not isinstance(candidate, dict):
            return ""
        provider = str(candidate.get("provider", "")).strip().lower()
        provider_id = str(candidate.get("provider_id", "")).strip()
        artist = str(candidate.get("artist", "")).strip()
        title = str(candidate.get("title", "")).strip()
        duration_seconds = float(candidate.get("duration_seconds", 0.0) or 0.0)
        try:
            if provider == "qq_music" and provider_id:
                return fetch_qq_music_synced_lyrics(song_mid=provider_id, logger=self.logger)
            if provider == "netease_music" and provider_id:
                return fetch_netease_music_synced_lyrics(song_id=provider_id, logger=self.logger)
            if provider == "kugou_music" and provider_id:
                return fetch_kugou_music_synced_lyrics(
                    lyric_id=provider_id,
                    accesskey=str(candidate.get("provider_accesskey", "")).strip(),
                    logger=self.logger,
                )
            if provider.startswith("syncedlyrics"):
                from music_video_pipeline.modules.module_a_v2.lyrics_lookup.providers.syncedlyrics import (
                    search_syncedlyrics_candidates,
                    search_syncedlyrics_candidates_by_provider,
                )

                if provider.startswith("syncedlyrics::"):
                    provider_name = provider.split("::", 1)[1].strip()
                    results = search_syncedlyrics_candidates_by_provider(
                        provider_name=provider_name,
                        artist=artist,
                        title=title,
                        logger=self.logger,
                        limit=1,
                    )
                else:
                    results = search_syncedlyrics_candidates(
                        artist=artist,
                        title=title,
                        logger=self.logger,
                        limit=1,
                    )
                if results:
                    return str(results[0].get("synced_lyrics", "")).strip()
            if provider == "lrclib" or (artist and title):
                result = query_lrclib_lyrics(
                    artist=artist,
                    title=title,
                    duration_seconds=duration_seconds,
                    logger=self.logger,
                )
                return str(result.get("synced_lyrics", "")).strip()
        except Exception as error:  # noqa: BLE001
            self.logger.warning(
                "[监督服务] 模块A候选歌词详情补拉失败，provider=%s，provider_id=%s，artist=%s，title=%s，错误=%s",
                provider,
                provider_id,
                artist,
                title,
                error,
            )
        return ""

    def _hydrate_module_a_candidate_detail(
        self,
        *,
        task_id: str,
        artifacts_dir: Path,
        raw_state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """
        功能说明：为候选详情补齐原文、翻译、罗马音并持久化回缓存。
        参数说明：
        - task_id: 任务ID。
        - artifacts_dir: 任务产物目录。
        - raw_state: 原始联网歌词缓存状态。
        - candidate: 当前候选对象。
        返回值：
        - dict[str, Any]: 补齐后的候选对象。
        异常说明：无；内部异常统一吞掉并返回原候选。
        边界条件：仅在缺字段时回源补拉，避免重复联网。
        """
        if not isinstance(candidate, dict):
            return {}
        enriched_candidate = dict(candidate)
        provider = str(enriched_candidate.get("provider", "")).strip().lower()
        needs_sync = not str(enriched_candidate.get("synced_lyrics", "")).strip()
        needs_translation = not str(enriched_candidate.get("translated_lyrics", "")).strip()
        needs_romanized = not str(enriched_candidate.get("romanized_lyrics", "")).strip()
        needs_word_timed = not str(enriched_candidate.get("word_timed_lyrics", "")).strip()
        if provider == "qq_music" and (needs_sync or needs_translation or needs_romanized):
            try:
                bundle = fetch_qq_music_lyrics_bundle(
                    song_mid=str(enriched_candidate.get("provider_id", "")).strip(),
                    song_id=str(enriched_candidate.get("provider_song_id", "")).strip(),
                    artist=str(enriched_candidate.get("artist", "")).strip(),
                    title=str(enriched_candidate.get("title", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选QQ富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if provider == "kugou_music" and (needs_sync or needs_word_timed or needs_translation or needs_romanized):
            try:
                bundle = fetch_kugou_music_lyrics_bundle(
                    lyric_id=str(enriched_candidate.get("provider_id", "")).strip(),
                    accesskey=str(enriched_candidate.get("provider_accesskey", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选酷狗富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_word_timed and str(bundle.get("word_timed_lyrics", "")).strip():
                enriched_candidate["word_timed_lyrics"] = str(bundle.get("word_timed_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if provider == "netease_music" and (needs_sync or needs_word_timed or needs_translation or needs_romanized):
            try:
                bundle = fetch_netease_music_lyrics_bundle(
                    song_id=str(enriched_candidate.get("provider_id", "")).strip(),
                    logger=self.logger,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.warning(
                    "[监督服务] 模块A候选网易云富歌词补拉失败，task_id=%s，candidate_id=%s，错误=%s",
                    task_id,
                    str(enriched_candidate.get("candidate_id", "")).strip(),
                    error,
                )
                bundle = {}
            if needs_sync and str(bundle.get("synced_lyrics", "")).strip():
                enriched_candidate["synced_lyrics"] = str(bundle.get("synced_lyrics", "")).strip()
            if needs_word_timed and str(bundle.get("word_timed_lyrics", "")).strip():
                enriched_candidate["word_timed_lyrics"] = str(bundle.get("word_timed_lyrics", "")).strip()
            if needs_translation and str(bundle.get("translated_lyrics", "")).strip():
                enriched_candidate["translated_lyrics"] = str(bundle.get("translated_lyrics", "")).strip()
            if needs_romanized and str(bundle.get("romanized_lyrics", "")).strip():
                enriched_candidate["romanized_lyrics"] = str(bundle.get("romanized_lyrics", "")).strip()
        if not str(enriched_candidate.get("synced_lyrics", "")).strip():
            synced_lyrics = self._fetch_module_a_candidate_synced_lyrics(candidate=enriched_candidate)
            if synced_lyrics:
                enriched_candidate["synced_lyrics"] = synced_lyrics
        if enriched_candidate != candidate:
            raw_candidates = raw_state.get("candidates", [])
            candidate_id = str(enriched_candidate.get("candidate_id", "")).strip()
            if isinstance(raw_candidates, list):
                for index, item in enumerate(raw_candidates):
                    if isinstance(item, dict) and str(item.get("candidate_id", "")).strip() == candidate_id:
                        raw_candidates[index] = dict(enriched_candidate)
                        break
            selected_candidate = raw_state.get("selected_candidate", {})
            if isinstance(selected_candidate, dict) and str(selected_candidate.get("candidate_id", "")).strip() == candidate_id:
                raw_state["selected_candidate"] = dict(enriched_candidate)
            write_module_a_network_lyrics_state(artifacts_dir=artifacts_dir, payload=raw_state)
        return enriched_candidate

    def _build_module_a_metadata_trace_summary(self, metadata_trace: Any) -> dict[str, Any]:
        """
        功能说明：将联网找词诊断摘要裁剪为模块A页面所需结构。
        参数说明：
        - metadata_trace: 原始诊断摘要对象。
        返回值：
        - dict[str, Any]: 前端稳定可用的摘要对象。
        异常说明：无。
        边界条件：非法输入时回退为空摘要。
        """
        if not isinstance(metadata_trace, dict):
            metadata_trace = {}
        return {
            "embedded_status": str(metadata_trace.get("embedded_status", "")).strip(),
            "embedded_source": str(metadata_trace.get("embedded_source", "")).strip(),
            "embedded_artist": str(metadata_trace.get("embedded_artist", "")).strip(),
            "embedded_title": str(metadata_trace.get("embedded_title", "")).strip(),
            "embedded_album": str(metadata_trace.get("embedded_album", "")).strip(),
            "embedded_error": str(metadata_trace.get("embedded_error", "")).strip(),
            "fingerprint_status": str(metadata_trace.get("fingerprint_status", "")).strip(),
            "fingerprint_error": str(metadata_trace.get("fingerprint_error", "")).strip(),
            "acoustid_status": str(metadata_trace.get("acoustid_status", "")).strip(),
            "matched_artist": str(metadata_trace.get("matched_artist", "")).strip(),
            "matched_title": str(metadata_trace.get("matched_title", "")).strip(),
            "matched_score": float(metadata_trace.get("matched_score", 0.0) or 0.0),
            "matched_error": str(metadata_trace.get("matched_error", "")).strip(),
        }

    def _build_module_a_candidate_summary(self, candidate: Any) -> dict[str, Any]:
        """
        功能说明：将模块 A 联网歌词候选裁剪为适合前端展示的摘要。
        参数说明：
        - candidate: 原始候选对象。
        返回值：
        - dict[str, Any]: 摘要对象。
        异常说明：无。
        边界条件：输入非法时返回空摘要。
        """
        if not isinstance(candidate, dict):
            return {
                "candidate_id": "",
                "artist": "",
                "title": "",
                "score": 0.0,
                "provider": "",
                "provider_id": "",
                "provider_song_id": "",
                "has_word_timed_lyrics": False,
                "has_translated_lyrics": False,
                "has_romanized_lyrics": False,
                "preview_lines": [],
                "preview_text": "",
            }
        preview_lines = candidate.get("preview_lines", [])
        normalized_preview_lines = [str(item).strip() for item in preview_lines if str(item).strip()] if isinstance(preview_lines, list) else []
        preview_text = str(candidate.get("preview_text", "")).strip()
        if not preview_text and normalized_preview_lines:
            preview_text = "\n".join(normalized_preview_lines)
        return {
            "candidate_id": str(candidate.get("candidate_id", "")).strip(),
            "artist": str(candidate.get("artist", "")).strip(),
            "title": str(candidate.get("title", "")).strip(),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "provider": str(candidate.get("provider", "lrclib")).strip(),
            "provider_id": str(candidate.get("provider_id", "")).strip(),
            "provider_song_id": str(candidate.get("provider_song_id", "")).strip(),
            "has_word_timed_lyrics": bool(str(candidate.get("word_timed_lyrics", "")).strip()),
            "has_translated_lyrics": bool(str(candidate.get("translated_lyrics", "")).strip()),
            "has_romanized_lyrics": bool(str(candidate.get("romanized_lyrics", "")).strip()),
            "preview_lines": normalized_preview_lines,
            "preview_text": preview_text,
        }

    def _build_module_a_provider_group_summaries(self, provider_groups: Any) -> list[dict[str, Any]]:
        """
        功能说明：将来源分组候选裁剪为适合前端展示的结构。
        参数说明：
        - provider_groups: 原始来源分组数组。
        返回值：
        - list[dict[str, Any]]: 前端稳定可用的来源分组摘要数组。
        异常说明：无。
        边界条件：非法输入时返回空数组。
        """
        if not isinstance(provider_groups, list):
            return []
        normalized_groups: list[dict[str, Any]] = []
        for provider_group in provider_groups:
            if not isinstance(provider_group, dict):
                continue
            candidates = provider_group.get("candidates", [])
            normalized_groups.append(
                {
                    "provider": str(provider_group.get("provider", "")).strip(),
                    "display_name": str(provider_group.get("display_name", "")).strip(),
                    "candidates": [
                        self._build_module_a_candidate_summary(item)
                        for item in candidates
                        if isinstance(item, dict)
                    ],
                }
            )
        return normalized_groups

    def _find_module_a_candidate_by_id(self, candidates: Any, candidate_id: str) -> dict[str, Any] | None:
        """
        功能说明：在缓存候选数组中按 candidate_id 查找目标项。
        参数说明：
        - candidates: 候选数组。
        - candidate_id: 目标候选ID。
        返回值：
        - dict[str, Any] | None: 命中返回候选对象，否则返回 None。
        异常说明：无。
        边界条件：仅接受字典数组。
        """
        if not isinstance(candidates, list):
            return None
        normalized_candidate_id = str(candidate_id).strip()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get("candidate_id", "")).strip() == normalized_candidate_id:
                return item
        return None

    def _handle_module_b_role_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 B role 级重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：当前默认仅提供占位入口，未来接入真实 handler 后复用同一 API。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        role_name = str(query.get("role_name", [""])[0]).strip().lower()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if role_name not in {"role1", "role2", "role3", "role4"}:
            return {"ok": False, "error": f"模块B role 重跑失败：非法 role_name={role_name or '<empty>'}。"}, HTTPStatus.BAD_REQUEST
        if self.module_b_role_rerun_handler is None:
            return {
                "ok": False,
                "error": f"当前 module_b 新方案 role 重跑尚未接通，role_name={role_name}。",
            }, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块B role 重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        active_process_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
        has_active_process = bool(active_process_meta.get("active"))
        if replace_running and (has_active_process or (active_thread is not None and active_thread.is_alive())):
            terminated = self._terminate_active_module_b_rerun_process(task_id=task_id)
            if active_thread is not None and active_thread.is_alive():
                active_thread.join(timeout=5.0)
            if not terminated and active_thread is not None and active_thread.is_alive():
                return {
                    "ok": False,
                    "error": f"模块B role 重跑失败：旧进程仍在退出中，task_id={task_id}",
                }, HTTPStatus.CONFLICT
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"模块B role 重跑失败：任务已有后台动作执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        if self._load_active_module_b_rerun_process_meta(task_id=task_id).get("active"):
            return {"ok": False, "error": f"模块B role 重跑失败：任务已有后台子进程执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        rerun_thread = threading.Thread(
            target=self._run_module_b_role_rerun_in_background,
            name=f"module-b-role-rerun-{task_id}-{role_name}",
            args=(task_id, role_name),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        self._rerun_thread_meta[task_id] = {
            "active": True,
            "status": "queued",
            "mode": "role",
            "role_name": role_name,
            "segment_id": "",
            "shot_id": "",
            "submitted_at": current_time_text(),
            "submitted_at_ms": int(time.time() * 1000),
            "started_at": "",
            "started_at_ms": 0,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()
        self.logger.info("[监督服务] 模块B role 重跑已提交，task_id=%s，role_name=%s", task_id, role_name)
        return {
            "ok": True,
            "task_id": task_id,
            "message": f"模块B role 重跑已提交，task_id={task_id}，role_name={role_name}",
        }, HTTPStatus.OK

    def _handle_module_b_role_segment_rerun_request(self, parsed: Any) -> tuple[dict[str, Any], HTTPStatus]:
        """
        功能说明：处理模块 B role 内 segment 级重跑请求。
        参数说明：
        - parsed: 已解析的请求URL对象。
        返回值：
        - tuple[dict[str, Any], HTTPStatus]: JSON响应与状态码。
        异常说明：无；错误统一转为 JSON。
        边界条件：当前仅开放 role3/role4；shot_id 由当前任务 segment 映射推导。
        """
        query = parse_qs(parsed.query)
        task_id = str(query.get("task_id", [self.task_id])[0]).strip() or self.task_id
        role_name = str(query.get("role_name", [""])[0]).strip().lower()
        segment_id = str(query.get("segment_id", [""])[0]).strip()
        replace_running = self._parse_truthy_flag(str(query.get("replace_running", ["0"])[0]))
        if role_name not in {"role3", "role4"}:
            return {
                "ok": False,
                "error": f"模块B segment 重跑失败：仅支持 role3/role4，role_name={role_name or '<empty>'}。",
            }, HTTPStatus.BAD_REQUEST
        if not segment_id:
            return {"ok": False, "error": "模块B segment 重跑失败：segment_id 不能为空。"}, HTTPStatus.BAD_REQUEST
        if self.module_b_role_segment_rerun_handler is None:
            return {
                "ok": False,
                "error": f"当前 module_b 新方案 segment 重跑尚未接通，role_name={role_name}，segment_id={segment_id}。",
            }, HTTPStatus.NOT_IMPLEMENTED
        task_record = self.state_store.get_task(task_id=task_id)
        if task_record is None:
            return {"ok": False, "error": f"模块B segment 重跑失败：任务不存在，task_id={task_id}"}, HTTPStatus.NOT_FOUND
        active_thread = self._rerun_threads.get(task_id)
        active_process_meta = self._load_active_module_b_rerun_process_meta(task_id=task_id)
        has_active_process = bool(active_process_meta.get("active"))
        if replace_running and (has_active_process or (active_thread is not None and active_thread.is_alive())):
            terminated = self._terminate_active_module_b_rerun_process(task_id=task_id)
            if active_thread is not None and active_thread.is_alive():
                active_thread.join(timeout=5.0)
            if not terminated and active_thread is not None and active_thread.is_alive():
                return {
                    "ok": False,
                    "error": f"模块B segment 重跑失败：旧进程仍在退出中，task_id={task_id}",
                }, HTTPStatus.CONFLICT
        if active_thread is not None and active_thread.is_alive():
            return {"ok": False, "error": f"模块B segment 重跑失败：任务已有后台动作执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        if self._load_active_module_b_rerun_process_meta(task_id=task_id).get("active"):
            return {"ok": False, "error": f"模块B segment 重跑失败：任务已有后台子进程执行中，task_id={task_id}"}, HTTPStatus.CONFLICT
        task_dir = self._resolve_task_dir(task_id=task_id)
        shot_id = self._resolve_module_b_shot_id_from_segment(task_dir=task_dir, task_id=task_id, segment_id=segment_id)
        if not shot_id:
            return {
                "ok": False,
                "error": f"模块B segment 重跑失败：无法从当前任务解析 shot_id，segment_id={segment_id}。",
            }, HTTPStatus.NOT_FOUND
        rerun_thread = threading.Thread(
            target=self._run_module_b_role_segment_rerun_in_background,
            name=f"module-b-role-segment-rerun-{task_id}-{role_name}-{segment_id}",
            args=(task_id, role_name, segment_id, shot_id),
            daemon=True,
        )
        self._rerun_threads[task_id] = rerun_thread
        self._rerun_thread_meta[task_id] = {
            "active": True,
            "status": "queued",
            "mode": "segment",
            "role_name": role_name,
            "segment_id": segment_id,
            "shot_id": shot_id,
            "submitted_at": current_time_text(),
            "submitted_at_ms": int(time.time() * 1000),
            "started_at": "",
            "started_at_ms": 0,
            "finished_at": "",
            "finished_at_ms": 0,
            "duration_ms": 0,
            "last_error": "",
            "failure_reason": "",
        }
        rerun_thread.start()
        self.logger.info(
            "[监督服务] 模块B segment 重跑已提交，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
            task_id,
            role_name,
            segment_id,
            shot_id,
        )
        return {
            "ok": True,
            "task_id": task_id,
            "message": (
                f"模块B segment 重跑已提交，task_id={task_id}，role_name={role_name}，"
                f"segment_id={segment_id}，shot_id={shot_id}"
            ),
        }, HTTPStatus.OK

    def _resolve_module_b_shot_id_from_segment(self, task_dir: Path, task_id: str, segment_id: str) -> str:
        """
        功能说明：根据当前任务的 segment 上下文解析对应 shot_id。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        - segment_id: 目标 segment 标识。
        返回值：
        - str: 命中的 shot_id；找不到时返回空字符串。
        异常说明：无。
        边界条件：优先复用当前模块 B 页面使用的 segment 列表。
        """
        normalized_segment_id = str(segment_id).strip()
        if not normalized_segment_id:
            return ""
        for item in self._load_module_b_segment_selector_items(task_dir=task_dir, task_id=task_id):
            if str(item.get("segment_id", "")).strip() == normalized_segment_id:
                return str(item.get("shot_id", "")).strip()
        return ""

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
            self.logger.info("[监督服务] 后台开始执行任务强制重跑，task_id=%s，from_module=A", task_id)
            self.rerun_handler(task_id)
            self.logger.info("[监督服务] 后台任务强制重跑执行结束，task_id=%s", task_id)
        except Exception as error:  # noqa: BLE001
            self.logger.error("[监督服务] 后台任务强制重跑失败，task_id=%s，错误信息=%s", task_id, error)
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)
                self._rerun_thread_meta.pop(task_id, None)

    def _run_module_b_role_rerun_in_background(self, task_id: str, role_name: str) -> None:
        """
        功能说明：在后台线程中执行模块 B role 级重跑。
        参数说明：
        - task_id: 任务唯一标识。
        - role_name: 模块 B 角色名。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        started_at_ms = int(time.time() * 1000)
        meta = self._rerun_thread_meta.get(task_id)
        if isinstance(meta, dict):
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at"] = current_time_text()
            meta["started_at_ms"] = started_at_ms
            meta["finished_at"] = ""
            meta["finished_at_ms"] = 0
            meta["duration_ms"] = 0
            meta["last_error"] = ""
            meta["failure_reason"] = ""
        try:
            self.logger.info("[监督服务] 后台开始执行模块B role 重跑，task_id=%s，role_name=%s", task_id, role_name)
            if self.module_b_role_rerun_handler is None:
                raise RuntimeError(f"模块B role 重跑 handler 缺失，task_id={task_id}，role_name={role_name}")
            self.module_b_role_rerun_handler(task_id, role_name)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
            self.logger.info("[监督服务] 后台模块B role 重跑执行结束，task_id=%s，role_name=%s", task_id, role_name)
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = self._classify_module_b_rerun_failure_reason(error)
            self.logger.error(
                "[监督服务] 后台模块B role 重跑失败，task_id=%s，role_name=%s，错误信息=%s",
                task_id,
                role_name,
                error,
            )
        finally:
            current_thread = self._rerun_threads.get(task_id)
            if current_thread is threading.current_thread():
                self._rerun_threads.pop(task_id, None)

    def _run_module_b_role_segment_rerun_in_background(
        self,
        task_id: str,
        role_name: str,
        segment_id: str,
        shot_id: str,
    ) -> None:
        """
        功能说明：在后台线程中执行模块 B role 内 segment 级重跑。
        参数说明：
        - task_id: 任务唯一标识。
        - role_name: 模块 B 角色名。
        - segment_id: 目标 segment 标识。
        - shot_id: 解析后的目标 shot_id。
        返回值：无。
        异常说明：异常统一记录日志，不向前端线程传播。
        边界条件：线程退出时必须清理并发占位。
        """
        started_at_ms = int(time.time() * 1000)
        meta = self._rerun_thread_meta.get(task_id)
        if isinstance(meta, dict):
            meta["active"] = True
            meta["status"] = "running"
            meta["started_at"] = current_time_text()
            meta["started_at_ms"] = started_at_ms
            meta["finished_at"] = ""
            meta["finished_at_ms"] = 0
            meta["duration_ms"] = 0
            meta["last_error"] = ""
            meta["failure_reason"] = ""
        try:
            self.logger.info(
                "[监督服务] 后台开始执行模块B segment 重跑，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
                task_id,
                role_name,
                segment_id,
                shot_id,
            )
            if self.module_b_role_segment_rerun_handler is None:
                raise RuntimeError(
                    f"模块B segment 重跑 handler 缺失，task_id={task_id}，role_name={role_name}，shot_id={shot_id}"
                )
            self.module_b_role_segment_rerun_handler(task_id, role_name, shot_id)
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "succeeded"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
            self.logger.info(
                "[监督服务] 后台模块B segment 重跑执行结束，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s",
                task_id,
                role_name,
                segment_id,
                shot_id,
            )
        except Exception as error:  # noqa: BLE001
            finished_at_ms = int(time.time() * 1000)
            meta = self._rerun_thread_meta.get(task_id)
            if isinstance(meta, dict):
                meta["active"] = False
                meta["status"] = "failed"
                meta["finished_at"] = current_time_text()
                meta["finished_at_ms"] = finished_at_ms
                meta["duration_ms"] = max(0, finished_at_ms - int(meta.get("started_at_ms", started_at_ms) or started_at_ms))
                meta["last_error"] = str(error).strip()
                meta["failure_reason"] = self._classify_module_b_rerun_failure_reason(error)
            self.logger.error(
                "[监督服务] 后台模块B segment 重跑失败，task_id=%s，role_name=%s，segment_id=%s，shot_id=%s，错误信息=%s",
                task_id,
                role_name,
                segment_id,
                shot_id,
                error,
            )
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

    def _load_lyric_units(
        self,
        task_dir: Path,
        review_segment_units: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        功能说明：读取歌词时间戳数组，并按审阅时间线重挂 segment_id。
        参数说明：
        - task_dir: 任务目录。
        - review_segment_units: 审阅页当前采用的 segment 时间线；传入时会把歌词挂到该时间线。
        返回值：
        - list[dict[str, Any]]: lyric_units 数组，供 Web 前端按时间滚动显示。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：当成片仍停留在旧 C/D 产物时，允许歌词沿用音频对齐结果，但 segment_id 必须对准旧审阅时间线。
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
        if not review_segment_units:
            return normalized_items
        return self._reattach_lyric_units_to_review_segments(
            lyric_units=normalized_items,
            review_segment_units=review_segment_units,
        )

    def _load_segment_units(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：构建审阅页 segment 数组；优先使用当前成片对应的模块D时间线。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: segments 数组，供 Web 前端按播放时间滚动高亮。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：若存在 module_d_output.json，则审阅页以旧 C/D 产物为准，避免在 A/B 重跑后混入新时间线。
        """
        module_d_segments = self._load_segment_units_from_module_d(task_dir=task_dir, task_id=task_id)
        if module_d_segments:
            return module_d_segments
        module_a_output_path = task_dir / "artifacts" / "module_a_output.json"
        if not module_a_output_path.exists():
            return []
        payload = self._load_json_file(module_a_output_path)
        if not isinstance(payload, dict):
            return []
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list):
            return []
        segment_item_map = self._load_module_b_segment_item_map(task_dir=task_dir, task_id=task_id)
        frame_item_map = self._load_frame_item_map_by_shot_id(task_dir=task_dir, task_id=task_id)
        normalized_items: list[dict[str, Any]] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id", "")).strip()
            segment_item = segment_item_map.get(segment_id, {})
            shot_id = str(segment_item.get("shot_id", "")).strip()
            frame_item = frame_item_map.get(shot_id, {}) if shot_id else {}
            normalized_items.append(
                {
                    "segment_id": segment_id,
                    "big_segment_id": str(item.get("big_segment_id", "")),
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", float(item.get("start_time", 0.0)))),
                    "label": str(item.get("label", "")),
                    "role": str(item.get("role", "")),
                    "scene_desc": str(segment_item.get("scene_desc", "")).strip(),
                    "shot_id": shot_id,
                    "camera_plan": segment_item.get("camera_plan", {}),
                    "keyframe_prompt_start_zh": str(frame_item.get("keyframe_prompt_start_zh", "") or segment_item.get("keyframe_prompt_start_zh", "")).strip(),
                    "keyframe_prompt_start_en": str(frame_item.get("keyframe_prompt_start_en", "") or segment_item.get("keyframe_prompt_start_en", "")).strip(),
                    "keyframe_prompt_end_zh": str(frame_item.get("keyframe_prompt_end_zh", "") or segment_item.get("keyframe_prompt_end_zh", "")).strip(),
                    "keyframe_prompt_end_en": str(frame_item.get("keyframe_prompt_end_en", "") or segment_item.get("keyframe_prompt_end_en", "")).strip(),
                    "video_prompt_zh": str(frame_item.get("video_prompt_zh", "") or segment_item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(frame_item.get("video_prompt_en", "") or segment_item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": str(frame_item.get("frame_path_start", "")).strip(),
                    "frame_path_end": str(frame_item.get("frame_path_end", "")).strip(),
                    "frame_url_start": str(frame_item.get("frame_url_start", "")).strip(),
                    "frame_url_end": str(frame_item.get("frame_url_end", "")).strip(),
                }
            )
        return normalized_items

    def _load_segment_units_from_module_d(self, task_dir: Path, task_id: str) -> list[dict[str, Any]]:
        """
        功能说明：从模块D标准输出恢复与当前成片一致的审阅页 segment 数组。
        参数说明：
        - task_dir: 任务目录。
        - task_id: 任务唯一标识。
        返回值：
        - list[dict[str, Any]]: 以模块D为准的旧 segment 时间线；不可用时返回空数组。
        异常说明：读取失败时返回空数组，不中断页面。
        边界条件：模块D不保存 frame_url，需按 shot_id 回查模块C关键帧文件。
        """
        module_d_output_path = task_dir / "artifacts" / "module_d_output.json"
        payload = self._load_json_file(module_d_output_path)
        raw_segment_items = payload.get("segment_items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_segment_items, list):
            return []
        frame_item_map = self._load_frame_item_map_by_shot_id(task_dir=task_dir, task_id=task_id)
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_segment_items):
            if not isinstance(item, dict):
                continue
            shot_id = str(item.get("shot_id", "")).strip()
            frame_item = frame_item_map.get(shot_id, {}) if shot_id else {}
            unit_index_raw = item.get("unit_index", index)
            try:
                unit_index = max(0, int(unit_index_raw))
            except (TypeError, ValueError):
                unit_index = max(0, index)
            segment_id = self._build_review_segment_id(
                explicit_segment_id=str(item.get("segment_id", "")).strip(),
                unit_index=unit_index,
                shot_id=shot_id,
                fallback_index=index,
            )
            segment_path = self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(item.get("segment_path", "")),
            )
            segment_frame_path_start, segment_frame_path_end = self._resolve_module_d_segment_frame_paths(
                task_id=task_id,
                shot_id=shot_id,
                segment_path=segment_path,
            )
            resolved_frame_path_start = segment_frame_path_start or self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(frame_item.get("frame_path_start", "")),
            )
            resolved_frame_path_end = segment_frame_path_end or self._resolve_task_artifact_path(
                task_id=task_id,
                raw_path=str(frame_item.get("frame_path_end", "")),
            )
            normalized_items.append(
                {
                    "segment_id": segment_id,
                    "big_segment_id": "",
                    "start_time": float(item.get("start_time", frame_item.get("start_time", 0.0))),
                    "end_time": float(item.get("end_time", frame_item.get("end_time", item.get("start_time", 0.0)))),
                    "label": "",
                    "role": "",
                    "scene_desc": str(item.get("scene_desc", "") or frame_item.get("scene_desc", "")).strip(),
                    "shot_id": shot_id,
                    "camera_plan": item.get("camera_plan", frame_item.get("camera_plan", {})),
                    "keyframe_prompt_start_zh": str(
                        item.get("keyframe_prompt_start_zh", "") or frame_item.get("keyframe_prompt_start_zh", "")
                    ).strip(),
                    "keyframe_prompt_start_en": str(
                        item.get("keyframe_prompt_start_en", "") or frame_item.get("keyframe_prompt_start_en", "")
                    ).strip(),
                    "keyframe_prompt_end_zh": str(
                        item.get("keyframe_prompt_end_zh", "") or frame_item.get("keyframe_prompt_end_zh", "")
                    ).strip(),
                    "keyframe_prompt_end_en": str(
                        item.get("keyframe_prompt_end_en", "") or frame_item.get("keyframe_prompt_end_en", "")
                    ).strip(),
                    "video_prompt_zh": str(item.get("video_prompt_zh", "") or frame_item.get("video_prompt_zh", "")).strip(),
                    "video_prompt_en": str(item.get("video_prompt_en", "") or frame_item.get("video_prompt_en", "")).strip(),
                    "frame_path_start": str(resolved_frame_path_start) if resolved_frame_path_start else "",
                    "frame_path_end": str(resolved_frame_path_end) if resolved_frame_path_end else "",
                    "frame_url_start": self._build_task_file_url(task_id=task_id, file_path=resolved_frame_path_start)
                    if resolved_frame_path_start
                    else "",
                    "frame_url_end": self._build_task_file_url(task_id=task_id, file_path=resolved_frame_path_end)
                    if resolved_frame_path_end
                    else "",
                }
            )
        return normalized_items

    def _resolve_module_d_segment_frame_paths(
        self,
        task_id: str,
        shot_id: str,
        segment_path: Path | None,
    ) -> tuple[Path | None, Path | None]:
        """
        功能说明：优先从模块D段视频对应的逐帧目录中定位真实首尾帧。
        参数说明：
        - task_id: 任务唯一标识。
        - shot_id: 模块D中的 shot 标识。
        - segment_path: 模块D记录的段视频路径。
        返回值：
        - tuple[Path | None, Path | None]: 真实首帧与尾帧路径；找不到时返回空。
        异常说明：无。
        边界条件：仅当段视频旁存在 `.shot_xxx_frames` 目录时命中，否则交给上层回退到模块C关键帧。
        """
        task_dir = self._resolve_task_dir(task_id=task_id)
        candidate_dirs: list[Path] = []
        if segment_path is not None:
            candidate_dirs.append((segment_path.parent / f".{str(shot_id).strip()}_frames").resolve())
        if str(shot_id).strip():
            candidate_dirs.append((task_dir / "artifacts" / "segments" / f".{str(shot_id).strip()}_frames").resolve())

        checked_dirs: set[Path] = set()
        for candidate_dir in candidate_dirs:
            if candidate_dir in checked_dirs:
                continue
            checked_dirs.add(candidate_dir)
            try:
                candidate_dir.relative_to(task_dir)
            except ValueError:
                continue
            if (not candidate_dir.exists()) or (not candidate_dir.is_dir()):
                continue
            frame_paths = sorted(
                [
                    file_path
                    for file_path in candidate_dir.iterdir()
                    if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                ],
                key=lambda file_path: file_path.name,
            )
            if not frame_paths:
                continue
            return frame_paths[0], frame_paths[-1]
        return None, None

    def _reattach_lyric_units_to_review_segments(
        self,
        lyric_units: list[dict[str, Any]],
        review_segment_units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        功能说明：按当前审阅时间线重挂歌词 segment_id，避免旧成片混入新 A/B 的 segment 标识。
        参数说明：
        - lyric_units: 原始歌词时间戳数组。
        - review_segment_units: 审阅页当前采用的 segment 时间线。
        返回值：
        - list[dict[str, Any]]: 已按审阅时间线修正 segment_id 的歌词数组。
        异常说明：无。
        边界条件：若歌词时刻找不到匹配 segment，则保留原始 segment_id 以避免信息丢失。
        """
        if not lyric_units or not review_segment_units:
            return lyric_units
        normalized_items: list[dict[str, Any]] = []
        for item in lyric_units:
            lyric_start_time = float(item.get("start_time", 0.0))
            matched_segment = self._find_review_segment_for_time(
                review_segment_units=review_segment_units,
                current_time=lyric_start_time,
            )
            normalized_item = dict(item)
            if matched_segment:
                normalized_item["segment_id"] = str(matched_segment.get("segment_id", "")).strip()
            normalized_items.append(normalized_item)
        return normalized_items

    def _find_review_segment_for_time(
        self,
        review_segment_units: list[dict[str, Any]],
        current_time: float,
    ) -> dict[str, Any] | None:
        """
        功能说明：按左闭右开规则查找给定时刻对应的审阅 segment。
        参数说明：
        - review_segment_units: 审阅页当前采用的 segment 时间线。
        - current_time: 当前时刻（秒）。
        返回值：
        - dict[str, Any] | None: 命中时返回 segment，未命中返回 None。
        异常说明：无。
        边界条件：最后一个 segment 允许命中右端点，避免播放器落在结尾时失焦。
        """
        if (not review_segment_units) or (not isinstance(current_time, (int, float))):
            return None
        last_index = len(review_segment_units) - 1
        for index, item in enumerate(review_segment_units):
            try:
                start_time = float(item.get("start_time", 0.0))
                end_time = max(start_time, float(item.get("end_time", start_time)))
            except (TypeError, ValueError):
                continue
            is_last_item = index == last_index
            if current_time >= start_time and (current_time < end_time or (is_last_item and current_time <= end_time)):
                return item
        return None

    def _build_review_segment_id(
        self,
        explicit_segment_id: str,
        unit_index: int,
        shot_id: str,
        fallback_index: int,
    ) -> str:
        """
        功能说明：为审阅页生成稳定的 `seg_xxxx` 标识。
        参数说明：
        - explicit_segment_id: 产物中显式保存的 segment_id。
        - unit_index: 模块D单元索引。
        - shot_id: 模块D shot 标识。
        - fallback_index: 当索引缺失时使用的枚举索引。
        返回值：
        - str: 标准化后的 segment_id。
        异常说明：无。
        边界条件：优先沿用显式字段；缺失时按 unit_index / shot_id / 枚举索引兜底。
        """
        normalized_explicit_segment_id = str(explicit_segment_id).strip()
        if normalized_explicit_segment_id:
            return normalized_explicit_segment_id
        if unit_index >= 0:
            return f"seg_{unit_index + 1:04d}"
        matched = re.search(r"(\d+)", str(shot_id))
        if matched is not None:
            return f"seg_{int(matched.group(1)):04d}"
        return f"seg_{fallback_index + 1:04d}"

    def _load_module_b_segment_item_map(self, task_dir: Path, task_id: str) -> dict[str, dict[str, Any]]:
        """
        功能说明：按 segment_id 加载模块B产物，供审阅页读取 scene_desc 与 prompt 数据。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, dict[str, Any]]: segment_id 到模块B单元载荷的映射。
        异常说明：读取失败时返回空映射，不中断页面。
        边界条件：优先使用状态库单元记录；缺失时回退 `artifacts/module_b_units/*.json`。
        """
        item_map: dict[str, dict[str, Any]] = {}
        try:
            state_rows = self.state_store.list_module_b_done_shot_items(task_id=task_id)
        except Exception:  # noqa: BLE001
            state_rows = []
        for row in state_rows:
            segment_id = str(row.get("unit_id", "")).strip()
            if not segment_id:
                continue
            artifact_path = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(row.get("artifact_path", "")))
            if artifact_path is None:
                continue
            payload = self._load_json_file(artifact_path)
            normalized_item = self._normalize_module_b_segment_item(payload=payload, segment_id=segment_id)
            if normalized_item:
                item_map[segment_id] = normalized_item

        module_b_units_dir = task_dir / "artifacts" / "module_b_units"
        if not module_b_units_dir.exists():
            return item_map
        for payload_path in sorted(module_b_units_dir.glob("*.json")):
            segment_id = self._derive_segment_id_from_name(name=payload_path.stem)
            if (not segment_id) or (segment_id in item_map):
                continue
            payload = self._load_json_file(payload_path)
            normalized_item = self._normalize_module_b_segment_item(payload=payload, segment_id=segment_id)
            if normalized_item:
                item_map[segment_id] = normalized_item
        return item_map

    def _normalize_module_b_segment_item(self, payload: Any, segment_id: str) -> dict[str, Any]:
        """
        功能说明：把模块B单元载荷收敛成审阅页需要的最小字段集合。
        参数说明：
        - payload: 读取出的模块B JSON 对象。
        - segment_id: 当前单元对应的 segment_id。
        返回值：
        - dict[str, Any]: 规范化后的审阅页字段；失败时返回空字典。
        异常说明：无。
        边界条件：源载荷里未写 segment_id 时使用外部传入值补齐。
        """
        if not isinstance(payload, dict):
            return {}
        normalized_segment_id = str(payload.get("segment_id", "")).strip() or str(segment_id).strip()
        if not normalized_segment_id:
            return {}
        camera_plan_payload = payload.get("camera_plan", {})
        return {
            "segment_id": normalized_segment_id,
            "shot_id": str(payload.get("shot_id", "")).strip(),
            "scene_desc": str(payload.get("scene_desc", "")).strip(),
            "camera_plan": camera_plan_payload if isinstance(camera_plan_payload, dict) else {},
            "keyframe_prompt_start_zh": str(payload.get("keyframe_prompt_start_zh", "")).strip(),
            "keyframe_prompt_start_en": str(payload.get("keyframe_prompt_start_en", "")).strip(),
            "keyframe_prompt_end_zh": str(payload.get("keyframe_prompt_end_zh", "")).strip(),
            "keyframe_prompt_end_en": str(payload.get("keyframe_prompt_end_en", "")).strip(),
            "video_prompt_zh": str(payload.get("video_prompt_zh", "")).strip(),
            "video_prompt_en": str(payload.get("video_prompt_en", "")).strip(),
        }

    def _load_frame_item_map_by_shot_id(self, task_dir: Path, task_id: str) -> dict[str, dict[str, Any]]:
        """
        功能说明：按 shot_id 加载模块C关键帧产物，并补齐前端可直接访问的 URL。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - dict[str, dict[str, Any]]: shot_id -> frame_item 映射。
        异常说明：读取失败时返回空映射，不中断页面。
        边界条件：优先使用状态库 sidecar 聚合结果；缺失时回退 `module_c_output.json`。
        """
        item_map: dict[str, dict[str, Any]] = {}
        try:
            state_frame_items = self.state_store.list_module_c_done_frame_items(task_id=task_id)
        except Exception:  # noqa: BLE001
            state_frame_items = []
        for item in state_frame_items:
            normalized_item = self._normalize_frame_item(task_id=task_id, payload=item)
            shot_id = str(normalized_item.get("shot_id", "")).strip()
            if shot_id:
                item_map[shot_id] = normalized_item

        module_c_output_path = task_dir / "artifacts" / "module_c_output.json"
        payload = self._load_json_file(module_c_output_path)
        raw_frame_items = payload.get("frame_items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_frame_items, list):
            return item_map
        for item in raw_frame_items:
            normalized_item = self._normalize_frame_item(task_id=task_id, payload=item)
            shot_id = str(normalized_item.get("shot_id", "")).strip()
            if not shot_id:
                continue
            existing_item = item_map.get(shot_id)
            if existing_item is None:
                item_map[shot_id] = normalized_item
                continue
            for field_name in ("frame_path", "frame_path_start", "frame_path_end", "frame_url_start", "frame_url_end"):
                if (not str(existing_item.get(field_name, "")).strip()) and str(normalized_item.get(field_name, "")).strip():
                    existing_item[field_name] = normalized_item[field_name]
        return item_map

    def _normalize_frame_item(self, task_id: str, payload: Any) -> dict[str, Any]:
        """
        功能说明：把模块C关键帧记录规整成审阅页需要的路径与URL字段。
        参数说明：
        - task_id: 任务唯一标识。
        - payload: 模块C原始 frame_item 或状态库聚合结果。
        返回值：
        - dict[str, Any]: 规范化后的关键帧信息；失败时返回空字典。
        异常说明：无。
        边界条件：若仅存在单张 `frame_path`，则同时回填起始与结束关键帧。
        """
        if not isinstance(payload, dict):
            return {}
        shot_id = str(payload.get("shot_id", "")).strip()
        if not shot_id:
            return {}
        frame_path = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path", "")))
        frame_path_start = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path_start", "")))
        frame_path_end = self._resolve_task_artifact_path(task_id=task_id, raw_path=str(payload.get("frame_path_end", "")))
        normalized_frame_path_start = frame_path_start or frame_path
        normalized_frame_path_end = frame_path_end or frame_path or frame_path_start
        normalized_frame_path = frame_path or normalized_frame_path_start or normalized_frame_path_end
        return {
            "shot_id": shot_id,
            "frame_path": str(normalized_frame_path) if normalized_frame_path else "",
            "frame_path_start": str(normalized_frame_path_start) if normalized_frame_path_start else "",
            "frame_path_end": str(normalized_frame_path_end) if normalized_frame_path_end else "",
            "frame_url_start": self._build_task_file_url(task_id=task_id, file_path=normalized_frame_path_start)
            if normalized_frame_path_start
            else "",
            "frame_url_end": self._build_task_file_url(task_id=task_id, file_path=normalized_frame_path_end)
            if normalized_frame_path_end
            else "",
        }

    def _resolve_task_artifact_path(self, task_id: str, raw_path: str) -> Path | None:
        """
        功能说明：把产物中记录的绝对或相对路径回映射到当前任务目录下的真实文件。
        参数说明：
        - task_id: 任务唯一标识。
        - raw_path: 产物JSON里记录的原始路径字符串。
        返回值：
        - Path | None: 找到可用文件时返回本地任务目录下路径，否则返回 None。
        异常说明：无。
        边界条件：兼容 `/root/data/runs/...` 这类运行时路径与本地 runs 目录映射。
        """
        normalized_raw_path = str(raw_path).strip()
        if not normalized_raw_path:
            return None
        task_dir = self._resolve_task_dir(task_id=task_id)
        raw_candidate = Path(normalized_raw_path)
        candidate_paths: list[Path] = []
        if not raw_candidate.is_absolute():
            candidate_paths.append((task_dir / raw_candidate).resolve())
        else:
            candidate_paths.append(raw_candidate.resolve())

        raw_parts = [str(part) for part in raw_candidate.parts if str(part).strip()]
        normalized_task_id = str(task_id).strip()
        if normalized_task_id in raw_parts:
            task_index = raw_parts.index(normalized_task_id)
            candidate_paths.append(task_dir.joinpath(*raw_parts[task_index + 1 :]).resolve())
        if "artifacts" in raw_parts:
            artifacts_index = max(index for index, part_text in enumerate(raw_parts) if part_text == "artifacts")
            candidate_paths.append(task_dir.joinpath("artifacts", *raw_parts[artifacts_index + 1 :]).resolve())

        for candidate_path in candidate_paths:
            try:
                candidate_path.relative_to(task_dir)
            except ValueError:
                continue
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path
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

    def _derive_segment_id_from_name(self, name: str) -> str:
        """
        功能说明：从模块B单元文件名中提取标准 segment_id。
        参数说明：
        - name: 不含扩展名的文件名。
        返回值：
        - str: 成功时返回 `seg_xxxx`，失败时返回空字符串。
        异常说明：无。
        边界条件：兼容 `seg_0001` 与 `segment_001_seg_0001` 两类历史命名。
        """
        matched = re.search(r"(seg_\d+)", str(name))
        if matched is None:
            return ""
        return str(matched.group(1)).strip()

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
