"""
文件用途：验证任务监督服务的HTTP与WebSocket行为。
核心流程：启动服务、访问页面、接收快照并验证终态自动停止。
输入输出：输入任务状态记录，输出监控服务行为断言。
依赖说明：依赖 urllib/asyncio 与项目内 monitoring/state_store。
维护说明：服务URL与推送协议变更时需同步更新本测试。
"""

# 标准库：用于异步运行WebSocket客户端
import asyncio
# 标准库：用于JSON解析
import json
# 标准库：用于日志对象
import logging
# 标准库：用于占用测试端口
import socket
# 标准库：用于线程执行异步客户端
import threading
# 标准库：用于时间轮询
import time
# 标准库：用于HTTP请求
from urllib.request import urlopen
# 标准库：用于URL解析
from urllib.parse import urlencode, urlparse
# 标准库：用于HTTP错误断言
from urllib.error import HTTPError
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于轻量命名空间对象
from types import SimpleNamespace
# 第三方库：测试框架
import pytest

# 项目内模块：任务监督服务
from music_video_pipeline.monitoring.server import TaskMonitorService
# 项目内模块：模块A联网歌词状态持久化
from music_video_pipeline.modules.module_a_v2.network_lyrics_state import write_module_a_network_lyrics_state
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def _seed_frontend_build(frontend_build_dir: Path) -> None:
    """
    功能说明：为监督服务测试构造最小可用的前端构建产物。
    参数说明：
    - frontend_build_dir: 前端构建目录。
    返回值：无。
    异常说明：无。
    边界条件：仅用于测试静态入口与静态资源路由，不追求真实打包结果。
    """
    assets_dir = frontend_build_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (frontend_build_dir / "index.html").write_text(
        (
            "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<title>MVPL Web 工作台</title></head><body>"
            "<div id=\"root\">MVPL Web 工作台</div>"
            "<script type=\"module\" src=\"/app/assets/main.js\"></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (assets_dir / "main.js").write_text("console.info('[Web前端] 测试构建产物已加载');", encoding="utf-8")


def _reserve_consecutive_ports(host: str = "127.0.0.1") -> tuple[socket.socket, socket.socket, int]:
    """
    功能说明：预留一对连续端口，便于测试“首个端口被占用时自动顺延”。
    参数说明：
    - host: 监听地址。
    返回值：
    - tuple[socket.socket, socket.socket, int]: (已占用端口socket, 下一端口socket, 起始端口号)。
    异常说明：
    - AssertionError: 扫描范围内未找到可用连续端口时抛出。
    边界条件：仅用于测试，返回的第二个 socket 需由调用方自行关闭释放。
    """
    for start_port in range(47050, 47250):
        first_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            first_socket.bind((host, start_port))
            first_socket.listen(1)
            second_socket.bind((host, start_port + 1))
            second_socket.listen(1)
            return first_socket, second_socket, start_port
        except OSError:
            first_socket.close()
            second_socket.close()
            continue
    raise AssertionError("未找到可用于测试的连续空闲端口对。")


def test_task_monitor_service_should_serve_page_and_push_snapshot(tmp_path: Path) -> None:
    """
    功能说明：验证监督服务可返回HTML并推送WebSocket快照。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：任务未结束时服务保持运行。
    """
    websockets = pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_001"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_serve_page_and_push_snapshot")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    try:
        html_text = urlopen(service.monitor_url, timeout=3).read().decode("utf-8")
        assert "MVPL Web 工作台" in html_text
        monitor_parsed = urlparse(service.monitor_url)
        snapshot_url = f"http://{monitor_parsed.netloc}/snapshot?task_id={task_id}"
        snapshot_payload = json.loads(urlopen(snapshot_url, timeout=3).read().decode("utf-8"))
        assert snapshot_payload["task_id"] == task_id
        assert snapshot_payload["task_status"] == "running"

        async def _recv_one_snapshot() -> dict:
            async with websockets.connect(service.websocket_url_for(task_id=task_id)) as websocket:
                payload = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                return json.loads(str(payload))

        snapshot = asyncio.run(_recv_one_snapshot())
        assert snapshot["task_id"] == task_id
        assert snapshot["task_status"] == "running"
        assert "module_overview" in snapshot
        assert "bcd_chains" in snapshot

        js_text = urlopen(f"http://{monitor_parsed.netloc}/app/assets/main.js", timeout=3).read().decode("utf-8")
        assert "测试构建产物已加载" in js_text
    finally:
        service.stop()


def test_task_monitor_service_should_return_503_when_frontend_build_missing(tmp_path: Path) -> None:
    """
    功能说明：验证正式前端构建产物缺失时会返回明确的 503 页面，而不是回退旧监督页。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：只验证 HTML 入口缺失场景。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_missing_build"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)

    logger = logging.getLogger("test_task_monitor_service_should_return_503_when_frontend_build_missing")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=tmp_path / "missing_frontend_build",
    )
    service.start()
    try:
        try:
            urlopen(service.monitor_url, timeout=3)
            assert False, "预期应返回 HTTP 503"
        except Exception as error:  # noqa: BLE001
            message = str(error)
            assert "503" in message
    finally:
        service.stop()


def test_task_monitor_service_should_open_task_list_without_default_task(tmp_path: Path) -> None:
    """
    功能说明：验证监督服务在未预选 task_id 时会直接打开任务列表页。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：状态库为空时也应允许启动。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "pipeline_state.sqlite3")
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    logger = logging.getLogger("test_task_monitor_service_should_open_task_list_without_default_task")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id="",
        logger=logger,
        tick_seconds=0.2,
        auto_stop_on_terminal=False,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    try:
        assert urlparse(service.monitor_url).path == "/tasks"
        html_text = urlopen(service.monitor_url, timeout=3).read().decode("utf-8")
        assert "MVPL Web 工作台" in html_text
        task_list_payload = json.loads(urlopen(f"http://{urlparse(service.monitor_url).netloc}/api/tasks", timeout=3).read().decode("utf-8"))
        assert task_list_payload["ok"] is True
        assert task_list_payload["current_task_id"] == ""
        assert task_list_payload["tasks"] == []
    finally:
        service.stop()


def test_task_monitor_service_should_remap_foreign_audio_path_for_search_lyrics_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能说明：验证模块 A 联网歌词检索会把旧外机音频路径重映射到当前工作区后再执行。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证路径解析与请求处理，不发起真实联网查询。
    """
    workspace_root = tmp_path / "workspace_monitor_audio"
    resources_dir = workspace_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    local_audio_path = (resources_dir / "jieranduhuo01.mp3").resolve()
    local_audio_path.write_bytes(b"fake-audio")

    config_path = workspace_root / "configs" / "music_windows_4060" / "jieranduhuo.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"paths": {"default_audio_path": "resources/jieranduhuo01.mp3"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    app_config = SimpleNamespace(
        paths=SimpleNamespace(default_audio_path="resources/jieranduhuo01.mp3"),
        ffmpeg=SimpleNamespace(ffprobe_bin="ffprobe"),
        module_a=SimpleNamespace(fpcalc_bin="fpcalc", acoustid_api_key_file=""),
    )

    runs_dir = workspace_root / "runs"
    task_id = "jieranduhuo01"
    state_store = StateStore(db_path=runs_dir / "pipeline_state.sqlite3")
    state_store.init_task(
        task_id=task_id,
        audio_path="\\root\\data\\t1\\resources\\jieranduhuo.mp3",
        config_path=str(config_path),
    )

    logger = logging.getLogger("test_task_monitor_service_should_remap_foreign_audio_path_for_search_lyrics_request")
    logger.setLevel(logging.INFO)
    frontend_build_dir = workspace_root / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        app_config=app_config,
        auto_stop_on_terminal=False,
        frontend_build_dir=frontend_build_dir,
    )

    captured: dict[str, Path] = {}

    def _fake_probe_audio_duration(*, audio_path, ffprobe_bin, logger):  # noqa: ANN001
        _ = (ffprobe_bin, logger)
        captured["probe_audio_path"] = audio_path
        return 123.45

    def _fake_search_synced_lrc_candidates(**kwargs):  # noqa: ANN003
        captured["search_audio_path"] = kwargs["audio_path"]
        return {
            "status": "ok",
            "metadata_trace": {
                "embedded_status": "missing",
                "embedded_source": "embedded_tags",
                "embedded_artist": "",
                "embedded_title": "",
                "embedded_album": "",
                "embedded_error": "artist/title 不完整",
                "fingerprint_status": "ok",
                "fingerprint_error": "",
                "acoustid_status": "ok",
                "matched_artist": "Artist",
                "matched_title": "Title",
                "matched_score": 0.98,
                "matched_error": "",
            },
            "candidates": [
                {
                    "candidate_id": "cand-001",
                    "artist": "Artist",
                    "title": "Title",
                    "score": 0.98,
                    "provider": "lrclib",
                    "provider_id": "lrclib-1",
                    "preview_lines": ["line-1"],
                }
            ],
            "fingerprint_result": {"status": "ok"},
            "acoustid_result": {"status": "ok"},
        }

    monkeypatch.setattr("music_video_pipeline.monitoring.server.probe_audio_duration", _fake_probe_audio_duration)
    monkeypatch.setattr(
        "music_video_pipeline.monitoring.server.search_synced_lrc_candidates",
        _fake_search_synced_lrc_candidates,
    )

    payload, status = service._handle_module_a_search_lyrics_request(
        urlparse(f"http://127.0.0.1:45706/api/module-a/search-lyrics?task_id={task_id}")
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["metadata_trace"]["embedded_status"] == "missing"
    assert payload["metadata_trace"]["matched_title"] == "Title"
    assert captured["probe_audio_path"] == local_audio_path
    assert captured["search_audio_path"] == local_audio_path
    task_record = state_store.get_task(task_id=task_id)
    assert task_record is not None
    assert task_record["audio_path"] == str(local_audio_path)


def test_task_monitor_service_should_redirect_root_path_to_task_list(tmp_path: Path) -> None:
    """
    功能说明：验证直接访问服务根路径时，总是跳转到通用任务列表页。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使服务携带默认 task_id，也不应把 `/` 重定向到单任务页。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "pipeline_state.sqlite3")
    task_id = "task_monitor_root_redirect"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    logger = logging.getLogger("test_task_monitor_service_should_redirect_root_path_to_task_list")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        auto_stop_on_terminal=False,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    try:
        base_url = f"http://{urlparse(service.monitor_url).netloc}"
        response = urlopen(f"{base_url}/", timeout=3)
        assert urlparse(response.geturl()).path == "/tasks"
        assert "MVPL Web 工作台" in response.read().decode("utf-8")
    finally:
        service.stop()


def test_task_monitor_service_should_return_cached_candidate_lyrics_detail(tmp_path: Path) -> None:
    """
    功能说明：验证模块 A 可按 candidate_id 返回缓存候选的完整歌词正文。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证已缓存候选读取，不触发新的联网检索。
    """
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_candidate_lyrics"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    logger = logging.getLogger("test_task_monitor_service_should_return_cached_candidate_lyrics_detail")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=frontend_build_dir,
    )
    task_dir = service._resolve_task_dir(task_id=task_id)
    write_module_a_network_lyrics_state(
        artifacts_dir=task_dir / "artifacts",
        payload={
            "candidates": [
                {
                    "candidate_id": "qq_music_001",
                    "artist": "ABbbb君",
                    "title": "子然妒火",
                    "provider": "qq_music",
                    "provider_id": "songmid_001",
                    "score": 0.0,
                    "preview_lines": ["[00:00.00]子然妒火"],
                    "preview_text": "[00:00.00]子然妒火",
                    "synced_lyrics": "[00:00.00]子然妒火\n[00:03.50]下一句",
                    "word_timed_lyrics": "[00:00.00]<00:00.00>子然<00:01.50><00:01.50>妒火<00:03.50>",
                    "translated_lyrics": "燃起嫉妒的火",
                    "romanized_lyrics": "Jieran du huo",
                }
            ]
        },
    )

    service.start()
    try:
        response = json.loads(
            urlopen(
                f"http://{urlparse(service.monitor_url).netloc}/api/module-a/candidate-lyrics?"
                f"{urlencode({'task_id': task_id, 'candidate_id': 'qq_music_001'})}",
                timeout=3,
            )
            .read()
            .decode("utf-8")
        )
        assert response["ok"] is True
        assert response["task_id"] == task_id
        assert response["candidate"]["candidate_id"] == "qq_music_001"
        assert response["synced_lyrics"] == "[00:00.00]子然妒火\n[00:03.50]下一句"
        assert response["word_timed_lyrics"] == "[00:00.00]<00:00.00>子然<00:01.50><00:01.50>妒火<00:03.50>"
        assert response["translated_lyrics"] == "燃起嫉妒的火"
        assert response["romanized_lyrics"] == "Jieran du huo"
    finally:
        service.stop()


def test_task_monitor_service_should_fallback_to_next_port_when_requested_port_is_occupied(tmp_path: Path) -> None:
    """
    功能说明：验证监督服务起始端口被占用时会自动顺延到下一个端口。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证单次顺延场景。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_port_fallback"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    occupied_socket, reserved_next_socket, requested_port = _reserve_consecutive_ports()
    reserved_next_socket.close()

    logger = logging.getLogger("test_task_monitor_service_should_fallback_to_next_port_when_requested_port_is_occupied")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        port=requested_port,
        tick_seconds=0.2,
        auto_stop_on_terminal=False,
        frontend_build_dir=frontend_build_dir,
    )
    try:
        service.start()
        parsed = urlparse(service.monitor_url)
        assert parsed.port == requested_port + 1
        html_text = urlopen(service.monitor_url, timeout=3).read().decode("utf-8")
        assert "MVPL Web 工作台" in html_text
    finally:
        service.stop()
        occupied_socket.close()


def test_task_monitor_service_should_not_serve_legacy_monitor_routes(tmp_path: Path) -> None:
    """
    功能说明：验证旧 `/web` 与 `/task-monitor` 页面入口已移除，不再继续兼容。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验旧 HTML 入口已不可访问，不影响新 `/tasks/...` 路由。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_legacy_removed"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    logger = logging.getLogger("test_task_monitor_service_should_not_serve_legacy_monitor_routes")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    try:
        monitor_parsed = urlparse(service.monitor_url)
        base_url = f"http://{monitor_parsed.netloc}"
        for legacy_path in ("/web", f"/web?task_id={task_id}", "/task-monitor", f"/task-monitor?task_id={task_id}"):
            try:
                urlopen(f"{base_url}{legacy_path}", timeout=3)
                assert False, f"预期旧入口应返回 404：{legacy_path}"
            except HTTPError as error:
                assert error.code == 404
    finally:
        service.stop()


def test_task_monitor_service_should_stop_when_task_finished(tmp_path: Path) -> None:
    """
    功能说明：验证任务进入终态后监督服务会自动停止。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：自动停止后再次 stop 应保持幂等。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_002"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_stop_when_task_finished")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    state_store.update_task_status(task_id=task_id, status="done", output_video_path="final.mp4")

    deadline = time.time() + 5.0
    while service.is_running and time.time() < deadline:
        time.sleep(0.1)

    assert service.is_running is False
    service.stop()


def test_task_monitor_service_should_wait_for_browser_close_after_terminal(tmp_path: Path) -> None:
    """
    功能说明：验证任务终态后若仍有WS连接，服务会等待连接关闭再停止。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证单连接场景。
    """
    websockets = pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "state.sqlite3")
    task_id = "task_monitor_server_003"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)
    state_store.update_task_status(task_id=task_id, status="running")

    logger = logging.getLogger("test_task_monitor_service_should_wait_for_browser_close_after_terminal")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()

    ws_opened = threading.Event()
    ws_release = threading.Event()

    def _hold_websocket() -> None:
        async def _run() -> None:
            async with websockets.connect(service.websocket_url_for(task_id=task_id)) as websocket:
                _ = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                ws_opened.set()
                while not ws_release.is_set():
                    await asyncio.sleep(0.1)

        asyncio.run(_run())

    ws_thread = threading.Thread(target=_hold_websocket, daemon=True)
    ws_thread.start()
    assert ws_opened.wait(timeout=3.0)

    state_store.update_task_status(task_id=task_id, status="done", output_video_path="final.mp4")
    time.sleep(0.8)
    assert service.is_running is True

    ws_release.set()
    ws_thread.join(timeout=3.0)

    deadline = time.time() + 5.0
    while service.is_running and time.time() < deadline:
        time.sleep(0.1)
    assert service.is_running is False


def test_task_monitor_service_should_support_home_task_management_apis(tmp_path: Path) -> None:
    """
    功能说明：验证主页相关任务列表、详情、新建、复制与改名接口可正常工作。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：接口只创建或迁移状态记录，不自动触发运行。
    """
    pytest.importorskip("websockets")
    state_store = StateStore(db_path=tmp_path / "pipeline_state.sqlite3")
    task_id = "task_home_api_001"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)

    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    state_store.set_module_status(task_id=task_id, module_name="A", status="done", artifact_path=str(task_dir / "artifacts" / "module_a_output.json"))
    state_store.update_task_status(task_id=task_id, status="running")
    frontend_build_dir = tmp_path / "frontend_build"
    _seed_frontend_build(frontend_build_dir=frontend_build_dir)

    logger = logging.getLogger("test_task_monitor_service_should_support_home_task_management_apis")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        auto_stop_on_terminal=False,
        frontend_build_dir=frontend_build_dir,
    )
    service.start()
    try:
        monitor_parsed = urlparse(service.monitor_url)
        base_url = f"http://{monitor_parsed.netloc}"

        task_list_payload = json.loads(urlopen(f"{base_url}/api/tasks", timeout=3).read().decode("utf-8"))
        assert task_list_payload["ok"] is True
        assert task_list_payload["tasks"][0]["task_id"] == task_id
        assert task_list_payload["tasks"][0]["module_status"]["A"] == "done"
        assert task_list_payload["tasks"][0]["output_video_path"] == ""

        task_detail_payload = json.loads(
            urlopen(f"{base_url}/api/task?task_id={task_id}", timeout=3).read().decode("utf-8")
        )
        assert task_detail_payload["ok"] is True
        assert task_detail_payload["task"]["task_id"] == task_id
        assert task_detail_payload["task"]["status"] == "running"

        create_query = urlencode(
            {
                "task_id": "task_home_api_002",
                "audio_path": str(tmp_path / "audio_002.mp3"),
                "config_path": str(tmp_path / "config_002.json"),
            }
        )
        create_payload = json.loads(urlopen(f"{base_url}/api/task/create?{create_query}", timeout=3).read().decode("utf-8"))
        assert create_payload["ok"] is True
        assert create_payload["task_id"] == "task_home_api_002"
        created_task = state_store.get_task(task_id="task_home_api_002")
        assert created_task is not None
        assert created_task["status"] == "pending"

        copy_query = urlencode(
            {
                "source_task_id": task_id,
                "new_task_id": "task_home_api_003",
                "audio_path": str(tmp_path / "audio_003.mp3"),
                "config_path": str(tmp_path / "config_003.json"),
            }
        )
        copy_payload = json.loads(urlopen(f"{base_url}/api/task/copy?{copy_query}", timeout=3).read().decode("utf-8"))
        assert copy_payload["ok"] is True
        copied_task = state_store.get_task(task_id="task_home_api_003")
        assert copied_task is not None
        assert copied_task["audio_path"] == str(tmp_path / "audio_003.mp3")

        rename_query = urlencode({"old_task_id": task_id, "new_task_id": "task_home_api_001_renamed"})
        rename_payload = json.loads(urlopen(f"{base_url}/api/task/rename?{rename_query}", timeout=3).read().decode("utf-8"))
        assert rename_payload["ok"] is True
        assert state_store.get_task(task_id=task_id) is None
        renamed_task = state_store.get_task(task_id="task_home_api_001_renamed")
        assert renamed_task is not None
        assert (tmp_path / "task_home_api_001_renamed").exists()
    finally:
        service.stop()


def test_task_monitor_service_should_build_web_payload_from_module_b_and_c_without_module_d(tmp_path: Path) -> None:
    """
    功能说明：验证审阅页数据可直接由模块A时间轴、模块B单元与模块C关键帧拼装，不依赖模块D摘要。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模块B单元文件允许缺失显式 segment_id，需从文件名补齐。
    """
    state_store = StateStore(db_path=tmp_path / "pipeline_state.sqlite3")
    task_id = "task_web_payload_001"
    _seed_task(state_store=state_store, task_id=task_id, workspace=tmp_path)

    task_dir = tmp_path / task_id
    artifacts_dir = task_dir / "artifacts"
    module_b_units_dir = artifacts_dir / "module_b_units"
    frames_dir = artifacts_dir / "frames"
    module_b_units_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "module_a_output.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "big_segment_id": "big_001",
                        "start_time": 0.0,
                        "end_time": 2.5,
                        "label": "verse",
                        "role": "lyric",
                    },
                    {
                        "segment_id": "seg_0002",
                        "big_segment_id": "big_001",
                        "start_time": 2.5,
                        "end_time": 5.0,
                        "label": "verse",
                        "role": "inst",
                    },
                ],
                "lyric_units": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (module_b_units_dir / "seg_0001.json").write_text(
        json.dumps(
            {
                "shot_id": "shot_001",
                "scene_desc": "黑白空巷里，少女站在巷口望向深处。",
                "keyframe_prompt_start_zh": "黑白空巷，少女站在巷口，强透视构图",
                "keyframe_prompt_start_en": "monochrome alley, girl at entrance, strong perspective",
                "keyframe_prompt_end_zh": "黑白空巷深处，少女身影被阴影吞没",
                "keyframe_prompt_end_en": "deep monochrome alley, girl's silhouette swallowed by shadow",
                "video_prompt_zh": "镜头沿着黑白巷道向深处推进，少女站在巷口。",
                "video_prompt_en": "camera pushes through monochrome alley, girl stands at entrance",
                "camera_plan": {"mode": "zoom", "direction": "center"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    frame_path = frames_dir / "frame_001.png"
    frame_path.write_bytes(b"fake-png")
    (artifacts_dir / "module_c_output.json").write_text(
        json.dumps(
            {
                "frame_items": [
                    {
                        "shot_id": "shot_001",
                        "frame_path": f"/root/data/runs/{task_id}/artifacts/frames/frame_001.png",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger = logging.getLogger("test_task_monitor_service_should_build_web_payload_from_module_b_and_c_without_module_d")
    logger.setLevel(logging.INFO)
    service = TaskMonitorService(
        state_store=state_store,
        task_id=task_id,
        logger=logger,
        tick_seconds=0.2,
        auto_stop_on_terminal=False,
    )

    payload = service._build_web_payload(task_id=task_id)
    assert payload["task_id"] == task_id
    assert len(payload["segment_units"]) == 2
    assert payload["segment_units"][0]["segment_id"] == "seg_0001"
    assert payload["segment_units"][0]["scene_desc"] == "黑白空巷里，少女站在巷口望向深处。"
    assert payload["segment_units"][0]["shot_id"] == "shot_001"
    assert payload["segment_units"][0]["keyframe_prompt_start_zh"] == "黑白空巷，少女站在巷口，强透视构图"
    assert payload["segment_units"][0]["frame_url_start"] == f"/task/{task_id}/artifacts/frames/frame_001.png"
    assert payload["segment_units"][0]["frame_url_end"] == f"/task/{task_id}/artifacts/frames/frame_001.png"
    assert payload["segment_units"][1]["segment_id"] == "seg_0002"
    assert payload["segment_units"][1]["scene_desc"] == ""
    assert payload["segment_units"][1]["shot_id"] == ""


def _seed_task(state_store: StateStore, task_id: str, workspace: Path) -> None:
    """
    功能说明：写入测试任务初始化记录。
    参数说明：
    - state_store: 状态库对象。
    - task_id: 任务标识。
    - workspace: 临时目录路径。
    返回值：无。
    异常说明：无。
    边界条件：仅用于监督服务测试，不要求完整模块产物。
    """
    audio_path = workspace / f"{task_id}.mp3"
    config_path = workspace / f"{task_id}.json"
    audio_path.write_bytes(b"fake")
    config_path.write_text("{}", encoding="utf-8")
    state_store.init_task(task_id=task_id, audio_path=str(audio_path), config_path=str(config_path))
