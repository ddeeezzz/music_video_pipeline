"""
文件用途：提供统一的 Remotion 渲染排队队列，限制并发渲染数，防止服务器资源耗尽。
核心流程：submit() 获取信号量后执行渲染函数，完成后释放。
依赖说明：仅依赖标准库 threading。
"""

import threading
from collections.abc import Callable
from typing import Any


class RemotionQueue:
    """
    功能说明：全局共享的 Remotion 渲染队列，所有 Module D 重跑路径共用此队列。
    参数说明：
    - max_concurrent: 最大并发渲染数（默认 3）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：max_concurrent 至少为 1；submit() 是阻塞调用，调用者线程等待渲染完成。
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._semaphore = threading.Semaphore(max(1, max_concurrent))
        self._active = 0
        self._waiting = 0
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        """当前正在渲染的数量。"""
        return self._active

    @property
    def waiting_count(self) -> int:
        """当前在队列中等待的数量。"""
        return self._waiting

    def submit(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """
        功能说明：提交一个渲染任务到队列。阻塞调用，直到获取到执行槽位并执行完成。
        参数说明：
        - func: 要执行的渲染函数（如 render_template_segment）。
        - args/kwargs: 传递给 func 的参数。
        返回值：
        - Any: func 的返回值。
        异常说明：
        - 透传 func 可能抛出的任何异常。
        边界条件：调用者线程会被阻塞，最多等待 max_concurrent 个前面的任务完成。
        """
        with self._lock:
            self._waiting += 1
        try:
            self._semaphore.acquire()
            with self._lock:
                self._waiting -= 1
                self._active += 1
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()
