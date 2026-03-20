"""
文件用途：提供统一的日志初始化方法。
核心流程：按配置初始化 logging，并统一日志格式。
输入输出：输入日志级别，输出可复用 Logger。
依赖说明：依赖标准库 logging。
维护说明：日志内容必须保持中文，便于评审与排障。
"""

# 标准库：用于日志记录
import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    功能说明：初始化项目日志配置并返回根 Logger。
    参数说明：
    - level: 日志级别字符串，例如 INFO/DEBUG。
    返回值：
    - logging.Logger: 根日志对象。
    异常说明：无。
    边界条件：非法级别将由 logging 自动处理为默认级别。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("music_video_pipeline")
