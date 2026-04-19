"""
文件用途：验证运行期日志过滤器的精准拦截与幂等安装行为。
核心流程：构造日志记录检查过滤结果，并验证重复安装不重复挂载 logger filter。
输入输出：输入为测试构造日志记录，输出为断言结果。
依赖说明：依赖 pytest 与项目内 log_filters 模块。
维护说明：若新增过滤规则，应补充对应命中/放行测试。
"""

# 标准库：用于日志记录对象与 logger 访问。
import logging

# 项目内模块：日志过滤器实现
from music_video_pipeline import log_filters
from music_video_pipeline.log_filters import INSTALL_SENTINEL_NAME, SuppressModelLoadNoiseFilter, install_runtime_noise_filters


def test_suppress_model_load_noise_filter_should_block_known_noise() -> None:
    """
    功能说明：验证过滤器仅拦截指定噪声关键字。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：分别覆盖 LOAD REPORT、UNEXPECTED 与普通 warning。
    """
    runtime_filter = SuppressModelLoadNoiseFilter()

    load_report_record = logging.LogRecord(
        name="diffusers",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="Some LOAD REPORT table row",
        args=(),
        exc_info=None,
    )
    unexpected_record = logging.LogRecord(
        name="transformers",
        level=logging.WARNING,
        pathname=__file__,
        lineno=11,
        msg="UNEXPECTED keys when loading checkpoint",
        args=(),
        exc_info=None,
    )
    normal_record = logging.LogRecord(
        name="transformers",
        level=logging.WARNING,
        pathname=__file__,
        lineno=12,
        msg="CUDA memory may be insufficient",
        args=(),
        exc_info=None,
    )

    assert runtime_filter.filter(load_report_record) is False
    assert runtime_filter.filter(unexpected_record) is False
    assert runtime_filter.filter(normal_record) is True


def test_install_runtime_noise_filters_should_be_idempotent(monkeypatch) -> None:
    """
    功能说明：验证过滤器安装过程可重复调用且不重复挂载 logger 过滤器。
    参数说明：
    - monkeypatch: pytest monkeypatch fixture。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：允许 warnings.filterwarnings 多次调用，但 logger filter 仅挂载一次。
    """
    root_logger = logging.getLogger()
    transformers_logger = logging.getLogger("transformers")
    diffusers_logger = logging.getLogger("diffusers")

    monkeypatch.setattr(root_logger, INSTALL_SENTINEL_NAME, False, raising=False)
    monkeypatch.setattr(transformers_logger, "filters", [])
    monkeypatch.setattr(diffusers_logger, "filters", [])

    warning_calls = {"count": 0}

    def _fake_filterwarnings(*_args, **_kwargs) -> None:
        warning_calls["count"] += 1

    monkeypatch.setattr(log_filters.warnings, "filterwarnings", _fake_filterwarnings)

    install_runtime_noise_filters()
    install_runtime_noise_filters()

    assert warning_calls["count"] == 2
    assert len(transformers_logger.filters) == 1
    assert len(diffusers_logger.filters) == 1
    assert getattr(root_logger, INSTALL_SENTINEL_NAME) is True
