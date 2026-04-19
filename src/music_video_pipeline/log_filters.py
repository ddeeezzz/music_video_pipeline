"""
文件用途：集中管理运行期日志与 warning 的精准过滤规则。
核心流程：在 CLI 启动早期安装过滤器，静默已知噪声输出。
输入输出：输入无，输出无（通过 logging/warnings 全局状态生效）。
依赖说明：依赖标准库 logging 与 warnings。
维护说明：只添加“已确认无害且高频”的规则，避免误伤有效告警。
"""

# 标准库：用于日志过滤器接口与 logger 管理。
import logging
# 标准库：用于 warning 过滤。
import warnings

# 常量：CLIPFeatureExtractor 废弃提示匹配正则。
CLIP_DEPRECATION_WARNING_PATTERN = ".*CLIPFeatureExtractor appears to have been deprecated.*"
# 常量：过滤器安装哨兵，挂在 root logger 上保证幂等。
INSTALL_SENTINEL_NAME = "_mvpl_runtime_noise_filters_installed"


class SuppressModelLoadNoiseFilter(logging.Filter):
    """
    功能说明：过滤模型加载阶段的已知噪声日志。
    参数说明：使用 logging.Filter.filter(record) 判定是否放行。
    返回值：
    - bool: False 表示拦截；True 表示放行。
    异常说明：无。
    边界条件：仅命中 LOAD REPORT / UNEXPECTED 关键字时拦截。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "LOAD REPORT" in message or "UNEXPECTED" in message:
            return False
        return True


def install_runtime_noise_filters() -> None:
    """
    功能说明：安装全局运行期噪声过滤规则。
    参数说明：无。
    返回值：无。
    异常说明：无。
    边界条件：重复调用时安全，最多安装一次 logger 过滤器。
    """
    warnings.filterwarnings(
        "ignore",
        message=CLIP_DEPRECATION_WARNING_PATTERN,
        category=FutureWarning,
    )

    root_logger = logging.getLogger()
    if getattr(root_logger, INSTALL_SENTINEL_NAME, False):
        return

    noise_filter = SuppressModelLoadNoiseFilter()
    logging.getLogger("transformers").addFilter(noise_filter)
    logging.getLogger("diffusers").addFilter(noise_filter)
    setattr(root_logger, INSTALL_SENTINEL_NAME, True)
