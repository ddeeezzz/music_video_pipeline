"""
文件用途：构建远端资源索引并计算本地下载状态。
核心流程：读取远端目录 -> 解析目录行 -> 生成菜单项。
输入输出：输入资源类型和项目根路径，输出可交互选择的条目列表。
依赖说明：依赖标准库 pathlib/re 与 bypy_client。
维护说明：该模块不处理下载动作，仅负责索引与展示数据准备。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于正则解析 bypy 输出
import re

try:
    # 包内导入：bypy 客户端异常类型
    from .bypy_client import BypyClient, BypyClientError
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from bypy_client import BypyClient, BypyClientError


# 常量：系列展示顺序。
SERIES_DISPLAY_ORDER = ("15", "xl", "fl")
# 常量：支持的资源类型。
VALID_RESOURCE_TYPES = ("lora", "base_model")
# 常量：支持的基础模型格式层级。
VALID_BASE_MODEL_FORMATS = ("single", "diffusers")
# 常量：资源远端根目录映射。
REMOTE_ROOT_MAP = {
    "lora": "/lora",
    "base_model": "/base_model",
}
# 常量：资源本地根目录映射（相对项目根目录）。
LOCAL_ROOT_MAP = {
    "lora": "models/lora",
    "base_model": "models/base_model",
}
# 常量：bypy list 的目录行解析规则。
BYPY_DIR_LINE_PATTERN = re.compile(
    r"^D\s+(.+?)\s+\d+\s+\d{4}-\d{2}-\d{2},\s+\d{2}:\d{2}:\d{2}\s*$"
)
# 常量：bypy list 的文件行解析规则。
BYPY_FILE_LINE_PATTERN = re.compile(
    r"^F\s+(.+?)\s+(\d+)\s+\d{4}-\d{2}-\d{2},\s+\d{2}:\d{2}:\d{2}\s+[0-9a-zA-Z]+$"
)


def parse_remote_dirs(list_output: str) -> list[str]:
    """
    功能说明：解析 bypy list 输出中的目录名。
    参数说明：
    - list_output: bypy list 输出文本。
    返回值：
    - list[str]: 目录名称列表。
    异常说明：无。
    边界条件：仅解析格式合法的 D 行。
    """
    dir_names: list[str] = []
    for line in list_output.splitlines():
        text = line.strip()
        if not text.startswith("D "):
            continue
        matched = BYPY_DIR_LINE_PATTERN.match(text)
        if not matched:
            continue
        name = matched.group(1).strip()
        if name:
            dir_names.append(name)
    return dir_names


def parse_remote_files(list_output: str) -> list[dict[str, str | int]]:
    """
    功能说明：解析 bypy list 输出中的文件项。
    参数说明：
    - list_output: bypy list 输出文本。
    返回值：
    - list[dict[str, str | int]]: 文件数组，每项包含 name/size。
    异常说明：无。
    边界条件：仅解析格式合法的 F 行。
    """
    files: list[dict[str, str | int]] = []
    for line in list_output.splitlines():
        text = line.strip()
        if not text.startswith("F "):
            continue
        matched = BYPY_FILE_LINE_PATTERN.match(text)
        if not matched:
            continue
        files.append({"name": matched.group(1).strip(), "size": int(matched.group(2))})
    return files


def build_remote_options(resource_type: str, project_root: Path, client: BypyClient, logger) -> list[dict]:
    """
    功能说明：按资源类型构建远端目录菜单项。
    参数说明：
    - resource_type: lora 或 base_model。
    - project_root: 项目根目录。
    - client: bypy 客户端。
    - logger: 日志对象。
    返回值：
    - list[dict]: 菜单项数组，包含 index/series/name/remote_dir/downloaded。
    异常说明：
    - ValueError: 资源类型非法时抛出。
    - BypyClientError: 根目录不可访问时抛出。
    边界条件：某个系列目录读取失败时仅跳过该系列。
    """
    if resource_type not in VALID_RESOURCE_TYPES:
        raise ValueError(f"资源类型非法：{resource_type}")

    remote_root = REMOTE_ROOT_MAP[resource_type]
    local_root = (project_root / LOCAL_ROOT_MAP[resource_type]).resolve()

    # 先探测根目录，确保远端结构存在。
    _ = client.list_remote(remote_root)

    options: list[dict] = []
    index_counter = 1
    for series in SERIES_DISPLAY_ORDER:
        series_remote_dir = f"{remote_root}/{series}"
        try:
            output = client.list_remote(series_remote_dir)
        except BypyClientError as error:
            logger.warning("读取远端目录失败，已跳过：%s，错误=%s", series_remote_dir, error)
            continue

        # LoRA 使用 /lora/{series}/{name} 叶子结构，保持原逻辑。
        if resource_type == "lora":
            dir_names = sorted(parse_remote_dirs(output), key=lambda item: item.lower())
            for dir_name in dir_names:
                local_dir = (local_root / series / dir_name).resolve()
                options.append(
                    {
                        "index": index_counter,
                        "series": series,
                        "name": dir_name,
                        "remote_dir": f"{series_remote_dir}/{dir_name}",
                        "downloaded": local_dir.exists(),
                    }
                )
                index_counter += 1
            continue

        # BaseModel 使用 /base_model/{series}/{single|diffusers}/{name} 结构。
        format_dirs = [item for item in parse_remote_dirs(output) if item in VALID_BASE_MODEL_FORMATS]
        for model_format in sorted(format_dirs):
            format_remote_dir = f"{series_remote_dir}/{model_format}"
            try:
                format_output = client.list_remote(format_remote_dir)
            except BypyClientError as error:
                logger.warning("读取远端目录失败，已跳过：%s，错误=%s", format_remote_dir, error)
                continue

            model_names = sorted(parse_remote_dirs(format_output), key=lambda item: item.lower())
            for model_name in model_names:
                local_dir = (local_root / series / model_format / model_name).resolve()
                options.append(
                    {
                        "index": index_counter,
                        "series": series,
                        "format": model_format,
                        "name": model_name,
                        "display_name": f"{model_format}/{model_name}",
                        "remote_dir": f"{format_remote_dir}/{model_name}",
                        "downloaded": local_dir.exists(),
                    }
                )
                index_counter += 1

    return options
