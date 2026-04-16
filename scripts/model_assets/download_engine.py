"""
文件用途：提供模型文件与 Hugging Face 仓库的统一下载能力。
核心流程：解析下载目标 -> 按后端顺序执行并重试 -> 校验下载结果有效性。
输入输出：输入下载参数与日志对象，输出本地文件或目录下载结果。
依赖说明：依赖标准库 pathlib/subprocess/shutil/os/time/urllib 与 requests。
维护说明：该模块只负责下载，不处理注册表或绑定关系写入。
"""

# 标准库：用于环境变量读取与传递
import os
# 标准库：用于可执行文件探测
import shutil
# 标准库：用于子进程调用外部命令
import subprocess
# 标准库：用于重试等待与进度节流
import time
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于 URL 路径解析
from urllib.parse import urlparse
# 标准库：用于类型标注
from typing import Callable

# 第三方库：用于 HTTP 流式下载回退。
import requests


# 常量：默认镜像站地址（当 HF_ENDPOINT 未设置时自动注入）。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
# 常量：默认最大重试次数。
DEFAULT_MAX_RETRIES = 3
# 常量：默认重试等待基准秒数（指数退避）。
DEFAULT_RETRY_WAIT_SECONDS = 2.0
# 常量：默认网络超时秒数。
DEFAULT_TIMEOUT_SECONDS = 60
# 常量：requests 下载分块大小（字节）。
REQUEST_CHUNK_SIZE = 1024 * 1024


class DownloadEngineError(RuntimeError):
    """
    功能说明：统一封装下载引擎异常。
    参数说明：继承 RuntimeError，直接传入中文错误消息。
    返回值：不适用。
    异常说明：不适用。
    边界条件：用于上层流程统一捕获与展示。
    """


def infer_filename_from_url(url: str) -> str:
    """
    功能说明：根据 URL 推断下载文件名。
    参数说明：
    - url: 下载地址。
    返回值：
    - str: 推断出的文件名；无法推断时返回 downloaded_asset.bin。
    异常说明：无。
    边界条件：会忽略 query 与 fragment 部分。
    """
    parsed = urlparse(str(url).strip())
    filename = Path(parsed.path).name.strip()
    return filename or "downloaded_asset.bin"


def infer_repo_name(repo_id: str) -> str:
    """
    功能说明：根据 Hugging Face repo_id 推断本地目录名。
    参数说明：
    - repo_id: 仓库标识，如 stabilityai/stable-diffusion-xl-base-1.0。
    返回值：
    - str: 目录名；无法推断时返回 unknown-repo。
    异常说明：无。
    边界条件：按最后一个 / 之后片段推断。
    """
    text = str(repo_id).strip()
    if "/" in text:
        tail = text.split("/")[-1].strip()
        return tail or "unknown-repo"
    return text or "unknown-repo"


def ensure_non_empty_file(file_path: Path) -> None:
    """
    功能说明：校验目标文件存在且非空。
    参数说明：
    - file_path: 目标文件路径。
    返回值：无。
    异常说明：
    - DownloadEngineError: 文件不存在或大小为 0 时抛出。
    边界条件：用于下载完成后的兜底校验。
    """
    if (not file_path.exists()) or (not file_path.is_file()) or file_path.stat().st_size <= 0:
        raise DownloadEngineError(f"下载结果校验失败：文件不存在或为空，path={file_path}")


def has_any_file(target_dir: Path) -> bool:
    """
    功能说明：判断目录中是否至少包含一个文件。
    参数说明：
    - target_dir: 目标目录。
    返回值：
    - bool: 存在文件返回 True。
    异常说明：无。
    边界条件：仅统计文件，不统计空目录。
    """
    if not target_dir.exists():
        return False
    for path in target_dir.rglob("*"):
        if path.is_file():
            return True
    return False


def run_with_retry(
    runner: Callable[[], None],
    action_name: str,
    max_retries: int,
    retry_wait_seconds: float,
    logger,
) -> None:
    """
    功能说明：执行动作并在失败时重试。
    参数说明：
    - runner: 无参执行函数。
    - action_name: 动作名称（用于日志）。
    - max_retries: 最大重试次数。
    - retry_wait_seconds: 重试等待基准秒数。
    - logger: 日志对象。
    返回值：无。
    异常说明：
    - DownloadEngineError: 重试耗尽后抛出。
    边界条件：重试等待采用指数退避。
    """
    normalized_retries = max(1, int(max_retries))
    normalized_wait = max(0.1, float(retry_wait_seconds))
    last_error: Exception | None = None

    for attempt_index in range(normalized_retries):
        attempt_no = attempt_index + 1
        try:
            logger.info("开始执行 %s，attempt=%s/%s", action_name, attempt_no, normalized_retries)
            runner()
            logger.info("%s 执行成功", action_name)
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
            logger.warning(
                "%s 执行失败，attempt=%s/%s，错误=%s",
                action_name,
                attempt_no,
                normalized_retries,
                error,
            )
            if attempt_no >= normalized_retries:
                break
            wait_seconds = normalized_wait * (2**attempt_index)
            logger.info("%s 准备重试，%s 秒后继续。", action_name, round(wait_seconds, 2))
            time.sleep(wait_seconds)

    raise DownloadEngineError(f"{action_name} 重试耗尽，最后错误：{last_error}")


def build_hf_environment(logger) -> dict[str, str]:
    """
    功能说明：构建 Hugging Face 下载环境变量。
    参数说明：
    - logger: 日志对象。
    返回值：
    - dict[str, str]: 传给子进程的环境变量副本。
    异常说明：无。
    边界条件：当 HF_ENDPOINT 缺失时自动注入默认镜像站。
    """
    env = dict(os.environ)
    endpoint = str(env.get("HF_ENDPOINT", "")).strip()
    if not endpoint:
        endpoint = DEFAULT_HF_ENDPOINT
        env["HF_ENDPOINT"] = endpoint
    logger.info("已启用 HF_ENDPOINT=%s", endpoint)
    return env


def _download_with_aria2c(url: str, save_path: Path, timeout_seconds: int) -> None:
    """
    功能说明：使用 aria2c 下载单文件。
    参数说明：
    - url: 下载地址。
    - save_path: 目标文件路径。
    - timeout_seconds: 网络超时秒数。
    返回值：无。
    异常说明：
    - DownloadEngineError: aria2c 返回非 0 时抛出。
    边界条件：保留 aria2c 原生进度输出。
    """
    command = [
        "aria2c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--split=8",
        "--max-connection-per-server=8",
        "--min-split-size=1M",
        "--timeout",
        str(max(1, int(timeout_seconds))),
        "--max-tries=1",
        "--dir",
        str(save_path.parent),
        "--out",
        save_path.name,
        url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise DownloadEngineError(f"aria2c 返回非零退出码：{result.returncode}")


def _download_with_wget(url: str, save_path: Path, timeout_seconds: int) -> None:
    """
    功能说明：使用 wget 下载单文件。
    参数说明：
    - url: 下载地址。
    - save_path: 目标文件路径。
    - timeout_seconds: 网络超时秒数。
    返回值：无。
    异常说明：
    - DownloadEngineError: wget 返回非 0 时抛出。
    边界条件：保留 wget 原生进度输出。
    """
    command = [
        "wget",
        "--continue",
        "--tries=1",
        f"--timeout={max(1, int(timeout_seconds))}",
        "--output-document",
        str(save_path),
        url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise DownloadEngineError(f"wget 返回非零退出码：{result.returncode}")


def _download_with_requests(url: str, save_path: Path, timeout_seconds: int, logger) -> None:
    """
    功能说明：使用 requests 流式下载单文件。
    参数说明：
    - url: 下载地址。
    - save_path: 目标文件路径。
    - timeout_seconds: 网络超时秒数。
    - logger: 日志对象。
    返回值：无。
    异常说明：
    - DownloadEngineError: HTTP/写盘失败时抛出。
    边界条件：先写 .part 临时文件，完成后原子替换。
    """
    temp_path = save_path.with_suffix(save_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with requests.get(url, stream=True, timeout=max(1, int(timeout_seconds)), allow_redirects=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", "0") or "0")
            downloaded_size = 0
            last_log_time = 0.0

            with temp_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=REQUEST_CHUNK_SIZE):
                    if not chunk:
                        continue
                    file_obj.write(chunk)
                    downloaded_size += len(chunk)

                    now = time.time()
                    if now - last_log_time >= 1.0:
                        if total_size > 0:
                            percent = downloaded_size / total_size * 100
                            logger.info(
                                "requests 下载进度：%.2f%%（%s/%s bytes）",
                                percent,
                                downloaded_size,
                                total_size,
                            )
                        else:
                            logger.info("requests 下载进度：已下载 %s bytes", downloaded_size)
                        last_log_time = now

        if (not temp_path.exists()) or temp_path.stat().st_size <= 0:
            raise DownloadEngineError("requests 下载结果为空文件")

        temp_path.replace(save_path)
    except Exception as error:  # noqa: BLE001
        if temp_path.exists():
            temp_path.unlink()
        raise DownloadEngineError(f"requests 下载失败：{error}") from error


def download_file(
    url: str,
    save_path: Path,
    logger,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """
    功能说明：按 aria2c -> wget -> requests 顺序下载文件并自动回退。
    参数说明：
    - url: 下载地址。
    - save_path: 目标文件路径。
    - logger: 日志对象。
    - max_retries: 每个后端最大重试次数。
    - retry_wait_seconds: 重试等待基准秒数。
    - timeout_seconds: 网络超时秒数。
    返回值：无。
    异常说明：
    - DownloadEngineError: 所有后端失败时抛出。
    边界条件：每个后端执行成功后会进行非空文件校验。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    backends: list[tuple[str, Callable[[], None]]] = []

    if shutil.which("aria2c"):
        backends.append(("aria2c", lambda: _download_with_aria2c(url=url, save_path=save_path, timeout_seconds=timeout_seconds)))
    else:
        logger.warning("未检测到 aria2c，自动回退到 wget/requests")

    if shutil.which("wget"):
        backends.append(("wget", lambda: _download_with_wget(url=url, save_path=save_path, timeout_seconds=timeout_seconds)))
    else:
        logger.warning("未检测到 wget，自动回退到 requests")

    backends.append(
        (
            "requests",
            lambda: _download_with_requests(
                url=url,
                save_path=save_path,
                timeout_seconds=timeout_seconds,
                logger=logger,
            ),
        )
    )

    errors: list[str] = []
    for backend_name, runner in backends:
        try:
            run_with_retry(
                runner=runner,
                action_name=f"文件下载后端({backend_name})",
                max_retries=max_retries,
                retry_wait_seconds=retry_wait_seconds,
                logger=logger,
            )
            ensure_non_empty_file(save_path)
            return
        except Exception as error:  # noqa: BLE001
            errors.append(f"{backend_name}: {error}")
            logger.warning("后端 %s 下载失败，准备降级下一后端。", backend_name)

    raise DownloadEngineError("文件下载失败，所有后端均不可用：" + " | ".join(errors))


def download_repo_with_hf_cli(
    repo_id: str,
    revision: str,
    local_dir: Path,
    logger,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
) -> None:
    """
    功能说明：使用 huggingface-cli 下载仓库快照到本地目录。
    参数说明：
    - repo_id: Hugging Face 仓库 ID。
    - revision: 仓库分支或提交号。
    - local_dir: 本地保存目录。
    - logger: 日志对象。
    - max_retries: 最大重试次数。
    - retry_wait_seconds: 重试等待基准秒数。
    - include_patterns: 仅下载匹配规则（可选）。
    - exclude_patterns: 排除下载匹配规则（可选）。
    返回值：无。
    异常说明：
    - DownloadEngineError: 命令不可用、执行失败或结果为空时抛出。
    边界条件：若本地目录不存在会自动创建。
    """
    cli_path = shutil.which("huggingface-cli")
    if not cli_path:
        raise DownloadEngineError("未找到 huggingface-cli，请先执行：pip install -U huggingface_hub")

    local_dir.mkdir(parents=True, exist_ok=True)
    env = build_hf_environment(logger=logger)

    command = [
        cli_path,
        "download",
        "--resume-download",
        repo_id,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
        "--local-dir-use-symlinks",
        "False",
    ]
    # huggingface-cli 的 include/exclude 参数是 nargs 形式，需一次性传入参数列表。
    if include_patterns:
        command.extend(["--include", *include_patterns])
    if exclude_patterns:
        command.extend(["--exclude", *exclude_patterns])

    def _runner() -> None:
        logger.info("执行 huggingface-cli 命令：%s", " ".join(command))
        result = subprocess.run(command, check=False, env=env)
        if result.returncode != 0:
            raise DownloadEngineError(f"huggingface-cli 返回非零退出码：{result.returncode}")

    run_with_retry(
        runner=_runner,
        action_name=f"HF仓库下载(repo_id={repo_id}, revision={revision})",
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds,
        logger=logger,
    )

    if not has_any_file(local_dir):
        raise DownloadEngineError(f"HF 仓库下载后目录为空：{local_dir}")
