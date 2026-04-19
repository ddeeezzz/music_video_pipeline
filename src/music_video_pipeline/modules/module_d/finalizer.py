"""
文件用途：提供模块 D 的拼接与 FFmpeg/FFprobe 通用执行能力。
核心流程：按两阶段终拼策略执行 concat，并在需要时从 copy 回退 reencode。
输入输出：输入片段路径与编码配置，输出最终拼接执行结果。
依赖说明：依赖标准库 subprocess/logging/pathlib/functools。
维护说明：本文件只负责终拼与工具函数，不承担单元调度职责。
"""

# 标准库：用于函数结果缓存
from functools import lru_cache
# 标准库：用于日志类型
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于子进程执行
import subprocess
# 标准库：用于类型提示
from typing import Any


@lru_cache(maxsize=8)
def _probe_ffmpeg_encoder_capabilities(ffmpeg_bin: str) -> dict[str, bool]:
    """
    功能说明：探测 ffmpeg 编码器能力并缓存结果。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    返回值：
    - dict[str, bool]: 编码器可用性字典（键为编码器名）。
    异常说明：探测失败时返回空能力集合，不在此处抛错。
    边界条件：缓存按 ffmpeg_bin 维度生效，避免重复探测开销。
    """
    command = [ffmpeg_bin, "-hide_banner", "-encoders"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except Exception:  # noqa: BLE001
        return {}

    output_text = str(result.stdout)
    return {
        "h264_nvenc": "h264_nvenc" in output_text,
        "hevc_nvenc": "hevc_nvenc" in output_text,
    }


def _normalize_video_accel_mode(video_accel_mode: str) -> str:
    """
    功能说明：归一化视频加速模式配置。
    参数说明：
    - video_accel_mode: 原始配置值。
    返回值：
    - str: 合法模式（auto/cpu_only/gpu_only）。
    异常说明：无。
    边界条件：非法值回退为 auto。
    """
    normalized = str(video_accel_mode).strip().lower()
    if normalized in {"auto", "cpu_only", "gpu_only"}:
        return normalized
    return "auto"


def _normalize_concat_video_mode(concat_video_mode: str) -> str:
    """
    功能说明：归一化最终拼接视频模式配置。
    参数说明：
    - concat_video_mode: 原始配置值。
    返回值：
    - str: 合法模式（copy/reencode）。
    异常说明：无。
    边界条件：非法值回退为 copy。
    """
    normalized = str(concat_video_mode).strip().lower()
    if normalized in {"copy", "reencode"}:
        return normalized
    return "copy"


def _clamp_nvenc_cq(gpu_cq: int | None, fallback_crf: int) -> int:
    """
    功能说明：获取 NVENC CQ 参数，未配置时从 CRF 近似映射。
    参数说明：
    - gpu_cq: 配置中的 GPU CQ（可选）。
    - fallback_crf: CPU 路径 CRF，作为 CQ 近似值来源。
    返回值：
    - int: 归一化后的 CQ 值。
    异常说明：无。
    边界条件：CQ 限制在 [0, 51]。
    """
    if gpu_cq is None:
        candidate = int(fallback_crf)
    else:
        candidate = int(gpu_cq)
    return max(0, min(51, candidate))


def _normalize_nvenc_rc_mode_for_preset(gpu_rc_mode: str, gpu_preset: str) -> str:
    """
    功能说明：在旧版 NVENC 约束下归一化 RC 模式，避免与 p1~p7 预设冲突。
    参数说明：
    - gpu_rc_mode: 原始 RC 模式配置。
    - gpu_preset: 原始 GPU 预设配置。
    返回值：
    - str: 兼容后的 RC 模式。
    异常说明：无。
    边界条件：当 preset 为 p1~p7 且 rc 为 vbr_hq/cbr_hq 时自动降级为 vbr/cbr。
    """
    normalized_rc = str(gpu_rc_mode).strip().lower() or "vbr"
    normalized_preset = str(gpu_preset).strip().lower()
    preset_is_p_series = normalized_preset.startswith("p") and normalized_preset[1:].isdigit()
    if preset_is_p_series and normalized_rc == "vbr_hq":
        return "vbr"
    if preset_is_p_series and normalized_rc == "cbr_hq":
        return "cbr"
    return normalized_rc


def _resolve_video_encoder_profile(
    ffmpeg_bin: str,
    video_accel_mode: str,
    cpu_video_codec: str,
    cpu_video_preset: str,
    cpu_video_crf: int,
    gpu_video_codec: str,
    gpu_preset: str,
    gpu_rc_mode: str,
    gpu_cq: int | None,
    gpu_bitrate: str | None,
) -> dict[str, Any]:
    """
    功能说明：根据配置与能力探测结果选择视频编码方案。
    参数说明：
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - cpu_video_codec/cpu_video_preset/cpu_video_crf: CPU 编码配置。
    - gpu_video_codec/gpu_preset/gpu_rc_mode/gpu_cq/gpu_bitrate: GPU 编码配置。
    返回值：
    - dict[str, Any]: 编码方案（含 use_gpu/command_args/fallback_cpu_profile）。
    异常说明：gpu_only 且编码器不可用时抛 RuntimeError。
    边界条件：auto 模式下 GPU 不可用自动回退 CPU。
    """
    normalized_mode = _normalize_video_accel_mode(video_accel_mode=video_accel_mode)
    normalized_gpu_codec = str(gpu_video_codec).strip().lower() or "h264_nvenc"
    capabilities = _probe_ffmpeg_encoder_capabilities(ffmpeg_bin=ffmpeg_bin)
    gpu_available = bool(capabilities.get(normalized_gpu_codec, False))

    cpu_profile = {
        "use_gpu": False,
        "name": "cpu",
        "codec": cpu_video_codec,
        "command_args": [
            "-c:v",
            cpu_video_codec,
            "-preset",
            cpu_video_preset,
            "-crf",
            str(cpu_video_crf),
        ],
        "fallback_cpu_profile": None,
    }

    normalized_gpu_rc_mode = _normalize_nvenc_rc_mode_for_preset(
        gpu_rc_mode=gpu_rc_mode,
        gpu_preset=gpu_preset,
    )
    gpu_profile = {
        "use_gpu": True,
        "name": "gpu",
        "codec": normalized_gpu_codec,
        "command_args": [
            "-c:v",
            normalized_gpu_codec,
            "-preset",
            str(gpu_preset),
            "-rc",
            str(normalized_gpu_rc_mode),
            "-cq",
            str(_clamp_nvenc_cq(gpu_cq=gpu_cq, fallback_crf=cpu_video_crf)),
        ],
        "fallback_cpu_profile": cpu_profile,
    }
    if gpu_bitrate:
        gpu_profile["command_args"].extend(["-b:v", str(gpu_bitrate)])
    else:
        gpu_profile["command_args"].extend(["-b:v", "0"])

    if normalized_mode == "cpu_only":
        return cpu_profile
    if normalized_mode == "gpu_only":
        if not gpu_available:
            raise RuntimeError(f"模块D-GPU编码不可用：未检测到编码器 {normalized_gpu_codec}")
        return {**gpu_profile, "fallback_cpu_profile": None}

    if gpu_available:
        return gpu_profile
    return cpu_profile


def _concat_segment_videos(
    segment_paths: list[Path],
    concat_file_path: Path,
    ffmpeg_bin: str,
    audio_path: Path,
    output_video_path: Path,
    audio_duration: float,
    fps: int,
    video_codec: str,
    audio_codec: str,
    video_preset: str,
    video_crf: int,
    *,
    video_accel_mode: str = "auto",
    gpu_video_codec: str = "h264_nvenc",
    gpu_preset: str = "p1",
    gpu_rc_mode: str = "vbr",
    gpu_cq: int | None = 34,
    gpu_bitrate: str | None = None,
    concat_video_mode: str = "copy",
    concat_copy_fallback_reencode: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    功能说明：拼接小视频片段并混入原音轨，生成最终成片。
    参数说明：
    - segment_paths: 小视频片段路径数组。
    - concat_file_path: concat 清单文件路径。
    - ffmpeg_bin: ffmpeg 可执行文件名或路径。
    - audio_path: 原音轨路径。
    - output_video_path: 输出视频路径。
    - audio_duration: 原音轨时长（秒）。
    - fps: 输出帧率。
    - video_codec: 视频编码器。
    - audio_codec: 音频编码器。
    - video_preset: 视频编码预设。
    - video_crf: 视频 CRF 参数。
    - video_accel_mode: 视频加速模式（auto/cpu_only/gpu_only）。
    - gpu_video_codec/gpu_preset/gpu_rc_mode/gpu_cq/gpu_bitrate: GPU 编码配置。
    - concat_video_mode: 最终拼接模式（copy/reencode）。
    - concat_copy_fallback_reencode: copy 失败时是否回退重编码。
    - logger: 日志对象（可选）。
    返回值：
    - dict[str, Any]: 拼接阶段执行信息（mode/copy_fallback_triggered）。
    异常说明：ffmpeg 调用失败时抛 RuntimeError。
    边界条件：显式使用 -t 音频时长，避免最终视频长于音频。
    """
    active_logger = logger or logging.getLogger("D")
    concat_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{_escape_concat_path(str(path))}'" for path in segment_paths]
    concat_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    normalized_concat_mode = _normalize_concat_video_mode(concat_video_mode=concat_video_mode)
    copy_command = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-t",
        f"{audio_duration:.3f}",
        str(output_video_path),
    ]

    profile = _resolve_video_encoder_profile(
        ffmpeg_bin=ffmpeg_bin,
        video_accel_mode=video_accel_mode,
        cpu_video_codec=video_codec,
        cpu_video_preset=video_preset,
        cpu_video_crf=video_crf,
        gpu_video_codec=gpu_video_codec,
        gpu_preset=gpu_preset,
        gpu_rc_mode=gpu_rc_mode,
        gpu_cq=gpu_cq,
        gpu_bitrate=gpu_bitrate,
    )
    reencode_command = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file_path),
        "-i",
        str(audio_path),
        *list(profile["command_args"]),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        audio_codec,
        "-t",
        f"{audio_duration:.3f}",
        str(output_video_path),
    ]

    if normalized_concat_mode == "reencode":
        _run_ffmpeg_command(command=reencode_command, command_name="拼接小片段并混音（reencode）")
        return {"mode": "reencode", "copy_fallback_triggered": False}

    try:
        _run_ffmpeg_command(command=copy_command, command_name="拼接小片段并混音（copy）")
        return {"mode": "copy", "copy_fallback_triggered": False}
    except RuntimeError as copy_error:
        if not bool(concat_copy_fallback_reencode):
            raise
        active_logger.warning("模块D-concat copy 失败，已回退 reencode，错误=%s", copy_error)
        _run_ffmpeg_command(command=reencode_command, command_name="拼接小片段并混音（copy回退reencode）")
        return {"mode": "copy_with_reencode_fallback", "copy_fallback_triggered": True}


def _probe_media_duration(media_path: Path, ffprobe_bin: str) -> float:
    """
    功能说明：使用 ffprobe 获取媒体时长（秒）。
    参数说明：
    - media_path: 媒体文件路径。
    - ffprobe_bin: ffprobe 可执行文件名或路径。
    返回值：
    - float: 媒体时长秒数。
    异常说明：ffprobe 执行失败或解析失败时抛 RuntimeError。
    边界条件：返回值最小为 0.1 秒，避免无效时长。
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"找不到 ffprobe 可执行文件，请检查 ffprobe_bin={ffprobe_bin}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffprobe 执行失败，stderr={error.stderr}") from error

    output_text = result.stdout.strip()
    try:
        duration = float(output_text)
    except ValueError as error:
        raise RuntimeError(f"ffprobe 时长解析失败，输出内容={output_text!r}") from error
    return max(0.1, duration)


def _run_ffmpeg_command(command: list[str], command_name: str) -> None:
    """
    功能说明：统一执行 ffmpeg 命令并抛出带上下文的错误信息。
    参数说明：
    - command: ffmpeg 命令参数数组。
    - command_name: 命令用途说明。
    返回值：无。
    异常说明：命令执行失败时抛 RuntimeError。
    边界条件：stderr 按 utf-8 replace 解码，避免 Windows 编码报错中断。
    """
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{command_name}失败：找不到 ffmpeg 可执行文件。") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{command_name}失败：ffmpeg 返回非零状态。\n命令：{' '.join(command)}\nstderr: {error.stderr}"
        ) from error


def _escape_concat_path(path_text: str) -> str:
    """
    功能说明：转义 concat 文件中的路径文本。
    参数说明：
    - path_text: 原始路径字符串。
    返回值：
    - str: 适合写入 concat 文件的路径文本。
    异常说明：无。
    边界条件：仅处理单引号转义，其他字符按原样保留。
    """
    return path_text.replace("'", "'\\''")
