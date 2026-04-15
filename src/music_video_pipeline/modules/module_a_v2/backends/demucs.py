"""
文件用途：提供模块A V2的 Demucs 准备与标准二轨输出能力。
核心流程：调用 allin1fix Demucs 分离并合成 no_vocals，再统一采样率。
输入输出：输入音频与运行参数，输出 vocals/no_vocals 及 stems_input。
依赖说明：依赖 importlib、subprocess、shutil 与 ffmpeg/ffprobe。
维护说明：本文件只承载分离与音轨标准化，不负责分段分析。
"""

# 标准库：动态导入
import importlib
# 标准库：子进程调用
import subprocess
# 标准库：路径处理
from pathlib import Path
# 标准库：命令探测
import shutil
# 标准库：类型提示
from typing import Any


def prepare_stems_with_allin1_demucs(
    audio_path: Path,
    output_dir: Path,
    device: str,
    model_name: str,
    logger,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    功能说明：复用 allin1fix 的 Demucs 分离能力，返回声轨路径与可供 allin1 分析的 stems_input。
    参数说明：
    - audio_path: 输入音频文件路径。
    - output_dir: allin1fix 分离结果输出目录。
    - device: 推理设备标识（如 cpu/cuda/auto）。
    - model_name: 分离模型名称（如 htdemucs）。
    - logger: 日志记录器，用于输出过程与异常信息。
    返回值：
    - tuple[Path, Path, dict[str, Any]]: `(vocals_path, no_vocals_path, stems_input)`。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅在 allin1fix 后端可用时生效，不支持时抛错由上层降级。
    """
    backend_name, backend_module = _import_allin1_backend()
    if backend_name != "allin1fix" or not hasattr(backend_module, "get_stems"):
        raise RuntimeError(f"当前后端={backend_name}，不支持复用 allin1fix 的 Demucs 分离")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_device = _normalize_allin1_runtime_device(device=device)
    provider_obj = None
    provider_cls = getattr(backend_module, "DemucsProvider", None)
    if provider_cls is not None:
        try:
            provider_obj = provider_cls(model_name=model_name, device=runtime_device)
        except TypeError:
            provider_obj = provider_cls(device=runtime_device)

    logger.info("模块A V2调用 allin1fix-Demucs 分离，模型=%s，设备=%s", model_name, runtime_device)
    stem_dirs = backend_module.get_stems([audio_path], output_dir, provider_obj, runtime_device)
    if not stem_dirs:
        raise RuntimeError("allin1fix Demucs 未返回可用分离目录")

    stem_dir = Path(stem_dirs[0])
    stem_paths = {
        "bass": stem_dir / "bass.wav",
        "drums": stem_dir / "drums.wav",
        "other": stem_dir / "other.wav",
        "vocals": stem_dir / "vocals.wav",
    }
    missing_files = [name for name, path in stem_paths.items() if not path.exists()]
    if missing_files:
        raise RuntimeError(f"allin1fix Demucs 缺少分离文件: {missing_files}")

    stems_input = {
        "bass": stem_paths["bass"],
        "drums": stem_paths["drums"],
        "other": stem_paths["other"],
        "vocals": stem_paths["vocals"],
        "identifier": audio_path.stem,
    }
    no_vocals_path = stem_dir / "no_vocals.wav"
    _mix_non_vocal_stems_to_no_vocals(
        bass_path=stem_paths["bass"],
        drums_path=stem_paths["drums"],
        other_path=stem_paths["other"],
        output_path=no_vocals_path,
        logger=logger,
    )
    target_sample_rate = _probe_audio_sample_rate(audio_path=audio_path, logger=logger)
    _normalize_standard_stems_sample_rate(
        vocals_path=stem_paths["vocals"],
        no_vocals_path=no_vocals_path,
        target_sample_rate=target_sample_rate,
        logger=logger,
    )
    return stem_paths["vocals"], no_vocals_path, stems_input


def _import_allin1_backend() -> tuple[str, Any]:
    """
    功能说明：按优先级导入 allin1 后端，兼容 allin1fix 包名。
    参数说明：无。
    返回值：
    - tuple[str, Any]: 后端模块名称与模块对象二元组。
    异常说明：导入失败时抛 RuntimeError。
    边界条件：依次尝试 allin1 与 allin1fix。
    """
    import_errors: list[str] = []
    for module_name in ("allin1", "allin1fix"):
        try:
            module_obj = importlib.import_module(module_name)
            return module_name, module_obj
        except Exception as error:  # noqa: BLE001
            import_errors.append(f"{module_name}: {error}")
    raise RuntimeError(f"allin1 导入失败，已尝试 allin1/allin1fix，错误详情：{' | '.join(import_errors)}")


def _normalize_allin1_runtime_device(device: str) -> str:
    """
    功能说明：归一化 allin1fix 分离阶段设备参数，仅输出 cpu/cuda。
    参数说明：
    - device: 推理设备标识（如 cpu/cuda/auto）。
    返回值：
    - str: 归一化后的设备字符串，仅为 cpu 或 cuda。
    异常说明：无。
    边界条件：当 torch 不可用或无 CUDA 时默认 cpu。
    """
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    try:
        # 第三方库：设备可用性探测
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _probe_audio_sample_rate(audio_path: Path, logger) -> int | None:
    """
    功能说明：通过 ffprobe 读取音频采样率（Hz）。
    参数说明：
    - audio_path: 输入音频路径。
    - logger: 日志记录器。
    返回值：
    - int | None: 采样率；失败时返回 None。
    异常说明：异常在函数内吞并并记录 warning，不中断主流程。
    边界条件：ffprobe 不可用时直接返回 None。
    """
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        logger.warning("模块A V2-未检测到 ffprobe，跳过采样率标准化探测")
        return None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        text = str(result.stdout).strip()
        if not text:
            return None
        return int(float(text.splitlines()[0]))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块A V2-采样率探测失败，路径=%s，错误=%s", audio_path, error)
        return None


def _resample_audio_file_inplace(audio_path: Path, target_sample_rate: int, logger) -> None:
    """
    功能说明：使用 ffmpeg 原地重采样音频文件并强制双声道输出。
    参数说明：
    - audio_path: 待重采样音频路径。
    - target_sample_rate: 目标采样率（Hz）。
    - logger: 日志记录器。
    返回值：无。
    异常说明：重采样失败抛错，由上层统一处理。
    边界条件：ffmpeg 不可用时抛错，避免静默格式漂移。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("未检测到 ffmpeg，可用性不足以执行采样率标准化")

    temp_output_path = audio_path.with_name(f"{audio_path.stem}.resample_tmp.wav")
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ar",
        str(int(target_sample_rate)),
        "-ac",
        "2",
        str(temp_output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    temp_output_path.replace(audio_path)
    logger.info("模块A V2-重采样完成，路径=%s，target_sr=%s", audio_path, target_sample_rate)


def _mix_non_vocal_stems_to_no_vocals(
    bass_path: Path,
    drums_path: Path,
    other_path: Path,
    output_path: Path,
    logger,
) -> None:
    """
    功能说明：将 bass/drums/other 三轨等权求和并做峰值保护，输出 no_vocals.wav。
    参数说明：
    - bass_path: bass 轨路径。
    - drums_path: drums 轨路径。
    - other_path: other 轨路径。
    - output_path: 输出 no_vocals 路径。
    - logger: 日志记录器。
    返回值：无。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：依赖 ffmpeg amix + alimiter，输出目录不存在时自动创建。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("未检测到 ffmpeg，可用性不足以合成 no_vocals")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(bass_path),
        "-i",
        str(drums_path),
        "-i",
        str(other_path),
        "-filter_complex",
        "[0:a][1:a][2:a]amix=inputs=3:normalize=0,alimiter=limit=0.98",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    logger.info("模块A V2-已合成 no_vocals，路径=%s", output_path)


def _normalize_standard_stems_sample_rate(
    vocals_path: Path,
    no_vocals_path: Path,
    target_sample_rate: int | None,
    logger,
) -> None:
    """
    功能说明：将标准二轨（vocals/no_vocals）统一重采样到指定采样率。
    参数说明：
    - vocals_path: vocals 轨路径。
    - no_vocals_path: no_vocals 轨路径。
    - target_sample_rate: 目标采样率（Hz）；为 None 时跳过。
    - logger: 日志记录器。
    返回值：无。
    异常说明：异常由调用方或上层流程统一处理。
    边界条件：仅在目标采样率有效时执行重采样。
    """
    if target_sample_rate is None or target_sample_rate <= 0:
        return
    _resample_audio_file_inplace(audio_path=vocals_path, target_sample_rate=target_sample_rate, logger=logger)
    _resample_audio_file_inplace(audio_path=no_vocals_path, target_sample_rate=target_sample_rate, logger=logger)
    logger.info(
        "模块A V2-标准二轨采样率已统一，target_sr=%s",
        target_sample_rate,
    )
