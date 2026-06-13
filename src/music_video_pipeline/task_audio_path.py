"""
文件用途：解析任务记录中的输入音频路径，并在跨机器迁移后尝试回映射到当前工作区。
核心流程：优先使用任务记录原始路径；若失效，则结合工作区根目录、resources 目录与任务配置默认音频做候选回退。
输入输出：输入任务记录中的 audio_path/config_path 与工作区候选根目录，输出当前机器可访问的真实音频路径。
依赖说明：依赖标准库 pathlib/re 与项目内配置加载器。
维护说明：新增输入路径约定时，需同步扩展候选生成逻辑与测试覆盖。
"""

import os
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于轻量 JSON 读取
import json
# 标准库：用于正则判断 Windows 盘符绝对路径
import re
# 标准库：用于日志输出
import logging
# 标准库：用于类型提示
from typing import Iterable

# 项目内模块：读取任务配置默认音频路径
from music_video_pipeline.config import load_config

# 模块级日志
logger = logging.getLogger(__name__)

# 常量：识别 `C:\\foo\\bar` 这类 Windows 盘符绝对路径。
WINDOWS_DRIVE_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def find_task_audio_path(
    *,
    raw_audio_path: str,
    config_path: str = "",
    workspace_roots: Iterable[Path] = (),
    fallback_default_audio_path: str = "",
) -> Path | None:
    """
    功能说明：尝试把任务记录中的音频路径映射到当前机器上真实存在的文件。
    参数说明：
    - raw_audio_path: 任务记录中的原始音频路径。
    - config_path: 任务记录中的配置路径；可用于读取默认音频回退。
    - workspace_roots: 当前工作区候选根目录数组。
    - fallback_default_audio_path: 可选的默认音频相对/绝对路径回退。
    返回值：
    - Path | None: 找到可用音频文件时返回绝对路径，否则返回 None。
    异常说明：无；配置解析失败时仅忽略该候选。
    边界条件：兼容 Linux 绝对路径、Windows 绝对路径与 Windows 下 `\\root\\...` 这类外机根路径文本。
    """
    normalized_audio_path = str(raw_audio_path).strip()
    if not normalized_audio_path:
        return None

    normalized_workspace_roots = _normalize_workspace_roots(workspace_roots=workspace_roots)
    candidate_paths: list[Path] = []
    raw_candidate = Path(normalized_audio_path)
    raw_parts = [str(part) for part in raw_candidate.parts if str(part).strip()]
    raw_parts_lower = [part.lower() for part in raw_parts]
    is_explicit_absolute = _looks_like_absolute_path_text(normalized_audio_path)

    if is_explicit_absolute or raw_candidate.is_absolute():
        _append_candidate(candidate_paths, raw_candidate)

    for workspace_root in normalized_workspace_roots:
        if (not is_explicit_absolute) and (not raw_candidate.is_absolute()):
            _append_candidate(candidate_paths, workspace_root / raw_candidate)
        if raw_candidate.name:
            _append_candidate(candidate_paths, workspace_root / "resources" / raw_candidate.name)
        if "resources" in raw_parts_lower:
            resources_index = max(index for index, part_text in enumerate(raw_parts_lower) if part_text == "resources")
            tail_parts = raw_parts[resources_index + 1 :]
            if tail_parts:
                _append_candidate(candidate_paths, workspace_root.joinpath("resources", *tail_parts))

    for resolved_config_path, config_workspace_root in _iter_config_candidates(
        config_path=config_path,
        workspace_roots=normalized_workspace_roots,
    ):
        default_audio_path_text = _read_default_audio_path_from_config(config_path=resolved_config_path)
        if not default_audio_path_text:
            continue
        _append_default_audio_candidates(
            candidate_paths=candidate_paths,
            default_audio_path=default_audio_path_text,
            workspace_root=config_workspace_root,
        )

    if str(fallback_default_audio_path).strip():
        for workspace_root in normalized_workspace_roots:
            _append_default_audio_candidates(
                candidate_paths=candidate_paths,
                default_audio_path=fallback_default_audio_path,
                workspace_root=workspace_root,
            )

    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    return None


def resolve_task_audio_path(
    *,
    raw_audio_path: str,
    config_path: str = "",
    workspace_roots: Iterable[Path] = (),
    fallback_default_audio_path: str = "",
) -> Path:
    """
    功能说明：解析任务输入音频路径；若无法回映射则抛出明确错误。
    参数说明：
    - raw_audio_path: 任务记录中的原始音频路径。
    - config_path: 任务记录中的配置路径。
    - workspace_roots: 当前工作区候选根目录数组。
    - fallback_default_audio_path: 可选默认音频路径回退。
    返回值：
    - Path: 当前机器上真实可访问的音频绝对路径。
    异常说明：
    - FileNotFoundError: 原始路径与本机回退路径均不可用时抛出。
    边界条件：错误信息保留原始路径文本，便于定位旧任务记录来源。
    """
    resolved_path = find_task_audio_path(
        raw_audio_path=raw_audio_path,
        config_path=config_path,
        workspace_roots=workspace_roots,
        fallback_default_audio_path=fallback_default_audio_path,
    )
    if resolved_path is not None:
        return resolved_path
    normalized_audio_path = str(raw_audio_path).strip()
    normalized_config_path = str(config_path).strip()
    error_parts = [f"原始路径={normalized_audio_path}"]
    if normalized_config_path:
        error_parts.append(f"config_path={normalized_config_path}")
    raise FileNotFoundError("音频文件不存在，且无法从当前工作区重映射：" + "，".join(error_parts))


def _normalize_workspace_roots(*, workspace_roots: Iterable[Path]) -> list[Path]:
    """
    功能说明：标准化并去重工作区根目录候选。
    参数说明：
    - workspace_roots: 原始工作区根目录迭代器。
    返回值：
    - list[Path]: 已去重的绝对路径数组。
    异常说明：无。
    边界条件：忽略 None、空字符串与无法解析的路径对象。
    """
    normalized_roots: list[Path] = []
    seen_keys: set[str] = set()
    for root in workspace_roots:
        if root is None:
            continue
        try:
            resolved_root = Path(root).resolve()
        except Exception:  # noqa: BLE001
            continue
        root_key = str(resolved_root).casefold()
        if root_key in seen_keys:
            continue
        seen_keys.add(root_key)
        normalized_roots.append(resolved_root)
    return normalized_roots


def _iter_config_candidates(*, config_path: str, workspace_roots: list[Path]) -> list[tuple[Path, Path]]:
    """
    功能说明：生成可用于读取任务配置的绝对路径及其对应工作区根目录。
    参数说明：
    - config_path: 任务记录中的配置路径文本。
    - workspace_roots: 工作区候选根目录数组。
    返回值：
    - list[tuple[Path, Path]]: `(配置绝对路径, 对应工作区根目录)` 数组。
    异常说明：无。
    边界条件：相对配置路径会针对所有工作区候选展开。
    """
    normalized_config_path = str(config_path).strip()
    if not normalized_config_path:
        return []
    config_candidate = Path(normalized_config_path)
    candidate_pairs: list[tuple[Path, Path]] = []
    seen_keys: set[str] = set()
    if _looks_like_absolute_path_text(normalized_config_path) or config_candidate.is_absolute():
        resolved_path = config_candidate.resolve()
        root_candidates = _derive_workspace_roots_from_config_path(resolved_path=resolved_path, fallback_roots=workspace_roots)
        for workspace_root in root_candidates:
            pair_key = f"{str(resolved_path).casefold()}::{str(workspace_root).casefold()}"
            if pair_key in seen_keys:
                continue
            seen_keys.add(pair_key)
            candidate_pairs.append((resolved_path, workspace_root))
        return candidate_pairs

    for workspace_root in workspace_roots:
        resolved_path = (workspace_root / config_candidate).resolve()
        pair_key = f"{str(resolved_path).casefold()}::{str(workspace_root).casefold()}"
        if pair_key in seen_keys:
            continue
        seen_keys.add(pair_key)
        candidate_pairs.append((resolved_path, workspace_root))
    return candidate_pairs


def _derive_workspace_roots_from_config_path(*, resolved_path: Path, fallback_roots: list[Path]) -> list[Path]:
    """
    功能说明：根据配置路径反推出可能的工作区根目录。
    参数说明：
    - resolved_path: 配置文件绝对路径。
    - fallback_roots: 调用方给出的工作区兜底候选。
    返回值：
    - list[Path]: 优先包含从 `configs/` 目录反推的工作区根目录。
    异常说明：无。
    边界条件：若路径中不含 `configs`，则仅返回调用方兜底候选。
    """
    candidates: list[Path] = []
    resolved_parts = list(resolved_path.parts)
    lowered_parts = [str(part).lower() for part in resolved_parts]
    if "configs" in lowered_parts:
        configs_index = lowered_parts.index("configs")
        if configs_index > 0:
            candidates.append(Path(*resolved_parts[:configs_index]).resolve())
    candidates.extend(fallback_roots)
    return _normalize_workspace_roots(workspace_roots=candidates)


def _append_default_audio_candidates(*, candidate_paths: list[Path], default_audio_path: str, workspace_root: Path) -> None:
    """
    功能说明：把配置默认音频路径展开为候选绝对路径。
    参数说明：
    - candidate_paths: 候选路径数组（原地追加）。
    - default_audio_path: 配置中的默认音频路径文本。
    - workspace_root: 对应工作区根目录。
    返回值：无。
    异常说明：无。
    边界条件：绝对路径直接使用；相对路径默认相对于工作区根目录。
    """
    normalized_default_audio_path = str(default_audio_path).strip()
    if not normalized_default_audio_path:
        return
    default_candidate = Path(normalized_default_audio_path)
    if _looks_like_absolute_path_text(normalized_default_audio_path) or default_candidate.is_absolute():
        _append_candidate(candidate_paths, default_candidate)
        return
    _append_candidate(candidate_paths, workspace_root / default_candidate)


def _read_default_audio_path_from_config(*, config_path: Path) -> str:
    """
    功能说明：尽量稳妥地从任务配置中读取 `paths.default_audio_path`。
    参数说明：
    - config_path: 配置文件绝对路径。
    返回值：
    - str: 读到则返回路径文本，否则返回空字符串。
    异常说明：无；读取失败时统一返回空字符串。
    边界条件：优先轻量读取原始 JSON；仅在必要时回退完整配置加载。
    """
    try:
        raw_payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if isinstance(raw_payload, dict):
            paths_data = raw_payload.get("paths", {})
            if isinstance(paths_data, dict):
                default_audio_path_text = str(paths_data.get("default_audio_path", "") or "").strip()
                if default_audio_path_text:
                    return default_audio_path_text
    except Exception:  # noqa: BLE001
        pass

    try:
        config = load_config(config_path)
    except Exception:  # noqa: BLE001
        return ""
    return str(getattr(getattr(config, "paths", None), "default_audio_path", "") or "").strip()


def _append_candidate(candidate_paths: list[Path], candidate_path: Path) -> None:
    """
    功能说明：把路径候选追加到数组，并按大小写不敏感键去重。
    参数说明：
    - candidate_paths: 候选路径数组。
    - candidate_path: 待追加路径。
    返回值：无。
    异常说明：无。
    边界条件：使用 `resolve(strict=False)`，允许保留尚不存在的候选路径文本。
    """
    normalized_candidate = candidate_path.resolve(strict=False)
    candidate_key = str(normalized_candidate).casefold()
    existing_keys = {str(item).casefold() for item in candidate_paths}
    if candidate_key in existing_keys:
        return
    candidate_paths.append(normalized_candidate)


def _looks_like_absolute_path_text(path_text: str) -> bool:
    """
    功能说明：基于原始文本判断路径是否显式表现为绝对/根路径。
    参数说明：
    - path_text: 原始路径文本。
    返回值：
    - bool: 是绝对/根路径文本时返回 True。
    异常说明：无。
    边界条件：在 Windows 上也会把 `\root\foo` 与 `/root/foo` 视为“非工作区相对路径”。
    """
    normalized_path_text = str(path_text).strip()
    if not normalized_path_text:
        return False
    if normalized_path_text.startswith(("/", "\\")):
        return True
    return bool(WINDOWS_DRIVE_ABSOLUTE_PATTERN.match(normalized_path_text))


def is_windows_drive_absolute_path(path_text: str) -> bool:
    """
    功能说明：判断路径文本是否为 Windows 盘符绝对路径（如 ``M:\\foo\\bar``）。
    参数说明：
    - path_text: 原始路径文本。
    返回值：
    - bool: 匹配盘符模式时返回 True。
    边界条件：不匹配以 ``/`` 或 ``\\`` 开头的 Linux/POSIX 绝对路径。
    """
    normalized = str(path_text).strip()
    if not normalized:
        return False
    return bool(WINDOWS_DRIVE_ABSOLUTE_PATTERN.match(normalized))


def remap_windows_absolute_path(*, workspace_root: Path, path_text: str) -> Path | None:
    """
    功能说明：将 Windows 盘符绝对路径回映射到当前工作区下的等价路径，
              若非 Windows 盘符路径则返回 None。
    参数说明：
    - workspace_root: 当前 Linux 工作区根目录。
    - path_text: 原始路径文本（可能来自 DB 中的 Windows 路径）。
    返回值：
    - Path | None: 回映射后的工作区绝对路径，或 None（无法处理时）。
    边界条件：若找不到已知项目目录标记（configs/resources/runs），
              则退化为取文件名拼接 workspace_root。
    """
    normalized = str(path_text).strip()
    if not normalized:
        return None
    if not is_windows_drive_absolute_path(normalized):
        return None
    # 统一分隔符以便后续匹配
    normalized = normalized.replace("\\", "/")
    # 按优先级查找已知项目目录标记
    for marker in ("configs", "resources", "runs"):
        marker_pattern = f"/{marker}/"
        if marker_pattern in normalized:
            marker_index = normalized.index(marker_pattern)
            # 取标记及之后的部分作为工作区相对路径
            relative_path = normalized[marker_index + 1:]
            return (workspace_root / relative_path).resolve()
    # 无已知标记：纯盘符路径（如 G:/ComfyUI）不存在于工作区内，返回 None
    # 让调用方按自身逻辑继续解析（如回退到原绝对路径或扫描磁盘候选）
    logger.warning(
        "Windows 盘符路径 remap 失败：路径 %r 不含已知项目标记（configs/resources/runs），"
        "无法映射到工作区 %s",
        normalized,
        workspace_root,
    )
    return None


def resolve_workspace_path(*, workspace_root: Path, path_text: str) -> Path:
    """
    功能说明：将来自状态库的路径（可能为 Windows 盘符绝对路径或 Linux 风格绝对路径）
              回映射到当前工作区下的等价绝对路径。
    参数说明：
    - workspace_root: 当前工作区根目录。
    - path_text: 原始路径文本（可能来自 DB 中的 config_path 或 audio_path）。
    返回值：
    - Path: 解析后的工作区绝对路径。
    边界条件：先尝试 Windows 盘符绝对路径映射；失败后尝试 Linux 风格绝对路径
               （以 / 或 \\ 开头）的已知标记映射；均不匹配时退化为工作区相对路径。
    """
    normalized = str(path_text).strip()
    if not normalized:
        return workspace_root.resolve()
    # 1) Windows 盘符绝对路径 remap
    remapped = remap_windows_absolute_path(workspace_root=workspace_root, path_text=normalized)
    if remapped is not None:
        return remapped
    # 2) Linux 风格绝对路径（在 Windows 上 is_absolute 为 False）
    if normalized.startswith(("/", "\\")):
        unified = normalized.replace("\\", "/")
        for marker in ("configs", "resources", "runs"):
            marker_pattern = f"/{marker}/"
            if marker_pattern in unified:
                marker_index = unified.index(marker_pattern)
                relative_path = unified[marker_index + 1:]
                return (workspace_root / relative_path).resolve()
        # 无已知标记：去掉前导分隔符作为工作区相对路径
        stripped = normalized.lstrip("/\\")
        if stripped:
            return (workspace_root / stripped).resolve()
    # 3) 其他路径：作为相对路径解析
    return (workspace_root / normalized).resolve()


def resolve_comfyui_root_dir(*, workspace_root: Path, root_dir_raw: str) -> Path:
    """
    功能说明：跨平台解析 ComfyUI 根目录。
    参数说明：
    - workspace_root: 项目工作区根目录。
    - root_dir_raw: 配置中的 root_dir 原始值（相对或绝对路径）。
    返回值：
    - Path: 解析后的 ComfyUI 根目录绝对路径。
    边界条件：
    - Linux: root_dir_raw 通常为相对路径 "ComfyUI"，相对于 workspace_root 解析。
    - Windows: 如果相对路径不存在，自动回退到 G:/ComfyUI。
    """
    # 第 1 步：尝试 Windows 盘符路径到 Linux 工作区的回映射
    remapped = remap_windows_absolute_path(workspace_root=workspace_root, path_text=root_dir_raw)
    if remapped is not None:
        # 验证 remap 结果：必须是有效的 ComfyUI 目录（含 output 子目录）
        # 防止 "G:/ComfyUI" 被错误映射到 workspace_root/ComfyUI
        if remapped.exists() and (remapped / "output").exists():
            return remapped
        # 无效则继续后续步骤
    # 第 2 步：按相对路径解析，要求 output 子目录存在（确认是真正的 ComfyUI）
    resolved = (workspace_root / root_dir_raw).resolve()
    if resolved.exists() and (resolved / "output").exists():
        return resolved
    # 第 3 步：Windows 回退 — 如果相对路径不存在而我们在 Windows 上，尝试常见位置
    if os.name == "nt":
        win_candidates = [
            Path("C:/ComfyUI"),
            Path("D:/ComfyUI"),
            Path("E:/ComfyUI"),
            Path("F:/ComfyUI"),
            Path("G:/ComfyUI"),
        ]
        for candidate in win_candidates:
            if candidate.exists():                return candidate.resolve()
    return resolved
