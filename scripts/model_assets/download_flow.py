"""
文件用途：提供 model_assets 交互式下载流程（LoRA直链、BaseModel仓库、BaseModel直链）。
核心流程：读取交互输入 -> 下载资源 -> 校验结果 -> 写入注册表或绑定清单。
输入输出：输入用户交互内容与配置路径，输出下载摘要与持久化结果。
依赖说明：依赖本包 download_engine/store/sync_base_model/ui。
维护说明：该模块聚合下载业务流程，不处理 bypy 远端目录索引。
"""

# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于目录删除
import shutil

try:
    # 包内导入：下载引擎
    from .download_engine import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_WAIT_SECONDS,
        DEFAULT_TIMEOUT_SECONDS,
        DownloadEngineError,
        download_file,
        download_repo_with_hf_cli,
        ensure_non_empty_file,
        has_any_file,
        infer_filename_from_url,
        infer_repo_name,
    )
    # 包内导入：BaseModel key 规则
    from .sync_base_model import build_base_model_key
    # 包内导入：存储工具
    from .store import (
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        now_iso_seconds,
        resolve_path,
        to_project_relative_path,
        upsert_base_model,
        upsert_lora_binding,
        validate_and_resolve_base_model,
        write_json,
    )
    # 包内导入：交互 UI
    from .ui import (
        prompt_base_model_key,
        prompt_confirm,
        prompt_download_action,
        prompt_model_series,
        prompt_text,
    )
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from download_engine import (  # type: ignore
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_WAIT_SECONDS,
        DEFAULT_TIMEOUT_SECONDS,
        DownloadEngineError,
        download_file,
        download_repo_with_hf_cli,
        ensure_non_empty_file,
        has_any_file,
        infer_filename_from_url,
        infer_repo_name,
    )
    from sync_base_model import build_base_model_key  # type: ignore
    from store import (  # type: ignore
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        now_iso_seconds,
        resolve_path,
        to_project_relative_path,
        upsert_base_model,
        upsert_lora_binding,
        validate_and_resolve_base_model,
        write_json,
    )
    from ui import (  # type: ignore
        prompt_base_model_key,
        prompt_confirm,
        prompt_download_action,
        prompt_model_series,
        prompt_text,
    )


# 常量：各系列默认 Hugging Face 仓库（空值表示必须手动输入）。
DEFAULT_REPO_BY_SERIES = {
    "15": "runwayml/stable-diffusion-v1-5",
    "xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "fl": "",
}
# 常量：默认 revision。
DEFAULT_REPO_REVISION = "main"
# 常量：支持的系列枚举。
VALID_MODEL_SERIES = ("15", "xl", "fl")
# 常量：XL 过滤下载包含规则（减少 full/openvino/示例资源混杂）。
XL_INCLUDE_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def _normalize_dir_name(name: str, fallback_name: str) -> str:
    """
    功能说明：规范化目录名，避免路径越界字符。
    参数说明：
    - name: 用户输入名称。
    - fallback_name: 兜底名称。
    返回值：
    - str: 规范化后的目录名。
    异常说明：无。
    边界条件：会替换斜杠与反斜杠。
    """
    text = str(name).strip().replace("/", "_").replace("\\", "_")
    return text or fallback_name


def _select_base_model_key(project_root: Path, base_registry_path: Path, model_series: str) -> str | None:
    """
    功能说明：为目标系列选择底模 key。
    参数说明：
    - project_root: 项目根目录。
    - base_registry_path: 底模注册表路径。
    - model_series: 目标系列。
    返回值：
    - str | None: 选中的 key；取消时返回 None。
    异常说明：
    - RuntimeError: 未找到可用候选时抛出。
    边界条件：候选仅一个时自动选择。
    """
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    candidates = get_enabled_base_model_candidates(registry_data=registry_data, model_series=model_series)
    if not candidates:
        raise RuntimeError(f"系列 {model_series} 未找到可用底模 key，请先补充底模注册表")
    if len(candidates) == 1:
        return str(candidates[0].get("key", "")).strip()
    return prompt_base_model_key(candidates=candidates)


def run_download_assets_flow(project_root: Path, logger, base_registry_path: Path, bindings_path: Path) -> int:
    """
    功能说明：运行模型资源下载子菜单流程。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - base_registry_path: 底模注册表路径。
    - bindings_path: LoRA 绑定清单路径。
    返回值：
    - int: 0 表示正常退出。
    异常说明：无（单次异常提示后继续循环）。
    边界条件：输入 q 时退出下载子菜单并返回主流程。
    """
    while True:
        action = prompt_download_action()
        if action is None:
            print("已退出下载子菜单。")
            return 0

        try:
            if action == "download_lora_direct":
                run_lora_direct_download_once(
                    project_root=project_root,
                    logger=logger,
                    base_registry_path=base_registry_path,
                    bindings_path=bindings_path,
                )
                continue

            if action == "download_base_repo":
                run_base_model_repo_download_once(
                    project_root=project_root,
                    logger=logger,
                    base_registry_path=base_registry_path,
                )
                continue

            run_base_model_direct_download_once(
                project_root=project_root,
                logger=logger,
                base_registry_path=base_registry_path,
            )
        except Exception as error:  # noqa: BLE001
            logger.error("下载流程执行失败：%s", error)
            print(f"下载失败：{error}")


def run_lora_direct_download_once(
    project_root: Path,
    logger,
    base_registry_path: Path,
    bindings_path: Path,
) -> None:
    """
    功能说明：执行一次 LoRA 直链下载并自动绑定底模。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - base_registry_path: 底模注册表路径。
    - bindings_path: LoRA 绑定清单路径。
    返回值：无。
    异常说明：
    - RuntimeError/DownloadEngineError: 输入或下载失败时抛出。
    边界条件：取消输入时直接返回，不抛异常。
    """
    model_series = prompt_model_series(title_text="请选择 LoRA 系列")
    if model_series is None:
        print("已取消本次 LoRA 直链下载。")
        return

    url = prompt_text(prompt_label="输入 LoRA 直链 URL（输入 q 取消）", default_value="")
    if url is None:
        print("已取消本次 LoRA 直链下载。")
        return

    filename = infer_filename_from_url(url)
    lora_filename = filename if filename.lower().endswith(".safetensors") else f"{Path(filename).stem or 'downloaded_lora'}.safetensors"
    default_binding_name = Path(lora_filename).stem or "downloaded_lora"
    binding_name_input = prompt_text(
        prompt_label=f"输入 binding 名称（默认 {default_binding_name}，输入 q 取消）",
        default_value=default_binding_name,
    )
    if binding_name_input is None:
        print("已取消本次 LoRA 直链下载。")
        return

    binding_name = _normalize_dir_name(binding_name_input, fallback_name=default_binding_name)
    series_dir = (project_root / "models" / "lora" / model_series).resolve()
    series_dir.mkdir(parents=True, exist_ok=True)
    target_dir = (series_dir / binding_name).resolve()
    if not target_dir.is_relative_to(series_dir):
        raise RuntimeError(f"LoRA 目标目录越界：{target_dir}")

    if target_dir.exists():
        confirmed = prompt_confirm("LoRA 目标目录已存在，是否覆盖删除后继续？", default_no=True)
        if not confirmed:
            print("已取消本次 LoRA 直链下载。")
            return
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_key = _select_base_model_key(
        project_root=project_root,
        base_registry_path=base_registry_path,
        model_series=model_series,
    )
    if not selected_key:
        print("已取消本次 LoRA 直链下载。")
        return

    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    _, base_model_path = validate_and_resolve_base_model(
        registry_data=registry_data,
        base_model_key=selected_key,
        model_series=model_series,
        project_root=project_root,
    )

    lora_local_path = (target_dir / lora_filename).resolve()
    download_file(
        url=url,
        save_path=lora_local_path,
        logger=logger,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_wait_seconds=DEFAULT_RETRY_WAIT_SECONDS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    ensure_non_empty_file(lora_local_path)

    bindings_data = load_or_init_lora_bindings(path=bindings_path)
    record = {
        "binding_name": binding_name,
        "model_series": model_series,
        "remote_dir": f"direct_url:{url}",
        "lora_file": to_project_relative_path(path=lora_local_path, project_root=project_root),
        "meta_file": "",
        "base_model_key": selected_key,
        "base_model_path": to_project_relative_path(path=base_model_path, project_root=project_root),
        "meta_base_model_text": "",
        "updated_at": now_iso_seconds(),
    }
    action = upsert_lora_binding(bindings_data=bindings_data, new_record=record)
    write_json(path=bindings_path, data=bindings_data)

    logger.info("LoRA 直链绑定写入完成，action=%s，binding=%s", action, binding_name)
    print(f"下载并绑定完成：[{model_series}] {binding_name} -> {lora_local_path}")


def run_base_model_repo_download_once(
    project_root: Path,
    logger,
    base_registry_path: Path,
) -> None:
    """
    功能说明：执行一次 BaseModel 仓库下载并更新注册表。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - base_registry_path: 底模注册表路径。
    返回值：无。
    异常说明：
    - RuntimeError/DownloadEngineError: 输入或下载失败时抛出。
    边界条件：取消输入时直接返回，不抛异常。
    """
    model_series = prompt_model_series(title_text="请选择 BaseModel 系列（HF 仓库）")
    if model_series is None:
        print("已取消本次 BaseModel 仓库下载。")
        return

    default_repo = DEFAULT_REPO_BY_SERIES.get(model_series, "")
    repo_id = prompt_text(
        prompt_label=(
            f"输入 HF repo_id（默认 {default_repo or '无'}，输入 q 取消）"
            if default_repo
            else "输入 HF repo_id（fl 系列无默认值，输入 q 取消）"
        ),
        default_value=default_repo,
    )
    if repo_id is None:
        print("已取消本次 BaseModel 仓库下载。")
        return

    repo_id = str(repo_id).strip()
    if not repo_id:
        raise RuntimeError(f"系列 {model_series} 必须输入 repo_id")

    revision = prompt_text(
        prompt_label=f"输入 revision（默认 {DEFAULT_REPO_REVISION}，输入 q 取消）",
        default_value=DEFAULT_REPO_REVISION,
    )
    if revision is None:
        print("已取消本次 BaseModel 仓库下载。")
        return
    revision = str(revision).strip() or DEFAULT_REPO_REVISION

    repo_name = infer_repo_name(repo_id=repo_id)
    local_dir = (project_root / "models" / "base_model" / model_series / "diffusers" / repo_name).resolve()
    if local_dir.exists():
        confirmed = prompt_confirm("BaseModel 目标目录已存在，是否覆盖删除后继续？", default_no=True)
        if not confirmed:
            print("已取消本次 BaseModel 仓库下载。")
            return
        shutil.rmtree(local_dir)

    include_patterns: tuple[str, ...] = ()
    if model_series == "xl":
        include_patterns = XL_INCLUDE_PATTERNS
        logger.info("已启用 XL 仓库过滤下载策略，仅保留推理必需文件集合。")

    download_repo_with_hf_cli(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        logger=logger,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_wait_seconds=DEFAULT_RETRY_WAIT_SECONDS,
        include_patterns=include_patterns,
    )

    if not has_any_file(local_dir):
        raise DownloadEngineError(f"仓库下载结果目录为空：{local_dir}")

    key_text = build_base_model_key(model_series=model_series, model_format="diffusers", model_name=repo_name)
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    record = {
        "key": key_text,
        "series": model_series,
        "format": "diffusers",
        "path": to_project_relative_path(path=local_dir, project_root=project_root),
        "type": "directory",
        "enabled": True,
        "description": f"HF 仓库下载基础模型目录：{repo_id}@{revision}",
    }
    action = upsert_base_model(registry_data=registry_data, new_record=record)
    write_json(path=base_registry_path, data=registry_data)

    logger.info("BaseModel 注册表写入完成，action=%s，key=%s", action, key_text)
    print(f"仓库下载完成：[{model_series}] {repo_id}@{revision} -> {local_dir}")


def run_base_model_direct_download_once(
    project_root: Path,
    logger,
    base_registry_path: Path,
) -> None:
    """
    功能说明：执行一次 BaseModel 直链下载并更新注册表。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - base_registry_path: 底模注册表路径。
    返回值：无。
    异常说明：
    - RuntimeError/DownloadEngineError: 输入或下载失败时抛出。
    边界条件：取消输入时直接返回，不抛异常。
    """
    model_series = prompt_model_series(title_text="请选择 BaseModel 系列（直链）")
    if model_series is None:
        print("已取消本次 BaseModel 直链下载。")
        return

    url = prompt_text(prompt_label="输入 BaseModel 直链 URL（输入 q 取消）", default_value="")
    if url is None:
        print("已取消本次 BaseModel 直链下载。")
        return

    default_filename = infer_filename_from_url(url)
    default_model_name = Path(default_filename).stem or "downloaded_base_model"
    default_save_rel = f"models/base_model/{model_series}/single/{default_model_name}/{default_filename}"
    save_path_text = prompt_text(
        prompt_label=f"输入保存路径（相对项目根，默认 {default_save_rel}，输入 q 取消）",
        default_value=default_save_rel,
    )
    if save_path_text is None:
        print("已取消本次 BaseModel 直链下载。")
        return

    target_path = resolve_path(project_root=project_root, raw_path=save_path_text)
    single_series_dir = (project_root / "models" / "base_model" / model_series / "single").resolve()
    single_series_dir.mkdir(parents=True, exist_ok=True)
    if not target_path.is_relative_to(single_series_dir):
        raise RuntimeError(f"保存路径必须位于 {single_series_dir} 下，当前路径={target_path}")

    if target_path.exists():
        confirmed = prompt_confirm("目标文件已存在，是否覆盖下载？", default_no=True)
        if not confirmed:
            print("已取消本次 BaseModel 直链下载。")
            return
        if target_path.is_file():
            target_path.unlink()
        else:
            shutil.rmtree(target_path)

    download_file(
        url=url,
        save_path=target_path,
        logger=logger,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_wait_seconds=DEFAULT_RETRY_WAIT_SECONDS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    ensure_non_empty_file(target_path)

    local_dir = target_path.parent
    key_text = build_base_model_key(model_series=model_series, model_format="single", model_name=local_dir.name)
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    record = {
        "key": key_text,
        "series": model_series,
        "format": "single",
        "path": to_project_relative_path(path=local_dir, project_root=project_root),
        "type": "directory",
        "enabled": True,
        "description": f"直链下载基础模型目录：{target_path.name}",
    }
    action = upsert_base_model(registry_data=registry_data, new_record=record)
    write_json(path=base_registry_path, data=registry_data)

    logger.info("BaseModel 注册表写入完成，action=%s，key=%s", action, key_text)
    print(f"直链下载完成：[{model_series}] {target_path}")
