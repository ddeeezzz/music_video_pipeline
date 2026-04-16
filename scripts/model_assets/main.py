"""
文件用途：model_assets 交互入口，统一管理同步、下载与 LoRA 重绑能力。
核心流程：初始化参数与日志 -> 选择功能 -> 菜单选择 -> 执行动作并写配置。
输入输出：输入命令行可选参数与交互输入，输出执行结果与错误提示。
依赖说明：依赖本包 bypy_client/indexer/store/sync_lora/sync_base_model/rebind_lora/download_flow/ui。
维护说明：本入口仅支持交互模式，不提供非交互批量参数模式。
"""

# 标准库：用于参数解析
import argparse

try:
    # 包内导入：bypy 客户端
    from .bypy_client import BypyClient
    # 包内导入：远端索引
    from .indexer import build_remote_options
    # 包内导入：LoRA 重绑逻辑
    from .rebind_lora import apply_lora_rebind, build_lora_binding_options, get_rebind_candidates
    # 包内导入：下载流程
    from .download_flow import run_download_assets_flow
    # 包内导入：BaseModel 同步
    from .sync_base_model import sync_base_model_item
    # 包内导入：LoRA 同步
    from .sync_lora import sync_lora_item
    # 包内导入：存储工具
    from .store import (
        DEFAULT_BASE_REGISTRY_PATH,
        DEFAULT_BINDINGS_PATH,
        DEFAULT_LOG_PATH,
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        resolve_path,
        resolve_project_root,
        setup_logger,
        write_json,
    )
    # 包内导入：交互 UI
    from .ui import (
        prompt_base_model_key,
        prompt_main_action,
        prompt_option,
        render_lora_binding_menu,
        render_series_menu,
    )
except ImportError:
    # 兼容脚本直跑：同目录模块导入
    from bypy_client import BypyClient
    from indexer import build_remote_options
    from rebind_lora import apply_lora_rebind, build_lora_binding_options, get_rebind_candidates
    from download_flow import run_download_assets_flow
    from sync_base_model import sync_base_model_item
    from sync_lora import sync_lora_item
    from store import (
        DEFAULT_BASE_REGISTRY_PATH,
        DEFAULT_BINDINGS_PATH,
        DEFAULT_LOG_PATH,
        get_enabled_base_model_candidates,
        load_or_init_base_registry,
        load_or_init_lora_bindings,
        resolve_path,
        resolve_project_root,
        setup_logger,
        write_json,
    )
    from ui import (
        prompt_base_model_key,
        prompt_main_action,
        prompt_option,
        render_lora_binding_menu,
        render_series_menu,
    )


def build_parser() -> argparse.ArgumentParser:
    """
    功能说明：构建命令行解析器。
    参数说明：无。
    返回值：
    - argparse.ArgumentParser: 已配置解析器。
    异常说明：无。
    边界条件：仅提供交互模式相关可选参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "统一管理 LoRA/BaseModel 的 bypy 同步、HF/直链下载与 LoRA 重绑（仅交互模式）。"
            "示例：uv run --no-sync model_assets"
        )
    )
    parser.add_argument(
        "--log-path",
        "--log_path",
        dest="log_path",
        default=DEFAULT_LOG_PATH,
        help="日志路径（可选），默认 log/model_assets.log",
    )
    return parser


def select_base_model_key_for_lora(
    project_root,
    base_registry_path,
    model_series: str,
) -> str:
    """
    功能说明：为 LoRA 同步选择底模 key。
    参数说明：
    - project_root: 项目根目录。
    - base_registry_path: 底模注册表路径。
    - model_series: LoRA 所属系列。
    返回值：
    - str: 选中的底模 key。
    异常说明：
    - RuntimeError: 没有可用底模或用户取消选择时抛出。
    边界条件：当候选仅 1 个时自动选择。
    """
    registry_data = load_or_init_base_registry(path=base_registry_path, project_root=project_root)
    candidates = get_enabled_base_model_candidates(registry_data=registry_data, model_series=model_series)
    if not candidates:
        raise RuntimeError(f"系列 {model_series} 未找到可用底模 key，请先补充底模注册表")

    if len(candidates) == 1:
        return str(candidates[0].get("key", "")).strip()

    selected_key = prompt_base_model_key(candidates=candidates)
    if not selected_key:
        raise RuntimeError("已取消本次 LoRA 下载")
    return selected_key


def run_sync_flow(
    resource_type: str,
    project_root,
    logger,
    client: BypyClient,
    base_registry_path,
    bindings_path,
) -> int:
    """
    功能说明：执行同步流程（LoRA 或 BaseModel）。
    参数说明：
    - resource_type: lora 或 base_model。
    - project_root: 项目根目录。
    - logger: 日志对象。
    - client: bypy 客户端。
    - base_registry_path: 底模注册表路径。
    - bindings_path: LoRA 绑定清单路径。
    返回值：
    - int: 0 表示正常退出。
    异常说明：无（单次失败会提示后继续循环）。
    边界条件：输入 q 退出。
    """
    resource_title = "LoRA" if resource_type == "lora" else "BaseModel"
    while True:
        options = build_remote_options(
            resource_type=resource_type,
            project_root=project_root,
            client=client,
            logger=logger,
        )
        if not options:
            print(f"远端 {resource_title} 暂无可下载目录，请稍后再试。")
            return 0

        render_series_menu(resource_title=resource_title, options=options)
        selected = prompt_option(options, prompt_text="输入序号下载/覆盖，输入 q 退出：")
        if selected is None:
            print("已退出。")
            return 0

        try:
            if resource_type == "lora":
                selected_key = select_base_model_key_for_lora(
                    project_root=project_root,
                    base_registry_path=base_registry_path,
                    model_series=str(selected.get("series", "")).strip(),
                )
                result = sync_lora_item(
                    project_root=project_root,
                    logger=logger,
                    client=client,
                    option=selected,
                    base_model_key=selected_key,
                    base_registry_path=base_registry_path,
                    bindings_path=bindings_path,
                )
                print(
                    f"同步完成：[{result['model_series']}] {result['name']} -> {result['local_dir']}"
                    f"（binding {result['action']}）"
                )
            else:
                result = sync_base_model_item(
                    project_root=project_root,
                    logger=logger,
                    client=client,
                    option=selected,
                    base_registry_path=base_registry_path,
                )
                print(
                    f"同步完成：[{result['model_series']}] {result['name']} -> {result['local_dir']}"
                    f"（registry key={result['key']}，{result['action']}）"
                )
        except Exception as step_error:  # noqa: BLE001
            logger.error("本次同步失败：%s", step_error)
            print(f"同步失败：{step_error}")
            continue


def run_lora_rebind_flow(project_root, logger, base_registry_path, bindings_path) -> int:
    """
    功能说明：执行 LoRA 单条重绑流程。
    参数说明：
    - project_root: 项目根目录。
    - logger: 日志对象。
    - base_registry_path: 底模注册表路径。
    - bindings_path: LoRA 绑定清单路径。
    返回值：
    - int: 0 表示正常退出。
    异常说明：无（单次失败会提示后继续循环）。
    边界条件：输入 q 退出。
    """
    while True:
        bindings_data = load_or_init_lora_bindings(path=bindings_path)
        options = build_lora_binding_options(bindings_data=bindings_data)
        if not options:
            print("当前没有可重绑的 LoRA 绑定记录，请先完成至少一次 LoRA 同步。")
            return 0

        render_lora_binding_menu(options=options)
        selected = prompt_option(options, prompt_text="输入序号重绑，输入 q 退出：")
        if selected is None:
            print("已退出。")
            return 0

        model_series = str(selected.get("series", "")).strip()
        candidates = get_rebind_candidates(
            project_root=project_root,
            base_registry_path=base_registry_path,
            model_series=model_series,
        )
        if not candidates:
            print(f"系列 {model_series} 没有可用底模 key，请先检查底模注册表。")
            continue

        selected_key = prompt_base_model_key(candidates=candidates)
        if not selected_key:
            print("已取消本次重绑。")
            continue

        try:
            result = apply_lora_rebind(
                project_root=project_root,
                bindings_data=bindings_data,
                option=selected,
                base_registry_path=base_registry_path,
                new_base_model_key=selected_key,
            )
            write_json(path=bindings_path, data=bindings_data)
            logger.info(
                "LoRA 重绑完成：binding=%s，old_key=%s，new_key=%s",
                result["binding_name"],
                result["old_base_model_key"],
                result["new_base_model_key"],
            )
            print(
                f"重绑完成：[{result['model_series']}] {result['binding_name']} "
                f"{result['old_base_model_key']} -> {result['new_base_model_key']}"
            )
        except Exception as step_error:  # noqa: BLE001
            logger.error("本次重绑失败：%s", step_error)
            print(f"重绑失败：{step_error}")
            continue


def main() -> int:
    """
    功能说明：脚本主入口。
    参数说明：无（参数从命令行读取）。
    返回值：
    - int: 0 表示成功退出，1 表示异常退出。
    异常说明：无（异常统一记录日志后返回 1）。
    边界条件：菜单中的 q 将以 0 退出。
    """
    args = build_parser().parse_args()
    project_root = resolve_project_root()

    log_path = resolve_path(project_root=project_root, raw_path=str(args.log_path))
    logger = setup_logger(log_path=log_path)

    base_registry_path = resolve_path(project_root=project_root, raw_path=DEFAULT_BASE_REGISTRY_PATH)
    bindings_path = resolve_path(project_root=project_root, raw_path=DEFAULT_BINDINGS_PATH)

    client = BypyClient(logger=logger)

    logger.info("模型资源下载与绑定，项目根目录=%s，日志路径=%s", project_root, log_path)

    try:
        action = prompt_main_action()
        if action is None:
            print("已退出。")
            return 0

        if action == "sync_lora":
            return run_sync_flow(
                resource_type="lora",
                project_root=project_root,
                logger=logger,
                client=client,
                base_registry_path=base_registry_path,
                bindings_path=bindings_path,
            )

        if action == "sync_base_model":
            return run_sync_flow(
                resource_type="base_model",
                project_root=project_root,
                logger=logger,
                client=client,
                base_registry_path=base_registry_path,
                bindings_path=bindings_path,
            )

        if action == "download_assets":
            return run_download_assets_flow(
                project_root=project_root,
                logger=logger,
                base_registry_path=base_registry_path,
                bindings_path=bindings_path,
            )

        return run_lora_rebind_flow(
            project_root=project_root,
            logger=logger,
            base_registry_path=base_registry_path,
            bindings_path=bindings_path,
        )
    except Exception as error:  # noqa: BLE001
        logger.error("执行失败：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
