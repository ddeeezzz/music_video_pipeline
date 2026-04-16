"""
文件用途：统一管理交互式 CLI 展示与输入。
核心流程：渲染菜单 -> 校验输入 -> 返回业务可用选择结果。
输入输出：输入菜单项，输出用户选择的功能或条目。
依赖说明：依赖标准库，无第三方依赖。
维护说明：UI 层不含下载与存储逻辑，仅做交互适配。
"""

# 常量：系列展示顺序。
SERIES_DISPLAY_ORDER = ("15", "xl", "fl")


def prompt_main_action() -> str | None:
    """
    功能说明：提示用户选择主功能。
    参数说明：无。
    返回值：
    - str | None: 返回 sync_lora/sync_base_model/rebind_lora/download_assets；输入 q 时返回 None。
    异常说明：无。
    边界条件：非法输入会循环提示。
    """
    while True:
        print("")
        print("请选择操作：")
        print("  [1] 同步 LoRA")
        print("  [2] 同步 BaseModel")
        print("  [3] 重绑 LoRA key")
        print("  [4] 下载模型资源（HF/直链）")
        answer = input("输入序号继续，输入 q 退出：").strip().lower()

        if answer in {"q", "quit", "exit"}:
            return None
        if answer == "1":
            return "sync_lora"
        if answer == "2":
            return "sync_base_model"
        if answer == "3":
            return "rebind_lora"
        if answer == "4":
            return "download_assets"
        print(f"输入无效：{answer}，请重新输入。")


def render_series_menu(resource_title: str, options: list[dict]) -> None:
    """
    功能说明：按 15/xl/fl 分段渲染资源菜单。
    参数说明：
    - resource_title: 资源标题文本。
    - options: 菜单项数组。
    返回值：无。
    异常说明：无。
    边界条件：系列为空时显示“(空)”。
    """
    print("")
    print(f"远端 {resource_title} 目录列表（输入序号下载，输入 q 退出）")
    print("--------------------------------------------------")
    for series in SERIES_DISPLAY_ORDER:
        print(f"[{series}]")
        group = [item for item in options if str(item.get("series", "")) == series]
        if not group:
            print("  (空)")
            continue
        for item in group:
            downloaded_mark = "（已下载）" if bool(item.get("downloaded", False)) else ""
            display_name = str(item.get("display_name", item.get("name", ""))).strip()
            print(f"  [{item['index']}] {display_name}{downloaded_mark}")
    print("--------------------------------------------------")


def render_lora_binding_menu(options: list[dict]) -> None:
    """
    功能说明：按系列分段渲染 LoRA 绑定重绑菜单。
    参数说明：
    - options: 重绑菜单项数组。
    返回值：无。
    异常说明：无。
    边界条件：系列为空时显示“(空)”。
    """
    print("")
    print("LoRA 绑定列表（输入序号重绑，输入 q 退出）")
    print("--------------------------------------------------")
    for series in SERIES_DISPLAY_ORDER:
        print(f"[{series}]")
        group = [item for item in options if str(item.get("series", "")) == series]
        if not group:
            print("  (空)")
            continue
        for item in group:
            binding_name = str(item.get("binding_name", "")).strip()
            current_key = str(item.get("current_key", "")).strip() or "<未绑定>"
            print(f"  [{item['index']}] {binding_name} -> {current_key}")
    print("--------------------------------------------------")


def prompt_option(options: list[dict], prompt_text: str) -> dict | None:
    """
    功能说明：读取用户输入的菜单序号。
    参数说明：
    - options: 菜单项数组。
    - prompt_text: 输入提示文本。
    返回值：
    - dict | None: 选中项；输入 q 时返回 None。
    异常说明：无。
    边界条件：非法输入会循环提示。
    """
    index_map = {int(item["index"]): item for item in options}
    while True:
        answer = input(prompt_text).strip().lower()
        if answer in {"q", "quit", "exit"}:
            return None
        if not answer.isdigit():
            print(f"输入无效：{answer}，请输入序号或 q。")
            continue

        selected_index = int(answer)
        selected_option = index_map.get(selected_index)
        if selected_option is None:
            print(f"未找到序号 [{selected_index}]，请重新输入。")
            continue
        return selected_option


def prompt_base_model_key(candidates: list[dict]) -> str | None:
    """
    功能说明：让用户选择绑定目标底模 key。
    参数说明：
    - candidates: 候选底模数组。
    返回值：
    - str | None: 选中的 key；输入 q 时返回 None。
    异常说明：无。
    边界条件：候选为空时返回 None。
    """
    if not candidates:
        return None

    print("")
    print("请选择绑定目标底模 key：")
    for idx, item in enumerate(candidates, start=1):
        key_text = str(item.get("key", "")).strip()
        path_text = str(item.get("path", "")).strip()
        print(f"  [{idx}] {key_text} -> {path_text}")

    while True:
        answer = input("输入序号确认，输入 q 取消本次操作：").strip().lower()
        if answer in {"q", "quit", "exit"}:
            return None
        if not answer.isdigit():
            print(f"输入无效：{answer}，请输入序号或 q。")
            continue

        index = int(answer)
        if index < 1 or index > len(candidates):
            print(f"序号越界：{index}，请重新输入。")
            continue
        return str(candidates[index - 1].get("key", "")).strip()


def prompt_download_action() -> str | None:
    """
    功能说明：提示用户选择下载子菜单功能。
    参数说明：无。
    返回值：
    - str | None: 返回 download_lora_direct/download_base_repo/download_base_direct；输入 q 时返回 None。
    异常说明：无。
    边界条件：非法输入会循环提示。
    """
    while True:
        print("")
        print("下载模型资源：")
        print("  [1] LoRA 直链下载并绑定")
        print("  [2] BaseModel HF 仓库下载")
        print("  [3] BaseModel 直链下载")
        answer = input("输入序号继续，输入 q 返回上级菜单：").strip().lower()
        if answer in {"q", "quit", "exit"}:
            return None
        if answer == "1":
            return "download_lora_direct"
        if answer == "2":
            return "download_base_repo"
        if answer == "3":
            return "download_base_direct"
        print(f"输入无效：{answer}，请重新输入。")


def prompt_model_series(title_text: str = "请选择模型系列") -> str | None:
    """
    功能说明：提示用户选择模型系列。
    参数说明：
    - title_text: 提示标题文本。
    返回值：
    - str | None: 返回 15/xl/fl；输入 q 时返回 None。
    异常说明：无。
    边界条件：非法输入会循环提示。
    """
    while True:
        print("")
        print(title_text)
        print("  [1] 15")
        print("  [2] xl")
        print("  [3] fl")
        answer = input("输入序号确认，输入 q 取消：").strip().lower()
        if answer in {"q", "quit", "exit"}:
            return None
        if answer == "1":
            return "15"
        if answer == "2":
            return "xl"
        if answer == "3":
            return "fl"
        print(f"输入无效：{answer}，请重新输入。")


def prompt_text(prompt_label: str, default_value: str = "") -> str | None:
    """
    功能说明：读取单行文本输入并支持默认值与取消。
    参数说明：
    - prompt_label: 输入提示。
    - default_value: 默认值（回车采用）。
    返回值：
    - str | None: 输入文本；输入 q 时返回 None。
    异常说明：无。
    边界条件：当默认值为空时，空输入会继续提示。
    """
    while True:
        answer = input(f"{prompt_label}：").strip()
        lowered = answer.lower()
        if lowered in {"q", "quit", "exit"}:
            return None
        if answer:
            return answer
        if default_value:
            return str(default_value)
        print("输入不能为空，请重新输入。")


def prompt_confirm(prompt_label: str, default_no: bool = True) -> bool:
    """
    功能说明：读取布尔确认输入。
    参数说明：
    - prompt_label: 提示文本。
    - default_no: True 表示默认否；False 表示默认是。
    返回值：
    - bool: 用户是否确认。
    异常说明：无。
    边界条件：空输入时采用默认值。
    """
    suffix = "[y/N]" if default_no else "[Y/n]"
    while True:
        answer = input(f"{prompt_label} {suffix}：").strip().lower()
        if not answer:
            return not default_no
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print(f"输入无效：{answer}，请输入 y 或 n。")
