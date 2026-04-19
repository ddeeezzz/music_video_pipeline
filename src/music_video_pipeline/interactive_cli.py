"""
文件用途：提供 mvpl 交互式命令行体验（无参进入）。
核心流程：菜单选择 -> 参数采集 -> 预览确认 -> 执行 -> 结果/失败恢复。
输入输出：输入用户键盘交互，输出执行摘要与下一步建议。
依赖说明：依赖标准库与 command_service 的 CommandRequest。
维护说明：本模块只负责交互编排，不承载 Pipeline 业务逻辑。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Callable

from music_video_pipeline.command_service import CommandRequest
from music_video_pipeline.constants import MODULE_ORDER
from music_video_pipeline.state_store import StateStore


_BACK = object()

ExecuteRequest = Callable[[CommandRequest], dict]


@dataclass(slots=True)
class SessionMemory:
    """交互会话默认值缓存。"""

    task_id: str = ""
    config_path: str = ""
    audio_path: str = ""


@dataclass(frozen=True, slots=True)
class MenuAction:
    """交互菜单动作定义。"""

    key: str
    title: str
    advanced: bool = False


ACTION_HINTS: dict[str, tuple[str, ...]] = {
    "run": (
        "适合首次执行全链路；若要覆盖音频可在参数里填写 --audio-path。",
        "建议使用与首次一致的 config，避免同一 task_id 行为漂移。",
        "若携带 --force-module，会重置该模块及下游状态并重建后续产物。",
    ),
    "resume": (
        "用于从断点继续；默认按当前状态跳过已完成模块。",
        "若需从某模块重跑，请在高级菜单使用 force 版本。",
        "强制恢复会覆盖目标模块及下游已有产物，任务状态会回退后再推进。",
    ),
    "run-module": (
        "仅执行单模块调试；不会自动替代完整全链路运行。",
        "首次初始化任务时可补充 audio_path，已有任务通常可留空。",
        "若启用 --force，会重置当前模块及下游状态再执行本模块。",
    ),
    "b-task-status": (
        "用于先看模块B失败/运行中单元，再决定是否重试。",
    ),
    "c-task-status": (
        "用于先看模块C失败/运行中单元，再决定是否重试。",
    ),
    "d-task-status": (
        "用于先看模块D失败/运行中单元，再决定是否重试。",
    ),
    "bcd-task-status": (
        "用于跨模块排障总览，先定位阻塞点再执行重试。",
    ),
    "monitor": (
        "仅手动启动监督服务，不会触发模块重跑。",
    ),
    "run-force": (
        "会从选定模块开始重跑并影响下游状态。",
        "任务由状态库选择，不再手填 task_id/config。",
        "已完成的下游模块会被置回待执行状态，相关历史产物可能被新结果替换。",
    ),
    "resume-force": (
        "在恢复模式下强制从指定模块重启执行。",
        "适合产物可能过期或中间状态不可信的场景。",
        "执行后会改写目标模块及其下游状态记录，历史结果将不再作为当前有效产物。",
    ),
    "run-module-force": (
        "会重置当前模块及下游后再执行本模块。",
        "用于单模块纠偏；执行前请确认目标模块选择正确。",
        "若模块选择错误，可能触发不必要的下游重建并覆盖现有结果。",
    ),
    "b-retry-segment": (
        "仅重试模块B指定 segment；不会自动执行 C/D。",
        "重试后通常需要再触发 run/resume 继续下游。",
        "该 segment 的 B 输出会被刷新，后续依赖该输出的链路需要重新衔接。",
    ),
    "c-retry-shot": (
        "仅重试模块C指定 shot，成功后会触发 D 重建。",
        "执行前建议先看 c-task-status 确认 shot_id。",
        "D 层对应片段会随新的 C 结果重建，最终合成内容会发生更新。",
    ),
    "d-retry-shot": (
        "仅重试模块D指定 shot，不重跑 A/B/C。",
        "适合最终渲染失败或局部素材异常场景。",
        "该 shot 的终稿片段会被替换，最终视频输出可能出现局部差异。",
    ),
    "bcd-retry-segment": (
        "从 B 指定 segment 贯通重试到 C/D。",
        "适合同一 segment 在 B/C/D 链路联动异常时使用。",
        "同一 segment 在三层链路的中间与终端产物都会被重算并覆盖。",
    ),
}


@dataclass(frozen=True, slots=True)
class RerunTaskContext:
    """重跑任务选择结果。"""

    db_path: Path
    task_id: str
    config_path: Path
    audio_path: Path
    task_status: str
    updated_at: str
    module_status_map: dict[str, str]


MAIN_ACTIONS: tuple[MenuAction, ...] = (
    MenuAction("run", "执行全链路运行"),
    MenuAction("resume", "从断点恢复运行"),
    MenuAction("run-module", "执行单模块调试"),
    MenuAction("b-task-status", "查看模块B单元状态"),
    MenuAction("c-task-status", "查看模块C单元状态"),
    MenuAction("d-task-status", "查看模块D单元状态"),
    MenuAction("bcd-task-status", "查看跨模块B/C/D链路状态"),
    MenuAction("monitor", "手动启动任务监督"),
)

ADVANCED_ACTIONS: tuple[MenuAction, ...] = (
    MenuAction("run-force", "run（含 --force-module）", advanced=True),
    MenuAction("resume-force", "resume（含 --force-module）", advanced=True),
    MenuAction("run-module-force", "run-module（含 --force）", advanced=True),
    MenuAction("b-retry-segment", "按 segment 重试模块B", advanced=True),
    MenuAction("c-retry-shot", "按 shot 重试模块C", advanced=True),
    MenuAction("d-retry-shot", "按 shot 重试模块D", advanced=True),
    MenuAction("bcd-retry-segment", "按 segment 重试跨模块B/C/D", advanced=True),
)

RERUN_RISK_COMMANDS = {
    "run-force",
    "resume-force",
    "run-module-force",
    "b-retry-segment",
    "c-retry-shot",
    "d-retry-shot",
    "bcd-retry-segment",
}


def run_interactive_cli(
    *,
    workspace_root: Path,
    default_config_path: Path,
    execute_request: ExecuteRequest,
) -> int:
    """交互模式主循环。"""
    memory = _load_session_memory()
    _render_startup(
        workspace_root=workspace_root,
        default_config_path=default_config_path,
        memory=memory,
    )

    while True:
        try:
            menu_choice = _prompt_main_menu()
            if menu_choice == "exit":
                print("已退出交互模式。")
                return 0
            if menu_choice == "clear-memory":
                _clear_session_memory(memory=memory)
                print("已清除最近输入。")
                continue
            if menu_choice == "advanced":
                confirmed = _prompt_yes_no("高级操作可能触发重跑，是否继续", default_no=True)
                if not confirmed:
                    continue
                advanced_result = _handle_advanced_menu(
                    workspace_root=workspace_root,
                    default_config_path=default_config_path,
                    memory=memory,
                    execute_request=execute_request,
                )
                if advanced_result == "exit":
                    print("已退出交互模式。")
                    return 0
                continue

            action = _find_action_by_key(MAIN_ACTIONS, menu_choice)
            if action is None:
                print(f"未知菜单项：{menu_choice}，请重试。")
                continue
            _render_action_hint(action=action)

            flow_result = _run_command_flow(
                action=action,
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
                execute_request=execute_request,
            )
            if flow_result == "exit":
                print("已退出交互模式。")
                return 0
        except KeyboardInterrupt:
            interrupt_action = _prompt_interrupt_action()
            if interrupt_action == "exit":
                print("已退出交互模式。")
                return 0


def _handle_advanced_menu(
    *,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
    execute_request: ExecuteRequest,
) -> str:
    while True:
        try:
            menu_choice = _prompt_advanced_menu()
            if menu_choice == "back":
                return "back"
            if menu_choice == "exit":
                return "exit"
            action = _find_action_by_key(ADVANCED_ACTIONS, menu_choice)
            if action is None:
                print(f"未知高级菜单项：{menu_choice}，请重试。")
                continue
            _render_action_hint(action=action)
            flow_result = _run_command_flow(
                action=action,
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
                execute_request=execute_request,
            )
            if flow_result == "exit":
                return "exit"
        except KeyboardInterrupt:
            interrupt_action = _prompt_interrupt_action()
            if interrupt_action == "exit":
                return "exit"


def _run_command_flow(
    *,
    action: MenuAction,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
    execute_request: ExecuteRequest,
) -> str:
    """单命令交互流：采集参数 -> 预览确认 -> 执行。"""
    while True:
        request = _build_request_for_action(
            action=action,
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            memory=memory,
        )
        if request is None:
            return "back"
        request = _attach_user_custom_prompt_override_if_needed(
            request=request,
            workspace_root=workspace_root,
        )

        _render_preview(action=action, request=request)
        confirmed = _prompt_yes_no("是否执行该命令", default_no=True)
        if not confirmed:
            keep_editing = _prompt_yes_no("是否继续修改参数", default_no=False)
            if keep_editing:
                continue
            return "back"

        _update_memory_from_request(memory=memory, request=request)
        _save_session_memory(memory=memory)

        execute_result = _execute_with_recovery(request=request, execute_request=execute_request)
        if execute_result == "retry":
            continue
        if execute_result == "edit":
            continue
        if execute_result == "exit":
            return "exit"
        return "done"


def _execute_with_recovery(*, request: CommandRequest, execute_request: ExecuteRequest) -> str:
    while True:
        try:
            summary = execute_request(request)
            print("\n执行成功，摘要如下：")
            print(_json_dumps(summary))
            print("\n建议下一步：")
            print("  [1] 返回主菜单继续操作")
            print("  [2] 退出交互模式")
            choice = _prompt_number_choice(max_index=2, prompt_text="输入序号：")
            if choice == 2:
                return "exit"
            return "done"
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001
            print(f"\n执行失败：{error}")
            print("请选择后续动作：")
            print("  [1] 重试本次参数")
            print("  [2] 修改参数")
            print("  [3] 返回主菜单")
            print("  [4] 退出交互模式")
            choice = _prompt_number_choice(max_index=4, prompt_text="输入序号：")
            if choice == 1:
                continue
            if choice == 2:
                return "edit"
            if choice == 3:
                return "done"
            return "exit"


def _build_request_for_action(
    *,
    action: MenuAction,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> CommandRequest | None:
    if action.key == "run":
        return _collect_run_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            memory=memory,
        )
    if action.key == "run-force":
        return _collect_run_force_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
    if action.key == "resume":
        return _collect_resume_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            memory=memory,
        )
    if action.key == "resume-force":
        return _collect_resume_force_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
    if action.key == "run-module":
        return _collect_run_module_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            memory=memory,
        )
    if action.key == "run-module-force":
        return _collect_run_module_force_request(
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
    if action.key in {"b-task-status", "c-task-status", "d-task-status", "bcd-task-status", "monitor"}:
        return _collect_task_command_request(
            command=action.key,
            workspace_root=workspace_root,
            default_config_path=default_config_path,
            memory=memory,
        )
    if action.key in {"b-retry-segment", "bcd-retry-segment"}:
        return _collect_rerun_segment_request(
            command=action.key,
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
    if action.key in {"c-retry-shot", "d-retry-shot"}:
        return _collect_rerun_shot_request(
            command=action.key,
            workspace_root=workspace_root,
            default_config_path=default_config_path,
        )
    raise RuntimeError(f"未实现的交互动作：{action.key}")


def _attach_user_custom_prompt_override_if_needed(*, request: CommandRequest, workspace_root: Path) -> CommandRequest:
    """
    功能说明：当命令可能触发模块B时，采集本次 user_custom_prompt 覆盖值并写入请求。
    参数说明：
    - request: 已采集基础参数的命令请求对象。
    - workspace_root: 项目根目录（用于扫描模板版本）。
    返回值：
    - CommandRequest: 注入覆盖值后的请求对象。
    异常说明：无。
    边界条件：不触发模块B的命令保持原样返回。
    """
    if not _request_can_trigger_module_b(request=request):
        return request
    if _should_skip_user_custom_prompt_for_resume(request=request, workspace_root=workspace_root):
        print("检测到模块B已完成，本次 resume 跳过 user_custom_prompt 输入。")
        return request
    prompt_override = _prompt_user_custom_prompt_override(workspace_root=workspace_root)
    return replace(request, user_custom_prompt_override=prompt_override)


def _request_can_trigger_module_b(*, request: CommandRequest) -> bool:
    """
    功能说明：判断当前命令是否可能触发模块B执行。
    参数说明：
    - request: 命令请求对象。
    返回值：
    - bool: True 表示命令会触发或可能触发模块B。
    异常说明：无。
    边界条件：run/resume 在 force-module 为 C/D 时视为不会触发模块B。
    """
    command = str(request.command).strip().lower()
    if command in {"b-retry-segment", "bcd-retry-segment"}:
        return True
    if command == "run-module":
        return str(request.module or "").strip().upper() == "B"
    if command in {"run", "resume"}:
        force_module = str(request.force_module or "").strip().upper()
        if force_module in {"C", "D"}:
            return False
        return True
    return False


def _should_skip_user_custom_prompt_for_resume(*, request: CommandRequest, workspace_root: Path) -> bool:
    """
    功能说明：判断 resume 场景是否可跳过 user_custom_prompt 采集。
    参数说明：
    - request: 命令请求对象。
    - workspace_root: 项目根目录。
    返回值：
    - bool: True 表示可跳过采集。
    异常说明：无。
    边界条件：仅在 resume 且模块B状态为 done 时返回 True。
    """
    command = str(request.command).strip().lower()
    if command != "resume":
        return False
    force_module = str(request.force_module or "").strip().upper()
    if force_module in {"A", "B"}:
        return False
    task_id = str(request.task_id or "").strip()
    if not task_id:
        return False
    module_status_map = _load_module_status_map_for_request(request=request, workspace_root=workspace_root)
    b_status = str(module_status_map.get("B", "")).strip().lower()
    return b_status == "done"


def _load_module_status_map_for_request(*, request: CommandRequest, workspace_root: Path) -> dict[str, str]:
    """
    功能说明：按请求配置定位状态库并读取模块状态映射。
    参数说明：
    - request: 命令请求对象（需包含 task_id/config_path）。
    - workspace_root: 项目根目录。
    返回值：
    - dict[str, str]: 模块状态映射；读取失败时返回空映射。
    异常说明：无（内部吞掉读取异常并返回空映射）。
    边界条件：状态库不存在或任务不存在时返回空映射。
    """
    task_id = str(request.task_id or "").strip()
    if not task_id:
        return {}
    db_path = _resolve_state_db_path_from_config(config_path=request.config_path, workspace_root=workspace_root)
    if db_path is None:
        return {}
    try:
        store = StateStore(db_path=db_path)
        store.reconcile_bcd_module_statuses_by_units(task_id=task_id)
        return store.get_module_status_map(task_id=task_id)
    except Exception:  # noqa: BLE001
        return {}


def _resolve_state_db_path_from_config(*, config_path: Path, workspace_root: Path) -> Path | None:
    """
    功能说明：从配置文件解析 runs_dir 并定位状态库路径。
    参数说明：
    - config_path: 配置文件路径。
    - workspace_root: 项目根目录。
    返回值：
    - Path | None: 状态库路径；不可解析时返回 None。
    异常说明：无（读取失败时返回 None）。
    边界条件：仅当 pipeline_state.sqlite3 实际存在时返回路径。
    """
    try:
        config_obj = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    runs_dir_text = str((config_obj.get("paths") or {}).get("runs_dir", "")).strip()
    if not runs_dir_text:
        return None
    runs_dir_path = Path(runs_dir_text)
    if not runs_dir_path.is_absolute():
        runs_dir_path = (workspace_root / runs_dir_path).resolve()
    db_path = (runs_dir_path / "pipeline_state.sqlite3").resolve()
    if not db_path.exists() or not db_path.is_file():
        return None
    return db_path


def _prompt_user_custom_prompt_override(*, workspace_root: Path) -> str:
    """
    功能说明：交互采集模块B user_custom_prompt 覆盖值（仅本次命令生效）。
    参数说明：
    - workspace_root: 项目根目录（用于扫描模板版本）。
    返回值：
    - str: 覆盖提示词文本；空字符串表示不注入额外用户提示。
    异常说明：无。
    边界条件：默认不添加（等价空字符串）。
    """
    while True:
        should_add = _prompt_yes_no("是否添加本次模块B user_custom_prompt（仅本次命令生效）", default_no=True)
        if not should_add:
            return ""
        selected_prompt = _prompt_user_custom_prompt_text(workspace_root=workspace_root)
        if selected_prompt is _BACK:
            continue
        return str(selected_prompt)


def _prompt_user_custom_prompt_text(*, workspace_root: Path) -> str | object:
    """
    功能说明：选择 user_custom_prompt 内容来源（版本模板/手动输入/空字符串）。
    参数说明：
    - workspace_root: 项目根目录（用于扫描模板版本）。
    返回值：
    - str | object: 返回文本；返回 _BACK 表示回到上一步。
    异常说明：无。
    边界条件：模板读取失败项会跳过并在终端提示。
    """
    template_options = _discover_user_prompt_template_options(workspace_root=workspace_root)
    print("\n请选择 user_custom_prompt 来源：")
    option_index = 1
    option_values: dict[int, str] = {}
    for option in template_options:
        preview = option["preview"]
        print(f"  [{option_index}] 使用模板版本 {option['version']}（{preview}）")
        option_values[option_index] = option["content"]
        option_index += 1
    manual_option_index = option_index
    empty_option_index = option_index + 1
    print(f"  [{manual_option_index}] 手动输入 user_custom_prompt")
    print(f"  [{empty_option_index}] 留空（不注入）")
    choice = _prompt_number_choice(max_index=empty_option_index, prompt_text="输入序号（输入 q 返回上一步）：")
    if choice is _BACK:
        return _BACK
    selected_index = int(choice)
    if selected_index in option_values:
        return option_values[selected_index]
    if selected_index == manual_option_index:
        manual_value = _prompt_optional_text(
            "输入 user_custom_prompt（支持一句话剧情/画风引导，输入 - 设为空）",
            default_value="",
        )
        if manual_value is _BACK:
            return _BACK
        return str(manual_value)
    return ""


def _discover_user_prompt_template_options(*, workspace_root: Path) -> list[dict[str, str]]:
    """
    功能说明：扫描 configs/prompts 下 module_b_prompt.v*.md 并提取 user_prompt_template 文本。
    参数说明：
    - workspace_root: 项目根目录。
    返回值：
    - list[dict[str, str]]: 版本与模板文本列表。
    异常说明：无。
    边界条件：解析失败文件会跳过。
    """
    prompt_dir = (workspace_root / "configs" / "prompts").resolve()
    if not prompt_dir.exists():
        return []
    options: list[dict[str, str]] = []
    for template_path in sorted(prompt_dir.glob("module_b_prompt.v*.md")):
        try:
            template_text = template_path.read_text(encoding="utf-8-sig")
        except OSError as error:
            print(f"提示：读取模板失败，已跳过：{template_path}，错误={error}")
            continue
        user_prompt_template = _extract_markdown_section_text(
            template_text=template_text,
            section_name="user_prompt_template",
        )
        if user_prompt_template is None:
            print(f"提示：模板缺少 user_prompt_template，已跳过：{template_path}")
            continue
        normalized_content = user_prompt_template.strip()
        if not normalized_content:
            print(f"提示：模板 user_prompt_template 为空，已跳过：{template_path}")
            continue
        preview = normalized_content.replace("\n", " ")
        if len(preview) > 48:
            preview = f"{preview[:48]}..."
        options.append(
            {
                "version": template_path.stem,
                "content": normalized_content,
                "preview": preview,
            }
        )
    return options


def _extract_markdown_section_text(*, template_text: str, section_name: str) -> str | None:
    """
    功能说明：从Markdown模板文本中提取指定二级标题段落内容。
    参数说明：
    - template_text: Markdown全文。
    - section_name: 段落名（如 user_prompt_template）。
    返回值：
    - str | None: 段落文本；不存在时返回 None。
    异常说明：无。
    边界条件：按 `## <name>` 切分段落。
    """
    target = str(section_name).strip().lower()
    current_key = ""
    buffer: list[str] = []
    sections: dict[str, str] = {}
    for line in str(template_text).splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = stripped[3:].strip().lower()
            buffer = []
            continue
        if current_key:
            buffer.append(line)
    if current_key:
        sections[current_key] = "\n".join(buffer).strip()
    return sections.get(target)


def _collect_run_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> CommandRequest | None:
    task_id = str(memory.task_id).strip()
    config_path: Path | None = None
    audio_text = str(memory.audio_path).strip()
    stage = 0

    while True:
        if stage == 0:
            answer = _prompt_required_text("任务ID", default_value=task_id)
            if answer is _BACK:
                return None
            task_id = str(answer).strip()
            stage = 1
            continue

        if stage == 1:
            answer = _prompt_config_path(
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
            )
            if answer is _BACK:
                stage = 0
                continue
            config_path = answer
            stage = 2
            continue

        if stage == 2:
            answer = _prompt_optional_text(
                "输入音频路径（回车使用配置默认；输入 - 清空）",
                default_value=audio_text,
            )
            if answer is _BACK:
                stage = 1
                continue
            audio_text = str(answer)
            break

    audio_path = None
    if audio_text.strip():
        audio_path = _resolve_path(workspace_root=workspace_root, raw_text=audio_text)

    assert config_path is not None
    return CommandRequest(
        command="run",
        task_id=task_id,
        config_path=config_path,
        audio_path=audio_path,
    )


def _collect_resume_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> CommandRequest | None:
    task_id = str(memory.task_id).strip()
    config_path: Path | None = None
    stage = 0

    while True:
        if stage == 0:
            answer = _prompt_required_text("任务ID", default_value=task_id)
            if answer is _BACK:
                return None
            task_id = str(answer).strip()
            stage = 1
            continue

        if stage == 1:
            answer = _prompt_config_path(
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
            )
            if answer is _BACK:
                stage = 0
                continue
            config_path = answer
            _render_resume_task_module_status(
                task_id=task_id,
                config_path=config_path,
                workspace_root=workspace_root,
            )
            break

    assert config_path is not None
    return CommandRequest(
        command="resume",
        task_id=task_id,
        config_path=config_path,
    )


def _render_resume_task_module_status(*, task_id: str, config_path: Path, workspace_root: Path) -> None:
    """
    功能说明：在 resume 采集阶段输出当前任务模块状态摘要。
    参数说明：
    - task_id: 任务标识。
    - config_path: 配置文件路径（用于定位状态库）。
    - workspace_root: 项目根目录。
    返回值：无。
    异常说明：无（读取失败时仅输出提示，不中断交互）。
    边界条件：任务不存在或状态库不可用时给出轻量提示。
    """
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        return
    request = CommandRequest(
        command="resume",
        task_id=normalized_task_id,
        config_path=config_path,
    )
    db_path = _resolve_state_db_path_from_config(config_path=config_path, workspace_root=workspace_root)
    if db_path is None:
        print("\n当前模块完成状态：未发现状态库（pipeline_state.sqlite3）。")
        return
    try:
        store = StateStore(db_path=db_path)
        task_record = store.get_task(task_id=normalized_task_id)
        module_status_map = _load_module_status_map_for_request(request=request, workspace_root=workspace_root)
    except Exception:  # noqa: BLE001
        print("\n当前模块完成状态：读取失败，请继续执行后再检查。")
        return

    if not task_record and not module_status_map:
        print(f"\n当前模块完成状态：未找到任务记录（task_id={normalized_task_id}）。")
        return

    task_status = str((task_record or {}).get("status", "unknown")).strip()
    a_status = str(module_status_map.get("A", "pending"))
    b_status = str(module_status_map.get("B", "pending"))
    c_status = str(module_status_map.get("C", "pending"))
    d_status = str(module_status_map.get("D", "pending"))
    print("\n当前模块完成状态：")
    print(f"  task={task_status} | A:{a_status} B:{b_status} C:{c_status} D:{d_status}")


def _collect_run_module_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> CommandRequest | None:
    task_id = str(memory.task_id).strip()
    module = "A"
    config_path: Path | None = None
    audio_text = str(memory.audio_path).strip()
    stage = 0

    while True:
        if stage == 0:
            answer = _prompt_required_text("任务ID", default_value=task_id)
            if answer is _BACK:
                return None
            task_id = str(answer).strip()
            stage = 1
            continue

        if stage == 1:
            answer = _prompt_module_choice("选择模块")
            if answer is _BACK:
                stage = 0
                continue
            module = str(answer).strip()
            stage = 2
            continue

        if stage == 2:
            answer = _prompt_config_path(
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
            )
            if answer is _BACK:
                stage = 1
                continue
            config_path = answer
            stage = 3
            continue

        if stage == 3:
            answer = _prompt_optional_text(
                "输入音频路径（首次初始化可能需要；回车留空；输入 - 清空）",
                default_value=audio_text,
            )
            if answer is _BACK:
                stage = 2
                continue
            audio_text = str(answer)
            break

    audio_path = None
    if audio_text.strip():
        audio_path = _resolve_path(workspace_root=workspace_root, raw_text=audio_text)

    assert config_path is not None
    return CommandRequest(
        command="run-module",
        task_id=task_id,
        module=module,
        config_path=config_path,
        audio_path=audio_path,
    )


def _collect_run_force_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
) -> CommandRequest | None:
    while True:
        task_context = _select_rerun_task_context(workspace_root=workspace_root, default_config_path=default_config_path)
        if task_context is None:
            return None
        force_module = _prompt_module_choice("选择 force-module")
        if force_module is _BACK:
            continue
        return CommandRequest(
            command="run",
            task_id=task_context.task_id,
            config_path=task_context.config_path,
            audio_path=task_context.audio_path,
            force_module=str(force_module).strip(),
        )


def _collect_resume_force_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
) -> CommandRequest | None:
    while True:
        task_context = _select_rerun_task_context(workspace_root=workspace_root, default_config_path=default_config_path)
        if task_context is None:
            return None
        force_module = _prompt_module_choice("选择 force-module")
        if force_module is _BACK:
            continue
        return CommandRequest(
            command="resume",
            task_id=task_context.task_id,
            config_path=task_context.config_path,
            force_module=str(force_module).strip(),
        )


def _collect_run_module_force_request(
    *,
    workspace_root: Path,
    default_config_path: Path,
) -> CommandRequest | None:
    while True:
        task_context = _select_rerun_task_context(workspace_root=workspace_root, default_config_path=default_config_path)
        if task_context is None:
            return None
        module = _prompt_module_choice("选择模块")
        if module is _BACK:
            continue
        return CommandRequest(
            command="run-module",
            task_id=task_context.task_id,
            module=str(module).strip(),
            config_path=task_context.config_path,
            audio_path=task_context.audio_path,
            force=True,
        )


def _collect_task_command_request(
    *,
    command: str,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> CommandRequest | None:
    task_id = str(memory.task_id).strip()
    config_path: Path | None = None
    stage = 0

    while True:
        if stage == 0:
            answer = _prompt_required_text("任务ID", default_value=task_id)
            if answer is _BACK:
                return None
            task_id = str(answer).strip()
            stage = 1
            continue

        if stage == 1:
            answer = _prompt_config_path(
                workspace_root=workspace_root,
                default_config_path=default_config_path,
                memory=memory,
            )
            if answer is _BACK:
                stage = 0
                continue
            config_path = answer
            break

    assert config_path is not None
    return CommandRequest(
        command=command,
        task_id=task_id,
        config_path=config_path,
    )


def _collect_rerun_segment_request(
    *,
    command: str,
    workspace_root: Path,
    default_config_path: Path,
) -> CommandRequest | None:
    while True:
        task_context = _select_rerun_task_context(workspace_root=workspace_root, default_config_path=default_config_path)
        if task_context is None:
            return None
        segment_id = _prompt_required_text("segment_id", default_value="")
        if segment_id is _BACK:
            continue
        return CommandRequest(
            command=command,
            task_id=task_context.task_id,
            segment_id=str(segment_id).strip(),
            config_path=task_context.config_path,
        )


def _collect_rerun_shot_request(
    *,
    command: str,
    workspace_root: Path,
    default_config_path: Path,
) -> CommandRequest | None:
    while True:
        task_context = _select_rerun_task_context(workspace_root=workspace_root, default_config_path=default_config_path)
        if task_context is None:
            return None
        shot_id = _prompt_required_text("shot_id", default_value="")
        if shot_id is _BACK:
            continue
        return CommandRequest(
            command=command,
            task_id=task_context.task_id,
            shot_id=str(shot_id).strip(),
            config_path=task_context.config_path,
        )


def _select_rerun_task_context(*, workspace_root: Path, default_config_path: Path) -> RerunTaskContext | None:
    db_paths = _discover_state_db_paths(workspace_root=workspace_root, default_config_path=default_config_path)
    if not db_paths:
        print("未发现可用状态库（pipeline_state.sqlite3），已返回高级菜单。")
        return None

    while True:
        selected_db_path = _prompt_state_db_choice(db_paths=db_paths)
        if selected_db_path is _BACK:
            return None

        store = StateStore(db_path=selected_db_path)
        task_rows = store.list_tasks()
        if not task_rows:
            print(f"所选状态库暂无任务：{selected_db_path}")
            return None
        task_ids = [str(item.get("task_id", "")).strip() for item in task_rows]
        module_summary_map = store.list_task_module_status_map(task_ids=task_ids)
        selected_task = _prompt_task_choice(task_rows=task_rows, module_summary_map=module_summary_map)
        if selected_task is None:
            continue

        selected_task_id = str(selected_task.get("task_id", "")).strip()
        selected_config_path_text = str(selected_task.get("config_path", "")).strip()
        selected_audio_path_text = str(selected_task.get("audio_path", "")).strip()
        if not selected_task_id or not selected_config_path_text or not selected_audio_path_text:
            print("所选任务缺少必要字段（task_id/config_path/audio_path），已返回高级菜单。")
            return None

        module_status_map = module_summary_map.get(
            selected_task_id,
            {module_name: "pending" for module_name in MODULE_ORDER},
        )
        return RerunTaskContext(
            db_path=selected_db_path,
            task_id=selected_task_id,
            config_path=Path(selected_config_path_text),
            audio_path=Path(selected_audio_path_text),
            task_status=str(selected_task.get("status", "unknown")),
            updated_at=str(selected_task.get("updated_at", "")),
            module_status_map=module_status_map,
        )


def _discover_state_db_paths(*, workspace_root: Path, default_config_path: Path) -> list[Path]:
    config_dir = workspace_root / "configs"
    config_paths: list[Path] = [default_config_path]
    if config_dir.exists():
        config_paths.extend(sorted(config_dir.rglob("*.json")))

    discovered: list[Path] = []
    seen: set[str] = set()
    for config_path in config_paths:
        try:
            config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        runs_dir_text = str((config_obj.get("paths") or {}).get("runs_dir", "")).strip()
        if not runs_dir_text:
            continue
        runs_dir_path = Path(runs_dir_text)
        if not runs_dir_path.is_absolute():
            runs_dir_path = (workspace_root / runs_dir_path).resolve()
        db_path = (runs_dir_path / "pipeline_state.sqlite3").resolve()
        db_path_key = str(db_path)
        if db_path_key in seen:
            continue
        seen.add(db_path_key)
        if db_path.exists() and db_path.is_file():
            discovered.append(db_path)
    return discovered


def _prompt_state_db_choice(*, db_paths: list[Path]) -> Path | object:
    print("\n请选择状态库：")
    for idx, db_path in enumerate(db_paths, start=1):
        print(f"  [{idx}] {db_path}")
    choice = _prompt_number_choice(max_index=len(db_paths), prompt_text="输入序号（输入 q 返回）：")
    if choice is _BACK:
        return _BACK
    return db_paths[int(choice) - 1]


def _prompt_task_choice(
    *,
    task_rows: list[dict[str, Any]],
    module_summary_map: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    print("\n请选择任务：")
    for idx, item in enumerate(task_rows, start=1):
        task_id = str(item.get("task_id", "")).strip()
        task_status = str(item.get("status", "unknown")).strip()
        updated_at = str(item.get("updated_at", "")).strip()
        module_status = module_summary_map.get(task_id, {module_name: "pending" for module_name in MODULE_ORDER})
        module_text = " ".join(f"{module_name}:{module_status.get(module_name, 'pending')}" for module_name in MODULE_ORDER)
        print(f"  [{idx}] {task_id} | task={task_status} | {module_text} | updated={updated_at}")
    choice = _prompt_number_choice(max_index=len(task_rows), prompt_text="输入序号（输入 q 返回）：")
    if choice is _BACK:
        return None
    return task_rows[int(choice) - 1]


def _prompt_main_menu() -> str:
    print("\n主菜单：")
    for idx, action in enumerate(MAIN_ACTIONS, start=1):
        print(f"  [{idx}] {action.title}")
    advanced_index = len(MAIN_ACTIONS) + 1
    clear_memory_index = len(MAIN_ACTIONS) + 2
    exit_index = len(MAIN_ACTIONS) + 3
    print(f"  [{advanced_index}] 显示高级操作（默认隐藏重跑相关命令）")
    print(f"  [{clear_memory_index}] 清除最近输入")
    print(f"  [{exit_index}] 退出")

    selection = _prompt_number_choice(max_index=exit_index, prompt_text="输入序号继续：")
    if selection == advanced_index:
        return "advanced"
    if selection == clear_memory_index:
        return "clear-memory"
    if selection == exit_index:
        return "exit"
    action = MAIN_ACTIONS[selection - 1]
    return action.key


def _prompt_advanced_menu() -> str:
    print("\n高级菜单（含重跑能力）：")
    for idx, action in enumerate(ADVANCED_ACTIONS, start=1):
        print(f"  [{idx}] {action.title}")
    back_index = len(ADVANCED_ACTIONS) + 1
    exit_index = len(ADVANCED_ACTIONS) + 2
    print(f"  [{back_index}] 返回主菜单")
    print(f"  [{exit_index}] 退出")

    selection = _prompt_number_choice(max_index=exit_index, prompt_text="输入序号继续：")
    if selection == back_index:
        return "back"
    if selection == exit_index:
        return "exit"
    action = ADVANCED_ACTIONS[selection - 1]
    return action.key


def _render_action_hint(*, action: MenuAction) -> None:
    hint_lines = ACTION_HINTS.get(action.key, ())
    if not hint_lines:
        return
    print(f"\n操作提示（{action.title}）：")
    for line in hint_lines:
        print(f"- {line}")


def _prompt_interrupt_action() -> str:
    print("\n检测到 Ctrl+C：")
    print("  [1] 返回主菜单")
    print("  [2] 退出程序")
    choice = _prompt_number_choice(max_index=2, prompt_text="输入序号：")
    if choice == 2:
        return "exit"
    return "menu"


def _read_interactive_input(prompt_text: str) -> str:
    """
    功能说明：读取交互输入并清理异常控制字符，增强在异常TTY状态下的健壮性。
    参数说明：
    - prompt_text: 输入提示文本。
    返回值：
    - str: 清理后的输入文本（已 strip）。
    异常说明：无（EOF/中断由调用方上层处理）。
    边界条件：会清除 \r/\x7f/\b 及不可打印控制字符，避免出现 ^M/^? 干扰菜单解析。
    """
    # 某些终端/代理环境下 stdout 可能延迟刷新，先手动 flush 保证菜单先可见。
    sys.stdout.flush()
    sys.stderr.flush()
    raw_text = input(prompt_text)
    normalized = str(raw_text)
    normalized = normalized.replace("\r", "").replace("\x7f", "").replace("\b", "")
    normalized = normalized.replace("^M", "").replace("^?", "")
    normalized = "".join(ch for ch in normalized if ch == "\t" or ord(ch) >= 32)
    return normalized.strip()


def _prompt_number_choice(*, max_index: int, prompt_text: str) -> int | object:
    while True:
        answer = _read_interactive_input(prompt_text).lower()
        if answer in {"q", "quit", "exit"}:
            return _BACK
        if not answer.isdigit():
            print(f"输入无效：{answer}，请输入数字序号。")
            continue
        number = int(answer)
        if number < 1 or number > max_index:
            print(f"序号越界：{number}，请输入 1-{max_index}。")
            continue
        return number


def _prompt_required_text(label: str, *, default_value: str) -> str | object:
    default_text = str(default_value).strip()
    while True:
        suffix = f"（默认：{default_text}）" if default_text else ""
        answer = _read_interactive_input(f"{label}{suffix}（输入 q 返回上一步）：")
        lowered = answer.lower()
        if lowered in {"q", "quit", "exit"}:
            return _BACK
        if answer:
            return answer
        if default_text:
            return default_text
        print(f"{label} 不能为空，请重新输入。")


def _prompt_optional_text(label: str, *, default_value: str) -> str | object:
    default_text = str(default_value).strip()
    while True:
        suffix = f"（默认：{default_text}）" if default_text else ""
        answer = _read_interactive_input(f"{label}{suffix}（输入 q 返回上一步）：")
        lowered = answer.lower()
        if lowered in {"q", "quit", "exit"}:
            return _BACK
        if answer == "-":
            return ""
        if answer:
            return answer
        return default_text


def _prompt_yes_no(label: str, *, default_no: bool) -> bool:
    suffix = "[y/N]" if default_no else "[Y/n]"
    while True:
        answer = _read_interactive_input(f"{label} {suffix}：").lower()
        if not answer:
            return not default_no
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print(f"输入无效：{answer}，请输入 y 或 n。")


def _prompt_module_choice(label: str) -> str | object:
    print(f"\n{label}：")
    print("  [1] A")
    print("  [2] B")
    print("  [3] C")
    print("  [4] D")
    choice = _prompt_number_choice(max_index=4, prompt_text="输入序号（输入 q 返回）：")
    if choice is _BACK:
        return _BACK
    mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
    return mapping[int(choice)]


def _prompt_config_path(
    *,
    workspace_root: Path,
    default_config_path: Path,
    memory: SessionMemory,
) -> Path | object:
    memory_default = str(memory.config_path).strip()
    default_text = memory_default or str(default_config_path)
    raw_text = _prompt_required_text("配置文件路径", default_value=default_text)
    if raw_text is _BACK:
        return _BACK
    return _resolve_path(workspace_root=workspace_root, raw_text=str(raw_text))


def _resolve_path(*, workspace_root: Path, raw_text: str) -> Path:
    path = Path(raw_text)
    if path.is_absolute():
        return path.resolve()
    return (workspace_root / path).resolve()


def _render_startup(*, workspace_root: Path, default_config_path: Path, memory: SessionMemory) -> None:
    runs_dir_text = _read_default_runs_dir(default_config_path=default_config_path)
    print("\n=== mvpl 交互模式 ===")
    print(f"项目根目录：{workspace_root}")
    print(f"默认配置：{default_config_path}")
    print(f"默认 runs_dir：{runs_dir_text}")
    print("执行模式：交互")
    if memory.task_id or memory.config_path or memory.audio_path:
        print("已加载最近输入：")
        print(f"  task_id: {memory.task_id or '<空>'}")
        print(f"  config: {memory.config_path or '<空>'}")
        print(f"  audio: {memory.audio_path or '<空>'}")


def _read_default_runs_dir(*, default_config_path: Path) -> str:
    try:
        from music_video_pipeline.config import load_config

        config = load_config(config_path=default_config_path)
        return str(config.paths.runs_dir)
    except Exception:  # noqa: BLE001
        return "<读取失败，执行时按具体配置解析>"


def _render_preview(*, action: MenuAction, request: CommandRequest) -> None:
    print("\n参数预览：")
    print(f"命令：{_build_command_preview(request)}")
    print(f"配置：{request.config_path}")
    if request.audio_path is not None:
        print(f"音频：{request.audio_path}")
    if request.command in {"run", "resume"} and request.force_module:
        print(f"force-module：{request.force_module}")
    if request.user_custom_prompt_override is not None:
        override_preview = str(request.user_custom_prompt_override).strip()
        if not override_preview:
            override_preview = "<空>"
        elif len(override_preview) > 80:
            override_preview = f"{override_preview[:80]}..."
        print(f"user_custom_prompt：{override_preview}")
    if action.key in RERUN_RISK_COMMANDS:
        print("风险提示：该命令可能触发重跑或下游重建，请确认任务状态与产物路径。")


def _build_command_preview(request: CommandRequest) -> str:
    parts = ["uv run mvpl", request.command]
    if request.task_id:
        parts.extend(["--task-id", request.task_id])
    if request.module:
        parts.extend(["--module", request.module])
    if request.audio_path:
        parts.extend(["--audio-path", str(request.audio_path)])
    if request.force_module:
        parts.extend(["--force-module", request.force_module])
    if request.force:
        parts.append("--force")
    if request.shot_id:
        parts.extend(["--shot-id", request.shot_id])
    if request.segment_id:
        parts.extend(["--segment-id", request.segment_id])
    parts.extend(["--config", str(request.config_path)])
    return " ".join(parts)


def _session_file_path() -> Path:
    return (Path.home() / ".cache" / "music-video-pipeline" / "interactive_session.json").resolve()


def _load_session_memory() -> SessionMemory:
    path = _session_file_path()
    if not path.exists():
        return SessionMemory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionMemory(
            task_id=str(data.get("task_id", "")).strip(),
            config_path=str(data.get("config_path", "")).strip(),
            audio_path=str(data.get("audio_path", "")).strip(),
        )
    except Exception:  # noqa: BLE001
        return SessionMemory()


def _save_session_memory(*, memory: SessionMemory) -> None:
    path = _session_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(asdict(memory)), encoding="utf-8")


def _clear_session_memory(*, memory: SessionMemory) -> None:
    memory.task_id = ""
    memory.config_path = ""
    memory.audio_path = ""
    path = _session_file_path()
    if path.exists():
        path.unlink()


def _update_memory_from_request(*, memory: SessionMemory, request: CommandRequest) -> None:
    if request.task_id:
        memory.task_id = str(request.task_id).strip()
    memory.config_path = str(request.config_path)
    if request.audio_path is not None:
        memory.audio_path = str(request.audio_path)


def _find_action_by_key(actions: tuple[MenuAction, ...], key: str) -> MenuAction | None:
    for action in actions:
        if action.key == key:
            return action
    return None


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
