"""
文件用途：提供模块 B 的统一编排入口。
核心流程：协调 role1 执行，并在完整编排未接通时显式报错。
输入输出：输入 RuntimeContext，输出 role1 产物路径或抛出未接通异常。
依赖说明：依赖运行上下文。
维护说明：编排顺序与上下游契约变更时需同步更新。
"""

# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于 JSON 读写。
import json
# 标准库：用于并行执行 role1/role2。
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
# 标准库：用于正则匹配。
import re
# 标准库：用于未定阶段的宽松参数占位。
from typing import Any

# 项目内模块：提供运行上下文对象。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：提供 role1 真实执行逻辑。
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber
# 项目内模块：提供 role2 真实执行逻辑。
from music_video_pipeline.modules.module_b.role2_story_planner import (
    Role2StoryPlanner,
    build_big_segment_catalog,
    build_big_segment_catalog_stub,
    build_big_segment_catalog_with_segments,
)
# 项目内模块：提供 role3 真实执行逻辑。
from music_video_pipeline.modules.module_b.role3_shot_planner import Role3ShotPlanner
# 项目内模块：提供 role4 真实执行逻辑。
from music_video_pipeline.modules.module_b.role4_prompt_builder import Role4PromptBuilder
# 项目内模块：提供模块 B role 工作目录路径。
from music_video_pipeline.modules.module_b.artifact_paths import (
    get_module_b_role_dir,
    get_module_b_role_result_path,
    get_module_b_streaming_dir,
)
# 项目内模块：提供 role2/role3 Markdown 契约解析器。
from music_video_pipeline.modules.module_b.markdown_contracts import (
    ScenePlan,
    ShotPlan,
    parse_scene_plans,
    parse_shot_plans,
)


class MultiRoleScriptGenerator:
    """
    功能说明：协调模块 B 多角色执行流程。
    参数说明：初始化时接收模块 B 运行依赖。
    返回值：不适用。
    异常说明：角色执行失败时向上抛出异常。
    边界条件：应保证角色执行顺序与数据流向一致。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def generate(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        功能说明：执行模块 B 生成流程。
        参数说明：接收编排所需的上下文与输入对象。
        返回值：
        - list[dict[str, Any]]: 模块 B 输出结果。
        异常说明：按具体实现定义。
        边界条件：输出结构应满足下游模块消费要求。
        """
        del args, kwargs
        raise NotImplementedError("module_b: orchestrator.generate is not implemented.")


def run_module_b(context: RuntimeContext):
    """
    功能说明：执行模块 B 顶层流程，role1 与 role2 并行调用。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path | dict | object: 模块 B 主流程产物。
    异常说明：任一 role 失败时抛出异常，并尝试等待另一 role 完成以保留其产物。
    边界条件：当前已接通 role1 与 role2，两者互不依赖可并行执行；role3/role4 仍未实现。
    """
    role1_output_path: Path | None = None
    role2_output_path: Path | None = None
    role1_error: Exception | None = None
    role2_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_role1 = executor.submit(run_module_b_role1, context)
        future_role2 = executor.submit(run_module_b_role2, context)
        for future in as_completed([future_role1, future_role2]):
            if future is future_role1:
                try:
                    role1_output_path = future.result()
                    context.state_store.set_module_unit_status(
                        task_id=context.task_id, module_name="B", unit_id="role1", status="done"
                    )
                except Exception as exc:  # noqa: BLE001
                    role1_error = exc
            else:
                try:
                    role2_output_path = future.result()
                    context.state_store.set_module_unit_status(
                        task_id=context.task_id, module_name="B", unit_id="role2", status="done"
                    )
                except Exception as exc:  # noqa: BLE001
                    role2_error = exc

    if role1_error is not None or role2_error is not None:
        error_parts: list[str] = []
        if role1_error is not None:
            error_parts.append(f"role1 失败：{role1_error}")
        if role2_error is not None:
            error_parts.append(f"role2 失败：{role2_error}")
        raise RuntimeError(
            "模块B role1/role2 执行失败。"
            + ("；" + "；".join(error_parts) if error_parts else "")
        )

    role3_output_path = run_module_b_role3(context)
    context.state_store.set_module_unit_status(
        task_id=context.task_id, module_name="B", unit_id="role3", status="done"
    )
    role4_output_path = run_module_b_role4(context)
    context.state_store.set_module_unit_status(
        task_id=context.task_id, module_name="B", unit_id="role4", status="done"
    )
    context.logger.info(
        "模块B 编排完成，task_id=%s，"
        "role1=%s，role2=%s，role3=%s，role4=%s",
        context.task_id,
        role1_output_path,
        role2_output_path,
        role3_output_path,
        role4_output_path,
    )
    return role4_output_path


def run_module_b_role1(context: RuntimeContext) -> Path:
    """
    功能说明：仅执行模块 B 的 role1，并写出 role1 Markdown 产物。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: role1 Markdown 产物路径。
    异常说明：按 role1 真实执行逻辑定义。
    边界条件：不生成 module_b_output.json，不触碰模块 B 单元状态。
    """
    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_markdown = _strip_remotion_section(storyboard_template_path.read_text(encoding="utf-8"))

    role1_describer = Role1ImageryDescriber(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )
    role1_items = role1_describer.generate(storyboard_template_markdown)
    output_path = _write_role1_markdown_output(context=context, role1_items=role1_items)
    context.logger.info(
        "模块B role1 执行完成，task_id=%s，role1_item_count=%s，artifact=%s",
        context.task_id,
        len(role1_items),
        output_path,
    )
    return output_path


def run_module_b_role2(context: RuntimeContext) -> Path:
    """
    功能说明：仅执行模块 B 的 role2，并写出 role2 Markdown 产物。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: role2 Markdown 产物路径。
    异常说明：按 role2 真实执行逻辑定义。
    边界条件：优先从 module_a_output.json 读取真实音频特征，缺失时 fallback 到占位数据。
    """
    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_markdown = _strip_remotion_section(storyboard_template_path.read_text(encoding="utf-8"))

    planner = Role2StoryPlanner(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )
    big_segment_catalog = _load_big_segment_catalog(context)
    scene_plans = planner.generate(storyboard_template_markdown, big_segment_catalog)
    output_path = _write_role2_markdown_output(context=context, scene_plans=scene_plans)
    context.logger.info(
        "模块B role2 执行完成，task_id=%s，scene_plan_count=%s，artifact=%s",
        context.task_id,
        len(scene_plans),
        output_path,
    )
    return output_path


def run_module_b_role3(context: RuntimeContext) -> Path:
    """
    功能说明：仅执行模块 B 的 role3，并写出 role3 Markdown 产物。
    role3 按 big_segment 并行调 LLM（ThreadPoolExecutor），依赖 role2 输出。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: role3 Markdown 产物路径。
    异常说明：按 role3 真实执行逻辑定义。
    边界条件：过滤黑屏/白屏大段；必须存在 role2 产物。
    """
    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_text = storyboard_template_path.read_text(encoding="utf-8")
    # role3 需要完整的 ## remotion模板 section（不移除）
    remotion_section = _extract_remotion_section(storyboard_template_text)

    # 加载带子段的 catalog（role3 专用，role2 不受影响）
    big_segment_catalog = _load_big_segment_catalog_with_segments(context)

    # 读取 role2 输出
    role2_output_path = get_module_b_role_result_path(context.artifacts_dir, "role2")
    if not role2_output_path.exists():
        raise FileNotFoundError(f"模块B role3 执行失败：缺少 role2 产物 {role2_output_path}")
    role2_markdown = role2_output_path.read_text(encoding="utf-8")
    scene_plans = parse_scene_plans(role2_markdown)

    # 解析 big_segment_catalog 中的 segment 元数据，用于过滤黑屏/白屏
    # 黑屏/白屏：label 为 intro/outro 且 歌词为"无"
    blacklist_segment_ids = _extract_blacklist_segment_ids(big_segment_catalog)

    # 解析 catalog 中每个 big_segment 的镜头信息（用于构建 big_segment_context）
    segment_catalog_by_big = _parse_segment_catalog_by_big(big_segment_catalog)

    scene_plan_map: dict[str, Any] = {}
    for scene_plan in scene_plans:
        bid = str(scene_plan.big_segment_id).strip()
        if not bid:
            continue
        scene_plan_map[bid] = scene_plan

    # 收集需要处理的大段 ID 列表（保持 Module A 原始顺序）
    big_segment_id_list = [bid for bid in _load_module_a_big_segment_ids(context) if bid]

    # 第一遍：处理直接落地的大段（黑屏/白屏/无剧情），无需 LLM
    all_shot_plans: list[ShotPlan] = []
    llm_segments: list[dict[str, str]] = []

    for bid in big_segment_id_list:
        if bid in blacklist_segment_ids:
            context.logger.info("模块B role3 识别到黑屏/白屏大段，直接落地并跳过 LLM：%s", bid)
            direct_plans = _build_direct_role3_shot_plans_for_skipped_big(
                big_segment_id=bid,
                big_segment_catalog=big_segment_catalog,
            )
            _write_role3_streaming_preview_for_big(context=context, big_segment_id=bid, shot_plans=direct_plans)
            all_shot_plans.extend(direct_plans)
            continue

        scene_plan = scene_plan_map.get(bid)
        if scene_plan is None:
            context.logger.info("模块B role3 未找到大段剧情，直接落地占位结果：%s", bid)
            direct_plans = _build_direct_role3_shot_plans_for_skipped_big(
                big_segment_id=bid,
                big_segment_catalog=big_segment_catalog,
            )
            _write_role3_streaming_preview_for_big(context=context, big_segment_id=bid, shot_plans=direct_plans)
            all_shot_plans.extend(direct_plans)
            continue

        # 构建 big_segment_context
        context_lines: list[str] = []
        context_lines.append(f"## {bid} 剧情")
        context_lines.append(str(scene_plan.story_outline_zh).strip())
        context_lines.append("")
        shots_section = segment_catalog_by_big.get(bid, "")
        if shots_section:
            context_lines.append(f"## {bid} 的镜头")
            context_lines.append(shots_section)
        big_segment_context = "\n".join(context_lines).strip()

        llm_segments.append({
            "bid": bid,
            "big_segment_context": big_segment_context,
            "story_outline_zh": str(scene_plan.story_outline_zh).strip(),
        })

    # 第二遍：需要 LLM 的大段并行处理
    if llm_segments:
        with ThreadPoolExecutor(max_workers=min(len(llm_segments), 4)) as executor:
            future_to_bid: dict[Future, str] = {}
            for seg in llm_segments:
                future = executor.submit(
                    _run_role3_llm_big_segment,
                    context=context,
                    project_root=project_root,
                    remotion_section=remotion_section,
                    bid=seg["bid"],
                    big_segment_context=seg["big_segment_context"],
                    story_outline_zh=seg["story_outline_zh"],
                )
                future_to_bid[future] = seg["bid"]

            for future in as_completed(future_to_bid):
                bid = future_to_bid[future]
                try:
                    shot_plans = future.result()
                    all_shot_plans.extend(shot_plans)
                except Exception as exc:  # noqa: BLE001
                    context.logger.error(
                        "模块B role3 大段 %s 并行处理失败，降级为直接落地：%s", bid, exc,
                    )
                    direct_plans = _build_direct_role3_shot_plans_for_skipped_big(
                        big_segment_id=bid,
                        big_segment_catalog=big_segment_catalog,
                    )
                    _write_role3_streaming_preview_for_big(
                        context=context, big_segment_id=bid, shot_plans=direct_plans,
                    )
                    all_shot_plans.extend(direct_plans)

    # 按原始 big_segment 顺序排序，保持输出稳定
    bid_order = {bid: idx for idx, bid in enumerate(big_segment_id_list)}
    all_shot_plans.sort(key=lambda sp: bid_order.get(str(getattr(sp, "big_segment_id", "")).strip(), 999))

    output_path = _write_role3_markdown_output(context=context, shot_plans=all_shot_plans)
    context.logger.info(
        "模块B role3 执行完成，task_id=%s，shot_plan_count=%s，artifact=%s",
        context.task_id,
        len(all_shot_plans),
        output_path,
    )
    return output_path


def _run_role3_llm_big_segment(
    context: RuntimeContext,
    project_root: Path,
    remotion_section: str,
    bid: str,
    big_segment_context: str,
    story_outline_zh: str,
) -> list[ShotPlan]:
    """为单个 big_segment 调 LLM 执行 role3（线程安全，每次创建独立 planner 实例）。"""
    planner = Role3ShotPlanner(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )
    context.logger.info("模块B role3 开始处理大段：%s", bid)
    shot_plans = planner.generate(
        storyboard_markdown=remotion_section,
        big_segment_context=big_segment_context,
    )
    for sp in shot_plans:
        sp.big_segment_id = bid
    _write_role3_streaming_preview_for_big(
        context=context,
        big_segment_id=bid,
        shot_plans=shot_plans,
        story_outline_zh=story_outline_zh,
    )
    context.logger.info(
        "模块B role3 大段 %s 完成，shot_count=%s",
        bid, len(shot_plans),
    )
    return shot_plans


def run_module_b_role3_big_segment(context: RuntimeContext, big_segment_id: str) -> Path:
    """
    功能说明：仅对单个 big_segment 重跑 role3，更新现有产物中对应大段。
    参数说明：
    - context: 运行上下文对象。
    - big_segment_id: 要重跑的大段 ID（如 big_001）。
    返回值：
    - Path: role3 Markdown 产物路径。
    异常说明：缺少上游产物或目标大段不存在时抛出。
    边界条件：会原地更新 role3_shot_output.md 中对应 ## 大段。
    """
    bid = str(big_segment_id).strip()
    if not bid:
        raise ValueError("big_segment_id 不能为空。")

    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_text = storyboard_template_path.read_text(encoding="utf-8")
    remotion_section = _extract_remotion_section(storyboard_template_text)

    planner = Role3ShotPlanner(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )

    big_segment_catalog = _load_big_segment_catalog_with_segments(context)

    role2_output_path = get_module_b_streaming_dir(context.artifacts_dir, "role2") / "role2_story_output.streaming.md"
    if not role2_output_path.exists():
        raise FileNotFoundError(f"模块B role3 big segment 重跑失败：缺少 role2 流式产物 {role2_output_path}")
    role2_markdown = role2_output_path.read_text(encoding="utf-8")
    scene_plans = parse_scene_plans(role2_markdown)

    segment_catalog_by_big = _parse_segment_catalog_by_big(big_segment_catalog)
    blacklist_segment_ids = _extract_blacklist_segment_ids(big_segment_catalog)
    target_scene = None
    for sp in scene_plans:
        if str(sp.big_segment_id).strip() == bid:
            target_scene = sp
            break
    if bid in blacklist_segment_ids or target_scene is None:
        context.logger.info("模块B role3 big segment 重跑：大段 %s 直接落地标准结果。", bid)
        direct_plans = _build_direct_role3_shot_plans_for_skipped_big(
            big_segment_id=bid,
            big_segment_catalog=big_segment_catalog,
        )
        _write_role3_streaming_preview_for_big(context=context, big_segment_id=bid, shot_plans=direct_plans)
        existing_plans: list = []
        role3_streaming_dir = get_module_b_streaming_dir(context.artifacts_dir, "role3")
        if role3_streaming_dir.exists():
            for sp_path in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
                try:
                    sp_content = sp_path.read_text(encoding="utf-8").strip()
                    if sp_content:
                        existing_plans.extend(parse_shot_plans(sp_content))
                except Exception:
                    pass
        merged_plans = [p for p in existing_plans if str(getattr(p, "big_segment_id", "")).strip() != bid]
        merged_plans.extend(direct_plans)
        return _write_role3_markdown_output(context=context, shot_plans=merged_plans)

    context_lines: list[str] = []
    context_lines.append(f"## {bid} 剧情")
    context_lines.append(str(target_scene.story_outline_zh).strip())
    context_lines.append("")
    shots_section = segment_catalog_by_big.get(bid, "")
    if shots_section:
        context_lines.append(f"## {bid} 的镜头")
        context_lines.append(shots_section)
    big_segment_context = "\n".join(context_lines).strip()

    context.logger.info("模块B role3 big segment 重跑开始处理大段：%s", bid)
    shot_plans = planner.generate(
        storyboard_markdown=remotion_section,
        big_segment_context=big_segment_context,
    )
    for sp in shot_plans:
        sp.big_segment_id = bid
    # 重写 streaming 文件，加入 story_outline_zh
    _write_role3_streaming_preview_for_big(
        context=context,
        big_segment_id=bid,
        shot_plans=shot_plans,
        story_outline_zh=str(target_scene.story_outline_zh).strip(),
    )

    existing_plans: list = []
    role3_streaming_dir = get_module_b_streaming_dir(context.artifacts_dir, "role3")
    if role3_streaming_dir.exists():
        for sp_path in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                sp_content = sp_path.read_text(encoding="utf-8").strip()
                if sp_content:
                    existing_plans.extend(parse_shot_plans(sp_content))
            except Exception:
                pass
    merged_plans = [p for p in existing_plans if str(getattr(p, "big_segment_id", "")).strip() != bid]
    merged_plans.extend(shot_plans)

    output_path = _write_role3_markdown_output(context=context, shot_plans=merged_plans)
    context.logger.info(
        "模块B role3 big segment 重跑完成，big_segment_id=%s，shot_count=%s，artifact=%s",
        bid, len(shot_plans), output_path,
    )
    return output_path


def _extract_remotion_section(text: str) -> str:
    """从 storyboard 模板中提取「## remotion模板」部分（可选后接 ### 子标题）。"""
    normalized = str(text or "").replace("\r\n", "\n")
    m = re.search(r"(## remotion模板\n(?:### .+\n(?:- .+\n?)*\n?)+)", normalized)
    if m:
        return m.group(1).strip()
    # fallback：查找 ## remotion模板 到下一个 ## 或文件末尾
    m = re.search(r"## remotion模板\n(.+)", normalized, re.DOTALL)
    if m:
        content = m.group(1).strip()
        # 截断到下一个 ## 标题
        next_section = re.search(r"\n## ", content)
        if next_section:
            content = content[: next_section.start()].strip()
        return f"## remotion模板\n{content}"
    raise ValueError("模块B role3 执行失败：storyboard 模板中未找到「## remotion模板」section。")


def _extract_blacklist_segment_ids(big_segment_catalog: str) -> set[str]:
    """从 catalog 中提取黑屏/白屏大段的 ID（intro/outro 且无歌词）。"""
    blacklist: set[str] = set()
    parts = re.split(r"\n(?=### )", str(big_segment_catalog or ""))
    for part in parts:
        part = part.strip()
        if not part.startswith("### "):
            continue
        lines = part.split("\n")
        seg_id = lines[0][4:].strip()
        label = ""
        has_lyric = False
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("- label:"):
                for token in stripped.split("|"):
                    token = token.strip()
                    if token.startswith("label:"):
                        label = token[len("label:"):].strip()
            if stripped.startswith("- 歌词:"):
                lyric_val = stripped[len("- 歌词:"):].strip()
                if lyric_val and lyric_val != "无":
                    has_lyric = True
        if label in ("intro", "outro", "start", "end") and not has_lyric:
            blacklist.add(seg_id)
    return blacklist


def _parse_segment_catalog_by_big(big_segment_catalog: str) -> dict[str, str]:
    """按 big_segment_id 拆分 catalog，返回每个大段下的 ### shot 及其字段行。"""
    result: dict[str, str] = {}
    # catalog 格式：### big_xxx \n - ... \n ### shot_xxx \n - ...
    # 按 ### big_ 拆分大段
    normalized = str(big_segment_catalog or "").replace("\r\n", "\n")
    big_parts = re.split(r"\n(?=### big_)", normalized)
    for big_part in big_parts:
        big_part = big_part.strip()
        if not big_part:
            continue
        lines = big_part.split("\n")
        big_heading = lines[0].strip()
        if not big_heading.startswith("### "):
            continue
        big_id = big_heading[4:].strip()
        # 提取该大段下的所有 ### shot_xxx 及字段行
        shot_body = "\n".join(lines[1:])
        shot_parts = re.split(r"\n(?=### )", shot_body)
        filtered_lines: list[str] = []
        for sp in shot_parts:
            sp = sp.strip()
            if not sp:
                continue
            sp_lines = sp.split("\n")
            sp_heading = sp_lines[0].strip()
            if sp_heading.startswith("### ") and not sp_heading.startswith("### big_"):
                filtered_lines.append(sp)
        if filtered_lines:
            result[big_id] = "\n".join(filtered_lines).strip()
    return result


def _build_direct_role3_shot_plans_for_skipped_big(*, big_segment_id: str, big_segment_catalog: str) -> list[ShotPlan]:
    """为黑屏或白屏大段直接构造 role3 标准结果，不经过 LLM。"""
    shots_section = _parse_segment_catalog_by_big(big_segment_catalog).get(str(big_segment_id).strip(), "")
    if not shots_section:
        return []
    shot_plans: list[ShotPlan] = []
    current_segment_id = ""
    current_label = ""
    for raw_line in shots_section.split("\n"):
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("### "):
            current_segment_id = line[4:].strip()
            current_label = ""
            continue
        if line.startswith("- label:"):
            current_label = line[len("- label:"):].strip()
        if current_segment_id and current_label:
            shot_plans.append(
                ShotPlan(
                    big_segment_id=str(big_segment_id).strip(),
                    segment_id=current_segment_id,
                    scene_desc_zh="黑屏过渡。" if current_label in {"intro", "start"} else "白屏或收束过渡。",
                    remotion_id="CenterTemplate",
                )
            )
            current_segment_id = ""
            current_label = ""
    return shot_plans


def _load_module_a_big_segment_ids(context: RuntimeContext) -> list[str]:
    """按 Module A 输出原始顺序返回 big_segment_id 列表。"""
    module_a_path = context.artifacts_dir / "module_a_output.json"
    if not module_a_path.exists():
        return []
    try:
        payload = json.loads(module_a_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    result: list[str] = []
    for item in (payload.get("big_segments", []) if isinstance(payload, dict) else []):
        if not isinstance(item, dict):
            continue
        big_id = str(item.get("segment_id", "")).strip()
        if big_id:
            result.append(big_id)
    return result


def _write_role3_streaming_preview_for_big(*, context: RuntimeContext, big_segment_id: str, shot_plans: list[ShotPlan], story_outline_zh: str = "") -> None:
    """为直接落地的大段同步写入 role3 streaming 预览文件。"""
    streaming_dir = context.artifacts_dir / "module_b_work" / "role3" / "streaming"
    streaming_dir.mkdir(parents=True, exist_ok=True)
    output_path = streaming_dir / f"role3_segment_output.streaming.{str(big_segment_id).strip()}.md"
    meta_path = streaming_dir / f"role3_segment_output.streaming.{str(big_segment_id).strip()}.meta.json"
    outline = str(story_outline_zh or "").strip()
    header = f"## {str(big_segment_id).strip()} / {outline}" if outline else f"## {str(big_segment_id).strip()}"
    lines: list[str] = [header]
    if outline:
        lines.append(f"- story_outline_zh: {outline}")
    for plan in shot_plans:
        lines.append(f"### {str(plan.segment_id).strip()}")
        lines.append(f"- scene_desc_zh: {str(plan.scene_desc_zh).strip()}")
        lines.append(f"- remotion_id: {str(plan.remotion_id).strip()}")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    meta_payload = {
        "current_attempt": 1,
        "first_chunk_at_ms": 0,
        "first_chunk_at": "",
        "last_chunk_at_ms": 0,
        "last_chunk_at": "",
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_role3_markdown_output(context: RuntimeContext, shot_plans: list) -> Path:
    """
    功能说明：把 role3 的结构化结果回写为 Markdown 主产物。
    参数说明：
    - context: 运行上下文。
    - shot_plans: role3 镜头规划数组（ShotPlan 对象）。
    返回值：
    - Path: Markdown 产物路径。
    异常说明：无。
    边界条件：当前固定写入 artifacts/role3_shot_output.md。
    """
    output_path = get_module_b_role_result_path(context.artifacts_dir, "role3")
    markdown_lines: list[str] = []
    current_big: str = ""
    for plan in shot_plans:
        bid = str(getattr(plan, "big_segment_id", "")).strip()
        if bid and bid != current_big:
            current_big = bid
            markdown_lines.append(f"## {bid}")
        segment_id = str(getattr(plan, "segment_id", "")).strip()
        markdown_lines.append(f"### {segment_id}")
        markdown_lines.append(f"- scene_desc_zh: {str(getattr(plan, 'scene_desc_zh', '')).strip()}")
        markdown_lines.append(f"- remotion_id: {str(getattr(plan, 'remotion_id', '')).strip()}")
        markdown_lines.append("")
    output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return output_path


def _is_multi_subject_template(remotion_id: str) -> bool:
    """GridTemplate 或 ScrollTemplate 为多主体模板。"""
    rid = str(remotion_id or "").strip()
    return "GridTemplate" in rid or "ScrollTemplate" in rid


def _parse_subject_descriptions(scene_desc: str, remotion_id: str) -> list[str]:
    """从 scene_desc_zh 解析主体描述列表。单主体返回全段；多主体只解析格子内主体。"""
    desc = str(scene_desc or "").strip()
    if not desc:
        return [""]
    if not _is_multi_subject_template(remotion_id):
        return [desc]
    m = re.search(r'出现(.+)', desc)
    if not m:
        return [desc]
    part = m.group(1)
    # 多主体模板中的“背景为/场景为”是共享画面背景，不是可单独生图的格子主体。
    part = re.split(r'(?:，|,|；|;)?\s*(?:背景|场景|环境)\s*(?:为|是|：|:)', part, maxsplit=1)[0]
    part = re.sub(r'[。；;]$', '', part).strip()
    subjects = re.split(r'[、，,]', part)
    return [s.strip() for s in subjects if s.strip()]


def _build_shot_id(segment_id: str, subject_index: int) -> str:
    """从 segment_id（如 seg_0001）和主体序号（1-based）构建 shot_id（如 shot_0001_1）。"""
    seg_number = str(segment_id).strip().replace("seg_", "")
    return f"shot_{seg_number}_{int(subject_index)}"


def _run_role4_llm_shot(
    context: RuntimeContext,
    project_root: Path,
    remotion_catalog: dict[str, str],
    visual_registry: dict[str, str],
    sp: ShotPlan,
    subj_idx: int,
    subject_desc: str,
) -> str:
    """为单个 shot 调 LLM 执行 role4（线程安全，每次创建独立 planner 实例）。"""
    segment_id = str(sp.segment_id).strip()
    remotion_id = str(sp.remotion_id).strip()
    scene_desc = str(sp.scene_desc_zh).strip()
    shot_id = _build_shot_id(segment_id, subj_idx)

    planner = Role4PromptBuilder(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )

    shot_brief_lines: list[str] = []
    shot_brief_lines.append(f"- segment_id: {segment_id}")
    shot_brief_lines.append(f"- shot_id: {shot_id}")
    shot_brief_lines.append(f"- big_segment_id: {str(sp.big_segment_id).strip()}")
    shot_brief_lines.append(f"- scene_desc_zh: {scene_desc}")
    shot_brief_lines.append(f"- remotion_id: {remotion_id}")
    subjects = _parse_subject_descriptions(scene_desc, remotion_id)
    if len(subjects) > 1:
        shot_brief_lines.append(f"- subject_index: {subj_idx}")
        shot_brief_lines.append(f"- subject_count: {len(subjects)}")
        shot_brief_lines.append(f"- subject_desc: {subject_desc}")
    shot_brief = "\n".join(shot_brief_lines)

    remotion_template = remotion_catalog.get(remotion_id, "")
    shot_visual_reference = _filter_visual_reference(visual_registry, scene_desc)

    user_variables: dict[str, str] = {
        "remotion_template": remotion_template,
        "shot_brief": shot_brief,
        "subject_desc": subject_desc,
        "visual_reference": shot_visual_reference,
    }

    return planner.generate(user_variables=user_variables, shot_id=shot_id)


def run_module_b_role4(context: RuntimeContext) -> Path:
    """
    功能说明：仅执行模块 B 的 role4，并写出 role4 Markdown 产物。
    role4 按 shot 逐个调 LLM，依赖 role1 与 role3 输出。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: role4 Markdown 产物路径。
    异常说明：按 role4 真实执行逻辑定义。
    边界条件：必须存在 role1 与 role3 产物。
    """
    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_text = storyboard_template_path.read_text(encoding="utf-8")
    # 解析模板目录为 {模板ID: 模板描述}，供 role4 按 remotion_id 精确匹配
    remotion_catalog = _parse_remotion_catalog(storyboard_template_text)

    # 读取 role1 streaming 输出（标题行正确），解析为 {意象名: 完整块} 供后续按 shot 筛选
    role1_output_path = get_module_b_role_result_path(context.artifacts_dir, "role1")
    role1_streaming_path = role1_output_path.parent / "streaming" / f"{role1_output_path.stem}.streaming.md"
    role1_source_path = role1_streaming_path if role1_streaming_path.exists() else role1_output_path
    if not role1_source_path.exists():
        raise FileNotFoundError(f"模块B role4 执行失败：缺少 role1 产物 {role1_source_path}")
    role1_markdown = role1_source_path.read_text(encoding="utf-8")
    visual_registry = _parse_visual_registry(role1_markdown)
    if not visual_registry:
        context.logger.warning("模块B role4 视觉注册表为空，请检查 role1 输出的 ## 标题行是否包含有效的意象名称。")

    # 读取 role3 流式文件作为 shot 来源
    role3_streaming_dir = get_module_b_streaming_dir(context.artifacts_dir, "role3")
    if not role3_streaming_dir.exists():
        raise FileNotFoundError(f"模块B role4 执行失败：缺少 role3 流式产物目录 {role3_streaming_dir}")
    shot_plans: list[ShotPlan] = []
    for sp_path in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
        try:
            sp_content = sp_path.read_text(encoding="utf-8").strip()
            if sp_content:
                shot_plans.extend(parse_shot_plans(sp_content))
        except Exception:
            continue
    if not shot_plans:
        raise FileNotFoundError(f"模块B role4 执行失败：role3 流式产物为空，请先执行 role3")

    # 收集所有 shot 任务（保持原始顺序）
    shot_tasks: list[dict[str, Any]] = []
    for sp in shot_plans:
        segment_id = str(sp.segment_id).strip()
        if not segment_id:
            continue
        remotion_id = str(sp.remotion_id).strip()
        scene_desc = str(sp.scene_desc_zh).strip()
        subjects = _parse_subject_descriptions(scene_desc, remotion_id)
        for subj_idx, subject_desc in enumerate(subjects, start=1):
            shot_tasks.append({
                "sp": sp,
                "subj_idx": subj_idx,
                "subject_desc": subject_desc,
            })

    # 统计每段含多少 shot，用于实时标记 B 单元完成
    segment_shot_counts: dict[str, int] = {}
    for task in shot_tasks:
        seg_id = str(task["sp"].segment_id).strip()
        if seg_id:
            segment_shot_counts[seg_id] = segment_shot_counts.get(seg_id, 0) + 1
    segment_done_counts: dict[str, int] = {}

    # 并行调 LLM
    output_parts_map: dict[int, str] = {}
    failed_count = 0
    with ThreadPoolExecutor(max_workers=min(len(shot_tasks), 4)) as executor:
        future_to_index: dict[Future, int] = {}
        for idx, task in enumerate(shot_tasks):
            future = executor.submit(
                _run_role4_llm_shot,
                context=context,
                project_root=project_root,
                remotion_catalog=remotion_catalog,
                visual_registry=visual_registry,
                sp=task["sp"],
                subj_idx=task["subj_idx"],
                subject_desc=task["subject_desc"],
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                output_parts_map[idx] = result
                seg_id = str(shot_tasks[idx]["sp"].segment_id).strip()
                if seg_id:
                    segment_done_counts[seg_id] = segment_done_counts.get(seg_id, 0) + 1
                    if segment_done_counts[seg_id] >= segment_shot_counts.get(seg_id, 0):
                        context.state_store.set_module_unit_status(
                            task_id=context.task_id,
                            module_name="B",
                            unit_id=seg_id,
                            status="done",
                        )
            except Exception as exc:  # noqa: BLE001
                context.logger.error("模块B role4 shot 处理失败（索引 %s）：%s", idx, exc)
                failed_count += 1

    # 按原始顺序拼接
    output_parts = [output_parts_map[i] for i in range(len(shot_tasks)) if i in output_parts_map]

    if failed_count:
        context.logger.warning("模块B role4 有 %s 个 shot 处理失败", failed_count)

    output_path = _write_role4_markdown_output(context=context, output_parts=output_parts)
    context.logger.info(
        "模块B role4 执行完成，task_id=%s，shot_count=%s，artifact=%s",
        context.task_id,
        len(shot_plans),
        output_path,
    )
    return output_path


def run_module_b_role4_shot(context: RuntimeContext, shot_id: str) -> Path:
    """仅对单个 shot 重跑 role4，写入 per-shot 文件。shot_id 格式：shot_XXXX_X。"""
    sid = str(shot_id).strip()
    if not sid:
        raise ValueError("shot_id 不能为空。")

    # 从 shot_id 解析 segment_id 和 subject_index
    m = re.match(r'^shot_(\d+)_(\d+)$', sid)
    if not m:
        raise ValueError(f"shot_id 格式不正确：{sid}（应为 shot_XXXX_X）")
    seg_number = m.group(1)
    subject_index = int(m.group(2))
    segment_id = f"seg_{seg_number}"

    project_root = _resolve_project_root()
    storyboard_template_path = _resolve_storyboard_template_path(context=context, project_root=project_root)
    storyboard_template_text = storyboard_template_path.read_text(encoding="utf-8")
    remotion_catalog = _parse_remotion_catalog(storyboard_template_text)

    planner = Role4PromptBuilder(
        logger=context.logger,
        llm_config=context.config.module_b.llm,
        project_root=project_root,
        artifacts_dir=context.artifacts_dir,
    )

    role1_output_path = get_module_b_role_result_path(context.artifacts_dir, "role1")
    role1_streaming_path = role1_output_path.parent / "streaming" / f"{role1_output_path.stem}.streaming.md"
    role1_source_path = role1_streaming_path if role1_streaming_path.exists() else role1_output_path
    if not role1_source_path.exists():
        raise FileNotFoundError(f"模块B role4 shot 重跑失败：缺少 role1 产物 {role1_source_path}")
    role1_markdown = role1_source_path.read_text(encoding="utf-8")
    visual_registry = _parse_visual_registry(role1_markdown)

    if not visual_registry:
        context.logger.warning("模块B role4 shot 重跑：视觉注册表为空，请检查 role1 输出的 ## 标题行是否包含有效的意象名称。")

    role3_streaming_dir = get_module_b_streaming_dir(context.artifacts_dir, "role3")
    if not role3_streaming_dir.exists():
        raise FileNotFoundError(f"模块B role4 shot 重跑失败：缺少 role3 流式产物目录 {role3_streaming_dir}")
    shot_plans: list[ShotPlan] = []
    for sp_path in sorted(role3_streaming_dir.glob("role3_segment_output.streaming.*.md")):
        try:
            sp_content = sp_path.read_text(encoding="utf-8").strip()
            if sp_content:
                shot_plans.extend(parse_shot_plans(sp_content))
        except Exception:
            continue
    if not shot_plans:
        raise FileNotFoundError(f"模块B role4 shot 重跑失败：role3 流式产物为空")

    target_sp = None
    for sp in shot_plans:
        if str(sp.segment_id).strip() == segment_id:
            target_sp = sp
            break
    if target_sp is None:
        raise RuntimeError(f"模块B role4 shot 重跑失败：role3 流式产物中找不到 segment_id={segment_id}（shot_id={sid}）")

    remotion_id = str(target_sp.remotion_id).strip()
    remotion_template = remotion_catalog.get(remotion_id, "")
    scene_desc = str(target_sp.scene_desc_zh).strip()
    subjects = _parse_subject_descriptions(scene_desc, remotion_id)

    if subject_index < 1 or subject_index > len(subjects):
        raise RuntimeError(
            f"模块B role4 shot 重跑失败：shot_id={sid} 主体序号 {subject_index} 超出范围（共 {len(subjects)} 个主体）"
        )
    subject_desc = subjects[subject_index - 1]

    shot_brief_lines: list[str] = []
    shot_brief_lines.append(f"- segment_id: {segment_id}")
    shot_brief_lines.append(f"- shot_id: {sid}")
    shot_brief_lines.append(f"- big_segment_id: {str(target_sp.big_segment_id).strip()}")
    shot_brief_lines.append(f"- scene_desc_zh: {scene_desc}")
    shot_brief_lines.append(f"- remotion_id: {remotion_id}")
    if len(subjects) > 1:
        shot_brief_lines.append(f"- subject_index: {subject_index}")
        shot_brief_lines.append(f"- subject_count: {len(subjects)}")
        shot_brief_lines.append(f"- subject_desc: {subject_desc}")
    shot_visual_reference = _filter_visual_reference(visual_registry, scene_desc)

    user_variables: dict[str, str] = {
        "remotion_template": remotion_template,
        "shot_brief": "\n".join(shot_brief_lines),
        "subject_desc": subject_desc,
        "visual_reference": shot_visual_reference,
    }

    context.logger.info("模块B role4 单 shot 重跑开始：%s", sid)
    planner.generate(user_variables=user_variables, shot_id=sid)
    context.logger.info("模块B role4 单 shot 重跑完成：%s", sid)

    # 级联回填：segment → role(1-4) → module B
    context.state_store.set_module_unit_status(
        task_id=context.task_id, module_name="B", unit_id=segment_id, status="done"
    )
    all_b_units = context.state_store.list_module_units(task_id=context.task_id, module_name="B")
    seg_units = [u for u in all_b_units if str(u.get("unit_id", "")).startswith("seg_")]
    if seg_units and all(str(u.get("status", "")).strip() == "done" for u in seg_units):
        for role_id in ("role1", "role2", "role3", "role4"):
            context.state_store.set_module_unit_status(
                task_id=context.task_id, module_name="B", unit_id=role_id, status="done"
            )
        context.state_store.set_module_status(
            task_id=context.task_id, module_name="B", status="done"
        )

    return get_module_b_streaming_dir(context.artifacts_dir, "role4") / f"role4_prompt_output.streaming.{sid}.md"


def _write_role4_markdown_output(context: RuntimeContext, output_parts: list[str]) -> Path:
    """把 role4 的原始 LLM 输出拼接写入单个 Markdown 文件。"""
    output_path = get_module_b_role_result_path(context.artifacts_dir, "role4")
    joined = "\n\n".join(part.strip() for part in output_parts if part.strip())
    output_path.write_text((joined + "\n") if joined else "", encoding="utf-8")
    return output_path


def _resolve_project_root() -> Path:
    """
    功能说明：解析项目根目录路径。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：约定当前文件位于 src/music_video_pipeline/modules/module_b 下。
    """
    return Path(__file__).resolve().parents[4]


def _resolve_storyboard_template_path(context: RuntimeContext, project_root: Path) -> Path:
    """
    功能说明：解析模块 B 当前使用的编排模板路径。
    参数说明：
    - context: 运行上下文。
    - project_root: 项目根目录。
    返回值：
    - Path: 模板绝对路径。
    异常说明：
    - FileNotFoundError: 模板文件不存在时抛出异常。
    边界条件：相对路径默认相对于项目根目录解析。
    """
    raw_template_path = str(context.config.module_b.storyboard_template_file).strip()
    if not raw_template_path:
        raise FileNotFoundError("模块B执行失败：storyboard_template_file 不能为空。")
    template_path = Path(raw_template_path)
    resolved_template_path = template_path if template_path.is_absolute() else (project_root / template_path)
    resolved_template_path = resolved_template_path.resolve()
    if not resolved_template_path.exists():
        raise FileNotFoundError(f"模块B执行失败：未找到编排模板文件 {resolved_template_path}")
    return resolved_template_path


def _write_role1_markdown_output(context: RuntimeContext, role1_items: list[Any]) -> Path:
    """
    功能说明：把 role1 的结构化结果回写为 Markdown 主产物。
    参数说明：
    - context: 运行上下文。
    - role1_items: role1 视觉描述数组。
    返回值：
    - Path: Markdown 产物路径。
    异常说明：无。
    边界条件：当前固定写入 artifacts/role1_visual_output.md。
    """
    output_path = get_module_b_role_result_path(context.artifacts_dir, "role1")
    markdown_lines: list[str] = []
    for item in role1_items:
        markdown_lines.extend(
            [
                f"## {str(getattr(item, 'imagery_name', '')).strip()}",
                f"- pos_zh: {str(getattr(item, 'pos_zh', '')).strip()}",
                f"- pos_en: {str(getattr(item, 'pos_en', '')).strip()}",
                "",
            ]
        )
    output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return output_path


def _write_role2_markdown_output(context: RuntimeContext, scene_plans: list[Any]) -> Path:
    """
    功能说明：把 role2 的结构化结果回写为 Markdown 主产物。
    参数说明：
    - context: 运行上下文。
    - scene_plans: role2 场景规划数组。
    返回值：
    - Path: Markdown 产物路径。
    异常说明：无。
    边界条件：当前固定写入 artifacts/role2_story_output.md。
    """
    output_path = get_module_b_role_result_path(context.artifacts_dir, "role2")
    markdown_lines: list[str] = []
    for plan in scene_plans:
        markdown_lines.extend(
            [
                f"## {str(getattr(plan, 'big_segment_id', '')).strip()}",
                f"- imagery_used: {str(getattr(plan, 'imagery_used', '')).strip()}",
                f"- story_outline_zh: {str(getattr(plan, 'story_outline_zh', '')).strip()}",
                "",
            ]
        )
    output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return output_path


def _load_big_segment_catalog(context: RuntimeContext) -> str:
    """尝试从 module_a_output.json 读取真实 catalog；失败时 fallback 到 stub。"""
    module_a_path = context.artifacts_dir / "module_a_output.json"
    if module_a_path.exists():
        try:
            data = json.loads(module_a_path.read_text(encoding="utf-8"))
            context.logger.info("模块B role2 从 %s 读取真实音频特征。", module_a_path)
            return build_big_segment_catalog(data)
        except Exception as exc:  # noqa: BLE001
            context.logger.warning(
                "模块B role2 读取 %s 失败：%s，fallback 到占位数据。",
                module_a_path,
                exc,
            )
    else:
        context.logger.warning(
            "模块B role2 未找到 %s，fallback 到占位数据。",
            module_a_path,
        )
    return build_big_segment_catalog_stub()


def _load_big_segment_catalog_with_segments(context: RuntimeContext) -> str:
    """尝试从 module_a_output.json 读取带子段的 catalog；失败时 fallback 到 stub。"""
    module_a_path = context.artifacts_dir / "module_a_output.json"
    if module_a_path.exists():
        try:
            data = json.loads(module_a_path.read_text(encoding="utf-8"))
            context.logger.info("模块B role3 从 %s 读取带子段的音频特征。", module_a_path)
            return build_big_segment_catalog_with_segments(data)
        except Exception as exc:  # noqa: BLE001
            context.logger.warning(
                "模块B role3 读取 %s 失败：%s，fallback 到占位数据。",
                module_a_path,
                exc,
            )
    else:
        context.logger.warning(
            "模块B role3 未找到 %s，fallback 到占位数据。",
            module_a_path,
        )
    return build_big_segment_catalog_stub()


def _strip_remotion_section(text: str) -> str:
    """从 storyboard 模板中移除「## remotion模板」部分，该部分仅用于后续模块 C/D。"""
    return re.sub(r"\n## remotion模板\n.*", "", str(text or ""), flags=re.DOTALL).strip()


def _parse_remotion_catalog(text: str) -> dict[str, str]:
    """从 storyboard 模板中解析「## remotion模板」为 {模板ID: 模板描述块} 映射。"""
    section = re.search(r"\n## remotion模板\n(.*)", str(text or ""), flags=re.DOTALL)
    if not section:
        return {}
    catalog: dict[str, str] = {}
    for block in re.split(r"\n(?=### )", section.group(1).strip()):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"### (\S+)", block)
        if m:
            catalog[m.group(1)] = block
    return catalog


def _parse_visual_registry(role1_markdown: str) -> dict[str, str]:
    """解析 role1 输出为 {意象名: 完整##块} 映射。"""
    text = str(role1_markdown or "").replace("\r\n", "\n")
    # 提取 ```md ... ``` 内的内容
    m = re.search(r"```(?:md|markdown)?[ \t]*\n(.*?)\n[ \t]*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    registry: dict[str, str] = {}
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"## (.+)", block)
        if m:
            registry[m.group(1).strip()] = block
    return registry


def _filter_visual_reference(visual_registry: dict[str, str], scene_desc_zh: str) -> str:
    """用 jieba 分词匹配 scene_desc_zh 中的意象名，避免子串误匹配。
    返回时把各意象块的 ## 标题降级为 ### ，避免占用整体 prompt 的二级标题层级。"""
    if not visual_registry or not scene_desc_zh:
        return ""
    import jieba
    tokens = jieba.lcut(str(scene_desc_zh))
    # 构建连续 token 组合（2~4 元），覆盖多字意象名如"昏暗小巷"
    ngram_set = set()
    for n in range(2, min(len(tokens) + 1, 5)):
        for i in range(len(tokens) - n + 1):
            ngram_set.add("".join(tokens[i:i + n]))
    matched_blocks: list[str] = []
    for imagery_name, block in visual_registry.items():
        if imagery_name in tokens or imagery_name in ngram_set:
            # 把块内的 ## 降级为 ###
            downgraded = re.sub(r'^## ', '### ', block, flags=re.MULTILINE)
            matched_blocks.append(downgraded)
    return "\n\n".join(matched_blocks)
