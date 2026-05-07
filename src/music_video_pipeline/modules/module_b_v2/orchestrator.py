"""
 文件用途：实现模块B v2 多角色编排主入口与新模式生成器。
 核心流程：加载模板 -> 规则增强音频 -> 调度4角色 -> 写入单元产物 -> 聚合 module_b_output。
 输入输出：输入 RuntimeContext 或模块A输出，输出模块B标准分镜数组/产物路径。
 依赖说明：依赖旧模块B单元模型/输出聚合器、v2 各角色、模板加载器与 JSON 工具。
 维护说明：本文件是 v2 唯一主编排入口，不回写旧 module_b 逻辑。
 """

# 标准库：用于异步 future 调度。
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于目录删除。
import shutil
# 标准库：用于并发回调结果保护。
from threading import Lock
# 标准库：用于阶段耗时统计。
from time import perf_counter
# 标准库：用于类型提示。
from typing import Any, Callable

# 项目内模块：运行上下文。
from music_video_pipeline.context import RuntimeContext
# 项目内模块：模块B配置。
from music_video_pipeline.config import ModuleBConfig
# 项目内模块：通用 JSON 读写。
from music_video_pipeline.io_utils import ensure_dir, read_json, write_json
# 项目内模块：旧模块B输出聚合器。
from music_video_pipeline.modules.module_b.output_builder import build_module_b_output
# 项目内模块：旧模块B单元模型。
from music_video_pipeline.modules.module_b.unit_models import build_module_b_units, build_unit_map, build_unit_sync_payload
# 项目内模块：类型契约校验。
from music_video_pipeline.types import validate_module_a_output, validate_module_b_output
# 项目内模块：v2 音频规则。
from music_video_pipeline.modules.module_b_v2.audio_rules import build_segment_audio_features_v2
# 项目内模块：v2 LLM 运行时。
from music_video_pipeline.modules.module_b_v2.llm_runtime import ModuleBV2LlmRuntime
# 项目内模块：v2 输出校验。
from music_video_pipeline.modules.module_b_v2.parser import (
    validate_role1_visual_catalog_output,
    validate_role2_big_segment_story_output,
    validate_role3_segment_directing_output,
    validate_role4_prompt_output,
)
# 项目内模块：v2 角色实现。
from music_video_pipeline.modules.module_b_v2.role1_visual_director import Role1VisualDirector
# 项目内模块：v2 角色实现。
from music_video_pipeline.modules.module_b_v2.role2_big_segment_director import Role2BigSegmentDirector
# 项目内模块：v2 角色实现。
from music_video_pipeline.modules.module_b_v2.role3_segment_director import Role3SegmentDirector
# 项目内模块：v2 角色实现。
from music_video_pipeline.modules.module_b_v2.role4_prompt_builder import Role4PromptBuilder
# 项目内模块：v2 模板加载器。
from music_video_pipeline.modules.module_b_v2.template_loader import (
    dump_storyboard_template_artifact,
    load_storyboard_template,
    resolve_storyboard_template_path,
)


# 常量：模块B v2 允许的角色级重试起点。
VALID_MODULE_B_V2_ROLE_NAMES = ("role1", "role2", "role3", "role4")


ModuleBUnitCompletedCallback = Callable[[str, int, dict[str, Any], Path], None]


class MultiRoleScriptGeneratorV2:
    """
    功能说明：为模块B v2 提供完整多角色编排能力。
    参数说明：
    - logger: 日志对象。
    - module_b_config: 模块B配置。
    - project_root: 项目根目录；为空时自动推断。
    返回值：不适用。
    异常说明：初始化阶段不抛业务异常。
    边界条件：本类只服务 v2 主链路与重跑入口，不再挂接旧 ScriptGenerator 抽象。
    """

    def __init__(
        self,
        *,
        logger: Any,
        module_b_config: ModuleBConfig | None = None,
        project_root: Path | None = None,
        prompt_dump_dir: Path | None = None,
    ) -> None:
        self.logger = logger
        self.module_b_config = module_b_config or ModuleBConfig()
        self.project_root = project_root or Path(__file__).resolve().parents[4]
        self._llm_runtime = ModuleBV2LlmRuntime(
            logger=self.logger,
            llm_config=self.module_b_config.llm,
            project_root=self.project_root,
        )
        self._llm_runtime.set_prompt_dump_dir(prompt_dump_dir)

    def generate(self, module_a_output: dict[str, Any]) -> list[dict[str, Any]]:
        """
        功能说明：执行完整 v2 批量多角色编排，返回全部 shot 数组。
        参数说明：
        - module_a_output: 模块A输出。
        返回值：
        - list[dict[str, Any]]: 模块B输出数组。
        异常说明：模板、规则、LLM 或校验失败时抛出异常。
        边界条件：仅返回内存结果，不直接写状态库与文件。
        """
        storyboard_template = load_storyboard_template(
            project_root=self.project_root,
            template_file=str(self.module_b_config.storyboard_template_file),
        )
        return self.generate_with_template(
            module_a_output=module_a_output,
            storyboard_template=storyboard_template,
        )

    def generate_with_template(
        self,
        *,
        module_a_output: dict[str, Any],
        storyboard_template: dict[str, Any],
        target_shot_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        功能说明：在给定模板对象下执行完整 v2 多角色编排。
        参数说明：
        - module_a_output: 模块A输出。
        - storyboard_template: 已编译模板。
        - target_shot_ids: 可选，仅为这些 shot 重新生成提示词块并组装结果。
        返回值：
        - list[dict[str, Any]]: 模块B输出数组。
        异常说明：模板、规则、LLM 或校验失败时抛出异常。
        边界条件：target_shot_ids 不为空时，仍会全量跑角色1-3以保持上下文一致。
        """
        role_outputs = self.generate_role_outputs(
            module_a_output=module_a_output,
            storyboard_template=storyboard_template,
            target_shot_ids=target_shot_ids,
        )
        return role_outputs["module_b_output"]

    def generate_role_outputs(
        self,
        *,
        module_a_output: dict[str, Any],
        storyboard_template: dict[str, Any],
        target_shot_ids: set[str] | None = None,
        role1_output: dict[str, Any] | None = None,
        role2_output: dict[str, Any] | None = None,
        existing_role3_shots: dict[str, dict[str, Any]] | None = None,
        existing_role4_shots: dict[str, dict[str, Any]] | None = None,
        on_role3_shot_completed: Callable[[dict[str, Any]], None] | None = None,
        on_role4_shot_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        功能说明：执行 4 角色链路并返回中间产物与最终 shot 结果。
        参数说明：
        - module_a_output: 模块A输出。
        - storyboard_template: 已编译模板。
        - target_shot_ids: 可选，仅要求这些 shot 在本轮输出中覆盖。
        - role1_output/role2_output: 可选，外部已缓存的 role 级产物。
        - existing_role3_shots/existing_role4_shots: 可选，已缓存的 shot 级产物。
        - on_role3_shot_completed/on_role4_shot_completed: 可选，单 shot 成功后的即时持久化回调。
        返回值：
        - dict[str, Any]: 包含 role1/2/3/4 输出、音频增强结果与最终分镜数组。
        异常说明：任一角色失败时抛出异常。
        边界条件：最终 shot 顺序与模块A segments 顺序一致；role3/4 支持按 shot 复用已有产物。
        """
        total_started_at = perf_counter()
        segments = [dict(item) for item in module_a_output.get("segments", []) if isinstance(item, dict)]
        for index, segment in enumerate(segments):
            segment["_global_index"] = index
        normalized_module_a_output = dict(module_a_output)
        normalized_module_a_output["segments"] = segments
        all_shot_ids = [f"shot_{index + 1:03d}" for index in range(len(segments))]
        required_shot_id_set = set(target_shot_ids or all_shot_ids)
        required_shot_ids = [shot_id for shot_id in all_shot_ids if shot_id in required_shot_id_set]
        cached_role3_shots = {
            str(shot_id).strip(): dict(shot_payload)
            for shot_id, shot_payload in (existing_role3_shots or {}).items()
            if str(shot_id).strip() and isinstance(shot_payload, dict)
        }
        cached_role4_shots = {
            str(shot_id).strip(): dict(shot_payload)
            for shot_id, shot_payload in (existing_role4_shots or {}).items()
            if str(shot_id).strip() and isinstance(shot_payload, dict)
        }

        role1_started_at = perf_counter()
        if role1_output is None:
            role1 = Role1VisualDirector(self._llm_runtime).generate(storyboard_template=storyboard_template)
        else:
            role1 = dict(role1_output)
            self.logger.info("模块B v2 role1 复用缓存结果")
        role1_elapsed = perf_counter() - role1_started_at
        self.logger.info("模块B v2 role1 耗时统计，elapsed_seconds=%.3f", role1_elapsed)

        role2_started_at = perf_counter()
        if role2_output is None:
            role2 = Role2BigSegmentDirector(self._llm_runtime).generate(
                module_a_output=normalized_module_a_output,
                storyboard_template=storyboard_template,
            )
        else:
            role2 = dict(role2_output)
            self.logger.info("模块B v2 role2 复用缓存结果")
        role2_elapsed = perf_counter() - role2_started_at
        self.logger.info("模块B v2 role2 耗时统计，elapsed_seconds=%.3f", role2_elapsed)

        audio_rules_started_at = perf_counter()
        segment_audio_features = build_segment_audio_features_v2(
            module_a_output=normalized_module_a_output,
            storyboard_template=storyboard_template,
        )
        audio_rules_elapsed = perf_counter() - audio_rules_started_at
        self.logger.info("模块B v2 音频规则增强耗时统计，elapsed_seconds=%.3f", audio_rules_elapsed)
        role4_builder = Role4PromptBuilder(self._llm_runtime)
        role4_generation_context = role4_builder.build_generation_context(
            storyboard_template=storyboard_template,
            role1_output=role1,
        )
        role4_results: dict[str, dict[str, Any]] = {}
        role4_errors: dict[str, str] = {}
        role4_result_lock = Lock()
        role4_started_at: float | None = None
        role4_finished_at: float | None = None
        role4_future_map: dict[Future[dict[str, Any]], str] = {}
        submitted_role4_shot_ids: set[str] = set()

        def _submit_role4_job(shot_item: dict[str, Any], executor: ThreadPoolExecutor) -> None:
            """
            功能说明：将单个 role3 shot 异步投递到独立 role4 消费池。
            参数说明：
            - shot_item: 已完成的角色3单镜头结果。
            - executor: role4 消费线程池。
            返回值：无。
            异常说明：线程池提交失败时向上抛出。
            边界条件：target_shot_ids 不为空时仅处理目标 shot。
            """
            shot_id = str(shot_item.get("shot_id", "")).strip()
            if shot_id not in required_shot_id_set:
                return
            if shot_id in cached_role4_shots:
                return

            nonlocal role4_started_at
            submitted_at = perf_counter()
            with role4_result_lock:
                if shot_id in submitted_role4_shot_ids:
                    return
                submitted_role4_shot_ids.add(shot_id)
                if role4_started_at is None:
                    role4_started_at = submitted_at
                future = executor.submit(
                    role4_builder.generate_one,
                    storyboard_template=storyboard_template,
                    shot_item=shot_item,
                    generation_context=role4_generation_context,
                )
                role4_future_map[future] = shot_id

        role3_started_at = perf_counter()
        role3_error: Exception | None = None
        role3: dict[str, Any] = {"shots": []}
        with ThreadPoolExecutor(
            max_workers=max(1, len(required_shot_ids) or 1)
        ) as role4_executor:
            for shot_id in required_shot_ids:
                if shot_id in cached_role3_shots and shot_id not in cached_role4_shots:
                    _submit_role4_job(cached_role3_shots[shot_id], role4_executor)
            try:
                role3 = Role3SegmentDirector(self._llm_runtime).generate(
                    module_a_output=normalized_module_a_output,
                    storyboard_template=storyboard_template,
                    role2_output=role2,
                    segment_audio_features=segment_audio_features,
                    target_shot_ids=required_shot_id_set,
                    existing_shots=cached_role3_shots,
                    on_shot_completed=lambda shot_item: _handle_role3_shot_completed(
                        shot_item=shot_item,
                        on_role3_shot_completed=on_role3_shot_completed,
                        submit_role4_job=lambda payload: _submit_role4_job(payload, role4_executor),
                    ),
                )
            except Exception as error:  # noqa: BLE001
                role3_error = error
            role3_elapsed = perf_counter() - role3_started_at
            self.logger.info("模块B v2 role3 耗时统计，elapsed_seconds=%.3f", role3_elapsed)
            future_lookup = dict(role4_future_map)
            for future in as_completed(list(future_lookup)):
                shot_id = future_lookup[future]
                try:
                    prompt_block = future.result()
                    role4_results[shot_id] = dict(prompt_block)
                    if on_role4_shot_completed is not None:
                        on_role4_shot_completed(dict(prompt_block))
                except Exception as error:  # noqa: BLE001
                    role4_errors[shot_id] = str(error)
                    self.logger.error("模块B v2 role4 执行失败，shot_id=%s，错误=%s", shot_id, error)
                finally:
                    with role4_result_lock:
                        role4_finished_at = perf_counter()
        if role3_error is not None:
            raise role3_error

        combined_role4_shot_map = {
            shot_id: dict(shot_payload)
            for shot_id, shot_payload in cached_role4_shots.items()
            if shot_id in required_shot_id_set
        }
        combined_role4_shot_map.update(role4_results)
        if role4_errors:
            raise RuntimeError(f"模块B v2 role4 存在失败 shot，failed_shot_ids={sorted(role4_errors)}")
        role4 = validate_role4_prompt_output(
            data={
                "shots": [
                    combined_role4_shot_map[shot_id]
                    for shot_id in required_shot_ids
                    if shot_id in combined_role4_shot_map
                ]
            },
            shot_ids=required_shot_ids,
        )
        role4_elapsed = (
            max(0.0, float(role4_finished_at) - float(role4_started_at))
            if role4_started_at is not None and role4_finished_at is not None
            else 0.0
        )
        self.logger.info(
            "模块B v2 role4 耗时统计，elapsed_seconds=%.3f，shot_count=%s，说明=role4 与 role3 重叠执行，wall time 不可直接与其他 role 相加",
            role4_elapsed,
            len(combined_role4_shot_map),
        )
        final_output = _assemble_module_b_output(
            module_a_output=normalized_module_a_output,
            role3_output=role3,
            role4_output=role4,
        )
        validate_module_b_output(final_output)
        total_elapsed = perf_counter() - total_started_at
        self.logger.info("模块B v2 角色链路总耗时统计，elapsed_seconds=%.3f", total_elapsed)
        return {
            "storyboard_template": storyboard_template,
            "segment_audio_features": segment_audio_features,
            "role1_output": role1,
            "role2_output": role2,
            "role3_output": role3,
            "role4_output": role4,
            "module_b_output": final_output,
        }


def _build_module_b_v2_artifact_dirs(artifacts_dir: Path) -> dict[str, Path]:
    """
    功能说明：构建模块B v2 专属产物目录结构。
    参数说明：
    - artifacts_dir: 任务 artifacts 根目录。
    返回值：
    - dict[str, Path]: 目录名到路径的映射。
    异常说明：无。
    边界条件：目录会在返回前确保存在。
    """
    module_b_v2_root = artifacts_dir / "module_b_v2"
    prompt_dir = module_b_v2_root / "llm_prompts"
    output_dir = module_b_v2_root / "artifacts"
    unit_outputs_dir = output_dir / "module_b_units"
    role3_shots_dir = output_dir / "role3_shots"
    role4_shots_dir = output_dir / "role4_shots"
    for path in [module_b_v2_root, prompt_dir, output_dir, unit_outputs_dir, role3_shots_dir, role4_shots_dir]:
        ensure_dir(path)
    return {
        "root": module_b_v2_root,
        "prompt_dir": prompt_dir,
        "output_dir": output_dir,
        "unit_outputs_dir": unit_outputs_dir,
        "role3_shots_dir": role3_shots_dir,
        "role4_shots_dir": role4_shots_dir,
    }


def _handle_role3_shot_completed(
    *,
    shot_item: dict[str, Any],
    on_role3_shot_completed: Callable[[dict[str, Any]], None] | None,
    submit_role4_job: Callable[[dict[str, Any]], None],
) -> None:
    """
    功能说明：统一处理 role3 单 shot 成功后的持久化与 role4 投递。
    参数说明：
    - shot_item: role3 单镜头结果。
    - on_role3_shot_completed: role3 外部持久化回调。
    - submit_role4_job: role4 投递函数。
    返回值：无。
    异常说明：回调抛错时向上抛出，避免静默吞错。
    边界条件：先持久化 role3，再投递 role4，保证断点恢复最小单位落盘。
    """
    if on_role3_shot_completed is not None:
        on_role3_shot_completed(dict(shot_item))
    submit_role4_job(dict(shot_item))


def _read_validated_artifact(
    *,
    artifact_path: Path,
    artifact_name: str,
    logger: Any,
    validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any] | None:
    """
    功能说明：读取并校验单个聚合 JSON 产物，失败时返回空以触发重算。
    参数说明：
    - artifact_path: 产物路径。
    - artifact_name: 产物名，用于日志。
    - logger: 日志对象。
    - validator: 校验函数。
    返回值：
    - dict[str, Any] | None: 校验通过则返回标准化对象，否则返回空。
    异常说明：无，内部吞掉坏文件并记录 warning。
    边界条件：仅用于 role1/role2 这类可整体复用的聚合产物。
    """
    if not artifact_path.exists():
        return None
    try:
        return validator(read_json(artifact_path))
    except Exception as error:  # noqa: BLE001
        logger.warning("模块B v2 缓存产物失效，artifact=%s，path=%s，错误=%s", artifact_name, artifact_path, error)
        return None


def _load_role_shot_cache(
    *,
    shot_dir: Path,
    shot_ids: list[str],
    role_name: str,
    logger: Any,
    validator: Callable[[dict[str, Any], str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    功能说明：读取并校验 role3/role4 的 shot 级缓存文件。
    参数说明：
    - shot_dir: shot 级缓存目录。
    - shot_ids: 允许加载的 shot_id 顺序。
    - role_name: role 名称，用于日志。
    - logger: 日志对象。
    - validator: 单 shot 校验器。
    返回值：
    - dict[str, dict[str, Any]]: shot_id 到标准化 payload 的映射。
    异常说明：无，坏文件仅跳过并记录 warning。
    边界条件：只读取显式给定的 shot_id，避免误吃历史脏文件。
    """
    shot_map: dict[str, dict[str, Any]] = {}
    for shot_id in shot_ids:
        artifact_path = shot_dir / f"{shot_id}.json"
        if not artifact_path.exists():
            continue
        try:
            shot_map[shot_id] = validator(read_json(artifact_path), shot_id)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "模块B v2 %s 缓存 shot 失效，shot_id=%s，path=%s，错误=%s",
                role_name,
                shot_id,
                artifact_path,
                error,
            )
    return shot_map


def _write_role_shot_aggregate(
    *,
    artifact_path: Path,
    key_name: str,
    shot_ids: list[str],
    shot_map: dict[str, dict[str, Any]],
) -> None:
    """
    功能说明：将 shot 级缓存聚合写回 role 级 JSON 产物。
    参数说明：
    - artifact_path: 聚合产物路径。
    - key_name: 顶层数组字段名。
    - shot_ids: 期望输出顺序。
    - shot_map: shot 映射。
    返回值：无。
    异常说明：写文件失败时向上抛出。
    边界条件：允许部分缺失；仅聚合当前已有 shot。
    """
    write_json(
        artifact_path,
        {
            key_name: [dict(shot_map[shot_id]) for shot_id in shot_ids if shot_id in shot_map],
        },
    )


def _persist_role3_and_maybe_emit_unit(
    *,
    shot_item: dict[str, Any],
    role3_shot_cache: dict[str, dict[str, Any]],
    role3_shots_dir: Path,
) -> None:
    """
    功能说明：持久化 role3 shot 并更新内存缓存。
    参数说明：
    - shot_item: role3 单镜头结果。
    - role3_shot_cache: role3 shot 内存缓存。
    - role3_shots_dir: role3 shot 落盘目录。
    返回值：无。
    异常说明：写文件失败时向上抛出。
    边界条件：仅更新 role3 缓存，不直接判定单元完成。
    """
    shot_id = str(shot_item.get("shot_id", "")).strip()
    if not shot_id:
        return
    role3_shot_cache[shot_id] = dict(shot_item)
    write_json(role3_shots_dir / f"{shot_id}.json", shot_item)


def _persist_role4_and_emit_unit_if_ready(
    *,
    shot_item: dict[str, Any],
    role4_shot_cache: dict[str, dict[str, Any]],
    role4_shots_dir: Path,
    emit_completed_unit: Callable[[str], None],
) -> None:
    """
    功能说明：持久化 role4 shot，并在成品条件满足时触发单元回调。
    参数说明：
    - shot_item: role4 单镜头结果。
    - role4_shot_cache: role4 shot 内存缓存。
    - role4_shots_dir: role4 shot 落盘目录。
    - emit_completed_unit: 成品回调函数。
    返回值：无。
    异常说明：写文件失败时向上抛出。
    边界条件：仅当 role3 与 role4 均齐备时才触发单元成品输出。
    """
    shot_id = str(shot_item.get("shot_id", "")).strip()
    if not shot_id:
        return
    role4_shot_cache[shot_id] = dict(shot_item)
    write_json(role4_shots_dir / f"{shot_id}.json", shot_item)
    emit_completed_unit(shot_id)


def invalidate_module_b_v2_role_outputs(*, task_dir: Path, role_name: str, logger: Any | None = None) -> dict[str, Any]:
    """
    功能说明：按角色级起点失效 module_b_v2 内部缓存，供断电恢复或定向重试复用。
    参数说明：
    - task_dir: 任务目录路径（runs/<task_id>）。
    - role_name: 角色名，仅允许 role1/role2/role3/role4。
    - logger: 可选日志对象，用于记录删除动作。
    返回值：
    - dict[str, Any]: 本次实际清理的路径摘要。
    异常说明：
    - ValueError: 角色名非法时抛出。
    边界条件：只失效模块B自身及其内部角色缓存，不直接触碰状态库。
    """
    normalized_role_name = str(role_name).strip().lower()
    if normalized_role_name not in VALID_MODULE_B_V2_ROLE_NAMES:
        raise ValueError(
            f"非法模块B v2 角色名: {role_name}，合法值={list(VALID_MODULE_B_V2_ROLE_NAMES)}"
        )

    artifacts_dir = task_dir / "artifacts"
    v2_dirs = _build_module_b_v2_artifact_dirs(artifacts_dir)
    removed_paths: list[str] = []

    def _remove_file(file_path: Path) -> None:
        """
        功能说明：删除单个文件并记录摘要。
        参数说明：
        - file_path: 目标文件路径。
        返回值：无。
        异常说明：删除失败时向上抛出。
        边界条件：文件不存在时静默跳过。
        """
        if not file_path.exists():
            return
        file_path.unlink()
        removed_paths.append(str(file_path))

    def _remove_dir_children(dir_path: Path) -> None:
        """
        功能说明：清空目录下所有直接子项，保留目录本身。
        参数说明：
        - dir_path: 目标目录。
        返回值：无。
        异常说明：删除失败时向上抛出。
        边界条件：目录不存在时静默跳过。
        """
        if not dir_path.exists():
            return
        for child in sorted(dir_path.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed_paths.append(str(child))

    # 最终模块B输出与单元产物从任意 role 起重跑时都应失效。
    _remove_file(artifacts_dir / "module_b_output.json")
    _remove_dir_children(v2_dirs["unit_outputs_dir"])

    output_dir = v2_dirs["output_dir"]
    prompt_dir = v2_dirs["prompt_dir"]

    if normalized_role_name == "role1":
        _remove_file(output_dir / "module_b_role1_visual_catalog.json")
        _remove_file(output_dir / "module_b_role4_prompt_blocks.json")
        _remove_dir_children(v2_dirs["role4_shots_dir"])
        for prompt_file in sorted(prompt_dir.glob("role1_visual_director*.prompt.md")):
            _remove_file(prompt_file)
        for prompt_file in sorted(prompt_dir.glob("role4_prompt_builder*.prompt.md")):
            _remove_file(prompt_file)
    elif normalized_role_name == "role2":
        _remove_file(output_dir / "module_b_role2_big_segment_story.json")
        _remove_file(output_dir / "module_b_role3_segment_directing.json")
        _remove_file(output_dir / "module_b_role4_prompt_blocks.json")
        _remove_dir_children(v2_dirs["role3_shots_dir"])
        _remove_dir_children(v2_dirs["role4_shots_dir"])
        for pattern in (
            "role2_big_segment_director*.prompt.md",
            "role3_segment_director*.prompt.md",
            "role4_prompt_builder*.prompt.md",
        ):
            for prompt_file in sorted(prompt_dir.glob(pattern)):
                _remove_file(prompt_file)
    elif normalized_role_name == "role3":
        _remove_file(output_dir / "module_b_role3_segment_directing.json")
        _remove_file(output_dir / "module_b_role4_prompt_blocks.json")
        _remove_dir_children(v2_dirs["role3_shots_dir"])
        _remove_dir_children(v2_dirs["role4_shots_dir"])
        for pattern in ("role3_segment_director*.prompt.md", "role4_prompt_builder*.prompt.md"):
            for prompt_file in sorted(prompt_dir.glob(pattern)):
                _remove_file(prompt_file)
    else:
        _remove_file(output_dir / "module_b_role4_prompt_blocks.json")
        _remove_dir_children(v2_dirs["role4_shots_dir"])
        for prompt_file in sorted(prompt_dir.glob("role4_prompt_builder*.prompt.md")):
            _remove_file(prompt_file)

    summary = {
        "role_name": normalized_role_name,
        "removed_path_count": len(removed_paths),
        "removed_paths": removed_paths,
    }
    if logger is not None:
        logger.info(
            "模块B v2 已按角色失效缓存，role_name=%s，removed_path_count=%s",
            normalized_role_name,
            len(removed_paths),
        )
    return summary


def invalidate_module_b_v2_role_shot_outputs(
    *,
    task_dir: Path,
    role_name: str,
    shot_id: str,
    logger: Any | None = None,
) -> dict[str, Any]:
    """
    功能说明：按角色内单 shot 失效 module_b_v2 内部缓存，供最细粒度断点重试复用。
    参数说明：
    - task_dir: 任务目录路径（runs/<task_id>）。
    - role_name: 角色名，仅允许 role3 或 role4。
    - shot_id: 目标 shot_id。
    - logger: 可选日志对象，用于记录删除动作。
    返回值：
    - dict[str, Any]: 本次实际清理的路径摘要。
    异常说明：
    - ValueError: 角色名非法或 shot_id 为空时抛出。
    边界条件：只失效对应 shot 的角色缓存；role3 shot 失效会级联失效同 shot 的 role4。
    """
    normalized_role_name = str(role_name).strip().lower()
    normalized_shot_id = str(shot_id).strip()
    if normalized_role_name not in {"role3", "role4"}:
        raise ValueError(f"非法模块B v2 shot级角色名: {role_name}，合法值=['role3', 'role4']")
    if not normalized_shot_id:
        raise ValueError("shot_id 不能为空。")

    artifacts_dir = task_dir / "artifacts"
    v2_dirs = _build_module_b_v2_artifact_dirs(artifacts_dir)
    removed_paths: list[str] = []

    def _remove_file(file_path: Path) -> None:
        if not file_path.exists():
            return
        file_path.unlink()
        removed_paths.append(str(file_path))

    _remove_file(artifacts_dir / "module_b_output.json")
    if normalized_role_name == "role3":
        _remove_file(v2_dirs["output_dir"] / "module_b_role3_segment_directing.json")
        _remove_file(v2_dirs["role3_shots_dir"] / f"{normalized_shot_id}.json")
        for prompt_file in sorted(v2_dirs["prompt_dir"].glob(f"role3_segment_director_{normalized_shot_id}.attempt_*.prompt.md")):
            _remove_file(prompt_file)

    _remove_file(v2_dirs["output_dir"] / "module_b_role4_prompt_blocks.json")
    _remove_file(v2_dirs["role4_shots_dir"] / f"{normalized_shot_id}.json")
    for prompt_file in sorted(v2_dirs["prompt_dir"].glob(f"role4_prompt_builder_{normalized_shot_id}.attempt_*.prompt.md")):
        _remove_file(prompt_file)

    summary = {
        "role_name": normalized_role_name,
        "shot_id": normalized_shot_id,
        "removed_path_count": len(removed_paths),
        "removed_paths": removed_paths,
    }
    if logger is not None:
        logger.info(
            "模块B v2 已按角色内 shot 失效缓存，role_name=%s，shot_id=%s，removed_path_count=%s",
            normalized_role_name,
            normalized_shot_id,
            len(removed_paths),
        )
    return summary


def run_module_b_v2(context: RuntimeContext) -> Path:
    """
    功能说明：执行模块B v2，并以旧状态机兼容的方式落盘单元级产物与聚合输出。
    参数说明：
    - context: 运行上下文对象。
    返回值：
    - Path: module_b_output.json 路径。
    异常说明：模板、规则、LLM、落盘或聚合校验失败时抛出异常。
    边界条件：done 单元复用已有产物，仅对 pending/failed/running 单元重新写出。
    """
    context.logger.info("模块B v2 开始执行，task_id=%s，mode=multi_role_llm_v2", context.task_id)
    return _run_module_b_v2_internal(context=context)


def run_module_b_v2_incremental(
    context: RuntimeContext,
    *,
    target_segment_ids: set[str] | None = None,
    on_unit_completed: ModuleBUnitCompletedCallback | None = None,
) -> Path:
    """
    功能说明：以增量方式执行模块B v2，并在单元完成时立刻回调成品产物。
    参数说明：
    - context: 运行上下文对象。
    - target_segment_ids: 可选，仅处理这些 segment_id 对应的模块B单元。
    - on_unit_completed: 可选，单元完成后的即时回调。
    返回值：
    - Path: module_b_output.json 路径。
    异常说明：模板、规则、LLM、落盘或聚合校验失败时抛出异常。
    边界条件：未命中的非目标单元不会被重置；已完成缓存会被直接复用并触发回调。
    """
    normalized_target_segment_ids = {
        str(segment_id).strip() for segment_id in (target_segment_ids or set()) if str(segment_id).strip()
    }
    return _run_module_b_v2_internal(
        context=context,
        target_segment_ids=normalized_target_segment_ids or None,
        on_unit_completed=on_unit_completed,
    )


def _run_module_b_v2_internal(
    *,
    context: RuntimeContext,
    target_segment_ids: set[str] | None = None,
    on_unit_completed: ModuleBUnitCompletedCallback | None = None,
) -> Path:
    """
    功能说明：执行模块B v2 内部主流程，可选按目标单元增量产出并回调。
    参数说明：
    - context: 运行上下文对象。
    - target_segment_ids: 可选，限制本轮处理的 segment_id 集合。
    - on_unit_completed: 可选，单元完成后的即时回调。
    返回值：
    - Path: module_b_output.json 路径。
    异常说明：模板、规则、LLM、落盘或聚合校验失败时抛出异常。
    边界条件：当目标集合不为空时，仅为目标单元写 module_b_units 单元产物与状态。
    """
    module_a_path = context.artifacts_dir / "module_a_output.json"
    module_a_output = read_json(module_a_path)
    validate_module_a_output(module_a_output)

    units = build_module_b_units(module_a_output=module_a_output)
    context.state_store.sync_module_units(
        task_id=context.task_id,
        module_name="B",
        units=build_unit_sync_payload(units=units),
    )
    units_by_id = build_unit_map(units=units)
    pending_records = context.state_store.list_module_units_by_status(
        task_id=context.task_id,
        module_name="B",
        statuses=["pending", "failed", "running"],
    )
    pending_unit_ids = {
        str(record.get("unit_id", "")).strip()
        for record in pending_records
        if str(record.get("unit_id", "")).strip()
    }
    if target_segment_ids is not None:
        pending_unit_ids = {unit_id for unit_id in pending_unit_ids if unit_id in target_segment_ids}
    all_shot_ids = [f"shot_{index + 1:03d}" for index in range(len(units))]
    target_shot_ids = [
        f"shot_{units_by_id[unit_id].unit_index + 1:03d}"
        for unit_id in pending_unit_ids
        if unit_id in units_by_id
    ]
    context.logger.info(
        "模块B v2 单元调度计划，task_id=%s，unit_total=%s，unit_to_run=%s",
        context.task_id,
        len(units),
        len(pending_unit_ids),
    )
    if not pending_unit_ids:
        if target_segment_ids is not None:
            done_unit_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
            module_b_output = build_module_b_output(
                done_unit_records=done_unit_records,
                module_a_output=module_a_output,
                instrumental_labels=context.config.module_a.instrumental_labels,
            )
            validate_module_b_output(module_b_output)
            output_path = context.artifacts_dir / "module_b_output.json"
            write_json(output_path, module_b_output)
            context.logger.info(
                "模块B v2 目标单元已全部完成，直接复用现有结果，task_id=%s，target_count=%s，输出=%s",
                context.task_id,
                len(target_segment_ids),
                output_path,
            )
            return output_path
        done_unit_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
        if len(done_unit_records) != len(units):
            done_unit_ids = {str(item["unit_id"]) for item in done_unit_records}
            missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_unit_ids]
            raise RuntimeError(f"模块B v2 执行失败：无待跑单元但仍缺失 done 单元，missing_unit_ids={missing_unit_ids}")
        module_b_output = build_module_b_output(
            done_unit_records=done_unit_records,
            module_a_output=module_a_output,
            instrumental_labels=context.config.module_a.instrumental_labels,
        )
        validate_module_b_output(module_b_output)
        output_path = context.artifacts_dir / "module_b_output.json"
        write_json(output_path, module_b_output)
        context.logger.info("模块B v2 无待执行单元，直接复用已完成结果，task_id=%s，输出=%s", context.task_id, output_path)
        return output_path

    v2_dirs = _build_module_b_v2_artifact_dirs(context.artifacts_dir)

    resolved_template_path = resolve_storyboard_template_path(
        project_root=Path.cwd(),
        template_file=str(context.config.module_b.storyboard_template_file),
    )
    storyboard_template = load_storyboard_template(
        project_root=Path.cwd(),
        template_file=str(resolved_template_path),
    )
    dump_storyboard_template_artifact(
        template_payload=storyboard_template,
        artifact_path=v2_dirs["output_dir"] / "module_b_storyboard_template.json",
    )
    write_json(
        v2_dirs["output_dir"] / "module_b_storyboard_template_meta.json",
        {
            "template_file": str(resolved_template_path),
            "template_id": str(storyboard_template.get("template_id", "")).strip(),
        },
    )
    scene_ids = [
        str(item.get("item_id", "")).strip()
        for item in storyboard_template.get("scene_catalog", [])
        if isinstance(item, dict)
    ]
    prop_ids = [
        str(item.get("item_id", "")).strip()
        for item in storyboard_template.get("prop_catalog", [])
        if isinstance(item, dict)
    ]
    character_ids = [
        str(item.get("item_id", "")).strip()
        for item in storyboard_template.get("character_catalog", [])
        if isinstance(item, dict)
    ]
    composition_ids = [
        str(item.get("composition_id", "")).strip()
        for item in storyboard_template.get("composition_catalog", [])
        if isinstance(item, dict)
    ]
    big_segment_ids = [
        str(item.get("segment_id", "")).strip()
        for item in module_a_output.get("big_segments", [])
        if isinstance(item, dict)
    ]
    reusable_role1_output = _read_validated_artifact(
        artifact_path=v2_dirs["output_dir"] / "module_b_role1_visual_catalog.json",
        artifact_name="role1",
        logger=context.logger,
        validator=lambda data: validate_role1_visual_catalog_output(
            data=data,
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
        ),
    )
    reusable_role2_output = _read_validated_artifact(
        artifact_path=v2_dirs["output_dir"] / "module_b_role2_big_segment_story.json",
        artifact_name="role2",
        logger=context.logger,
        validator=lambda data: validate_role2_big_segment_story_output(
            data=data,
            big_segment_ids=big_segment_ids,
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
        ),
    )
    existing_role3_shots = _load_role_shot_cache(
        shot_dir=v2_dirs["role3_shots_dir"],
        shot_ids=all_shot_ids,
        role_name="role3",
        logger=context.logger,
        validator=lambda data, shot_id: validate_role3_segment_directing_output(
            data={"shots": [data]},
            shot_ids=[shot_id],
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
            composition_ids=composition_ids,
        )["shots"][0],
    )
    existing_role4_shots = _load_role_shot_cache(
        shot_dir=v2_dirs["role4_shots_dir"],
        shot_ids=target_shot_ids,
        role_name="role4",
        logger=context.logger,
        validator=lambda data, shot_id: validate_role4_prompt_output(
            data={"shots": [data]},
            shot_ids=[shot_id],
        )["shots"][0],
    )
    completed_unit_ids: set[str] = set()

    def _emit_completed_unit_if_ready(shot_id: str) -> None:
        if shot_id not in role3_shot_cache or shot_id not in role4_shot_cache:
            return
        unit = shot_unit_map.get(shot_id)
        if unit is None or unit.unit_id in completed_unit_ids:
            return
        shot_payload = _assemble_module_b_output_item(
            segment=segment_by_shot_id[shot_id],
            directing=role3_shot_cache[shot_id],
            prompt_block=role4_shot_cache[shot_id],
            shot_id=shot_id,
        )
        artifact_path = v2_dirs["unit_outputs_dir"] / f"{unit.unit_id}.json"
        write_json(artifact_path, shot_payload)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(artifact_path),
            error_message="",
        )
        completed_unit_ids.add(unit.unit_id)
        if on_unit_completed is not None:
            on_unit_completed(unit.unit_id, int(unit.unit_index), dict(shot_payload), artifact_path)

    role3_shot_cache = dict(existing_role3_shots)
    role4_shot_cache = dict(existing_role4_shots)
    shot_unit_map = {
        f"shot_{unit.unit_index + 1:03d}": unit
        for unit_id, unit in units_by_id.items()
        if unit_id in pending_unit_ids
    }
    segment_by_shot_id = {
        f"shot_{index + 1:03d}": dict(segment)
        for index, segment in enumerate(module_a_output.get("segments", []))
        if isinstance(segment, dict) and f"shot_{index + 1:03d}" in shot_unit_map
    }
    for cached_shot_id in sorted(role4_shot_cache):
        _emit_completed_unit_if_ready(cached_shot_id)
    generator = MultiRoleScriptGeneratorV2(
        logger=context.logger,
        module_b_config=context.config.module_b,
        project_root=Path.cwd(),
        prompt_dump_dir=v2_dirs["prompt_dir"],
    )
    role_outputs = generator.generate_role_outputs(
        module_a_output=module_a_output,
        storyboard_template=storyboard_template,
        target_shot_ids=set(target_shot_ids),
        role1_output=reusable_role1_output,
        role2_output=reusable_role2_output,
        existing_role3_shots=role3_shot_cache,
        existing_role4_shots=role4_shot_cache,
        on_role3_shot_completed=lambda shot_item: _persist_role3_and_maybe_emit_unit(
            shot_item=shot_item,
            role3_shot_cache=role3_shot_cache,
            role3_shots_dir=v2_dirs["role3_shots_dir"],
        ),
        on_role4_shot_completed=lambda shot_item: _persist_role4_and_emit_unit_if_ready(
            shot_item=shot_item,
            role4_shot_cache=role4_shot_cache,
            role4_shots_dir=v2_dirs["role4_shots_dir"],
            emit_completed_unit=_emit_completed_unit_if_ready,
        ),
    )
    write_json(v2_dirs["output_dir"] / "module_b_role1_visual_catalog.json", role_outputs["role1_output"])
    write_json(v2_dirs["output_dir"] / "module_b_role2_big_segment_story.json", role_outputs["role2_output"])
    write_json(v2_dirs["output_dir"] / "module_b_segment_audio_features_v2.json", role_outputs["segment_audio_features"])
    aggregated_role3_shots = _load_role_shot_cache(
        shot_dir=v2_dirs["role3_shots_dir"],
        shot_ids=all_shot_ids,
        role_name="role3",
        logger=context.logger,
        validator=lambda data, shot_id: validate_role3_segment_directing_output(
            data={"shots": [data]},
            shot_ids=[shot_id],
            scene_ids=scene_ids,
            prop_ids=prop_ids,
            character_ids=character_ids,
            composition_ids=composition_ids,
        )["shots"][0],
    )
    aggregated_role4_shots = _load_role_shot_cache(
        shot_dir=v2_dirs["role4_shots_dir"],
        shot_ids=all_shot_ids,
        role_name="role4",
        logger=context.logger,
        validator=lambda data, shot_id: validate_role4_prompt_output(
            data={"shots": [data]},
            shot_ids=[shot_id],
        )["shots"][0],
    )
    _write_role_shot_aggregate(
        artifact_path=v2_dirs["output_dir"] / "module_b_role3_segment_directing.json",
        key_name="shots",
        shot_ids=all_shot_ids,
        shot_map=aggregated_role3_shots,
    )
    _write_role_shot_aggregate(
        artifact_path=v2_dirs["output_dir"] / "module_b_role4_prompt_blocks.json",
        key_name="shots",
        shot_ids=all_shot_ids,
        shot_map=aggregated_role4_shots,
    )

    shot_map = {
        str(item.get("shot_id", "")).strip(): dict(item)
        for item in role_outputs["module_b_output"]
        if isinstance(item, dict)
    }
    for unit_id in pending_unit_ids:
        unit = units_by_id.get(unit_id)
        if unit is None:
            continue
        shot_id = f"shot_{unit.unit_index + 1:03d}"
        shot_payload = shot_map.get(shot_id)
        if not shot_payload:
            raise RuntimeError(f"模块B v2 写单元失败：缺失目标 shot 结果，unit_id={unit_id}，shot_id={shot_id}")
        artifact_path = v2_dirs["unit_outputs_dir"] / f"{unit.unit_id}.json"
        write_json(artifact_path, shot_payload)
        context.state_store.set_module_unit_status(
            task_id=context.task_id,
            module_name="B",
            unit_id=unit.unit_id,
            status="done",
            artifact_path=str(artifact_path),
            error_message="",
        )
        if unit_id in completed_unit_ids:
            continue
        completed_unit_ids.add(unit_id)
        if on_unit_completed is not None:
            on_unit_completed(unit_id, int(unit.unit_index), dict(shot_payload), artifact_path)

    done_unit_records = context.state_store.list_module_b_done_shot_items(task_id=context.task_id)
    if len(done_unit_records) != len(units):
        done_unit_ids = {str(item["unit_id"]) for item in done_unit_records}
        missing_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in done_unit_ids]
        raise RuntimeError(f"模块B v2 执行失败：存在未完成单元，missing_unit_ids={missing_unit_ids}")
    module_b_output = build_module_b_output(
        done_unit_records=done_unit_records,
        module_a_output=module_a_output,
        instrumental_labels=context.config.module_a.instrumental_labels,
    )
    validate_module_b_output(module_b_output)
    output_path = context.artifacts_dir / "module_b_output.json"
    write_json(output_path, module_b_output)
    context.logger.info("模块B v2 执行完成，task_id=%s，输出=%s", context.task_id, output_path)
    return output_path


def _assemble_module_b_output(
    *,
    module_a_output: dict[str, Any],
    role3_output: dict[str, Any],
    role4_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    功能说明：将角色3与角色4结果装配为最终模块B输出。
    参数说明：
    - module_a_output: 模块A输出。
    - role3_output: 角色3输出。
    - role4_output: 角色4输出。
    返回值：
    - list[dict[str, Any]]: 模块B输出数组。
    异常说明：字段缺失导致无法匹配时抛出异常。
    边界条件：shot 顺序严格按模块A segments 顺序输出。
    """
    segments = [dict(item) for item in module_a_output.get("segments", []) if isinstance(item, dict)]
    role3_map = {str(item.get("shot_id", "")).strip(): dict(item) for item in role3_output.get("shots", []) if isinstance(item, dict)}
    role4_map = {str(item.get("shot_id", "")).strip(): dict(item) for item in role4_output.get("shots", []) if isinstance(item, dict)}
    shots: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        shot_id = f"shot_{index + 1:03d}"
        directing = role3_map.get(shot_id)
        prompt_block = role4_map.get(shot_id)
        if not directing:
            continue
        if not prompt_block:
            continue
        shot_item = _assemble_module_b_output_item(
            segment=segment,
            directing=directing,
            prompt_block=prompt_block,
            shot_id=shot_id,
        )
        shots.append(shot_item)
    return shots


def _assemble_module_b_output_item(
    *,
    segment: dict[str, Any],
    directing: dict[str, Any],
    prompt_block: dict[str, Any],
    shot_id: str,
) -> dict[str, Any]:
    """
    功能说明：将单个 role3/role4 shot 结果装配为标准模块B输出项。
    参数说明：
    - segment: 当前 segment 元信息。
    - directing: role3 单镜头结果。
    - prompt_block: role4 单镜头结果。
    - shot_id: 标准 shot_id。
    返回值：
    - dict[str, Any]: 单个模块B输出项。
    异常说明：无。
    边界条件：scene_desc 优先取 role4 回填，缺失时退回 role3 中文描述。
    """
    return {
        "shot_id": shot_id,
        "start_time": float(segment.get("start_time", 0.0)),
        "end_time": float(segment.get("end_time", segment.get("start_time", 0.0))),
        "scene_desc": str(prompt_block.get("scene_desc", directing.get("scene_desc_zh", ""))).strip(),
        "keyframe_prompt_start_zh": str(prompt_block.get("keyframe_prompt_start_zh", "")).strip(),
        "keyframe_prompt_start_en": str(prompt_block.get("keyframe_prompt_start_en", "")).strip(),
        "keyframe_negative_prompt_start_zh": str(prompt_block.get("keyframe_negative_prompt_start_zh", "")).strip(),
        "keyframe_negative_prompt_start_en": str(prompt_block.get("keyframe_negative_prompt_start_en", "")).strip(),
        "keyframe_prompt_end_zh": str(prompt_block.get("keyframe_prompt_end_zh", "")).strip(),
        "keyframe_prompt_end_en": str(prompt_block.get("keyframe_prompt_end_en", "")).strip(),
        "keyframe_negative_prompt_end_zh": str(prompt_block.get("keyframe_negative_prompt_end_zh", "")).strip(),
        "keyframe_negative_prompt_end_en": str(prompt_block.get("keyframe_negative_prompt_end_en", "")).strip(),
        "video_prompt_zh": str(prompt_block.get("video_prompt_zh", "")).strip(),
        "video_prompt_en": str(prompt_block.get("video_prompt_en", "")).strip(),
        "keyframe_prompt_start_tokens_zh": [dict(item) for item in prompt_block.get("keyframe_prompt_start_tokens_zh", []) if isinstance(item, dict)],
        "keyframe_prompt_start_tokens_en": [dict(item) for item in prompt_block.get("keyframe_prompt_start_tokens_en", []) if isinstance(item, dict)],
        "keyframe_negative_prompt_start_tokens_zh_increment": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_start_tokens_zh_increment", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_start_tokens_en_increment": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_start_tokens_en_increment", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_start_tokens_zh": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_start_tokens_zh", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_start_tokens_en": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_start_tokens_en", []) if isinstance(item, dict)
        ],
        "keyframe_prompt_end_tokens_zh": [dict(item) for item in prompt_block.get("keyframe_prompt_end_tokens_zh", []) if isinstance(item, dict)],
        "keyframe_prompt_end_tokens_en": [dict(item) for item in prompt_block.get("keyframe_prompt_end_tokens_en", []) if isinstance(item, dict)],
        "keyframe_negative_prompt_end_tokens_zh_increment": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_end_tokens_zh_increment", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_end_tokens_en_increment": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_end_tokens_en_increment", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_end_tokens_zh": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_end_tokens_zh", []) if isinstance(item, dict)
        ],
        "keyframe_negative_prompt_end_tokens_en": [
            dict(item) for item in prompt_block.get("keyframe_negative_prompt_end_tokens_en", []) if isinstance(item, dict)
        ],
        "video_prompt_tokens_zh": [dict(item) for item in prompt_block.get("video_prompt_tokens_zh", []) if isinstance(item, dict)],
        "video_prompt_tokens_en": [dict(item) for item in prompt_block.get("video_prompt_tokens_en", []) if isinstance(item, dict)],
        "camera_plan_preset_id": str(directing.get("camera_plan_preset_id", "")).strip(),
        "transition_plan_preset_id": str(directing.get("transition_plan_preset_id", "")).strip(),
        "motion_delta_label": str(directing.get("motion_delta_label", "")).strip(),
        "motion_speed_label": str(directing.get("motion_speed_label", "")).strip(),
        "composition_stability": str(directing.get("composition_stability", "")).strip(),
        "camera_plan": dict(directing.get("camera_plan", {})),
        "transition_plan": dict(directing.get("transition_plan", {})),
        "constraints": {"must_keep_style": True, "must_align_to_beat": True},
    }
