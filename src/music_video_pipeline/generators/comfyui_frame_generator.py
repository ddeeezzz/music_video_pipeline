"""
文件用途：实现模块 C 的 ComfyUI 双关键帧生成器。
核心流程：首关键帧走 txt2img workflow，末关键帧走首帧图 + end prompt 的 img2img workflow。
输入输出：输入 shot 与输出目录，输出符合模块 D 双锚点契约的 frame_item。
依赖说明：依赖项目内 ComfyUI 客户端与工作流契约工具。
维护说明：本文件只负责模块 C 的 ComfyUI 出图，不承担 resident daemon 或模块 D 视频逻辑。
"""

# 标准库：用于日志输出。
import logging
# 标准库：用于正则解析。
import re
# 标准库：用于文件复制。
import shutil
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any

# 项目内模块：应用配置定义。
from music_video_pipeline.config import AppConfig
# 项目内模块：ComfyUI API 客户端。
from music_video_pipeline.comfyui import (
    ComfyUIClient,
    ComfyUIServiceOptions,
    load_workflow_contract,
    render_workflow_from_contract,
)
# 项目内模块：Windows 路径回映射。
from music_video_pipeline.task_audio_path import remap_windows_absolute_path


class ComfyUIFrameGenerator:
    """
    功能说明：基于 ComfyUI workflow 生成双关键帧。
    参数说明：
    - app_config: 全局配置对象。
    - logger: 日志对象。
    返回值：不适用。
    异常说明：运行时异常由 generate_one 抛出。
    边界条件：默认假设 ComfyUI 服务已经启动且可访问。
    """

    def __init__(self, app_config: AppConfig, logger: logging.Logger, seed: int = 42) -> None:
        self._config = app_config
        self._logger = logger
        self._seed = int(seed)
        project_root = Path(__file__).resolve().parents[3]
        comfyui_cfg = app_config.comfyui
        self._project_root = project_root
        comfyui_root_text = str(comfyui_cfg.root_dir)
        remapped_root = remap_windows_absolute_path(workspace_root=project_root, path_text=comfyui_root_text)
        if remapped_root is not None:
            resolved_root = remapped_root
        else:
            resolved_root = (project_root / comfyui_root_text).resolve()
        self._client = ComfyUIClient(
            ComfyUIServiceOptions(
                root_dir=resolved_root,
                server_url=str(comfyui_cfg.server_url),
                request_timeout_seconds=float(comfyui_cfg.request_timeout_seconds),
                poll_interval_seconds=float(comfyui_cfg.poll_interval_seconds),
                execution_timeout_seconds=float(comfyui_cfg.execution_timeout_seconds),
            )
        )
        self._contract_start = load_workflow_contract(app_config.module_c.comfyui.contract_start_file)
        self._contract_end = load_workflow_contract(app_config.module_c.comfyui.contract_end_file)

    def prewarm(self) -> None:
        """
        功能说明：校验 ComfyUI 服务可达并记录工作流契约信息。
        参数说明：无。
        返回值：无。
        异常说明：
        - RuntimeError: 服务不可达时抛出。
        边界条件：模型资产由 ComfyUI 直接管理，不做本地文件存在性校验。
        """
        self._client.ensure_service_ready()
        comfyui_cfg = self._config.module_c.comfyui
        required_asset_paths = {
            "checkpoint_file": (self._project_root / str(comfyui_cfg.checkpoint_file)).resolve(),
            "scene_lora_file": (self._project_root / str(comfyui_cfg.turbo_lora_file)).resolve(),
            "char_lora_file": (self._project_root / str(comfyui_cfg.char_lora_file)).resolve(),
        }
        missing_assets = [
            f"{field_name}={asset_path}"
            for field_name, asset_path in required_asset_paths.items()
            if not asset_path.exists()
        ]
        if missing_assets:
            self._logger.warning(
                "模块C ComfyUI 部分本地模型资产缺失（ComfyUI 侧可能已有对应文件），missing=%s",
                missing_assets,
            )
        self._logger.info(
            "模块C ComfyUI 预热完成，start=%s，end=%s",
            self._contract_start.workflow_api_file,
            self._contract_end.workflow_api_file,
        )

    @staticmethod
    def _assemble_prompt(cfg: Any, prompt_body: str, subject_kind: str = "character") -> str:
        """组装正向提示词：前缀 + LLM 输出主体 + （可选后缀）。
        场景类主体跳过后缀中的白色背景约束，避免与场景描述冲突。"""
        prefix = str(getattr(cfg, "prompt_prefix", "").strip())
        suffix = str(getattr(cfg, "prompt_suffix", "").strip())
        parts = [p for p in [prefix, prompt_body] if p]
        normalized_kind = str(subject_kind or "character").strip().lower()
        if normalized_kind != "scene" and suffix:
            parts.append(suffix)
        return "\n".join(parts)

    def generate_one(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
    ) -> dict[str, Any]:
        """
        功能说明：执行单个 shot 的双关键帧生成。
        参数说明：
        - shot: 模块 B 单元产物字典。
        - output_dir: 关键帧输出目录。
        - width/height: 输出分辨率。
        - shot_index: shot 顺序索引（0 基）。
        返回值：
        - dict[str, Any]: 符合模块 D 双锚点契约的 frame_item。
        异常说明：
        - RuntimeError: ComfyUI 服务不可用、workflow 执行失败或必要字段缺失时抛出。
        边界条件：不做单帧兼容回退。
        """
        shot_id = str(shot.get("shot_id", "")).strip()
        if not shot_id:
            raise RuntimeError("模块C ComfyUI 生成失败：shot_id 不能为空。")
        raw_prompt_start = str(shot.get("keyframe_prompt_start_en", "")).strip()
        raw_prompt_end = str(shot.get("keyframe_prompt_end_en", "")).strip()
        shot_subject_kind = str(shot.get("subject_kind", "character")).strip().lower()
        prompt_start = self._assemble_prompt(self._config.module_c.comfyui, raw_prompt_start, shot_subject_kind)
        prompt_end = self._assemble_prompt(self._config.module_c.comfyui, raw_prompt_end, shot_subject_kind)
        negative_prompt_start_zh = str(shot.get("keyframe_negative_prompt_start_zh", "")).strip()
        negative_prompt_start = str(shot.get("keyframe_negative_prompt_start_en", "")).strip()
        negative_prompt_end_zh = str(shot.get("keyframe_negative_prompt_end_zh", "")).strip()
        negative_prompt_end = str(shot.get("keyframe_negative_prompt_end_en", "")).strip()
        keyframe_prompt_start_zh = str(shot.get("keyframe_prompt_start_zh", "")).strip()
        keyframe_prompt_end_zh = str(shot.get("keyframe_prompt_end_zh", "")).strip()
        video_prompt_zh = str(shot.get("video_prompt_zh", "")).strip()
        video_prompt_en = str(shot.get("video_prompt_en", "")).strip()
        missing_fields = [
            key
            for key, value in {
                "keyframe_prompt_start_zh": keyframe_prompt_start_zh,
                "keyframe_prompt_start_en": prompt_start,
                "keyframe_prompt_end_zh": keyframe_prompt_end_zh,
                "keyframe_prompt_end_en": prompt_end,
                "video_prompt_zh": video_prompt_zh,
                "video_prompt_en": video_prompt_en,
            }.items()
            if not value
        ]
        if missing_fields:
            raise RuntimeError(
                "模块C ComfyUI 生成失败：分镜缺失必要提示词字段，"
                f"shot_id={shot_id}，missing={missing_fields}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        comfyui_cfg = self._config.module_c.comfyui
        asset_kind = _resolve_asset_kind(shot=shot)
        contract_start, contract_end = self._resolve_contract_pair(asset_kind=asset_kind)
        checkpoint_name = Path(str(comfyui_cfg.checkpoint_file)).name

        # 解析 seg 序号与资产序号，构建文件名前缀。
        big_seg = str(shot.get("big_segment_id", "")).strip()
        seg_match = re.search(r"(\d+)", big_seg)
        seg_idx = int(seg_match.group(1)) if seg_match else (shot_index + 1)
        asset_idx = 1  # 当前每个 shot 只有一个主体
        file_prefix = f"mvpl/module_c/shot{seg_idx:04d}-{asset_idx}"
        scene_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.turbo_lora_file),
            category_folder="lora",
        )
        char_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.char_lora_file),
            category_folder="lora",
        )

        workflow_start = render_workflow_from_contract(
            contract=contract_start,
            binding_values=self._build_binding_values(
                asset_kind=asset_kind,
                checkpoint_name=checkpoint_name,
                scene_lora_name=scene_lora_name,
                char_lora_name=char_lora_name,
                positive_prompt=prompt_start,
                negative_prompt=negative_prompt_start or str(comfyui_cfg.negative_prompt),
                width=width,
                height=height,
                seed=self._seed,
                filename_prefix=f"{file_prefix}/start",
                subject_kind=shot_subject_kind,
            ),
        )
        start_outputs = self._client.execute_prompt(
            workflow_prompt=workflow_start,
            output_node_id=contract_start.output_node_id,
        )
        if not start_outputs:
            raise RuntimeError(f"模块C ComfyUI 生成失败：首关键帧未返回产物，shot_id={shot_id}")

        # 保存首帧到磁盘，并 stage 到 ComfyUI input 目录作为尾帧图生图的 init_image
        image_path_start = output_dir / f"{shot_id}_start.png"
        image_path_start.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(start_outputs[0], image_path_start)
        staged_init_path = self._client.stage_input_image(
            image_path_start, prefix=f"{shot_id}_end_init"
        )

        workflow_end = render_workflow_from_contract(
            contract=contract_end,
            binding_values=self._build_binding_values(
                asset_kind=asset_kind,
                checkpoint_name=checkpoint_name,
                scene_lora_name=scene_lora_name,
                char_lora_name=char_lora_name,
                positive_prompt=prompt_end,
                negative_prompt=negative_prompt_end or str(comfyui_cfg.negative_prompt),
                width=width,
                height=height,
                seed=self._seed,
                filename_prefix=f"{file_prefix}/end",
                init_image=staged_init_path,
                denoise=comfyui_cfg.end_denoise,
                subject_kind=shot_subject_kind,
            ),
        )
        end_outputs = self._client.execute_prompt(
            workflow_prompt=workflow_end,
            output_node_id=contract_end.output_node_id,
        )
        if not end_outputs:
            raise RuntimeError(f"模块C ComfyUI 生成失败：末关键帧未返回产物，shot_id={shot_id}")

        image_path_end = output_dir / f"{shot_id}_end.png"
        shutil.copy2(end_outputs[0], image_path_end)

        start_time = float(shot["start_time"])
        end_time = float(shot["end_time"])
        duration = round(max(0.5, end_time - start_time), 3)
        self._logger.info(
            "模块C ComfyUI 单元生成完成，shot_id=%s，start=%s，end=%s",
            shot_id,
            image_path_start,
            image_path_end,
        )
        return {
            "shot_id": shot_id,
            "frame_path": str(image_path_start),
            "frame_path_start": str(image_path_start),
            "frame_path_end": str(image_path_end),
            "control_frame_paths": [str(image_path_start), str(image_path_end)],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "scene_desc": str(shot.get("scene_desc", "")),
            "camera_plan": dict(shot.get("camera_plan", {})) if isinstance(shot.get("camera_plan"), dict) else {},
            "transition_plan": (
                dict(shot.get("transition_plan", {})) if isinstance(shot.get("transition_plan"), dict) else {}
            ),
            "keyframe_prompt_start_zh": keyframe_prompt_start_zh,
            "keyframe_prompt_start_en": prompt_start,
            "keyframe_negative_prompt_start_zh": negative_prompt_start_zh,
            "keyframe_negative_prompt_start_en": negative_prompt_start,
            "keyframe_prompt_end_zh": keyframe_prompt_end_zh,
            "keyframe_prompt_end_en": prompt_end,
            "keyframe_negative_prompt_end_zh": negative_prompt_end_zh,
            "keyframe_negative_prompt_end_en": negative_prompt_end,
            "video_prompt_zh": video_prompt_zh,
            "video_prompt_en": video_prompt_en,
            "asset_kind": asset_kind,
            "binding_name": "comfyui",
            "base_model_key": Path(str(comfyui_cfg.checkpoint_file)).stem,
            "scene_lora_file": str((self._project_root / str(comfyui_cfg.turbo_lora_file)).resolve()),
            "char_lora_file": str((self._project_root / str(comfyui_cfg.char_lora_file)).resolve()),
        }

    def generate_one_frame(
        self,
        shot: dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        shot_index: int,
        frame_type: str,
    ) -> dict[str, Any]:
        """
        功能说明：仅生成单个 shot 的首帧或尾帧。
        参数说明：
        - shot: 模块 B 单元产物字典。
        - output_dir: 关键帧输出目录。
        - width/height: 输出分辨率。
        - shot_index: shot 顺序索引（0 基）。
        - frame_type: "start" 或 "end"。
        返回值：
        - dict[str, Any]: 单帧结果字典。
        异常说明：
        - RuntimeError: 服务不可用、workflow 执行失败或必要字段缺失时抛出。
        边界条件：end 当前仍直接走 end prompt workflow，不依赖 start 图。
        """
        normalized_frame_type = str(frame_type or "").strip().lower()
        if normalized_frame_type not in {"start", "end"}:
            raise RuntimeError(f"模块C ComfyUI 单帧生成失败：非法 frame_type={frame_type}")

        shot_id = str(shot.get("shot_id", "")).strip()
        if not shot_id:
            raise RuntimeError("模块C ComfyUI 单帧生成失败：shot_id 不能为空。")
        output_dir.mkdir(parents=True, exist_ok=True)
        comfyui_cfg = self._config.module_c.comfyui
        asset_kind = _resolve_asset_kind(shot=shot)
        contract_start, contract_end = self._resolve_contract_pair(asset_kind=asset_kind)
        checkpoint_name = Path(str(comfyui_cfg.checkpoint_file)).name

        big_seg = str(shot.get("big_segment_id", "")).strip()
        seg_match = re.search(r"(\d+)", big_seg)
        seg_idx = int(seg_match.group(1)) if seg_match else (shot_index + 1)
        asset_idx = 1
        file_prefix = f"mvpl/module_c/shot{seg_idx:04d}-{asset_idx}"
        scene_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.turbo_lora_file),
            category_folder="lora",
        )
        char_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.char_lora_file),
            category_folder="lora",
        )

        raw_prompt_start = str(shot.get("keyframe_prompt_start_en", "")).strip()
        raw_prompt_end = str(shot.get("keyframe_prompt_end_en", "")).strip()
        shot_subject_kind = str(shot.get("subject_kind", "character")).strip().lower()
        prompt_start = self._assemble_prompt(self._config.module_c.comfyui, raw_prompt_start, shot_subject_kind)
        prompt_end = self._assemble_prompt(self._config.module_c.comfyui, raw_prompt_end, shot_subject_kind)
        keyframe_prompt_start_zh = str(shot.get("keyframe_prompt_start_zh", "")).strip()
        keyframe_prompt_end_zh = str(shot.get("keyframe_prompt_end_zh", "")).strip()
        video_prompt_zh = str(shot.get("video_prompt_zh", "")).strip()
        video_prompt_en = str(shot.get("video_prompt_en", "")).strip()
        negative_prompt_start_zh = str(shot.get("keyframe_negative_prompt_start_zh", "")).strip()
        negative_prompt_start = str(shot.get("keyframe_negative_prompt_start_en", "")).strip()
        negative_prompt_end_zh = str(shot.get("keyframe_negative_prompt_end_zh", "")).strip()
        negative_prompt_end = str(shot.get("keyframe_negative_prompt_end_en", "")).strip()

        if normalized_frame_type == "start":
            self._logger.info(
                "模块C ComfyUI 单帧生成开始，shot_id=%s，frame_type=start，width=%s，height=%s",
                shot_id,
                width,
                height,
            )
            if not keyframe_prompt_start_zh or not prompt_start:
                raise RuntimeError(
                    f"模块C ComfyUI 首帧生成失败：缺失首帧提示词，shot_id={shot_id}"
                )
            workflow_prompt = render_workflow_from_contract(
                contract=contract_start,
                binding_values=self._build_binding_values(
                    asset_kind=asset_kind,
                    checkpoint_name=checkpoint_name,
                    scene_lora_name=scene_lora_name,
                    char_lora_name=char_lora_name,
                    positive_prompt=prompt_start,
                    negative_prompt=negative_prompt_start or str(comfyui_cfg.negative_prompt),
                    width=width,
                    height=height,
                    seed=self._seed,
                    filename_prefix=f"{file_prefix}/start",
                    subject_kind=shot_subject_kind,
                ),
            )
            outputs = self._client.execute_prompt(
                workflow_prompt=workflow_prompt,
                output_node_id=contract_start.output_node_id,
            )
            if not outputs:
                raise RuntimeError(f"模块C ComfyUI 首帧生成失败：未返回产物，shot_id={shot_id}")
            image_path = output_dir / f"{shot_id}_start.png"
            shutil.copy2(outputs[0], image_path)
            self._logger.info(
                "模块C ComfyUI 单帧生成完成，shot_id=%s，frame_type=start，image=%s",
                shot_id,
                image_path,
            )
            return {
                "shot_id": shot_id,
                "frame_type": "start",
                "frame_path_start": str(image_path),
                "keyframe_prompt_start_zh": keyframe_prompt_start_zh,
                "keyframe_prompt_start_en": prompt_start,
                "video_prompt_zh": video_prompt_zh,
                "video_prompt_en": video_prompt_en,
                "scene_desc": str(shot.get("scene_desc", "")),
            }

        self._logger.info(
            "模块C ComfyUI 单帧生成开始，shot_id=%s，frame_type=end，width=%s，height=%s",
            shot_id,
            width,
            height,
        )
        if not keyframe_prompt_end_zh or not prompt_end:
            raise RuntimeError(
                f"模块C ComfyUI 尾帧生成失败：缺失尾帧提示词，shot_id={shot_id}"
            )
        # 单帧重跑时，从磁盘读取已存在的首帧图作为 img2img init
        start_frame_path = output_dir / f"{shot_id}_start.png"
        if start_frame_path.exists():
            staged_init_path_end = self._client.stage_input_image(
                start_frame_path, prefix=f"{shot_id}_end_init"
            )
        else:
            self._logger.warning(
                "模块C ComfyUI 尾帧单帧生成未找到首帧图，回退 txt2img：shot_id=%s，path=%s",
                shot_id, start_frame_path,
            )
            staged_init_path_end = None
        workflow_prompt = render_workflow_from_contract(
            contract=contract_end,
            binding_values=self._build_binding_values(
                asset_kind=asset_kind,
                checkpoint_name=checkpoint_name,
                scene_lora_name=scene_lora_name,
                char_lora_name=char_lora_name,
                positive_prompt=prompt_end,
                negative_prompt=negative_prompt_end or str(comfyui_cfg.negative_prompt),
                width=width,
                height=height,
                seed=self._seed,
                filename_prefix=f"{file_prefix}/end",
                init_image=staged_init_path_end,
                denoise=comfyui_cfg.end_denoise if staged_init_path_end else None,
                subject_kind=shot_subject_kind,
            ),
        )
        outputs = self._client.execute_prompt(
            workflow_prompt=workflow_prompt,
            output_node_id=contract_end.output_node_id,
        )
        if not outputs:
            raise RuntimeError(f"模块C ComfyUI 尾帧生成失败：未返回产物，shot_id={shot_id}")
        image_path = output_dir / f"{shot_id}_end.png"
        shutil.copy2(outputs[0], image_path)
        self._logger.info(
            "模块C ComfyUI 单帧生成完成，shot_id=%s，frame_type=end，image=%s",
            shot_id,
            image_path,
        )
        return {
            "shot_id": shot_id,
            "frame_type": "end",
            "frame_path_end": str(image_path),
            "keyframe_prompt_end_zh": keyframe_prompt_end_zh,
            "keyframe_prompt_end_en": prompt_end,
            "video_prompt_zh": video_prompt_zh,
            "video_prompt_en": video_prompt_en,
            "scene_desc": str(shot.get("scene_desc", "")),
        }

    def _resolve_contract_pair(self, asset_kind: str):
        """
        功能说明：返回 start/end 工作流契约。
        返回值：
        - tuple[object, object]: 依次为 start/end 契约对象。
        """
        return self._contract_start, self._contract_end

    def _build_binding_values(
        self,
        *,
        asset_kind: str,
        checkpoint_name: str,
        scene_lora_name: str,
        char_lora_name: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        filename_prefix: str,
        init_image: str | None = None,
        denoise: float | None = None,
        subject_kind: str = "character",
    ) -> dict[str, Any]:
        """
        功能说明：构建一次 ComfyUI workflow 渲染所需的绑定值字典。
        参数说明：
        - asset_kind: 素材类型。
        - subject_kind: 主体类型（character/scene），场景类主体跳过角色 LoRA。
        - checkpoint_name: checkpoint 文件名。
        - scene_lora_name: 场景 LoRA 相对名。
        - char_lora_name: 角色 LoRA 相对名。
        - positive_prompt: 正向提示词。
        - negative_prompt: 负向提示词。
        - width/height: 输出宽高。
        - seed: 稳定种子。
        - filename_prefix: ComfyUI 输出前缀。
        - init_image: 可选 img2img 输入图。
        - denoise: 可选 img2img denoise。
        返回值：
        - dict[str, Any]: 契约绑定值字典。
        异常说明：无。
        边界条件：prop 路径不注入 char LoRA 绑定。
        """
        comfyui_cfg = self._config.module_c.comfyui
        binding_values: dict[str, Any] = {
            "checkpoint_name": checkpoint_name,
            "scene_lora_name": scene_lora_name,
            "scene_lora_strength_model": float(comfyui_cfg.scene_lora_strength),
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": int(seed),
            "steps": int(comfyui_cfg.steps),
            "cfg": float(comfyui_cfg.guidance_scale),
            "sampler_name": str(comfyui_cfg.sampler_name),
            "scheduler": str(comfyui_cfg.scheduler),
            "filename_prefix": filename_prefix,
        }
        if init_image is None:
            binding_values["width"] = int(width)
            binding_values["height"] = int(height)
        else:
            binding_values["init_image"] = init_image
            binding_values["denoise"] = float(denoise if denoise is not None else comfyui_cfg.end_denoise)
        normalized_sk = str(subject_kind or "character").strip().lower()
        binding_values["char_lora_name"] = char_lora_name
        if asset_kind == "character" and normalized_sk != "scene":
            binding_values["char_lora_strength_model"] = float(comfyui_cfg.char_lora_strength)
        else:
            binding_values["char_lora_strength_model"] = 0.0
        return binding_values


def _resolve_catalog_asset_name(asset_file: str, category_folder: str) -> str:
    """
    功能说明：将项目内模型文件路径转换为 ComfyUI catalog 相对路径。
    参数说明：
    - asset_file: 模型文件路径。
    - category_folder: 目录锚点名称，如 lora。
    返回值：
    - str: 相对于 ComfyUI 搜索根目录的相对路径（使用正斜杠分隔符以兼容 Linux ComfyUI，亦适用于 Windows）。
    异常说明：无。
    边界条件：当无法识别目录锚点时退回文件名。
    """
    parts = Path(str(asset_file)).parts
    for index, part in enumerate(parts):
        if part != category_folder:
            continue
        relative_parts = parts[index + 1 :]
        if relative_parts:
            return "/".join(relative_parts)
    return Path(str(asset_file)).name


def _resolve_asset_kind(shot: dict[str, Any]) -> str:
    """
    功能说明：解析当前 shot 的素材类型。
    参数说明：
    - shot: 模块 B 单元产物字典。
    返回值：
    - str: 归一化后的素材类型，当前为 character 或 prop。
    异常说明：无。
    边界条件：未知值统一回退为 character。
    """
    normalized = str(shot.get("asset_kind", "character")).strip().lower()
    if normalized == "prop":
        return "prop"
    return "character"
