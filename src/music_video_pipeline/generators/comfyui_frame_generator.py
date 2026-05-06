"""
文件用途：实现模块 C 的 ComfyUI 双关键帧生成器。
核心流程：首关键帧走 txt2img workflow，末关键帧走首帧图 + end prompt 的 img2img workflow。
输入输出：输入 shot 与输出目录，输出符合模块 D 双锚点契约的 frame_item。
依赖说明：依赖项目内 ComfyUI 客户端与工作流契约工具。
维护说明：本文件只负责模块 C 的 ComfyUI 出图，不承担 resident daemon 或模块 D 视频逻辑。
"""

# 标准库：用于稳定随机种子。
import hashlib
# 标准库：用于日志输出。
import logging
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

    def __init__(self, app_config: AppConfig, logger: logging.Logger) -> None:
        self._config = app_config
        self._logger = logger
        project_root = Path(__file__).resolve().parents[3]
        comfyui_cfg = app_config.comfyui
        self._project_root = project_root
        self._client = ComfyUIClient(
            ComfyUIServiceOptions(
                root_dir=(project_root / str(comfyui_cfg.root_dir)).resolve(),
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
        功能说明：在模块 C 批量生成前显式校验 ComfyUI 服务、契约与关键模型资产。
        参数说明：无。
        返回值：无。
        异常说明：
        - RuntimeError: 服务不可达、契约异常或关键模型资产缺失时抛出。
        边界条件：本预热不提交真实生成任务，只做前置条件校验。
        """
        self._client.ensure_service_ready()
        comfyui_cfg = self._config.module_c.comfyui
        required_asset_paths = {
            "checkpoint_file": (self._project_root / str(comfyui_cfg.checkpoint_file)).resolve(),
            "scene_lora_file": (self._project_root / str(comfyui_cfg.scene_lora_file)).resolve(),
            "char_lora_file": (self._project_root / str(comfyui_cfg.char_lora_file)).resolve(),
        }
        missing_assets = [
            f"{field_name}={asset_path}"
            for field_name, asset_path in required_asset_paths.items()
            if not asset_path.exists()
        ]
        if missing_assets:
            raise RuntimeError(
                "模块C ComfyUI 预热失败：关键模型资产缺失，"
                f"missing={missing_assets}"
            )
        self._logger.info(
            "模块C ComfyUI 预热完成，contract_start=%s，contract_end=%s",
            self._contract_start.workflow_api_file,
            self._contract_end.workflow_api_file,
        )

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
        prompt_start = str(shot.get("keyframe_prompt_start_en", "")).strip()
        prompt_end = str(shot.get("keyframe_prompt_end_en", "")).strip()
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
        checkpoint_name = Path(str(comfyui_cfg.checkpoint_file)).name
        scene_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.scene_lora_file),
            category_folder="lora",
        )
        char_lora_name = _resolve_catalog_asset_name(
            asset_file=str(comfyui_cfg.char_lora_file),
            category_folder="lora",
        )

        workflow_start = render_workflow_from_contract(
            contract=self._contract_start,
            binding_values={
                "checkpoint_name": checkpoint_name,
                "scene_lora_name": scene_lora_name,
                "scene_lora_strength_model": float(comfyui_cfg.scene_lora_strength),
                "scene_lora_strength_clip": float(comfyui_cfg.scene_lora_strength),
                "char_lora_name": char_lora_name,
                "char_lora_strength_model": float(comfyui_cfg.char_lora_strength),
                "char_lora_strength_clip": float(comfyui_cfg.char_lora_strength),
                "positive_prompt": prompt_start,
                "negative_prompt": negative_prompt_start or str(comfyui_cfg.negative_prompt),
                "width": int(width),
                "height": int(height),
                "seed": _resolve_seed_value(shot_id=shot_id, shot_index=shot_index, seed_variant="start"),
                "steps": int(comfyui_cfg.steps),
                "cfg": float(comfyui_cfg.guidance_scale),
                "sampler_name": str(comfyui_cfg.sampler_name),
                "scheduler": str(comfyui_cfg.scheduler),
                "filename_prefix": f"mvpl/module_c/{shot_id}/start",
            },
        )
        start_outputs = self._client.execute_prompt(
            workflow_prompt=workflow_start,
            output_node_id=self._contract_start.output_node_id,
        )
        if not start_outputs:
            raise RuntimeError(f"模块C ComfyUI 生成失败：首关键帧未返回产物，shot_id={shot_id}")
        staged_init_image = self._client.stage_input_image(start_outputs[0], prefix=f"{shot_id}_start")
        workflow_end = render_workflow_from_contract(
            contract=self._contract_end,
            binding_values={
                "checkpoint_name": checkpoint_name,
                "scene_lora_name": scene_lora_name,
                "scene_lora_strength_model": float(comfyui_cfg.scene_lora_strength),
                "scene_lora_strength_clip": float(comfyui_cfg.scene_lora_strength),
                "char_lora_name": char_lora_name,
                "char_lora_strength_model": float(comfyui_cfg.char_lora_strength),
                "char_lora_strength_clip": float(comfyui_cfg.char_lora_strength),
                "init_image": staged_init_image,
                "positive_prompt": prompt_end,
                "negative_prompt": negative_prompt_end or str(comfyui_cfg.negative_prompt),
                "seed": _resolve_seed_value(shot_id=shot_id, shot_index=shot_index, seed_variant="end"),
                "steps": int(comfyui_cfg.steps),
                "cfg": float(comfyui_cfg.guidance_scale),
                "sampler_name": str(comfyui_cfg.sampler_name),
                "scheduler": str(comfyui_cfg.scheduler),
                "denoise": float(comfyui_cfg.end_denoise),
                "filename_prefix": f"mvpl/module_c/{shot_id}/end",
            },
        )
        end_outputs = self._client.execute_prompt(
            workflow_prompt=workflow_end,
            output_node_id=self._contract_end.output_node_id,
        )
        if not end_outputs:
            raise RuntimeError(f"模块C ComfyUI 生成失败：末关键帧未返回产物，shot_id={shot_id}")

        image_path_start = output_dir / f"frame_{shot_index + 1:03d}.png"
        image_path_end = output_dir / f"frame_{shot_index + 1:03d}_end.png"
        shutil.copy2(start_outputs[0], image_path_start)
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
            "binding_name": "comfyui",
            "base_model_key": Path(str(comfyui_cfg.checkpoint_file)).stem,
            "scene_lora_file": str((self._project_root / str(comfyui_cfg.scene_lora_file)).resolve()),
            "char_lora_file": str((self._project_root / str(comfyui_cfg.char_lora_file)).resolve()),
        }


def _resolve_seed_value(shot_id: str, shot_index: int, seed_variant: str) -> int:
    """
    功能说明：为 ComfyUI workflow 生成稳定随机种子。
    参数说明：
    - shot_id: 镜头 ID。
    - shot_index: 镜头索引。
    - seed_variant: 种子变体标签（start/end）。
    返回值：
    - int: 32 位正整数种子。
    异常说明：无。
    边界条件：同一 shot 不同变体返回不同种子。
    """
    seed_source = f"{shot_id}|{shot_index}|{seed_variant}".encode("utf-8")
    return int(hashlib.sha256(seed_source).hexdigest()[:8], 16)


def _resolve_catalog_asset_name(asset_file: str, category_folder: str) -> str:
    """
    功能说明：将项目内模型文件路径转换为 ComfyUI catalog 相对路径。
    参数说明：
    - asset_file: 模型文件路径。
    - category_folder: 目录锚点名称，如 lora。
    返回值：
    - str: 相对于 ComfyUI 搜索根目录的 POSIX 路径。
    异常说明：无。
    边界条件：当无法识别目录锚点时退回文件名。
    """
    parts = Path(str(asset_file)).parts
    for index, part in enumerate(parts):
        if part != category_folder:
            continue
        relative_parts = parts[index + 2 :]
        if relative_parts:
            return Path(*relative_parts).as_posix()
    return Path(str(asset_file)).name
