"""
文件用途：提供 ComfyUI API 客户端与输入输出文件编排能力。
核心流程：准备输入文件 -> POST /prompt -> 轮询 /history/{prompt_id} -> 收集输出文件。
输入输出：输入服务配置、工作流 prompt 与文件路径，输出 ComfyUI 产物路径列表。
依赖说明：依赖标准库 shutil/time/uuid/pathlib，以及第三方 requests。
维护说明：模块 C/D 不应各自拼 HTTP 请求；统一通过本文件访问 ComfyUI。
"""

# 标准库：用于数据类声明。
from dataclasses import dataclass
# 标准库：用于文件复制。
import shutil
# 标准库：用于时间轮询。
import time
# 标准库：用于唯一路径前缀。
import uuid
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any

# 第三方库：用于 HTTP 请求。
import requests


@dataclass(frozen=True)
class ComfyUIServiceOptions:
    """
    功能说明：定义 ComfyUI 服务访问参数。
    参数说明：
    - root_dir: ComfyUI 根目录。
    - server_url: ComfyUI API 地址。
    - request_timeout_seconds: 单次 HTTP 请求超时。
    - poll_interval_seconds: 历史轮询间隔。
    - execution_timeout_seconds: 单个 prompt 总超时。
    返回值：不适用。
    异常说明：不适用。
    边界条件：input/output 目录固定挂在 root_dir 下。
    """

    root_dir: Path
    server_url: str = "http://127.0.0.1:8188"
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 1.0
    execution_timeout_seconds: float = 600.0

    @property
    def input_dir(self) -> Path:
        return self.root_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output"


class ComfyUIClient:
    """
    功能说明：封装 ComfyUI prompt 提交与输出收集。
    参数说明：
    - options: 服务访问参数。
    返回值：不适用。
    异常说明：不适用。
    边界条件：假设 ComfyUI 已由外部进程启动；本类不负责拉起服务。
    """

    def __init__(self, options: ComfyUIServiceOptions) -> None:
        self._options = options
        self._session = requests.Session()

    def ensure_service_ready(self) -> None:
        """
        功能说明：探测 ComfyUI 服务是否可访问。
        参数说明：无。
        返回值：无。
        异常说明：
        - RuntimeError: 服务不可访问时抛出。
        边界条件：只做轻量 GET 探测，不触发执行。
        """
        target_url = f"{self._options.server_url.rstrip('/')}/system_stats"
        try:
            response = self._session.get(target_url, timeout=self._options.request_timeout_seconds)
            response.raise_for_status()
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "ComfyUI 服务不可用，请先启动 ComfyUI API 服务，"
                f"url={self._options.server_url}，错误={error}"
            ) from error

    def stage_input_image(self, source_path: Path | str, prefix: str) -> str:
        """
        功能说明：将输入图片复制到 ComfyUI input 目录，并返回相对文件名。
        参数说明：
        - source_path: 原始图片路径。
        - prefix: 目标文件名前缀。
        返回值：
        - str: 相对 input 目录的 POSIX 路径。
        异常说明：
        - RuntimeError: 原图不存在或复制失败时抛出。
        边界条件：使用 `mvpl/` 子目录隔离项目输入。
        """
        source_file = Path(source_path).resolve()
        if not source_file.exists():
            raise RuntimeError(f"ComfyUI 输入图片不存在：{source_file}")
        safe_prefix = str(prefix).strip().replace(" ", "_") or "asset"
        target_dir = self._options.input_dir / "mvpl"
        target_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{safe_prefix}_{uuid.uuid4().hex[:8]}{source_file.suffix.lower() or '.png'}"
        target_file = target_dir / unique_name
        try:
            shutil.copy2(source_file, target_file)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"ComfyUI 输入图片复制失败：source={source_file}，target={target_file}，错误={error}"
            ) from error
        return target_file.relative_to(self._options.input_dir).as_posix()

    def execute_prompt(
        self,
        workflow_prompt: dict[str, Any],
        output_node_id: str,
    ) -> list[Path]:
        """
        功能说明：提交 workflow prompt 并等待指定输出节点产物就绪。
        参数说明：
        - workflow_prompt: API workflow prompt。
        - output_node_id: 产物输出节点 ID。
        返回值：
        - list[Path]: 输出文件绝对路径数组。
        异常说明：
        - RuntimeError: prompt 提交失败、超时或未产生输出时抛出。
        边界条件：当前只收集 history 中的 images 文件列表。
        """
        self.ensure_service_ready()
        prompt_url = f"{self._options.server_url.rstrip('/')}/prompt"
        try:
            response = self._session.post(
                prompt_url,
                json={"prompt": workflow_prompt},
                timeout=self._options.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"ComfyUI prompt 提交失败：错误={error}") from error
        prompt_id = str(payload.get("prompt_id", "")).strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI prompt 响应缺失 prompt_id：payload={payload}")
        return self._wait_for_output_files(prompt_id=prompt_id, output_node_id=output_node_id)

    def _wait_for_output_files(self, prompt_id: str, output_node_id: str) -> list[Path]:
        """
        功能说明：轮询 ComfyUI history 直到指定输出节点生成图片文件。
        参数说明：
        - prompt_id: ComfyUI prompt_id。
        - output_node_id: 输出节点 ID。
        返回值：
        - list[Path]: 输出文件绝对路径数组。
        异常说明：
        - RuntimeError: 超时、history 结构非法或无产物时抛出。
        边界条件：按 history 返回顺序收集图片，不额外打乱顺序。
        """
        history_url = f"{self._options.server_url.rstrip('/')}/history/{prompt_id}"
        deadline = time.time() + max(self._options.execution_timeout_seconds, 1.0)
        last_payload: Any = None
        while time.time() < deadline:
            try:
                response = self._session.get(history_url, timeout=self._options.request_timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                last_payload = payload
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"ComfyUI history 查询失败：prompt_id={prompt_id}，错误={error}") from error

            prompt_payload = payload.get(prompt_id)
            if isinstance(prompt_payload, dict):
                outputs_payload = prompt_payload.get("outputs")
                if isinstance(outputs_payload, dict):
                    node_output = outputs_payload.get(str(output_node_id))
                    image_files = self._extract_image_files(node_output=node_output)
                    if image_files:
                        return image_files
            time.sleep(max(self._options.poll_interval_seconds, 0.2))
        raise RuntimeError(
            "ComfyUI 执行超时或未产生输出，"
            f"prompt_id={prompt_id}，output_node_id={output_node_id}，last_payload={last_payload}"
        )

    def _extract_image_files(self, node_output: Any) -> list[Path]:
        """
        功能说明：从 history 某个输出节点记录中提取图片文件路径。
        参数说明：
        - node_output: history.outputs[node_id] 的值。
        返回值：
        - list[Path]: 图片绝对路径数组。
        异常说明：无。
        边界条件：只识别 `images` 字段。
        """
        if not isinstance(node_output, dict):
            return []
        images_payload = node_output.get("images")
        if not isinstance(images_payload, list):
            return []
        resolved_paths: list[Path] = []
        for item in images_payload:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "")).strip()
            if not filename:
                continue
            subfolder = str(item.get("subfolder", "")).strip()
            path_obj = (self._options.output_dir / subfolder / filename).resolve()
            if path_obj.exists():
                resolved_paths.append(path_obj)
        return resolved_paths
