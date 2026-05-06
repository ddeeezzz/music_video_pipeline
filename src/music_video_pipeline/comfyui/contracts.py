"""
文件用途：加载并渲染 ComfyUI 工作流契约。
核心流程：读取 contract JSON -> 读取 workflow API JSON -> 按绑定表回填变量 -> 生成可提交 prompt。
输入输出：输入契约路径与绑定值，输出渲染后的工作流字典。
依赖说明：依赖标准库 dataclasses/copy/json/pathlib。
维护说明：契约文件只描述“字段绑定关系”，不要把业务逻辑硬编码到 workflow JSON 里。
"""

# 标准库：用于深拷贝工作流模板。
from copy import deepcopy
# 标准库：用于数据类声明。
from dataclasses import dataclass
# 标准库：用于 JSON 读取。
import json
# 标准库：用于路径处理。
from pathlib import Path
# 标准库：用于类型提示。
from typing import Any


@dataclass(frozen=True)
class ComfyUIWorkflowContract:
    """
    功能说明：描述一个 ComfyUI API 工作流契约。
    参数说明：
    - name: 契约名称。
    - workflow_api_file: workflow API JSON 文件绝对路径。
    - bindings: 绑定表，key 为业务字段名，value 为 node_id/input_name 映射。
    - output_node_id: 用于收集产物的输出节点 ID。
    - output_kind: 输出类型（images/image_sequence）。
    返回值：不适用。
    异常说明：不适用。
    边界条件：bindings 中未声明的字段不会被自动注入。
    """

    name: str
    workflow_api_file: Path
    bindings: dict[str, dict[str, Any]]
    output_node_id: str
    output_kind: str


def _resolve_project_root() -> Path:
    """
    功能说明：解析项目根目录路径。
    参数说明：无。
    返回值：
    - Path: 项目根目录绝对路径。
    异常说明：无。
    边界条件：假设本文件位于 src/music_video_pipeline/comfyui 目录。
    """
    return Path(__file__).resolve().parents[3]


def _load_json_object(path: Path, source_name: str) -> dict[str, Any]:
    """
    功能说明：读取并校验 JSON 顶层对象。
    参数说明：
    - path: JSON 文件路径。
    - source_name: 数据来源名。
    返回值：
    - dict[str, Any]: JSON 对象。
    异常说明：
    - RuntimeError: 文件不存在、解析失败或顶层不是对象时抛出。
    边界条件：读取编码统一使用 utf-8-sig。
    """
    if not path.exists():
        raise RuntimeError(f"{source_name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file_obj:
            payload = json.load(file_obj)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"{source_name} 读取失败：path={path}，错误={error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{source_name} 顶层结构必须是对象：path={path}")
    return payload


def load_workflow_contract(contract_path: Path | str) -> ComfyUIWorkflowContract:
    """
    功能说明：加载 ComfyUI 工作流契约。
    参数说明：
    - contract_path: 契约 JSON 路径（支持相对项目根）。
    返回值：
    - ComfyUIWorkflowContract: 解析后的契约对象。
    异常说明：
    - RuntimeError: 契约字段非法或 workflow_api_file 缺失时抛出。
    边界条件：workflow_api_file 相对路径按项目根解析。
    """
    project_root = _resolve_project_root()
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = (project_root / contract_file).resolve()
    raw_contract = _load_json_object(path=contract_file, source_name="ComfyUI 工作流契约")

    name = str(raw_contract.get("name", "")).strip()
    workflow_api_file_text = str(raw_contract.get("workflow_api_file", "")).strip()
    output_node_id = str(raw_contract.get("output_node_id", "")).strip()
    output_kind = str(raw_contract.get("output_kind", "images")).strip() or "images"
    bindings = raw_contract.get("bindings")
    if not name:
        raise RuntimeError(f"ComfyUI 工作流契约缺失 name：{contract_file}")
    if not workflow_api_file_text:
        raise RuntimeError(f"ComfyUI 工作流契约缺失 workflow_api_file：{contract_file}")
    if not output_node_id:
        raise RuntimeError(f"ComfyUI 工作流契约缺失 output_node_id：{contract_file}")
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError(f"ComfyUI 工作流契约 bindings 非法：{contract_file}")

    workflow_api_file = Path(workflow_api_file_text)
    if not workflow_api_file.is_absolute():
        workflow_api_file = (project_root / workflow_api_file).resolve()
    return ComfyUIWorkflowContract(
        name=name,
        workflow_api_file=workflow_api_file,
        bindings=dict(bindings),
        output_node_id=output_node_id,
        output_kind=output_kind,
    )


def render_workflow_from_contract(
    contract: ComfyUIWorkflowContract,
    binding_values: dict[str, Any],
) -> dict[str, Any]:
    """
    功能说明：根据契约与绑定值生成可提交给 ComfyUI 的工作流。
    参数说明：
    - contract: 已加载的工作流契约。
    - binding_values: 业务字段到具体值的映射。
    返回值：
    - dict[str, Any]: 渲染完成的 workflow prompt。
    异常说明：
    - RuntimeError: workflow 文件结构非法、缺绑定字段或节点/输入不存在时抛出。
    边界条件：仅按 bindings 白名单回填字段，不做隐式兼容。
    """
    raw_workflow = _load_json_object(path=contract.workflow_api_file, source_name=f"ComfyUI workflow[{contract.name}]")
    if not raw_workflow:
        raise RuntimeError(
            f"ComfyUI workflow[{contract.name}] 为空：{contract.workflow_api_file}。"
            "请先提供可用的 API workflow JSON。"
        )
    workflow = deepcopy(raw_workflow)

    for binding_name, binding_spec_raw in contract.bindings.items():
        if binding_name not in binding_values:
            raise RuntimeError(
                f"ComfyUI workflow[{contract.name}] 缺少绑定值：binding={binding_name}"
            )
        if not isinstance(binding_spec_raw, dict):
            raise RuntimeError(
                f"ComfyUI workflow[{contract.name}] binding 配置非法：binding={binding_name}"
            )
        node_id = str(binding_spec_raw.get("node_id", "")).strip()
        input_name = str(binding_spec_raw.get("input_name", "")).strip()
        if not node_id or not input_name:
            raise RuntimeError(
                f"ComfyUI workflow[{contract.name}] binding 缺失 node_id/input_name：binding={binding_name}"
            )
        node_payload = workflow.get(node_id)
        if not isinstance(node_payload, dict):
            raise RuntimeError(
                f"ComfyUI workflow[{contract.name}] 找不到节点：node_id={node_id}，binding={binding_name}"
            )
        inputs_payload = node_payload.get("inputs")
        if not isinstance(inputs_payload, dict):
            raise RuntimeError(
                f"ComfyUI workflow[{contract.name}] 节点 inputs 非法：node_id={node_id}，binding={binding_name}"
            )
        inputs_payload[input_name] = binding_values[binding_name]
    return workflow
