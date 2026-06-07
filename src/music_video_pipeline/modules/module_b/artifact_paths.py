"""
文件用途：统一管理模块 B 各 role 的工作产物目录。
核心流程：基于 artifacts_dir 生成 module_b_work/roleX 以及 prompt、streaming 子目录路径。
输入输出：输入任务 artifacts_dir 与 role 名称，输出标准化 Path。
依赖说明：依赖 pathlib 进行路径拼接与解析。
维护说明：新增模块 B 工作产物时优先复用本文件，避免路径规则散落。
"""

# 标准库：用于路径处理。
from pathlib import Path


# 常量：模块 B 当前唯一正式工作目录名。
MODULE_B_WORK_DIR_NAME = "module_b_work"

# 常量：模块 B 支持的四个固定角色目录名。
MODULE_B_ROLE_NAMES = ("role1", "role2", "role3", "role4")

# 常量：每个 role 下用于保存 prompt 类产物的子目录名。
MODULE_B_PROMPT_DIR_NAME = "prompt"

# 常量：每个 role 下用于保存流式预览类产物的子目录名。
MODULE_B_STREAMING_DIR_NAME = "streaming"


def get_module_b_work_dir(artifacts_dir: Path) -> Path:
    """
    功能说明：返回模块 B 的统一工作目录。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    返回值：
    - Path: artifacts/module_b_work 的绝对路径。
    异常说明：无。
    边界条件：仅解析路径，不主动创建目录。
    """
    return (artifacts_dir / MODULE_B_WORK_DIR_NAME).resolve()


def get_module_b_role_dir(artifacts_dir: Path, role_name: str) -> Path:
    """
    功能说明：返回模块 B 指定 role 的工作目录。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    - role_name: role1/role2/role3/role4。
    返回值：
    - Path: artifacts/module_b_work/<role_name> 的绝对路径。
    异常说明：
    - ValueError: role_name 不在固定角色集合内。
    边界条件：仅解析路径，不主动创建目录。
    """
    safe_role_name = str(role_name or "").strip().lower()
    if safe_role_name not in MODULE_B_ROLE_NAMES:
        raise ValueError(f"模块 B role 名称非法：{role_name}")
    return (get_module_b_work_dir(artifacts_dir) / safe_role_name).resolve()


def get_module_b_prompt_dir(artifacts_dir: Path, role_name: str) -> Path:
    """
    功能说明：返回模块 B 指定 role 的 prompt 产物目录。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    - role_name: role1/role2/role3/role4。
    返回值：
    - Path: artifacts/module_b_work/<role_name>/prompt 的绝对路径。
    异常说明：
    - ValueError: role_name 不在固定角色集合内。
    边界条件：仅解析路径，不主动创建目录。
    """
    return (get_module_b_role_dir(artifacts_dir, role_name) / MODULE_B_PROMPT_DIR_NAME).resolve()


def get_module_b_streaming_dir(artifacts_dir: Path, role_name: str) -> Path:
    """
    功能说明：返回模块 B 指定 role 的 streaming 产物目录。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    - role_name: role1/role2/role3/role4。
    返回值：
    - Path: artifacts/module_b_work/<role_name>/streaming 的绝对路径。
    异常说明：
    - ValueError: role_name 不在固定角色集合内。
    边界条件：仅解析路径，不主动创建目录。
    """
    return (get_module_b_role_dir(artifacts_dir, role_name) / MODULE_B_STREAMING_DIR_NAME).resolve()


def ensure_module_b_role_layout(artifacts_dir: Path) -> None:
    """
    功能说明：确保模块 B 四个 role 目录及其 prompt、streaming 子目录全部存在。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    返回值：无。
    异常说明：目录创建失败时由 pathlib 向上抛出。
    边界条件：不会删除或迁移任何已有文件。
    """
    for role_name in MODULE_B_ROLE_NAMES:
        get_module_b_prompt_dir(artifacts_dir, role_name).mkdir(parents=True, exist_ok=True)
        get_module_b_streaming_dir(artifacts_dir, role_name).mkdir(parents=True, exist_ok=True)


def get_module_b_role_result_path(artifacts_dir: Path, role_name: str) -> Path:
    """
    功能说明：返回模块 B 指定 role 的正式汇总 Markdown 产物路径。
    参数说明：
    - artifacts_dir: 当前任务 artifacts 目录。
    - role_name: role1/role2/role3/role4。
    返回值：
    - Path: 对应 role 的正式汇总 Markdown 文件路径。
    异常说明：
    - ValueError: role_name 不在固定角色集合内。
    边界条件：仅解析路径，不主动创建目录。
    """
    role_dir = get_module_b_role_dir(artifacts_dir, role_name)
    filename_map = {
        "role1": "role1_visual_output.md",
        "role2": "role2_story_output.md",
        "role3": "role3_shot_output.md",
        "role4": "role4_prompt_output.md",
    }
    safe_role_name = str(role_name or "").strip().lower()
    if safe_role_name not in filename_map:
        raise ValueError(f"模块 B role 名称非法：{role_name}")
    return (role_dir / filename_map[safe_role_name]).resolve()
