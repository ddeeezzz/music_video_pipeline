"""
文件用途：验证 model_assets 的 BaseModel 远端索引与同步兼容“single 目录下直接文件”的布局。
核心流程：构造 bypy 远端输出桩对象 -> 构建菜单项 -> 执行单文件同步 -> 断言注册表写入。
输入输出：输入临时项目目录与 monkeypatch/桩客户端，输出断言结果。
依赖说明：依赖 pytest 与 scripts.model_assets 的 indexer/sync_base_model/store 模块。
维护说明：当 BaseModel 远端目录契约扩展时，应同步补充本测试。
"""

# 标准库：用于 JSON 读写断言
import json
# 标准库：用于日志对象构造
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于导入路径补齐
import sys

# 第三方库：用于测试断言
import pytest

# 兼容 pytest 默认 pythonpath=src：补齐项目根目录，确保可导入 scripts.* 包。
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 项目内模块：远端菜单索引
from scripts.model_assets.indexer import build_remote_options
# 项目内模块：底模同步逻辑
from scripts.model_assets.sync_base_model import sync_base_model_item


class _FakeListClient:
    """
    功能说明：提供最小 bypy list 读取能力的测试桩客户端。
    参数说明：
    - outputs: 远端目录到输出文本的映射。
    返回值：不适用。
    异常说明：未命中路径时抛 AssertionError，便于测试快速定位。
    边界条件：仅覆盖本测试需要的 list_remote 接口。
    """

    def __init__(self, outputs: dict[str, str]) -> None:
        self._outputs = outputs

    def list_remote(self, remote_dir: str) -> str:
        """
        功能说明：返回指定远端目录的伪造 bypy 输出。
        参数说明：
        - remote_dir: 远端目录路径。
        返回值：
        - str: 对应目录输出文本。
        异常说明：
        - AssertionError: 测试未提供该目录输出时抛出。
        边界条件：无。
        """
        assert remote_dir in self._outputs, f"测试未提供远端目录输出：{remote_dir}"
        return self._outputs[remote_dir]


class _FakeSyncClient:
    """
    功能说明：提供最小 BaseModel 下载能力的测试桩客户端。
    参数说明：无。
    返回值：不适用。
    异常说明：不适用。
    边界条件：仅覆盖本测试需要的 downfile/downdir 接口。
    """

    def downfile(self, remote_file: str, local_path: Path) -> None:
        """
        功能说明：模拟下载单文件 BaseModel。
        参数说明：
        - remote_file: 远端文件路径。
        - local_path: 本地文件路径。
        返回值：无。
        异常说明：无。
        边界条件：会自动创建父目录并写入非空内容。
        """
        _ = remote_file
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake-checkpoint")

    def downdir(self, remote_dir: str, local_dir: Path) -> None:
        """
        功能说明：模拟下载目录型 BaseModel。
        参数说明：
        - remote_dir: 远端目录路径。
        - local_dir: 本地目录路径。
        返回值：无。
        异常说明：无。
        边界条件：当前测试不应触发该分支，若触发则直接失败。
        """
        raise AssertionError(f"当前测试不应调用 downdir：{remote_dir} -> {local_dir}")


def test_build_remote_options_should_include_single_files_for_base_model(tmp_path: Path) -> None:
    """
    功能说明：验证 BaseModel 远端索引会识别 single 目录下直接存放的文件。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅校验 single 文件型条目，不覆盖 diffusers 目录型分支。
    """
    logger = logging.getLogger("test_model_assets_index_single_file")
    logger.setLevel(logging.INFO)
    client = _FakeListClient(
        outputs={
            "/base_model": "/apps/bypy/base_model ($t $f $s $m $d):\nD 15 0 2026-04-16, 19:48:19 \n",
            "/base_model/15": (
                "/apps/bypy/base_model/15 ($t $f $s $m $d):\n"
                "D diffusers 0 2026-04-16, 19:48:19 \n"
                "D single 0 2026-04-27, 16:00:37 \n"
            ),
            "/base_model/xl": "/apps/bypy/base_model/xl ($t $f $s $m $d):\n",
            "/base_model/fl": "/apps/bypy/base_model/fl ($t $f $s $m $d):\n",
            "/base_model/15/diffusers": "/apps/bypy/base_model/15/diffusers ($t $f $s $m $d):\n",
            "/base_model/15/single": (
                "/apps/bypy/base_model/15/single ($t $f $s $m $d):\n"
                "F anything-v5.safetensors 2132626102 2026-04-27, 16:00:37 cde123c19saee3e1fc34f61c1054ec31\n"
            ),
        }
    )

    options = build_remote_options(
        resource_type="base_model",
        project_root=tmp_path,
        client=client,
        logger=logger,
    )

    assert len(options) == 1
    assert options[0]["series"] == "15"
    assert options[0]["format"] == "single"
    assert options[0]["name"] == "anything-v5.safetensors"
    assert options[0]["remote_kind"] == "file"
    assert options[0]["remote_file"] == "/base_model/15/single/anything-v5.safetensors"


def test_sync_base_model_item_should_support_single_remote_file(tmp_path: Path) -> None:
    """
    功能说明：验证 BaseModel 同步可处理 single 目录下的远端单文件并写入 file 类型注册表。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不依赖真实 bypy 网络请求。
    """
    logger = logging.getLogger("test_model_assets_sync_single_file")
    logger.setLevel(logging.INFO)
    base_registry_path = tmp_path / "configs" / "base_model_registry.json"

    result = sync_base_model_item(
        project_root=tmp_path,
        logger=logger,
        client=_FakeSyncClient(),
        option={
            "series": "15",
            "format": "single",
            "name": "anything-v5.safetensors",
            "remote_file": "/base_model/15/single/anything-v5.safetensors",
            "remote_kind": "file",
        },
        base_registry_path=base_registry_path,
    )

    local_file = tmp_path / "models" / "base_model" / "15" / "single" / "anything-v5.safetensors"
    assert local_file.exists()
    assert local_file.read_bytes() == b"fake-checkpoint"
    assert result["key"] == "base_15_single_anything_v5_safetensors"
    assert result["remote_dir"] == "/base_model/15/single/anything-v5.safetensors"

    registry_data = json.loads(base_registry_path.read_text(encoding="utf-8"))
    matched = next(
        item for item in registry_data["base_models"] if item.get("key") == "base_15_single_anything_v5_safetensors"
    )
    assert matched["type"] == "file"
    assert matched["path"] == "models/base_model/15/single/anything-v5.safetensors"
    assert matched["format"] == "single"

