"""
文件用途：验证任务输入音频路径在跨机器迁移场景下的回映射逻辑。
核心流程：构造旧外机绝对路径、本地 resources 音频与任务配置，断言能解析为当前工作区真实文件。
输入输出：输入临时工作区与任务配置，输出路径解析结果断言。
依赖说明：依赖 pytest 与项目内任务音频路径解析工具。
维护说明：若任务音频来源候选规则调整，需同步更新本测试。
"""

# 标准库：用于 JSON 配置写入
import json
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：任务音频路径解析
from music_video_pipeline.task_audio_path import find_task_audio_path, resolve_task_audio_path


def test_resolve_task_audio_path_should_fallback_to_config_default_audio(tmp_path: Path) -> None:
    """
    功能说明：验证旧外机绝对路径失效时，可回退到任务配置中的默认本地音频。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：默认音频路径使用工作区相对路径表示。
    """
    workspace_root = tmp_path / "workspace_audio_from_config"
    resources_dir = workspace_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    local_audio_path = (resources_dir / "jieranduhuo01.mp3").resolve()
    local_audio_path.write_bytes(b"fake-audio")

    config_path = workspace_root / "configs" / "music_windows_4060" / "jieranduhuo.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"paths": {"default_audio_path": "resources/jieranduhuo01.mp3"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    resolved_path = resolve_task_audio_path(
        raw_audio_path="\\root\\data\\t1\\resources\\jieranduhuo.mp3",
        config_path=str(config_path),
        workspace_roots=[workspace_root],
    )

    assert resolved_path == local_audio_path


def test_find_task_audio_path_should_map_foreign_resources_tail_into_workspace(tmp_path: Path) -> None:
    """
    功能说明：验证旧外机路径与本地 resources 同名文件可直接重映射。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不依赖任务配置文件即可命中 basename/resources 回退。
    """
    workspace_root = tmp_path / "workspace_audio_from_resources"
    resources_dir = workspace_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    local_audio_path = (resources_dir / "demo-song.mp3").resolve()
    local_audio_path.write_bytes(b"fake-audio")

    resolved_path = find_task_audio_path(
        raw_audio_path="/root/data/t1/resources/demo-song.mp3",
        workspace_roots=[workspace_root],
    )

    assert resolved_path == local_audio_path
