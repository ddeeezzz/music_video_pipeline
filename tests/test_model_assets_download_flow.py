"""
文件用途：验证 model_assets 下载扩展流程（交互路由、下载写入、回退策略）。
核心流程：构造临时项目目录 -> 打桩交互与下载执行 -> 断言配置写入与命令参数。
输入输出：输入临时路径与 monkeypatch，输出断言结果。
依赖说明：依赖 pytest 与 scripts.model_assets 相关模块。
维护说明：下载子菜单或注册表写入规则变更时需同步更新本测试。
"""

# 标准库：用于 JSON 读写
import json
# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path
# 标准库：用于导入路径补齐
import sys
# 标准库：用于简单命名空间桩对象
from types import SimpleNamespace

# 第三方库：用于测试断言与异常校验
import pytest

# 兼容 pytest 默认 pythonpath=src：补齐项目根目录，确保可导入 scripts.* 包。
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 项目内模块：model_assets 主入口
import scripts.model_assets.main as main_cli
# 项目内模块：下载引擎
from scripts.model_assets import download_engine
# 项目内模块：下载流程
from scripts.model_assets import download_flow


def test_main_should_route_to_download_assets_flow(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证主菜单选择下载功能时会路由到下载子流程。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩避免触发真实 bypy 调用。
    """

    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(log_path="log/model_assets.log")

    called: dict[str, str] = {}

    monkeypatch.setattr(main_cli, "build_parser", lambda: _FakeParser())
    monkeypatch.setattr(main_cli, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(main_cli, "BypyClient", lambda logger: object())
    monkeypatch.setattr(main_cli, "prompt_main_action", lambda: "download_assets")

    def _fake_run_download_assets_flow(project_root, logger, base_registry_path, bindings_path):
        called["project_root"] = str(project_root)
        called["base_registry_path"] = str(base_registry_path)
        called["bindings_path"] = str(bindings_path)
        return 0

    monkeypatch.setattr(main_cli, "run_download_assets_flow", _fake_run_download_assets_flow)
    exit_code = main_cli.main()
    assert exit_code == 0
    assert called["project_root"] == str(tmp_path)
    assert called["base_registry_path"].endswith("configs/base_model_registry.json")
    assert called["bindings_path"].endswith("configs/lora_bindings.json")


def test_main_should_route_to_task_sync_flow(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证主菜单选择“任务同步（手动）”时会路由到 task_sync 子流程。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩避免触发真实 bypy 与交互循环。
    """

    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(log_path="log/model_assets.log")

    called: dict[str, str] = {}

    monkeypatch.setattr(main_cli, "build_parser", lambda: _FakeParser())
    monkeypatch.setattr(main_cli, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(main_cli, "BypyClient", lambda logger: object())
    monkeypatch.setattr(main_cli, "prompt_main_action", lambda: "sync_tasks")

    def _fake_run_task_sync_flow(project_root, logger):  # noqa: ANN001
        _ = logger
        called["project_root"] = str(project_root)
        return 0

    monkeypatch.setattr(main_cli, "run_task_sync_flow", _fake_run_task_sync_flow)
    exit_code = main_cli.main()
    assert exit_code == 0
    assert called["project_root"] == str(tmp_path)


def test_run_lora_direct_download_should_write_binding_with_direct_url(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 LoRA 直链下载会正确写入 direct_url 来源绑定。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：下载执行通过桩函数直接写入文件。
    """
    project_root = tmp_path
    logger = logging.getLogger("test_lora_direct")
    logger.setLevel(logging.INFO)

    base_model_dir = project_root / "models" / "base_model" / "xl" / "diffusers" / "base_ok"
    base_model_dir.mkdir(parents=True, exist_ok=True)
    (base_model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    base_registry_path = project_root / "configs" / "base_model_registry.json"
    base_registry_path.parent.mkdir(parents=True, exist_ok=True)
    base_registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "base_models": [
                    {
                        "key": "base_xl_diffusers_base_ok",
                        "series": "xl",
                        "format": "diffusers",
                        "path": "models/base_model/xl/diffusers/base_ok",
                        "enabled": True,
                        "type": "directory",
                        "description": "test",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bindings_path = project_root / "configs" / "lora_bindings.json"
    bindings_path.write_text(json.dumps({"version": 1, "bindings": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(download_flow, "prompt_model_series", lambda title_text="": "xl")
    prompt_values = iter(
        [
            "https://example.com/assets/my_lora.safetensors",
            "my_lora_binding",
        ]
    )
    monkeypatch.setattr(download_flow, "prompt_text", lambda prompt_label, default_value="": next(prompt_values))
    monkeypatch.setattr(download_flow, "prompt_confirm", lambda prompt_label, default_no=True: True)

    def _fake_download_file(url, save_path, logger, max_retries, retry_wait_seconds, timeout_seconds):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"fake-lora")

    monkeypatch.setattr(download_flow, "download_file", _fake_download_file)

    download_flow.run_lora_direct_download_once(
        project_root=project_root,
        logger=logger,
        base_registry_path=base_registry_path,
        bindings_path=bindings_path,
    )

    bindings_data = json.loads(bindings_path.read_text(encoding="utf-8"))
    assert len(bindings_data["bindings"]) == 1
    record = bindings_data["bindings"][0]
    assert record["binding_name"] == "my_lora_binding"
    assert record["model_series"] == "xl"
    assert record["remote_dir"] == "direct_url:https://example.com/assets/my_lora.safetensors"
    assert record["base_model_key"] == "base_xl_diffusers_base_ok"
    assert record["lora_file"].startswith("models/lora/xl/my_lora_binding/")


def test_download_repo_with_hf_cli_should_include_required_command_parts(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 HF 仓库下载命令包含必要参数与镜像站环境变量。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过假 subprocess.run 避免真实网络下载。
    """
    logger = logging.getLogger("test_hf_cli")
    logger.setLevel(logging.INFO)
    target_dir = tmp_path / "repo_dir"
    captured: dict[str, object] = {}

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(download_engine.shutil, "which", lambda name: "/usr/bin/huggingface-cli")

    def _fake_subprocess_run(command, check=False, env=None):
        local_dir = Path(command[command.index("--local-dir") + 1])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model_index.json").write_text("{}", encoding="utf-8")
        captured["command"] = command
        captured["env"] = dict(env or {})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(download_engine.subprocess, "run", _fake_subprocess_run)

    download_engine.download_repo_with_hf_cli(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision="main",
        local_dir=target_dir,
        logger=logger,
        max_retries=1,
        retry_wait_seconds=0.1,
        include_patterns=("model_index.json", "unet/*"),
    )

    command = list(captured["command"])
    assert command[0] == "/usr/bin/huggingface-cli"
    assert "--resume-download" in command
    assert "--local-dir-use-symlinks" in command
    assert "False" in command
    assert "--include" in command
    assert captured["env"]["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_run_download_assets_flow_should_continue_when_repo_id_missing_for_fl(tmp_path: Path, monkeypatch, capsys) -> None:
    """
    功能说明：验证 fl 系列仓库下载缺少 repo_id 时不会中断下载主循环。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    - capsys: pytest 输出捕获工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：首次失败后应允许继续输入下一次菜单动作。
    """
    logger = logging.getLogger("test_download_loop")
    logger.setLevel(logging.INFO)

    action_values = iter(["download_base_repo", None])
    monkeypatch.setattr(download_flow, "prompt_download_action", lambda: next(action_values))
    monkeypatch.setattr(download_flow, "prompt_model_series", lambda title_text="": "fl")
    text_values = iter(["", "main"])
    monkeypatch.setattr(download_flow, "prompt_text", lambda prompt_label, default_value="": next(text_values))

    exit_code = download_flow.run_download_assets_flow(
        project_root=tmp_path,
        logger=logger,
        base_registry_path=tmp_path / "configs" / "base_model_registry.json",
        bindings_path=tmp_path / "configs" / "lora_bindings.json",
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "系列 fl 必须输入 repo_id" in captured.out


def test_run_base_model_direct_download_should_reject_outside_single_dir(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证 BaseModel 直链下载路径越界会被拒绝。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：
    - RuntimeError: 越界路径应抛异常。
    边界条件：不触发真实下载执行。
    """
    logger = logging.getLogger("test_base_direct_reject")
    logger.setLevel(logging.INFO)

    monkeypatch.setattr(download_flow, "prompt_model_series", lambda title_text="": "xl")
    text_values = iter(["https://example.com/model.safetensors", "../outside/model.safetensors"])
    monkeypatch.setattr(download_flow, "prompt_text", lambda prompt_label, default_value="": next(text_values))

    with pytest.raises(RuntimeError, match="保存路径必须位于"):
        download_flow.run_base_model_direct_download_once(
            project_root=tmp_path,
            logger=logger,
            base_registry_path=tmp_path / "configs" / "base_model_registry.json",
        )


def test_download_file_should_fallback_from_aria2c_to_wget(tmp_path: Path, monkeypatch) -> None:
    """
    功能说明：验证文件下载后端会从 aria2c 失败回退到 wget。
    参数说明：
    - tmp_path: pytest 临时目录。
    - monkeypatch: pytest 打桩工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：各后端执行通过打桩实现，不触发真实网络。
    """
    logger = logging.getLogger("test_download_fallback")
    logger.setLevel(logging.INFO)
    save_path = tmp_path / "model.safetensors"
    calls: list[str] = []

    monkeypatch.setattr(
        download_engine.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"aria2c", "wget"} else None,
    )

    def _fake_aria2c(url, save_path, timeout_seconds):
        calls.append("aria2c")
        raise download_engine.DownloadEngineError("aria2c failed")

    def _fake_wget(url, save_path, timeout_seconds):
        calls.append("wget")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"ok")

    monkeypatch.setattr(download_engine, "_download_with_aria2c", _fake_aria2c)
    monkeypatch.setattr(download_engine, "_download_with_wget", _fake_wget)

    download_engine.download_file(
        url="https://example.com/model.safetensors",
        save_path=save_path,
        logger=logger,
        max_retries=1,
        retry_wait_seconds=0.1,
        timeout_seconds=10,
    )

    assert calls == ["aria2c", "wget"]
    assert save_path.exists()
    assert save_path.stat().st_size > 0
