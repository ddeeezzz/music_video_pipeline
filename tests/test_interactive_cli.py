"""
文件用途：验证交互式 CLI 的关键交互行为（菜单、执行、会话记忆）。
核心流程：打桩 input 与 execute_request，断言请求构造和退出流程。
"""

from __future__ import annotations

import json
from pathlib import Path

from music_video_pipeline import interactive_cli
from music_video_pipeline.state_store import StateStore


def test_session_memory_save_load_and_clear(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "interactive_session.json"
    monkeypatch.setattr(interactive_cli, "_session_file_path", lambda: session_path)

    memory = interactive_cli.SessionMemory(task_id="task_001", config_path="/tmp/cfg.json", audio_path="/tmp/a.mp3")
    interactive_cli._save_session_memory(memory=memory)

    loaded = interactive_cli._load_session_memory()
    assert loaded.task_id == "task_001"
    assert loaded.config_path == "/tmp/cfg.json"
    assert loaded.audio_path == "/tmp/a.mp3"

    interactive_cli._clear_session_memory(memory=loaded)
    assert loaded.task_id == ""
    assert loaded.config_path == ""
    assert loaded.audio_path == ""
    assert not session_path.exists()


def test_run_interactive_cli_should_collect_run_and_execute(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "interactive_session.json"
    monkeypatch.setattr(interactive_cli, "_session_file_path", lambda: session_path)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    default_config_path = workspace_root / "configs" / "default.json"
    default_config_path.parent.mkdir(parents=True, exist_ok=True)
    default_config_path.write_text("{}", encoding="utf-8")

    main_exit_index = len(interactive_cli.MAIN_ACTIONS) + 3
    inputs = iter(
        [
            "1",  # 主菜单 -> run
            "task_interactive_001",  # task_id
            "",  # config 使用默认
            "-",  # 清空 audio，回退到配置默认
            "",  # 模块B user_custom_prompt 默认不添加（等价空）
            "y",  # 确认执行
            "1",  # 成功后返回主菜单
            str(main_exit_index),  # 主菜单 -> 退出
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    captured: list[object] = []

    def _fake_execute_request(request):  # noqa: ANN001
        captured.append(request)
        return {"ok": True, "command": request.command, "task_id": request.task_id}

    exit_code = interactive_cli.run_interactive_cli(
        workspace_root=workspace_root,
        default_config_path=default_config_path,
        execute_request=_fake_execute_request,
    )

    assert exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.command == "run"
    assert request.task_id == "task_interactive_001"
    assert request.config_path == default_config_path.resolve()
    assert request.audio_path is None
    assert request.user_custom_prompt_override == ""

    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["task_id"] == "task_interactive_001"
    assert str(saved["config_path"]).endswith("configs/default.json")


def test_select_rerun_task_context_should_return_none_when_no_db(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(interactive_cli, "_discover_state_db_paths", lambda **_: [])
    context = interactive_cli._select_rerun_task_context(
        workspace_root=tmp_path / "workspace",
        default_config_path=tmp_path / "workspace" / "configs" / "default.json",
    )
    assert context is None


def test_collect_run_force_should_pick_task_from_db_and_reuse_audio_config(tmp_path: Path, monkeypatch, capsys: object) -> None:
    session_path = tmp_path / "interactive_session.json"
    monkeypatch.setattr(interactive_cli, "_session_file_path", lambda: session_path)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "pipeline_state.sqlite3"
    store = StateStore(db_path=db_path)
    store.init_task(
        task_id="task_force_001",
        audio_path="/abs/audio_force.mp3",
        config_path="/abs/config_force.json",
    )
    store.set_module_status(task_id="task_force_001", module_name="A", status="done", artifact_path="a.json")
    store.set_module_status(task_id="task_force_001", module_name="B", status="running")

    monkeypatch.setattr(interactive_cli, "_discover_state_db_paths", lambda **_: [db_path])

    main_advanced_index = len(interactive_cli.MAIN_ACTIONS) + 1
    main_exit_index = len(interactive_cli.MAIN_ACTIONS) + 3
    advanced_back_index = len(interactive_cli.ADVANCED_ACTIONS) + 1
    inputs = iter(
        [
            str(main_advanced_index),  # 主菜单 -> 高级菜单
            "y",  # 确认进入高级菜单
            "1",  # 高级菜单 -> run-force
            "1",  # 选择状态库
            "1",  # 选择任务
            "3",  # force-module -> C
            "y",  # 执行确认
            "1",  # 执行成功后留在高级菜单
            str(advanced_back_index),  # 高级菜单 -> 返回主菜单
            str(main_exit_index),  # 主菜单 -> 退出
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    captured: list[object] = []

    def _fake_execute_request(request):  # noqa: ANN001
        captured.append(request)
        return {"ok": True, "command": request.command, "task_id": request.task_id}

    exit_code = interactive_cli.run_interactive_cli(
        workspace_root=workspace_root,
        default_config_path=workspace_root / "configs" / "default.json",
        execute_request=_fake_execute_request,
    )
    assert exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.command == "run"
    assert request.task_id == "task_force_001"
    assert request.force_module == "C"
    assert str(request.config_path) == "/abs/config_force.json"
    assert str(request.audio_path) == "/abs/audio_force.mp3"
    captured = capsys.readouterr()
    assert "操作提示（run（含 --force-module））" in captured.out
    assert "请选择状态库" in captured.out
    assert "请选择任务" in captured.out
    assert "任务ID（输入 q 返回上一步）" not in captured.out
    assert "配置文件路径（输入 q 返回上一步）" not in captured.out


def test_prompt_task_choice_should_show_module_statuses(capsys: object) -> None:
    task_rows = [
        {
            "task_id": "task_ui_001",
            "status": "running",
            "updated_at": "2026-04-17T18:00:00+08:00",
        }
    ]
    module_summary_map = {
        "task_ui_001": {
            "A": "done",
            "B": "running",
            "C": "pending",
            "D": "failed",
        }
    }
    # 使用 monkeypatch 不方便时，直接验证渲染函数输出内容。
    from unittest.mock import patch

    with patch("builtins.input", return_value="1"):
        selected = interactive_cli._prompt_task_choice(task_rows=task_rows, module_summary_map=module_summary_map)
    captured = capsys.readouterr()
    assert selected is not None
    assert "A:done" in captured.out
    assert "B:running" in captured.out
    assert "C:pending" in captured.out
    assert "D:failed" in captured.out


def test_select_rerun_task_context_should_return_none_when_selected_db_has_no_tasks(
    tmp_path: Path, monkeypatch, capsys: object
) -> None:
    db_path = tmp_path / "runs" / "pipeline_state.sqlite3"
    StateStore(db_path=db_path)
    monkeypatch.setattr(interactive_cli, "_discover_state_db_paths", lambda **_: [db_path])
    inputs = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    context = interactive_cli._select_rerun_task_context(
        workspace_root=tmp_path / "workspace",
        default_config_path=tmp_path / "workspace" / "configs" / "default.json",
    )
    assert context is None
    captured = capsys.readouterr()
    assert "所选状态库暂无任务" in captured.out


def test_select_rerun_task_context_should_allow_back_from_task_to_db(
    tmp_path: Path, monkeypatch
) -> None:
    db_path_1 = tmp_path / "runs_1" / "pipeline_state.sqlite3"
    db_path_2 = tmp_path / "runs_2" / "pipeline_state.sqlite3"
    store_1 = StateStore(db_path=db_path_1)
    store_2 = StateStore(db_path=db_path_2)
    store_1.init_task(task_id="task_db_1", audio_path="/a/1.mp3", config_path="/a/1.json")
    store_2.init_task(task_id="task_db_2", audio_path="/a/2.mp3", config_path="/a/2.json")

    monkeypatch.setattr(interactive_cli, "_discover_state_db_paths", lambda **_: [db_path_1, db_path_2])
    inputs = iter(
        [
            "1",  # 首次选 db_path_1
            "q",  # 任务选择层返回，回到 DB 选择层
            "2",  # 改选 db_path_2
            "1",  # 选择 task_db_2
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    context = interactive_cli._select_rerun_task_context(
        workspace_root=tmp_path / "workspace",
        default_config_path=tmp_path / "workspace" / "configs" / "default.json",
    )
    assert context is not None
    assert context.task_id == "task_db_2"
    assert context.db_path == db_path_2


def test_request_can_trigger_module_b_should_match_expected_commands() -> None:
    assert interactive_cli._request_can_trigger_module_b(
        request=interactive_cli.CommandRequest(command="run", config_path=Path("/tmp/config.json"))
    )
    assert not interactive_cli._request_can_trigger_module_b(
        request=interactive_cli.CommandRequest(
            command="resume",
            config_path=Path("/tmp/config.json"),
            force_module="C",
        )
    )
    assert interactive_cli._request_can_trigger_module_b(
        request=interactive_cli.CommandRequest(
            command="run-module",
            config_path=Path("/tmp/config.json"),
            module="B",
        )
    )
    assert not interactive_cli._request_can_trigger_module_b(
        request=interactive_cli.CommandRequest(
            command="run-module",
            config_path=Path("/tmp/config.json"),
            module="D",
        )
    )


def test_collect_resume_request_should_render_current_module_status(tmp_path: Path, monkeypatch, capsys: object) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "pipeline_state.sqlite3"
    store = StateStore(db_path=db_path)
    store.init_task(task_id="task_resume_status_001", audio_path="/a.mp3", config_path="/cfg.json")
    store.update_task_status(task_id="task_resume_status_001", status="running")
    store.set_module_status(task_id="task_resume_status_001", module_name="A", status="done")
    store.set_module_status(task_id="task_resume_status_001", module_name="B", status="done")
    store.set_module_status(task_id="task_resume_status_001", module_name="C", status="running")
    store.set_module_status(task_id="task_resume_status_001", module_name="D", status="pending")

    config_path = workspace_root / "configs" / "resume_status.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"paths": {"runs_dir": str(runs_dir)}}), encoding="utf-8")

    memory = interactive_cli.SessionMemory()
    inputs = iter(
        [
            "task_resume_status_001",
            str(config_path),
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    request = interactive_cli._collect_resume_request(
        workspace_root=workspace_root,
        default_config_path=config_path,
        memory=memory,
    )
    assert request is not None
    assert request.command == "resume"
    assert request.task_id == "task_resume_status_001"
    captured = capsys.readouterr()
    assert "当前模块完成状态：" in captured.out
    assert "task=running" in captured.out
    assert "A:done" in captured.out
    assert "B:done" in captured.out
    assert "C:running" in captured.out
    assert "D:pending" in captured.out


def test_discover_user_prompt_template_options_should_scan_versions(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    prompt_dir = workspace_root / "configs" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "module_b_prompt.v1.md").write_text(
        "# Module B Prompt Template v1\n\n## system_prompt\nX\n\n## user_prompt_template\n版本一提示词\n\n## retry_hint_template\n补救要求：{{retry_hint}}\n",
        encoding="utf-8",
    )
    (prompt_dir / "module_b_prompt.v2.md").write_text(
        "# Module B Prompt Template v2\n\n## system_prompt\nX\n\n## user_prompt_template\n版本二提示词\n\n## retry_hint_template\n补救要求：{{retry_hint}}\n",
        encoding="utf-8",
    )
    (prompt_dir / "module_b_prompt.v3.md").write_text(
        "# Module B Prompt Template v3\n\n## system_prompt\nX\n\n## retry_hint_template\n补救要求：{{retry_hint}}\n",
        encoding="utf-8",
    )

    options = interactive_cli._discover_user_prompt_template_options(workspace_root=workspace_root)
    assert [item["version"] for item in options] == ["module_b_prompt.v1", "module_b_prompt.v2"]
    assert options[0]["content"] == "版本一提示词"
    assert options[1]["content"] == "版本二提示词"


def test_attach_user_custom_prompt_override_should_only_apply_to_b_commands(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    called = {"value": 0}

    def _fake_prompt(*, workspace_root: Path) -> str:  # noqa: ANN001
        _ = workspace_root
        called["value"] += 1
        return "赛博朋克女孩"

    monkeypatch.setattr(interactive_cli, "_prompt_user_custom_prompt_override", _fake_prompt)

    request_b = interactive_cli.CommandRequest(
        command="run-module",
        config_path=Path("/tmp/config.json"),
        module="B",
    )
    patched_b = interactive_cli._attach_user_custom_prompt_override_if_needed(
        request=request_b,
        workspace_root=workspace_root,
    )
    assert patched_b.user_custom_prompt_override == "赛博朋克女孩"
    assert called["value"] == 1

    request_d = interactive_cli.CommandRequest(
        command="run-module",
        config_path=Path("/tmp/config.json"),
        module="D",
    )
    patched_d = interactive_cli._attach_user_custom_prompt_override_if_needed(
        request=request_d,
        workspace_root=workspace_root,
    )
    assert patched_d.user_custom_prompt_override is None
    assert called["value"] == 1


def test_attach_user_custom_prompt_override_should_skip_resume_when_module_b_done(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "pipeline_state.sqlite3"
    store = StateStore(db_path=db_path)
    store.init_task(task_id="task_resume_001", audio_path="/a.mp3", config_path="/cfg.json")
    store.set_module_status(task_id="task_resume_001", module_name="B", status="done")

    config_path = workspace_root / "configs" / "resume.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"paths": {"runs_dir": str(runs_dir)}}), encoding="utf-8")

    called = {"value": 0}

    def _fake_prompt(*, workspace_root: Path) -> str:  # noqa: ANN001
        _ = workspace_root
        called["value"] += 1
        return "不应被调用"

    monkeypatch.setattr(interactive_cli, "_prompt_user_custom_prompt_override", _fake_prompt)
    request = interactive_cli.CommandRequest(
        command="resume",
        task_id="task_resume_001",
        config_path=config_path,
    )
    patched = interactive_cli._attach_user_custom_prompt_override_if_needed(
        request=request,
        workspace_root=workspace_root,
    )
    assert patched.user_custom_prompt_override is None
    assert called["value"] == 0


def test_attach_user_custom_prompt_override_should_skip_resume_when_module_b_units_all_done(
    tmp_path: Path, monkeypatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "pipeline_state.sqlite3"
    store = StateStore(db_path=db_path)
    store.init_task(task_id="task_resume_002", audio_path="/a.mp3", config_path="/cfg.json")
    store.set_module_status(task_id="task_resume_002", module_name="B", status="running")
    store.sync_module_units(
        task_id="task_resume_002",
        module_name="B",
        units=[{"unit_id": "seg_001", "unit_index": 0, "start_time": 0.0, "end_time": 1.0, "duration": 1.0}],
    )
    store.set_module_unit_status(task_id="task_resume_002", module_name="B", unit_id="seg_001", status="done")

    config_path = workspace_root / "configs" / "resume_units_done.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"paths": {"runs_dir": str(runs_dir)}}), encoding="utf-8")

    called = {"value": 0}

    def _fake_prompt(*, workspace_root: Path) -> str:  # noqa: ANN001
        _ = workspace_root
        called["value"] += 1
        return "不应被调用"

    monkeypatch.setattr(interactive_cli, "_prompt_user_custom_prompt_override", _fake_prompt)
    request = interactive_cli.CommandRequest(
        command="resume",
        task_id="task_resume_002",
        config_path=config_path,
    )
    patched = interactive_cli._attach_user_custom_prompt_override_if_needed(
        request=request,
        workspace_root=workspace_root,
    )
    assert patched.user_custom_prompt_override is None
    assert called["value"] == 0
