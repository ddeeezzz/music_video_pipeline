"""
文件用途：验证模块B LLM分镜生成器的双语提示词回填与失败行为。
核心流程：打桩真实LLM调用函数，检查 generate_one 返回结构。
输入输出：输入伪造模块A与segment数据，输出shot断言结果。
依赖说明：依赖 pytest 与 LlmScriptGenerator。
维护说明：若分镜字段契约调整，需同步更新本测试。
"""

# 标准库：用于日志对象构建
import logging
# 标准库：用于路径处理
from pathlib import Path

# 第三方库：用于异常断言
import pytest

# 项目内模块：模块B配置类型
from music_video_pipeline.config import ModuleBConfig, ModuleBLlmConfig
# 项目内模块：模块B LLM生成异常类型
from music_video_pipeline.modules.module_b import llm_generator as llm_generator_module
from music_video_pipeline.modules.module_b.llm_generator import ModuleBLlmGenerationError, generate_module_b_prompts
# 项目内模块：分镜生成器
from music_video_pipeline.generators import script_generator as script_generator_module


def test_llm_script_generator_should_fill_bilingual_prompt_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能说明：验证LLM分镜生成器可回填 scene_desc 与 keyframe/video 的中英文字段。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：camera_motion/transition 仍由规则逻辑生成。
    """

    def _fake_generate_module_b_prompts(logger, llm_config, llm_input_payload, project_root):
        _ = (logger, llm_config, project_root)
        assert "memory_context" in llm_input_payload
        assert "current_segment" in llm_input_payload
        assert llm_input_payload["current_segment"]["segment_id"] == "seg_0001"
        assert llm_input_payload["memory_context"]["recent_history"] == []
        return {
            "scene_desc": "雨夜街口，人物在霓虹下停步回望，镜头缓慢推进。",
            "keyframe_prompt_zh": "电影感关键帧，雨夜霓虹街道，孤立主体，中景，构图稳定",
            "keyframe_prompt_en": "cinematic keyframe, rainy neon street, lone figure, soft rim light, medium shot, film still",
            "video_prompt_zh": "电影感视频提示词，雨夜街道，慢速推进，风雨细微运动，忧郁氛围",
            "video_prompt_en": "cinematic video prompt, rainy neon street, slow push-in, gentle wind and rain motion, melancholic mood",
            "keyframe_prompt": "cinematic keyframe, rainy neon street, lone figure, soft rim light, medium shot, film still",
            "video_prompt": "cinematic video prompt, rainy neon street, slow push-in, gentle wind and rain motion, melancholic mood",
        }

    monkeypatch.setattr(script_generator_module, "generate_module_b_prompts", _fake_generate_module_b_prompts)
    generator = script_generator_module.LlmScriptGenerator(
        logger=logging.getLogger("test_llm_script_generator"),
        module_b_config=ModuleBConfig(),
    )

    shot = generator.generate_one(
        module_a_output={
            "big_segments": [{"segment_id": "big_001", "label": "verse"}],
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "big_segment_id": "big_001",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "label": "verse",
                }
            ],
            "energy_features": [{"energy_level": "mid", "trend": "flat"}],
            "lyric_units": [{"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.5, "text": "第一句", "confidence": 0.8}],
        },
        segment={
            "segment_id": "seg_0001",
            "big_segment_id": "big_001",
            "start_time": 0.0,
            "end_time": 2.0,
            "label": "verse",
        },
        segment_index=0,
    )

    assert shot["scene_desc"].startswith("雨夜街口")
    assert "雨夜霓虹街道" in shot["keyframe_prompt_zh"]
    assert "rainy neon street" in shot["keyframe_prompt_en"]
    assert "雨夜街道" in shot["video_prompt_zh"]
    assert "rainy neon street" in shot["video_prompt_en"]
    # 兼容字段默认指向英文版本。
    assert "neon street" in shot["keyframe_prompt"]
    assert "video prompt" in shot["video_prompt"]
    assert shot["camera_motion"] in {"none", "slow_pan", "zoom_in", "shake", "push_pull"}
    assert shot["transition"] in {"crossfade", "hard_cut"}


def test_llm_script_generator_should_raise_when_llm_generation_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能说明：验证LLM双语提示词生成失败时会抛出可定位异常。
    参数说明：
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：错误信息应包含 segment_id，便于模块B单元排障。
    """

    def _fake_generate_module_b_prompts(logger, llm_config, llm_input_payload, project_root):
        _ = (logger, llm_config, llm_input_payload, project_root)
        raise ModuleBLlmGenerationError("mock llm failed")

    monkeypatch.setattr(script_generator_module, "generate_module_b_prompts", _fake_generate_module_b_prompts)
    generator = script_generator_module.LlmScriptGenerator(
        logger=logging.getLogger("test_llm_script_generator_fail"),
        module_b_config=ModuleBConfig(),
    )

    with pytest.raises(RuntimeError, match="segment_id=seg_0001"):
        generator.generate_one(
            module_a_output={
                "big_segments": [{"segment_id": "big_001", "label": "verse"}],
                "segments": [{"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 1.0, "label": "verse"}],
                "energy_features": [{"energy_level": "mid", "trend": "flat"}],
                "lyric_units": [],
            },
            segment={
                "segment_id": "seg_0001",
                "big_segment_id": "big_001",
                "start_time": 0.0,
                "end_time": 1.0,
                "label": "verse",
            },
            segment_index=0,
        )


def test_generate_module_b_prompts_should_render_external_prompt_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能说明：验证模块B会加载外置模板并替换 input_payload 占位符。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：无需真实网络请求，LLM客户端由补丁函数模拟。
    """
    template_path = tmp_path / "module_b_prompt.v1.md"
    template_path.write_text(
        (
            "# Module B Prompt Template v1\n\n"
            "## system_prompt\n"
            "你是测试系统提示。输入数据：{{input_payload_json}}\n\n"
            "## user_prompt_template\n"
            "{{user_custom_prompt}}\n\n"
            "## retry_hint_template\n"
            "补救要求：{{retry_hint}}\n"
        ),
        encoding="utf-8",
    )
    llm_config = ModuleBLlmConfig(
        prompt_template_file=str(template_path),
        json_retry_times=0,
        user_custom_prompt="赛博朋克女孩",
    )
    captured_messages: list[list[dict[str, str]]] = []

    def _fake_call_module_b_llm_chat(logger, llm_config, messages, project_root):
        _ = (logger, llm_config, project_root)
        captured_messages.append(messages)
        return (
            "{\"scene_desc\":\"中文描述\","
            "\"keyframe_prompt_zh\":\"中文关键帧\","
            "\"keyframe_prompt_en\":\"english keyframe\","
            "\"video_prompt_zh\":\"中文视频\","
            "\"video_prompt_en\":\"english video\"}"
        )

    monkeypatch.setattr(llm_generator_module, "call_module_b_llm_chat", _fake_call_module_b_llm_chat)
    result = generate_module_b_prompts(
        logger=logging.getLogger("test_generate_module_b_prompts_template"),
        llm_config=llm_config,
        llm_input_payload={"segment_id": "seg_0001", "lyric_text": "第一句"},
        project_root=tmp_path,
    )

    assert result["scene_desc"] == "中文描述"
    assert len(captured_messages) == 1
    assert "\"segment_id\":\"seg_0001\"" in captured_messages[0][0]["content"]
    assert "{{input_payload_json}}" not in captured_messages[0][0]["content"]
    assert captured_messages[0][1]["content"] == "赛博朋克女孩"


def test_generate_module_b_prompts_should_fail_when_template_missing(tmp_path: Path) -> None:
    """
    功能说明：验证模板文件不存在时，模块B会直接失败。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：失败发生在调用LLM前，不进入网络请求流程。
    """
    llm_config = ModuleBLlmConfig(
        prompt_template_file=str(tmp_path / "not_exists.md"),
        json_retry_times=0,
    )
    with pytest.raises(ModuleBLlmGenerationError, match="模板加载失败"):
        generate_module_b_prompts(
            logger=logging.getLogger("test_generate_module_b_prompts_missing_template"),
            llm_config=llm_config,
            llm_input_payload={"segment_id": "seg_0001"},
            project_root=tmp_path,
        )


def test_generate_module_b_prompts_should_allow_empty_user_custom_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能说明：验证 user_custom_prompt 为空时，user message 可为空字符串。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：system message 仍应包含已渲染的输入 payload。
    """
    template_path = tmp_path / "module_b_prompt_empty_user.v1.md"
    template_path.write_text(
        (
            "# Module B Prompt Template v1\n\n"
            "## system_prompt\n"
            "系统提示：{{input_payload_json}}\n\n"
            "## user_prompt_template\n"
            "{{user_custom_prompt}}\n\n"
            "## retry_hint_template\n"
            "补救要求：{{retry_hint}}\n"
        ),
        encoding="utf-8",
    )
    llm_config = ModuleBLlmConfig(prompt_template_file=str(template_path), json_retry_times=0, user_custom_prompt="")
    captured_messages: list[list[dict[str, str]]] = []

    def _fake_call_module_b_llm_chat(logger, llm_config, messages, project_root):
        _ = (logger, llm_config, project_root)
        captured_messages.append(messages)
        return (
            "{\"scene_desc\":\"中文描述\","
            "\"keyframe_prompt_zh\":\"中文关键帧\","
            "\"keyframe_prompt_en\":\"english keyframe\","
            "\"video_prompt_zh\":\"中文视频\","
            "\"video_prompt_en\":\"english video\"}"
        )

    monkeypatch.setattr(llm_generator_module, "call_module_b_llm_chat", _fake_call_module_b_llm_chat)
    generate_module_b_prompts(
        logger=logging.getLogger("test_generate_module_b_prompts_empty_user_prompt"),
        llm_config=llm_config,
        llm_input_payload={"segment_id": "seg_0002"},
        project_root=tmp_path,
    )

    assert len(captured_messages) == 1
    assert "\"segment_id\":\"seg_0002\"" in captured_messages[0][0]["content"]
    assert captured_messages[0][1]["content"] == ""


def test_generate_module_b_prompts_should_reject_json_template_format(tmp_path: Path) -> None:
    """
    功能说明：验证模块B不再支持JSON模板格式。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：即使JSON内容合法，也应按格式不支持处理。
    """
    template_path = tmp_path / "module_b_prompt.v1.json"
    template_path.write_text(
        (
            "{"
            "\"version\":1,"
            "\"system_prompt\":\"系统提示\","
            "\"user_prompt_template\":\"{{input_payload_json}}\","
            "\"retry_hint_template\":\"补救要求：{{retry_hint}}\""
            "}"
        ),
        encoding="utf-8",
    )
    llm_config = ModuleBLlmConfig(
        prompt_template_file=str(template_path),
        json_retry_times=0,
    )
    with pytest.raises(ModuleBLlmGenerationError, match="格式不支持"):
        generate_module_b_prompts(
            logger=logging.getLogger("test_generate_module_b_prompts_json_not_supported"),
            llm_config=llm_config,
            llm_input_payload={"segment_id": "seg_0001"},
            project_root=tmp_path,
        )


def test_generate_module_b_prompts_should_fail_when_template_placeholder_missing(
    tmp_path: Path,
) -> None:
    """
    功能说明：验证模板缺失占位符时，模块B会直接失败。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：模板字段齐全但占位符不合法也应失败。
    """
    template_path = tmp_path / "bad_prompt.v1.md"
    template_path.write_text(
        (
            "# Module B Prompt Template v1\n\n"
            "## system_prompt\n"
            "系统提示，不含占位符。\n\n"
            "## user_prompt_template\n"
            "{{user_custom_prompt}}\n\n"
            "## retry_hint_template\n"
            "补救要求：{{retry_hint}}\n"
        ),
        encoding="utf-8",
    )
    llm_config = ModuleBLlmConfig(
        prompt_template_file=str(template_path),
        json_retry_times=0,
    )
    with pytest.raises(ModuleBLlmGenerationError, match="缺失占位符"):
        generate_module_b_prompts(
            logger=logging.getLogger("test_generate_module_b_prompts_bad_template"),
            llm_config=llm_config,
            llm_input_payload={"segment_id": "seg_0001"},
            project_root=tmp_path,
        )


def test_generate_module_b_prompts_should_append_retry_hint_from_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能说明：验证解析重试时会按模板追加 retry_hint 文本。
    参数说明：
    - tmp_path: pytest 提供的临时目录。
    - monkeypatch: pytest 提供的补丁工具。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：首次返回非法JSON字段触发解析失败，第二次返回合法JSON成功。
    """
    template_path = tmp_path / "module_b_prompt_retry.v1.md"
    template_path.write_text(
        (
            "# Module B Prompt Template v1\n\n"
            "## system_prompt\n"
            "系统提示：{{input_payload_json}}\n\n"
            "## user_prompt_template\n"
            "{{user_custom_prompt}}\n\n"
            "## retry_hint_template\n"
            "补救要求：{{retry_hint}}\n"
        ),
        encoding="utf-8",
    )
    llm_config = ModuleBLlmConfig(
        prompt_template_file=str(template_path),
        json_retry_times=1,
        user_custom_prompt="请强调镜头连贯",
    )
    captured_messages: list[list[dict[str, str]]] = []
    call_count = {"value": 0}

    def _fake_call_module_b_llm_chat(logger, llm_config, messages, project_root):
        _ = (logger, llm_config, project_root)
        captured_messages.append(messages)
        call_count["value"] += 1
        if call_count["value"] == 1:
            return (
                "{\"scene_desc\":\"中文描述\","
                "\"keyframe_prompt_zh\":\"中文关键帧\","
                "\"keyframe_prompt_en\":\"english keyframe\","
                "\"video_prompt_zh\":\"中文视频\"}"
            )
        return (
            "{\"scene_desc\":\"中文描述\","
            "\"keyframe_prompt_zh\":\"中文关键帧\","
            "\"keyframe_prompt_en\":\"english keyframe\","
            "\"video_prompt_zh\":\"中文视频\","
            "\"video_prompt_en\":\"english video\"}"
        )

    monkeypatch.setattr(llm_generator_module, "call_module_b_llm_chat", _fake_call_module_b_llm_chat)
    result = generate_module_b_prompts(
        logger=logging.getLogger("test_generate_module_b_prompts_retry_hint"),
        llm_config=llm_config,
        llm_input_payload={"segment_id": "seg_0001"},
        project_root=tmp_path,
    )

    assert result["video_prompt_en"] == "english video"
    assert len(captured_messages) == 2
    assert captured_messages[0][1]["content"] == "请强调镜头连贯"
    assert "补救要求：上次输出不符合要求：" in captured_messages[1][1]["content"]
