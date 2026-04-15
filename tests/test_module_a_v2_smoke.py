"""
文件用途：验证模块A V2在路由开启时可完成基础执行与产物落盘。
核心流程：打桩感知层与算法层，执行 run_module_a 并检查输出文件。
输入输出：输入临时上下文，输出断言结果。
依赖说明：依赖 pytest 与模块A V2入口实现。
维护说明：本测试聚焦“可跑通”，不覆盖算法细节正确性。
"""

# 标准库：用于日志对象
import logging
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：配置对象
from music_video_pipeline.config import AppConfig, FfmpegConfig, LoggingConfig, MockConfig, ModeConfig, ModuleAConfig, PathsConfig
# 项目内模块：运行上下文
from music_video_pipeline.context import RuntimeContext
# 项目内模块：JSON读取
from music_video_pipeline.io_utils import read_json
# 项目内模块：模块A V2入口
from music_video_pipeline.modules.module_a_v2 import run_module_a_v2
# 项目内模块：V2算法层产物类型
from music_video_pipeline.modules.module_a_v2.algorithm import AlgorithmBundle
# 项目内模块：V2产物路径构建
from music_video_pipeline.modules.module_a_v2.artifacts import build_module_a_v2_artifacts
# 项目内模块：V2感知层产物类型
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def _build_context(tmp_path: Path) -> RuntimeContext:
    """
    功能说明：构建模块A V2测试上下文。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - RuntimeContext: 运行上下文对象。
    异常说明：无。
    边界条件：输入音频使用占位文件，不依赖真实解码。
    """
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")
    app_config = AppConfig(
        mode=ModeConfig(script_generator="mock", frame_generator="mock"),
        paths=PathsConfig(runs_dir=str(tmp_path / "runs"), default_audio_path="demo.wav"),
        ffmpeg=FfmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            video_codec="libx264",
            audio_codec="aac",
            fps=24,
            video_preset="veryfast",
            video_crf=30,
        ),
        logging=LoggingConfig(level="INFO"),
        mock=MockConfig(beat_interval_seconds=0.5, video_width=640, video_height=360),
        module_a=ModuleAConfig(funasr_language="auto", implementation="v2"),
    )
    artifacts_dir = tmp_path / "runs" / "task_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(
        task_id="task_v2",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "task_v2",
        artifacts_dir=artifacts_dir,
        config=app_config,
        logger=logging.getLogger("test_module_a_v2_smoke"),
        state_store=StateStore(db_path=tmp_path / "state.sqlite3"),
    )


def _build_fake_perception_bundle(artifacts_dir: Path) -> PerceptionBundle:
    """
    功能说明：构造模块A V2测试用感知层打桩结果。
    参数说明：
    - artifacts_dir: 任务 artifacts 目录。
    返回值：
    - PerceptionBundle: 可用于编排层测试的感知结果。
    异常说明：无。
    边界条件：仅用于测试，不代表真实模型输出分布。
    """
    artifacts = build_module_a_v2_artifacts(artifacts_dir / "module_a_work_v2")
    runtime_vocals_path = artifacts.perception_model_demucs_runtime_dir / "vocals.wav"
    runtime_bass_path = artifacts.perception_model_demucs_runtime_dir / "bass.wav"
    runtime_drums_path = artifacts.perception_model_demucs_runtime_dir / "drums.wav"
    runtime_other_path = artifacts.perception_model_demucs_runtime_dir / "other.wav"
    runtime_no_vocals_path = artifacts.perception_model_demucs_runtime_dir / "no_vocals.wav"
    return PerceptionBundle(
        big_segments_stage1=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "verse"}],
        beat_candidates=[0.0, 1.0, 2.0, 3.0, 12.0],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "minor", "source": "allin1"},
        ],
        lyric_sentence_units=[],
        sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35, "skipped": True},
        vocals_path=runtime_vocals_path,
        no_vocals_path=runtime_no_vocals_path,
        demucs_stems={
            "vocals": runtime_vocals_path,
            "bass": runtime_bass_path,
            "drums": runtime_drums_path,
            "other": runtime_other_path,
            "no_vocals": runtime_no_vocals_path,
        },
        onset_candidates=[0.0, 1.0, 2.0, 12.0],
        rms_times=[0.0, 1.0, 2.0, 12.0],
        rms_values=[0.1, 0.2, 0.3, 0.2],
        vocal_onset_candidates=[0.0, 1.0, 2.0, 12.0],
        vocal_rms_times=[0.0, 1.0, 2.0, 12.0],
        vocal_rms_values=[0.1, 0.2, 0.3, 0.2],
        funasr_skipped_for_silent_vocals=True,
    )


def _build_fake_algorithm_bundle() -> AlgorithmBundle:
    """
    功能说明：构造模块A V2测试用算法层打桩结果。
    参数说明：无。
    返回值：
    - AlgorithmBundle: 可用于编排层测试的算法结果。
    异常说明：无。
    边界条件：字段完整覆盖 ModuleAOutput 最低契约。
    """
    return AlgorithmBundle(
        big_segments_stage1=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "verse"}],
        big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "verse"}],
        segments=[{"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 12.0, "label": "verse"}],
        beats=[
            {"time": 0.0, "type": "major", "source": "allin1"},
            {"time": 12.0, "type": "minor", "source": "allin1"},
        ],
        lyric_units=[],
        energy_features=[{"start_time": 0.0, "end_time": 12.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.5}],
    )


def _patch_module_a_v2_core_stages(monkeypatch, context: RuntimeContext) -> None:
    """
    功能说明：统一打桩模块A V2核心阶段（时长探测/感知层/算法层）。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - context: 测试上下文。
    返回值：无。
    异常说明：无。
    边界条件：用于隔离真实模型依赖，确保测试可复现。
    """

    def _fake_probe_audio_duration(*_args, **_kwargs) -> float:
        return 12.0

    def _fake_perception_stage(*_args, **_kwargs) -> PerceptionBundle:
        return _build_fake_perception_bundle(artifacts_dir=context.artifacts_dir)

    def _fake_algorithm_stage(*_args, **_kwargs) -> AlgorithmBundle:
        return _build_fake_algorithm_bundle()

    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator.probe_audio_duration", _fake_probe_audio_duration)
    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator._run_perception_stage", _fake_perception_stage)
    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator._run_algorithm_stage", _fake_algorithm_stage)


def test_module_a_v2_should_run_and_write_output(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 implementation=v2 时可跑通并写出 module_a_output.json。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：感知层与算法层使用测试打桩。
    """
    context = _build_context(tmp_path=tmp_path)
    _patch_module_a_v2_core_stages(monkeypatch=monkeypatch, context=context)

    output_path = run_module_a_v2(context)
    output_data = read_json(output_path)
    assert output_path.exists()
    assert output_data["task_id"] == "task_v2"
    assert len(output_data["big_segments"]) == 1
    assert len(output_data["segments"]) == 1
    assert len(output_data["beats"]) == 2


def test_module_a_v2_should_auto_refresh_visualization_after_run(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证模块A V2每次执行后都会自动触发可视化重绘。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：可视化函数通过打桩隔离真实HTML模板细节。
    """
    context = _build_context(tmp_path=tmp_path)
    _patch_module_a_v2_core_stages(monkeypatch=monkeypatch, context=context)
    call_trace: dict[str, Path | str] = {}

    def _fake_collect_visualization_payload(task_dir: Path) -> dict:
        call_trace["task_dir"] = task_dir
        return {"task_id": context.task_id, "audio_path": str(context.audio_path)}

    def _fake_render_visualization_html(payload: dict, output_html_path: Path, audio_mode: str = "copy") -> Path:
        call_trace["output_html_path"] = output_html_path
        call_trace["audio_mode"] = audio_mode
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        output_html_path.write_text("<html><body>auto-viz</body></html>", encoding="utf-8")
        assert payload["task_id"] == context.task_id
        return output_html_path

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.orchestrator.collect_visualization_payload",
        _fake_collect_visualization_payload,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.orchestrator.render_visualization_html",
        _fake_render_visualization_html,
    )

    output_path = run_module_a_v2(context)
    expected_html_path = context.task_dir / f"{context.task_id}_module_a_v2_visualization.html"
    assert output_path.exists()
    assert call_trace["task_dir"] == context.task_dir
    assert call_trace["output_html_path"] == expected_html_path
    assert call_trace["audio_mode"] == "copy"
    assert expected_html_path.exists()


def test_module_a_v2_should_continue_when_auto_visualization_failed(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证自动可视化失败时不影响模块A主流程产物落盘。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：可视化故障应被编排层捕获并降级为告警日志。
    """
    context = _build_context(tmp_path=tmp_path)
    _patch_module_a_v2_core_stages(monkeypatch=monkeypatch, context=context)

    def _raise_visualization_error(*_args, **_kwargs):
        raise RuntimeError("模拟可视化失败")

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a_v2.orchestrator.collect_visualization_payload",
        _raise_visualization_error,
    )

    output_path = run_module_a_v2(context)
    output_data = read_json(output_path)
    assert output_path.exists()
    assert output_data["task_id"] == "task_v2"
