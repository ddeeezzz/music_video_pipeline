"""
文件用途：验证模块A V2输出契约在空歌词场景下仍合法。
核心流程：打桩执行 run_module_a_v2，并调用 validate_module_a_output 校验。
输入输出：输入临时上下文，输出断言结果。
依赖说明：依赖 pytest 与模块A V2编排实现。
维护说明：若 ModuleAOutput 契约变更，需同步更新本测试样本。
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
# 项目内模块：V2编排入口
from music_video_pipeline.modules.module_a_v2.orchestrator import run_module_a_v2
# 项目内模块：类型契约校验
from music_video_pipeline.types import validate_module_a_output
# 项目内模块：V2算法层产物类型
from music_video_pipeline.modules.module_a_v2.algorithm import AlgorithmBundle
# 项目内模块：V2感知层产物类型
from music_video_pipeline.modules.module_a_v2.perception import PerceptionBundle
# 项目内模块：状态存储
from music_video_pipeline.state_store import StateStore


def _build_context(tmp_path: Path) -> RuntimeContext:
    """
    功能说明：构建模块A V2契约测试上下文。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：
    - RuntimeContext: 运行上下文对象。
    异常说明：无。
    边界条件：输入音频使用占位文件。
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
    artifacts_dir = tmp_path / "runs" / "task_contract_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(
        task_id="task_contract_v2",
        audio_path=audio_path,
        task_dir=tmp_path / "runs" / "task_contract_v2",
        artifacts_dir=artifacts_dir,
        config=app_config,
        logger=logging.getLogger("test_module_a_v2_contracts"),
        state_store=StateStore(db_path=tmp_path / "state.sqlite3"),
    )


def test_module_a_v2_should_allow_empty_lyric_units(monkeypatch, tmp_path: Path) -> None:
    """
    功能说明：验证 FunASR 空输出时，V2仍可输出合法契约。
    参数说明：
    - monkeypatch: pytest monkeypatch 工具。
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过打桩保证不依赖真实模型环境。
    """
    context = _build_context(tmp_path=tmp_path)

    def _fake_probe_audio_duration(*_args, **_kwargs) -> float:
        return 8.0

    def _fake_perception_stage(*_args, **_kwargs) -> PerceptionBundle:
        return PerceptionBundle(
            big_segments_stage1=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "inst"}],
            beat_candidates=[0.0, 2.0, 4.0, 8.0],
            beats=[
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 8.0, "type": "minor", "source": "allin1"},
            ],
            lyric_sentence_units=[],
            sentence_split_stats={"dynamic_gap_threshold_seconds": 0.35, "skipped": True},
            vocals_path=tmp_path / "vocals.wav",
            no_vocals_path=tmp_path / "no_vocals.wav",
            demucs_stems={},
            onset_candidates=[0.0, 2.0, 4.0, 8.0],
            rms_times=[0.0, 2.0, 4.0, 8.0],
            rms_values=[0.1, 0.1, 0.1, 0.1],
            vocal_onset_candidates=[0.0, 2.0, 4.0, 8.0],
            vocal_rms_times=[0.0, 2.0, 4.0, 8.0],
            vocal_rms_values=[0.1, 0.1, 0.1, 0.1],
            funasr_skipped_for_silent_vocals=True,
        )

    def _fake_algorithm_stage(*_args, **_kwargs) -> AlgorithmBundle:
        return AlgorithmBundle(
            big_segments_stage1=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "inst"}],
            big_segments=[{"segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "inst"}],
            segments=[{"segment_id": "seg_001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 8.0, "label": "inst"}],
            beats=[
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 8.0, "type": "minor", "source": "allin1"},
            ],
            lyric_units=[],
            energy_features=[{"start_time": 0.0, "end_time": 8.0, "energy_level": "low", "trend": "flat", "rhythm_tension": 0.1}],
        )

    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator.probe_audio_duration", _fake_probe_audio_duration)
    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator._run_perception_stage", _fake_perception_stage)
    monkeypatch.setattr("music_video_pipeline.modules.module_a_v2.orchestrator._run_algorithm_stage", _fake_algorithm_stage)

    output_path = run_module_a_v2(context)
    output_data = read_json(output_path)
    validate_module_a_output(output_data)
    assert output_data["lyric_units"] == []
