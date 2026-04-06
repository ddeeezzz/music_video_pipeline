"""
文件用途：验证模块A在调用 Allin1 时可保存原始响应 JSON。
核心流程：打桩 Allin1 后端返回值，调用检测函数并断言 JSON 落盘。
输入输出：输入临时目录与后端桩对象，输出断言结果。
依赖说明：依赖 pytest 与模块A backends 实现。
维护说明：若 Allin1 接口字段调整，应同步更新本测试桩数据。
"""

# 标准库：日志构建
import logging
# 标准库：路径处理
from pathlib import Path
# 标准库：命令结果对象
import subprocess

# 第三方库：测试框架
import pytest

# 项目内模块：被测函数
from music_video_pipeline.modules.module_a.backends import _analyze_with_allin1
from music_video_pipeline.modules.module_a.backends import _detect_big_segments_with_allin1
from music_video_pipeline.modules.module_a.backends import _prepare_stems_with_allin1_demucs
from music_video_pipeline.modules.module_a.backends import _separate_with_demucs


def test_detect_big_segments_with_allin1_should_save_raw_response_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能说明：验证传入 raw_response_path 时会保存 Allin1 原始响应 JSON。
    参数说明：
    - monkeypatch: pytest 提供的运行时打桩工具。
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：后端返回对象为字典结构，需同时覆盖段落解析与原始落盘两条路径。
    """

    class _FakeAllin1Backend:
        """
        功能说明：用于模拟 allin1 后端最小行为的测试桩。
        参数说明：无。
        返回值：不适用。
        异常说明：不适用。
        边界条件：仅实现 analyze 接口，覆盖当前主路径。
        """

        @staticmethod
        def analyze(_: str) -> dict:
            """
            功能说明：返回用于测试的固定 allin1 原始结果。
            参数说明：
            - _: 输入音频路径（测试中不使用）。
            返回值：
            - dict: 包含 segments 与附加元数据的模拟响应。
            异常说明：无。
            边界条件：返回字段满足解析逻辑最小输入要求。
            """
            return {
                "segments": [
                    {"start_time": 0.0, "end_time": 1.2, "label": "intro"},
                    {"start_time": 1.2, "end_time": 2.5, "label": "verse"},
                ],
                "meta": {"source": "fake_allin1"},
            }

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._import_allin1_backend",
        lambda: ("allin1", _FakeAllin1Backend),
    )

    raw_output_path = tmp_path / "allin1_raw_response.json"
    logger = logging.getLogger("test_module_a_allin1_raw_dump")
    logger.setLevel(logging.INFO)

    normalized_segments = _detect_big_segments_with_allin1(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=2.5,
        logger=logger,
        raw_response_path=raw_output_path,
    )

    assert raw_output_path.exists()
    saved_text = raw_output_path.read_text(encoding="utf-8")
    assert "\"segments\"" in saved_text
    assert "\"label\": \"intro\"" in saved_text
    assert "\"source\": \"fake_allin1\"" in saved_text

    assert normalized_segments
    assert normalized_segments[0]["label"] == "intro"


def test_analyze_with_allin1_should_parse_beats_and_positions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能说明：验证 allin1 分析可提取 beats/beat_positions 并映射为模块A beats 结构。
    参数说明：
    - monkeypatch: pytest 提供的运行时打桩工具。
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅允许裁剪/去重，不应注入额外节拍点。
    """

    class _FakeAllin1Backend:
        """
        功能说明：用于模拟 allin1 后端最小行为的测试桩。
        参数说明：无。
        返回值：不适用。
        异常说明：不适用。
        边界条件：仅覆盖 analyze 主路径。
        """

        @staticmethod
        def analyze(_: str) -> dict:
            """
            功能说明：返回用于测试的固定 allin1 原始结果。
            参数说明：
            - _: 输入音频路径（测试中不使用）。
            返回值：
            - dict: 包含 segments/beats/beat_positions 的模拟响应。
            异常说明：无。
            边界条件：beats 中包含重复与越界值用于验证规范化行为。
            """
            return {
                "segments": [
                    {"start_time": 0.0, "end_time": 1.2, "label": "intro"},
                    {"start_time": 1.2, "end_time": 2.5, "label": "verse"},
                ],
                "beats": [0.2, 0.8, 0.8, 1.6, 3.1],
                "beat_positions": [1, 2, 3, 4, 1],
            }

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._import_allin1_backend",
        lambda: ("allin1", _FakeAllin1Backend),
    )

    raw_output_path = tmp_path / "allin1_raw_response.json"
    logger = logging.getLogger("test_module_a_allin1_beats_parse")
    logger.setLevel(logging.INFO)

    analysis = _analyze_with_allin1(
        audio_path=tmp_path / "demo.wav",
        duration_seconds=2.5,
        logger=logger,
        raw_response_path=raw_output_path,
    )

    assert raw_output_path.exists()
    assert analysis["big_segments"]
    assert analysis["beat_times"] == [0.2, 0.8, 1.6, 2.5]
    assert [item["source"] for item in analysis["beats"]] == ["allin1", "allin1", "allin1", "allin1"]
    assert [item["type"] for item in analysis["beats"]] == ["major", "minor", "minor", "major"]


def test_prepare_stems_with_allin1_demucs_should_return_no_vocals_and_keep_other(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能说明：验证 allin1-demucs 复用路径会返回 no_vocals，并保留 other 供 allin1 四轨分析。
    参数说明：
    - monkeypatch: pytest 提供的运行时打桩工具。
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过桩函数隔离外部模型与 ffmpeg 依赖。
    """
    created_no_vocals_paths: list[Path] = []
    resample_calls: list[tuple[Path, int]] = []

    class _FakeAllin1FixBackend:
        """
        功能说明：模拟 allin1fix 后端最小接口（DemucsProvider/get_stems）。
        参数说明：无。
        返回值：不适用。
        异常说明：不适用。
        边界条件：仅覆盖当前函数调用所需最小行为。
        """

        class DemucsProvider:
            """
            功能说明：兼容 DemucsProvider 构造签名的测试桩。
            参数说明：无。
            返回值：不适用。
            异常说明：不适用。
            边界条件：构造参数仅缓存，不参与推理。
            """

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        @staticmethod
        def get_stems(audio_paths, output_dir, _provider_obj, _runtime_device):
            stem_dir = Path(output_dir) / "htdemucs" / Path(audio_paths[0]).stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            for stem_name in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
                (stem_dir / stem_name).write_bytes(b"fake")
            return [stem_dir]

    def _fake_mix_non_vocal_stems_to_no_vocals(*, output_path: Path, **_kwargs):
        output_path.write_bytes(b"mixed")
        created_no_vocals_paths.append(output_path)

    def _fake_resample_audio_file_inplace(*, audio_path: Path, target_sample_rate: int, **_kwargs):
        resample_calls.append((audio_path, int(target_sample_rate)))

    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._import_allin1_backend",
        lambda: ("allin1fix", _FakeAllin1FixBackend),
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._mix_non_vocal_stems_to_no_vocals",
        _fake_mix_non_vocal_stems_to_no_vocals,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._probe_audio_sample_rate",
        lambda *args, **kwargs: 48000,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._resample_audio_file_inplace",
        _fake_resample_audio_file_inplace,
    )

    logger = logging.getLogger("test_prepare_stems_allin1_demucs")
    logger.setLevel(logging.INFO)
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")

    vocals_path, no_vocals_path, stems_input = _prepare_stems_with_allin1_demucs(
        audio_path=audio_path,
        output_dir=tmp_path / "allin1_demucs",
        device="auto",
        model_name="htdemucs",
        logger=logger,
    )

    assert vocals_path.name == "vocals.wav"
    assert no_vocals_path.name == "no_vocals.wav"
    assert no_vocals_path in created_no_vocals_paths
    assert stems_input["other"].name == "other.wav"
    assert stems_input["bass"].name == "bass.wav"
    assert stems_input["drums"].name == "drums.wav"
    assert (vocals_path, 48000) in resample_calls
    assert (no_vocals_path, 48000) in resample_calls


def test_separate_with_demucs_should_trigger_stem_sample_rate_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能说明：验证独立 Demucs 路径会调用标准二轨采样率统一流程。
    参数说明：
    - monkeypatch: pytest 提供的运行时打桩工具。
    - tmp_path: pytest 提供的临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：通过命令桩避免真实 demucs/ffmpeg 依赖。
    """
    normalize_calls: list[tuple[Path, Path, int | None]] = []

    def _fake_which(command_name: str) -> str | None:
        if command_name == "demucs":
            return "/usr/bin/demucs"
        return None

    def _fake_run(command, **_kwargs):
        if command and str(command[0]) == "/usr/bin/demucs":
            output_dir = Path(command[command.index("-o") + 1])
            audio_file = Path(command[-1])
            stem_dir = output_dir / "htdemucs" / audio_file.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            (stem_dir / "vocals.wav").write_bytes(b"vocals")
            (stem_dir / "no_vocals.wav").write_bytes(b"no_vocals")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def _fake_normalize_stem_pair(*, vocals_path: Path, no_vocals_path: Path, target_sample_rate: int | None, **_kwargs):
        normalize_calls.append((vocals_path, no_vocals_path, target_sample_rate))

    monkeypatch.setattr("music_video_pipeline.modules.module_a.backends.shutil.which", _fake_which)
    monkeypatch.setattr("music_video_pipeline.modules.module_a.backends.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._probe_audio_sample_rate",
        lambda *args, **kwargs: 48000,
    )
    monkeypatch.setattr(
        "music_video_pipeline.modules.module_a.backends._normalize_standard_stems_sample_rate",
        _fake_normalize_stem_pair,
    )

    logger = logging.getLogger("test_separate_with_demucs_normalize")
    logger.setLevel(logging.INFO)
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")

    vocals_path, no_vocals_path = _separate_with_demucs(
        audio_path=audio_path,
        output_dir=tmp_path / "demucs",
        device="auto",
        model_name="htdemucs",
        logger=logger,
    )

    assert vocals_path.name == "vocals.wav"
    assert no_vocals_path.name == "no_vocals.wav"
    assert len(normalize_calls) == 1
    assert normalize_calls[0][0] == vocals_path
    assert normalize_calls[0][1] == no_vocals_path
    assert normalize_calls[0][2] == 48000
