"""
文件用途：验证模块A V2可视化负载聚合与HTML渲染能力。
核心流程：构造最小任务产物目录，调用可视化函数与脚本并断言输出。
输入输出：输入临时任务目录，输出断言结果。
依赖说明：依赖 pytest 与模块A V2可视化实现。
维护说明：若可视化数据源路径变更，需同步更新测试产物构造逻辑。
"""

# 标准库：用于子进程执行脚本
import subprocess
# 标准库：用于系统解释器路径
import sys
# 标准库：用于环境变量拷贝
import os
# 标准库：用于路径处理
from pathlib import Path

# 项目内模块：JSON读写
from music_video_pipeline.io_utils import read_json, write_json
# 项目内模块：模块A V2可视化函数
from music_video_pipeline.modules.module_a_v2.visualization import collect_visualization_payload, render_visualization_html


def _build_minimal_task_dir(tmp_path: Path, with_optional_files: bool = True) -> Path:
    """
    功能说明：构造模块A V2可视化测试所需最小任务目录。
    参数说明：
    - tmp_path: pytest 临时目录。
    - with_optional_files: 是否写入可选产物文件。
    返回值：
    - Path: 任务目录路径。
    异常说明：无。
    边界条件：任务目录结构固定为 runs/<task_id>/artifacts/module_a_work_v2。
    """
    task_dir = tmp_path / "runs" / "viz_task"
    artifacts_dir = task_dir / "artifacts"
    work_dir = artifacts_dir / "module_a_work_v2"
    algorithm_dir = work_dir / "algorithm"
    window_dir = algorithm_dir / "window"
    timeline_dir = algorithm_dir / "timeline"
    final_dir = algorithm_dir / "final"
    librosa_dir = work_dir / "perception" / "signal" / "librosa"
    funasr_dir = work_dir / "perception" / "model" / "funasr"
    window_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    librosa_dir.mkdir(parents=True, exist_ok=True)
    funasr_dir.mkdir(parents=True, exist_ok=True)

    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio")

    write_json(
        artifacts_dir / "module_a_output.json",
        {
            "task_id": "viz_task",
            "audio_path": str(audio_path),
            "big_segments": [
                {"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"},
                {"segment_id": "big_002", "start_time": 4.0, "end_time": 8.0, "label": "chorus"},
            ],
            "segments": [
                {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
                {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 2.0, "end_time": 4.0, "label": "verse"},
                {"segment_id": "seg_0003", "big_segment_id": "big_002", "start_time": 4.0, "end_time": 8.0, "label": "chorus"},
            ],
            "beats": [
                {"time": 0.0, "type": "major", "source": "allin1"},
                {"time": 2.0, "type": "minor", "source": "allin1"},
                {"time": 4.0, "type": "major", "source": "allin1"},
                {"time": 8.0, "type": "minor", "source": "allin1"},
            ],
            "lyric_units": [
                {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.6, "text": "第一句", "confidence": 0.9}
            ],
            "energy_features": [
                {"start_time": 0.0, "end_time": 2.0, "energy_level": "low", "trend": "up", "rhythm_tension": 0.2},
                {"start_time": 2.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.4},
            ],
            "alias_map": {"version": "module_a_alias_v1"},
        },
    )

    write_json(
        timeline_dir / "stage_big_a0.json",
        [
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 4.2, "label": "verse"},
            {"segment_id": "big_002", "start_time": 4.2, "end_time": 8.0, "label": "chorus"},
        ],
    )
    write_json(
        timeline_dir / "stage_big_a1.json",
        [
            {"segment_id": "big_001", "start_time": 0.0, "end_time": 4.0, "label": "verse"},
            {"segment_id": "big_002", "start_time": 4.0, "end_time": 8.0, "label": "chorus"},
        ],
    )
    write_json(
        final_dir / "stage_segments_final.json",
        [
            {"segment_id": "seg_0001", "big_segment_id": "big_001", "start_time": 0.0, "end_time": 2.0, "label": "verse"},
            {"segment_id": "seg_0002", "big_segment_id": "big_001", "start_time": 2.0, "end_time": 4.0, "label": "verse"},
            {"segment_id": "seg_0003", "big_segment_id": "big_002", "start_time": 4.0, "end_time": 8.0, "label": "chorus"},
        ],
    )
    write_json(
        final_dir / "stage_energy.json",
        [
            {"start_time": 0.0, "end_time": 2.0, "energy_level": "low", "trend": "up", "rhythm_tension": 0.2},
            {"start_time": 2.0, "end_time": 4.0, "energy_level": "mid", "trend": "flat", "rhythm_tension": 0.4},
        ],
    )
    write_json(
        window_dir / "stage_windows_classified.json",
        [
            {"window_id": "win_0001", "start_time": 0.0, "end_time": 2.0, "window_type": "lyric_sentence", "role": "lyric"},
            {"window_id": "win_0002", "start_time": 2.0, "end_time": 4.0, "window_type": "other_between", "role": "chant"},
            {"window_id": "win_0003", "start_time": 4.0, "end_time": 8.0, "window_type": "other_trailing", "role": "inst"},
        ],
    )
    write_json(
        window_dir / "stage_windows_merged.json",
        [
            {"window_id": "win_0001", "start_time": 0.0, "end_time": 2.0, "window_type": "lyric_sentence", "role": "lyric"},
            {"window_id": "win_0002", "start_time": 2.0, "end_time": 8.0, "window_type": "other_merged", "role": "chant"},
        ],
    )

    if with_optional_files:
        write_json(
            funasr_dir / "lyric_sentence_units.json",
            [
                {"start_time": 0.1, "end_time": 1.1, "text": "第一句(全量)", "confidence": 0.9},
                {"start_time": 1.3, "end_time": 2.2, "text": "第二句(全量)", "confidence": 0.88},
            ],
        )
        write_json(
            timeline_dir / "stage_lyric_sentence_units_cleaned.json",
            [
                {"start_time": 0.05, "end_time": 1.1, "text": "第一句(清洗后)", "confidence": 0.9},
                {"start_time": 1.25, "end_time": 2.2, "text": "第二句(清洗后)", "confidence": 0.88},
            ],
        )
        write_json(
            final_dir / "stage_lyric_attached.json",
            [
                {"segment_id": "seg_0001", "start_time": 0.2, "end_time": 1.6, "text": "第一句", "confidence": 0.9}
            ],
        )
        write_json(
            librosa_dir / "accompaniment_candidates.json",
            {
                "onset_candidates": [0.0, 1.5, 3.1, 4.8],
                "onset_points": [
                    {"time": 0.0, "energy_raw": 0.0},
                    {"time": 1.5, "energy_raw": 0.2},
                    {"time": 3.1, "energy_raw": 0.9},
                    {"time": 4.8, "energy_raw": 0.4},
                ],
                "rms_times": [0.0, 1.0],
                "rms_values": [0.2, 0.3],
            },
        )
        write_json(
            librosa_dir / "vocal_precheck_rms.json",
            {
                "rms_times": [0.0, 0.5, 1.0],
                "rms_values": [0.01, 0.02, 0.015],
                "should_skip_funasr": False,
                "peak_rms": 0.02,
                "active_ratio": 0.66,
                "peak_threshold": 0.01,
                "active_ratio_threshold": 0.02,
            },
        )
        write_json(funasr_dir / "funasr_raw_response.json", {"skipped": False, "result": []})
        write_json(
            funasr_dir / "sentence_split_stats.json",
            {
                "dynamic_gap_threshold_seconds": 0.42,
                "sample_source": "punctuation_neighbor",
                "sample_count_raw": 3,
                "sample_count_kept": 2,
                "sample_count_outlier": 1,
                "outlier_samples": [1.12],
            },
        )

    return task_dir


def test_collect_visualization_payload_should_include_required_layers(tmp_path: Path) -> None:
    """
    功能说明：验证聚合函数可读取必需产物并组装核心图层数据。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：本用例包含可选文件。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=True)
    payload = collect_visualization_payload(task_dir=task_dir)
    assert payload["task_id"] == "viz_task"
    assert len(payload["a0_segments"]) == 2
    assert len(payload["al_segments"]) == 2
    assert len(payload["b_segments"]) == 2
    assert len(payload["s_segments"]) == 3
    assert len(payload["content_roles"]) == 3
    assert payload["content_roles"][0]["display_text"] == "lyric"
    assert payload["s_segments"][0]["display_text"] == "verse"
    assert len(payload["beats"]) == 4
    assert len(payload["lyric_units"]) == 2
    assert payload["summary"]["lyric_count"] == 2
    assert payload["summary"]["lyric_attached_count"] == 1
    assert len(payload["energy_features"]) == 2
    assert len(payload["onset_points"]) == 4
    assert payload["onset_points"][2]["energy_raw"] == 0.9
    assert payload["onset_points"][2]["energy_norm"] >= payload["onset_points"][1]["energy_norm"]
    assert len(payload["accompaniment_rms"]["times"]) == 2
    assert payload["vocal_precheck_rms"]["sample_source"] == "punctuation_neighbor"
    assert payload["vocal_precheck_rms"]["sample_count_raw"] == 3
    assert payload["vocal_precheck_rms"]["sample_count_outlier"] == 1
    assert payload["summary"]["boundary_shift"]["adjusted_count"] >= 1
    assert payload["duration_seconds"] >= 8.0


def test_collect_visualization_payload_should_allow_missing_optional_files(tmp_path: Path) -> None:
    """
    功能说明：验证可选文件缺失时仍可聚合并生成HTML。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：不写入 optional 文件，走降级路径。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=False)
    payload = collect_visualization_payload(task_dir=task_dir)
    assert payload["onset_candidates"] == []
    assert payload["onset_points"] == []
    assert payload["vocal_precheck_rms"]["times"] == []
    assert payload["accompaniment_rms"]["times"] == []

    output_path = task_dir / "module_a_v2_visualization.html"
    render_visualization_html(payload=payload, output_html_path=output_path, audio_mode="none")
    assert output_path.exists()
    html_text = output_path.read_text(encoding="utf-8")
    assert "模块A V2 可视化" in html_text
    assert "const PAYLOAD =" in html_text
    assert "A0段（stage_big_a0）" in html_text
    assert "A1段（timeline/stage_big_a1）" in html_text
    assert "Beats（module_a_output.beats）" in html_text
    assert "Lyrics（全量分句：timeline_cleaned/funasr）" in html_text
    assert "Lyrics（挂载结果：final/stage_lyric_attached）" in html_text
    assert "伴奏RMS（no_vocals）" in html_text
    assert html_text.index('key: "precheck"') < html_text.index('key: "accompaniment_rms"')


def test_render_visualization_html_should_include_split_stats_anchor(tmp_path: Path) -> None:
    """
    功能说明：验证人声RMS预检轨包含分句统计信息渲染锚点。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证关键脚本锚点存在，不校验像素级绘制结果。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=True)
    payload = collect_visualization_payload(task_dir=task_dir)
    output_path = task_dir / "module_a_v2_visualization.html"
    render_visualization_html(payload=payload, output_html_path=output_path, audio_mode="none")
    html_text = output_path.read_text(encoding="utf-8")
    assert "sample_count_raw" in html_text
    assert "dynamic_gap_threshold" in html_text
    assert "silence_floor_rms" not in html_text


def test_collect_visualization_payload_should_raise_when_required_file_missing(tmp_path: Path) -> None:
    """
    功能说明：验证必需文件缺失时抛出明确错误信息。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：通过断言异常文本完成校验。
    边界条件：删除 stage_big_a1.json 触发报错。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=True)
    required_file = task_dir / "artifacts" / "module_a_work_v2" / "algorithm" / "timeline" / "stage_big_a1.json"
    required_file.unlink()

    try:
        collect_visualization_payload(task_dir=task_dir)
        raise AssertionError("预期应抛出 RuntimeError，但函数未抛错")
    except RuntimeError as error:
        error_text = str(error)
        assert "缺少必需产物文件" in error_text
        assert "stage_big_a1.json" in error_text


def test_module_a_v2_visualization_script_smoke_should_generate_html(tmp_path: Path) -> None:
    """
    功能说明：验证独立脚本可基于 task_dir 生成可视化HTML文件。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：脚本失败时由 subprocess 返回非零码并导致断言失败。
    边界条件：脚本通过 --task-dir 运行，不依赖真实 runs 状态库。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=True)
    output_html_path = task_dir / "custom_visualization.html"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "_module_a_v2_visualize.py"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    current_python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{current_python_path}" if current_python_path else str(repo_root / "src")

    command = [
        sys.executable,
        str(script_path),
        "--task-dir",
        str(task_dir),
        "--output",
        str(output_html_path),
        "--audio-mode",
        "copy",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    assert "可视化页面生成完成" in completed.stdout
    assert output_html_path.exists()
    html_text = output_html_path.read_text(encoding="utf-8")
    assert "custom_visualization_audio" in html_text
    payload_loaded = read_json(task_dir / "artifacts" / "module_a_output.json")
    assert payload_loaded["task_id"] == "viz_task"


def test_render_visualization_html_should_include_onset_energy_tooltip_fields(tmp_path: Path) -> None:
    """
    功能说明：验证可视化HTML包含onset能量渲染与tooltip字段。
    参数说明：
    - tmp_path: pytest 临时目录。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：仅验证模板中关键脚本文本存在。
    """
    task_dir = _build_minimal_task_dir(tmp_path=tmp_path, with_optional_files=True)
    payload = collect_visualization_payload(task_dir=task_dir)
    output_path = task_dir / "onset_energy_visualization.html"
    render_visualization_html(payload=payload, output_html_path=output_path, audio_mode="none")
    html_text = output_path.read_text(encoding="utf-8")
    assert "energy_raw" in html_text
    assert "energy_norm" in html_text
    assert "0.15 + 0.80 * energyNorm" in html_text
