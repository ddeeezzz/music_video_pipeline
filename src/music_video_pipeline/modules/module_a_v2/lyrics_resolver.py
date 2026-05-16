"""
文件用途：统一解析模块A V2 的歌词来源优先级。
核心流程：元数据直查 LRCLIB -> 根据结果选用歌词、终止或回退 FunASR。
输入输出：输入音频、时长、产物路径与 FunASR 回调，输出统一歌词解析结果。
依赖说明：依赖元数据读取、LRCLIB 查询、LRC 解析与产物落盘工具。
维护说明：阶段一只实现元数据直查与 FunASR 兜底，不引入指纹补充链。
"""

# 标准库：用于路径类型
from pathlib import Path
# 标准库：用于类型提示
from typing import Any, Callable

# 项目内模块：产物路径与落盘工具
from music_video_pipeline.modules.module_a_v2.artifacts import ModuleAV2Artifacts, dump_json_artifact
# 项目内模块：AcoustID 指纹与识别补充链
from music_video_pipeline.modules.module_a_v2.acoustid_client import build_fingerprint_result, query_acoustid_match
# 项目内模块：LRC 解析器
from music_video_pipeline.modules.module_a_v2.lrc_parser import parse_synced_lyrics_to_sentence_units
# 项目内模块：LRCLIB 客户端
from music_video_pipeline.modules.module_a_v2.lrclib_client import query_lrclib_lyrics
# 项目内模块：元数据读取器
from music_video_pipeline.modules.module_a_v2.metadata_reader import read_embedded_metadata


def resolve_lyrics_with_priority(
    audio_path: Path,
    duration_seconds: float,
    artifacts: ModuleAV2Artifacts,
    logger,
    fpcalc_bin: str,
    acoustid_api_key_file: str,
    enable_fingerprint_lookup: bool,
    funasr_fallback_runner: Callable[[], tuple[Any, list[dict[str, Any]], dict[str, Any]] | None],
) -> dict[str, Any]:
    """
    功能说明：统一解析模块A V2歌词来源，优先元数据直查，必要时回退 FunASR。
    参数说明：
    - audio_path: 输入音频路径。
    - duration_seconds: 音频总时长（秒）。
    - artifacts: 模块A V2产物路径对象。
    - logger: 日志记录器。
    - fpcalc_bin: 指纹命令路径或命令名。
    - acoustid_api_key_file: AcoustID API Key 文件路径。
    - enable_fingerprint_lookup: 是否启用指纹补充链。
    - funasr_fallback_runner: FunASR 兜底执行回调。
    返回值：
    - dict[str, Any]: 统一歌词解析结果。
    异常说明：异常在函数内尽量转为 fallback，避免中断模块A主链。
    边界条件：当 LRCLIB 明确标记 instrumental 时，不再进入 FunASR 兜底。
    """
    metadata_result = read_embedded_metadata(
        audio_path=audio_path,
        duration_seconds=duration_seconds,
        logger=logger,
    )
    dump_json_artifact(
        output_path=artifacts.perception_model_lyrics_metadata_result_path,
        payload=metadata_result,
        logger=logger,
        artifact_name="metadata_result",
    )
    if metadata_result.get("status") == "ok":
        lrclib_result = query_lrclib_lyrics(
            artist=str(metadata_result.get("artist", "")),
            title=str(metadata_result.get("title", "")),
            duration_seconds=float(metadata_result.get("duration_seconds", duration_seconds)),
            logger=logger,
        )
        dump_json_artifact(
            output_path=artifacts.perception_model_lrclib_match_path,
            payload=lrclib_result,
            logger=logger,
            artifact_name="lrclib_match",
        )
        if lrclib_result.get("status") == "instrumental" or bool(lrclib_result.get("instrumental", False)):
            result = {
                "provider": "lrclib",
                "reason": "instrumental",
                "lyric_sentence_units": [],
                "sentence_split_stats": {"reason": "instrumental", "sample_source": "lrclib"},
                "funasr_raw_result": {"skipped": True, "reason": "instrumental"},
            }
            _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
            dump_json_artifact(
                output_path=artifacts.perception_model_lrclib_lyric_sentence_units_path,
                payload=[],
                logger=logger,
                artifact_name="lrclib_lyric_sentence_units",
            )
            return result
        if lrclib_result.get("status") == "synced":
            lyric_sentence_units = parse_synced_lyrics_to_sentence_units(
                synced_lyrics=str(lrclib_result.get("synced_lyrics", "")),
                audio_duration=duration_seconds,
                logger=logger,
            )
            if lyric_sentence_units:
                dump_json_artifact(
                    output_path=artifacts.perception_model_lrclib_lyric_sentence_units_path,
                    payload=lyric_sentence_units,
                    logger=logger,
                    artifact_name="lrclib_lyric_sentence_units",
                )
                result = {
                    "provider": "lrclib",
                    "reason": "metadata_synced",
                    "lyric_sentence_units": lyric_sentence_units,
                    "sentence_split_stats": {
                        "reason": "metadata_synced",
                        "sample_source": "lrclib",
                        "sentence_count": len(lyric_sentence_units),
                    },
                    "funasr_raw_result": {"skipped": True, "reason": "lrclib_selected"},
                }
                _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
                return result

    if bool(enable_fingerprint_lookup):
        fingerprint_result = build_fingerprint_result(
            audio_path=audio_path,
            duration_seconds=duration_seconds,
            fpcalc_bin=fpcalc_bin,
            logger=logger,
        )
        dump_json_artifact(
            output_path=artifacts.perception_model_acoustid_fingerprint_result_path,
            payload=fingerprint_result,
            logger=logger,
            artifact_name="fingerprint_result",
        )
        acoustid_result = query_acoustid_match(
            fingerprint_result=fingerprint_result,
            acoustid_api_key_file=acoustid_api_key_file,
            logger=logger,
        )
        dump_json_artifact(
            output_path=artifacts.perception_model_acoustid_match_path,
            payload=acoustid_result,
            logger=logger,
            artifact_name="acoustid_match",
        )
        if acoustid_result.get("status") == "ok":
            lrclib_result = query_lrclib_lyrics(
                artist=str(acoustid_result.get("artist", "")),
                title=str(acoustid_result.get("title", "")),
                duration_seconds=float(acoustid_result.get("duration_seconds", duration_seconds)),
                logger=logger,
            )
            dump_json_artifact(
                output_path=artifacts.perception_model_lrclib_match_path,
                payload=lrclib_result,
                logger=logger,
                artifact_name="lrclib_match",
            )
            if lrclib_result.get("status") == "instrumental" or bool(lrclib_result.get("instrumental", False)):
                result = {
                    "provider": "lrclib",
                    "reason": "instrumental",
                    "lyric_sentence_units": [],
                    "sentence_split_stats": {"reason": "instrumental", "sample_source": "lrclib"},
                    "funasr_raw_result": {"skipped": True, "reason": "instrumental"},
                }
                _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
                dump_json_artifact(
                    output_path=artifacts.perception_model_lrclib_lyric_sentence_units_path,
                    payload=[],
                    logger=logger,
                    artifact_name="lrclib_lyric_sentence_units",
                )
                return result
            if lrclib_result.get("status") == "synced":
                lyric_sentence_units = parse_synced_lyrics_to_sentence_units(
                    synced_lyrics=str(lrclib_result.get("synced_lyrics", "")),
                    audio_duration=duration_seconds,
                    logger=logger,
                )
                if lyric_sentence_units:
                    dump_json_artifact(
                        output_path=artifacts.perception_model_lrclib_lyric_sentence_units_path,
                        payload=lyric_sentence_units,
                        logger=logger,
                        artifact_name="lrclib_lyric_sentence_units",
                    )
                    result = {
                        "provider": "lrclib",
                        "reason": "fingerprint_synced",
                        "lyric_sentence_units": lyric_sentence_units,
                        "sentence_split_stats": {
                            "reason": "fingerprint_synced",
                            "sample_source": "lrclib",
                            "sentence_count": len(lyric_sentence_units),
                        },
                        "funasr_raw_result": {"skipped": True, "reason": "lrclib_selected"},
                    }
                    _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
                    return result

    funasr_result = funasr_fallback_runner()
    if funasr_result is None:
        result = {
            "provider": "funasr",
            "reason": "silent_vocals_precheck",
            "lyric_sentence_units": [],
            "sentence_split_stats": {
                "skipped": True,
                "reason": "silent_vocals_precheck",
                "dynamic_gap_threshold_seconds": 0.35,
                "sample_source": "none",
                "sample_count_raw": 0,
                "sample_count_kept": 0,
                "sample_count_outlier": 0,
                "outlier_samples": [],
            },
            "funasr_raw_result": {"skipped": True, "reason": "silent_vocals_precheck"},
        }
        _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
        return result

    funasr_raw_result, lyric_sentence_units, sentence_split_stats = funasr_result
    result = {
        "provider": "funasr",
        "reason": "fallback",
        "lyric_sentence_units": list(lyric_sentence_units),
        "sentence_split_stats": dict(sentence_split_stats),
        "funasr_raw_result": funasr_raw_result,
    }
    _dump_selected_provider(artifacts=artifacts, payload=result, logger=logger)
    return result


def _dump_selected_provider(artifacts: ModuleAV2Artifacts, payload: dict[str, Any], logger) -> None:
    """
    功能说明：写入统一歌词来源选择结果。
    参数说明：
    - artifacts: 模块A V2产物路径对象。
    - payload: 统一歌词解析结果。
    - logger: 日志记录器。
    返回值：无。
    异常说明：异常由上层统一处理。
    边界条件：只落盘最小必要字段，避免重复堆积大对象。
    """
    dump_json_artifact(
        output_path=artifacts.perception_model_lyrics_selected_provider_path,
        payload={
            "provider": str(payload.get("provider", "")),
            "reason": str(payload.get("reason", "")),
            "lyric_unit_count": len(list(payload.get("lyric_sentence_units", []))),
        },
        logger=logger,
        artifact_name="selected_provider",
    )
