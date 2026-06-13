"""
文件用途：提供 FunASR 歌词矫正能力——将用户选中的网络歌词与 FunASR 识别结果
          通过 LLM 逐批匹配时间戳，获得每句歌词首字准确时间。
核心流程：读取网络歌词与 FunASR 产物 → 每次发 5 句网络歌词 + 对应范围 FunASR 字级时间戳
        → LLM 返回 JSON 匹配 start_time → 写入结果。
输入输出：输入网络歌词文本与任务 artifacts 目录，输出矫正后的带时间戳歌词。
维护说明：硬编码配置，不读配置文件。修改批次大小请改 BATCH_SIZE。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from music_video_pipeline.modules.module_a_v2.lrc_parser import (
    LRC_TIMESTAMP_PATTERN as LRC_TS,
    parse_synced_lyrics_to_sentence_units,
)
from music_video_pipeline.modules.module_a_v2.network_lyrics_state import (
    load_module_a_network_lyrics_state,
)
from music_video_pipeline.modules.module_a_v2.utils.time_utils import round_time
from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat

# 硬编码：每次发 5 句正确歌词给 LLM
BATCH_SIZE = 5
# 硬编码：每批前后额外加的 FunASR 语句数
CONTEXT_MARGIN = 2

LRC_LINE_PATTERN = re.compile(r"^\[(?P<time>\d{1,2}:\d{2}(?:\.\d{1,3})?)\](?P<text>.*)$")

# 硬编码 system prompt — 仅返回 JSON
SYSTEM_PROMPT = """你是一个歌词时间戳匹配助手。

【任务】
我会给你：
1. 歌名和歌手
2. 若干句正确的歌词（带时间标签）
3. FunASR 语音识别结果（含每个字的起始/结束时间戳，格式为 [00:00.00]字 秒 秒）

你的任务：对每句正确歌词，在 FunASR 字级时间戳中找到该句第一个字符的起始时间。

【规则】
- FunASR 识别可能有错字、多字、漏字，不要被干扰
- 匹配依据：**正确歌词的第一个字符**在 FunASR 字级时间戳中出现在哪个位置
- 如果无法精确匹配第一个字符，用 FunASR 句子的 start_time 作为近似
- **只返回 JSON 数组**，不要多余文字
- 每项格式：{"text": "<正确歌词全文>", "start_time": "<MM:SS.xx>"}
- start_time 保留两位小数

【例子】
正确歌词第一句: "悪戯は知らん顔で"
FunASR 中匹配到 "悪" 字的起始时间是 00:22.69
→ {"text": "悪戯は知らん顔で", "start_time": "00:22.69"}"""


def _read_funasr_raw(artifacts_dir: Path, logger) -> list[dict[str, Any]]:
    """读取 FunASR 原始识别结果。"""
    path = artifacts_dir / "module_a_work_v2" / "perception" / "model" / "funasr" / "funasr_raw_response.json"
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return [raw]
            if isinstance(raw, list):
                return raw
        logger.warning("FunASR 矫正：原始结果文件不存在，path=%s", path)
    except Exception as error:
        logger.warning("FunASR 矫正：原始结果读取失败，path=%s，错误=%s", path, error)
    return []


def _read_selected_network_lyrics(artifacts_dir: Path, logger) -> dict[str, Any]:
    """读取用户已启用的联网歌词选择结果。"""
    state = load_module_a_network_lyrics_state(artifacts_dir=artifacts_dir)
    if not bool(state.get("enabled", False)):
        logger.warning("FunASR 矫正：未启用联网歌词，无法矫正。")
        return {}
    candidate = state.get("selected_candidate", {})
    if not isinstance(candidate, dict):
        return {}
    synced = str(candidate.get("synced_lyrics", "")).strip()
    if not synced:
        logger.warning("FunASR 矫正：已启用但选中候选无同步歌词。")
        return {}
    return {
        "synced_lyrics": synced,
        "artist": str(candidate.get("artist", "")).strip(),
        "title": str(candidate.get("title", "")).strip(),
        "provider": str(candidate.get("provider", "")).strip(),
        "romanized_lyrics": str(candidate.get("romanized_lyrics", "")).strip(),
        "translated_lyrics": str(candidate.get("translated_lyrics", "")).strip(),
    }


def _extract_lrc_lines(lrc_text: str) -> list[dict[str, Any]]:
    """从 LRC 文本中按行提取时间戳与歌词文本。"""
    lines: list[dict[str, Any]] = []
    for raw_line in lrc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LRC_LINE_PATTERN.match(line)
        if not match:
            continue
        time_label = str(match.group("time")).strip()
        text = str(match.group("text")).strip()
        if not text:
            continue
        minutes, seconds, *fraction = time_label.replace(".", ":").split(":")
        time_sec = int(minutes) * 60 + int(seconds) + (float(f"0.{fraction[0]}") if fraction else 0.0)
        lines.append({"time": time_sec, "text": text, "raw_time_label": time_label})
    return lines


def _extract_funasr_utterances(funasr_raw: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """从 FunASR 原始结果中提取带字级时间戳的语句列表。"""
    utterances: list[dict[str, Any]] = []
    raw_uts: list[Any] = []
    punct_chars = set("，、；：。！？!?,.;: ")
    if isinstance(funasr_raw, list):
        for item in funasr_raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            timestamps = item.get("timestamps") or item.get("token_timestamps") or []
            tokens: list[dict[str, Any]] = []
            if isinstance(timestamps, list):
                for ts_item in timestamps:
                    if isinstance(ts_item, dict):
                        tok_text = str(ts_item.get("text", ts_item.get("token", ""))).strip()
                        if tok_text:
                            tokens.append({
                                "text": tok_text,
                                "start": float(ts_item.get("start", ts_item.get("start_time", 0.0))),
                                "end": float(ts_item.get("end", ts_item.get("end_time", 0.0))),
                            })
                    elif isinstance(ts_item, (list, tuple)) and len(ts_item) >= 3:
                        tok_text = str(ts_item[2]).strip()
                        if tok_text:
                            tokens.append({"text": tok_text, "start": float(ts_item[0]), "end": float(ts_item[1])})
            start = float(item.get("start", item.get("start_time", 0.0)))
            end = float(item.get("end", item.get("end_time", 0.0)))
            confidence = float(item.get("confidence", item.get("score", 0.65)))
            if tokens and len([u for u in utterances if u.get("tokens")]) == 0:
                sentence_parts: list[list[dict[str, Any]]] = [[]]
                for tok in tokens:
                    sentence_parts[-1].append(tok)
                    if tok["text"] in punct_chars and len(sentence_parts[-1]) > 1:
                        sentence_parts.append([])
                sentence_parts = [sp for sp in sentence_parts if sp]
                for sp in sentence_parts:
                    st = round_time(sp[0]["start"])
                    et = round_time(sp[-1]["end"])
                    sp_text = "".join(t["text"] for t in sp).strip()
                    sp_text = sp_text.strip("，、；：。！？!?,.;: ")
                    if sp_text:
                        utterances.append({
                            "start": st, "end": et, "text": sp_text,
                            "confidence": confidence,
                            "tokens": [{"text": t["text"], "start": t["start"], "end": t["end"]} for t in sp],
                        })
            else:
                utterances.append({
                    "start": round_time(start), "end": round_time(end),
                    "text": text, "confidence": confidence, "tokens": tokens,
                })
    elif isinstance(funasr_raw, dict):
        raw_uts = funasr_raw.get("utterances") or funasr_raw.get("sentences") or []
    else:
        raw_uts = []
    if isinstance(raw_uts, list):
        for item in raw_uts:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", item.get("start_time", 0.0)))
            end = float(item.get("end", item.get("end_time", 0.0)))
            tokens_raw = item.get("tokens") or item.get("words") or []
            tokens: list[dict[str, Any]] = []
            if isinstance(tokens_raw, list):
                for tok in tokens_raw:
                    if not isinstance(tok, dict):
                        continue
                    tok_text = str(tok.get("text", "")).strip()
                    if not tok_text:
                        continue
                    tokens.append({
                        "text": tok_text,
                        "start": float(tok.get("start", tok.get("start_time", start))),
                        "end": float(tok.get("end", tok.get("end_time", end))),
                    })
            utterances.append({
                "start": round_time(start),
                "end": round_time(end),
                "text": text,
                "confidence": float(item.get("confidence", item.get("score", 0.65))),
                "tokens": tokens,
            })
    return utterances


def _select_funasr_context(
    funasr_utterances: list[dict[str, Any]],
    batch_network_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为当前批次的网络歌词选择对应的 FunASR 上下文语句。
    取这批网络歌词起止时间前后各扩展 CONTEXT_MARGIN 个 FunASR 语句。"""
    batch_start_idx = -1
    batch_end_idx = -1
    batch_time_start = batch_network_lines[0]["time"]
    batch_time_end = batch_network_lines[-1]["time"]

    for idx, ut in enumerate(funasr_utterances):
        if ut["end"] >= batch_time_start and batch_start_idx == -1:
            batch_start_idx = max(0, idx - CONTEXT_MARGIN)
        if ut["start"] <= batch_time_end:
            batch_end_idx = idx
    if batch_end_idx != -1:
        batch_end_idx = min(len(funasr_utterances) - 1, batch_end_idx + CONTEXT_MARGIN)

    if batch_start_idx == -1:
        batch_start_idx = 0
    if batch_end_idx == -1:
        batch_end_idx = len(funasr_utterances) - 1

    return funasr_utterances[batch_start_idx : batch_end_idx + 1]


def _fmt_sec(seconds: float) -> str:
    """将秒数格式化为 MM:SS.xx"""
    total_seconds = max(0, seconds)
    m = int(total_seconds // 60)
    s = total_seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _build_funasr_text(funasr_lines: list[dict[str, Any]]) -> str:
    """将 FunASR 语句格式化为 LLM 可读文本。
    每行格式：[句起始时间] 句子全文
    字级时间戳： 字 (起始秒, 结束秒)
    """
    parts: list[str] = []
    for ut in funasr_lines:
        tokens = ut.get("tokens", [])
        ut_start = _fmt_sec(ut["start"])
        parts.append(f"[{ut_start}] {ut['text']}")
        if tokens:
            token_lines = []
            for tok in tokens:
                t_text = tok.get("text", "")
                t_start = tok.get("start", 0.0)
                t_end = tok.get("end", 0.0)
                token_lines.append(f"  {t_text} ({t_start:.2f}, {t_end:.2f})")
            parts.append("\n".join(token_lines))
    return "\n".join(parts)


def _build_batch_user_prompt(
    artist: str,
    title: str,
    batch_network_lines: list[dict[str, Any]],
    funasr_lines: list[dict[str, Any]],
) -> str:
    """为当前批次构建 user prompt。"""
    network_text = "\n".join(f"{ln['raw_time_label']} {ln['text']}" for ln in batch_network_lines)
    funasr_text = _build_funasr_text(funasr_lines)
    return (
        f"歌曲：{artist} - {title}\n\n"
        f"正确歌词（{len(batch_network_lines)}句）：\n{network_text}\n\n"
        f"FunASR 识别结果（含字级时间戳）：\n{funasr_text}\n\n"
        "请返回 JSON 数组，每项格式为 {\"text\": \"正确歌词全文\", \"start_time\": \"MM:SS.xx\"}\n"
        "start_time 是该句第一个字符在 FunASR 中的起始时间。"
    )


def _parse_json_output(llm_text: str) -> list[dict[str, Any]]:
    """从 LLM 返回的文本中提取 JSON 数组。"""
    text = llm_text.strip()
    # 尝试直接解析
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取 [] 之间的内容
    bracket_match = re.search(r"(\[.*?\])", text, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1))
        except json.JSONDecodeError:
            pass
    return []


def correct_funasr_with_llm(
    artifacts_dir: Path,
    logger,
    llm_config: Any,
    project_root: Path,
    network_data_override: dict[str, Any] | None = None,
    on_chunk_corrected: Callable[[int, int, list[dict[str, Any]]], None] | None = None,
    on_stream_chunk: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """
    执行 FunASR 歌词矫正流程（新版本 — 逐批 5 句发送）。
    每次发 5 句正确歌词 + 对应范围的 FunASR 字级时间戳给 LLM，
    LLM 返回 JSON 匹配每句首字时间戳。
    """
    if network_data_override:
        network_data = network_data_override
    else:
        network_data = _read_selected_network_lyrics(artifacts_dir=artifacts_dir, logger=logger)
    if not network_data:
        logger.warning("FunASR 矫正：没有已启用的网络歌词，跳过矫正。")
        return []

    funasr_raw = _read_funasr_raw(artifacts_dir=artifacts_dir, logger=logger)
    funasr_utterances = _extract_funasr_utterances(funasr_raw)
    if not funasr_utterances:
        logger.warning("FunASR 矫正：FunASR 原始结果为空或格式不兼容。")
        return []

    network_lines = _extract_lrc_lines(network_data["synced_lyrics"])
    if not network_lines:
        logger.warning("FunASR 矫正：网络歌词为空。")
        return []

    # 按 BATCH_SIZE 分批
    batches = [network_lines[i:i + BATCH_SIZE] for i in range(0, len(network_lines), BATCH_SIZE)]
    logger.info("FunASR 矫正开始：网络歌词 %s 行，FunASR %s 句，分为 %s 批",
                len(network_lines), len(funasr_utterances), len(batches))

    all_corrected_lines: list[dict[str, Any]] = []

    for batch_idx, batch_network_lines in enumerate(batches):
        funasr_context = _select_funasr_context(funasr_utterances, batch_network_lines)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_batch_user_prompt(
                artist=network_data["artist"],
                title=network_data["title"],
                batch_network_lines=batch_network_lines,
                funasr_lines=funasr_context,
            )},
        ]

        accumulated_llm_text: list[str] = []

        def _on_chunk(text: str, _delta: str = "") -> None:
            accumulated_llm_text.append(text)
            if on_stream_chunk is not None:
                on_stream_chunk(text)

        try:
            llm_text, _usage = call_module_b_llm_chat(
                logger=logger,
                llm_config=llm_config,
                messages=messages,
                project_root=project_root,
                on_stream_chunk=_on_chunk,
            )
        except Exception as error:
            logger.warning("FunASR 矫正：第 %s 批 LLM 调用失败，错误=%s", batch_idx + 1, error)
            continue

        parsed = _parse_json_output(llm_text)
        if not parsed:
            logger.warning("FunASR 矫正：第 %s 批 LLM 返回无法解析，原始=%s", batch_idx + 1, llm_text[:200])
            continue

        logger.info("FunASR 矫正：第 %s/%s 批 LLM 返回 %s 个时间戳",
                    batch_idx + 1, len(batches), len(parsed))

        for item in parsed:
            text = str(item.get("text", "")).strip()
            start_time_str = str(item.get("start_time", "")).strip()
            if not text or not start_time_str:
                continue
            # 在原始 network_lines 中找到匹配的文本
            matched_line = None
            for nl in network_lines:
                if nl["text"] == text:
                    matched_line = nl
                    break
            if matched_line is None:
                # 未找到匹配，仍然写入，保留网络歌词的文本
                all_corrected_lines.append({
                    "time": 0.0,
                    "text": text,
                    "raw_time_label": start_time_str,
                    "romanized_text": "",
                    "translated_text": "",
                    "token_timestamps": [],
                })
                continue

            try:
                parts = start_time_str.replace(".", ":").split(":")
                minutes = int(parts[0])
                seconds_parts = parts[1].split(":")
                seconds = int(seconds_parts[0])
                fraction = float(f"0.{seconds_parts[1]}") if len(seconds_parts) > 1 else 0.0
                time_sec = minutes * 60 + seconds + fraction
            except Exception:
                time_sec = matched_line["time"]

            all_corrected_lines.append({
                "time": round_time(time_sec),
                "text": matched_line["text"],
                "raw_time_label": start_time_str,
                "romanized_text": "",
                "translated_text": "",
                "token_timestamps": [],
            })

        if on_chunk_corrected is not None:
            # 返回当前批次的矫正结果（前端展示用）
            batch_corrected = [
                {"time": 0.0, "text": text, "raw_time_label": item.get("start_time", ""),
                 "romanized_text": "", "translated_text": ""}
                for item in parsed if item.get("text") and item.get("start_time")
            ]
            on_chunk_corrected(batch_idx, len(batches), batch_corrected)

    all_corrected_lines.sort(key=lambda x: x["time"])
    logger.info("FunASR 矫正完成：共 %s 行", len(all_corrected_lines))

    # 按时间戳匹配翻译和罗马音
    if network_data:
        roma_map: dict[str, str] = {}
        trans_map: dict[str, str] = {}
        for raw_line in str(network_data.get("romanized_lyrics", "")).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = LRC_LINE_PATTERN.match(line)
            if m and m.group("text").strip():
                roma_map[m.group("time").strip()] = m.group("text").strip()
        for raw_line in str(network_data.get("translated_lyrics", "")).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = LRC_LINE_PATTERN.match(line)
            if m and m.group("text").strip():
                trans_map[m.group("time").strip()] = m.group("text").strip()
        for line in all_corrected_lines:
            rt = line.get("raw_time_label", "")
            if not line.get("romanized_text") and rt in roma_map:
                line["romanized_text"] = roma_map[rt]
            if rt in trans_map:
                line["translated_text"] = trans_map[rt]

    return all_corrected_lines
