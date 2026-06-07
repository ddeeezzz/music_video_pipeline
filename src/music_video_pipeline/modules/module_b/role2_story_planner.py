"""
文件用途：提供模块 B role2 的剧情规划器。
核心流程：将故事模板与大段音频特征传给 LLM，再对返回结果做 Markdown 解析校验。
输入输出：输入故事模板与 big_segment_catalog，输出解析后的场景规划数组。
依赖说明：依赖模块 B prompt 模板、LLM 客户端与 Markdown 契约解析器。
维护说明：big_segment_catalog 已接通 module_a_output.json 真实数据，stub 仅作 fallback。
"""

# 标准库：用于复制 dataclass 配置对象。
from dataclasses import replace
# 标准库：用于路径类型标注。
from pathlib import Path
# 标准库：用于 JSON 元信息写盘。
import json
# 标准库：用于日志类型标注。
import logging
# 标准库：用于时间戳。
import time

# 项目内模块：提供模块 B LLM 配置对象。
from music_video_pipeline.config import ModuleBLlmConfig
# 项目内模块：提供模块 B LLM 调用函数。
from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat
# 项目内模块：提供模块 B role 工作目录路径。
from music_video_pipeline.modules.module_b.artifact_paths import (
    get_module_b_prompt_dir,
    get_module_b_role_dir,
    get_module_b_streaming_dir,
)
# 项目内模块：复用 role1 的 usage 元信息更新工具。
from music_video_pipeline.modules.module_b.role1_imagery_describer import _update_meta_with_usage
# 项目内模块：提供 role2 Markdown 契约解析器与数据结构。
from music_video_pipeline.modules.module_b.markdown_contracts import (
    ScenePlan,
    parse_scene_plans,
)
# 项目内模块：提供 role2 prompt 模板装配能力。
from music_video_pipeline.modules.module_b.prompt_templates import (
    ROLE2_PROMPT_TEMPLATE_REF,
    render_prompt_asset,
)


def build_big_segment_catalog_stub() -> str:
    """返回一份与 prompt 模板格式一致的占位 catalog 示例，用于后续真实数据聚合前的接通过程。"""
    return """
### big_001
- label: intro | 时长: 10.2s (0.0s ~ 10.2s)
- 人声占比: 0% | 器乐: 90% | 留白: 10%
- 能量: low → low，趋势: flat，节奏紧张度: 0.18
- 歌词: 无

### big_002
- label: verse | 时长: 22.6s (10.2s ~ 32.8s)
- 人声占比: 65% | 器乐: 25% | 留白: 10%
- 能量: low → mid，趋势: up，节奏紧张度: 0.42
- 歌词: 有（"一个人站了很久 / 看光线慢慢移动"）

### big_003
- label: chorus | 时长: 22.3s (32.8s ~ 55.1s)
- 人声占比: 72% | 器乐: 20% | 留白: 8%
- 能量: mid → high，趋势: up，节奏紧张度: 0.71
- 歌词: 有（"让它走吧 / 让它消失在风中"）

### big_004
- label: verse | 时长: 22.4s (55.1s ~ 77.5s)
- 人声占比: 60% | 器乐: 28% | 留白: 12%
- 能量: mid → mid，趋势: flat，节奏紧张度: 0.45
- 歌词: 有（"那些没说出口的 / 被时间慢慢带走"）

### big_005
- label: chorus | 时长: 22.3s (77.5s ~ 99.8s)
- 人声占比: 75% | 器乐: 18% | 留白: 7%
- 能量: high → high，趋势: flat，节奏紧张度: 0.78
- 歌词: 有（"不再等了 / 该走的就让它走吧"）

### big_006
- label: outro | 时长: 15.2s (99.8s ~ 115.0s)
- 人声占比: 10% | 器乐: 70% | 留白: 20%
- 能量: mid → low，趋势: down，节奏紧张度: 0.22
- 歌词: 有（"一切都安静下来了"）
""".strip()


def build_big_segment_catalog(module_a_output: dict) -> str:
    """从 module_a_output 提取 big_segment 音频特征并转为 catalog markdown。"""
    logger = logging.getLogger(__name__)
    big_segments: list[dict] = module_a_output.get("big_segments", [])
    segments: list[dict] = module_a_output.get("segments", [])
    energy_features: list[dict] = module_a_output.get("energy_features", [])
    lyric_units: list[dict] = module_a_output.get("lyric_units", [])

    # 建立索引：segments 按 big_segment_id 分组
    segs_by_big: dict[str, list[dict]] = {}
    for seg in segments:
        bid = str(seg.get("big_segment_id", ""))
        if bid:
            segs_by_big.setdefault(bid, []).append(seg)

    # 建立索引：energy_features 按 (start_time, end_time) 快速查找
    energy_by_time: dict[tuple[float, float], dict] = {}
    for ef in energy_features:
        key = (float(ef["start_time"]), float(ef["end_time"]))
        energy_by_time[key] = ef

    # 建立索引：lyric_units 按 segment_id 分组
    lyrics_by_seg: dict[str, list[dict]] = {}
    for lu in lyric_units:
        sid = str(lu.get("segment_id", ""))
        if sid:
            lyrics_by_seg.setdefault(sid, []).append(lu)

    lines: list[str] = []
    for big in big_segments:
        bid = str(big["segment_id"])
        label = str(big.get("label", ""))
        start = float(big["start_time"])
        end = float(big["end_time"])
        duration = end - start

        child_segs = segs_by_big.get(bid, [])

        # 过滤 Module A 的残余空段（< 0.05s 视为空段），仅当无子段时跳过。
        if duration < 0.05 and not child_segs:
            logger.info(
                "build_big_segment_catalog 过滤残余空段：%s label=%s duration=%.3fs child_segs=%s",
                bid, label, duration, len(child_segs),
            )
            continue

        lines.append(f"### {bid}")
        lines.append(f"- label: {label} | 时长: {duration:.3f}s ({start:.3f}s ~ {end:.3f}s)")

        # 人声占比：按子段时长加权
        if child_segs:
            vocal_dur = sum(
                s["end_time"] - s["start_time"]
                for s in child_segs
                if str(s.get("role", "")).lower() in ("lyric", "chant")
            )
            inst_dur = sum(
                s["end_time"] - s["start_time"]
                for s in child_segs
                if str(s.get("role", "")).lower() == "inst"
            )
            if duration > 0:
                vocal_pct = round(vocal_dur / duration * 100)
                inst_pct = round(inst_dur / duration * 100)
                silence_pct = max(0, 100 - vocal_pct - inst_pct)
                lines.append(f"- 人声占比: {vocal_pct}% | 器乐: {inst_pct}% | 留白: {silence_pct}%")
            else:
                lines.append("- 人声占比: — | 器乐: — | 留白: —")
        else:
            lines.append("- 人声占比: — | 器乐: — | 留白: —")

        # 能量聚合
        if child_segs:
            sorted_segs = sorted(child_segs, key=lambda s: float(s["start_time"]))
            tensions: list[float] = []
            peak_ef: dict | None = None
            peak_tension = -1.0

            for seg in sorted_segs:
                ef_key = (round(float(seg["start_time"]), 3), round(float(seg["end_time"]), 3))
                ef = energy_by_time.get(ef_key)
                if ef is not None:
                    rt = float(ef["rhythm_tension"])
                    tensions.append(rt)
                    if rt > peak_tension:
                        peak_tension = rt
                        peak_ef = ef

            first_ef = energy_by_time.get(
                (round(float(sorted_segs[0]["start_time"]), 3), round(float(sorted_segs[0]["end_time"]), 3))
            )
            last_ef = energy_by_time.get(
                (round(float(sorted_segs[-1]["start_time"]), 3), round(float(sorted_segs[-1]["end_time"]), 3))
            )

            if first_ef is not None and last_ef is not None and peak_ef is not None and tensions:
                energy_str = f"{first_ef['energy_level']} → {last_ef['energy_level']}"
                trend = str(peak_ef["trend"])
                avg_tension = sum(tensions) / len(tensions)
                lines.append(f"- 能量: {energy_str}，趋势: {trend}，节奏紧张度: {avg_tension:.2f}")
            else:
                lines.append("- 能量: —，趋势: —，节奏紧张度: —")
        else:
            lines.append("- 能量: —，趋势: —，节奏紧张度: —")

        # 歌词聚合
        lyric_texts: list[str] = []
        for seg in child_segs:
            seg_lyrics = lyrics_by_seg.get(str(seg["segment_id"]), [])
            for lu in seg_lyrics:
                text = str(lu.get("text", "")).strip()
                if text:
                    lyric_texts.append(text)

        if lyric_texts:
            combined = " / ".join(lyric_texts)
            if len(combined) > 200:
                combined = combined[:200] + "..."
            lines.append("- 歌词: 有（" + combined + "）")
        else:
            lines.append("- 歌词: 无")

        # 黑屏许可标注（仅 start/end 需要，其余 label 不输出此字段）
        if label.lower() in ("start", "end"):
            has_vocals = bool(lyric_texts)
            if duration < 4.0 and not has_vocals:
                lines.append("- 黑白屏许可: 允许")
            else:
                lines.append("- 黑白屏许可: 禁止")

        lines.append("")

    return "\n".join(lines).strip()


def build_big_segment_catalog_with_segments(module_a_output: dict) -> str:
    """从 module_a_output 提取 big_segment 音频特征并转为 catalog markdown，
    包含每个大段下的子段（seg_xxxx）条目，供 role3 使用全局 segment_id。"""
    logger = logging.getLogger(__name__)
    big_segments: list[dict] = module_a_output.get("big_segments", [])
    segments: list[dict] = module_a_output.get("segments", [])
    energy_features: list[dict] = module_a_output.get("energy_features", [])
    lyric_units: list[dict] = module_a_output.get("lyric_units", [])

    segs_by_big: dict[str, list[dict]] = {}
    for seg in segments:
        bid = str(seg.get("big_segment_id", ""))
        if bid:
            segs_by_big.setdefault(bid, []).append(seg)

    energy_by_time: dict[tuple[float, float], dict] = {}
    for ef in energy_features:
        key = (float(ef["start_time"]), float(ef["end_time"]))
        energy_by_time[key] = ef

    lyrics_by_seg: dict[str, list[dict]] = {}
    for lu in lyric_units:
        sid = str(lu.get("segment_id", ""))
        if sid:
            lyrics_by_seg.setdefault(sid, []).append(lu)

    lines: list[str] = []
    for big in big_segments:
        bid = str(big["segment_id"])
        label = str(big.get("label", ""))
        start = float(big["start_time"])
        end = float(big["end_time"])
        duration = end - start

        child_segs = segs_by_big.get(bid, [])

        if duration < 0.05 and not child_segs:
            logger.info(
                "build_big_segment_catalog_with_segments 过滤残余空段：%s label=%s duration=%.3fs child_segs=%s",
                bid, label, duration, len(child_segs),
            )
            continue

        lines.append(f"### {bid}")
        lines.append(f"- label: {label} | 时长: {duration:.3f}s ({start:.3f}s ~ {end:.3f}s)")

        if child_segs:
            vocal_dur = sum(
                s["end_time"] - s["start_time"]
                for s in child_segs
                if str(s.get("role", "")).lower() in ("lyric", "chant")
            )
            inst_dur = sum(
                s["end_time"] - s["start_time"]
                for s in child_segs
                if str(s.get("role", "")).lower() == "inst"
            )
            if duration > 0:
                vocal_pct = round(vocal_dur / duration * 100)
                inst_pct = round(inst_dur / duration * 100)
                silence_pct = max(0, 100 - vocal_pct - inst_pct)
                lines.append(f"- 人声占比: {vocal_pct}% | 器乐: {inst_pct}% | 留白: {silence_pct}%")
            else:
                lines.append("- 人声占比: — | 器乐: — | 留白: —")
        else:
            lines.append("- 人声占比: — | 器乐: — | 留白: —")

        if child_segs:
            sorted_segs = sorted(child_segs, key=lambda s: float(s["start_time"]))
            tensions: list[float] = []
            peak_ef: dict | None = None
            peak_tension = -1.0

            for seg in sorted_segs:
                ef_key = (round(float(seg["start_time"]), 3), round(float(seg["end_time"]), 3))
                ef = energy_by_time.get(ef_key)
                if ef is not None:
                    rt = float(ef["rhythm_tension"])
                    tensions.append(rt)
                    if rt > peak_tension:
                        peak_tension = rt
                        peak_ef = ef

            first_ef = energy_by_time.get(
                (round(float(sorted_segs[0]["start_time"]), 3), round(float(sorted_segs[0]["end_time"]), 3))
            )
            last_ef = energy_by_time.get(
                (round(float(sorted_segs[-1]["start_time"]), 3), round(float(sorted_segs[-1]["end_time"]), 3))
            )

            if first_ef is not None and last_ef is not None and peak_ef is not None and tensions:
                energy_str = f"{first_ef['energy_level']} → {last_ef['energy_level']}"
                trend = str(peak_ef["trend"])
                avg_tension = sum(tensions) / len(tensions)
                lines.append(f"- 能量: {energy_str}，趋势: {trend}，节奏紧张度: {avg_tension:.2f}")
            else:
                lines.append("- 能量: —，趋势: —，节奏紧张度: —")
        else:
            lines.append("- 能量: —，趋势: —，节奏紧张度: —")

        lyric_texts: list[str] = []
        for seg in child_segs:
            seg_lyrics = lyrics_by_seg.get(str(seg["segment_id"]), [])
            for lu in seg_lyrics:
                text = str(lu.get("text", "")).strip()
                if text:
                    lyric_texts.append(text)

        if lyric_texts:
            combined = " / ".join(lyric_texts)
            if len(combined) > 200:
                combined = combined[:200] + "..."
            lines.append("- 歌词: 有（" + combined + "）")
        else:
            lines.append("- 歌词: 无")

        # 黑屏许可标注（仅 start/end 需要，其余 label 不输出此字段）
        if label.lower() in ("start", "end"):
            has_vocals = bool(lyric_texts)
            if duration < 4.0 and not has_vocals:
                lines.append("- 黑白屏许可: 允许")
            else:
                lines.append("- 黑白屏许可: 禁止")

        lines.append("")

        # 子段（shot）信息：保留 Module A 的全局 segment_id
        for seg in sorted(child_segs, key=lambda s: float(s["start_time"])):
            sid = str(seg["segment_id"])
            seg_start = float(seg["start_time"])
            seg_end = float(seg["end_time"])
            seg_duration = seg_end - seg_start
            seg_label = str(seg.get("label", ""))
            lines.append(f"### {sid}")
            lines.append(f"- label: {seg_label} | 时长: {seg_duration:.3f}s ({seg_start:.3f}s ~ {seg_end:.3f}s)")

            ef = energy_by_time.get((round(seg_start, 3), round(seg_end, 3)))
            if ef is not None:
                lines.append(f"- 能量: {ef['energy_level']}，趋势: {ef['trend']}，节奏紧张度: {float(ef['rhythm_tension']):.2f}")
            else:
                lines.append("- 能量: —，趋势: —，节奏紧张度: —")

            seg_lyrics_list = lyrics_by_seg.get(sid, [])
            seg_texts = [str(lu.get("text", "")).strip() for lu in seg_lyrics_list if str(lu.get("text", "")).strip()]
            if seg_texts:
                combined_text = " / ".join(seg_texts)
                if len(combined_text) > 200:
                    combined_text = combined_text[:200] + "..."
                lines.append("- 歌词: 有（" + combined_text + "）")
            else:
                lines.append("- 歌词: 无")

            lines.append("")

    return "\n".join(lines).strip()


class Role2StoryPlanner:
    """执行模块 B role2 剧情规划。"""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm_config: ModuleBLlmConfig,
        project_root: Path,
        artifacts_dir: Path | None = None,
    ) -> None:
        self._logger = logger
        self._llm_config = llm_config
        self._project_root = project_root
        self._artifacts_dir = artifacts_dir.resolve() if isinstance(artifacts_dir, Path) else None
        self._work_dir = (
            get_module_b_role_dir(self._artifacts_dir, "role2")
            if self._artifacts_dir is not None
            else None
        )
        self._prompt_dir = (
            get_module_b_prompt_dir(self._artifacts_dir, "role2")
            if self._artifacts_dir is not None
            else None
        )
        self._streaming_dir = (
            get_module_b_streaming_dir(self._artifacts_dir, "role2")
            if self._artifacts_dir is not None
            else None
        )
        self._stream_preview_path = (
            (self._streaming_dir / "role2_story_output.streaming.md").resolve()
            if self._streaming_dir is not None
            else None
        )
        self._stream_preview_meta_path = (
            (self._streaming_dir / "role2_story_output.streaming.meta.json").resolve()
            if self._streaming_dir is not None
            else None
        )

    def generate(self, story_template_markdown: str, big_segment_catalog: str) -> list[ScenePlan]:
        """根据故事模板与大段音频特征生成并校验 role2 结果。"""
        prompt_template_ref = ROLE2_PROMPT_TEMPLATE_REF
        prompt_template_file_override = str(self._llm_config.prompt_template_file).strip()
        if prompt_template_file_override:
            prompt_template_ref = replace(prompt_template_ref, template_file=prompt_template_file_override)

        prompt_asset = render_prompt_asset(
            project_root=self._project_root,
            prompt_template_ref=prompt_template_ref,
            user_variables={
                "模板的## 故事和## 意象": _normalize_markdown_text(
                    "role2.story_template", story_template_markdown
                ),
                "big_segment_catalog": str(big_segment_catalog or "").strip(),
            },
        )
        if self._artifacts_dir is not None:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._prompt_dir.mkdir(parents=True, exist_ok=True)
            (self._prompt_dir / "role2_rendered_prompt.md").write_text(
                prompt_asset["user_prompt_markdown"], encoding="utf-8"
            )
        call_llm_config = self._llm_config
        if self._stream_preview_path is not None:
            self._stream_preview_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream_preview_path.write_text("", encoding="utf-8")
        if self._stream_preview_meta_path is not None:
            self._write_stream_preview_meta(
                current_attempt=1,
                first_chunk_at_ms=0,
                last_chunk_at_ms=0,
            )

        last_error: Exception | None = None
        response_text = ""
        try:
            response_text, usage = call_module_b_llm_chat(**self._build_llm_call_kwargs(
                call_llm_config=call_llm_config,
                prompt_asset=prompt_asset,
            ))
            _update_meta_with_usage(self._stream_preview_meta_path, usage, fallback_text=response_text)
            response_markdown = _normalize_markdown_text("role2.response_markdown", response_text)
            return parse_scene_plans(response_markdown)
        except Exception as error:  # noqa: BLE001
            last_error = self._persist_failure_artifacts(response_text=response_text, error=error)
        raise RuntimeError(f"module_b: role2 执行失败：{last_error}")

    def _write_stream_preview_meta(
        self,
        *,
        current_attempt: int,
        first_chunk_at_ms: int,
        last_chunk_at_ms: int,
        completion_tokens: int | None = None,
        speed_tokens_per_sec: float | None = None,
    ) -> None:
        """写入 role2 流式预览元信息。"""
        if self._stream_preview_meta_path is None:
            return
        try:
            payload = {
                "current_attempt": max(1, int(current_attempt)),
                "first_chunk_at_ms": max(0, int(first_chunk_at_ms)),
                "first_chunk_at": _format_timestamp_ms(first_chunk_at_ms),
                "last_chunk_at_ms": max(0, int(last_chunk_at_ms)),
                "last_chunk_at": _format_timestamp_ms(last_chunk_at_ms),
            }
            if completion_tokens is not None and completion_tokens > 0:
                payload["completion_tokens"] = completion_tokens
            if speed_tokens_per_sec is not None and speed_tokens_per_sec > 0:
                payload["speed_tokens_per_sec"] = round(speed_tokens_per_sec, 2)
            self._stream_preview_meta_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            return

    def _persist_failure_artifacts(self, response_text: str, error: Exception) -> Exception:
        """在 role2 失败时保留原始返回与失败原因文件。"""
        if self._work_dir is None:
            return error
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            raw_output_path = self._work_dir / "role2_story_output.failed.md"
            reason_path = self._work_dir / "role2_story_output.failed.reason.txt"
            if str(response_text).strip():
                raw_output_path.write_text(str(response_text).strip() + "\n", encoding="utf-8")
            reason_path.write_text(str(error).strip() + "\n", encoding="utf-8")
            self._logger.warning(
                "模块 B role2 执行失败，已保留原始返回与失败原因，raw_path=%s，reason_path=%s",
                raw_output_path,
                reason_path,
            )
            return RuntimeError(
                "模块 B role2 执行失败："
                f"{str(error).strip()}；"
                f"原始返回路径={raw_output_path if str(response_text).strip() else '<空响应未落盘>'}；"
                f"失败原因路径={reason_path}"
            )
        except Exception as persist_error:  # noqa: BLE001
            self._logger.warning(
                "模块 B role2 失败落盘原始返回时出错，error=%s，persist_error=%s",
                error,
                persist_error,
            )
            return RuntimeError(
                "模块 B role2 执行失败："
                f"{str(error).strip()}；"
                f"原始返回落盘失败={persist_error}"
            )

    def _build_llm_call_kwargs(
        self,
        *,
        call_llm_config: ModuleBLlmConfig,
        prompt_asset: dict[str, str],
    ) -> dict[str, object]:
        """构造 role2 LLM 调用参数，并在可用时注入流式预览与重试回调。"""
        kwargs: dict[str, object] = {
            "logger": self._logger,
            "llm_config": call_llm_config,
            "messages": _build_messages(
                system_prompt=prompt_asset["system_prompt"],
                user_prompt_markdown=prompt_asset["user_prompt_markdown"],
            ),
            "project_root": self._project_root,
        }
        stream_callback = self._build_stream_preview_callback()
        if stream_callback is not None:
            kwargs["on_stream_chunk"] = stream_callback
        retry_hint_callback = self._build_retry_hint_callback()
        if retry_hint_callback is not None:
            kwargs["on_retry_hint"] = retry_hint_callback
        return kwargs

    def _build_retry_hint_callback(self):
        """构造 role2 重试提示回调。"""
        def _on_retry(attempt_index: int, error: Exception) -> str:
            del attempt_index
            return (
                f"上次输出不符合要求：{error}。"
                "这次必须严格输出 Markdown，每个 `## big_segment_id` 下只保留"
                " `- imagery_used:` 和 `- story_outline_zh:` 两个字段。"
            )
        return _on_retry

    def _build_stream_preview_callback(self):
        """构造 role2 流式输出写盘回调。"""
        if self._stream_preview_path is None:
            return None
        self._write_stream_preview_meta(
            current_attempt=1,
            first_chunk_at_ms=0,
            last_chunk_at_ms=0,
        )
        first_chunk_at_ms = 0
        has_written_chunk = False
        stream_start_ms = int(time.time() * 1000)
        call_count = 0

        def _on_stream_chunk(aggregated_text: str, _delta_text: str) -> None:
            nonlocal first_chunk_at_ms, has_written_chunk, call_count
            call_count += 1
            current_time_ms = int(time.time() * 1000)
            delta_text = str(_delta_text or "")
            if not delta_text:
                return
            if first_chunk_at_ms <= 0:
                first_chunk_at_ms = current_time_ms
                elapsed_ms = first_chunk_at_ms - stream_start_ms
                self._logger.info(
                    "模块 B role2 流式写盘收到首个 chunk：耗时=%sms",
                    elapsed_ms,
                )
            if not has_written_chunk:
                if call_count > 1 and self._stream_preview_path.exists():
                    existing_text = self._stream_preview_path.read_text(encoding="utf-8").rstrip()
                    if existing_text:
                        self._stream_preview_path.write_text(
                            f"{existing_text}\n\n[Retry]\n", encoding="utf-8"
                        )
                has_written_chunk = True
            with self._stream_preview_path.open("a", encoding="utf-8") as preview_file:
                preview_file.write(delta_text)
            self._logger.debug(
                "模块 B role2 流式写盘：delta_chars=%s total_chars=%s",
                len(delta_text),
                len(aggregated_text),
            )
            self._write_stream_preview_meta(
                current_attempt=1,
                first_chunk_at_ms=first_chunk_at_ms,
                last_chunk_at_ms=current_time_ms,
            )

        return _on_stream_chunk


def _format_timestamp_ms(timestamp_ms: int) -> str:
    """把毫秒时间戳格式化为本地可读时间文本。"""
    normalized_timestamp_ms = int(timestamp_ms or 0)
    if normalized_timestamp_ms <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(normalized_timestamp_ms / 1000))


def _normalize_markdown_text(field_name: str, value: str) -> str:
    """标准化并校验非空 Markdown 字符串。"""
    normalized_text = str(value or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        raise ValueError(f"{field_name} 不能为空。")
    return normalized_text


def _build_messages(
    *,
    system_prompt: str,
    user_prompt_markdown: str,
) -> list[dict[str, str]]:
    """构建 role2 的标准 messages 数组。"""
    return [
        {"role": "system", "content": str(system_prompt or "").strip()},
        {"role": "user", "content": str(user_prompt_markdown or "").strip()},
    ]
