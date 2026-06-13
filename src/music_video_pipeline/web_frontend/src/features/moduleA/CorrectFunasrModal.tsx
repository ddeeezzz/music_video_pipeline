import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Alert, App, Button, Checkbox, Modal, Space, Tag, Typography } from "antd";

import { buildTaskModuleACorrectFunasrSocketUrl } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import type { TaskModuleALyricCandidate, TaskModuleALyricDetail } from "@/schemas/moduleA";

interface CorrectFunasrModalProps {
  open: boolean;
  onClose: () => void;
  taskId: string;
  lyricsPreviewText: string;
  lyricsPreviewCandidate: TaskModuleALyricCandidate | null;
  onSave: (data: TaskModuleALyricCandidate & Record<string, unknown>) => void;
  onResegment: () => void;
  selectedCandidateId: string;
  lyricsPreviewWordTimedText: string;
  lyricsPreviewTranslatedText: string;
  lyricsPreviewRomanizedText: string;
  lyricsPreviewRows: Array<{ timeLabel: string; originalText: string; tokens: Array<{ text: string; start: string; end: string }> }>;
}

const LRC_LINE_PATTERN = /^\[(?<time>\d{2}:\d{2}(?:\.\d{1,3})?)\](?<text>.*)$/;
const ENHANCED_TOKEN_PATTERN = /<(?<start>\d{2}:\d{2}(?:\.\d{1,3})?)>(?<text>.*?)<(?<end>\d{2}:\d{2}(?:\.\d{1,3})?)>/g;

export function CorrectFunasrModal({
  open, onClose, taskId, lyricsPreviewText, lyricsPreviewCandidate,
  onSave, onResegment, selectedCandidateId, lyricsPreviewWordTimedText,
  lyricsPreviewTranslatedText, lyricsPreviewRomanizedText, lyricsPreviewRows,
}: CorrectFunasrModalProps) {
  const { message } = App.useApp();
  const socketRef = useRef<WebSocket | null>(null);
  const wtRef = useRef("");
  const romaRef = useRef("");
  const transRef = useRef("");
  const lyricsRef = useRef("");
  const artistRef = useRef("");
  const titleRef = useRef("");
  const [connected, setConnected] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [correctedLines, setCorrectedLines] = useState<Array<{ time: number; text: string; raw_time_label: string }>>([]);
  const [funasrUtterances, setFunasrUtterances] = useState<Array<{ start: number; end: number; text: string; confidence: number; tokens: Array<{ text: string; start: number; end: number }> }>>([]);
  const [networkLines, setNetworkLines] = useState<Array<{ time: number; text: string; raw_time_label: string }>>([]);
  const [networkRomanizedText, setNetworkRomanizedText] = useState("");
  const [networkTranslatedText, setNetworkTranslatedText] = useState("");
  const [networkWordTimedText, setNetworkWordTimedText] = useState("");
  const [chunkInfo, setChunkInfo] = useState<string>("");
  const [correctionDone, setCorrectionDone] = useState(false);
  const [activeToken, setActiveToken] = useState("");

  const effectiveLyricsText = useMemo(() => {
    if (lyricsPreviewText) return lyricsPreviewText;
    const cand = lyricsPreviewCandidate as Record<string, unknown> | null;
    const rows = cand?.__saved_rows as Array<{ timeLabel: string; original: string }> | undefined;
    if (rows && rows.length > 0) {
      return rows.map((r) => r.timeLabel ? `[${r.timeLabel}]${r.original}` : r.original).join("\n");
    }
    return "";
  }, [lyricsPreviewText, lyricsPreviewCandidate]);

  useEffect(() => {
    if (!open) return;
    setConnected(false);
    setStreamText("");
    setCorrectedLines([]);
    setFunasrUtterances([]);
    setNetworkLines([]);
    setNetworkRomanizedText("");
    setNetworkTranslatedText("");
    setNetworkWordTimedText("");
    setChunkInfo("");
    setCorrectionDone(false);
    setActiveToken("");

    wtRef.current = lyricsPreviewWordTimedText || "";
    romaRef.current = lyricsPreviewRomanizedText || "";
    transRef.current = lyricsPreviewTranslatedText || "";
    lyricsRef.current = effectiveLyricsText || "";
    const cand = lyricsPreviewCandidate;
    artistRef.current = cand?.artist || "";
    titleRef.current = cand?.title || "";

    const socket = new WebSocket(buildTaskModuleACorrectFunasrSocketUrl(taskId));
    socketRef.current = socket;

    socket.onopen = () => {
      setConnected(true);
      appLogger.info("模块A矫正", "FunASR 矫正 WebSocket 已建立", { taskId });
      socket.send(JSON.stringify({
        event: "init_lyrics",
        data: {
          lyrics_text: lyricsRef.current,
          artist: artistRef.current,
          title: titleRef.current,
          word_timed_text: wtRef.current,
          romanized_text: romaRef.current,
          translated_text: transRef.current,
        },
      }));
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(String(event.data || "{}"));
        const eventName = String(payload.event || "");
        const data = (payload.data || {}) as Record<string, unknown>;
        if (eventName === "error") {
          message.warning(String(data.message || "矫正失败"));
          return;
        }
        if (eventName === "init") {
          setNetworkLines((data.network_lines || []) as typeof networkLines);
          setNetworkRomanizedText(String(data.romanized_text || ""));
          setNetworkTranslatedText(String(data.translated_text || ""));
          setNetworkWordTimedText(String(data.word_timed_text || ""));
          setFunasrUtterances((data.funasr_utterances || []) as typeof funasrUtterances);
          return;
        }
        if (eventName === "stream_chunk") {
          setStreamText((prev) => prev + String(data.text || ""));
          return;
        }
        if (eventName === "chunk_complete") {
          setChunkInfo(`第 ${Number(data.chunk_index) + 1}/${data.total_chunks} 块完成`);
          return;
        }
        if (eventName === "complete") {
          setCorrectedLines((data.corrected_lines || []) as typeof correctedLines);
          setCorrectionDone(true);
          setChunkInfo("矫正完成");
          return;
        }
      } catch { /* ignore */ }
    };

    socket.onclose = () => {
      setConnected(false);
      if (socketRef.current === socket) socketRef.current = null;
    };

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, [open, taskId, message, effectiveLyricsText, lyricsPreviewCandidate, lyricsPreviewWordTimedText, lyricsPreviewRomanizedText, lyricsPreviewTranslatedText]);

  const handleSave = () => {
    if (!correctedLines.length && !streamText) { message.info("没有可保存的矫正结果。"); return; }
    if (correctedLines.length === 0) { message.info("矫正尚未完成，请等待完成后再保存。"); return; }
    const savedId = "saved_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6);
    const romaLines = networkRomanizedText ? networkRomanizedText.split(/\r?\n/).filter(Boolean) : [];
    const transLines = networkTranslatedText ? networkTranslatedText.split(/\r?\n/).filter(Boolean) : [];
    const savedRows = correctedLines.map((cl, i) => ({
      timeLabel: cl.raw_time_label,
      original: cl.text,
      romanized: romaLines[i] || "",
      translated: transLines[i] || "",
      tokens: [] as Array<{ text: string; start: string; end: string }>,
    }));
    const candidate: TaskModuleALyricCandidate & Record<string, unknown> = {
      candidate_id: savedId,
      artist: lyricsPreviewCandidate?.artist || "",
      title: lyricsPreviewCandidate?.title || "",
      provider: "saved" as const,
      provider_id: savedId,
      has_word_timed_lyrics: Boolean(wtRef.current),
      has_translated_lyrics: Boolean(transLines.some(Boolean)),
      has_romanized_lyrics: Boolean(romaLines.some(Boolean)),
      preview_lines: correctedLines.slice(0, 4).map((cl) => cl.text),
      preview_text: correctedLines.map((cl) => cl.text).join("\n").slice(0, 200),
      __saved_at: Date.now(),
      __saved_word_timed: wtRef.current,
      __saved_rows: savedRows,
    };
    onSave(candidate);
    message.success("矫正结果已保存到「已保存」列。");
    onClose();
  };

  const renderTokenBtn = (token: { text: string; start: string | number; end: string | number }, prefix: string) => {
    const tk = `${prefix}`;
    const isActive = activeToken === tk;
    const timeLabel = typeof token.start === "number" ? fmtSec(token.start) : token.start;
    return (
      <Button key={tk} type="text" size="small" style={{ fontSize: 11, padding: "0 2px", height: 20, background: isActive ? "#e6f4ff" : undefined }}
        onClick={() => setActiveToken((c) => c === tk ? "" : tk)}>
        {token.text}
        {isActive ? <Typography.Text type="secondary" style={{ fontSize: 10, marginLeft: 2 }}>{timeLabel}</Typography.Text> : null}
      </Button>
    );
  };

  const buildTokenMap = (lines: Array<{ raw_time_label: string }>): Record<number, Array<{ text: string; start: string; end: string }>> => {
    const effectiveWt = lyricsPreviewWordTimedText || networkWordTimedText;
    if (!effectiveWt) return {};
    const wtLines = effectiveWt.split(/\r?\n/).filter(Boolean);
    const map: Record<number, Array<{ text: string; start: string; end: string }>> = {};
    // 预构建 lines 的时间标签 → 索引映射，带归一化去尾 0
    const timeToIdx: Record<string, number> = {};
    lines.forEach((l, i) => {
      const raw = l.raw_time_label || "";
      timeToIdx[raw] = i;
      // 去掉尾部多余的 0：00:22.690 → 00:22.69
      const normalized = raw.replace(/(\.\d+?)0+$/, "$1");
      if (normalized !== raw) timeToIdx[normalized] = i;
      const alt = raw.replace(/\.(\d)$/, ".$10");
      if (alt !== raw && !(alt in timeToIdx)) timeToIdx[alt] = i;
    });
    wtLines.forEach((line) => {
      const m = line.match(LRC_LINE_PATTERN);
      const timeLabel = m?.groups?.time || "";
      const content = m?.groups?.text || "";
      const tokens: Array<{ text: string; start: string; end: string }> = [];
      Array.from(content.matchAll(ENHANCED_TOKEN_PATTERN)).forEach((match) => {
        tokens.push({ start: match.groups?.start || "", end: match.groups?.end || "", text: match.groups?.text || "" });
      });
      if (tokens.length > 0) {
        const idx = timeToIdx[timeLabel];
        if (idx !== undefined && idx >= 0) map[idx] = tokens;
      }
    });
    return map;
  };
  const tokenMap = useMemo(() => buildTokenMap(networkLines), [networkLines, lyricsPreviewWordTimedText, networkWordTimedText]);

  return (
    <Modal title="LLM FunASR 矫正" open={open} onCancel={onClose}
      footer={[
        <Button key="close" onClick={onClose}>关闭</Button>,
        <Button key="save" type="primary" disabled={!correctionDone} onClick={handleSave}>保存</Button>,
        <Button key="resegment" disabled={!correctionDone} onClick={() => { handleSave(); onResegment(); }}>更新歌词和时间戳</Button>,
      ]}
      width={1400} destroyOnHidden>
      <Space direction="vertical" style={{ width: "100%" }} size={8}>
        <Tag color={connected ? "success" : "default"}>{connected ? "已连接" : "未连接"}</Tag>
        {chunkInfo ? <Tag color="processing">{chunkInfo}</Tag> : null}
      </Space>
      <div style={{ display: "flex", gap: 12, height: 520, marginTop: 12 }}>
        <div style={{ flex: 1, overflow: "auto", border: "1px solid #e8ecf0", borderRadius: 8, padding: 12 }}>
          <Typography.Title level={5}>网络歌词</Typography.Title>
          {(() => {
            const romaLines = networkRomanizedText.split(/\r?\n/).filter(Boolean);
            const transLines = networkTranslatedText.split(/\r?\n/).filter(Boolean);
            return networkLines.map((nl, i) => {
              const tokens = tokenMap[i];
              const romaText = romaLines[i] || "";
              const transText = transLines[i] || "";
              return (
              <div key={`net-${i}`} style={{ marginBottom: 4, padding: "2px 0", borderBottom: "1px solid #f0f0f0" }}>
                <Typography.Text type="secondary" style={{ fontSize: 11 }}>{nl.raw_time_label}</Typography.Text>
                <div style={{ fontSize: 13, display: "flex", flexWrap: "wrap", gap: 1 }}>
                  {tokens && tokens.length > 0 ? tokens.map((t, ti) => renderTokenBtn(t, `net-${i}-${ti}`)) : nl.text}
                </div>
                <div style={{ fontSize: 12, color: "#8c8c8c" }}>{[transText, romaText].filter(Boolean).join(" | ")}</div>
              </div>);
            });
          })()}
        </div>
        <div style={{ flex: 1, overflow: "auto", border: "1px solid #e8ecf0", borderRadius: 8, padding: 12 }}>
          <Typography.Title level={5}>LLM 矫正流式输出</Typography.Title>
          {correctedLines.length > 0 ? (() => {
            const romaLines = networkRomanizedText.split(/\r?\n/).filter(Boolean);
            const transLines = networkTranslatedText.split(/\r?\n/).filter(Boolean);
            return correctedLines.map((cl, i) => {
              const tokens = tokenMap[i];
              const romaText = romaLines[i] || "";
              const transText = transLines[i] || "";
              return (
              <div key={`cl-${i}`} style={{ marginBottom: 4, padding: "2px 0", borderBottom: "1px solid #f0f0f0" }}>
                <Typography.Text type="secondary" style={{ fontSize: 11 }}>{cl.raw_time_label}</Typography.Text>
                <div style={{ fontSize: 13, display: "flex", flexWrap: "wrap", gap: 1 }}>
                  {tokens?.length ? tokens.map((t, ti) => renderTokenBtn(t, `cl-${i}-${ti}`)) : cl.text}
                </div>
                <div style={{ fontSize: 12, color: "#8c8c8c" }}>{[transText, romaText].filter(Boolean).join(" | ")}</div>
              </div>);
            });
          })() : (
            <Typography.Text style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{streamText}</Typography.Text>
          )}
        </div>
        <div style={{ flex: 1, overflow: "auto", border: "1px solid #e8ecf0", borderRadius: 8, padding: 12 }}>
          <Typography.Title level={5}>FunASR 原始识别</Typography.Title>
          {funasrUtterances.map((fu, i) => (
            <div key={`fu-${i}`} style={{ marginBottom: 4, padding: "2px 0", borderBottom: "1px solid #f0f0f0" }}>
              <Typography.Text type="secondary" style={{ fontSize: 11 }}>{fmtSec(fu.start)} 置信度{fu.confidence.toFixed(2)}</Typography.Text>
              <div style={{ fontSize: 13, display: "flex", flexWrap: "wrap", gap: 1 }}>
                {fu.tokens?.length > 0 ? fu.tokens.map((t, ti) => renderTokenBtn(t, `fu-${i}-${ti}`)) : fu.text}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}

function fmtSec(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  return `${m.toString().padStart(2, "0")}:${(s % 60).toFixed(2).padStart(5, "0")}`;
}
