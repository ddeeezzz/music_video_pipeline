import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  ExportOutlined,
  GlobalOutlined,
  LeftOutlined,
  MinusCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  RightOutlined,
  SearchOutlined,
  SoundOutlined,
} from "@ant-design/icons";
import {
  Alert,
  App,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Empty,
  Input,
  Modal,
  Radio,
  Space,
  Tag,
  Typography,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  buildTaskModuleALyricsSearchSocketUrl,
  getTaskModuleALyricDetail,
  getTaskModuleAData,
  rerunTask,
  resetModuleAStatus,
  selectTaskModuleALyrics,
  taskQueryKeys,
} from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { ModuleAVisualization } from "@/features/moduleA/visualization/ModuleAVisualization";
import { CorrectFunasrModal } from "@/features/moduleA/CorrectFunasrModal";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import type {
  TaskModuleALyricCandidate,
  TaskModuleALyricDetail,
  TaskModuleALyricProviderGroup,
  TaskModuleAMetadataTrace,
} from "@/schemas/moduleA";

type SearchMode = "automatic" | "manual_query" | "fingerprint" | "metadata" | null;
type LyricTokenUnit = { text: string; start: string; end: string };
type LyricDisplayLine = { timeLabel: string; text: string };
type WordTimedDisplayLine = { timeLabel: string; text: string; tokens: LyricTokenUnit[] };
type LyricPreviewRow = {
  timeLabel: string;
  originalText: string;
  tokens: LyricTokenUnit[];
  translatedText: string;
  romanizedText: string;
};

const PROVIDER_COLUMN_WIDTH = 320;
const PROVIDER_COLUMN_GAP = 12;
const LYRICS_MODAL_MIN_WIDTH = 760;
const LYRICS_MODAL_MAX_WIDTH = 1480;
const LYRICS_MODAL_VIEWPORT_MARGIN = 32;
const LRC_LINE_PATTERN = /^\[(?<time>\d{2}:\d{2}(?:\.\d{1,3})?)\](?<text>.*)$/;
const ENHANCED_TOKEN_PATTERN = /<(?<start>\d{2}:\d{2}(?:\.\d{1,3})?)>(?<text>.*?)<(?<end>\d{2}:\d{2}(?:\.\d{1,3})?)>/g;

function getNetworkStatusTag(displayStatus: string) {
  if (displayStatus === "enabled") {
    return <Tag color="success">已启用</Tag>;
  }
  if (displayStatus === "searched_not_enabled") {
    return <Tag color="warning">已查找未启用</Tag>;
  }
  return <Tag>未启用</Tag>;
}

function buildCandidateLabel(candidate: TaskModuleALyricCandidate): string {
  const artist = candidate.artist.trim();
  const title = candidate.title.trim();
  return buildSongLabel(artist, title) || candidate.candidate_id;
}

function buildSongLabel(artist: string, title: string): string {
  const normalizedArtist = artist.trim();
  const normalizedTitle = title.trim();
  if (normalizedArtist && normalizedTitle) {
    return `${normalizedArtist} - ${normalizedTitle}`;
  }
  return normalizedArtist || normalizedTitle;
}

function formatDuration(seconds: number | undefined | null): string {
  if (!seconds || seconds <= 0) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function parseLyricDisplayLines(text: string): LyricDisplayLine[] {
  return text
    .split(/\r?\n/)
    .map((rawLine) => rawLine.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(LRC_LINE_PATTERN);
      if (!match?.groups) {
        return { timeLabel: "", text: line };
      }
      return {
        timeLabel: match.groups.time || "",
        text: (match.groups.text || "").trim(),
      };
    });
}

function buildLyricTimeKey(timeLabel: string, fallbackIndex: number): string {
  const normalizedTimeLabel = String(timeLabel || "").trim();
  if (normalizedTimeLabel) {
    return `time:${normalizedTimeLabel}`;
  }
  return `index:${fallbackIndex}`;
}

function parseWordTimedDisplayLines(text: string): WordTimedDisplayLine[] {
  return text
    .split(/\r?\n/)
    .map((rawLine) => rawLine.trim())
    .filter(Boolean)
    .map((line) => {
      const lineMatch = line.match(LRC_LINE_PATTERN);
      const timeLabel = lineMatch?.groups?.time || "";
      const content = (lineMatch?.groups?.text || line).trim();
      const tokens = Array.from(content.matchAll(ENHANCED_TOKEN_PATTERN)).map((match) => ({
        start: match.groups?.start || "",
        end: match.groups?.end || "",
        text: match.groups?.text || "",
      }));
      return {
        timeLabel,
        text: tokens.length ? tokens.map((token) => token.text).join("") : content,
        tokens,
      };
    });
}

function buildLyricPreviewRows(
  originalText: string,
  wordTimedText: string,
  translatedText: string,
  romanizedText: string,
): LyricPreviewRow[] {
  const wordTimedLines = parseWordTimedDisplayLines(wordTimedText);
  const originalLines = parseLyricDisplayLines(originalText);
  const translatedLines = parseLyricDisplayLines(translatedText);
  const romanizedLines = parseLyricDisplayLines(romanizedText);
  console.log("[buildLyricPreviewRows] originalLines[0]:", JSON.stringify(originalLines[0]), "originalLines.length:", originalLines.length);
  console.log("[buildLyricPreviewRows] translatedLines[0]:", JSON.stringify(translatedLines[0]));
  console.log("[buildLyricPreviewRows] romanizedLines[0]:", JSON.stringify(romanizedLines[0]));
  const baseLines = wordTimedLines.length
    ? wordTimedLines
    : originalLines.map((line) => ({
        timeLabel: line.timeLabel,
        text: line.text,
        tokens: [] as LyricTokenUnit[],
      }));
  const originalLineMap = new Map(originalLines.map((line, index) => [buildLyricTimeKey(line.timeLabel, index), line]));
  const translatedLineMap = new Map(translatedLines.map((line, index) => [buildLyricTimeKey(line.timeLabel, index), line]));
  const romanizedLineMap = new Map(romanizedLines.map((line, index) => [buildLyricTimeKey(line.timeLabel, index), line]));
  return baseLines.map((line, index) => {
    const timeKey = buildLyricTimeKey(line.timeLabel, index);
    const matchedOriginalLine = originalLineMap.get(timeKey);
    // 若按时间戳匹配失败（如纯文本翻译），则回退到按行号 index 匹配
    const matchedTranslatedLine = translatedLineMap.get(timeKey) || translatedLineMap.get(`index:${index}`);
    const matchedRomanizedLine = romanizedLineMap.get(timeKey) || romanizedLineMap.get(`index:${index}`);
    return {
      timeLabel: line.timeLabel || matchedOriginalLine?.timeLabel || "",
      originalText: matchedOriginalLine?.text || line.text || "",
      tokens: line.tokens,
      translatedText: matchedTranslatedLine?.text || "",
      romanizedText: matchedRomanizedLine?.text || "",
    };
  });
}

function buildTokenTimestampText(token: LyricTokenUnit): string {
  return `${token.start} - ${token.end}`;
}

const LRC_TIME_PATTERN = /\[\d{2}:\d{2}(?:\.\d{1,3})?\]/g;
const NON_WORD_PATTERN = /[\s\-_/|:：·,，.。!！?？'"]+/g;

function stripLrcTimestamps(text: string): string {
  return text.replace(LRC_TIME_PATTERN, "").trim();
}

function normalizeLyricsForCompare(text: string): string {
  return stripLrcTimestamps(text).replace(NON_WORD_PATTERN, "").toLowerCase();
}

function computeLyricsSimilarity(textA: string, textB: string): number {
  const cleanA = normalizeLyricsForCompare(textA);
  const cleanB = normalizeLyricsForCompare(textB);
  if (!cleanA || !cleanB || cleanA.length < 10 || cleanB.length < 10) return 0;
  const setA = new Set(cleanA);
  const setB = new Set(cleanB);
  let common = 0;
  for (const c of setA) { if (setB.has(c)) common++; }
  return Math.round((2 * common * 100) / (setA.size + setB.size));
}

const MISSING_FIELD_LABELS = {
  word_timed_lyrics: "逐字时间戳",
  translated_lyrics: "翻译",
  romanized_lyrics: "罗马音",
};

function getMissingFieldLabels(candidate: TaskModuleALyricCandidate | null): string[] {
  if (!candidate) return [];
  const labels: string[] = [];
  if (!candidate.has_word_timed_lyrics) labels.push(MISSING_FIELD_LABELS.word_timed_lyrics);
  if (!candidate.has_translated_lyrics) labels.push(MISSING_FIELD_LABELS.translated_lyrics);
  if (!candidate.has_romanized_lyrics) labels.push(MISSING_FIELD_LABELS.romanized_lyrics);
  return labels;
}

function getAvailableMergeFields(detail: { word_timed_lyrics: string; translated_lyrics: string; romanized_lyrics: string } | null, missingLabels: string[]): string[] {
  if (!detail) return [];
  const map: Record<string, keyof typeof detail> = { "逐字时间戳": "word_timed_lyrics", "翻译": "translated_lyrics", "罗马音": "romanized_lyrics" };
  return missingLabels.filter((label) => { const key = map[label]; return key && Boolean(String(detail[key] || "").trim()); });
}

function renderEmbeddedMetadata(trace?: TaskModuleAMetadataTrace | null): ReactNode {
  if (!trace || (!trace.embedded_status && !trace.embedded_artist && !trace.embedded_title && !trace.embedded_error)) {
    return "尚未联网查找";
  }
  const songLabel = buildSongLabel(trace.embedded_artist, trace.embedded_title);
  if (trace.embedded_status === "ok") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag color="success">已从文件读取</Tag>
        <Typography.Text>{songLabel || "已读取到元信息"}</Typography.Text>
        {trace.embedded_album ? <Typography.Text type="secondary">专辑：{trace.embedded_album}</Typography.Text> : null}
      </Space>
    );
  }
  if (trace.embedded_status === "missing") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag>文件未提供完整元信息</Tag>
        <Typography.Text type="secondary">{songLabel || trace.embedded_error || "artist/title 不完整"}</Typography.Text>
      </Space>
    );
  }
  if (trace.embedded_status === "failed") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag color="error">文件元信息读取失败</Tag>
        <Typography.Text type="secondary">{trace.embedded_error || "未提供错误信息"}</Typography.Text>
      </Space>
    );
  }
  if (trace.embedded_status === "skipped") {
    return <Tag>未读取文件元信息</Tag>;
  }
  return <Typography.Text type="secondary">{trace.embedded_status}</Typography.Text>;
}

function renderFingerprintMetadata(trace?: TaskModuleAMetadataTrace | null): ReactNode {
  if (!trace || (!trace.fingerprint_status && !trace.acoustid_status && !trace.matched_artist && !trace.matched_title)) {
    return "尚未联网查找";
  }
  const matchedLabel = buildSongLabel(trace.matched_artist, trace.matched_title);
  if (trace.acoustid_status === "ok") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag color="success">指纹已匹配曲目</Tag>
        <Typography.Text>{matchedLabel || "已匹配到曲目信息"}</Typography.Text>
      </Space>
    );
  }
  if (trace.fingerprint_status === "failed") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag color="error">指纹生成失败</Tag>
        <Typography.Text type="secondary">{trace.fingerprint_error || "未提供错误信息"}</Typography.Text>
      </Space>
    );
  }
  if (trace.acoustid_status === "failed") {
    return (
      <Space wrap size={[8, 8]}>
        <Tag color="error">指纹曲目识别失败</Tag>
        <Typography.Text type="secondary">{trace.matched_error || "未提供错误信息"}</Typography.Text>
      </Space>
    );
  }
  if (trace.acoustid_status === "no_match" || trace.acoustid_status === "not_found") {
    return <Tag>指纹未匹配到曲目信息</Tag>;
  }
  if (trace.fingerprint_status === "skipped" || trace.acoustid_status === "skipped") {
    return <Tag>未执行指纹识别</Tag>;
  }
  if (trace.fingerprint_status === "ok") {
    return <Tag color="processing">指纹已生成，未拿到曲目信息</Tag>;
  }
  return <Typography.Text type="secondary">{trace.acoustid_status || trace.fingerprint_status}</Typography.Text>;
}

function sortProviderGroups(groups: TaskModuleALyricProviderGroup[]): TaskModuleALyricProviderGroup[] {
  return [...groups]
    .filter((group) => group.candidates.length > 0)
    .sort((left, right) => {
      const leftAt = left.first_result_at_ms ?? Number.MAX_SAFE_INTEGER;
      const rightAt = right.first_result_at_ms ?? Number.MAX_SAFE_INTEGER;
      if (leftAt !== rightAt) {
        return leftAt - rightAt;
      }
      return left.display_name.localeCompare(right.display_name, "zh-CN");
    });
}

function buildLockedProviderOrder(groups: TaskModuleALyricProviderGroup[]): string[] {
  return sortProviderGroups(groups).map((group) => group.provider);
}

function mergeProviderOrder(
  currentOrder: string[],
  incomingGroups: TaskModuleALyricProviderGroup | TaskModuleALyricProviderGroup[],
): string[] {
  const normalizedIncomingGroups = Array.isArray(incomingGroups) ? incomingGroups : [incomingGroups];
  let nextOrder = [...currentOrder];
  for (const incomingGroup of normalizedIncomingGroups) {
    if (!incomingGroup.provider || !incomingGroup.candidates.length || nextOrder.includes(incomingGroup.provider)) {
      continue;
    }
    nextOrder = [...nextOrder, incomingGroup.provider];
  }
  return nextOrder;
}

function orderProviderGroups(groups: TaskModuleALyricProviderGroup[], providerOrder: string[]): TaskModuleALyricProviderGroup[] {
  const nonEmptyGroups = groups.filter((group) => group.candidates.length > 0);
  const groupMap = new Map(nonEmptyGroups.map((group) => [group.provider, group]));
  const orderedGroups = providerOrder.map((provider) => groupMap.get(provider)).filter(Boolean) as TaskModuleALyricProviderGroup[];
  const tailGroups = nonEmptyGroups.filter((group) => !providerOrder.includes(group.provider));
  return [...orderedGroups, ...tailGroups];
}

function flattenCandidates(groups: TaskModuleALyricProviderGroup[]): TaskModuleALyricCandidate[] {
  return groups.flatMap((group) => group.candidates);
}

function mergeProviderGroup(
  existingGroups: TaskModuleALyricProviderGroup[],
  incomingGroup: TaskModuleALyricProviderGroup,
): TaskModuleALyricProviderGroup[] {
  const existingGroup = existingGroups.find((group) => group.provider === incomingGroup.provider) || null;
  const mergedGroup: TaskModuleALyricProviderGroup = {
    ...incomingGroup,
    first_result_at_ms: incomingGroup.first_result_at_ms ?? existingGroup?.first_result_at_ms ?? null,
  };
  const nextGroups = existingGroups.filter((group) => group.provider !== incomingGroup.provider);
  nextGroups.push(mergedGroup);
  return nextGroups;
}

function readProviderPage(providerPages: Record<string, number>, provider: string): number {
  return Math.max(0, providerPages[provider] ?? 0);
}

function shouldLoadProviderPage(providerGroup: TaskModuleALyricProviderGroup, pageIndex: number): boolean {
  const pageSize = providerGroup.page_size || 10;
  const requiredCount = (pageIndex + 1) * pageSize;
  const totalCount = providerGroup.total_count || providerGroup.candidates.length;
  return requiredCount > providerGroup.candidates.length && providerGroup.candidates.length < totalCount;
}

export function TaskModuleAPage() {
  const taskId = useTaskIdParam();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const searchSocketRef = useRef<WebSocket | null>(null);
  const [lyricsModalOpen, setLyricsModalOpen] = useState(false);
  const [confirmModalOpen, setConfirmModalOpen] = useState(false);
  const [manualSearchModalOpen, setManualSearchModalOpen] = useState(false);
  const [manualSearchArtist, setManualSearchArtist] = useState("");
  const [manualSearchTitle, setManualSearchTitle] = useState("");
  const [providerGroups, setProviderGroups] = useState<TaskModuleALyricProviderGroup[]>([]);
  const [providerOrder, setProviderOrder] = useState<string[]>([]);
  const [providerPages, setProviderPages] = useState<Record<string, number>>({});
  const [selectedCandidateId, setSelectedCandidateId] = useState("");
  const [lyricsPreviewCandidate, setLyricsPreviewCandidate] = useState<TaskModuleALyricCandidate | null>(null);
  const [lyricsPreviewText, setLyricsPreviewText] = useState("");
  const [lyricsPreviewWordTimedText, setLyricsPreviewWordTimedText] = useState("");
  const [lyricsPreviewTranslatedText, setLyricsPreviewTranslatedText] = useState("");
  const [lyricsPreviewRomanizedText, setLyricsPreviewRomanizedText] = useState("");
  const [showWordTimedOverlay, setShowWordTimedOverlay] = useState(true);
  const [showTranslatedLyrics, setShowTranslatedLyrics] = useState(true);
  const [showRomanizedLyrics, setShowRomanizedLyrics] = useState(true);
  const [activeTokenKey, setActiveTokenKey] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchMode, setSearchMode] = useState<SearchMode>(null);
  const [showCachedResults, setShowCachedResults] = useState(false);
  const [viewportWidth, setViewportWidth] = useState<number>(() => window.innerWidth);
  const [viewportHeight, setViewportHeight] = useState<number>(() => window.innerHeight);
  const hasUserPickedCandidateRef = useRef(false);
  const [rowIndices, setRowIndices] = useState<number[][]>([]);
  const [savedCandidates, setSavedCandidates] = useState<TaskModuleALyricCandidate[]>([]);
  const [correctFunasrOpen, setCorrectFunasrOpen] = useState(false);
  const [mergeSourceModalOpen, setMergeSourceModalOpen] = useState(false);
  const [mergeFieldSelectOpen, setMergeFieldSelectOpen] = useState(false);
  const [mergeInspectedCandidate, setMergeInspectedCandidate] = useState<TaskModuleALyricCandidate | null>(null);
  const [mergeInspectedSynced, setMergeInspectedSynced] = useState("");
  const [mergeInspectedWordTimed, setMergeInspectedWordTimed] = useState("");
  const [mergeInspectedTranslated, setMergeInspectedTranslated] = useState("");
  const [mergeInspectedRomanized, setMergeInspectedRomanized] = useState("");
  const [mergeInspecting, setMergeInspecting] = useState(false);
  const [mergeSimilarity, setMergeSimilarity] = useState(0);
  const [mergeInspectedAvailable, setMergeInspectedAvailable] = useState<string[]>([]);
  const [mergeSelectedFields, setMergeSelectedFields] = useState<Set<string>>(new Set());
  const [mergeMissingFields, setMergeMissingFields] = useState<string[]>([]);
  const [compareModalOpen, setCompareModalOpen] = useState(false);
  const [mergeCompareRows, setMergeCompareRows] = useState<Array<{timeLabel:string;original:string;romanized:string;translated:string;tokens:LyricTokenUnit[]}>>([]);
  const [candTransOffset, setCandTransOffset] = useState(0);
  const [savedRowsData, setSavedRowsData] = useState<Array<{ timeLabel: string; original: string; romanized: string; translated: string; tokens: LyricTokenUnit[] }> | null>(null);
  const [previewRows, setPreviewRows] = useState<Array<{timeLabel:string;originalText:string;romanizedText:string;translatedText:string;tokens:LyricTokenUnit[]}>>([]);

  useEffect(() => {
    appLogger.info("模块A页面", "模块 A 可视化页已进入", { taskId });
  }, [taskId]);

  useEffect(() => {
    setProviderGroups([]);
    setProviderOrder([]);
    setProviderPages({});
    setSelectedCandidateId("");
    setSearching(false);
    setSearchMode(null);
    setShowCachedResults(false);
    setLyricsPreviewCandidate(null);
    setLyricsPreviewText("");
    setLyricsPreviewWordTimedText("");
    setLyricsPreviewTranslatedText("");
    setLyricsPreviewRomanizedText("");
    setShowWordTimedOverlay(true);
    setShowTranslatedLyrics(true);
    setShowRomanizedLyrics(true);
    setActiveTokenKey("");
    hasUserPickedCandidateRef.current = false;
  }, [taskId]);

  useEffect(() => {
    return () => {
      if (searchSocketRef.current) {
        searchSocketRef.current.close();
        searchSocketRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
      setViewportHeight(window.innerHeight);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  const { data, isLoading, refetch, isFetching, error } = useQuery({
    queryKey: taskQueryKeys.moduleA(taskId),
    queryFn: () => getTaskModuleAData(taskId),
    enabled: Boolean(taskId),
  });
  const queryErrorText = error instanceof Error ? error.message : "";

  const invalidateTaskScopes = async () => {
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.snapshot(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.webData(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleA(taskId) });
  };

  const ensureProviderPageLoaded = async (providerName: string, pageIndex: number) => {
    const currentProviderGroup = providerGroups.find((group) => group.provider === providerName);
    if (!currentProviderGroup || !shouldLoadProviderPage(currentProviderGroup, pageIndex)) {
      return;
    }
    const latestModuleAData = await queryClient.fetchQuery({
      queryKey: taskQueryKeys.moduleA(taskId),
      queryFn: () => getTaskModuleAData(taskId),
    });
    const nextProviderGroup = latestModuleAData.network_lrc_state.provider_groups.find(
      (group) => group.provider === providerName,
    );
    if (!nextProviderGroup) {
      return;
    }
    setProviderGroups((current) => mergeProviderGroup(current, nextProviderGroup));
  };

  const rerunMutation = useMutation({
    mutationFn: (force?: boolean) => rerunTask(taskId, force ?? false),
    onSuccess: async (payload) => {
      await invalidateTaskScopes();
      message.success(payload.message || "模块 A 重跑请求已提交");
    },
    onError: (mutationError) => {
      const errorText = mutationError instanceof Error ? mutationError.message : String(mutationError);
      appLogger.warn("模块A页面", "模块 A 重跑入口反馈", { taskId, error: errorText });
      message.warning(errorText);
    },
  });

  const resetStatusMutation = useMutation({
    mutationFn: (status?: string) => resetModuleAStatus(taskId, status ?? "pending"),
    onSuccess: async () => {
      await invalidateTaskScopes();
      message.success("模块 A 状态已重置");
    },
    onError: (mutationError) => {
      const errorText = mutationError instanceof Error ? mutationError.message : String(mutationError);
      appLogger.warn("模块A页面", "重置模块 A 状态失败", { taskId, error: errorText });
      message.warning(errorText);
    },
  });

  const handleRerunClick = () => {
    const moduleAStatus = data?.module_a_status;
    if (moduleAStatus === "running") {
      Modal.confirm({
        title: "确认强制重跑",
        content: `当前模块 A 状态为 running，是否强行重跑？`,
        okText: "强行重跑",
        cancelText: "取消",
        onOk: () => rerunMutation.mutate(true),
      });
    } else {
      rerunMutation.mutate(false);
    }
  };

  const instrumentalRerunMutation = useMutation({
    mutationFn: () => selectTaskModuleALyrics(taskId, "instrumental", true, undefined, undefined, undefined, undefined, undefined, true),
    onSuccess: async (payload) => {
      await invalidateTaskScopes();
      message.success(payload.message || "纯音乐模式已启用，开始重跑模块A");
    },
    onError: (mutationError) => {
      const errorText = mutationError instanceof Error ? mutationError.message : String(mutationError);
      appLogger.warn("模块A页面", "纯音乐模式重跑失败", { taskId, error: errorText });
      message.warning(errorText);
    },
  });

  const handleInstrumentalRerun = () => {
    Modal.confirm({
      title: "确认纯音乐重跑",
      content: "将创建空歌词并重跑模块A（纯音乐/无歌词模式）。已选的歌词候选将被覆盖，确定继续？",
      okText: "确定",
      cancelText: "取消",
      onOk: () => instrumentalRerunMutation.mutate(),
    });
  };

  const handleResetModuleAStatus = () => {
    Modal.confirm({
      title: "确认重置模块 A 状态",
      content: "将模块 A 状态重置为 pending，确定继续？",
      okText: "确定重置",
      cancelText: "取消",
      onOk: () => resetStatusMutation.mutate("pending"),
    });
  };

  const selectLyricsMutation = useMutation({
    mutationFn: ({ candidateId, enable, rerunMode, lyricsText, artist, title, wordTimedLyrics }: { candidateId: string; enable: boolean; rerunMode?: string; lyricsText?: string; artist?: string; title?: string; wordTimedLyrics?: string }) =>
      selectTaskModuleALyrics(taskId, candidateId, enable, rerunMode, lyricsText, artist, title, wordTimedLyrics),
    onSuccess: async (payload, variables) => {
      await invalidateTaskScopes();
      setConfirmModalOpen(false);
      if (variables.enable) {
        setLyricsModalOpen(false);
      }
      message.success(payload.message || (variables.enable ? "模块 A 已按选中歌词开始重跑" : "已联网查找lrc但未启用"));
      // 轻量重跑：轮询等待子进程完成，显示完成横幅
      if (variables.enable && variables.rerunMode === "lyrics_only") {
        const pollInterval = 2000;
        const maxPolls = 15; // 最长等待 30 秒
        let pollCount = 0;
        const pollTimer = setInterval(async () => {
          pollCount++;
          try {
            const moduleAData = await getTaskModuleAData(taskId);
            if (moduleAData.module_a_status === "done" || moduleAData.task_status === "done") {
              clearInterval(pollTimer);
              await invalidateTaskScopes();
              message.success("轻量重跑完成，模块 A 分段已更新");
            } else if (pollCount >= maxPolls) {
              clearInterval(pollTimer);
            }
          } catch {
            if (pollCount >= maxPolls) {
              clearInterval(pollTimer);
            }
          }
        }, pollInterval);
      }
    },
    onError: async (mutationError) => {
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleA(taskId) });
      const errorText = mutationError instanceof Error ? mutationError.message : String(mutationError);
      appLogger.warn("模块A页面", "联网歌词启用反馈", { taskId, error: errorText });
      message.warning(errorText);
    },
  });

  const lyricDetailMutation = useMutation({
    mutationFn: ({ candidateId }: { candidateId: string }) => getTaskModuleALyricDetail(taskId, candidateId),
    onSuccess: (payload: TaskModuleALyricDetail) => {
      setLyricsPreviewCandidate(payload.candidate);
      setLyricsPreviewText(payload.synced_lyrics);
      setLyricsPreviewWordTimedText(payload.word_timed_lyrics);
      setLyricsPreviewTranslatedText(payload.translated_lyrics);
      setLyricsPreviewRomanizedText(payload.romanized_lyrics);
      // 合并三轨 LRC 为统一 previewRows
      const merged = buildLyricPreviewRows(payload.synced_lyrics, payload.word_timed_lyrics, payload.translated_lyrics, payload.romanized_lyrics);
      setPreviewRows(merged);
      const m = getMissingFieldLabels(payload.candidate).filter((label) => {
        if (label === "逐字时间戳" && !payload.word_timed_lyrics) return true;
        if (label === "翻译" && !payload.translated_lyrics) return true;
        if (label === "罗马音" && !payload.romanized_lyrics) return true;
        return false;
      });
      setMergeMissingFields(m);
    },
    onError: (mutationError) => {
      const errorText = mutationError instanceof Error ? mutationError.message : String(mutationError);
      setLyricsPreviewText("");
      setLyricsPreviewWordTimedText("");
      setLyricsPreviewTranslatedText("");
      setLyricsPreviewRomanizedText("");
      message.warning(errorText);
    },
  });

  // lyricsPreviewText 变化时重建 rowIndices
  useEffect(() => {
    if (lyricsPreviewText) {
      const lines = lyricsPreviewText.split(/\r?\n/).filter(Boolean);
      const lrcLines = lines.filter((l) => LRC_LINE_PATTERN.test(l));
      const n = lrcLines.length;
      const idx: number[][] = [[n, n, n, n], ...lrcLines.map((_, i) => [i, i, i, i])];
      setRowIndices(idx);
    }
  }, [lyricsPreviewText]);

  const networkState = data?.network_lrc_state;
  const cachedProviderGroups = useMemo(
    () => sortProviderGroups(networkState?.provider_groups || []),
    [networkState?.provider_groups],
  );
  const savedProviderGroup: TaskModuleALyricProviderGroup | null = useMemo(() => {
    const sorted = [...savedCandidates].sort((a, b) => {
      const ta = (a as Record<string, unknown>).__saved_at as number || 0;
      const tb = (b as Record<string, unknown>).__saved_at as number || 0;
      return tb - ta;
    });
    return sorted.length > 0 ? {
      provider: "saved", display_name: "已保存", candidates: sorted,
      first_result_at_ms: null, page_size: 20, total_count: sorted.length, has_more: false,
    } : null;
  }, [savedCandidates]);
  const hasCachedLyricsResults = Boolean((networkState?.cached_candidates_count || 0) > 0 && cachedProviderGroups.length);
  const displayProviderGroups = (() => {
    const groups = providerGroups.length
      ? orderProviderGroups(providerGroups, providerOrder)
      : showCachedResults
        ? orderProviderGroups(cachedProviderGroups, providerOrder)
        : [];
    if (savedProviderGroup && !groups.find((g) => g.provider === "saved")) {
      groups.push(savedProviderGroup);
    }
    return groups;
  })();
  const displayCandidates = useMemo(() => flattenCandidates(displayProviderGroups), [displayProviderGroups]);
  const selectedCandidate = useMemo(() => {
    return displayCandidates.find((item) => item.candidate_id === selectedCandidateId) || null;
  }, [displayCandidates, selectedCandidateId]);
  const lyricsModalWidth = useMemo(() => {
    const providerCount = Math.max(1, displayProviderGroups.length);
    const laneWidth = providerCount * PROVIDER_COLUMN_WIDTH + Math.max(0, providerCount - 1) * PROVIDER_COLUMN_GAP;
    const desiredWidth = Math.min(LYRICS_MODAL_MAX_WIDTH, Math.max(LYRICS_MODAL_MIN_WIDTH, laneWidth + 96));
    const maxViewportWidth = Math.max(360, viewportWidth - LYRICS_MODAL_VIEWPORT_MARGIN);
    return Math.min(desiredWidth, maxViewportWidth);
  }, [displayProviderGroups.length, viewportWidth]);
  const lyricsModalBodyHeight = useMemo(() => Math.max(320, viewportHeight - 260), [viewportHeight]);

  useEffect(() => {
    setShowWordTimedOverlay(Boolean(lyricsPreviewWordTimedText));
    setShowTranslatedLyrics(Boolean(lyricsPreviewTranslatedText));
    setShowRomanizedLyrics(Boolean(lyricsPreviewRomanizedText));
    setActiveTokenKey("");
  }, [lyricsPreviewRomanizedText, lyricsPreviewTranslatedText, lyricsPreviewWordTimedText, lyricsPreviewCandidate]);

  const openVisualization = () => {
    if (!data?.module_a_visualization.available || !data.module_a_visualization.url) {
      message.info("当前任务还没有模块 A 可视化 HTML 产物。");
      return;
    }
    window.open(data.module_a_visualization.url, "_blank", "noopener,noreferrer");
  };

  const closeActiveSearchSocket = () => {
    if (searchSocketRef.current) {
      searchSocketRef.current.close();
      searchSocketRef.current = null;
    }
  };

  const openCachedLyricsResults = () => {
    closeActiveSearchSocket();
    setProviderGroups([]);
    setProviderPages({});
    setSearching(false);
    setSearchMode("automatic");
    setShowCachedResults(true);
    hasUserPickedCandidateRef.current = false;
    setLyricsModalOpen(true);
    setConfirmModalOpen(false);
    setManualSearchModalOpen(false);
    setProviderOrder(buildLockedProviderOrder(cachedProviderGroups));
    setSelectedCandidateId((current) => current || cachedProviderGroups[0]?.candidates[0]?.candidate_id || "");
  };

  const startLyricsSearch = (options?: { manualQuery?: string; manualArtist?: string; manualTitle?: string }) => {
    closeActiveSearchSocket();
    setProviderGroups([]);
    setProviderOrder([]);
    setProviderPages({});
    setSelectedCandidateId("");
    setLyricsPreviewCandidate(null);
    setLyricsPreviewText("");
    setLyricsPreviewWordTimedText("");
    setLyricsPreviewTranslatedText("");
    setLyricsPreviewRomanizedText("");
    setShowCachedResults(false);
    hasUserPickedCandidateRef.current = false;
    setSearching(true);
    setLyricsModalOpen(true);
    setConfirmModalOpen(false);
    setManualSearchModalOpen(false);
    const socketUrl = buildTaskModuleALyricsSearchSocketUrl(taskId, options);
    const socket = new WebSocket(socketUrl);
    searchSocketRef.current = socket;

    socket.onopen = () => {
      appLogger.info("模块A页面", "模块 A 歌词搜索流已建立", { taskId, socketUrl });
    };

    socket.onmessage = (event) => {
      let payload: unknown = {};
      try {
        payload = JSON.parse(String(event.data || "{}"));
      } catch (err) {
        appLogger.error("模块A页面", "WebSocket 消息 JSON 解析失败", { error: String(err) });
        return;
      }
      const parsedPayload = payload as { event?: string; data?: Record<string, unknown> };
      const eventName = String(parsedPayload.event || "").trim();
      const eventData = (parsedPayload.data || {}) as Record<string, unknown>;
      if (eventName === "search_started") {
        setSearchMode((String(eventData.search_mode || "").trim() as SearchMode) || "automatic");
        return;
      }
      if (eventName === "stage") {
        return;
      }
      if (eventName === "provider_group") {
        const nextGroup = eventData as unknown as TaskModuleALyricProviderGroup;
        if (!nextGroup.provider) {
          return;
        }
        setProviderGroups((current) => mergeProviderGroup(current, nextGroup));
        setProviderOrder((current) => mergeProviderOrder(current, nextGroup));
        setProviderPages((current) => ({ ...current, [nextGroup.provider]: current[nextGroup.provider] ?? 0 }));
        setSelectedCandidateId((current) => current || nextGroup.candidates[0]?.candidate_id || "");
        return;
      }
      if (eventName === "complete") {
        const nextProviderGroups = ((eventData.provider_groups || []) as TaskModuleALyricProviderGroup[]) || [];
        const finalCandidates = ((eventData.candidates || []) as TaskModuleALyricCandidate[]) || [];
        const flattenedCandidates = finalCandidates.length ? finalCandidates : flattenCandidates(nextProviderGroups);
        setProviderGroups((current) => nextProviderGroups.reduce(mergeProviderGroup, current));
        setProviderOrder((current) => mergeProviderOrder(current, sortProviderGroups(nextProviderGroups)));
        setSearchMode((String(eventData.search_mode || "").trim() as SearchMode) || "automatic");
        setSelectedCandidateId((current) => {
          if (hasUserPickedCandidateRef.current) {
            return current;
          }
          return flattenedCandidates[0]?.candidate_id || current || "";
        });
        setSearching(false);
        // 先确保 providerGroups 已更新，再异步刷新 REST 缓存
        if (!flattenedCandidates.length && eventData.suggest_manual_query) {
          setManualSearchModalOpen(true);
        }
        // 延迟刷新 REST API 数据，避免覆盖正在显示的搜索结果
        setTimeout(() => { invalidateTaskScopes(); }, 100);
        return;
      }
      if (eventName === "error") {
        const nextError = String(eventData.message || "联网歌词搜索失败").trim();
        setSearching(false);
        message.warning(nextError);
      }
    };

    socket.onerror = () => {
      setSearching(false);
      appLogger.error("模块A页面", "模块 A 歌词搜索流发生错误", { taskId });
    };

    socket.onclose = () => {
      if (searchSocketRef.current === socket) {
        searchSocketRef.current = null;
      }
      setSearching(false);
    };
  };

  const openLyricsSelectionConfirm = () => {
    if (!selectedCandidateId) {
      message.info("请先选择一份歌词候选。");
      return;
    }
    setLyricsModalOpen(false);
    setConfirmModalOpen(true);
  };

  const submitSelectedLyrics = (enable: boolean, rerunMode?: string) => {
    if (!selectedCandidateId) {
      message.info("请先选择一份歌词候选。");
      return;
    }
    const selCand = displayCandidates.find((c) => c.candidate_id === selectedCandidateId);
    const isSaved = selCand?.provider === "saved";
    let lyricsText: string | undefined;
    let wordTimedLyrics: string | undefined;
    if (isSaved) {
      const _sd = selCand as Record<string, unknown>;
      const _r = _sd.__saved_rows as Array<{ timeLabel: string; original: string; tokens?: { text: string; start: string; end: string }[] }> | undefined;
      if (_r && _r.length > 0) {
        lyricsText = _r.map((x) => x.timeLabel ? "[" + x.timeLabel + "]" + x.original : x.original).join("\n");
      }
      // 已保存候选由后端直接读文件，不传 word_timed_lyrics 避免 URL 超长
    }
    selectLyricsMutation.mutate({
      candidateId: selectedCandidateId, enable, rerunMode,
      lyricsText, wordTimedLyrics,
      artist: selCand?.artist, title: selCand?.title,
    });
  };

  const openLyricsPreview = (candidate: TaskModuleALyricCandidate) => {
    if (!candidate || !candidate.candidate_id) {
      appLogger.warn("模块A页面", "openLyricsPreview candidate_id 为空", { candidate });
      message.warning("该候选数据异常，请重新搜索。");
      return;
    }
    hasUserPickedCandidateRef.current = true;
    setSelectedCandidateId(candidate.candidate_id);
    setLyricsPreviewCandidate(candidate);
    setLyricsPreviewText("");
    setLyricsPreviewWordTimedText("");
    setLyricsPreviewTranslatedText("");
    setLyricsPreviewRomanizedText("");
    setShowRomanizedLyrics(true);
    setRowIndices([]);
    const _sd = candidate as Record<string, unknown>;
    if (_sd.__saved_rows) {
      const _r = _sd.__saved_rows as Array<{ timeLabel: string; original: string; romanized: string; translated: string; tokens: LyricTokenUnit[] }>;
      console.log("[打开已保存] _r[0]:", JSON.stringify(_r[0]));
      console.log("[打开已保存] _r.length:", _r.length);
      // 如果有词级时间戳文本，重新解析 tokens 回填到每行
      setSavedRowsData(_r);
      setPreviewRows(_r.map((x) => ({timeLabel:x.timeLabel,originalText:x.original,romanizedText:x.romanized,translatedText:x.translated,tokens:x.tokens})));
      // 同步重建 rowIndices：savedRows 有多少行，rowIndices 就重建多少
      const syncedText = _r.map((x) => x.timeLabel ? "[" + x.timeLabel + "]" + x.original : x.original).join("\n");
      const syncedLines = syncedText.split(/\r?\n/).filter(Boolean).filter((l) => LRC_LINE_PATTERN.test(l));
      console.log("[打开已保存] syncedText第一行:", syncedText.split("\n")[0]);
      console.log("[打开已保存] syncedLines.length:", syncedLines.length);
      console.log("[打开已保存] syncedLines[0]:", syncedLines[0]);
      setRowIndices([[syncedLines.length, syncedLines.length, syncedLines.length, syncedLines.length], ...syncedLines.map((_, i) => [i, i, i, i])]);
      setLyricsPreviewText(syncedText);
      setLyricsPreviewWordTimedText(String(_sd.__saved_word_timed || ""));
      const rawWordTimedForDisplay = String(_sd.__saved_word_timed || "").trim();
      const transText = _r.map((x) => x.translated ? (x.timeLabel ? `[${x.timeLabel}]${x.translated}` : x.translated) : "").join("\n");
      const romaText = _r.map((x) => x.romanized ? (x.timeLabel ? `[${x.timeLabel}]${x.romanized}` : x.romanized) : "").join("\n");
      console.log("[打开已保存] transText第一行:", transText.split("\n")[0]);
      console.log("[打开已保存] romaText第一行:", romaText.split("\n")[0]);
      setLyricsPreviewTranslatedText(transText);
      setLyricsPreviewRomanizedText(romaText);
      // 词级时间戳直接走 tokens 不再依赖 __saved_word_timed
      if (!rawWordTimedForDisplay || rawWordTimedForDisplay === "tokens_present") {
        setLyricsPreviewWordTimedText(_r.some((x) => x.tokens.length > 0) ? "tokens_present" : "");
      }
      return;
    }
    setSavedRowsData(null);
    lyricDetailMutation.mutate({ candidateId: candidate.candidate_id });
  };

  const startAutomaticLyricsSearch = () => {
    if (hasCachedLyricsResults) {
      openCachedLyricsResults();
      return;
    }
    startLyricsSearch({ manualQuery: "" });
  };

  const openManualLyricsSearchModal = () => {
    setManualSearchModalOpen(true);
  };

  const submitManualLyricsSearch = () => {
    const normalizedManualSearchArtist = manualSearchArtist.trim();
    const normalizedManualSearchTitle = manualSearchTitle.trim();
    if (!normalizedManualSearchTitle) {
      message.info("请先输入歌名。");
      return;
    }
    startLyricsSearch({
      manualArtist: normalizedManualSearchArtist,
      manualTitle: normalizedManualSearchTitle,
    });
  };

  const openMergeSourceModal = () => { setMergeFieldSelectOpen(true); };

  const confirmMergeFieldSelection = () => {
    setMergeFieldSelectOpen(false);
    setMergeMissingFields(Array.from(mergeSelectedFields));
    // 生成基准行数据，供对比弹窗直接使用
    const rows: Array<{timeLabel:string;original:string;romanized:string;translated:string;tokens:LyricTokenUnit[]}> = [];
    for (let i = 0; i < previewRows.length; i++) {
      const row = previewRows[i];
      const _ri = rowIndices[i + 1];
      if (!_ri || _ri[0] === -1) continue;
      rows.push({
        timeLabel: row.timeLabel || "",
        original: row.originalText || "",
        romanized: _ri[1] !== -1 ? (row.romanizedText || "") : "",
        translated: row.translatedText || "",
        tokens: row.tokens || [],
      });
    }
    setMergeCompareRows(rows);
    console.log("[合并] 生成基准行数:", rows.length, "第一行:", rows[0]?.original?.slice(0,20));
    setMergeSourceModalOpen(true);
    setMergeInspectedCandidate(null); setMergeInspectedSynced(""); setMergeInspectedWordTimed(""); setMergeInspectedTranslated(""); setMergeInspectedRomanized("");
    setMergeSimilarity(0); setMergeInspectedAvailable([]); setMergeInspecting(false);
  };

  const stripLrc = (text: string): string => text.replace(LRC_TIME_PATTERN, "").trim();

  const inspectMergeCandidate = async (candidate: TaskModuleALyricCandidate) => {
    setMergeInspectedCandidate(candidate); setMergeInspecting(true);
    setMergeInspectedSynced(""); setMergeInspectedWordTimed(""); setMergeInspectedTranslated(""); setMergeInspectedRomanized("");
    setMergeSelectedFields(new Set()); setMergeInspectedAvailable([]);
    try {
      const d = await getTaskModuleALyricDetail(taskId, candidate.candidate_id);
      setMergeInspectedSynced(d.synced_lyrics);
      setMergeInspectedWordTimed(d.word_timed_lyrics);
      setMergeInspectedTranslated(stripLrc(d.translated_lyrics));
      setMergeInspectedRomanized(stripLrc(d.romanized_lyrics));
      setMergeSimilarity(computeLyricsSimilarity(lyricsPreviewText, d.synced_lyrics));
      const a = getAvailableMergeFields({ word_timed_lyrics: d.word_timed_lyrics, translated_lyrics: d.translated_lyrics, romanized_lyrics: d.romanized_lyrics }, mergeMissingFields);
      setMergeInspectedAvailable(a);
      if (a.length > 0) setMergeSelectedFields(new Set(a));
    } catch (err) {
      appLogger.error("模块A页面", "获取合并候选歌词详情失败", { error: String(err), candidateId: candidate.candidate_id });
      message.warning("无法获取该候选的歌词详情。");
    }
    finally { setMergeInspecting(false); }
  };

  const applyMerge = () => {
    for (const l of mergeSelectedFields) {
      if (l === "逐字时间戳" && mergeInspectedWordTimed) setLyricsPreviewWordTimedText(mergeInspectedWordTimed);
      if (l === "翻译" && mergeInspectedTranslated) setLyricsPreviewTranslatedText(mergeInspectedTranslated);
      if (l === "罗马音" && mergeInspectedRomanized) setLyricsPreviewRomanizedText(mergeInspectedRomanized);
    }
    if (lyricsPreviewCandidate) {
      const mwt = mergeSelectedFields.has("逐字时间戳") && mergeInspectedWordTimed ? mergeInspectedWordTimed : lyricsPreviewWordTimedText;
      const mt = mergeSelectedFields.has("翻译") && mergeInspectedTranslated ? mergeInspectedTranslated : lyricsPreviewTranslatedText;
      const mr = mergeSelectedFields.has("罗马音") && mergeInspectedRomanized ? mergeInspectedRomanized : lyricsPreviewRomanizedText;
      const uc = { ...lyricsPreviewCandidate, has_word_timed_lyrics: Boolean(mwt || lyricsPreviewCandidate.has_word_timed_lyrics), has_translated_lyrics: Boolean(mt || lyricsPreviewCandidate.has_translated_lyrics), has_romanized_lyrics: Boolean(mr || lyricsPreviewCandidate.has_romanized_lyrics) };
      setLyricsPreviewCandidate(uc);
      setMergeMissingFields(getMissingFieldLabels(uc));
    }
    // 刷新 mergeCompareRows 以反映刚合并的翻译/罗马音
    const newRows: Array<{timeLabel:string;original:string;romanized:string;translated:string;tokens:LyricTokenUnit[]}> = [];
    for (let i = 0; i < previewRows.length; i++) {
      const row = previewRows[i];
      const _ri = rowIndices[i + 1];
      if (!_ri || _ri[0] === -1) continue;
      newRows.push({
        timeLabel: row.timeLabel || "",
        original: row.originalText || "",
        romanized: _ri[1] !== -1 ? (row.romanizedText || "") : "",
        translated: row.translatedText || "",
        tokens: row.tokens || [],
      });
    }
    setMergeCompareRows(newRows);
    setMergeSourceModalOpen(false);
    message.success("已合并补全所选字段。");
  };

  const extractTimeLabelFromLrc = (index: number): string => {
    try {
      const lines = (lyricsPreviewText || "").split(/\r?\n/).filter(Boolean);
      return lines[index]?.match(LRC_LINE_PATTERN)?.groups?.time || "";
    } catch (err) {
      appLogger.error("模块A页面", "从 LRC 提取时间标签失败", { error: String(err), index });
      return "";
    };
  };

  const handleSaveCurrentLyrics = () => {
    if (!lyricsPreviewCandidate) { message.info("没有可保存的歌词。"); return; }
    const savedId = "saved_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6);
    const sourceRows = savedRowsData
      ? savedRowsData.map((r) => ({ timeLabel: r.timeLabel, originalText: r.original, romanizedText: r.romanized, translatedText: r.translated, tokens: r.tokens }))
      : previewRows;
    const savedRows: { timeLabel: string; original: string; romanized: string; translated: string; tokens: LyricTokenUnit[] }[] = [];
    for (let i = 0; i < sourceRows.length; i++) {
      const row = sourceRows[i];
      const _ri = rowIndices[i + 1];
      if (!_ri || _ri[0] === -1) continue;
      savedRows.push({
        timeLabel: row.timeLabel || "",
        original: row.originalText || "",
        romanized: _ri[1] !== -1 ? (row.romanizedText || "") : "",
        translated: _ri[2] !== -1 ? (row.translatedText || "") : "",
        tokens: row.tokens,
      });
    }
    // 重建词级时间戳 LRC（优先使用已编辑的 word_timed，否则从 tokens 重建）
    let savedWordTimed = String(lyricsPreviewWordTimedText || "").trim();
    console.log("[保存歌词] 数据流 step1: savedWordTimed 原始值", {
      from: savedWordTimed ? "lyricsPreviewWordTimedText" : "空/假值",
      length: savedWordTimed.length,
      isEmpty: !savedWordTimed,
      isTokensPresent: savedWordTimed === "tokens_present",
      preview: savedWordTimed.slice(0, 200),
    });
    if (!savedWordTimed || savedWordTimed === "tokens_present") {
      // token 重建分支
      const tokenLines: string[] = [];
      for (const row of savedRows) {
        if (row.tokens && row.tokens.length > 0) {
          tokenLines.push((row.timeLabel ? "[" + row.timeLabel + "]" : "") + row.tokens.map((tk) => "<" + tk.start + ">" + tk.text + "<" + tk.end + ">").join(""));
        } else {
          tokenLines.push(row.timeLabel ? "[" + row.timeLabel + "]" + row.original : row.original);
        }
      }
      savedWordTimed = tokenLines.join("\n");
      console.log("[保存歌词] 数据流 step2: 走了 tokens 重建", {
        tokenLinesCount: tokenLines.length,
        savedWordTimedPreview: savedWordTimed.slice(0, 300),
      });
    } else {
      console.log("[保存歌词] 数据流 step2: 不走 tokens 重建，走裁剪", {
        savedRowsCount: savedRows.length,
        savedTimeLabels: savedRows.map(r => r.timeLabel),
        beforeCropLineCount: savedWordTimed.split("\n").length,
        beforeCropPreview: savedWordTimed.slice(0, 300),
      });
      // 裁剪 word_timed_lyrics 中多余的行（只保留 savedRows 中存在的行）
      const savedTimeLabels = new Set(savedRows.map((r) => r.timeLabel).filter(Boolean));
      console.log("[保存歌词] 数据流 step3: savedTimeLabels 集合", {
        size: savedTimeLabels.size,
        labels: [...savedTimeLabels].slice(0, 10),
      });
      if (savedTimeLabels.size > 0) {
        const LRC_TIME = /^\[(\d{2}:\d{2}(?:\.\d{1,3})?)\]/;
        const lines = savedWordTimed.split("\n");
        let kept = 0, removed = 0;
        const filtered = lines.filter((line, idx) => {
          const m = line.match(LRC_TIME);
          if (m) {
            const has = savedTimeLabels.has(m[1]);
            if (!has) removed++;
            else kept++;
            return has;
          }
          kept++;
          return true;
        });
        savedWordTimed = filtered.join("\n");
        console.log("[保存歌词] 数据流 step4: 裁剪结果", {
          before: { lines: lines.length },
          after: { lines: filtered.length, kept, removed },
          preview: savedWordTimed.slice(0, 300),
        });
      } else {
        console.log("[保存歌词] 数据流 step3b: savedTimeLabels 为空，跳过裁剪", {
          savedWordTimedPreview: savedWordTimed.slice(0, 200),
        });
      }
    }
    if (savedRows.length === 0) {
      savedWordTimed = "";
    }
    const sc: TaskModuleALyricCandidate & Record<string, unknown> = {
      ...lyricsPreviewCandidate, candidate_id: savedId, provider: "saved", provider_id: savedId,
      synced_lyrics: savedWordTimed || (savedRows.length > 0 ? (lyricsPreviewCandidate?.synced_lyrics || "") : ""),
      word_timed_lyrics: savedWordTimed || (savedRows.length > 0 ? (lyricsPreviewCandidate?.word_timed_lyrics || "") : ""),
      __saved_at: Date.now(), __saved_word_timed: savedWordTimed,
      preview_lines: savedRows.length > 0 ? savedRows.slice(0, 4).map((r) => r.original || r.romanized || "") : ["纯音乐/无歌词"],
      preview_text: savedRows.length > 0 ? savedRows.map((r) => r.original).join("\n").slice(0, 200) : "纯音乐/无歌词",
      has_word_timed_lyrics: Boolean(savedWordTimed),
      has_translated_lyrics: Boolean(savedRows.some((r) => r.translated)),
      has_romanized_lyrics: Boolean(savedRows.some((r) => r.romanized)),
      __saved_rows: savedRows,
    };
    console.log("[保存歌词] savedRows 逐行:", savedRows.map((r, idx) => `i=${idx} tl=${r.timeLabel} orig=${r.original.slice(0,20)} roma=${(r.romanized||"").slice(0,20)} trans=${(r.translated||"").slice(0,20)} tokens=${r.tokens?.length||0}`));
    console.log("[保存歌词] word_timed_lyrics 内容预览:", (savedWordTimed || "").slice(0, 300));
    setSavedCandidates((prev) => { const n = [...prev, sc]; try { localStorage.setItem("saved_lyrics_" + taskId, JSON.stringify(n)); } catch (err) { appLogger.error("模块A页面", "保存歌词到 localStorage 失败", { error: String(err), taskId }); } return n; });
    // 保存到后端磁盘（独立端口，json POST）
    const _saveLyricsPromise = (async () => {
      try {
        const _savePort = data?.save_lyrics_port;
        if (!_savePort || _savePort <= 0) {
          throw new Error(`保存歌词 POST 服务未启动（端口=${_savePort}）。`);
        }
        const _hostname = window.location.hostname || "127.0.0.1";
        console.log("[保存歌词] 尝试保存到后端:", {
          savePort: _savePort,
          url: `http://${_hostname}:${_savePort}/api/module-a/save-lyrics`,
        });
        const _body = JSON.stringify({ task_id: taskId, candidate: sc });
        const _resp = await fetch(`http://${_hostname}:${_savePort}/api/module-a/save-lyrics`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: _body,
        });
        if (!_resp.ok) {
          const _respBody = await _resp.text().catch(() => "");
          throw new Error(`HTTP ${_resp.status}: ${_respBody.slice(0, 100)}`);
        }
        console.log("[保存歌词] 后端保存成功");
      } catch (_err) {
        appLogger.warn("模块A页面", "保存歌词到后端失败", {
          error: String(_err), taskId,
          savePort: data?.save_lyrics_port, hostname: window.location.hostname,
        });
        throw _err;
      }
    })();
    _saveLyricsPromise
      .then(() => { message.success("歌词已保存到本地和后端。"); })
      .catch(() => { message.warning("歌词已保存到本地，但保存到后端失败，重跑时不会携带词级时间戳。"); });
  };

  const deleteSavedCandidate = (candidateId: string) => {
    setSavedCandidates((prev) => { const n = prev.filter((c) => c.candidate_id !== candidateId); try { localStorage.setItem("saved_lyrics_" + taskId, JSON.stringify(n)); } catch (err) { appLogger.error("模块A页面", "删除已保存歌词时写入 localStorage 失败", { error: String(err), taskId }); } return n; });
    // 删除后端磁盘文件
    fetch(`http://${window.location.hostname}:${data?.save_lyrics_port || 45706}/api/module-a/save-lyrics`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, candidate_id: candidateId }),
    }).catch((_err) => {
      appLogger.warn("模块A页面", "删除后端歌词文件失败", { error: String(_err), taskId });
    });
  };

  useEffect(() => {
    try {
      const raw = localStorage.getItem("saved_lyrics_" + taskId);
      if (raw) { const p = JSON.parse(raw); if (Array.isArray(p) && p.length > 0) setSavedCandidates(p); }
    } catch (err) {
      appLogger.error("模块A页面", "从 localStorage 读取已保存歌词失败", { error: String(err), taskId });
    }
  }, [taskId]);

  if (!data && !isLoading) {
    return (
      <Card bordered={false}>
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Alert
            type="error"
            showIcon
            message="模块 A 页面数据加载失败"
            description={queryErrorText || `没有找到任务：${taskId}`}
          />
          <Empty description={`当前无法打开模块 A 页面：${taskId}`} />
        </Space>
      </Card>
    );
  }

  return (
    <div className="page-stack">
      <Card bordered={false} loading={isLoading}>
        <div className="page-toolbar">
          <div>
            <Typography.Title level={3} className="page-title">
              模块 A 可视化
            </Typography.Title>
            <Typography.Text type="secondary">
              自动链会先读取文件元信息，再按元信息搜词；仍未命中时继续生成音乐指纹并识别曲目，最后按识别结果补词。
            </Typography.Text>
          </div>
          <Space wrap>
            <Button icon={<ReloadOutlined />} loading={isFetching && !isLoading} onClick={() => void refetch()}>
              刷新状态
            </Button>
            <Button type="primary" icon={<ReloadOutlined />} loading={rerunMutation.isPending} onClick={handleRerunClick}>
              重跑模块A
            </Button>
            <Button icon={<MinusCircleOutlined />} loading={instrumentalRerunMutation.isPending} onClick={handleInstrumentalRerun}>
              无歌词重跑模块A
            </Button>
            <Button icon={<PlayCircleOutlined />} loading={rerunMutation.isPending} onClick={handleRerunClick}>
              重跑task(全链路)
            </Button>
            <Button icon={<GlobalOutlined />} loading={searching && searchMode !== "manual_query"} onClick={startAutomaticLyricsSearch}>
              根据曲目自动查找歌词
            </Button>
            <Button icon={<SearchOutlined />} loading={searching && searchMode === "manual_query"} onClick={openManualLyricsSearchModal}>
              手动搜歌名
            </Button>
            <Button icon={<ExportOutlined />} disabled={!data?.module_a_visualization.available} onClick={openVisualization}>
              新标签页打开
            </Button>
          </Space>
        </div>

        {data ? (
          <Descriptions column={2} bordered className="detail-descriptions">
            <Descriptions.Item label="任务 ID">{data.task_id}</Descriptions.Item>
            <Descriptions.Item label="任务状态">{data.task_status}</Descriptions.Item>
            <Descriptions.Item label="模块 A 状态">
              {data.module_a_status === "running" ? (
                <Space>
                  <span style={{ color: "#faad14", fontWeight: "bold" }}>running</span>
                  <Button size="small" loading={resetStatusMutation.isPending} onClick={handleResetModuleAStatus}>
                    重置为pending
                  </Button>
                </Space>
              ) : (
                data.module_a_status
              )}
            </Descriptions.Item>
            <Descriptions.Item label="音频时长">{data.duration_seconds ? formatDuration(data.duration_seconds) : "未知"}</Descriptions.Item>
            <Descriptions.Item label="联网lrc状态">{getNetworkStatusTag(data.network_lrc_state.display_status)}</Descriptions.Item>
            <Descriptions.Item label="最近一次查找">{data.network_lrc_state.last_search_at || "尚未查找"}</Descriptions.Item>
            <Descriptions.Item label="文件元信息" span={2}>
              {renderEmbeddedMetadata(data.network_lrc_state.metadata_trace)}
            </Descriptions.Item>
            <Descriptions.Item label="指纹曲目信息" span={2}>
              {renderFingerprintMetadata(data.network_lrc_state.metadata_trace)}
            </Descriptions.Item>
            <Descriptions.Item label="当前选中歌词" span={2}>
              {data.network_lrc_state.selected_candidate.artist || data.network_lrc_state.selected_candidate.title
                ? buildCandidateLabel(data.network_lrc_state.selected_candidate)
                : "尚未选中"}
            </Descriptions.Item>
          </Descriptions>
        ) : null}
      </Card>

      {networkState?.display_status === "enabled" ? (
        <Alert
          type="success"
          showIcon
          message="已经启用联网查找的lrc"
          description={
            networkState.selected_candidate.artist || networkState.selected_candidate.title
              ? `当前选中：${buildCandidateLabel(networkState.selected_candidate)}`
              : undefined
          }
        />
      ) : null}

      {networkState?.display_status === "searched_not_enabled" ? (
        <Alert
          type="warning"
          showIcon
          message="已联网查找lrc但未启用"
          description={
            networkState.selected_candidate.artist || networkState.selected_candidate.title
              ? `当前选中：${buildCandidateLabel(networkState.selected_candidate)}`
              : undefined
          }
        />
      ) : null}

      {networkState?.lookup_error &&
      networkState.display_status !== "enabled" &&
      networkState.display_status !== "searched_not_enabled" ? (
        <Alert type="warning" showIcon message="最近一次联网查找lrc未完成" description={networkState.lookup_error} />
      ) : null}

      {data?.module_a_visualization.available ? (
        <ModuleAVisualization taskId={taskId} />
      ) : (
        <Card bordered={false}>
          <Empty description="当前任务还没有模块 A 可视化产物数据。" />
        </Card>
      )}

      <Modal
        title="手动输入歌曲名搜索"
        open={manualSearchModalOpen}
        onOk={submitManualLyricsSearch}
        onCancel={() => setManualSearchModalOpen(false)}
        okText="开始搜索"
        cancelText="取消"
        confirmLoading={searching && searchMode === "manual_query"}
        destroyOnHidden
      >
        <div className="page-stack">
          <Typography.Text type="secondary">自动搜索未命中时，可以在这里直接输入歌曲信息。歌手可选，歌名必填。</Typography.Text>
          <Input
            placeholder="歌手（可选），例如：郭顶"
            value={manualSearchArtist}
            onChange={(event) => setManualSearchArtist(event.target.value)}
            onPressEnter={() => void submitManualLyricsSearch()}
            maxLength={80}
          />
          <Input
            placeholder="歌名（必填），例如：水星记"
            value={manualSearchTitle}
            onChange={(event) => setManualSearchTitle(event.target.value)}
            onPressEnter={() => void submitManualLyricsSearch()}
            maxLength={120}
          />
        </div>
      </Modal>

      <Modal
        title="联网歌词候选"
        className="module-a-lyrics-modal"
        open={lyricsModalOpen}
        onOk={openLyricsSelectionConfirm}
        onCancel={() => {
          closeActiveSearchSocket();
          setLyricsModalOpen(false);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              closeActiveSearchSocket();
              setLyricsModalOpen(false);
            }}
          >
            关闭
          </Button>,
          <Button
            key="research"
            icon={<ReloadOutlined />}
            loading={searching && searchMode !== "manual_query"}
            onClick={() => startLyricsSearch({ manualQuery: "" })}
          >
            重新搜索
          </Button>,
          <Button key="confirm" type="primary" onClick={openLyricsSelectionConfirm}>
            确定
          </Button>,
        ]}
        width={lyricsModalWidth}
        destroyOnHidden
      >
        <div className="page-stack module-a-lyrics-modal__body" style={{ height: lyricsModalBodyHeight }}>
          <Radio.Group
            value={selectedCandidateId}
            onChange={(event) => {
              hasUserPickedCandidateRef.current = true;
              setSelectedCandidateId(String(event.target.value));
            }}
            className="module-a-provider-group"
          >
            {displayProviderGroups.length ? (
              <div className="module-a-provider-viewport">
                <div className="module-a-provider-lane">
                  {displayProviderGroups.map((providerGroup) => {
                    const pageIndex = readProviderPage(providerPages, providerGroup.provider);
                    const pageSize = providerGroup.page_size || 10;
                    const totalCount = providerGroup.total_count || providerGroup.candidates.length;
                    const pageCount = Math.max(1, Math.ceil(totalCount / pageSize));
                    const visibleCandidates = providerGroup.candidates.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
                  return (
                    <Card
                      key={providerGroup.provider}
                      size="small"
                      className="module-a-provider-column"
                        title={
                        <Space wrap size={[8, 8]}>
                          <Typography.Text strong>{providerGroup.display_name || providerGroup.provider}</Typography.Text>
                          <Tag>{Math.min(pageSize, totalCount)} 条</Tag>
                        </Space>
                      }
                        extra={
                          <Space size={4}>
                            <Button
                              icon={<LeftOutlined />}
                              size="small"
                              disabled={pageIndex <= 0}
                              onClick={() =>
                                setProviderPages((current) => ({
                                  ...current,
                                  [providerGroup.provider]: Math.max(0, pageIndex - 1),
                                }))
                              }
                            />
                            <Typography.Text type="secondary">{pageIndex + 1}/{pageCount}</Typography.Text>
                            <Button
                              icon={<RightOutlined />}
                              size="small"
                              disabled={pageIndex >= pageCount - 1}
                              onClick={async () => {
                                const nextPageIndex = Math.min(pageCount - 1, pageIndex + 1);
                                await ensureProviderPageLoaded(providerGroup.provider, nextPageIndex);
                                setProviderPages((current) => ({
                                  ...current,
                                  [providerGroup.provider]: nextPageIndex,
                                }));
                              }}
                            />
                          </Space>
                        }
                      >
                        <div className="module-a-provider-scroll">
                          {visibleCandidates.length ? (
                            <Space direction="vertical" size={12} style={{ width: "100%" }}>
                              {visibleCandidates.map((candidate) => (
                                <Radio key={candidate.candidate_id} value={candidate.candidate_id} className="module-a-candidate-radio">
                                  <div
                                    className="module-a-candidate-card"
                                    onClick={() => {
                                      openLyricsPreview(candidate);
                                    }}
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter" || event.key === " ") {
                                        event.preventDefault();
                                        openLyricsPreview(candidate);
                                      }
                                    }}
                                    role="button"
                                    tabIndex={0}
                                  >
                                    <Space wrap size={[8, 8]} className="module-a-candidate-card__headline">
                                      <Typography.Text strong>{buildCandidateLabel(candidate)}</Typography.Text>
                                      {candidate.duration_seconds ? <Tag>{formatDuration(candidate.duration_seconds)}</Tag> : null}
                                      {candidate.has_word_timed_lyrics ? <Tag color="processing">词级</Tag> : null}
                                      {candidate.has_translated_lyrics ? <Tag color="green">翻译</Tag> : null}
                                      {candidate.has_romanized_lyrics ? <Tag color="purple">罗马音</Tag> : null}
                                      {providerGroup.provider === "saved" ? (() => {
                                        const _st = (candidate as Record<string, unknown>).__saved_at as number || 0;
                                        const _d = _st ? new Date(_st) : null;
                                        return <><Typography.Text type="secondary" style={{ fontSize: 10 }}>{_d ? String(_d.getHours()).padStart(2,"0") + ":" + String(_d.getMinutes()).padStart(2,"0") : ""}</Typography.Text>
                                          <Button type="text" size="small" danger onClick={(e) => { e.stopPropagation(); deleteSavedCandidate(candidate.candidate_id); }} style={{ padding: 0, minWidth: 20, fontSize: 14 }}>✕</Button>
                                        </>;
                                      })() : null}
                                    </Space>
                                  </div>
                                </Radio>
                              ))}
                            </Space>
                          ) : (
                            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前页没有结果" />
                          )}
                        </div>
                      </Card>
                    );
                  })}
                </div>
              </div>
            ) : (
              <Empty
                description={
                  searching
                    ? "正在等待第一个来源返回结果"
                    : "当前还没有可展示的歌词来源结果"
                }
              />
            )}
          </Radio.Group>
        </div>
      </Modal>

      <Modal
        title="是否根据这份歌词运行模块A"
        open={confirmModalOpen}
        onCancel={() => setConfirmModalOpen(false)}
        footer={[
          <Button
            key="cancel"
            onClick={() => setConfirmModalOpen(false)}
          >
            取消
          </Button>,
          <Button key="lyrics-only" icon={<SoundOutlined />} loading={selectLyricsMutation.isPending && selectLyricsMutation.variables?.candidateId === selectedCandidateId && selectLyricsMutation.variables?.enable === true && selectLyricsMutation.variables?.rerunMode === "lyrics_only"} onClick={() => void submitSelectedLyrics(true, "lyrics_only")}>仅更新歌词</Button>,
          <Button key="ok" type="primary" loading={selectLyricsMutation.isPending && selectLyricsMutation.variables?.candidateId === selectedCandidateId && selectLyricsMutation.variables?.enable === true && !selectLyricsMutation.variables?.rerunMode} onClick={() => void submitSelectedLyrics(true)}>确定</Button>,
          <Button key="correct" disabled={!selectedCandidateId} onClick={async () => {
            setConfirmModalOpen(false);
            if (selectedCandidateId) {
              const cand = displayCandidates.find((c) => c.candidate_id === selectedCandidateId);
              if (cand) {
                setLyricsPreviewCandidate(cand);
                const saved = cand as Record<string, unknown>;
                const rows = saved.__saved_rows as Array<{ timeLabel: string; original: string; romanized: string; translated: string }> | undefined;
                const wt = saved.__saved_word_timed as string || "";
                if (rows) {
                  setLyricsPreviewText(rows.map((r) => r.timeLabel ? "[" + r.timeLabel + "]" + r.original : r.original).join("\n"));
                  setLyricsPreviewRomanizedText(rows.map((r) => r.romanized || "").join("\n"));
                  setLyricsPreviewTranslatedText(rows.map((r) => r.translated || "").join("\n"));
                  setLyricsPreviewWordTimedText(wt);
                } else {
                  try { const d = await getTaskModuleALyricDetail(taskId, selectedCandidateId);
                    setLyricsPreviewText(d.synced_lyrics); setLyricsPreviewWordTimedText(d.word_timed_lyrics);
                    setLyricsPreviewRomanizedText(d.romanized_lyrics); setLyricsPreviewTranslatedText(d.translated_lyrics);
                  } catch (err) {
                    appLogger.error("模块A页面", "矫正 Modal 加载歌词详情失败", { error: String(err), taskId, selectedCandidateId });
                  } }}
            }
            setCorrectFunasrOpen(true);
          }}>LLM 矫正</Button>,
        ]}
        width={760}
        maskClosable={false}
        closable={false}
        keyboard={false}
        destroyOnHidden
      >
        {selectedCandidate ? (
          <div className="page-stack">
            <Alert
              type="info"
              showIcon
              message={buildCandidateLabel(selectedCandidate)}
              description={
                <>
                  <div>确定后会把这份歌词设为已启用，并从模块 A 开始重跑。</div>
                  <div style={{ marginTop: 8 }}>
                    <Typography.Text type="secondary">仅更新歌词仅运行歌词+算法层（跳过信号处理），适合跳过环境检查或不需换 Demucs/Allin1 的微调场景。</Typography.Text>
                  </div>
                </>
              }
            />
          </div>
        ) : (
          <Empty description="当前没有可确认的歌词候选。" />
        )}
      </Modal>

      <Modal
        title={lyricsPreviewCandidate ? buildCandidateLabel(lyricsPreviewCandidate) : "歌词预览"}
        open={Boolean(lyricsPreviewCandidate)}
        onCancel={() => {
          setLyricsPreviewCandidate(null);
          setLyricsPreviewText("");
          setLyricsPreviewWordTimedText("");
          setLyricsPreviewTranslatedText("");
          setLyricsPreviewRomanizedText("");
          setActiveTokenKey("");
          lyricDetailMutation.reset();
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setLyricsPreviewCandidate(null);
              setLyricsPreviewText("");
              setLyricsPreviewWordTimedText("");
              setLyricsPreviewTranslatedText("");
              setLyricsPreviewRomanizedText("");
              setActiveTokenKey("");
              lyricDetailMutation.reset();
            }}
          >
            关闭
          </Button>,
          <Button key="save" disabled={!lyricsPreviewCandidate} onClick={handleSaveCurrentLyrics}>
            保存
          </Button>,
        ]}
        width={960}
        destroyOnHidden
      >
        {lyricsPreviewCandidate ? (
          <div className="page-stack module-a-lyrics-preview-shell">
            <div className="module-a-lyrics-preview-toolbar">
              <Space wrap size={[8, 8]}>
                <Tag>{lyricsPreviewCandidate.provider}</Tag>
                {lyricsPreviewCandidate.has_word_timed_lyrics ? (
                  <Checkbox checked={showWordTimedOverlay} onChange={(event) => setShowWordTimedOverlay(event.target.checked)}>
                    词级时间戳
                  </Checkbox>
                ) : null}
                {lyricsPreviewCandidate.has_translated_lyrics ? (
                  <Checkbox checked={showTranslatedLyrics} onChange={(event) => setShowTranslatedLyrics(event.target.checked)}>
                    翻译
                  </Checkbox>
                ) : null}
                {lyricsPreviewCandidate.has_romanized_lyrics ? (
                  <Checkbox checked={showRomanizedLyrics} onChange={(event) => setShowRomanizedLyrics(event.target.checked)}>
                    罗马音
                  </Checkbox>
                ) : null}
                {(() => { const rows = rowIndices.slice(1); const n = rows.filter(r => r[0]===-1).length; return n > 0 ? <Button size="small" onClick={() => setRowIndices([rowIndices[0],...rowIndices.slice(1).map(r => [r[0]>=0?0:-1,r[1],r[2],r[3]])])}>原文已滤 {n}</Button> : null; })()}
                {(() => { const rows = rowIndices.slice(1); const n = rows.filter(r => r[1]===-1).length; return n > 0 ? <Button size="small" onClick={() => setRowIndices([rowIndices[0],...rowIndices.slice(1).map(r => [r[0],r[1]>=0?0:-1,r[2],r[3]])])}>罗马音已滤 {n}</Button> : null; })()}
                {(() => { const rows = rowIndices.slice(1); const n = rows.filter(r => r[2]===-1).length; return n > 0 ? <Button size="small" onClick={() => setRowIndices([rowIndices[0],...rowIndices.slice(1).map(r => [r[0],r[1],r[2]>=0?0:-1,r[3]])])}>翻译已滤 {n}</Button> : null; })()}
                <Button size="small" onClick={openMergeSourceModal}>合并</Button>
              </Space>
            </div>
            {lyricDetailMutation.isPending ? (
              <Alert type="info" showIcon message="正在加载这条候选的歌词内容" />
            ) : (
              <div className="module-a-lyrics-preview-board">
                {previewRows.length ? (
                  previewRows.map((row, rowIndex) => {
                    const _ri = rowIndices[rowIndex + 1];
                    if (!_ri || !row.originalText) return null;
                    if (rowIndex < 3) console.log("渲染行", rowIndex, "_ri:", JSON.stringify(_ri), "orig:", row.originalText?.slice(0,30), "roma:", row.romanizedText?.slice(0,30));
                    const _toggle = (fi: number) => setRowIndices((p) => { if (!p.length || !p[rowIndex + 1]) return p; const n = p.map(r=>[...r]); const idx = rowIndex + 1; n[idx][fi] = n[idx][fi]===-1 ? (n[0][fi] ?? idx) : -1; return n; });
                    return (
                    <div key={`${row.timeLabel}-${rowIndex}`} className="module-a-lyrics-row" style={{ display: _ri[0] === -1 ? "none" : undefined }}>
                      <div className="module-a-lyrics-row__time">
                        <Typography.Text type="secondary">{row.timeLabel || "--:--.--"}</Typography.Text>
                      </div>
                      <div className="module-a-lyrics-row__content">
                        {showRomanizedLyrics && row.romanizedText && _ri[1] !== -1 ? (
                          <div className="module-a-lyrics-line module-a-lyrics-line--romanized">
                            <Checkbox checked={_ri[1] !== -1} onChange={() => _toggle(1)} style={{ marginRight: 4 }} />
                            {row.romanizedText}
                          </div>
                        ) : null}
                        <div className="module-a-lyrics-line module-a-lyrics-line--original">
                          <Checkbox checked={_ri[0] !== -1} onChange={() => _toggle(0)} style={{ marginRight: 4 }} />
                          {showWordTimedOverlay && row.tokens.length
                            ? row.tokens.map((token, tokenIndex) => {
                                const tokenKey = `${rowIndex}-${tokenIndex}`;
                                const isActive = activeTokenKey === tokenKey;
                                return (
                                  <button
                                    key={tokenKey}
                                    type="button"
                                    className={`module-a-word-token${isActive ? " is-active" : ""}`}
                                    onClick={() => setActiveTokenKey((current) => (current === tokenKey ? "" : tokenKey))}
                                  >
                                    {token.text}
                                  </button>
                                );
                              })
                            : row.originalText}
                        </div>
                        {showWordTimedOverlay && row.tokens.length && activeTokenKey.startsWith(`${rowIndex}-`) ? (
                          <div className="module-a-lyrics-token-meta">
                            {row.tokens.map((token, tokenIndex) => {
                              const tokenKey = `${rowIndex}-${tokenIndex}`;
                              if (activeTokenKey !== tokenKey) {
                                return null;
                              }
                              return (
                                <Tag key={tokenKey} color="processing">
                                  {token.text} {buildTokenTimestampText(token)}
                                </Tag>
                              );
                            })}
                          </div>
                        ) : null}
                        {showTranslatedLyrics && row.translatedText && _ri[2] !== -1 ? (
                          <div className="module-a-lyrics-line module-a-lyrics-line--translated">
                            <Checkbox checked={_ri[2] !== -1} onChange={() => _toggle(2)} style={{ marginRight: 4 }} />
                            {row.translatedText}
                          </div>
                        ) : null}
                      </div>
                    </div>
                    );
                  })
                ) : (
                  <Alert type="warning" showIcon message="当前来源没有返回可展示的歌词正文。" />
                )}
              </div>
            )}
          </div>
        ) : null}
      </Modal>

      {/* Merge field select */}
      <Modal title="选择要合并的字段" open={mergeFieldSelectOpen} onCancel={() => setMergeFieldSelectOpen(false)}
        footer={[<Button key="cancel" onClick={() => setMergeFieldSelectOpen(false)}>取消</Button>, <Button key="ok" type="primary" disabled={mergeSelectedFields.size === 0} onClick={confirmMergeFieldSelection}>确认 ({mergeSelectedFields.size})</Button>]}
        width={400} destroyOnHidden>
        <Space direction="vertical" size={12}>
          <Typography.Text>选择要从其他来源补充的字段：</Typography.Text>
          <Checkbox.Group value={Array.from(mergeSelectedFields)} onChange={(values) => setMergeSelectedFields(new Set(values as string[]))}>
            <Space direction="vertical">{["翻译","罗马音","逐字时间戳"].map((label) => (<Checkbox key={label} value={label}><Tag color="processing">{label}</Tag></Checkbox>))}</Space>
          </Checkbox.Group>
        </Space>
      </Modal>

      {/* Merge source */}
      <Modal title="选择歌词来源以补全缺失字段" open={mergeSourceModalOpen} onCancel={() => setMergeSourceModalOpen(false)}
        footer={[<Button key="close" onClick={() => setMergeSourceModalOpen(false)}>关闭</Button>, <Button key="apply" type="primary" disabled={mergeSelectedFields.size === 0 || !mergeInspectedCandidate} onClick={applyMerge}>确认补充 ({mergeSelectedFields.size})</Button>]}
        width={lyricsModalWidth} destroyOnHidden>
        <div style={{ height: lyricsModalBodyHeight, overflow: "auto" }}>
          {displayProviderGroups.length ? (
            <Radio.Group value={mergeInspectedCandidate?.candidate_id || ""} onChange={(event) => { const cid = String(event.target.value); const cand = displayCandidates.find((c) => c.candidate_id === cid); if (cand) void inspectMergeCandidate(cand); }} className="module-a-provider-group">
              <div className="module-a-provider-viewport"><div className="module-a-provider-lane">
                {displayProviderGroups.map((pg) => {
                  const ps = pg.page_size || 10; const cs = pg.candidates.slice(0, ps);
                  return <Card key={pg.provider} size="small" className="module-a-provider-column"
                    title={<Space wrap size={[8,8]}><Typography.Text strong>{pg.display_name || pg.provider}</Typography.Text><Tag>{cs.length}</Tag></Space>}>
                    <div className="module-a-provider-scroll"><Space direction="vertical" size={12} style={{width:"100%"}}>
                      {cs.map((cand) => (<Radio key={cand.candidate_id} value={cand.candidate_id} className="module-a-candidate-radio">
                        <div className="module-a-candidate-card" onClick={() => void inspectMergeCandidate(cand)} role="button" tabIndex={0}>
                          <Space wrap size={[8,8]} className="module-a-candidate-card__headline">
                            <Typography.Text strong>{buildCandidateLabel(cand)}</Typography.Text>
                            {cand.duration_seconds ? <Tag>{formatDuration(cand.duration_seconds)}</Tag> : null}
                            {cand.has_word_timed_lyrics ? <Tag color="processing">词级</Tag> : null}
                            {cand.has_translated_lyrics ? <Tag color="green">翻译</Tag> : null}
                            {cand.has_romanized_lyrics ? <Tag color="purple">罗马音</Tag> : null}
                            {mergeInspectedCandidate?.candidate_id === cand.candidate_id && mergeSimilarity > 0 ? <Tag color={mergeSimilarity>=80?"success":mergeSimilarity>=60?"warning":"error"}>相似度 {mergeSimilarity}%</Tag> : null}
                            {mergeInspectedCandidate?.candidate_id === cand.candidate_id && mergeInspectedSynced ? <Button size="small" type="link" onClick={(e) => { e.stopPropagation(); setCompareModalOpen(true); }}>对比</Button> : null}
                          </Space>
                        </div>
                      </Radio>))}
                </Space></div>
                  </Card>;
                })}
              </div></div>
            </Radio.Group>
          ) : <Empty description="没有可用的歌词来源" />}
          {mergeInspectedCandidate ? (<Card size="small" style={{marginTop:16}}>
            {mergeInspecting ? <Alert type="info" showIcon message="正在加载该候选的歌词详情..." />
            : <Space direction="vertical" size={12} style={{width:"100%"}}>
                {mergeInspectedAvailable.length > 0 ? <><Typography.Text>可用补充字段（可多选）：</Typography.Text>
                  <Checkbox.Group value={Array.from(mergeSelectedFields)} onChange={(values) => setMergeSelectedFields(new Set(values as string[]))}>
                    <Space direction="vertical">{mergeInspectedAvailable.map((f) => <Checkbox key={f} value={f}><Tag color="processing">{f}</Tag></Checkbox>)}</Space>
                  </Checkbox.Group>
                </> : <Alert type="warning" showIcon message="当前来源没有可用的补充字段。" />}
              </Space>}
          </Card>) : null}
        </div>
      </Modal>

      {/* Compare */}
      <Modal title="对比合并" open={compareModalOpen} onCancel={() => setCompareModalOpen(false)}
        footer={[<Button key="close" onClick={() => setCompareModalOpen(false)}>关闭</Button>, <Button key="save" type="primary" onClick={() => {
          const tL = mergeInspectedTranslated ? mergeInspectedTranslated.split(/\r?\n/).filter(Boolean) : [];
          const rL = mergeInspectedRomanized ? mergeInspectedRomanized.split(/\r?\n/).filter(Boolean) : [];
          if (!lyricsPreviewCandidate) { message.warning("没有基础源可选"); return; }
          if (tL.length) setLyricsPreviewTranslatedText(tL.slice(Math.max(0, candTransOffset)).join("\n"));
          if (rL.length) setLyricsPreviewRomanizedText(rL.slice(Math.max(0, candTransOffset)).join("\n"));
          const scId = "saved_" + Date.now() + "_" + Math.random().toString(36).slice(2,6);
          const rd: {timeLabel:string;original:string;romanized:string;translated:string;tokens:{text:string;start:string;end:string}[]}[] = [];
          for (let _j = 0; _j < mergeCompareRows.length; _j++) {
            const _r = mergeCompareRows[_j];
            const _ci = _j + candTransOffset;
            rd.push({
              timeLabel: _r.timeLabel,
              original: _r.original,
              romanized: _r.romanized,
              translated: _ci >= 0 && _ci < tL.length ? (tL[_ci] || "") : _r.translated,
              tokens: _r.tokens || [],
            });
          }
          const cand_ = { ...(lyricsPreviewCandidate as TaskModuleALyricCandidate), candidate_id: scId, provider: "saved" as const, provider_id: scId, __saved_at: Date.now(), __saved_word_timed: mergeInspectedWordTimed || lyricsPreviewWordTimedText, preview_lines: rd.slice(0,4).map((x:any)=>x.original), preview_text: rd.map((x:any)=>x.original).join("\n").slice(0,200), has_word_timed_lyrics: Boolean(mergeInspectedWordTimed || lyricsPreviewWordTimedText), has_translated_lyrics: Boolean(tL.some(Boolean)), has_romanized_lyrics: Boolean(rd.some((r:any) => r.romanized)), __saved_rows: rd } as TaskModuleALyricCandidate & Record<string, unknown>;
          setSavedCandidates((prev) => { const n = [...prev, cand_]; try { localStorage.setItem("saved_lyrics_"+taskId, JSON.stringify(n)); } catch (err) { appLogger.error("模块A页面", "对比合并保存到 localStorage 失败", { error: String(err), taskId }); } return n; });
          setSelectedCandidateId(scId); setLyricsPreviewCandidate(cand_ as TaskModuleALyricCandidate); setSavedRowsData(rd as any);
          setCompareModalOpen(false); setMergeSourceModalOpen(false);
          message.success("已合并并保存。");
        }}>保存</Button>]}
        width={1200} destroyOnHidden>
        <div style={{display:"flex", gap:16, height:500}}>
          <div style={{flex:1, overflow:"auto", border:"1px solid #e8ecf0", borderRadius:8, padding:12}}>
            <Typography.Title level={5}>基础源{lyricsPreviewCandidate ? <Tag style={{marginLeft:8}}>{buildCandidateLabel(lyricsPreviewCandidate)}</Tag> : null}
              <Button.Group size="small" style={{marginLeft:8}}>
                <Button onClick={() => setCandTransOffset((v) => Math.min(20, v+1))}>↑</Button>
                <Button disabled style={{color:"#333", cursor:"default"}}>翻译{candTransOffset}</Button>
                <Button onClick={() => setCandTransOffset((v) => Math.max(-20, v-1))}>↓</Button>
              </Button.Group>
            </Typography.Title>
            {(() => {
              const _t = mergeInspectedTranslated ? mergeInspectedTranslated.split(/\r?\n/).filter(Boolean) : [];
              const _res: ReactNode[] = [];
              for (let _j = 0; _j < mergeCompareRows.length; _j++) {
                const _r = mergeCompareRows[_j];
                const _ci = _j + candTransOffset;
                const _tt = _ci >= 0 && _ci < _t.length ? _t[_ci] || "" : "";
                _res.push(<div key={"b-"+_j} style={{marginBottom:8, padding:"4px 0", borderBottom:"1px solid #f0f0f0"}}>
                  <Typography.Text type="secondary" style={{fontSize:11}}>{_r.timeLabel}</Typography.Text>
                  <div style={{fontSize:13}}>{_r.original}</div>
                  {_r.romanized ? <div style={{fontSize:12, color:"#722ed1"}}>{_r.romanized}</div> : null}
                  {_tt ? <div style={{fontSize:12, color:"#52c41a"}}>{_tt}</div> : null}
                </div>);
              }
              return _res;
            })()}
          </div>
          <div style={{flex:1, overflow:"auto", border:"1px solid #e8ecf0", borderRadius:8, padding:12}}>
            <Typography.Title level={5}>候选源{mergeInspectedCandidate ? <Tag style={{marginLeft:8}}>{buildCandidateLabel(mergeInspectedCandidate)}</Tag> : null}</Typography.Title>
            {mergeInspectedSynced ? (() => { const _cl = mergeInspectedSynced.split(/\r?\n/).filter(Boolean); return _cl.map((l,i) => { const _m = l.match(LRC_LINE_PATTERN); return <div key={"c-"+i} style={{marginBottom:8, padding:"4px 0", borderBottom:"1px solid #f0f0f0"}}><Typography.Text type="secondary" style={{fontSize:11}}>{_m?.groups?.time||""}</Typography.Text><div style={{fontSize:13}}>{_m?.groups?.text?.trim()||l}</div></div>; }); })() : mergeInspecting ? <Alert type="info" showIcon message="加载中..." /> : null}
          </div>
        </div>
      </Modal>

      <CorrectFunasrModal
        open={correctFunasrOpen} onClose={() => setCorrectFunasrOpen(false)}
        taskId={taskId} lyricsPreviewText={lyricsPreviewText} lyricsPreviewCandidate={lyricsPreviewCandidate}
        onSave={(candidate) => { setSavedCandidates((prev) => { const n = [...prev, candidate]; try { localStorage.setItem("saved_lyrics_" + taskId, JSON.stringify(n)); } catch (err) { appLogger.error("模块A页面", "LLM 矫正保存到 localStorage 失败", { error: String(err), taskId }); } return n; }); }}
        onResegment={() => { setCorrectFunasrOpen(false); setConfirmModalOpen(true); }}
        selectedCandidateId={selectedCandidateId}
        lyricsPreviewWordTimedText={lyricsPreviewWordTimedText} lyricsPreviewTranslatedText={lyricsPreviewTranslatedText}
        lyricsPreviewRomanizedText={lyricsPreviewRomanizedText} lyricsPreviewRows={previewRows} />
    </div>
  );
}
