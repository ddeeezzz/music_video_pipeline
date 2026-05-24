import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  ExportOutlined,
  GlobalOutlined,
  LeftOutlined,
  ReloadOutlined,
  RightOutlined,
  SearchOutlined,
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
  selectTaskModuleALyrics,
  taskQueryKeys,
} from "@/api/taskApi";
import { appLogger } from "@/app/logger";
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
    const matchedTranslatedLine = translatedLineMap.get(timeKey);
    const matchedRomanizedLine = romanizedLineMap.get(timeKey);
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
    mutationFn: () => rerunTask(taskId),
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

  const selectLyricsMutation = useMutation({
    mutationFn: ({ candidateId, enable }: { candidateId: string; enable: boolean }) =>
      selectTaskModuleALyrics(taskId, candidateId, enable),
    onSuccess: async (payload, variables) => {
      await invalidateTaskScopes();
      setConfirmModalOpen(false);
      if (variables.enable) {
        setLyricsModalOpen(false);
      }
      message.success(payload.message || (variables.enable ? "模块 A 已按选中歌词开始重跑" : "已联网查找lrc但未启用"));
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

  const networkState = data?.network_lrc_state;
  const cachedProviderGroups = useMemo(
    () => sortProviderGroups(networkState?.provider_groups || []),
    [networkState?.provider_groups],
  );
  const hasCachedLyricsResults = Boolean((networkState?.cached_candidates_count || 0) > 0 && cachedProviderGroups.length);
  const displayProviderGroups = providerGroups.length
    ? orderProviderGroups(providerGroups, providerOrder)
    : showCachedResults
      ? orderProviderGroups(cachedProviderGroups, providerOrder)
      : [];
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
  const lyricsPreviewRows = useMemo(
    () =>
      buildLyricPreviewRows(
        lyricsPreviewText,
        lyricsPreviewWordTimedText,
        lyricsPreviewTranslatedText,
        lyricsPreviewRomanizedText,
      ),
    [lyricsPreviewRomanizedText, lyricsPreviewText, lyricsPreviewTranslatedText, lyricsPreviewWordTimedText],
  );

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
      } catch {
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
        void invalidateTaskScopes();
        if (!flattenedCandidates.length && eventData.suggest_manual_query) {
          setManualSearchModalOpen(true);
        }
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

  const submitSelectedLyrics = (enable: boolean) => {
    if (!selectedCandidateId) {
      message.info("请先选择一份歌词候选。");
      return;
    }
    selectLyricsMutation.mutate({ candidateId: selectedCandidateId, enable });
  };

  const openLyricsPreview = (candidate: TaskModuleALyricCandidate) => {
    hasUserPickedCandidateRef.current = true;
    setSelectedCandidateId(candidate.candidate_id);
    setLyricsPreviewCandidate(candidate);
    setLyricsPreviewText("");
    setLyricsPreviewWordTimedText("");
    setLyricsPreviewTranslatedText("");
    setLyricsPreviewRomanizedText("");
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
            <Button type="primary" icon={<ReloadOutlined />} loading={rerunMutation.isPending} onClick={() => rerunMutation.mutate()}>
              重跑模块A
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
            <Descriptions.Item label="模块 A 状态">{data.module_a_status}</Descriptions.Item>
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
        <Card bordered={false} className="iframe-card">
          <Alert type="info" showIcon message={`来源文件：${data.module_a_visualization.path}`} style={{ marginBottom: 16 }} />
          <iframe title={`module-a-${taskId}`} src={data.module_a_visualization.url} className="module-a-iframe" />
        </Card>
      ) : (
        <Card bordered={false}>
          <Empty description="当前任务还没有模块 A 可视化 HTML 产物。" />
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
                                      {candidate.has_word_timed_lyrics ? <Tag color="processing">词级</Tag> : null}
                                      {candidate.has_translated_lyrics ? <Tag color="green">翻译</Tag> : null}
                                      {candidate.has_romanized_lyrics ? <Tag color="purple">罗马音</Tag> : null}
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
        onCancel={() => void submitSelectedLyrics(false)}
        footer={[
          <Button
            key="cancel"
            loading={
              selectLyricsMutation.isPending &&
              selectLyricsMutation.variables?.candidateId === selectedCandidateId &&
              selectLyricsMutation.variables?.enable === false
            }
            onClick={() => void submitSelectedLyrics(false)}
          >
            取消
          </Button>,
          <Button
            key="ok"
            type="primary"
            loading={
              selectLyricsMutation.isPending &&
              selectLyricsMutation.variables?.candidateId === selectedCandidateId &&
              selectLyricsMutation.variables?.enable === true
            }
            onClick={() => void submitSelectedLyrics(true)}
          >
            确定
          </Button>,
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
              description="确定后会把这份歌词设为已启用，并从模块 A 开始重跑；取消则只记录为已联网查找但未启用。"
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
              </Space>
            </div>
            {lyricDetailMutation.isPending ? (
              <Alert type="info" showIcon message="正在加载这条候选的歌词内容" />
            ) : (
              <div className="module-a-lyrics-preview-board">
                {lyricsPreviewRows.length ? (
                  lyricsPreviewRows.map((row, rowIndex) => (
                    <div key={`${row.timeLabel}-${rowIndex}`} className="module-a-lyrics-row">
                      <div className="module-a-lyrics-row__time">
                        <Typography.Text type="secondary">{row.timeLabel || "--:--.--"}</Typography.Text>
                      </div>
                      <div className="module-a-lyrics-row__content">
                        {showRomanizedLyrics && row.romanizedText ? (
                          <Typography.Paragraph className="module-a-lyrics-line module-a-lyrics-line--romanized">
                            {row.romanizedText}
                          </Typography.Paragraph>
                        ) : null}
                        <div className="module-a-lyrics-line module-a-lyrics-line--original">
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
                        {showWordTimedOverlay && row.tokens.length ? (
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
                        {showTranslatedLyrics && row.translatedText ? (
                          <Typography.Paragraph className="module-a-lyrics-line module-a-lyrics-line--translated">
                            {row.translatedText}
                          </Typography.Paragraph>
                        ) : null}
                      </div>
                    </div>
                  ))
                ) : (
                  <Alert type="warning" showIcon message="当前来源没有返回可展示的歌词正文。" />
                )}
              </div>
            )}
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
