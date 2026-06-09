import { useEffect, useMemo, useState } from "react";
import {
  PlaySquareOutlined,
  ReloadOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Image,
  Modal,
  Radio,
  Row,
  Space,
  Tag,
  Typography,
} from "antd";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getModuleDSegmentVideos,
  getRebuildAudioCandidates,
  getTaskModuleDData,
  getToonCrafterMode,
  rebuildModuleDFinal,
  rerunModuleDSegment,
  rerunModuleDBothFrames,
  rerunModuleDAll,
  rerunModuleDSegmentToonCrafter,
  rerunModuleDAllToonCrafter,
  rerunModuleDSegmentRemotion,
  rerunModuleDAllRemotion,
  setToonCrafterMode,
  setToonCrafterShotMode,
  taskQueryKeys,
} from "@/api/taskApi";
import type { AudioCandidate } from "@/api/taskApi";
import {
  buildSegmentVideoUrl,
  probeSegmentVideoExists,
  resolveSegmentVideoBasePath,
} from "@/features/moduleD/segmentVideo";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import type {
  TaskModuleDData,
  TaskModuleDSegment,
  TaskModuleDSegmentVideoFile,
  TaskModuleDShot,
} from "@/schemas/moduleD";

interface RerunSegmentModalProps {
  open: boolean;
  segment: TaskModuleDSegment;
  onCancel: () => void;
  onConfirm: (frameType: "start" | "end", transitionBg?: "white" | "black") => void;
  loading?: boolean;
}

function RerunSegmentModal({
  open,
  segment,
  onCancel,
  onConfirm,
  loading,
}: RerunSegmentModalProps) {
  const [frameType, setFrameType] = useState<"start" | "end">("start");
  const [transitionBg, setTransitionBg] = useState<"white" | "black">("white");
  const remotionId = segment.remotion_id || "";
  const isTransition = remotionId && new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]).has(remotionId);
  const multiSubjectTemplates = new Set(["GridTemplate", "ScrollTemplate"]);
  const hasMulti = segment.shots.length > 1;

  return (
    <Modal
      title={`重跑 segment：${segment.segment_id}`}
      open={open}
      onCancel={onCancel}
      onOk={() => onConfirm(frameType, isTransition ? transitionBg : undefined)}
      confirmLoading={loading}
      okText="确认重跑"
      cancelText="取消"
      destroyOnClose
      width={hasMulti ? 720 : 500}
    >
      <Space direction="vertical" style={{ width: "100%" }}>
        <Space>
          {remotionId ? <Tag>{remotionId}</Tag> : null}
          {hasMulti ? <Tag color="orange">{segment.shots.length} 主体</Tag> : null}
          <Radio.Group value={frameType} onChange={(e) => setFrameType(e.target.value)}>
            <Radio.Button value="start">首帧</Radio.Button>
            <Radio.Button value="end">尾帧</Radio.Button>
          </Radio.Group>
        </Space>

        <Row gutter={[8, 8]}>
          {segment.shots.map((shot) => (
            <Col span={hasMulti ? 12 : 24} key={shot.shot_id}>
              <div style={{ position: "relative" }}>
                <Typography.Text style={{ fontSize: 11, position: "absolute", top: 4, left: 6, zIndex: 1, background: "rgba(0,0,0,0.5)", color: "#fff", padding: "0 6px", borderRadius: 3 }}>
                  {shot.shot_id}
                </Typography.Text>
                {(frameType === "start" && shot.keyframe_start_url) || (frameType === "end" && shot.keyframe_end_url) ? (
                  <Image
                    src={frameType === "start" ? shot.keyframe_start_url : shot.keyframe_end_url}
                    alt={frameType}
                    style={{ width: "100%", maxHeight: 320, objectFit: "contain", borderRadius: 4 }}
                    preview={{ mask: "预览" }}
                  />
                ) : (
                  <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", background: "#fafafa", border: "1px dashed #d9d9d9", borderRadius: 4 }}>
                    <Typography.Text type="secondary">无可预览帧</Typography.Text>
                  </div>
                )}
              </div>
            </Col>
          ))}
        </Row>

        {isTransition ? (
          <Space direction="vertical" style={{ width: "100%" }}>
            <Alert type="info" showIcon message={`使用 ${transitionBg === "white" ? "白屏" : "黑屏"} + 当前 segment 选择帧进行过渡渲染。`} />
            <Radio.Group value={transitionBg} onChange={(e) => setTransitionBg(e.target.value)}>
              <Radio.Button value="white">白屏</Radio.Button>
              <Radio.Button value="black">黑屏</Radio.Button>
            </Radio.Group>
          </Space>
        ) : null}
        {remotionId && multiSubjectTemplates.has(remotionId) ? (
          <Alert type="info" showIcon message={`多主体模板：使用全部 ${segment.shots.length} 个 shot 的对应帧。`} />
        ) : null}
      </Space>
    </Modal>
  );
}

function SegmentVideoPlayer({
  taskId,
  segmentId,
  segmentVideo,
  watchFast,
}: {
  taskId: string;
  segmentId: string;
  segmentVideo?: TaskModuleDSegmentVideoFile;
  watchFast: boolean;
}) {
  const basePath = useMemo(
    () => resolveSegmentVideoBasePath(taskId, segmentId),
    [taskId, segmentId],
  );
  const pollMs = watchFast ? 1000 : 2000;
  const hasApiMeta = Boolean(segmentVideo?.exists);
  const [localTick, setLocalTick] = useState(() => Date.now());

  const { data: probedExists } = useQuery({
    queryKey: ["segment-video-probe", taskId, segmentId],
    queryFn: () => probeSegmentVideoExists(basePath),
    enabled: Boolean(basePath) && !hasApiMeta,
    staleTime: 0,
    refetchInterval: pollMs,
  });

  useEffect(() => {
    if (hasApiMeta) {
      return;
    }
    const timerId = window.setInterval(() => setLocalTick(Date.now()), pollMs);
    return () => window.clearInterval(timerId);
  }, [hasApiMeta, pollMs]);

  const cacheToken = segmentVideo?.mtime ?? localTick;
  const videoUrl = buildSegmentVideoUrl(basePath, cacheToken);
  const showVideo = hasApiMeta || probedExists === true;

  if (!basePath || !showVideo) {
    return (
      <div
        style={{
          maxWidth: 400,
          height: 100,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#fafafa",
          border: "1px dashed #d9d9d9",
          borderRadius: 4,
        }}
      >
        <Typography.Text type="secondary">无合成视频</Typography.Text>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 400 }}>
      <video
        key={videoUrl}
        src={videoUrl}
        controls
        preload="metadata"
        style={{ width: "100%", height: "auto", background: "#000", borderRadius: 4 }}
      >
        您的浏览器不支持视频播放。
      </video>
    </div>
  );
}

function SegmentCard({
  taskId,
  segment,
  segmentVideo,
  watchFast,
  onRerun,
  onRerunBoth,
  onToonCrafter,
  onRemotion,
  rerunBothLoading,
  tooncrafterLoading,
  remotionLoading,
}: {
  taskId: string;
  segment: TaskModuleDSegment;
  segmentVideo?: TaskModuleDSegmentVideoFile;
  watchFast: boolean;
  onRerun: (seg: TaskModuleDSegment) => void;
  onRerunBoth: (seg: TaskModuleDSegment) => void;
  onToonCrafter: (seg: TaskModuleDSegment) => void;
  onRemotion: (seg: TaskModuleDSegment) => void;
  rerunBothLoading: boolean;
  tooncrafterLoading: boolean;
  remotionLoading: boolean;
}) {
  const remotionId = segment.remotion_id;
  const basePath = resolveSegmentVideoBasePath(taskId, segment.segment_id);
  const hasVideo = Boolean(segmentVideo?.exists) || Boolean(basePath);
  const allDone = segment.status ? segment.status === "done" : segment.shots.every((s) => s.status === "done");
  const anyVideo = hasVideo || segment.shots.some((s) => s.video_url);
  const segStatus = allDone ? "done" : anyVideo ? "running" : "pending";

  const hasMultiSubjects = segment.shots.length > 1;

  // segment 级别时间可能为 0（旧任务），从 shot 数据回退
  const segStart = segment.start_time || segment.shots[0]?.start_time || 0;
  const segEnd = segment.end_time || segment.shots[0]?.end_time || 0;
  const segDuration = segment.duration || (segEnd > segStart ? segEnd - segStart : 0);

  return (
    <Card
      size="small"
      title={
        <Space size={6}>
          <PlaySquareOutlined />
          <Typography.Text strong style={{ fontSize: 13 }}>{segment.segment_id}</Typography.Text>
          {remotionId ? <Tag color="blue" style={{ fontSize: 10, lineHeight: "16px" }}>{remotionId}</Tag> : null}
          <TaskStatusTag status={segStatus} />
          {hasMultiSubjects ? (
            <Tag color="orange" style={{ fontSize: 10, lineHeight: "16px" }}>{segment.shots.length} 主体</Tag>
          ) : null}
          {segStatus === "done" && segDuration ? (
            <Typography.Text style={{ fontSize: 11, color: "#999" }}>
              {segStart.toFixed(2)}~{segEnd.toFixed(2)}s
            </Typography.Text>
          ) : null}
        </Space>
      }
    >
      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          {segment.scene_desc_zh ? (
            <Typography.Paragraph
              type="secondary"
              style={{ fontSize: 11, marginBottom: 8 }}
              ellipsis={{ rows: 1, expandable: true }}
            >
              {segment.scene_desc_zh}
            </Typography.Paragraph>
          ) : null}

          <SegmentVideoPlayer
            taskId={taskId}
            segmentId={segment.segment_id}
            segmentVideo={segmentVideo}
            watchFast={watchFast}
          />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, justifyContent: "flex-start" }}>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => onRerunBoth(segment)}
            loading={rerunBothLoading}
          >
            首尾帧
          </Button>
          <Button
            type="primary"
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => onRerun(segment)}
          >
            单帧
          </Button>
          <Button
            size="small"
            icon={<ExperimentOutlined />}
            onClick={() => onToonCrafter(segment)}
            loading={tooncrafterLoading}
          >
            ToonCrafter
          </Button>
          <Button
            size="small"
            icon={<ExperimentOutlined />}
            onClick={() => onRemotion(segment)}
            loading={remotionLoading}
          >
            重渲
          </Button>
        </div>
      </div>
    </Card>
  );
}

function _modeRadio(label: string, value: string, savedMode: string, hasSaved: boolean) {
  const suffix = hasSaved ? (value === savedMode ? "（当前）" : "") : (value === "slow" ? "（默认）" : "");
  return (
    <Radio value={value}>
      <Typography.Text strong>{label}{suffix}</Typography.Text>
    </Radio>
  );
}

interface SegmentSelectorModalProps {
  open: boolean;
  segments: TaskModuleDSegment[];
  selectedSegments: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;
  onCancel: () => void;
  onConfirm: () => void;
  loading: boolean;
  audioCandidates: AudioCandidate[];
  selectedAudioPath: string;
  onAudioPathChange: (path: string) => void;
}

function SegmentSelectorModal({
  open,
  segments,
  selectedSegments,
  onSelectionChange,
  onCancel,
  onConfirm,
  loading,
  audioCandidates,
  selectedAudioPath,
  onAudioPathChange,
}: SegmentSelectorModalProps) {
  const bigSegments = useMemo(() => {
    const map = new Map<string, TaskModuleDSegment[]>();
    for (const seg of segments) {
      const bigId = seg.big_segment_id || "unknown";
      if (!map.has(bigId)) map.set(bigId, []);
      map.get(bigId)!.push(seg);
    }
    return Array.from(map.entries());
  }, [segments]);

  const allSelected = segments.length > 0 && segments.every((s) => selectedSegments.has(s.segment_id));

  const toggleSegment = (sid: string) => {
    const next = new Set(selectedSegments);
    if (next.has(sid)) next.delete(sid);
    else next.add(sid);
    onSelectionChange(next);
  };

  const toggleAll = () => {
    if (allSelected) {
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(segments.map((s) => s.segment_id)));
    }
  };

  const toggleBig = (bigId: string, segIds: string[]) => {
    const bigAll = segIds.every((sid) => selectedSegments.has(sid));
    const next = new Set(selectedSegments);
    for (const sid of segIds) {
      if (bigAll) next.delete(sid);
      else next.add(sid);
    }
    onSelectionChange(next);
  };

  const humanSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  return (
    <Modal
      title="选择 segment 输出成片"
      open={open}
      onCancel={onCancel}
      onOk={onConfirm}
      confirmLoading={loading}
      okText={`确认输出（${selectedSegments.size} 段）`}
      cancelText="取消"
      destroyOnClose
      width={560}
    >
      <Space direction="vertical" style={{ width: "100%" }}>
        <Typography.Text strong style={{ fontSize: 13 }}>选择片段</Typography.Text>
        <Checkbox checked={allSelected} onChange={toggleAll}>
          <Typography.Text strong>全选（{segments.length} 段）</Typography.Text>
        </Checkbox>
        {bigSegments.length > 0 ? (
          bigSegments.map(([bigId, segs]) => {
            const bigAll = segs.every((s) => selectedSegments.has(s.segment_id));
            const someSelected = segs.some((s) => selectedSegments.has(s.segment_id));
            return (
              <Card key={bigId} size="small" style={{ width: "100%" }}>
                <Checkbox
                  checked={bigAll}
                  indeterminate={!bigAll && someSelected}
                  onChange={() => toggleBig(bigId, segs.map((s) => s.segment_id))}
                >
                  <Typography.Text strong code style={{ fontSize: 12 }}>{bigId}</Typography.Text>
                  <Typography.Text style={{ fontSize: 12, marginLeft: 6 }}>{segs.length} 段</Typography.Text>
                </Checkbox>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 6, marginLeft: 24 }}>
                  {segs.map((seg) => (
                    <Checkbox
                      key={seg.segment_id}
                      checked={selectedSegments.has(seg.segment_id)}
                      onChange={() => toggleSegment(seg.segment_id)}
                    >
                      <Typography.Text code style={{ fontSize: 11 }}>{seg.segment_id}</Typography.Text>
                    </Checkbox>
                  ))}
                </div>
              </Card>
            );
          })
        ) : (
          <Typography.Text type="secondary">无 segment 数据</Typography.Text>
        )}

        <Typography.Text strong style={{ fontSize: 13, marginTop: 8 }}>选择音频</Typography.Text>
        {audioCandidates.length > 0 ? (
          <Radio.Group value={selectedAudioPath} onChange={(e) => onAudioPathChange(e.target.value)} style={{ width: "100%" }}>
            <Space direction="vertical" style={{ width: "100%" }}>
              {audioCandidates.map((ac) => (
                <Radio key={ac.path} value={ac.path}>
                  <Typography.Text style={{ fontSize: 12 }}>{ac.label}</Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 11, marginLeft: 8 }}>{humanSize(ac.size_bytes)}</Typography.Text>
                  {ac.default ? <Tag color="blue" style={{ marginLeft: 6, fontSize: 10 }}>默认</Tag> : null}
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        ) : (
          <Typography.Text type="secondary">未找到音频候选，将使用任务记录的音频路径</Typography.Text>
        )}
      </Space>
    </Modal>
  );
}

interface FrameModeModalProps {
  segment: TaskModuleDSegment;
  action: "tooncrafter" | "remotion";
  frameMode: string;
  shotModes: Record<string, "slow" | "pingpong" | "holdtail">;
  transitionBg: string;
  savedMode: string;
  hasSavedMode: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  onFrameModeChange: (mode: "slow" | "pingpong" | "holdtail") => void;
  onShotModeChange: (shotId: string, mode: "slow" | "pingpong" | "holdtail") => void;
  onTransitionBgChange: (bg: string) => void;
  loading: boolean;
}

function FrameModeModal({
  segment,
  action,
  frameMode,
  shotModes,
  transitionBg,
  savedMode,
  hasSavedMode,
  onCancel,
  onConfirm,
  onFrameModeChange,
  onShotModeChange,
  onTransitionBgChange,
  loading,
}: FrameModeModalProps) {
  const isMulti = segment.shots.length > 1 && new Set(["GridTemplate", "ScrollTemplate"]).has(segment.remotion_id || "");
  const isTransition = segment.remotion_id && new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]).has(segment.remotion_id);
  const desc = action === "tooncrafter"
    ? "ToonCrafter 将在首尾帧之间生成 16 帧插值序列，然后由 Remotion 渲染为最终视频。"
    : "复用已有 ToonCrafter 帧，直接由 Remotion 渲染。";

  return (
    <Modal
      title={`${action === "tooncrafter" ? "ToonCrafter" : "Remotion 重渲"}：${segment.segment_id}`}
      open
      onCancel={onCancel}
      onOk={onConfirm}
      confirmLoading={loading}
      okText="确认执行"
      cancelText="取消"
      destroyOnClose
      width={isMulti ? 600 : 500}
    >
      <Space direction="vertical" style={{ width: "100%" }}>
        <Alert type="info" showIcon message={`${desc}选择帧填充模式：`} />

        {isMulti ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {segment.shots.map((shot) => {
              const shotMode = shotModes[shot.shot_id] || frameMode as any;
              return (
                <Card key={shot.shot_id} size="small" title={<Typography.Text code style={{ fontSize: 12 }}>{shot.shot_id}</Typography.Text>}>
                  <Radio.Group
                    value={shotMode}
                    onChange={(e) => onShotModeChange(shot.shot_id, e.target.value)}
                  >
                    <Space direction="vertical">
                      {_modeRadio("慢放", "slow", savedMode, hasSavedMode)}
                      {_modeRadio("Ping-pong 循环", "pingpong", savedMode, hasSavedMode)}
                      {_modeRadio("尾帧保持", "holdtail", savedMode, hasSavedMode)}
                    </Space>
                  </Radio.Group>
                </Card>
              );
            })}
          </div>
        ) : (
          <Radio.Group value={frameMode} onChange={(e) => onFrameModeChange(e.target.value)}>
            <Space direction="vertical">
              {_modeRadio("慢放", "slow", savedMode, hasSavedMode)}
              {_modeRadio("Ping-pong 循环", "pingpong", savedMode, hasSavedMode)}
              {_modeRadio("尾帧保持", "holdtail", savedMode, hasSavedMode)}
            </Space>
          </Radio.Group>
        )}

          {isTransition ? (
            <>
              <div style={{ borderTop: "1px solid #f0f0f0", marginTop: 8 }} />
              <Alert type="warning" showIcon message="转场模板需要选择前半段的背景来源。" />
              <Typography.Text strong style={{ fontSize: 13 }}>前半段前一帧来源：</Typography.Text>
              <Radio.Group value={transitionBg} onChange={(e) => onTransitionBgChange(e.target.value)}>
                <Space direction="vertical">
                  <Radio value="">上一个 segment 尾帧</Radio>
                  <Radio value="white">白屏</Radio>
                  <Radio value="black">黑屏</Radio>
                </Space>
              </Radio.Group>
            </>
          ) : null}
      </Space>
    </Modal>
  );
}

function _fmtNow(): string {
  const now = new Date();
  const y = now.getFullYear();
  const M = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  const h = String(now.getHours()).padStart(2, "0");
  const m = String(now.getMinutes()).padStart(2, "0");
  const s = String(now.getSeconds()).padStart(2, "0");
  return `${y}-${M}-${d} ${h}:${m}:${s}`;
}

export function TaskModuleDPage() {
  const taskId = useTaskIdParam();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [rerunTarget, setRerunTarget] = useState<TaskModuleDSegment | null>(null);
  const [rerunBothTarget, setRerunBothTarget] = useState<TaskModuleDSegment | null>(null);
  const [rerunBothTransitionBg, setRerunBothTransitionBg] = useState<"white" | "black">();
  const [frameModeTarget, setFrameModeTarget] = useState<{segment: TaskModuleDSegment; action: "tooncrafter" | "remotion"} | null>(null);
  const [frameMode, setFrameMode] = useState<"slow" | "pingpong" | "holdtail">("slow");
  const [shotModes, setShotModes] = useState<Record<string, "slow" | "pingpong" | "holdtail">>({});
  const [transitionBg, setTransitionBg] = useState("");
  const [savedMode, setSavedMode] = useState("slow");
  const [hasSavedMode, setHasSavedMode] = useState(false);

  const { data } = useQuery({
    queryKey: taskQueryKeys.moduleD(taskId),
    queryFn: () => getTaskModuleDData(taskId),
    staleTime: 0,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) return 5000;
      if (payload.active_rerun?.active) return 3000;
      if (payload.active_rerun) return 5000;
      // 成片输出已提交但未收到完成态：强制轮询避免错过更新
      if (rebuildSubmitted && !rebuildFinalVideoUrl) return 5000;
      if (payload.module_d_status === "running") return 5000;
      return false;
    },
  });

  const activeRerun = data?.active_rerun;

  const { data: segmentVideos } = useQuery({
    queryKey: taskQueryKeys.moduleDSegmentVideos(taskId),
    queryFn: () => getModuleDSegmentVideos(taskId),
    enabled: Boolean(taskId),
    staleTime: 0,
    refetchInterval: activeRerun?.active ? 800 : 2000,
  });

  const rerunMutation = useMutation({
    mutationFn: ({
      segmentId,
      frameType,
      transitionBg,
    }: {
      segmentId: string;
      frameType: "start" | "end";
      transitionBg?: "white" | "black";
    }) => rerunModuleDSegment(taskId, segmentId, frameType, transitionBg),
    onSuccess: (_data, variables) => {
      setRerunTarget(null);
      queryClient.setQueryData<TaskModuleDData>(taskQueryKeys.moduleD(taskId), (old) => {
        if (!old) return old;
        return {
          ...old,
          active_rerun: {
            active: true,
            status: "queued",
            segment_id: variables.segmentId,
            frame_type: variables.frameType,
            submitted_at: _fmtNow(),
            submitted_at_ms: Date.now(),
            started_at_ms: 0,
            last_error: "",
            failure_reason: "",
          },
        };
      });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const rerunBothMutation = useMutation({
    mutationFn: ({
      segmentId,
      transitionBg,
    }: {
      segmentId: string;
      transitionBg?: "white" | "black";
    }) => rerunModuleDBothFrames(taskId, segmentId, transitionBg),
    onSuccess: (_data, variables) => {
      setRerunBothTarget(null);
      queryClient.setQueryData<TaskModuleDData>(taskQueryKeys.moduleD(taskId), (old) => {
        if (!old) return old;
        return {
          ...old,
          active_rerun: {
            active: true,
            status: "queued",
            segment_id: variables.segmentId,
            frame_type: "both",
            submitted_at: _fmtNow(),
            submitted_at_ms: Date.now(),
            started_at_ms: 0,
            last_error: "",
            failure_reason: "",
          },
        };
      });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  // 打开 modal 时加载该 segment 保存的模式
  useEffect(() => {
    if (frameModeTarget) {
      const seg = frameModeTarget.segment;
      const isMulti = seg.shots.length > 1 && new Set(["GridTemplate", "ScrollTemplate"]).has(seg.remotion_id || "");
      getToonCrafterMode(taskId, seg.segment_id).then((result) => {
        if (result.modes && isMulti) {
          const sm: Record<string, "slow" | "pingpong" | "holdtail"> = {};
          for (const shot of seg.shots) {
            const saved = result.modes[shot.shot_id];
            sm[shot.shot_id] = (saved === "slow" || saved === "pingpong" || saved === "holdtail") ? saved : result.mode as any;
          }
          setShotModes(sm);
        } else {
          setShotModes({});
        }
        if (result.mode === "slow" || result.mode === "pingpong" || result.mode === "holdtail") {
          setFrameMode(result.mode);
        }
        setSavedMode(result.mode);
        setHasSavedMode(true);
      });
    }
  }, [frameModeTarget, taskId]);

  const rerunToonCrafterMutation = useMutation({
    mutationFn: ({
      segmentId,
      mode,
      transitionBg,
    }: {
      segmentId: string;
      mode?: string;
      transitionBg?: string;
    }) => rerunModuleDSegmentToonCrafter(taskId, segmentId, mode, transitionBg),
    onSuccess: (_data, variables) => {
      setFrameModeTarget(null);
      queryClient.setQueryData<TaskModuleDData>(taskQueryKeys.moduleD(taskId), (old) => {
        if (!old) return old;
        return {
          ...old,
          active_rerun: {
            active: true,
            status: "queued",
            segment_id: variables.segmentId,
            frame_type: "tooncrafter",
            submitted_at: _fmtNow(),
            submitted_at_ms: Date.now(),
            started_at_ms: 0,
            last_error: "",
            failure_reason: "",
          },
        };
      });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const rerunModuleAllToonCrafterMutation = useMutation({
    mutationFn: () => rerunModuleDAllToonCrafter(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const rerunRemotionMutation = useMutation({
    mutationFn: ({ segmentId, mode, transitionBg }: { segmentId: string; mode?: string; transitionBg?: string }) => rerunModuleDSegmentRemotion(taskId, segmentId, mode, transitionBg),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const rerunModuleAllRemotionMutation = useMutation({
    mutationFn: () => rerunModuleDAllRemotion(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const [rebuildSelectionOpen, setRebuildSelectionOpen] = useState(false);
  const [selectedSegments, setSelectedSegments] = useState<Set<string>>(new Set());
  const [rebuildFinalVideoUrl, setRebuildFinalVideoUrl] = useState<string | null>(null);
  const [rebuildSubmitted, setRebuildSubmitted] = useState(false);
  const [audioCandidates, setAudioCandidates] = useState<AudioCandidate[]>([]);
  const [selectedAudioPath, setSelectedAudioPath] = useState<string>("");
  const rebuildMutation = useMutation({
    mutationFn: ({ segmentIds, audioPath }: { segmentIds: string[]; audioPath?: string }) =>
      rebuildModuleDFinal(taskId, segmentIds, audioPath),
    onSuccess: (data) => {
      const payload = data as Record<string, unknown>;
      if (payload?.ok === true) {
        setRebuildSubmitted(true);
        setRebuildFinalVideoUrl(null);
      } else {
        const videoUrl = typeof payload?.video_url === "string" ? payload.video_url : null;
        if (videoUrl) {
          setRebuildFinalVideoUrl(videoUrl);
          setRebuildSubmitted(false);
        }
      }
    },
  });

  // 轮询检测成片完成
  useEffect(() => {
    if (activeRerun?.frame_type === "rebuild_final") {
      if (activeRerun.video_url) {
        setRebuildFinalVideoUrl(activeRerun.video_url);
        setRebuildSubmitted(false);
      } else if (activeRerun.status === "done") {
        setRebuildFinalVideoUrl(`/task/${taskId}/final_output.mp4?t=${Date.now()}`);
        setRebuildSubmitted(false);
      } else if (activeRerun.status === "failed") {
        setRebuildSubmitted(false);
      }
    }
  }, [activeRerun, taskId]);

  const rerunModuleMutation = useMutation({
    mutationFn: ({
      frameType,
    }: {
      frameType: "start" | "end" | "both";
    }) => rerunModuleDAll(taskId, frameType),
    onSuccess: (_data, variables) => {
      queryClient.setQueryData<TaskModuleDData>(taskQueryKeys.moduleD(taskId), (old) => {
        if (!old) return old;
        return {
          ...old,
          active_rerun: {
            active: true,
            status: "queued",
            frame_type: variables.frameType,
            submitted_at: _fmtNow(),
            submitted_at_ms: Date.now(),
            started_at_ms: 0,
            last_error: "",
            failure_reason: "",
          },
        };
      });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const unitSummary = data?.unit_summary;
  const segments = data?.segments ?? [];

  const handleRerun = (segment: TaskModuleDSegment) => {
    setRerunTarget(segment);
  };

  const handleRerunConfirm = (frameType: "start" | "end", transitionBg?: "white" | "black") => {
    if (!rerunTarget) return;
    rerunMutation.mutate({
      segmentId: rerunTarget.segment_id,
      frameType,
      transitionBg,
    });
  };

  const handleRerunBoth = (segment: TaskModuleDSegment) => {
    const remotionId = segment.remotion_id || "";
    const isTransition = new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]).has(remotionId);
    if (isTransition) {
      setRerunBothTarget(segment);
      setRerunBothTransitionBg(undefined);
    } else {
      rerunBothMutation.mutate({ segmentId: segment.segment_id });
    }
  };

  const handleToonCrafter = (segment: TaskModuleDSegment) => {
    setFrameModeTarget({ segment, action: "tooncrafter" });
  };

  const handleRemotion = (segment: TaskModuleDSegment) => {
    setFrameModeTarget({ segment, action: "remotion" });
  };

  return (
    <div className="module-d-page">
      <Typography.Title level={3}>Module D — 视频合成</Typography.Title>

      {activeRerun?.status === "failed" ? (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message={`最近一次重跑失败：${activeRerun.failure_reason || "未知原因"}`}
          description={activeRerun.last_error || "请检查后台日志。"}
        />
      ) : null}

      {activeRerun?.active ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={`当前正在重跑：${activeRerun.segment_id || "-"}`}
          description={`提交时间：${activeRerun.submitted_at || "-"}`}
        />
      ) : null}

      {unitSummary ? (
        <Space wrap style={{ marginBottom: 16 }}>
          <Tag>total: {unitSummary.total_units}</Tag>
          <Tag color="green">done: {unitSummary.status_counts?.done ?? 0}</Tag>
          <Tag color="blue">running: {unitSummary.status_counts?.running ?? 0}</Tag>
          <Tag>pending: {unitSummary.status_counts?.pending ?? 0}</Tag>
          <Tag color="red">failed: {unitSummary.status_counts?.failed ?? 0}</Tag>
          {activeRerun?.active ? (
            <Tag color="processing">重跑中: {activeRerun.segment_id}</Tag>
          ) : null}
          {activeRerun?.status === "failed" ? <Tag color="error">最近重跑失败</Tag> : null}
        </Space>
      ) : null}

      <Space wrap style={{ marginBottom: 16 }}>
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={rerunModuleMutation.isPending}
          onClick={() => rerunModuleMutation.mutate({ frameType: "both" })}
        >
          首尾帧重跑 module
        </Button>
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={rerunModuleMutation.isPending}
          onClick={() => rerunModuleMutation.mutate({ frameType: "start" })}
        >
          按首帧重跑 module
        </Button>
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={rerunModuleMutation.isPending}
          onClick={() => rerunModuleMutation.mutate({ frameType: "end" })}
        >
          按尾帧重跑 module
        </Button>
        <Button
          size="small"
          icon={<ExperimentOutlined />}
          loading={rerunModuleAllToonCrafterMutation.isPending}
          onClick={() => rerunModuleAllToonCrafterMutation.mutate()}
        >
          ToonCrafter 重跑 module
        </Button>
        <Button
          size="small"
          icon={<ExperimentOutlined />}
          loading={rerunModuleAllRemotionMutation.isPending}
          onClick={() => rerunModuleAllRemotionMutation.mutate()}
        >
          Remotion 重渲 module
        </Button>
        <Button
          size="small"
          icon={<ExperimentOutlined />}
          onClick={() => {
            setSelectedSegments(new Set(segments.map((s) => s.segment_id)));
            setRebuildFinalVideoUrl(null);
            setRebuildSubmitted(false);
            getRebuildAudioCandidates(taskId).then((res) => {
              setAudioCandidates(res.candidates);
              setSelectedAudioPath(res.defaultPath);
            });
            setRebuildSelectionOpen(true);
          }}
        >
          输出成片
        </Button>
      </Space>

      {rebuildSubmitted && !rebuildFinalVideoUrl ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="成片输出已提交"
          description="后台正在拼接片段并混入音频，请稍候..."
        />
      ) : null}
      {rebuildFinalVideoUrl ? (
        <Alert
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
          message="成片已生成"
          description={
            <span>
              请前往<Button type="link" size="small" onClick={() => navigate(`/tasks/${taskId}/review`)}>审阅页面</Button>查看最终效果。
            </span>
          }
        />
      ) : null}

      <SegmentSelectorModal
        open={rebuildSelectionOpen}
        segments={segments}
        selectedSegments={selectedSegments}
        onSelectionChange={setSelectedSegments}
        onCancel={() => setRebuildSelectionOpen(false)}
        onConfirm={() => {
          if (selectedSegments.size > 0) {
            const ap = selectedAudioPath || undefined;
            rebuildMutation.mutate({ segmentIds: Array.from(selectedSegments), audioPath: ap });
          }
          setRebuildSelectionOpen(false);
        }}
        loading={rebuildMutation.isPending}
        audioCandidates={audioCandidates}
        selectedAudioPath={selectedAudioPath}
        onAudioPathChange={setSelectedAudioPath}
      />

      {segments.length === 0 ? (
        <Typography.Text type="secondary">暂无 segment 数据，请先执行模块 B。</Typography.Text>
      ) : (
        <Row gutter={[16, 16]}>
          {segments.map((segment) => (
            <Col span={12} key={segment.segment_id}>
              <SegmentCard
                taskId={taskId}
                segment={segment}
                segmentVideo={segmentVideos?.items?.[segment.segment_id]}
                watchFast={
                  Boolean(activeRerun?.active)
                  && (activeRerun?.segment_id === segment.segment_id || !activeRerun?.segment_id)
                }
                onRerun={handleRerun}
                onRerunBoth={handleRerunBoth}
                onToonCrafter={handleToonCrafter}
                onRemotion={handleRemotion}
                rerunBothLoading={rerunBothMutation.isPending}
                tooncrafterLoading={rerunToonCrafterMutation.isPending}
                remotionLoading={rerunRemotionMutation.isPending}
              />
            </Col>
          ))}
        </Row>
      )}

      {rerunTarget ? (
        <RerunSegmentModal
          open
          segment={rerunTarget}
          onCancel={() => setRerunTarget(null)}
          onConfirm={handleRerunConfirm}
          loading={rerunMutation.isPending}
        />
      ) : null}

      {frameModeTarget ? (
        <FrameModeModal
          segment={frameModeTarget.segment}
          action={frameModeTarget.action}
          frameMode={frameMode}
          shotModes={shotModes}
          transitionBg={transitionBg}
          savedMode={savedMode}
          hasSavedMode={hasSavedMode}
          onCancel={() => {
            setFrameModeTarget(null);
            setHasSavedMode(false);
          }}
          onTransitionBgChange={(bg) => setTransitionBg(bg)}
          onConfirm={() => {
            const segId = frameModeTarget.segment.segment_id;
            const isMulti = frameModeTarget.segment.shots.length > 1
              && new Set(["GridTemplate", "ScrollTemplate"]).has(frameModeTarget.segment.remotion_id || "");
            if (isMulti) {
              for (const shot of frameModeTarget.segment.shots) {
                const sm = shotModes[shot.shot_id] || frameMode;
                setToonCrafterShotMode(taskId, segId, shot.shot_id, sm);
              }
            } else {
              setToonCrafterMode(taskId, segId, frameMode);
            }
            const tb = transitionBg || undefined;
            if (frameModeTarget.action === "tooncrafter") {
              rerunToonCrafterMutation.mutate({ segmentId: segId, mode: frameMode, transitionBg: tb });
            } else {
              rerunRemotionMutation.mutate({ segmentId: segId, mode: frameMode, transitionBg: tb });
            }
            setFrameModeTarget(null);
          }}
          onFrameModeChange={(m) => setFrameMode(m)}
          onShotModeChange={(shotId, m) => setShotModes((prev) => ({ ...prev, [shotId]: m }))}
          loading={rerunToonCrafterMutation.isPending || rerunRemotionMutation.isPending}
        />
      ) : null}

      {rerunBothTarget && new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]).has(rerunBothTarget.remotion_id || "") ? (
        <Modal
          title={`首尾帧重跑：${rerunBothTarget.segment_id}（过渡模板）`}
          open
          onCancel={() => setRerunBothTarget(null)}
          onOk={() => {
            rerunBothMutation.mutate({
              segmentId: rerunBothTarget.segment_id,
              transitionBg: rerunBothTransitionBg,
            });
          }}
          confirmLoading={rerunBothMutation.isPending}
          okText="确认重跑"
          cancelText="取消"
          destroyOnClose
        >
          <Space direction="vertical" style={{ width: "100%" }}>
            <Alert type="info" showIcon message="过渡模板需要选择前半段前一帧的来源。" />
            <div>
              <Typography.Text strong>前半段前一帧：</Typography.Text>
              <Radio.Group
                value={rerunBothTransitionBg || "auto"}
                onChange={(e) => {
                  setRerunBothTransitionBg(e.target.value === "auto" ? undefined : e.target.value as "white" | "black");
                }}
                style={{ marginTop: 8 }}
              >
                <Radio.Button value="auto">上一个 segment 尾帧</Radio.Button>
                <Radio.Button value="white">白屏</Radio.Button>
                <Radio.Button value="black">黑屏</Radio.Button>
              </Radio.Group>
            </div>
          </Space>
        </Modal>
      ) : null}
    </div>
  );
}
