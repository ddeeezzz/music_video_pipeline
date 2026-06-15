import { useEffect, useMemo, useState } from "react";
import {
  PlaySquareOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
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
  rerunModuleDBatchToonCrafter,
  rerunModuleDSegmentRemotion,
  rerunModuleDAllRemotion,
  rerunModuleDBatchRerender,
  resumeModuleD,
  setToonCrafterMode,
  setToonCrafterShotMode,
  taskQueryKeys,
} from "@/api/taskApi";
import type { AudioCandidate, BatchRerenderSegmentConfig } from "@/api/taskApi";
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
  const [transitionBg, setTransitionBg] = useState("");
  const remotionId = segment.remotion_id || "";
  const isTransition = remotionId && new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]).has(remotionId);
  const multiSubjectTemplates = new Set(["GridTemplate", "ScrollTemplate"]);
  const hasMulti = segment.shots.length > 1;

  return (
    <Modal
      title={`重跑 segment：${segment.segment_id}`}
      open={open}
      onCancel={onCancel}
      onOk={() => onConfirm(frameType, isTransition ? (transitionBg || undefined) as "white" | "black" | undefined : undefined)}
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
            <Alert type="info" showIcon message={
              transitionBg === ""
                ? "使用上一个 segment 尾帧 + 当前 segment 选择帧进行过渡渲染。"
                : `使用 ${transitionBg === "white" ? "白屏" : "黑屏"} + 当前 segment 选择帧进行过渡渲染。`
            } />
            <Radio.Group value={transitionBg} onChange={(e) => setTransitionBg(e.target.value)}>
              <Radio.Button value="">上一个 segment 尾帧</Radio.Button>
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
  phase,
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
  phase?: string;
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
          {phase === "tooncrafter" ? <Tag color="purple" style={{ fontSize: 10, lineHeight: "16px" }}>ToonCrafter</Tag> : null}
          {phase === "remotion" ? <Tag color="blue" style={{ fontSize: 10, lineHeight: "16px" }}>Remotion</Tag> : null}
          {phase === "rebuild" ? <Tag color="orange" style={{ fontSize: 10, lineHeight: "16px" }}>拼接中</Tag> : null}
          {hasMultiSubjects ? (
            <Tag color="orange" style={{ fontSize: 10, lineHeight: "16px" }}>{segment.shots.length} 主体</Tag>
          ) : null}
          {segStatus === "done" && segDuration ? (
            <Typography.Text style={{ fontSize: 11, color: "#999" }}>
              {segStart.toFixed(2)}~{segEnd.toFixed(2)}s
            </Typography.Text>
          ) : null}
          {segment.lyrics && segment.lyrics.length > 0 ? (
            <Tag color="purple" style={{ fontSize: 10, lineHeight: "16px", maxWidth: 280 }} title={segment.lyrics.join(" | ")}>
              {segment.lyrics.slice(0, 2).join(" / ")}{segment.lyrics.length > 2 ? "…" : ""}
            </Tag>
          ) : (
            <Typography.Text style={{ fontSize: 11, color: "#bbb" }}>无歌词</Typography.Text>
          )}
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

interface BatchToonCrafterModalProps {
  open: boolean;
  segments: TaskModuleDSegment[];
  selectedSegments: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;
  frameMode: "slow" | "pingpong" | "holdtail";
  onFrameModeChange: (mode: "slow" | "pingpong" | "holdtail") => void;
  onCancel: () => void;
  onConfirm: () => void;
  loading: boolean;
}

function BatchToonCrafterModal({
  open,
  segments,
  selectedSegments,
  onSelectionChange,
  frameMode,
  onFrameModeChange,
  onCancel,
  onConfirm,
  loading,
}: BatchToonCrafterModalProps) {
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

  return (
    <Modal
      title="ToonCrafter 批量重跑"
      open={open}
      onCancel={onCancel}
      onOk={onConfirm}
      confirmLoading={loading}
      okText={`确认重跑（${selectedSegments.size} 段）`}
      cancelText="取消"
      destroyOnClose
      width={580}
    >
      <Space direction="vertical" style={{ width: "100%" }}>
        <Typography.Text strong style={{ fontSize: 13 }}>
          选择需要重跑 ToonCrafter 的片段
        </Typography.Text>

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

        <div style={{ borderTop: "1px solid #f0f0f0", marginTop: 8, paddingTop: 12 }}>
          <Typography.Text strong style={{ fontSize: 13, display: "block", marginBottom: 8 }}>
            选择帧填充模式（适用于所有选中 segment）
          </Typography.Text>
          <Radio.Group value={frameMode} onChange={(e) => onFrameModeChange(e.target.value)}>
            <Space direction="vertical">
              {_modeRadio("慢放", "slow", frameMode, true)}
              {_modeRadio("Ping-pong 循环", "pingpong", frameMode, true)}
              {_modeRadio("尾帧保持", "holdtail", frameMode, true)}
            </Space>
          </Radio.Group>
        </div>
      </Space>
    </Modal>
  );
}

const _TRANSITION_TEMPLATE_SET = new Set(["TiltUpTemplate", "TiltDownTemplate", "PanRightTemplate"]);
const _MULTI_SUBJECT_TEMPLATE_SET = new Set(["GridTemplate", "ScrollTemplate"]);

interface BatchRerenderModalProps {
  open: boolean;
  segments: TaskModuleDSegment[];
  selectedSegments: Set<string>;
  segmentModes: Record<string, "slow" | "pingpong" | "holdtail">;
  segmentTransitionBgs: Record<string, string>;
  onSelectionChange: (selected: Set<string>) => void;
  onSegmentModeChange: (segmentId: string, mode: "slow" | "pingpong" | "holdtail") => void;
  onSegmentTransitionBgChange: (segmentId: string, bg: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
  loading: boolean;
}

function BatchRerenderModal({
  open,
  segments,
  selectedSegments,
  segmentModes,
  segmentTransitionBgs,
  onSelectionChange,
  onSegmentModeChange,
  onSegmentTransitionBgChange,
  onCancel,
  onConfirm,
  loading,
}: BatchRerenderModalProps) {
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

  return (
    <Modal
      title="批量重渲 — 逐 segment 配置"
      open={open}
      onCancel={onCancel}
      onOk={onConfirm}
      confirmLoading={loading}
      okText={`确认重渲（${selectedSegments.size} 段）`}
      cancelText="取消"
      destroyOnClose
      width={900}
    >
      <Space direction="vertical" style={{ width: "100%" }}>
        <Alert type="info" showIcon message="勾选需要重渲的 segment，逐项设置帧填充模式和过渡背景后提交。" />

        <Checkbox checked={allSelected} onChange={toggleAll}>
          <Typography.Text strong>全选（{segments.length} 段）</Typography.Text>
        </Checkbox>

        <div style={{ maxHeight: 500, overflowY: "auto" }}>
          {bigSegments.length > 0 ? (
            bigSegments.map(([bigId, segs]) => {
              const bigAll = segs.every((s) => selectedSegments.has(s.segment_id));
              const someSelected = segs.some((s) => selectedSegments.has(s.segment_id));
              return (
                <Card key={bigId} size="small" style={{ width: "100%", marginBottom: 8 }}>
                  <Checkbox
                    checked={bigAll}
                    indeterminate={!bigAll && someSelected}
                    onChange={() => toggleBig(bigId, segs.map((s) => s.segment_id))}
                  >
                    <Typography.Text strong code style={{ fontSize: 12 }}>{bigId}</Typography.Text>
                    <Typography.Text style={{ fontSize: 12, marginLeft: 6 }}>{segs.length} 段</Typography.Text>
                  </Checkbox>
                  <div style={{ marginTop: 8 }}>
                    {segs.map((seg) => {
                      const checked = selectedSegments.has(seg.segment_id);
                      const isTransition = _TRANSITION_TEMPLATE_SET.has(seg.remotion_id || "");
                      return (
                        <Card
                          key={seg.segment_id}
                          size="small"
                          style={{
                            marginBottom: 6,
                            borderLeft: checked ? "3px solid #1677ff" : "3px solid transparent",
                            opacity: checked ? 1 : 0.6,
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                            <Checkbox checked={checked} onChange={() => toggleSegment(seg.segment_id)}>
                              <Typography.Text code style={{ fontSize: 12 }}>{seg.segment_id}</Typography.Text>
                            </Checkbox>
                            {seg.remotion_id ? <Tag style={{ fontSize: 10 }}>{seg.remotion_id}</Tag> : null}
                            {seg.shots.length > 1 ? <Tag color="orange" style={{ fontSize: 10 }}>{seg.shots.length}主体</Tag> : null}
                          </div>

                          {checked ? (
                            <div style={{ marginTop: 8, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
                              {/* Mode */}
                              <Space size={4}>
                                <Typography.Text style={{ fontSize: 11, color: "#888" }}>模式:</Typography.Text>
                                <Radio.Group
                                  size="small"
                                  value={segmentModes[seg.segment_id] || "slow"}
                                  onChange={(e) => onSegmentModeChange(seg.segment_id, e.target.value)}
                                >
                                  <Radio.Button value="slow" style={{ fontSize: 11, lineHeight: "18px" }}>慢放</Radio.Button>
                                  <Radio.Button value="pingpong" style={{ fontSize: 11, lineHeight: "18px" }}>循环</Radio.Button>
                                  <Radio.Button value="holdtail" style={{ fontSize: 11, lineHeight: "18px" }}>尾帧保持</Radio.Button>
                                </Radio.Group>
                              </Space>

                              {/* Transition bg */}
                              {isTransition ? (
                                <Space size={4}>
                                  <Typography.Text style={{ fontSize: 11, color: "#888" }}>过渡:</Typography.Text>
                                  <Radio.Group
                                    size="small"
                                    value={segmentTransitionBgs[seg.segment_id] || ""}
                                    onChange={(e) => onSegmentTransitionBgChange(seg.segment_id, e.target.value)}
                                  >
                                    <Radio.Button value="" style={{ fontSize: 11, lineHeight: "18px" }}>上段尾帧</Radio.Button>
                                    <Radio.Button value="white" style={{ fontSize: 11, lineHeight: "18px" }}>白屏</Radio.Button>
                                    <Radio.Button value="black" style={{ fontSize: 11, lineHeight: "18px" }}>黑屏</Radio.Button>
                                  </Radio.Group>
                                </Space>
                              ) : null}
                            </div>
                          ) : null}
                        </Card>
                      );
                    })}
                  </div>
                </Card>
              );
            })
          ) : (
            <Typography.Text type="secondary">无 segment 数据</Typography.Text>
          )}
        </div>
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
  const [rebuildSelectionOpen, setRebuildSelectionOpen] = useState(false);
  const [selectedSegments, setSelectedSegments] = useState<Set<string>>(new Set());
  const [rebuildFinalVideoUrl, setRebuildFinalVideoUrl] = useState<string | null>(null);
  const [rebuildSubmitted, setRebuildSubmitted] = useState(false);
  const [audioCandidates, setAudioCandidates] = useState<AudioCandidate[]>([]);
  const [selectedAudioPath, setSelectedAudioPath] = useState<string>("");
  const [batchToonCrafterOpen, setBatchToonCrafterOpen] = useState(false);
  const [batchToonCrafterSegments, setBatchToonCrafterSegments] = useState<Set<string>>(new Set());
  const [batchToonCrafterMode, setBatchToonCrafterMode] = useState<"slow" | "pingpong" | "holdtail">("slow");
  const [batchRerenderOpen, setBatchRerenderOpen] = useState(false);
  const [batchRerenderSegments, setBatchRerenderSegments] = useState<Set<string>>(new Set());
  const [batchRerenderModes, setBatchRerenderModes] = useState<Record<string, "slow" | "pingpong" | "holdtail">>({});
  const [batchRerenderTransitionBgs, setBatchRerenderTransitionBgs] = useState<Record<string, string>>({});

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
      // 有 unit 仍在 running 中：持续轮询
      const unitSummary = payload.unit_summary;
      if (unitSummary?.status_counts?.running && unitSummary.status_counts.running > 0) return 5000;
      return false;
    },
  });

  const activeRerun = data?.active_rerun;

  const { data: segmentVideos } = useQuery({
    queryKey: taskQueryKeys.moduleDSegmentVideos(taskId),
    queryFn: () => getModuleDSegmentVideos(taskId),
    enabled: Boolean(taskId),
    staleTime: 0,
    refetchInterval: activeRerun?.active ? 800 : (data?.module_d_status === "running" ? 2000 : 2000),
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

  const rerunModuleBatchToonCrafterMutation = useMutation({
    mutationFn: ({
      segmentIds,
      mode,
    }: {
      segmentIds: string[];
      mode: string;
    }) => rerunModuleDBatchToonCrafter(taskId, segmentIds, mode),
    onSuccess: () => {
      setBatchToonCrafterOpen(false);
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
  });

  const rerunBatchRerenderMutation = useMutation({
    mutationFn: async (segmentConfigs: BatchRerenderSegmentConfig[]) =>
      rerunModuleDBatchRerender(taskId, segmentConfigs),
    onSuccess: () => {
      setBatchRerenderOpen(false);
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

  const resumeMutation = useMutation({
    mutationFn: () => resumeModuleD(taskId),
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleD(taskId) });
      queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleDSegmentVideos(taskId) });
    },
    onError: () => {
      // 错误由 refetch 自动显示
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
          loading={rerunModuleBatchToonCrafterMutation.isPending}
          onClick={() => {
            setBatchToonCrafterSegments(new Set(segments.map((s) => s.segment_id)));
            setBatchToonCrafterMode("slow");
            setBatchToonCrafterOpen(true);
          }}
        >
          ToonCrafter 批量重跑
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
          icon={<ThunderboltOutlined />}
          loading={rerunBatchRerenderMutation.isPending}
          onClick={() => {
            setBatchRerenderSegments(new Set(segments.map((s) => s.segment_id)));
            setBatchRerenderModes({});
            setBatchRerenderTransitionBgs({});
            setBatchRerenderOpen(true);
          }}
        >
          批量重渲
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
        <Button
          icon={<ReloadOutlined />}
          size="small"
          loading={resumeMutation.isPending}
          onClick={() => {
            Modal.confirm({
              title: "断点续跑 Module D",
              content: "将扫描所有 segment，对缺少视频产物的 segment 逐个补跑。已有产物的 segment 会跳过。",
              onOk: () => resumeMutation.mutate(),
            });
          }}
        >
          断点续跑
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

      <BatchToonCrafterModal
        open={batchToonCrafterOpen}
        segments={segments}
        selectedSegments={batchToonCrafterSegments}
        onSelectionChange={setBatchToonCrafterSegments}
        frameMode={batchToonCrafterMode}
        onFrameModeChange={(m) => setBatchToonCrafterMode(m)}
        onCancel={() => setBatchToonCrafterOpen(false)}
        onConfirm={() => {
          if (batchToonCrafterSegments.size > 0) {
            rerunModuleBatchToonCrafterMutation.mutate({
              segmentIds: Array.from(batchToonCrafterSegments),
              mode: batchToonCrafterMode,
            });
          }
        }}
        loading={rerunModuleBatchToonCrafterMutation.isPending}
      />

      <BatchRerenderModal
        open={batchRerenderOpen}
        segments={segments}
        selectedSegments={batchRerenderSegments}
        segmentModes={batchRerenderModes}
        segmentTransitionBgs={batchRerenderTransitionBgs}
        onSelectionChange={setBatchRerenderSegments}
        onSegmentModeChange={(segmentId, mode) =>
          setBatchRerenderModes((prev) => ({ ...prev, [segmentId]: mode }))
        }
        onSegmentTransitionBgChange={(segmentId, bg) =>
          setBatchRerenderTransitionBgs((prev) => ({ ...prev, [segmentId]: bg }))
        }
        onCancel={() => setBatchRerenderOpen(false)}
        onConfirm={() => {
          if (batchRerenderSegments.size > 0) {
            const configs = Array.from(batchRerenderSegments).map((sid) => ({
              segment_id: sid,
              mode: batchRerenderModes[sid] || "slow",
              transition_bg: batchRerenderTransitionBgs[sid] ?? "",
              action: "remotion" as const,
            })) as BatchRerenderSegmentConfig[];
            rerunBatchRerenderMutation.mutate(configs);
          }
        }}
        loading={rerunBatchRerenderMutation.isPending}
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
                phase={
                  activeRerun?.active
                  && (activeRerun?.segment_id === segment.segment_id || !activeRerun?.segment_id)
                    ? activeRerun?.phase
                    : undefined
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
