import { useEffect, useMemo, useState } from "react";
import {
  PlaySquareOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Col,
  Image,
  Modal,
  Radio,
  Row,
  Space,
  Tag,
  Typography,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getModuleDSegmentVideos,
  getTaskModuleDData,
  rerunModuleDSegment,
  rerunModuleDBothFrames,
  rerunModuleDAll,
  taskQueryKeys,
} from "@/api/taskApi";
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
  rerunBothLoading,
}: {
  taskId: string;
  segment: TaskModuleDSegment;
  segmentVideo?: TaskModuleDSegmentVideoFile;
  watchFast: boolean;
  onRerun: (seg: TaskModuleDSegment) => void;
  onRerunBoth: (seg: TaskModuleDSegment) => void;
  rerunBothLoading: boolean;
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
        </Space>
      }
      extra={
        <Space size={4}>
          {segStatus === "done" && segDuration ? (
            <Typography.Text style={{ fontSize: 11, color: "#999" }}>
              {segStart.toFixed(2)}~{segEnd.toFixed(2)}s
            </Typography.Text>
          ) : null}
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
        </Space>
      }
    >
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
    </Card>
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
  const [rerunTarget, setRerunTarget] = useState<TaskModuleDSegment | null>(null);
  const [rerunBothTarget, setRerunBothTarget] = useState<TaskModuleDSegment | null>(null);
  const [rerunBothTransitionBg, setRerunBothTransitionBg] = useState<"white" | "black">();

  const { data } = useQuery({
    queryKey: taskQueryKeys.moduleD(taskId),
    queryFn: () => getTaskModuleDData(taskId),
    staleTime: 0,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) return 5000;
      if (payload.active_rerun?.active) return 3000;
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
      </Space>

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
                rerunBothLoading={rerunBothMutation.isPending}
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
