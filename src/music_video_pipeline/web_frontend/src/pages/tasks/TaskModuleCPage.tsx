import {
  ReloadOutlined,
  EyeOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Image,
  Modal,
  Row,
  Space,
  Tag,
  Typography,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  getTaskModuleCData,
  rerunModuleCFrame,
  rerunModuleCShot,
  resumeModuleB,
  resumeModuleC,
  taskQueryKeys,
} from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import type { TaskModuleCShot } from "@/schemas/moduleC";

function formatTimeRange(startTime: number, endTime: number): string {
  return `${startTime.toFixed(2)}s ~ ${endTime.toFixed(2)}s`;
}

function frameStatusColor(status: string): string {
  switch (status) {
    case "done":
      return "green";
    case "running":
      return "blue";
    case "failed":
      return "red";
    default:
      return "default";
  }
}

function ShotCard({
  shot,
  onRerunShot,
  onRerunFrame,
  rerunLoading,
}: {
  shot: TaskModuleCShot;
  onRerunShot: (shotId: string) => void;
  onRerunFrame: (shotId: string, frameType: "start" | "end") => void;
  rerunLoading: boolean;
}) {
  const [promptModalOpen, setPromptModalOpen] = useState(false);
  const [refreshKeyStart, setRefreshKeyStart] = useState(0);
  const [refreshKeyEnd, setRefreshKeyEnd] = useState(0);

  const frameUrlStart = shot.frame_url_start
    ? `${shot.frame_url_start}${shot.frame_url_start.includes("?") ? "&" : "?"}_=${refreshKeyStart}`
    : "";
  const frameUrlEnd = shot.frame_url_end
    ? `${shot.frame_url_end}${shot.frame_url_end.includes("?") ? "&" : "?"}_=${refreshKeyEnd}`
    : "";

  return (
    <>
      <Card
        size="small"
        style={{ marginBottom: 16 }}
        title={
          <Space>
            <Typography.Text strong>{shot.shot_id}</Typography.Text>
            <TaskStatusTag status={shot.status} />
            <Typography.Text type="secondary">
              {formatTimeRange(shot.start_time, shot.end_time)}
            </Typography.Text>
            {shot.lyrics && shot.lyrics.length > 0 ? (
              <Tag color="purple" style={{ fontSize: 10, lineHeight: "16px", maxWidth: 280 }} title={shot.lyrics.join(" | ")}>
                {shot.lyrics.slice(0, 2).join(" / ")}{shot.lyrics.length > 2 ? "…" : ""}
              </Tag>
            ) : (
              <Typography.Text style={{ fontSize: 11, color: "#bbb" }}>无歌词</Typography.Text>
            )}
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={12}>
            <Typography.Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
              首帧{" "}
              <Tag color={frameStatusColor(shot.frame_status_start)}>
                {shot.frame_status_start}
              </Tag>
              <Button type="link" size="small" icon={<ReloadOutlined />} onClick={() => setRefreshKeyStart((k) => k + 1)} style={{ padding: 0, marginLeft: 8 }} />
            </Typography.Text>
            {shot.frame_url_start ? (
              <Image
                src={frameUrlStart}
                alt={`${shot.shot_id} 首帧`}
                style={{ maxHeight: 300, width: "100%", objectFit: "contain" }}
                fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
              />
            ) : (
              <div
                style={{
                  height: 100,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fafafa",
                  border: "1px dashed #d9d9d9",
                  borderRadius: 4,
                }}
              >
                <Typography.Text type="secondary">首帧未生成</Typography.Text>
              </div>
            )}
          </Col>
          <Col span={12}>
            <Typography.Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
              尾帧{" "}
              <Tag color={frameStatusColor(shot.frame_status_end)}>
                {shot.frame_status_end}
              </Tag>
              <Button type="link" size="small" icon={<ReloadOutlined />} onClick={() => setRefreshKeyEnd((k) => k + 1)} style={{ padding: 0, marginLeft: 8 }} />
            </Typography.Text>
            {shot.frame_url_end ? (
              <Image
                src={frameUrlEnd}
                alt={`${shot.shot_id} 尾帧`}
                style={{ maxHeight: 300, width: "100%", objectFit: "contain" }}
                fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
              />
            ) : (
              <div
                style={{
                  height: 100,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fafafa",
                  border: "1px dashed #d9d9d9",
                  borderRadius: 4,
                }}
              >
                <Typography.Text type="secondary">尾帧未生成</Typography.Text>
              </div>
            )}
          </Col>
        </Row>
        <div style={{ marginTop: 12 }}>
          <Space>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => onRerunShot(shot.shot_id)}
              loading={rerunLoading}
            >
              重跑 Shot
            </Button>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => onRerunFrame(shot.shot_id, "start")}
              loading={rerunLoading}
            >
              重跑首帧
            </Button>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => onRerunFrame(shot.shot_id, "end")}
              loading={rerunLoading}
            >
              重跑尾帧
            </Button>
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => setPromptModalOpen(true)}
            >
              查看完整 Prompt
            </Button>
          </Space>
        </div>
      </Card>

      <Modal
        title={`完整 Prompt — ${shot.shot_id}`}
        open={promptModalOpen}
        onCancel={() => setPromptModalOpen(false)}
        footer={null}
        width={800}
      >
        <Typography.Title level={5}>原始 role4 字段</Typography.Title>
        {shot.role4_prompt?.keyframe_prompt_start_zh ? (
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 12px", fontSize: 13, marginBottom: 24 }}>
            <Typography.Text type="secondary">首帧描述(中):</Typography.Text>
            <Typography.Text>{shot.role4_prompt.keyframe_prompt_start_zh}</Typography.Text>
            <Typography.Text type="secondary">首帧描述(英):</Typography.Text>
            <Typography.Text style={{ wordBreak: "break-word" }}>{shot.role4_prompt.keyframe_prompt_start_en}</Typography.Text>
            <Typography.Text type="secondary">尾帧描述(中):</Typography.Text>
            <Typography.Text>{shot.role4_prompt.keyframe_prompt_end_zh}</Typography.Text>
            <Typography.Text type="secondary">尾帧描述(英):</Typography.Text>
            <Typography.Text style={{ wordBreak: "break-word" }}>{shot.role4_prompt.keyframe_prompt_end_en}</Typography.Text>
            <Typography.Text type="secondary">视频 prompt(中):</Typography.Text>
            <Typography.Text>{shot.role4_prompt.video_prompt_zh}</Typography.Text>
            <Typography.Text type="secondary">视频 prompt(英):</Typography.Text>
            <Typography.Text style={{ wordBreak: "break-word" }}>{shot.role4_prompt.video_prompt_en}</Typography.Text>
          </div>
        ) : (
          <Typography.Text type="secondary" style={{ marginBottom: 24, display: "block" }}>无 role4 数据</Typography.Text>
        )}

        <Typography.Title level={5}>组装后完整 Prompt</Typography.Title>
        <Typography.Text type="secondary" style={{ display: "block", marginBottom: 4 }}>首帧完整 prompt（prefix + body + suffix）：</Typography.Text>
        <pre
          style={{
            padding: 12,
            background: "#f5f5f5",
            borderRadius: 6,
            fontSize: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: 300,
            overflow: "auto",
            marginBottom: 16,
          }}
        >
          {shot.assembled_prompt_start || "暂无"}
        </pre>
        <Typography.Text type="secondary" style={{ display: "block", marginBottom: 4 }}>尾帧完整 prompt（prefix + body + suffix）：</Typography.Text>
        <pre
          style={{
            padding: 12,
            background: "#f5f5f5",
            borderRadius: 6,
            fontSize: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: 300,
            overflow: "auto",
          }}
        >
          {shot.assembled_prompt_end || "暂无"}
        </pre>
      </Modal>
    </>
  );
}

export function TaskModuleCPage() {
  const taskId = useTaskIdParam();
  const { message } = App.useApp();
  const queryClient = useQueryClient();

  const { data, refetch } = useQuery({
    queryKey: taskQueryKeys.moduleC(taskId),
    queryFn: () => getTaskModuleCData(taskId),
    staleTime: 0,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) return 2000;
      if (payload.active_rerun?.active) return 1000;
      if (payload.module_c_status === "running") return 2000;
      return 5000;
    },
  });

  const rerunShotMutation = useMutation({
    mutationFn: (shotId: string) =>
      rerunModuleCShot(taskId, shotId, { replaceRunning: true }),
    onSuccess: () => {
      message.success("Shot 重跑已提交");
      refetch();
    },
    onError: (error: Error) => {
      appLogger.error("模块C", "Shot 重跑失败", { error: error.message });
      message.error(error.message);
    },
  });

  const rerunFrameMutation = useMutation({
    mutationFn: ({
      shotId,
      frameType,
    }: {
      shotId: string;
      frameType: "start" | "end";
    }) => rerunModuleCFrame(taskId, shotId, frameType, { replaceRunning: true }),
    onSuccess: () => {
      message.success("帧重跑已提交");
      refetch();
    },
    onError: (error: Error) => {
      appLogger.error("模块C", "帧重跑失败", { error: error.message });
      message.error(error.message);
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => resumeModuleC(taskId),
    onSuccess: async (payload) => {
      await refetch();
      message.success(payload.message || "断点续跑已开始");
    },
    onError: (error) => {
      const errorText = error instanceof Error ? error.message : String(error);
      message.warning(errorText);
    },
  });

  const bcdResumeMutation = useMutation({
    mutationFn: () => resumeModuleB(taskId),
    onSuccess: async (payload) => {
      await refetch();
      message.success(payload.message || "BCD 续跑已开始");
    },
    onError: (error) => {
      const errorText = error instanceof Error ? error.message : String(error);
      message.warning(errorText);
    },
  });

  const handleRerunShot = (shotId: string) => {
    if (activeRerun?.active && activeRerun.shot_id === shotId) {
      const runningSeconds = activeRerun.started_at_ms
        ? Math.floor((Date.now() - activeRerun.started_at_ms) / 1000)
        : 0;
      Modal.confirm({
        title: "当前任务运行中",
        content: `该 shot 已有后台任务正在运行，已运行 ${runningSeconds} 秒。是否取消当前任务并重新执行？`,
        okText: "取消并重新执行",
        onOk: () => rerunShotMutation.mutate(shotId),
      });
    } else {
      Modal.confirm({
        title: "确认重跑",
        content: `将重跑 ${shotId} 的整个镜头，会重新生成首尾帧。`,
        onOk: () => rerunShotMutation.mutate(shotId),
      });
    }
  };

  const handleRerunFrame = (shotId: string, frameType: "start" | "end") => {
    const label = frameType === "start" ? "首帧" : "尾帧";
    if (activeRerun?.active && activeRerun.shot_id === shotId) {
      const runningSeconds = activeRerun.started_at_ms
        ? Math.floor((Date.now() - activeRerun.started_at_ms) / 1000)
        : 0;
      Modal.confirm({
        title: "当前任务运行中",
        content: `该 shot 已有后台任务正在运行，已运行 ${runningSeconds} 秒。是否取消当前任务并重新执行${label}？`,
        okText: "取消并重新执行",
        onOk: () => rerunFrameMutation.mutate({ shotId, frameType }),
      });
    } else {
      Modal.confirm({
        title: "确认重跑",
        content: `将重跑 ${shotId} 的${label}。`,
        onOk: () => rerunFrameMutation.mutate({ shotId, frameType }),
      });
    }
  };

  const isRerunLoading = rerunShotMutation.isPending || rerunFrameMutation.isPending;

  const unitSummary = data?.unit_summary;
  const shots = data?.shots ?? [];
  const activeRerun = data?.active_rerun;

  const sortedShots = [...shots].sort((a, b) => a.shot_id.localeCompare(b.shot_id));

  return (
    <div className="module-c-page">
      <Typography.Title level={3}>Module C — 镜头首尾帧</Typography.Title>

      {activeRerun?.status === "failed" ? (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message={`最近一次重跑失败：${activeRerun.failure_reason || "未知原因"}`}
          description={activeRerun.last_error || "请检查后台日志与当前 shot 状态。"}
        />
      ) : null}

      {activeRerun?.active ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={`当前正在重跑：${activeRerun.shot_id || "-"}${activeRerun.frame_type ? ` ${activeRerun.frame_type === "start" ? "首帧" : "尾帧"}` : " 整个shot"}`}
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
          {activeRerun?.active ? <Tag color="processing">重跑中: {activeRerun.shot_id}</Tag> : null}
          {activeRerun?.status === "failed" ? <Tag color="error">最近重跑失败</Tag> : null}
          <Button
            icon={<ToolOutlined />}
            size="small"
            loading={resumeMutation.isPending}
            onClick={() => {
              Modal.confirm({
                title: "断点续跑 Module C",
                content: "将扫描所有 shot，对首帧/尾帧缺失的 shot 逐个补跑。已有产物的 shot 会跳过。",
                onOk: () => resumeMutation.mutate(),
              });
            }}
          >
            C 续跑
          </Button>
          <Button
            size="small"
            loading={bcdResumeMutation.isPending}
            onClick={() => {
              Modal.confirm({
                title: "BCD 续跑",
                content: "将自动扫描 B→C→D 链路，检查各 role、shot 和 segment 的完成状态，仅补充缺失环节。",
                onOk: () => bcdResumeMutation.mutate(),
              });
            }}
          >
            BCD 续跑
          </Button>
        </Space>
      ) : null}

      {sortedShots.length === 0 ? (
        <Typography.Text type="secondary">暂无镜头数据，请先执行模块 B 和模块 C。</Typography.Text>
      ) : (
        <Row gutter={[16, 16]}>
          {sortedShots.map((shot) => (
            <Col span={12} key={shot.shot_id}>
              <ShotCard
                shot={shot}
                onRerunShot={handleRerunShot}
                onRerunFrame={handleRerunFrame}
                rerunLoading={isRerunLoading}
              />
            </Col>
          ))}
        </Row>
      )}
    </div>
  );
}
