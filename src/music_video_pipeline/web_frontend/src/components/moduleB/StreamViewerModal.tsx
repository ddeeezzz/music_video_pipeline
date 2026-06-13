import { useEffect, useRef, useState } from "react";
import { Alert, Modal, Select, Space, Tag, Typography } from "antd";
import type { TaskModuleBRole } from "@/schemas/moduleB";

interface StreamViewerModalProps {
  role: TaskModuleBRole | undefined;
  open: boolean;
  onClose: () => void;
}

function formatStopwatchSeconds(totalSeconds: number): string {
  const normalized = Math.max(0, Number(totalSeconds) || 0);
  const totalTenths = Math.floor(normalized * 10);
  const hours = Math.floor(totalTenths / 36000);
  const minutes = Math.floor((totalTenths % 36000) / 600);
  const seconds = Math.floor((totalTenths % 600) / 10);
  const tenths = totalTenths % 10;
  const hh = String(hours).padStart(2, "0");
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  if (hours > 0) {
    return `${hh}:${mm}:${ss}.${tenths}`;
  }
  return `${mm}:${ss}.${tenths}`;
}

export function StreamViewerModal({ role, open, onClose }: StreamViewerModalProps) {
  const [segmentId, setSegmentId] = useState("");

  const roleName = role?.role_name || "";
  const isSegmentedRole = roleName === "role3" || roleName === "role4";
  const isRole3 = roleName === "role3";
  const allSegments = role?.segment_items || [];
  const previewSegments = role?.stream_preview_segments || [];
  const isRunning = Boolean(role?.active_rerun?.active);
  const isCompleted = role?.active_rerun?.status === "succeeded";
  const isFailed = role?.active_rerun?.status === "failed";

  // 默认选中第一个 segment（role4 用）
  useEffect(() => {
    if (!isSegmentedRole || isRole3 || !allSegments.length) return;
    setSegmentId((prev) => {
      if (prev && allSegments.some((s) => s.segment_id === prev)) return prev;
      return allSegments[0].segment_id;
    });
  }, [isSegmentedRole, isRole3, allSegments]);

  // 计时器
  const [tick, setTick] = useState(0);
  const elapsedMsRef = useRef(0);
  useEffect(() => {
    if (!open) return;
    const timerId = window.setInterval(() => setTick((t) => t + 1), 100);
    return () => window.clearInterval(timerId);
  }, [open]);

  const elapsedMs = (() => {
    if (!role) return 0;
    const ar = role.active_rerun;
    if (ar?.active) {
      const startMs = ar.started_at_ms || ar.submitted_at_ms || 0;
      if (startMs > 0) {
        elapsedMsRef.current = Math.max(0, Date.now() - startMs);
      }
      return elapsedMsRef.current;
    }
    return ar?.duration_ms || elapsedMsRef.current || 0;
  })();

  const segmentContent =
    !isRole3 && isSegmentedRole && segmentId
      ? (previewSegments.find((s) => s.segment_id === segmentId)?.content || "")
      : "";
  const displayContent = isRole3
    ? ""
    : isSegmentedRole
      ? (segmentContent || role?.result_text?.content || "")
      : (role?.stream_preview?.content || role?.result_text?.content || "");

  const hasContent = Boolean(previewSegments.length > 0 ? true : displayContent);

  return (
    <Modal
      title={
        <Space direction="vertical" size={4}>
          <Typography.Text strong>
            {role?.title || ""} 流式输出
          </Typography.Text>
          <Space wrap size={[8, 4]}>
            <Tag color={isRunning ? "processing" : "default"}>
              计时器 {formatStopwatchSeconds(elapsedMs / 1000)}
            </Tag>
          </Space>
        </Space>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={920}
      destroyOnClose={false}
    >
      {role ? (
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          {isCompleted || isFailed ? (
            <Alert
              type={isFailed ? "error" : "success"}
              showIcon
              message={
                isFailed
                  ? `重跑失败：${role.active_rerun?.failure_reason || "未知原因"}`
                  : "当前展示的是最近一次已收到的全部内容。"
              }
            />
          ) : null}

          {isRole3 && previewSegments.length > 0 ? (
            <div className="stream-viewer-big-list">
              {previewSegments.map((seg) => (
                <div key={seg.segment_id} className="stream-viewer-big-section">
                  <div className="stream-viewer-big-header">
                    <Typography.Text strong className="stream-viewer-big-title">
                      {seg.segment_id}
                    </Typography.Text>
                  </div>
                  <pre className="stream-viewer-big-content">
                    {seg.content || "…"}
                  </pre>
                </div>
              ))}
            </div>
          ) : (
            <>
              {isSegmentedRole && allSegments.length > 0 && hasContent ? (
                <Select
                  value={segmentId || undefined}
                  onChange={(value: string) => setSegmentId(value)}
                  options={allSegments.map((seg) => ({
                    value: seg.segment_id,
                    label: (seg as any).display_title || seg.segment_id,
                  }))}
                  style={{ width: "100%" }}
                  placeholder={roleName === "role3" ? "选择 big_segment" : "选择 shot"}
                />
              ) : null}

              <pre
                style={{
                  margin: 0,
                  padding: 16,
                  background: "var(--ant-color-fill-quaternary)",
                  borderRadius: 8,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontSize: 12,
                  lineHeight: 1.6,
                }}
              >
                {!hasContent && isRunning
                  ? "等待中..."
                  : displayContent || "暂无内容"}
              </pre>
            </>
          )}
        </Space>
      ) : null}
    </Modal>
  );
}
