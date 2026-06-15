import { useEffect, useRef, useState } from "react";
import { Alert, Modal, Select, Space, Tag, Typography } from "antd";
import type { TaskModuleBRole } from "@/schemas/moduleB";
import { appLogger } from "@/app/logger";

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
  // ===== 所有 ref/state 声明放在最前面 =====
  const rolePrevRef = useRef(role?.active_rerun?.active);
  const runCountRef = useRef(0);
  const [tick, setTick] = useState(0);
  const elapsedMsRef = useRef(0);
  const timerActiveRef = useRef(false);
  const timerWasActiveRef = useRef(false);
  const localSubmittedAtMsRef = useRef(0);
  const [segmentId, setSegmentId] = useState("");

  // ===== 派生变量 =====
  const roleName = role?.role_name || "";
  const isSegmentedRole = roleName === "role3" || roleName === "role4";
  const isRole3 = roleName === "role3";
  const allSegments = role?.segment_items || [];
  const previewSegments = role?.stream_preview_segments || [];
  const rerunStatus = role?.active_rerun?.status || "";
  // isRunning 的改进逻辑：本地计时器激活 + 非终态 → 视为运行中
  // 避免后端 active=false 导致的计时器鬼畜
  const isRunning =
    Boolean(role?.active_rerun?.active) ||
    (open && timerWasActiveRef.current && rerunStatus !== "succeeded" && rerunStatus !== "failed");
  const isCompleted = rerunStatus === "succeeded";
  const isFailed = rerunStatus === "failed";
  const implementationStatus = role?.implementation_status || "";

  const isRunningRef = useRef(isRunning);
  const implStatusRef = useRef(implementationStatus);
  const prevSegmentsLenRef = useRef(previewSegments.length);
  const prevSegmentsContentRef = useRef("");

  // ========== 日志：modal 打开/关闭 ==========
  useEffect(() => {
    appLogger.info("StreamViewerModal", `模态框 ${open ? "打开" : "关闭"}`, {
      roleName: role?.role_name,
      hasRole: !!role,
    });
    if (!open) {
      appLogger.info("StreamViewerModal", "计时器停止（模态框关闭）", {
        lastElapsedMs: elapsedMsRef.current,
        runCount: runCountRef.current,
      });
      runCountRef.current = 0;
    }
  }, [open, role?.role_name]);

  // ========== 日志：active_rerun.active 状态变化 ==========
  useEffect(() => {
    const currentActive = Boolean(role?.active_rerun?.active);
    const prevActive = rolePrevRef.current;
    if (prevActive !== undefined && prevActive !== currentActive) {
      appLogger.info("StreamViewerModal", `active_rerun.active 变化: ${prevActive} → ${currentActive}`, {
        roleName: role?.role_name,
        prevStatus: prevActive ? "运行中" : "非运行",
        currentStatus: currentActive ? "运行中" : "非运行",
        activeRerun: role?.active_rerun,
        elapsedMsRef: elapsedMsRef.current,
      });
    }
    rolePrevRef.current = currentActive;
  }, [role?.active_rerun?.active, role?.active_rerun, role?.role_name]);

  // ========== 修复：本地计时器状态 ==========
  // 一旦见过 active:true，就持续本地计时，不依赖后端 active 字段
  // 这样即使后端尚未确认重跑、返回 active=false，计时器也不会鬼畜
  useEffect(() => {
    if (!open) {
      // 关闭模态框时重置本地计时器
      if (timerWasActiveRef.current) {
        appLogger.info("StreamViewerModal", "本地计时器重置（模态框关闭）", {
          roleName,
          totalElapsedMs: elapsedMsRef.current,
        });
      }
      timerWasActiveRef.current = false;
      localSubmittedAtMsRef.current = 0;
      return;
    }
    // 当 active 从 false→true 时，记录 submitted_at_ms
    if (role?.active_rerun?.active) {
      const submittedMs = role.active_rerun.submitted_at_ms || Date.now();
      if (!timerWasActiveRef.current) {
        timerWasActiveRef.current = true;
        localSubmittedAtMsRef.current = submittedMs;
        appLogger.info("StreamViewerModal", "本地计时器激活", {
          roleName,
          submitted_at_ms: submittedMs,
          activeRerunStatusCode: role.active_rerun.status,
        });
      }
    }
  }, [open, role?.active_rerun?.active, role?.active_rerun?.submitted_at_ms, roleName]);

  // ========== 日志：isRunning / implementation_status 变化 ==========
  useEffect(() => {
    if (isRunningRef.current !== isRunning) {
      appLogger.info("StreamViewerModal", `isRunning 变化: ${isRunningRef.current} → ${isRunning}`, {
        roleName,
        implementationStatus,
        activeRerun: role?.active_rerun,
      });
      isRunningRef.current = isRunning;
    }
  }, [isRunning, roleName, implementationStatus, role?.active_rerun]);

  useEffect(() => {
    if (implStatusRef.current !== implementationStatus) {
      appLogger.info("StreamViewerModal", `implementation_status 变化: ${implStatusRef.current} → ${implementationStatus}`, {
        roleName,
        isRunning,
        activeRerun: role?.active_rerun,
      });
      implStatusRef.current = implementationStatus;
    }
  }, [implementationStatus, roleName, isRunning, role?.active_rerun]);

  // ========== 日志：stream_preview_segments 内容变化 ==========
  useEffect(() => {
    const currentContent = JSON.stringify(previewSegments.map(s => ({ id: s.segment_id, len: s.content?.length || 0 })));
    if (prevSegmentsLenRef.current !== previewSegments.length || prevSegmentsContentRef.current !== currentContent) {
      appLogger.info("StreamViewerModal", "stream_preview_segments 更新", {
        roleName,
        segmentsCount: previewSegments.length,
        segmentsDetail: previewSegments.map(s => ({
          segment_id: s.segment_id,
          contentLength: s.content?.length || 0,
          updated_at_ms: s.updated_at_ms,
        })),
      });
      prevSegmentsLenRef.current = previewSegments.length;
      prevSegmentsContentRef.current = currentContent;
    }
  }, [previewSegments, roleName]);

  // 默认选中第一个 segment（role4 用）
  useEffect(() => {
    if (!isSegmentedRole || isRole3 || !allSegments.length) return;
    setSegmentId((prev) => {
      if (prev && allSegments.some((s) => s.segment_id === prev)) return prev;
      return allSegments[0].segment_id;
    });
  }, [isSegmentedRole, isRole3, allSegments]);

  // ========== 计时器 ==========
  useEffect(() => {
    if (!open) return;
    appLogger.info("StreamViewerModal", `计时器 setInterval 启动 (每100ms)`, {
      roleName,
      isRunning,
      activeRerun: role?.active_rerun,
    });
    timerActiveRef.current = true;
    const timerId = window.setInterval(() => {
      setTick((t) => t + 1);
      runCountRef.current++;
    }, 100);
    return () => {
      timerActiveRef.current = false;
      appLogger.info("StreamViewerModal", `计时器 clearInterval 清理`, {
        roleName,
        finalTick: runCountRef.current,
        finalElapsedMs: elapsedMsRef.current,
      });
      window.clearInterval(timerId);
    };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const elapsedMs = (() => {
    if (!role) return 0;

    // 本地计时器：一旦激活，持续计时直到终态，不受后端 active 字段影响
    if (open && timerWasActiveRef.current && localSubmittedAtMsRef.current > 0) {
      const newVal = Math.max(0, Date.now() - localSubmittedAtMsRef.current);
      const diff = newVal - elapsedMsRef.current;
      if (Math.abs(diff) > 500) {
        appLogger.info("StreamViewerModal", `[本地计时器] elapsedMs: ${elapsedMsRef.current} → ${newVal} (diff=${diff}ms)`, {
          roleName,
          localSubmittedAtMs: localSubmittedAtMsRef.current,
          tick,
        });
      }
      elapsedMsRef.current = newVal;
      return elapsedMsRef.current;
    }

    const ar = role.active_rerun;
    if (ar?.active) {
      const startMs = ar.started_at_ms || ar.submitted_at_ms || 0;
      if (startMs > 0) {
        const newVal = Math.max(0, Date.now() - startMs);
        const diff = newVal - elapsedMsRef.current;
        if (Math.abs(diff) > 500) {
          appLogger.info("StreamViewerModal", `[后端计时器] elapsedMs 跳变: ${elapsedMsRef.current} → ${newVal} (diff=${diff}ms)`, {
            roleName,
            active: true,
            startMs,
            started_at_ms: ar.started_at_ms,
            submitted_at_ms: ar.submitted_at_ms,
            tick,
          });
        }
        elapsedMsRef.current = newVal;
      }
      return elapsedMsRef.current;
    }
    const fallback = ar?.duration_ms || elapsedMsRef.current || 0;
    if (elapsedMsRef.current > 0 && fallback === 0) {
      appLogger.info("StreamViewerModal", `elapsedMs: 未激活回退到 0`, {
        roleName,
        duration_ms: ar?.duration_ms,
        elapsedMsRefPrev: elapsedMsRef.current,
        tick,
        timerWasActive: timerWasActiveRef.current,
        localSubmitted: localSubmittedAtMsRef.current,
      });
    }
    return fallback;
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
