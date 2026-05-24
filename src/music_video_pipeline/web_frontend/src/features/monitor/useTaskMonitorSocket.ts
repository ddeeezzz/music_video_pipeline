import { useEffect, useMemo, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { taskMonitorSnapshotSchema } from "@/schemas/monitor";

type ConnectionState = "idle" | "connecting" | "open" | "closed" | "error";

function buildWebSocketUrl(taskId: string): string {
  const configuredBaseUrl = import.meta.env.VITE_WS_BASE_URL?.trim();
  if (configuredBaseUrl) {
    const normalizedBaseUrl = configuredBaseUrl.replace(/\/+$/, "");
    return `${normalizedBaseUrl}/ws?task_id=${encodeURIComponent(taskId)}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws?task_id=${encodeURIComponent(taskId)}`;
}

export function useTaskMonitorSocket(taskId: string, enabled: boolean): {
  connectionState: ConnectionState;
  lastMessageAt: number | null;
} {
  const queryClient = useQueryClient();
  const reconnectTimerRef = useRef<number | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const reconnectUrl = useMemo(() => (taskId ? buildWebSocketUrl(taskId) : ""), [taskId]);

  useEffect(() => {
    if (!enabled || !taskId) {
      setConnectionState("idle");
      return;
    }

    let cancelled = false;
    let socket: WebSocket | null = null;

    const connect = (): void => {
      if (cancelled) {
        return;
      }

      setConnectionState("connecting");
      appLogger.info("任务监督", "准备建立 WebSocket 连接", { taskId });
      socket = new WebSocket(reconnectUrl);

      socket.onopen = () => {
        setConnectionState("open");
        appLogger.info("任务监督", "WebSocket 连接已建立", { taskId });
      };

      socket.onmessage = (event) => {
        let rawPayload: unknown = {};
        try {
          rawPayload = JSON.parse(String(event.data || "{}"));
        } catch (_error) {
          appLogger.warn("任务监督", "收到无法解析的 WebSocket 消息，已忽略", { taskId });
          return;
        }
        const parsed = taskMonitorSnapshotSchema.safeParse(rawPayload);
        if (!parsed.success) {
          appLogger.warn("任务监督", "收到无法校验的监督快照，已忽略", {
            taskId,
            issues: parsed.error.issues,
          });
          return;
        }
        queryClient.setQueryData(taskQueryKeys.snapshot(taskId), parsed.data);
        setLastMessageAt(Date.now());
      };

      socket.onerror = () => {
        setConnectionState("error");
        appLogger.error("任务监督", "WebSocket 连接发生错误", { taskId });
      };

      socket.onclose = () => {
        if (cancelled) {
          return;
        }
        setConnectionState("closed");
        appLogger.warn("任务监督", "WebSocket 连接已关闭，准备重连", { taskId });
        reconnectTimerRef.current = window.setTimeout(connect, 1500);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (socket) {
        socket.close();
      }
      appLogger.info("任务监督", "任务监督页已卸载，WebSocket 连接已清理", { taskId });
    };
  }, [enabled, queryClient, reconnectUrl, taskId]);

  return {
    connectionState,
    lastMessageAt,
  };
}
