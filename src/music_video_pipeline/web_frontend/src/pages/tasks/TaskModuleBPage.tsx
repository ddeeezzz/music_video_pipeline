import { useEffect, useMemo, useState } from "react";

import {
  ExportOutlined,
  EyeOutlined,
  FileSearchOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  App,
  Button,
  Card,
  Descriptions,
  Empty,
  Modal,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getTaskModuleBData,
  rerunModuleBRole,
  rerunModuleBRoleSegment,
  taskQueryKeys,
} from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import type { TaskModuleBData, TaskModuleBRole } from "@/schemas/moduleB";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";

function getImplementationTag(role: TaskModuleBRole) {
  if (role.implementation_status === "implemented") {
    return <Tag color="success">已接入</Tag>;
  }
  if (role.implementation_status === "placeholder") {
    return <Tag color="warning">占位中</Tag>;
  }
  if (role.implementation_status === "missing") {
    return <Tag color="error">缺源码</Tag>;
  }
  return <Tag>待确认</Tag>;
}

function formatSegmentOptionLabel(segmentId: string, shotId: string, label: string): string {
  const normalizedLabel = label.trim();
  if (!normalizedLabel) {
    return `${segmentId} / ${shotId}`;
  }
  return `${segmentId} / ${shotId} / ${normalizedLabel}`;
}

function formatTimeRange(startTime: number, endTime: number): string {
  return `${startTime.toFixed(2)} - ${endTime.toFixed(2)}`;
}

function formatDurationMs(durationMs: number): string {
  const normalized = Math.max(0, Number(durationMs) || 0);
  if (normalized < 1000) {
    return `${normalized} ms`;
  }
  return `${(normalized / 1000).toFixed(normalized >= 10000 ? 1 : 2)} s`;
}

function formatTimestampOrDash(value: string): string {
  return value.trim() || "-";
}

function formatStopwatchSeconds(totalSeconds: number): string {
  const normalizedSeconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const hours = Math.floor(normalizedSeconds / 3600);
  const minutes = Math.floor((normalizedSeconds % 3600) / 60);
  const seconds = normalizedSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function isActiveRerunConflictMessage(errorText: string): boolean {
  const normalizedText = errorText.trim();
  return [
    "任务已有后台动作执行中",
    "任务已有后台子进程执行中",
    "旧进程仍在退出中",
  ].some((fragment) => normalizedText.includes(fragment));
}

function buildRoleResultStatusText(
  updatedAt: string,
  updatedAtMs: number,
  rerunSubmittedAtMs: number,
  rerunActive: boolean,
): string {
  if (!updatedAt) {
    return rerunActive ? "当前尚未生成成果，正在等待新产物落盘。" : "当前尚未生成成果。";
  }
  if (rerunActive && updatedAtMs > 0 && rerunSubmittedAtMs > 0 && updatedAtMs < rerunSubmittedAtMs) {
    return `已有旧成果，最近更新于 ${updatedAt}；本次重跑的新产物尚未落盘。`;
  }
  if (rerunActive) {
    return `成果文件最近更新于 ${updatedAt}，页面正在继续跟踪本次重跑。`;
  }
  return `成果文件最近更新于 ${updatedAt}。`;
}

function buildRerunStatusMessage(
  activeRerun: TaskModuleBData["active_rerun"] | undefined,
  roleName: string,
): { type: "info" | "success" | "error"; text: string } | null {
  if (!activeRerun || activeRerun.role_name !== roleName || !activeRerun.status) {
    return null;
  }
  if (activeRerun.active) {
    if (activeRerun.mode === "segment") {
      return {
        type: "info",
        text: `正在按 Segment 重跑，已提交于 ${activeRerun.submitted_at || "-"}。`,
      };
    }
    return {
      type: "info",
      text: `正在按 Role 重跑，已提交于 ${activeRerun.submitted_at || "-"}。`,
    };
  }
  if (activeRerun.status === "succeeded") {
    return {
      type: "success",
      text: `最近一次重跑已完成，后端耗时 ${formatDurationMs(activeRerun.duration_ms)}。`,
    };
  }
  if (activeRerun.status === "failed") {
    const reason = activeRerun.failure_reason || "未知原因";
    const detail = activeRerun.last_error ? `；详情：${activeRerun.last_error}` : "";
    return {
      type: "error",
      text: `最近一次重跑失败，后端耗时 ${formatDurationMs(activeRerun.duration_ms)}，原因：${reason}${detail}`,
    };
  }
  return null;
}

export function TaskModuleBPage() {
  const taskId = useTaskIdParam();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const [promptRoleName, setPromptRoleName] = useState("");
  const [segmentSelection, setSegmentSelection] = useState<Record<string, string>>({});
  const [clientRerunStartedAtMs, setClientRerunStartedAtMs] = useState<Record<string, number>>({});
  const [clientRerunFrozenElapsedMs, setClientRerunFrozenElapsedMs] = useState<Record<string, number>>({});
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [streamViewerRoleName, setStreamViewerRoleName] = useState("");
  const [streamViewerMode, setStreamViewerMode] = useState<"rerun" | "follow-current">("rerun");

  useEffect(() => {
    appLogger.info("模块B页面", "模块 B 观察页已进入", { taskId });
  }, [taskId]);

  const { data, isLoading, refetch, isFetching, error } = useQuery({
    queryKey: taskQueryKeys.moduleB(taskId),
    queryFn: () => getTaskModuleBData(taskId),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (payload?.active_rerun?.active) {
        return 1000;
      }
      if (payload?.task_status === "running" || payload?.module_b_status === "running") {
        return 2000;
      }
      return false;
    },
  });
  const queryErrorText = error instanceof Error ? error.message : "";

  const roles = data?.roles || [];
  const roleMap = useMemo(
    () => Object.fromEntries(roles.map((role) => [role.role_name, role])),
    [roles],
  );
  const segmentItems = data?.segment_items || [];

  useEffect(() => {
    setSegmentSelection((current) => {
      const nextSelection = { ...current };
      for (const roleName of ["role3", "role4"]) {
        const currentValue = nextSelection[roleName];
        if (currentValue && segmentItems.some((item) => item.segment_id === currentValue)) {
          continue;
        }
        nextSelection[roleName] = segmentItems[0]?.segment_id || "";
      }
      return nextSelection;
    });
  }, [segmentItems]);

  useEffect(() => {
    const hasLocalTimer = Object.keys(clientRerunStartedAtMs).length > 0;
    const hasActiveRerun = Boolean(data?.active_rerun?.active);
    if (!hasLocalTimer && !hasActiveRerun) {
      return undefined;
    }
    const timerId = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timerId);
  }, [clientRerunStartedAtMs, data?.active_rerun?.active]);

  useEffect(() => {
    const activeRerun = data?.active_rerun;
    if (!activeRerun || !activeRerun.role_name) {
      return;
    }
    if (activeRerun.active) {
      return;
    }
    const roleName = activeRerun.role_name;
    const startedAtMs = clientRerunStartedAtMs[roleName] || 0;
    const fallbackElapsedMs =
      activeRerun.duration_ms > 0
        ? activeRerun.duration_ms
        : startedAtMs > 0
          ? Math.max(0, nowMs - startedAtMs)
          : 0;
    setClientRerunFrozenElapsedMs((current) => ({
      ...current,
      [roleName]: fallbackElapsedMs,
    }));
    setClientRerunStartedAtMs((current) => {
      if (!(roleName in current)) {
        return current;
      }
      const nextState = { ...current };
      delete nextState[roleName];
      return nextState;
    });
  }, [clientRerunStartedAtMs, data?.active_rerun, nowMs]);

  const invalidateTaskScopes = async () => {
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.snapshot(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.webData(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.moduleB(taskId) });
  };

  const roleRerunMutation = useMutation({
    mutationFn: ({ roleName, replaceRunning }: { roleName: string; replaceRunning?: boolean }) =>
      rerunModuleBRole(taskId, roleName, { replaceRunning }),
    onSuccess: async (payload, variables) => {
      await invalidateTaskScopes();
      message.success(payload.message || `模块 B ${variables.roleName} 重跑请求已提交`);
    },
    onError: async (error, variables) => {
      setClientRerunStartedAtMs((current) => {
        if (!(variables.roleName in current)) {
          return current;
        }
        const nextState = { ...current };
        delete nextState[variables.roleName];
        return nextState;
      });
      const errorText = error instanceof Error ? error.message : String(error);
      appLogger.warn("模块B页面", "模块 B role 重跑入口反馈", { taskId, error: errorText });
      if (isActiveRerunConflictMessage(errorText)) {
        await invalidateTaskScopes();
        confirmRunningRoleAction(
          variables.roleName,
          () => submitRoleRerun(variables.roleName, true),
        );
        return;
      }
      message.warning(errorText);
    },
  });

  const segmentRerunMutation = useMutation({
    mutationFn: ({ roleName, segmentId, replaceRunning }: { roleName: string; segmentId: string; replaceRunning?: boolean }) =>
      rerunModuleBRoleSegment(taskId, roleName, segmentId, { replaceRunning }),
    onSuccess: async (payload, variables) => {
      await invalidateTaskScopes();
      message.success(payload.message || `模块 B ${variables.roleName} / ${variables.segmentId} 重跑请求已提交`);
    },
    onError: async (error, variables) => {
      setClientRerunStartedAtMs((current) => {
        if (!(variables.roleName in current)) {
          return current;
        }
        const nextState = { ...current };
        delete nextState[variables.roleName];
        return nextState;
      });
      const errorText = error instanceof Error ? error.message : String(error);
      appLogger.warn("模块B页面", "模块 B segment 重跑入口反馈", { taskId, error: errorText });
      if (isActiveRerunConflictMessage(errorText)) {
        await invalidateTaskScopes();
        confirmRunningRoleAction(
          variables.roleName,
          () => submitSegmentRerun(variables.roleName, variables.segmentId, true),
          {
            content: "是否先取消当前后台进程，再重新发起新的 Segment 重跑？",
            okText: "取消并重跑",
            cancelText: variables.roleName === "role1" ? "继续查看当前输出" : "暂不重跑",
            onCancelView:
              variables.roleName === "role1"
                ? () => openRoleStreamViewer(variables.roleName, "follow-current")
                : undefined,
          },
        );
        return;
      }
      message.warning(errorText);
    },
  });

  const promptRole = promptRoleName ? roleMap[promptRoleName] : undefined;
  const activeRerun = data?.active_rerun;
  const streamViewerRole = streamViewerRoleName ? roleMap[streamViewerRoleName] : undefined;
  const streamViewerContent = streamViewerRole?.stream_preview.content || "";
  const streamViewerUpdatedAt = streamViewerRole?.stream_preview.updated_at || "";
  const streamViewerMeta = streamViewerRole?.stream_preview_meta;
  const streamViewerActive = Boolean(
    activeRerun?.active && activeRerun.role_name === streamViewerRoleName,
  );
  const streamViewerElapsedMs = streamViewerRoleName
    ? (
      clientRerunStartedAtMs[streamViewerRoleName]
        ? Math.max(0, nowMs - clientRerunStartedAtMs[streamViewerRoleName])
        : clientRerunFrozenElapsedMs[streamViewerRoleName]
          || (
            streamViewerActive && Math.max(activeRerun?.submitted_at_ms || 0, activeRerun?.started_at_ms || 0) > 0
              ? Math.max(0, nowMs - Math.max(activeRerun?.submitted_at_ms || 0, activeRerun?.started_at_ms || 0))
              : activeRerun?.role_name === streamViewerRoleName
                ? activeRerun.duration_ms || 0
                : 0
          )
    )
    : 0;
  const streamViewerTimerText = formatStopwatchSeconds(Math.floor(streamViewerElapsedMs / 1000));

  const openRoleStreamViewer = (roleName: string, mode: "rerun" | "follow-current") => {
    setStreamViewerRoleName(roleName);
    setStreamViewerMode(mode);
  };

  const confirmRunningRoleAction = (
    roleName: string,
    onConfirmReplace: () => void,
    options?: {
      content?: string;
      okText?: string;
      cancelText?: string;
      onCancelView?: () => void;
    },
  ) => {
    Modal.confirm({
      title: "检测到该角色仍在运行",
      content: options?.content || "是否先取消当前后台进程，再重新发起新的 Role 重跑？",
      okText: options?.okText || "取消并重跑",
      cancelText: options?.cancelText || "继续查看当前输出",
      onOk: onConfirmReplace,
      onCancel: options?.onCancelView || (() => openRoleStreamViewer(roleName, "follow-current")),
    });
  };

  const submitRoleRerun = (roleName: string, replaceRunning = false) => {
    setClientRerunFrozenElapsedMs((current) => {
      if (!(roleName in current)) {
        return current;
      }
      const nextState = { ...current };
      delete nextState[roleName];
      return nextState;
    });
    openRoleStreamViewer(roleName, "rerun");
    setClientRerunStartedAtMs((current) => ({
      ...current,
      [roleName]: Date.now(),
    }));
    roleRerunMutation.mutate({ roleName, replaceRunning });
  };

  const submitSegmentRerun = (roleName: string, segmentId: string, replaceRunning = false) => {
    setClientRerunStartedAtMs((current) => ({
      ...current,
      [roleName]: Date.now(),
    }));
    segmentRerunMutation.mutate({ roleName, segmentId, replaceRunning });
  };

  const handleRoleRerunClick = (roleName: string) => {
    const hasActiveRoleRerun = Boolean(activeRerun?.active && activeRerun.role_name === roleName);
    if (!hasActiveRoleRerun) {
      submitRoleRerun(roleName, false);
      return;
    }
    confirmRunningRoleAction(roleName, () => submitRoleRerun(roleName, true));
  };

  const handleSegmentRerunClick = (roleName: string, segmentId: string) => {
    const hasActiveRoleRerun = Boolean(activeRerun?.active && activeRerun.role_name === roleName);
    if (!hasActiveRoleRerun) {
      submitSegmentRerun(roleName, segmentId, false);
      return;
    }
    confirmRunningRoleAction(
      roleName,
      () => submitSegmentRerun(roleName, segmentId, true),
      {
        content: "是否先取消当前后台进程，再重新发起新的 Segment 重跑？",
        okText: "取消并重跑",
        cancelText: roleName === "role1" ? "继续查看当前输出" : "暂不重跑",
        onCancelView: roleName === "role1" ? () => openRoleStreamViewer(roleName, "follow-current") : undefined,
      },
    );
  };

  const openRoleResult = (role: TaskModuleBRole) => {
    if (role.role_name === "role1") {
      openRoleStreamViewer(role.role_name, "follow-current");
      return;
    }
    if (!role.result.available || !role.result.url) {
      message.info(`当前任务还没有 ${role.title} 的角色级成果文件。`);
      return;
    }
    window.open(role.result.url, "_blank", "noopener,noreferrer");
  };

  const openAggregateOutput = () => {
    if (!data?.aggregate_output.available || !data.aggregate_output.url) {
      message.info("当前任务还没有模块 B 聚合产物。");
      return;
    }
    window.open(data.aggregate_output.url, "_blank", "noopener,noreferrer");
  };

  if (!data && !isLoading) {
    return (
      <Card bordered={false}>
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Alert
            type="error"
            showIcon
            message="模块 B 页面数据加载失败"
            description={queryErrorText || `没有找到任务：${taskId}`}
          />
          <Empty description={`当前无法打开模块 B 页面：${taskId}`} />
        </Space>
      </Card>
    );
  }

  return (
    <div className="page-stack">
      <Alert
        type="info"
        showIcon
        message="当前页只展示 module_b 当前源码方案。"
      />

      <Card bordered={false} loading={isLoading}>
        <div className="page-toolbar">
          <div>
            <Typography.Title level={3} className="page-title">
              模块 B
            </Typography.Title>
            <Typography.Text type="secondary">
              按当前源码观察 role 模板、实现状态、segment 入口和角色级产物落盘情况。
            </Typography.Text>
          </div>
          <Space wrap>
            <Button icon={<ReloadOutlined />} loading={isFetching && !isLoading} onClick={() => void refetch()}>
              刷新状态
            </Button>
            <Button icon={<ExportOutlined />} onClick={openAggregateOutput}>
              查看聚合产物
            </Button>
          </Space>
        </div>

        {data ? (
          <Descriptions column={2} bordered className="detail-descriptions">
            <Descriptions.Item label="任务 ID">{data.task_id}</Descriptions.Item>
            <Descriptions.Item label="任务状态">{data.task_status}</Descriptions.Item>
            <Descriptions.Item label="模块 B 状态">{data.module_b_status}</Descriptions.Item>
            <Descriptions.Item label="模块 B 单元数">{data.module_b_unit_summary.total_units}</Descriptions.Item>
            <Descriptions.Item label="已完成单元">
              {data.module_b_unit_summary.done_unit_ids.length}
            </Descriptions.Item>
            <Descriptions.Item label="待处理单元">
              {data.module_b_unit_summary.problem_unit_ids.length}
            </Descriptions.Item>
          </Descriptions>
        ) : null}
      </Card>

      {roles.map((role) => {
        const selectedSegmentId = segmentSelection[role.role_name] || "";
        const selectedSegment = segmentItems.find((item) => item.segment_id === selectedSegmentId);
        const roleRerunActive =
          activeRerun?.active &&
          activeRerun.role_name === role.role_name &&
          activeRerun.mode === "role";
        const segmentRerunActive =
          activeRerun?.active &&
          activeRerun.role_name === role.role_name &&
          activeRerun.mode === "segment";
        const roleRerunLoading =
          roleRerunActive || (roleRerunMutation.isPending && roleRerunMutation.variables?.roleName === role.role_name);
        const segmentRerunLoading =
          segmentRerunActive ||
          (
            segmentRerunMutation.isPending &&
            segmentRerunMutation.variables?.roleName === role.role_name &&
            segmentRerunMutation.variables?.segmentId === selectedSegmentId
          );
        const roleResultStatusText = buildRoleResultStatusText(
          role.result.updated_at,
          role.result.updated_at_ms,
          activeRerun?.submitted_at_ms || 0,
          Boolean(roleRerunActive || segmentRerunActive),
        );
        const rerunStatusMessage = buildRerunStatusMessage(activeRerun, role.role_name);

        return (
          <Card key={role.role_name} bordered={false} className="module-b-role-card">
            <div className="module-b-role-head">
              <div className="module-b-role-meta">
                <Space wrap size={[8, 8]}>
                  <Typography.Title level={4} className="page-title">
                    {role.title}
                  </Typography.Title>
                  {getImplementationTag(role)}
                </Space>
                <Typography.Paragraph type="secondary" className="page-paragraph">
                  {role.description}
                </Typography.Paragraph>
                <Typography.Paragraph type="secondary" className="module-b-role-path">
                  {role.source_path}
                </Typography.Paragraph>
              </div>
              <Space wrap>
                <Button
                  icon={<FileSearchOutlined />}
                  onClick={() => setPromptRoleName(role.role_name)}
                  disabled={!role.prompt_template.available}
                >
                  查看 Prompt
                </Button>
                <Button icon={<EyeOutlined />} onClick={() => openRoleResult(role)}>
                  查看成果
                </Button>
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  loading={roleRerunLoading}
                  disabled={!role.supports_role_rerun}
                  onClick={() => handleRoleRerunClick(role.role_name)}
                >
                  按 Role 重跑
                </Button>
              </Space>
            </div>

            <div className="module-b-role-body">
              <div className="module-b-contract-row">
                {role.contract_fields.map((fieldName) => (
                  <Tag key={`${role.role_name}-${fieldName}`}>{fieldName}</Tag>
                ))}
              </div>
              <Typography.Paragraph className="page-paragraph">
                {role.implementation_detail}
              </Typography.Paragraph>
              <Typography.Paragraph type="secondary" className="page-paragraph">
                {roleResultStatusText}
              </Typography.Paragraph>
              {rerunStatusMessage ? <Alert type={rerunStatusMessage.type} showIcon message={rerunStatusMessage.text} /> : null}

              {role.supports_segment_retry ? (
                <div className="module-b-segment-box">
                  <div className="module-b-segment-toolbar">
                    <Select
                      value={selectedSegmentId || undefined}
                      placeholder="选择 segment"
                      options={segmentItems.map((item) => ({
                        value: item.segment_id,
                        label: formatSegmentOptionLabel(item.segment_id, item.shot_id, item.label),
                      }))}
                      onChange={(value) =>
                        setSegmentSelection((current) => ({ ...current, [role.role_name]: String(value) }))
                      }
                      className="module-b-segment-select"
                    />
                    <Button
                      icon={<ReloadOutlined />}
                      loading={segmentRerunLoading}
                      onClick={() => {
                        if (!selectedSegmentId) {
                          message.info("请先选择一个 segment。");
                          return;
                        }
                        handleSegmentRerunClick(role.role_name, selectedSegmentId);
                      }}
                    >
                      按 Segment 重跑
                    </Button>
                  </div>
                  {selectedSegment ? (
                    <div className="module-b-segment-summary">
                      <Typography.Text strong>
                        {selectedSegment.segment_id} / {selectedSegment.shot_id}
                      </Typography.Text>
                      <Typography.Text type="secondary">
                        {formatTimeRange(selectedSegment.start_time, selectedSegment.end_time)}
                      </Typography.Text>
                      <Typography.Text type="secondary">
                        {selectedSegment.label || selectedSegment.role || "未标注"}
                      </Typography.Text>
                      <Typography.Paragraph className="page-paragraph">
                        {selectedSegment.scene_desc || "当前 segment 还没有 scene_desc。"}
                      </Typography.Paragraph>
                    </div>
                  ) : (
                    <Empty description="当前任务还没有可用于 role3/role4 的 segment 列表。" />
                  )}
                </div>
              ) : null}
            </div>
          </Card>
        );
      })}

      <Modal
        title={promptRole ? `${promptRole.title} Prompt` : "Prompt"}
        open={Boolean(promptRole)}
        onCancel={() => setPromptRoleName("")}
        footer={null}
        width={980}
        destroyOnClose
      >
        {promptRole ? (
          <div className="module-b-prompt-modal">
            <Descriptions column={1} bordered size="small">
              <Descriptions.Item label="模板路径">{promptRole.prompt_template.path || "-"}</Descriptions.Item>
              <Descriptions.Item label="当前状态">{promptRole.implementation_detail}</Descriptions.Item>
            </Descriptions>
            <pre className="module-b-prompt-pre">{promptRole.prompt_template.content || "当前 prompt 模板不可用。"}</pre>
          </div>
        ) : null}
      </Modal>

      <Modal
        title={
          streamViewerRole ? (
            <Space direction="vertical" size={4}>
              <Typography.Text strong>
                {streamViewerRole.title} 流式输出{streamViewerMode === "follow-current" ? "（当前进程）" : ""}
              </Typography.Text>
              <Space wrap size={[8, 4]}>
                <Tag color={streamViewerActive ? "processing" : "default"}>
                  计时器 {streamViewerTimerText}
                </Tag>
                <Tag color="blue">Retry 第 {Math.max(1, streamViewerMeta?.current_attempt || 1)} 次</Tag>
                <Tag>首个 chunk {formatTimestampOrDash(streamViewerMeta?.first_chunk_at || "")}</Tag>
                <Tag>最近 chunk {formatTimestampOrDash(streamViewerMeta?.last_chunk_at || "")}</Tag>
              </Space>
            </Space>
          ) : "流式输出"
        }
        open={Boolean(streamViewerRoleName)}
        onCancel={() => setStreamViewerRoleName("")}
        footer={null}
        width={920}
        destroyOnClose={false}
      >
        {streamViewerRole ? (
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Alert
              type={streamViewerActive ? "info" : "success"}
              showIcon
              message={
                streamViewerActive
                  ? "当前正在持续接收 role1 流式输出。"
                  : streamViewerContent
                    ? "当前展示的是最近一次已收到的全部内容。"
                    : "当前还没有收到任何流式输出。"
              }
              description={
                streamViewerUpdatedAt
                  ? `最近文本更新时间：${streamViewerUpdatedAt}；首个 chunk：${formatTimestampOrDash(streamViewerMeta?.first_chunk_at || "")}；最近 chunk：${formatTimestampOrDash(streamViewerMeta?.last_chunk_at || "")}；当前 retry：第 ${Math.max(1, streamViewerMeta?.current_attempt || 1)} 次`
                  : "一旦后端收到首个内容 chunk，这里会自动刷新。"
              }
            />
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
              {streamViewerContent || "正在等待流式输出..."}
            </pre>
          </Space>
        ) : null}
      </Modal>
    </div>
  );
}
