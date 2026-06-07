import { useEffect, useState } from "react";

import {
  ExportOutlined,
  EyeOutlined,
  FileSearchOutlined,
  ReloadOutlined,
  ToolOutlined,
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
  rebuildModuleBOutput,
  rerunModuleBRole,
  rerunModuleBRoleSegment,
  resumeModuleB,
  taskQueryKeys,
} from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import type { TaskModuleBData, TaskModuleBRole } from "@/schemas/moduleB";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import { Role3SegmentBody } from "@/components/moduleB/Role3SegmentBody";
import { Role4SegmentBody } from "@/components/moduleB/Role4SegmentBody";
import { StreamViewerModal } from "@/components/moduleB/StreamViewerModal";

function getImplementationTag(role: TaskModuleBRole) {
  if (role.implementation_status === "implemented") {
    return <Tag color="success">已生成</Tag>;
  }
  if (role.implementation_status === "placeholder") {
    return <Tag color="warning">占位中</Tag>;
  }
  if (role.implementation_status === "missing") {
    return <Tag color="error">当前未生成结果</Tag>;
  }
  return <Tag>待确认</Tag>;
}

function formatDurationMs(durationMs: number): string {
  const normalized = Math.max(0, Number(durationMs) || 0);
  if (normalized < 1000) {
    return `${normalized} ms`;
  }
  return `${(normalized / 1000).toFixed(normalized >= 10000 ? 1 : 2)} s`;
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

function computeEffectiveDurationMs(ar: NonNullable<TaskModuleBRole["active_rerun"]>): number {
  if (ar.duration_ms && ar.duration_ms > 0) return ar.duration_ms;
  if (ar.finished_at_ms && ar.started_at_ms) return Math.max(0, ar.finished_at_ms - ar.started_at_ms);
  if (ar.finished_at_ms && ar.submitted_at_ms) return Math.max(0, ar.finished_at_ms - ar.submitted_at_ms);
  return 0;
}

function buildRerunStatusMessage(
  activeRerun: TaskModuleBRole["active_rerun"] | undefined,
  roleName: string,
): { type: "info" | "success" | "error"; text: string } | null {
  if (!activeRerun || activeRerun.role_name !== roleName || !activeRerun.status) {
    return null;
  }
  if (activeRerun.active) {
    if (activeRerun.mode === "segment") {
      return {
        type: "info",
        text: `正在按 ${roleName === "role3" ? "Big Segment" : "Shot"} 重跑，已提交于 ${activeRerun.submitted_at || "-"}。`,
      };
    }
    return {
      type: "info",
      text: `正在按 Role 重跑，已提交于 ${activeRerun.submitted_at || "-"}。`,
    };
  }
  const durationMs = computeEffectiveDurationMs(activeRerun);
  if (activeRerun.status === "succeeded") {
    return {
      type: "success",
      text: `最近一次重跑已完成，耗时 ${formatDurationMs(durationMs)}。`,
    };
  }
  if (activeRerun.status === "failed") {
    const reason = activeRerun.failure_reason || "未知原因";
    const detail = activeRerun.last_error ? `；详情：${activeRerun.last_error}` : "";
    return {
      type: "error",
      text: `最近一次重跑失败，耗时 ${formatDurationMs(durationMs)}，原因：${reason}${detail}`,
    };
  }
  return null;
}

export function TaskModuleBPage() {
  const taskId = useTaskIdParam();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const [promptRoleName, setPromptRoleName] = useState("");
  const [promptSegmentId, setPromptSegmentId] = useState("");
  const [segmentSelection, setSegmentSelection] = useState<Record<string, string>>({});
  const [streamViewerRoleName, setStreamViewerRoleName] = useState("");

  useEffect(() => {
    appLogger.info("模块B页面", "模块 B 观察页已进入", { taskId });
  }, [taskId]);

  const { data, isLoading, refetch, isFetching, error } = useQuery({
    queryKey: taskQueryKeys.moduleB(taskId),
    queryFn: () => getTaskModuleBData(taskId),
    enabled: Boolean(taskId),
    placeholderData: (previousData: any) => previousData,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (payload?.roles?.some((r) => r.active_rerun?.active)) {
        return 1000;
      }
      // 流式查看器打开时持续轮询，避免后端未确认时 stop polling
      if (streamViewerRoleName) {
        return 1500;
      }
      if (payload?.task_status === "running" || payload?.module_b_status === "running") {
        return 2000;
      }
      return false;
    },
  });
  const queryErrorText = error instanceof Error ? error.message : "";

  const roles = data?.roles || [];
  const roleMap = Object.fromEntries(roles.map((role) => [role.role_name, role]));

  // 初始化 role3/role4 默认选中（已有选中值则保留）
  useEffect(() => {
    setSegmentSelection((current) => {
      const next = { ...current };
      for (const rn of ["role3", "role4"] as const) {
        const items = roleMap[rn]?.segment_items || [];
        if (!items.length) continue;
        if (next[rn]) {
          // 校验已有选中在当前 segment_items 中仍然有效
          const currentId = next[rn];
          const stillValid = items.some((item) => {
            const id = rn === "role3"
              ? (item.big_segment_id || item.segment_id)
              : item.segment_id;
            return id === currentId;
          });
          if (stillValid) continue;
          // 无效则回退到第一项
        }
        next[rn] = (rn === "role3")
          ? (items[0].big_segment_id || items[0].segment_id)
          : items[0].segment_id;
      }
      return next;
    });
  }, [roleMap["role3"]?.segment_items, roleMap["role4"]?.segment_items]);

  // prompt 默认选中
  useEffect(() => {
    if (promptRoleName !== "role3" && promptRoleName !== "role4") return;
    const segments = roleMap[promptRoleName]?.segment_items || [];
    if (!segments.length) return;
    setPromptSegmentId((prev) => {
      if (prev) {
        const stillValid = segments.some((s) => {
          const id = promptRoleName === "role3"
            ? (s.big_segment_id || s.segment_id)
            : s.segment_id;
          return id === prev;
        });
        if (stillValid) return prev;
      }
      if (promptRoleName === "role3") {
        return segments[0].big_segment_id || segments[0].segment_id;
      }
      return segments[0].segment_id;
    });
  }, [promptRoleName, roleMap[promptRoleName]?.segment_items]);

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
    onMutate: async (variables) => {
      queryClient.setQueryData(taskQueryKeys.moduleB(taskId), (old: any) => {
        if (!old?.roles) return old;
        return {
          ...old,
          roles: old.roles.map((r: any) => {
            if (r.role_name === variables.roleName) {
              return {
                ...r,
                active_rerun: {
                  active: true,
                  status: "queued",
                  mode: "role",
                  role_name: variables.roleName,
                  segment_id: "",
                  shot_id: "",
                  submitted_at: new Date().toISOString(),
                  submitted_at_ms: Date.now(),
                  started_at: "",
                  started_at_ms: 0,
                  finished_at: "",
                  finished_at_ms: 0,
                  duration_ms: 0,
                  last_error: "",
                  failure_reason: "",
                },
                stream_preview: { ...r.stream_preview, content: "", available: false },
                stream_preview_segments: (r.stream_preview_segments || []).map((seg: any) => ({ ...seg, content: "" })),
                result_text: { ...r.result_text, content: "" },
              };
            }
            return r;
          }),
        };
      });
    },
    onSuccess: async (payload, variables) => {
      // 失效其他查询范围（list/detail/snapshot/webData），保留 moduleB 手动控制
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(taskId) });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.snapshot(taskId) });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.webData(taskId) });
      // 重新拉取 moduleB，如果后端还没确认重跑，重填乐观状态保持轮询不中断
      await queryClient.refetchQueries({ queryKey: taskQueryKeys.moduleB(taskId) });
      queryClient.setQueryData(taskQueryKeys.moduleB(taskId), (old: any) => {
        if (!old?.roles) return old;
        const targetRole = old.roles.find((r: any) => r.role_name === variables.roleName);
        if (targetRole && !targetRole.active_rerun?.active) {
          return {
            ...old,
            roles: old.roles.map((r: any) =>
              r.role_name === variables.roleName
                ? {
                    ...r,
                    active_rerun: { ...r.active_rerun, active: true, status: "queued" },
                    stream_preview: { ...r.stream_preview, content: "", available: false },
                    stream_preview_segments: (r.stream_preview_segments || []).map((seg: any) => ({ ...seg, content: "" })),
                    result_text: { ...r.result_text, content: "" },
                  }
                : r
            ),
          };
        }
        return old;
      });
      message.success(payload.message || `模块 B ${variables.roleName} 重跑请求已提交`);
    },
    onError: async (error, variables) => {
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
    onMutate: async (variables) => {
      queryClient.setQueryData(taskQueryKeys.moduleB(taskId), (old: any) => {
        if (!old?.roles) return old;
        return {
          ...old,
          roles: old.roles.map((r: any) => {
            if (r.role_name === variables.roleName) {
              return {
                ...r,
                active_rerun: {
                  active: true,
                  status: "queued",
                  mode: "segment",
                  role_name: variables.roleName,
                  segment_id: variables.segmentId,
                  shot_id: "",
                  submitted_at: new Date().toISOString(),
                  submitted_at_ms: Date.now(),
                  started_at: "",
                  started_at_ms: 0,
                  finished_at: "",
                  finished_at_ms: 0,
                  duration_ms: 0,
                  last_error: "",
                  failure_reason: "",
                },
                stream_preview_segments: (r.stream_preview_segments || []).map((seg: any) =>
                  seg.segment_id === variables.segmentId ? { ...seg, content: "" } : seg
                ),
              };
            }
            return r;
          }),
        };
      });
    },
    onSuccess: async (payload, variables) => {
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(taskId) });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.snapshot(taskId) });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.webData(taskId) });
      await queryClient.refetchQueries({ queryKey: taskQueryKeys.moduleB(taskId) });
      queryClient.setQueryData(taskQueryKeys.moduleB(taskId), (old: any) => {
        if (!old?.roles) return old;
        const targetRole = old.roles.find((r: any) => r.role_name === variables.roleName);
        if (targetRole && !targetRole.active_rerun?.active) {
          return {
            ...old,
            roles: old.roles.map((r: any) =>
              r.role_name === variables.roleName
                ? {
                    ...r,
                    active_rerun: { ...r.active_rerun, active: true, status: "queued" },
                    stream_preview: { ...r.stream_preview, content: "", available: false },
                    stream_preview_segments: (r.stream_preview_segments || []).map((seg: any) =>
                      seg.segment_id === variables.segmentId ? { ...seg, content: "" } : seg
                    ),
                  }
                : r
            ),
          };
        }
        return old;
      });
      message.success(payload.message || `模块 B ${variables.roleName} / ${variables.segmentId} 重跑请求已提交`);
    },
    onError: async (error, variables) => {
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
                ? () => openRoleStreamViewer(variables.roleName)
                : undefined,
          },
        );
        return;
      }
      message.warning(errorText);
    },
  });

  const rebuildOutputMutation = useMutation({
    mutationFn: () => rebuildModuleBOutput(taskId),
    onSuccess: async (payload) => {
      await invalidateTaskScopes();
      message.success(payload.message || "模块 B 输出已重建");
    },
    onError: (error) => {
      const errorText = error instanceof Error ? error.message : String(error);
      message.warning(errorText);
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => resumeModuleB(taskId),
    onSuccess: async (payload) => {
      await invalidateTaskScopes();
      message.success(payload.message || "断点续跑已开始");
    },
    onError: (error) => {
      const errorText = error instanceof Error ? error.message : String(error);
      if (isActiveRerunConflictMessage(errorText)) {
        message.warning(errorText);
        return;
      }
      message.warning(errorText);
    },
  });

  const promptRole = promptRoleName ? roleMap[promptRoleName] : undefined;
  const promptRoleAllSegments = promptRole?.segment_items || [];
  const promptRoleSegmentOptions = (() => {
    if (!promptRoleAllSegments.length) return [];
    if (promptRoleName === "role3") {
      const seen = new Set<string>();
      const options: { value: string; label: string }[] = [];
      for (const s of promptRoleAllSegments) {
        const bid = s.big_segment_id || s.segment_id;
        if (!bid || seen.has(bid)) continue;
        seen.add(bid);
        const story = s.story_outline_zh || "";
        options.push({ value: bid, label: story ? `${bid} / ${story}` : bid });
      }
      return options;
    }
    return promptRoleAllSegments.map((s) => ({
      value: s.segment_id,
      label: s.segment_id,
    }));
  })();
  const promptRoleFirstSegId = (() => {
    if (!promptRoleAllSegments.length) return "";
    if (promptRoleName === "role3") {
      return promptRoleAllSegments[0].big_segment_id || promptRoleAllSegments[0].segment_id;
    }
    return promptRoleAllSegments[0].segment_id;
  })();
  const promptRoleSegmentContent = (() => {
    if (promptRole?.role_name !== "role3" && promptRole?.role_name !== "role4") {
      return promptRole?.rendered_prompt.content || "";
    }
    if (!promptRoleAllSegments.length) return "";
    const selectedId = promptSegmentId || promptRoleFirstSegId || "";
    return (promptRole?.rendered_prompt_segments || []).find((s) => s.segment_id === selectedId)?.content || "";
  })();

  const openRoleStreamViewer = (roleName: string) => {
    setStreamViewerRoleName(roleName);
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
      onCancel: options?.onCancelView || (() => openRoleStreamViewer(roleName)),
    });
  };

  const submitRoleRerun = (roleName: string, replaceRunning = false) => {
    openRoleStreamViewer(roleName);
    roleRerunMutation.mutate({ roleName, replaceRunning });
  };

  const submitSegmentRerun = (roleName: string, segmentId: string, replaceRunning = false) => {
    segmentRerunMutation.mutate({ roleName, segmentId, replaceRunning });
  };

  const handleRoleRerunClick = (roleName: string) => {
    const roleActiveRerun = roles.find((r) => r.role_name === roleName)?.active_rerun;
    const hasActiveRoleRerun = Boolean(
      roleActiveRerun?.active &&
      roleActiveRerun.role_name === roleName,
    );
    if (!hasActiveRoleRerun) {
      submitRoleRerun(roleName, false);
      return;
    }
    confirmRunningRoleAction(roleName, () => submitRoleRerun(roleName, true));
  };

  const handleSegmentRerunClick = (roleName: string, segmentId: string) => {
    const roleActiveRerun = roles.find((r) => r.role_name === roleName)?.active_rerun;
    const hasActiveRoleRerun = Boolean(
      roleActiveRerun?.active &&
      roleActiveRerun.role_name === roleName,
    );
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
        onCancelView: roleName === "role1" ? () => openRoleStreamViewer(roleName) : undefined,
      },
    );
  };

  const openRoleResult = (role: TaskModuleBRole) => {
    openRoleStreamViewer(role.role_name);
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
            <Button
              icon={<ToolOutlined />}
              loading={rebuildOutputMutation.isPending}
              onClick={() => {
                Modal.confirm({
                  title: "重建模块 B 输出",
                  content: "将根据已有的 role3/role4 markdown 产物重新生成 module_b_output.json，不会重新调用 LLM。",
                  onOk: () => rebuildOutputMutation.mutate(),
                });
              }}
            >
              重建 B 输出
            </Button>
            <Button
              icon={<ReloadOutlined />}
              loading={resumeMutation.isPending}
              onClick={() => {
                Modal.confirm({
                  title: "断点续跑 Role 4",
                  content: "将扫描 role3 输出的所有 shot，对缺少 role4 产物的 shot 逐个补跑。已有产物的 shot 会跳过。",
                  onOk: () => resumeMutation.mutate(),
                });
              }}
            >
              断点续跑
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
        const roleRerunActive =
          role.active_rerun?.active &&
          role.active_rerun.mode === "role" &&
          role.active_rerun.role_name === role.role_name;
        const segmentRerunActive =
          role.active_rerun?.active &&
          role.active_rerun.mode === "segment" &&
          role.active_rerun.role_name === role.role_name;
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
          role.active_rerun?.submitted_at_ms || 0,
          Boolean(roleRerunActive || segmentRerunActive),
        );
        const rerunStatusMessage = buildRerunStatusMessage(role.active_rerun, role.role_name);

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
              <Typography.Paragraph type="secondary" className="page-paragraph">
                {roleResultStatusText}
              </Typography.Paragraph>
              {rerunStatusMessage ? <Alert type={rerunStatusMessage.type} showIcon message={rerunStatusMessage.text} /> : null}

              {role.supports_segment_retry ? (
                role.role_name === "role3" ? (
                  <Role3SegmentBody
                    role={role}
                    selectedBigSegmentId={selectedSegmentId}
                    onBigSegmentChange={(bid) => {
                      appLogger.info("模块B页面", "role3 big_segment 选中变更", { rid: role.role_name, bid });
                      setSegmentSelection((cur) => ({ ...cur, [role.role_name]: bid }));
                    }}
                    onSegmentRerun={(bid) => handleSegmentRerunClick(role.role_name, bid)}
                    roleRerunLoading={roleRerunLoading}
                    segmentRerunLoading={segmentRerunLoading}
                  />
                ) : role.role_name === "role4" ? (
                  <Role4SegmentBody
                    role={role}
                    selectedSegmentId={selectedSegmentId}
                    onSegmentChange={(sid) =>
                      setSegmentSelection((cur) => ({ ...cur, [role.role_name]: sid }))
                    }
                    onSegmentRerun={(sid) => handleSegmentRerunClick(role.role_name, sid)}
                    roleRerunLoading={roleRerunLoading}
                    segmentRerunLoading={segmentRerunLoading}
                  />
                ) : null
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
            </Descriptions>
            {(promptRole.role_name === "role3" || promptRole.role_name === "role4") && promptRoleSegmentOptions.length > 0 ? (
              <div style={{ marginTop: 16, marginBottom: 8 }}>
                <Space align="center">
                  <Typography.Text strong>选择：</Typography.Text>
                  <Select
                    value={promptSegmentId || promptRoleFirstSegId || undefined}
                    options={promptRoleSegmentOptions}
                    onChange={(value) => setPromptSegmentId(String(value))}
                    style={{ width: 340 }}
                    popupClassName="module-b-prompt-select-dropdown"
                  />
                </Space>
              </div>
            ) : null}
            {promptRole.rendered_prompt.available || promptRoleSegmentContent ? (
              <>
                <Typography.Title level={5} style={{ marginTop: 16, marginBottom: 8 }}>
                  渲染后 User Prompt（已替换变量）
                </Typography.Title>
                <pre className="module-b-prompt-pre">{promptRoleSegmentContent || promptRole.rendered_prompt.content}</pre>
                <Typography.Title level={5} style={{ marginTop: 16, marginBottom: 8 }}>
                  原始模板（含占位符）
                </Typography.Title>
              </>
            ) : null}
            <pre className="module-b-prompt-pre">{promptRole.prompt_template.content || "当前 prompt 模板不可用。"}</pre>
          </div>
        ) : null}
      </Modal>

      <StreamViewerModal
        role={streamViewerRoleName ? roleMap[streamViewerRoleName] : undefined}
        open={Boolean(streamViewerRoleName)}
        onClose={() => { setStreamViewerRoleName(""); }}
      />
    </div>
  );
}
