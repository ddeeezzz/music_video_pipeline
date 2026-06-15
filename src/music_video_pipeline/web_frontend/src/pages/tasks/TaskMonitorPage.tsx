import { useEffect } from "react";

import { Alert, Button, Card, Col, Row, Space, Statistic, Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useQuery } from "@tanstack/react-query";

import { getTaskSnapshot, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { EmptyState } from "@/components/common/EmptyState";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import { useTaskMonitorSocket } from "@/features/monitor/useTaskMonitorSocket";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import type { TaskMonitorChainRow } from "@/schemas/monitor";

export function TaskMonitorPage() {
  const taskId = useTaskIdParam();

  useEffect(() => {
    appLogger.info("任务监督", "任务监督页已进入", { taskId });
  }, [taskId]);

  const { connectionState, lastMessageAt } = useTaskMonitorSocket(taskId, Boolean(taskId));

  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: taskQueryKeys.snapshot(taskId),
    queryFn: () => getTaskSnapshot(taskId),
    enabled: Boolean(taskId),
    refetchInterval: connectionState === "open" ? false : 5_000,
    staleTime: 0,
  });

  const columns: ColumnsType<TaskMonitorChainRow> = [
    {
      title: "序号",
      dataIndex: "unit_index",
      key: "unit_index",
      width: 80,
    },
    {
      title: "segment_id",
      dataIndex: "segment_id",
      key: "segment_id",
      width: 220,
      ellipsis: true,
    },
    {
      title: "B",
      dataIndex: "b_status",
      key: "b_status",
      width: 100,
      render: (value: string) => <TaskStatusTag status={value} />,
    },
    {
      title: "C",
      dataIndex: "c_status",
      key: "c_status",
      width: 100,
      render: (value: string) => <TaskStatusTag status={value} />,
    },
    {
      title: "D",
      dataIndex: "d_status",
      key: "d_status",
      width: 100,
      render: (value: string) => <TaskStatusTag status={value} />,
    },
    {
      title: "链路状态",
      dataIndex: "chain_status",
      key: "chain_status",
      width: 120,
      render: (value: string) => <TaskStatusTag status={value} />,
    },
    {
      title: "错误摘要",
      key: "error_summary",
      render: (_, row) => row.b_error_message || row.c_error_message || row.d_error_message || "-",
    },
  ];

  return (
    <div className="page-stack">
      <Card bordered={false}>
        <div className="page-toolbar">
          <div>
            <Typography.Title level={3} className="page-title">
              任务监督
            </Typography.Title>
            <Typography.Text type="secondary">
              首屏用 `/snapshot` 启动，后续优先吃 `/ws` 实时推送。这里不再手工 patch DOM，而是直接更新 Query 缓存。
            </Typography.Text>
          </div>
          <Space wrap>
            <Typography.Text type="secondary">
              连接状态：{connectionState}
              {lastMessageAt ? ` / 最近快照：${new Date(lastMessageAt).toLocaleTimeString()}` : ""}
            </Typography.Text>
            <Button
              loading={isFetching && !isLoading}
              onClick={() => {
                appLogger.info("任务监督", "手动刷新监督快照", { taskId });
                void refetch();
              }}
            >
              刷新快照
            </Button>
          </Space>
        </div>
        <Alert
          type={connectionState === "open" ? "success" : connectionState === "error" ? "error" : "info"}
          showIcon
          message={
            connectionState === "open"
              ? "WebSocket 已连接，监督页正在实时接收快照。"
              : connectionState === "error"
                ? "WebSocket 当前不可用，页面会继续按间隔拉取快照。"
                : "监督页正在建立实时连接。"
          }
        />
      </Card>

      <Card bordered={false} loading={isLoading}>
        <Typography.Title level={4} className="page-title">
          模块概览
        </Typography.Title>
        <Row gutter={[16, 16]}>
          {Object.entries(data?.module_overview || {}).map(([moduleName, moduleOverview]) => (
            <Col key={moduleName} xs={24} sm={12} lg={8} xl={6}>
              <Card size="small" className="module-card">
                <Space direction="vertical" size={10}>
                  <Typography.Text strong>{moduleName}</Typography.Text>
                  <TaskStatusTag status={moduleOverview.status} />
                  <Typography.Text type="secondary">
                    进度：{moduleOverview.done}/{moduleOverview.total} ({moduleOverview.progress}%)
                  </Typography.Text>
                  {moduleOverview.error_message ? (
                    <Typography.Text type="danger">{moduleOverview.error_message}</Typography.Text>
                  ) : null}
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      <Card bordered={false}>
        <Typography.Title level={4} className="page-title">
          链路统计
        </Typography.Title>
        <Row gutter={[16, 16]}>
          <Col xs={12} md={6}>
            <Statistic title="pending" value={data?.chain_counts.pending || 0} />
          </Col>
          <Col xs={12} md={6}>
            <Statistic title="running" value={data?.chain_counts.running || 0} />
          </Col>
          <Col xs={12} md={6}>
            <Statistic title="done" value={data?.chain_counts.done || 0} />
          </Col>
          <Col xs={12} md={6}>
            <Statistic title="failed" value={data?.chain_counts.failed || 0} />
          </Col>
        </Row>
      </Card>

      <Card bordered={false}>
        <Typography.Title level={4} className="page-title">
          B / C / D 链路表
        </Typography.Title>
        <Table
          rowKey={(row) => `${row.unit_index}-${row.segment_id}-${row.shot_id}`}
          columns={columns}
          dataSource={data?.bcd_chains || []}
          locale={{
            emptyText: <EmptyState description="当前任务还没有链路单元，监督页会在模块 B/C/D 建立后自动出现。" />,
          }}
          pagination={{ pageSize: 12, showSizeChanger: false }}
          scroll={{ x: 1000 }}
        />
      </Card>
    </div>
  );
}
