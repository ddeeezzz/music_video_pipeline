import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";

import { Button, Card, Input, Select, Space, Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";

import { listTasks, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { routes } from "@/app/routes";
import { EmptyState } from "@/components/common/EmptyState";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import type { TaskSummary } from "@/schemas/tasks";

function renderModuleStatus(moduleStatus: Record<string, string>): ReactNode {
  const entries = Object.entries(moduleStatus || {});
  if (!entries.length) {
    return <Typography.Text type="secondary">暂无模块状态</Typography.Text>;
  }

  return (
    <Space wrap size={[4, 4]}>
      {entries.map(([moduleName, status]) => (
        <Space key={moduleName} size={4}>
          <Typography.Text strong>{moduleName}</Typography.Text>
          <TaskStatusTag status={status} />
        </Space>
      ))}
    </Space>
  );
}

export function TaskListPage() {
  const navigate = useNavigate();
  const [keyword, setKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const { data, isLoading, isFetching } = useQuery({
    queryKey: taskQueryKeys.list,
    queryFn: listTasks,
  });

  useEffect(() => {
    appLogger.info("任务列表", "任务列表页已进入");
  }, []);

  useEffect(() => {
    if (data) {
      appLogger.info("任务列表", "任务列表数据已更新", {
        taskCount: data.tasks.length,
        currentTaskId: data.current_task_id,
      });
    }
  }, [data]);

  const statusOptions = useMemo(() => {
    const statuses = new Set<string>();
    for (const task of data?.tasks || []) {
      statuses.add(task.status);
    }
    return ["all", ...Array.from(statuses)];
  }, [data?.tasks]);

  const filteredTasks = useMemo(() => {
    const normalizedKeyword = keyword.trim().toLowerCase();
    return (data?.tasks || []).filter((task) => {
      const matchesKeyword =
        !normalizedKeyword ||
        task.task_id.toLowerCase().includes(normalizedKeyword) ||
        task.audio_path.toLowerCase().includes(normalizedKeyword) ||
        task.config_path.toLowerCase().includes(normalizedKeyword);
      const matchesStatus = statusFilter === "all" || task.status === statusFilter;
      return matchesKeyword && matchesStatus;
    });
  }, [data?.tasks, keyword, statusFilter]);

  const columns: ColumnsType<TaskSummary> = [
    {
      title: "任务 ID",
      dataIndex: "task_id",
      key: "task_id",
      width: 220,
      render: (value: string) => (
        <Button type="link" className="link-button" onClick={() => navigate(routes.taskDetail(value))}>
          {value}
        </Button>
      ),
    },
    {
      title: "当前状态",
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (value: string) => <TaskStatusTag status={value} />,
    },
    {
      title: "模块进度",
      dataIndex: "module_status",
      key: "module_status",
      render: (value: Record<string, string>) => renderModuleStatus(value),
    },
    {
      title: "音频路径",
      dataIndex: "audio_path",
      key: "audio_path",
      ellipsis: true,
    },
    {
      title: "更新时间",
      dataIndex: "updated_at",
      key: "updated_at",
      width: 220,
    },
    {
      title: "操作",
      key: "actions",
      width: 220,
      render: (_, task) => (
        <Space wrap>
          <Link to={routes.taskDetail(task.task_id)}>详情</Link>
          <Link to={routes.taskMonitor(task.task_id)}>监督</Link>
          <Link to={routes.taskReview(task.task_id)}>审阅</Link>
        </Space>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <Card bordered={false}>
        <div className="page-toolbar">
          <div>
            <Typography.Title level={3} className="page-title">
              任务列表
            </Typography.Title>
            <Typography.Text type="secondary">
              这里先打通现有 `/api/tasks`，后续筛选、搜索和批量操作都在这个工作台里继续长。
            </Typography.Text>
          </div>
          <Space wrap>
            <Button loading={isFetching && !isLoading} onClick={() => navigate(0)}>
              刷新页面
            </Button>
            <Button type="primary" onClick={() => navigate(routes.taskCreate)}>
              创建任务
            </Button>
          </Space>
        </div>
      </Card>

      <Card bordered={false}>
        <Space wrap className="page-toolbar">
          <Input.Search
            allowClear
            placeholder="按 task_id / 音频路径 / 配置路径搜索"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            style={{ maxWidth: 360 }}
          />
          <Select
            value={statusFilter}
            onChange={setStatusFilter}
            style={{ width: 180 }}
            options={statusOptions.map((status) => ({
              label: status === "all" ? "全部状态" : status,
              value: status,
            }))}
          />
        </Space>
        <Table
          rowKey="task_id"
          loading={isLoading}
          columns={columns}
          dataSource={filteredTasks}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          locale={{
            emptyText: <EmptyState description="当前还没有任务记录，先创建一个任务再回来。" />,
          }}
          scroll={{ x: 1100 }}
        />
      </Card>
    </div>
  );
}
