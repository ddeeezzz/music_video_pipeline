import { useEffect, useState } from "react";

import {
  DeploymentUnitOutlined,
  FileAddOutlined,
  FolderOpenOutlined,
  PictureOutlined,
  PlaySquareOutlined,
  ProfileOutlined,
  RadarChartOutlined,
  ReadOutlined,
  UnorderedListOutlined,
} from "@ant-design/icons";
import { Button, Descriptions, Layout, Menu, Modal, Typography } from "antd";
import { useQuery } from "@tanstack/react-query";
import { Link, Outlet, useLocation, useParams } from "react-router-dom";

import { getTaskDetail, taskQueryKeys } from "@/api/taskApi";
import { routes } from "@/app/routes";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";

const { Sider, Content } = Layout;
const LAST_TASK_ID_STORAGE_KEY = "mvpl:last-task-id";

function getSelectedMenuKey(pathname: string): string {
  if (pathname === routes.taskCreate) {
    return "create";
  }
  return "tasks";
}

function getSelectedTaskMenuKey(pathname: string): string {
  if (pathname.endsWith("/monitor")) {
    return "task-monitor";
  }
  if (pathname.endsWith("/review")) {
    return "task-review";
  }
  if (pathname.endsWith("/module-a")) {
    return "task-module-a";
  }
  if (pathname.endsWith("/module-b")) {
    return "task-module-b";
  }
  if (pathname.endsWith("/module-c")) {
    return "task-module-c";
  }
  if (pathname.endsWith("/module-d")) {
    return "task-module-d";
  }
  return "task-detail";
}

export function AppShell() {
  const location = useLocation();
  const params = useParams();
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [rememberedTaskId, setRememberedTaskId] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return window.sessionStorage.getItem(LAST_TASK_ID_STORAGE_KEY)?.trim() || "";
  });
  const routeTaskId = String(params.taskId || "").trim();
  const taskId = routeTaskId || rememberedTaskId;
  const selectedMenuKey = getSelectedMenuKey(location.pathname);
  const selectedTaskMenuKeys = routeTaskId ? [getSelectedTaskMenuKey(location.pathname)] : [];

  useEffect(() => {
    if (!routeTaskId) {
      return;
    }
    setRememberedTaskId(routeTaskId);
    window.sessionStorage.setItem(LAST_TASK_ID_STORAGE_KEY, routeTaskId);
  }, [routeTaskId]);

  const { data, isLoading } = useQuery({
    queryKey: taskQueryKeys.detail(taskId),
    queryFn: () => getTaskDetail(taskId),
    enabled: Boolean(taskId),
  });
  const task = data?.task;
  const menuItems = [
    {
      key: "tasks",
      icon: <UnorderedListOutlined />,
      label: <Link to={routes.taskList}>任务列表</Link>,
    },
    {
      key: "create",
      icon: <FileAddOutlined />,
      label: <Link to={routes.taskCreate}>创建任务</Link>,
    },
  ];
  const taskMenuItems = taskId
    ? [
        {
          key: "task-detail",
          icon: <ProfileOutlined />,
          label: <Link to={routes.taskDetail(taskId)}>详情</Link>,
        },
        {
          key: "task-monitor",
          icon: <RadarChartOutlined />,
          label: <Link to={routes.taskMonitor(taskId)}>监督</Link>,
        },
        {
          key: "task-review",
          icon: <ReadOutlined />,
          label: <Link to={routes.taskReview(taskId)}>审阅</Link>,
        },
        {
          key: "task-module-a",
          icon: <FolderOpenOutlined />,
          label: <Link to={routes.taskModuleA(taskId)}>模块 A</Link>,
        },
        {
          key: "task-module-b",
          icon: <DeploymentUnitOutlined />,
          label: <Link to={routes.taskModuleB(taskId)}>模块 B</Link>,
        },
        {
          key: "task-module-c",
          icon: <PictureOutlined />,
          label: <Link to={routes.taskModuleC(taskId)}>模块 C</Link>,
        },
        {
          key: "task-module-d",
          icon: <PlaySquareOutlined />,
          label: <Link to={routes.taskModuleD(taskId)}>模块 D</Link>,
        },
      ]
    : [];

  return (
    <Layout className="app-shell">
      <Sider
        width={272}
        breakpoint="lg"
        collapsedWidth={72}
        theme="light"
        className="app-shell__sider"
      >
        <div className="app-shell__sider-inner">
          <div className="app-shell__brand">
            <Typography.Text className="app-shell__brand-kicker">MVPL</Typography.Text>
          </div>
          {taskId ? (
            <div className="app-shell__task-panel">
              <div className="app-shell__task-panel-head">
                <Typography.Text className="app-shell__active-task-label">当前任务</Typography.Text>
                <Typography.Title level={5} className="app-shell__task-title">
                  {taskId}
                </Typography.Title>
                {task ? <TaskStatusTag status={task.status} /> : null}
              </div>
              <Button block onClick={() => setTaskDetailOpen(true)} loading={isLoading} disabled={!task}>
                查看详情
              </Button>
              <Menu
                mode="inline"
                selectedKeys={selectedTaskMenuKeys}
                className="app-shell__task-menu"
                items={taskMenuItems}
              />
            </div>
          ) : null}
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            items={menuItems}
          />
        </div>
      </Sider>
      <Layout>
        <Content className="app-shell__content">
          <div className="app-shell__page">
            <div className="app-shell__page-body">
              <Outlet />
            </div>
          </div>
        </Content>
      </Layout>
      <Modal
        title={task ? `任务详情：${task.task_id}` : "任务详情"}
        open={taskDetailOpen}
        onCancel={() => setTaskDetailOpen(false)}
        footer={null}
        destroyOnClose
        width={820}
      >
        {task ? (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="任务 ID">{task.task_id}</Descriptions.Item>
            <Descriptions.Item label="当前状态">
              <TaskStatusTag status={task.status} />
            </Descriptions.Item>
            <Descriptions.Item label="音频路径">
              <Typography.Paragraph copyable className="task-path-text">
                {task.audio_path || "-"}
              </Typography.Paragraph>
            </Descriptions.Item>
            <Descriptions.Item label="配置路径">
              <Typography.Paragraph copyable className="task-path-text">
                {task.config_path || "-"}
              </Typography.Paragraph>
            </Descriptions.Item>
            <Descriptions.Item label="输出视频路径">
              <Typography.Paragraph copyable className="task-path-text">
                {task.output_video_path || "-"}
              </Typography.Paragraph>
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">{task.created_at || "-"}</Descriptions.Item>
            <Descriptions.Item label="更新时间">{task.updated_at || "-"}</Descriptions.Item>
          </Descriptions>
        ) : (
          <Typography.Text type="secondary">当前还没有拿到任务详情数据。</Typography.Text>
        )}
      </Modal>
    </Layout>
  );
}
