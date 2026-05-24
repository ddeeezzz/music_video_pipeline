import { useEffect, useMemo, useState } from "react";

import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  Modal,
  Row,
  Space,
  Typography,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { copyTask, getTaskDetail, renameTask, rerunTask, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { routes } from "@/app/routes";
import { EmptyState } from "@/components/common/EmptyState";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";

type RenameFormValues = {
  newTaskId: string;
};

type CopyFormValues = {
  newTaskId: string;
  audioPath: string;
  configPath: string;
};

export function TaskDetailPage() {
  const taskId = useTaskIdParam();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const [renameOpen, setRenameOpen] = useState(false);
  const [copyOpen, setCopyOpen] = useState(false);
  const [renameForm] = Form.useForm<RenameFormValues>();
  const [copyForm] = Form.useForm<CopyFormValues>();

  useEffect(() => {
    appLogger.info("任务详情", "任务详情页已进入", { taskId });
  }, [taskId]);

  const { data, isLoading } = useQuery({
    queryKey: taskQueryKeys.detail(taskId),
    queryFn: () => getTaskDetail(taskId),
    enabled: Boolean(taskId),
  });

  const task = data?.task;

  const invalidateTaskScopes = async (targetTaskId?: string): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.snapshot(taskId) });
    await queryClient.invalidateQueries({ queryKey: taskQueryKeys.webData(taskId) });
    if (targetTaskId) {
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(targetTaskId) });
    }
  };

  const renameMutation = useMutation({
    mutationFn: renameTask,
    onSuccess: async (payload) => {
      const nextTaskId = payload.task_id || "";
      await invalidateTaskScopes(nextTaskId);
      message.success(`任务已改名为：${nextTaskId}`);
      setRenameOpen(false);
      renameForm.resetFields();
      navigate(routes.taskDetail(nextTaskId));
    },
    onError: (error) => {
      appLogger.error("任务详情", "任务改名失败", { error: error instanceof Error ? error.message : String(error) });
      message.error(error instanceof Error ? error.message : String(error));
    },
  });

  const copyMutation = useMutation({
    mutationFn: copyTask,
    onSuccess: async (payload) => {
      const nextTaskId = payload.task_id || "";
      await invalidateTaskScopes(nextTaskId);
      message.success(`任务已复制为：${nextTaskId}`);
      setCopyOpen(false);
      copyForm.resetFields();
      navigate(routes.taskDetail(nextTaskId));
    },
    onError: (error) => {
      appLogger.error("任务详情", "任务复制失败", { error: error instanceof Error ? error.message : String(error) });
      message.error(error instanceof Error ? error.message : String(error));
    },
  });

  const rerunMutation = useMutation({
    mutationFn: () => rerunTask(taskId),
    onSuccess: async (payload) => {
      await invalidateTaskScopes(taskId);
      message.success(payload.message || `任务已开始重跑：${taskId}`);
      navigate(routes.taskMonitor(taskId));
    },
    onError: (error) => {
      appLogger.error("任务详情", "任务重跑失败", { error: error instanceof Error ? error.message : String(error) });
      message.error(error instanceof Error ? error.message : String(error));
    },
  });

  const moduleEntries = useMemo(() => Object.entries(task?.module_status || {}), [task?.module_status]);

  if (!task && !isLoading) {
    return (
      <Card bordered={false}>
        <EmptyState description={`没有找到任务：${taskId}`} />
      </Card>
    );
  }

  return (
    <div className="page-stack">
      {task?.error_message ? <Alert type="error" showIcon message="任务错误摘要" description={task.error_message} /> : null}

      <Card bordered={false} loading={isLoading}>
        <div className="page-toolbar">
          <div>
            <Typography.Title level={3} className="page-title">
              任务详情
            </Typography.Title>
            <Typography.Text type="secondary">
              这一页先把任务基础信息、模块状态和常用动作统一收口，为后续监督页和审阅页提供稳定入口。
            </Typography.Text>
          </div>
          <Space wrap>
            <Button onClick={() => setRenameOpen(true)} disabled={!task}>
              改名
            </Button>
            <Button onClick={() => setCopyOpen(true)} disabled={!task}>
              复制
            </Button>
            <Button type="primary" loading={rerunMutation.isPending} onClick={() => rerunMutation.mutate()} disabled={!task}>
              重跑任务
            </Button>
          </Space>
        </div>

        {task ? (
          <Descriptions column={2} bordered className="detail-descriptions">
            <Descriptions.Item label="任务 ID">{task.task_id}</Descriptions.Item>
            <Descriptions.Item label="当前状态">
              <TaskStatusTag status={task.status} />
            </Descriptions.Item>
            <Descriptions.Item label="音频路径">{task.audio_path || "-"}</Descriptions.Item>
            <Descriptions.Item label="配置路径">{task.config_path || "-"}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{task.created_at || "-"}</Descriptions.Item>
            <Descriptions.Item label="更新时间">{task.updated_at || "-"}</Descriptions.Item>
            <Descriptions.Item label="输出视频" span={2}>
              {task.output_video_path || "-"}
            </Descriptions.Item>
          </Descriptions>
        ) : null}
      </Card>

      <Card bordered={false}>
        <div className="page-toolbar">
          <Typography.Title level={4} className="page-title">
            模块状态
          </Typography.Title>
          <Space wrap>
            <Button onClick={() => navigate(routes.taskMonitor(taskId))}>进入监督页</Button>
            <Button onClick={() => navigate(routes.taskReview(taskId))}>进入审阅页</Button>
            <Button onClick={() => navigate(routes.taskModuleA(taskId))}>进入模块 A 页面</Button>
            <Button onClick={() => navigate(routes.taskModuleB(taskId))}>进入模块 B 页面</Button>
          </Space>
        </div>
        {moduleEntries.length ? (
          <Row gutter={[16, 16]}>
            {moduleEntries.map(([moduleName, status]) => (
              <Col key={moduleName} xs={24} sm={12} lg={8} xl={6}>
                <Card size="small" className="module-card">
                  <Space direction="vertical" size={8}>
                    <Typography.Text strong>{moduleName}</Typography.Text>
                    <TaskStatusTag status={status} />
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <EmptyState description="当前任务还没有模块状态摘要。" />
        )}
      </Card>

      <Modal
        title="任务改名"
        open={renameOpen}
        onCancel={() => setRenameOpen(false)}
        onOk={() => renameForm.submit()}
        confirmLoading={renameMutation.isPending}
        destroyOnClose
      >
        <Form<RenameFormValues>
          layout="vertical"
          form={renameForm}
          initialValues={{ newTaskId: taskId }}
          onFinish={(values) => {
            renameMutation.mutate({
              oldTaskId: taskId,
              newTaskId: values.newTaskId.trim(),
            });
          }}
        >
          <Form.Item
            label="新任务 ID"
            name="newTaskId"
            rules={[{ required: true, message: "请输入新的任务 ID" }]}
          >
            <Input />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="复制任务"
        open={copyOpen}
        onCancel={() => setCopyOpen(false)}
        onOk={() => copyForm.submit()}
        confirmLoading={copyMutation.isPending}
        destroyOnClose
      >
        <Form<CopyFormValues>
          layout="vertical"
          form={copyForm}
          initialValues={{
            newTaskId: `${taskId}_copy`,
            audioPath: task?.audio_path || "",
            configPath: task?.config_path || "",
          }}
          onFinish={(values) => {
            copyMutation.mutate({
              sourceTaskId: taskId,
              newTaskId: values.newTaskId.trim(),
              audioPath: values.audioPath.trim(),
              configPath: values.configPath.trim(),
            });
          }}
        >
          <Form.Item
            label="新任务 ID"
            name="newTaskId"
            rules={[{ required: true, message: "请输入新的任务 ID" }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            label="音频路径"
            name="audioPath"
            rules={[{ required: true, message: "请输入音频路径" }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            label="配置路径"
            name="configPath"
            rules={[{ required: true, message: "请输入配置路径" }]}
          >
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
