import { useEffect } from "react";

import { App, Button, Card, Form, Input, Space, Typography } from "antd";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { createTask, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { routes } from "@/app/routes";

type TaskCreateFormValues = {
  taskId: string;
  audioPath: string;
  configPath: string;
};

export function TaskCreatePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const [form] = Form.useForm<TaskCreateFormValues>();

  useEffect(() => {
    appLogger.info("任务创建", "创建任务页已进入");
  }, []);

  const createTaskMutation = useMutation({
    mutationFn: createTask,
    onSuccess: async (payload) => {
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.list });
      await queryClient.invalidateQueries({ queryKey: taskQueryKeys.detail(payload.task_id || "") });
      message.success(`任务已创建：${payload.task_id}`);
      navigate(routes.taskDetail(payload.task_id || ""));
    },
    onError: (error) => {
      appLogger.error("任务创建", "创建任务失败", { error: error instanceof Error ? error.message : String(error) });
      message.error(error instanceof Error ? error.message : String(error));
    },
  });

  return (
    <div className="page-stack">
      <Card bordered={false}>
        <Typography.Title level={3} className="page-title">
          创建任务
        </Typography.Title>
        <Typography.Paragraph type="secondary" className="page-paragraph">
          第一阶段先严格对齐现有后端创建接口，只提交 `task_id`、`audio_path` 和 `config_path`。
          额外业务字段等后端契约确认后再往前接。
        </Typography.Paragraph>
      </Card>

      <Card bordered={false}>
        <Form<TaskCreateFormValues>
          layout="vertical"
          form={form}
          onFinish={(values) => {
            createTaskMutation.mutate({
              taskId: values.taskId.trim(),
              audioPath: values.audioPath.trim(),
              configPath: values.configPath.trim(),
            });
          }}
        >
          <Form.Item
            label="任务 ID"
            name="taskId"
            rules={[{ required: true, message: "请输入任务 ID" }]}
          >
            <Input placeholder="例如 demo_2025_001" />
          </Form.Item>
          <Form.Item
            label="音频路径"
            name="audioPath"
            rules={[{ required: true, message: "请输入音频路径" }]}
          >
            <Input placeholder="例如 resources/juebieshu.m4a" />
          </Form.Item>
          <Form.Item
            label="配置路径"
            name="configPath"
            rules={[{ required: true, message: "请输入配置路径" }]}
          >
            <Input placeholder="例如 configs/music_wsl/default.json" />
          </Form.Item>
          <Space>
            <Button onClick={() => navigate(routes.taskList)}>取消</Button>
            <Button type="primary" htmlType="submit" loading={createTaskMutation.isPending}>
              创建任务
            </Button>
          </Space>
        </Form>
      </Card>
    </div>
  );
}
