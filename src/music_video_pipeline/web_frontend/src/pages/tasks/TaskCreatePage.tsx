import { useEffect } from "react";

import { App, Button, Card, Form, Input, Space, Typography } from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { configQueryKeys, createTask, getDefaultConfig, saveTaskConfig, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { routes } from "@/app/routes";

type TaskCreateFormValues = {
  taskId: string;
  audioPath: string;
  runsDir: string;
  storyboardTemplateFile: string;
};

export function TaskCreatePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { message } = App.useApp();
  const [form] = Form.useForm<TaskCreateFormValues>();

  useEffect(() => {
    appLogger.info("任务创建", "创建任务页已进入");
  }, []);

  const { data: defaultConfigData } = useQuery({
    queryKey: configQueryKeys.default,
    queryFn: getDefaultConfig,
  });

  const cfg = (defaultConfigData?.config || {}) as Record<string, Record<string, unknown>>;
  const defaultRunsDir = (cfg.paths?.runs_dir as string) ?? "runs";
  const defaultStoryboardTemplateFile = (cfg.module_b?.storyboard_template_file as string) ?? "";

  useEffect(() => {
    if (defaultConfigData) {
      form.setFieldsValue({
        runsDir: defaultRunsDir,
        storyboardTemplateFile: defaultStoryboardTemplateFile,
      });
    }
  }, [defaultConfigData, form, defaultRunsDir, defaultStoryboardTemplateFile]);

  const createAndSaveMutation = useMutation({
    mutationFn: async (values: TaskCreateFormValues) => {
      const createResult = await createTask({
        taskId: values.taskId.trim(),
        audioPath: values.audioPath.trim(),
        configPath: "configs/common.json",
      });
      const overrides: Record<string, unknown> = {};
      if (values.runsDir?.trim() && values.runsDir.trim() !== defaultRunsDir) {
        overrides.paths = { runs_dir: values.runsDir.trim() };
      }
      if (values.storyboardTemplateFile?.trim() && values.storyboardTemplateFile.trim() !== defaultStoryboardTemplateFile) {
        overrides.module_b = { storyboard_template_file: values.storyboardTemplateFile.trim() };
      }
      if (Object.keys(overrides).length > 0) {
        await saveTaskConfig(values.taskId.trim(), overrides);
      }
      return createResult;
    },
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
          填写任务信息和配置项。配置默认值来自公共配置，可按需修改。
        </Typography.Paragraph>
      </Card>

      <Card bordered={false}>
        <Form<TaskCreateFormValues>
          layout="vertical"
          form={form}
          onFinish={(values) => {
            createAndSaveMutation.mutate(values);
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
            label="运行目录 (runs_dir)"
            name="runsDir"
          >
            <Input placeholder={`默认: ${defaultRunsDir}`} />
          </Form.Item>
          <Form.Item
            label="故事板模板文件 (storyboard_template_file)"
            name="storyboardTemplateFile"
          >
            <Input placeholder={`默认: ${defaultStoryboardTemplateFile || "未设置"}`} />
          </Form.Item>
          <Space>
            <Button onClick={() => navigate(routes.taskList)}>取消</Button>
            <Button type="primary" htmlType="submit" loading={createAndSaveMutation.isPending}>
              创建任务
            </Button>
          </Space>
        </Form>
      </Card>
    </div>
  );
}
