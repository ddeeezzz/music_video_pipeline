import { z } from "zod";

const moduleStatusSchema = z.record(z.string(), z.string());

export const taskSummarySchema = z.object({
  task_id: z.string(),
  status: z.string(),
  audio_path: z.string(),
  config_path: z.string(),
  output_video_path: z.string(),
  updated_at: z.string(),
  module_status: moduleStatusSchema,
});

export const taskDetailSchema = z.object({
  task_id: z.string(),
  status: z.string(),
  audio_path: z.string(),
  config_path: z.string(),
  output_video_path: z.string(),
  updated_at: z.string(),
  created_at: z.string(),
  error_message: z.string(),
  module_status: moduleStatusSchema,
});

export const taskListResponseSchema = z.object({
  ok: z.boolean(),
  current_task_id: z.string(),
  tasks: z.array(taskSummarySchema),
});

export const taskDetailResponseSchema = z.object({
  ok: z.boolean(),
  task: taskDetailSchema.nullable(),
});

export const taskActionResponseSchema = z.object({
  ok: z.boolean(),
  task_id: z.string().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
  task: taskDetailSchema.nullable().optional(),
});

export type TaskSummary = z.infer<typeof taskSummarySchema>;
export type TaskDetail = z.infer<typeof taskDetailSchema>;
