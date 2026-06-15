import { z } from "zod";

/** 宽松的配置对象 schema（接受任意嵌套的 JSON 结构） */
export const looseConfigSchema = z.record(z.string(), z.unknown());

/** 默认配置响应 schema */
export const defaultConfigResponseSchema = z.object({
  ok: z.boolean(),
  config: looseConfigSchema.optional(),
  error: z.string().optional(),
});

/** 任务配置响应 schema */
export const taskConfigResponseSchema = z.object({
  ok: z.boolean(),
  task_id: z.string().optional(),
  config: looseConfigSchema.optional(),
  overrides: looseConfigSchema.optional(),
  error: z.string().optional(),
});

/** 任务配置保存响应 schema */
export const taskConfigSaveResponseSchema = z.object({
  ok: z.boolean(),
  task_id: z.string().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
});

export type DefaultConfigResponse = z.infer<typeof defaultConfigResponseSchema>;
export type TaskConfigResponse = z.infer<typeof taskConfigResponseSchema>;
export type TaskConfigSaveResponse = z.infer<typeof taskConfigSaveResponseSchema>;
