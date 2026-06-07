import { z } from "zod";

const moduleCShotSchema = z.object({
  shot_id: z.string(),
  unit_index: z.number(),
  segment_id: z.string(),
  status: z.enum(["pending", "running", "done", "failed"]),
  frame_status_start: z.string(),
  frame_status_end: z.string(),
  frame_url_start: z.string(),
  frame_url_end: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  duration: z.number(),
  error_message: z.string(),
  assembled_prompt_start: z.string().optional().default(""),
  assembled_prompt_end: z.string().optional().default(""),
  role4_prompt: z
    .object({
      keyframe_prompt_start_zh: z.string().optional().default(""),
      keyframe_prompt_start_en: z.string().optional().default(""),
      keyframe_prompt_end_zh: z.string().optional().default(""),
      keyframe_prompt_end_en: z.string().optional().default(""),
      video_prompt_zh: z.string().optional().default(""),
      video_prompt_en: z.string().optional().default(""),
    })
    .optional()
    .default({
      keyframe_prompt_start_zh: "",
      keyframe_prompt_start_en: "",
      keyframe_prompt_end_zh: "",
      keyframe_prompt_end_en: "",
      video_prompt_zh: "",
      video_prompt_en: "",
    }),
});

const moduleCActiveRerunSchema = z.object({
  active: z.boolean(),
  status: z.string(),
  shot_id: z.string(),
  frame_type: z.string().optional().default(""),
  submitted_at: z.string(),
  submitted_at_ms: z.number().optional().default(0),
  started_at_ms: z.number().optional().default(0),
  last_error: z.string().optional().default(""),
  failure_reason: z.string().optional().default(""),
});

const moduleCUnitSummarySchema = z.object({
  module_name: z.string(),
  total_units: z.number(),
  status_counts: z.record(z.string(), z.number()),
  pending_unit_ids: z.array(z.string()),
  running_unit_ids: z.array(z.string()),
  failed_unit_ids: z.array(z.string()),
  done_unit_ids: z.array(z.string()),
  problem_unit_ids: z.array(z.string()),
});

export const taskModuleCDataSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  module_c_status: z.string(),
  unit_summary: moduleCUnitSummarySchema,
  shots: z.array(moduleCShotSchema),
  active_rerun: moduleCActiveRerunSchema.nullable(),
});

export type TaskModuleCData = z.infer<typeof taskModuleCDataSchema>;
export type TaskModuleCShot = z.infer<typeof moduleCShotSchema>;
