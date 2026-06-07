import { z } from "zod";

const fileAssetSchema = z.object({
  available: z.boolean(),
  url: z.string(),
  path: z.string(),
  updated_at: z.string(),
  updated_at_ms: z.number(),
});

const textFileAssetSchema = z.object({
  available: z.boolean(),
  path: z.string(),
  content: z.string(),
});

const taskTextFileAssetSchema = z.object({
  available: z.boolean(),
  path: z.string(),
  content: z.string(),
  updated_at: z.string(),
  updated_at_ms: z.number(),
});

const streamPreviewMetaSchema = z.object({
  available: z.boolean(),
  current_attempt: z.number(),
  first_chunk_at: z.string(),
  first_chunk_at_ms: z.number(),
  last_chunk_at: z.string(),
  last_chunk_at_ms: z.number(),
  completion_tokens: z.number().optional(),
  speed_tokens_per_sec: z.number().optional(),
});

const moduleBUnitSummarySchema = z.object({
  module_name: z.string(),
  total_units: z.number(),
  status_counts: z.record(z.string(), z.number()),
  pending_unit_ids: z.array(z.string()),
  running_unit_ids: z.array(z.string()),
  failed_unit_ids: z.array(z.string()),
  done_unit_ids: z.array(z.string()),
  problem_unit_ids: z.array(z.string()),
});

const moduleBSegmentItemSchema = z.object({
  segment_id: z.string(),
  shot_id: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  label: z.string(),
  role: z.string(),
  scene_desc: z.string(),
  story_outline_zh: z.string().optional().default(""),
  big_segment_id: z.string().optional().default(""),
  display_shot_id: z.string().optional().default(""),
  display_title: z.string().optional().default(""),
  display_subtitle: z.string().optional().default(""),
  subject_index: z.number().optional(),
  subject_title: z.string().optional().default(""),
});

const moduleBActiveRerunSchema = z.object({
  active: z.boolean(),
  status: z.string(),
  mode: z.string(),
  role_name: z.string(),
  segment_id: z.string(),
  shot_id: z.string(),
  submitted_at: z.string(),
  submitted_at_ms: z.number(),
  started_at: z.string(),
  started_at_ms: z.number(),
  finished_at: z.string(),
  finished_at_ms: z.number(),
  duration_ms: z.number(),
  last_error: z.string(),
  failure_reason: z.string(),
});

const renderedPromptSegmentSchema = z.object({
  segment_id: z.string(),
  content: z.string(),
  updated_at_ms: z.number().optional(),
});

const moduleBRoleSchema = z.object({
  role_name: z.string(),
  title: z.string(),
  description: z.string(),
  source_path: z.string(),
  contract_fields: z.array(z.string()),
  implementation_status: z.string(),
  supports_role_rerun: z.boolean(),
  supports_segment_retry: z.boolean(),
  segment_items: z.array(moduleBSegmentItemSchema).optional().default([]),
  active_rerun: moduleBActiveRerunSchema,
  prompt_template: textFileAssetSchema,
  rendered_prompt: taskTextFileAssetSchema,
  rendered_prompt_segments: z.array(renderedPromptSegmentSchema),
  stream_preview_segments: z.array(renderedPromptSegmentSchema),
  stream_preview: taskTextFileAssetSchema,
  stream_preview_meta: streamPreviewMetaSchema,
  result: fileAssetSchema,
  result_text: taskTextFileAssetSchema,
});

export const taskModuleBDataSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  task_status: z.string(),
  module_status: z.record(z.string(), z.string()),
  module_b_status: z.string(),
  module_b_unit_summary: moduleBUnitSummarySchema,
  aggregate_output: fileAssetSchema,
  roles: z.array(moduleBRoleSchema),
  segment_items: z.array(moduleBSegmentItemSchema),
});

export type TaskModuleBData = z.infer<typeof taskModuleBDataSchema>;
export type TaskModuleBRole = z.infer<typeof moduleBRoleSchema>;
export type TaskModuleBSegmentItem = z.infer<typeof moduleBSegmentItemSchema>;
