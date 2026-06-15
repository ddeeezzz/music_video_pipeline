import { z } from "zod";

const moduleDShotSchema = z.object({
  shot_id: z.string(),
  unit_index: z.number(),
  segment_id: z.string(),
  status: z.enum(["pending", "running", "done", "failed"]),
  video_url: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  duration: z.number(),
  error_message: z.string(),
  scene_desc: z.string().optional(),
  big_segment_id: z.string().optional(),
  keyframe_start_url: z.string().optional(),
  keyframe_end_url: z.string().optional(),
  keyframe_prompt_start_zh: z.string().optional(),
  keyframe_prompt_start_en: z.string().optional(),
  keyframe_prompt_end_zh: z.string().optional(),
  keyframe_prompt_end_en: z.string().optional(),
  video_prompt_zh: z.string().optional(),
  video_prompt_en: z.string().optional(),
});

const moduleDActiveRerunSchema = z.object({
  active: z.boolean(),
  status: z.string(),
  big_segment_id: z.string().optional(),
  segment_id: z.string().optional(),
  frame_type: z.string().optional(),
  phase: z.string().optional(),
  submitted_at: z.string(),
  submitted_at_ms: z.number().optional().default(0),
  started_at_ms: z.number().optional().default(0),
  last_error: z.string().optional().default(""),
  failure_reason: z.string().optional().default(""),
  video_url: z.string().optional(),
});

const moduleDUnitSummarySchema = z.object({
  module_name: z.string(),
  total_units: z.number(),
  status_counts: z.record(z.string(), z.number()),
  pending_unit_ids: z.array(z.string()),
  running_unit_ids: z.array(z.string()),
  failed_unit_ids: z.array(z.string()),
  done_unit_ids: z.array(z.string()),
  problem_unit_ids: z.array(z.string()),
});

const moduleDSegmentSchema = z.object({
  segment_id: z.string(),
  big_segment_id: z.string(),
  remotion_id: z.string().optional(),
  scene_desc_zh: z.string().optional(),
  status: z.enum(["pending", "running", "done", "failed"]).optional(),
  video_url: z.string().optional().default(""),
  start_time: z.number().optional().default(0),
  end_time: z.number().optional().default(0),
  duration: z.number().optional().default(0),
  error_message: z.string().optional().default(""),
  shots: z.array(moduleDShotSchema),
  lyrics: z.array(z.string()).optional().default([]),
});

export const taskModuleDDataSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  module_d_status: z.string(),
  unit_summary: moduleDUnitSummarySchema.optional(),
  segments: z.array(moduleDSegmentSchema),
  output_video_url: z.string().optional(),
  active_rerun: moduleDActiveRerunSchema.nullable().optional(),
});

const moduleDSegmentVideoFileSchema = z.object({
  segment_id: z.string(),
  exists: z.boolean(),
  mtime: z.number(),
  size_bytes: z.number(),
  video_url: z.string(),
});

export const taskModuleDSegmentVideosSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  items: z.record(z.string(), moduleDSegmentVideoFileSchema),
});

export type TaskModuleDData = z.infer<typeof taskModuleDDataSchema>;
export type TaskModuleDSegment = z.infer<typeof moduleDSegmentSchema>;
export type TaskModuleDShot = z.infer<typeof moduleDShotSchema>;
export type TaskModuleDSegmentVideoFile = z.infer<typeof moduleDSegmentVideoFileSchema>;
