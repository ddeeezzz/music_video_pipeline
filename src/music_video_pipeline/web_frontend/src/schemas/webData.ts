import { z } from "zod";

const mediaAssetSchema = z.object({
  available: z.boolean(),
  url: z.string(),
  path: z.string(),
});

const lyricTokenSchema = z.object({
  text: z.string(),
  start_time: z.number(),
  end_time: z.number(),
});

const lyricUnitSchema = z.object({
  segment_id: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  text: z.string(),
  confidence: z.number(),
  token_units: z.array(lyricTokenSchema),
});

const segmentUnitSchema = z.object({
  segment_id: z.string(),
  big_segment_id: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  label: z.string(),
  role: z.string(),
  scene_desc: z.string(),
  shot_id: z.string(),
  camera_plan: z.record(z.string(), z.unknown()),
  keyframe_prompt_start_zh: z.string(),
  keyframe_prompt_start_en: z.string(),
  keyframe_prompt_end_zh: z.string(),
  keyframe_prompt_end_en: z.string(),
  video_prompt_zh: z.string(),
  video_prompt_en: z.string(),
  frame_path_start: z.string(),
  frame_path_end: z.string(),
  frame_url_start: z.string(),
  frame_url_end: z.string(),
});

export const taskWebDataSchema = z.object({
  task_id: z.string(),
  task_status: z.string(),
  video: mediaAssetSchema,
  module_a_visualization: mediaAssetSchema,
  lyric_units: z.array(lyricUnitSchema),
  segment_units: z.array(segmentUnitSchema),
});

export type TaskWebData = z.infer<typeof taskWebDataSchema>;
export type TaskSegmentUnit = z.infer<typeof segmentUnitSchema>;
