import { z } from "zod";

// 段落方块（A0/A1/B/S/ROLE 通用）
const segmentItemSchema = z.object({
  id: z.string(),
  segment_id: z.string(),
  window_id: z.string(),
  big_segment_id: z.string(),
  label: z.string(),
  role: z.string(),
  display_text: z.string(),
  merge_action: z.string(),
  source_segment_ids: z.array(z.string()),
  start_time: z.number(),
  end_time: z.number(),
  duration: z.number(),
  layer: z.string(),
});

// 节拍
const beatItemSchema = z.object({
  id: z.string(),
  time: z.number(),
  type: z.string(),
  source: z.string(),
});

// 歌词
const lyricItemSchema = z.object({
  id: z.string(),
  segment_id: z.string(),
  text: z.string(),
  display_text: z.string(),
  confidence: z.number(),
  start_time: z.number(),
  end_time: z.number(),
  duration: z.number(),
});

// 能量特征
const energyItemSchema = z.object({
  id: z.string(),
  start_time: z.number(),
  end_time: z.number(),
  duration: z.number(),
  energy_level: z.string(),
  trend: z.string(),
  rhythm_tension: z.number(),
});

// Onset 点
const onsetPointItemSchema = z.object({
  id: z.string(),
  time: z.number(),
  energy_raw: z.number(),
  energy_norm: z.number(),
});

// RMS 折线
const rmsSeriesSchema = z.object({
  times: z.array(z.number()),
  values: z.array(z.number()),
});

// 人声预检 RMS
const vocalPrecheckRmsSchema = rmsSeriesSchema.extend({
  should_skip_funasr: z.boolean(),
  peak_rms: z.number(),
  active_ratio: z.number(),
  peak_threshold: z.number(),
  active_ratio_threshold: z.number(),
  sample_source: z.string(),
  sample_count_raw: z.number(),
  sample_count_kept: z.number(),
  sample_count_outlier: z.number(),
  dynamic_gap_threshold_seconds: z.number(),
});

// A0→A1 边界调整统计
const boundaryShiftStatsSchema = z.object({
  compared_count: z.number(),
  adjusted_count: z.number(),
  adjusted_ratio: z.number(),
  average_abs_shift_seconds: z.number(),
  max_abs_shift_seconds: z.number(),
});

// 摘要统计
const visualizationSummarySchema = z.object({
  a0_count: z.number(),
  al_count: z.number(),
  b_count: z.number(),
  s_count: z.number(),
  beat_count: z.number(),
  lyric_count: z.number(),
  lyric_attached_count: z.number(),
  energy_count: z.number(),
  boundary_shift: boundaryShiftStatsSchema,
});

// Onset candidates（仅时间列表）
const onsetCandidatesSchema = z.array(z.number());

// 完整可视化 payload
export const moduleAVisualizationPayloadSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  task_dir: z.string(),
  audio_path: z.string(),
  audio_url: z.string(),
  audio_available: z.boolean(),
  module_a_output_path: z.string(),
  duration_seconds: z.number(),
  a0_segments: z.array(segmentItemSchema),
  al_segments: z.array(segmentItemSchema),
  b_segments: z.array(segmentItemSchema),
  s_segments: z.array(segmentItemSchema),
  content_roles: z.array(segmentItemSchema),
  beats: z.array(beatItemSchema),
  lyric_units: z.array(lyricItemSchema),
  lyric_units_attached: z.array(lyricItemSchema),
  energy_features: z.array(energyItemSchema),
  onset_candidates: onsetCandidatesSchema,
  onset_points: z.array(onsetPointItemSchema),
  vocal_precheck_rms: vocalPrecheckRmsSchema,
  accompaniment_rms: rmsSeriesSchema,
  summary: visualizationSummarySchema,
});

// 导出类型
export type ModuleAVisualizationPayload = z.infer<typeof moduleAVisualizationPayloadSchema>;
export type SegmentItem = z.infer<typeof segmentItemSchema>;
export type BeatItem = z.infer<typeof beatItemSchema>;
export type LyricItem = z.infer<typeof lyricItemSchema>;
export type EnergyItem = z.infer<typeof energyItemSchema>;
export type OnsetPointItem = z.infer<typeof onsetPointItemSchema>;
export type RmsSeries = z.infer<typeof rmsSeriesSchema>;
export type VocalPrecheckRms = z.infer<typeof vocalPrecheckRmsSchema>;
export type VisualizationSummary = z.infer<typeof visualizationSummarySchema>;
export type BoundaryShiftStats = z.infer<typeof boundaryShiftStatsSchema>;
