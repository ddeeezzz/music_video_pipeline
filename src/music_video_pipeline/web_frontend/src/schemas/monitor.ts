import { z } from "zod";

export const taskMonitorModuleOverviewSchema = z.object({
  status: z.string(),
  progress: z.number(),
  done: z.number(),
  total: z.number(),
  error_message: z.string(),
});

export const taskMonitorChainRowSchema = z.object({
  unit_index: z.number(),
  segment_id: z.string(),
  shot_id: z.string(),
  b_status: z.string(),
  c_status: z.string(),
  d_status: z.string(),
  chain_status: z.string(),
  b_error_message: z.string(),
  c_error_message: z.string(),
  d_error_message: z.string(),
});

export const taskMonitorSnapshotSchema = z.object({
  task_id: z.string(),
  task_status: z.string(),
  updated_at: z.string(),
  module_overview: z.record(z.string(), taskMonitorModuleOverviewSchema),
  bcd_chains: z.array(taskMonitorChainRowSchema),
  chain_counts: z.record(z.string(), z.number()),
  output_video_path: z.string(),
});

export type TaskMonitorChainRow = z.infer<typeof taskMonitorChainRowSchema>;
