import { z } from "zod";

const mediaAssetSchema = z.object({
  available: z.boolean(),
  url: z.string(),
  path: z.string(),
});

const moduleALyricCandidateSchema = z.object({
  candidate_id: z.string(),
  artist: z.string(),
  title: z.string(),
  provider: z.string(),
  provider_id: z.string(),
  provider_song_id: z.string().optional(),
  duration_seconds: z.number().optional(),
  has_word_timed_lyrics: z.boolean().optional(),
  has_translated_lyrics: z.boolean().optional(),
  has_romanized_lyrics: z.boolean().optional(),
  synced_lyrics: z.string().optional(),
  word_timed_lyrics: z.string().optional(),
  preview_lines: z.array(z.string()),
  preview_text: z.string(),
});

const moduleALyricProviderGroupSchema = z.object({
  provider: z.string(),
  display_name: z.string(),
  candidates: z.array(moduleALyricCandidateSchema),
  first_result_at_ms: z.number().nullable().optional(),
  page_size: z.number().optional(),
  total_count: z.number().optional(),
  has_more: z.boolean().optional(),
});

const moduleAMetadataTraceSchema = z.object({
  embedded_status: z.string(),
  embedded_source: z.string(),
  embedded_artist: z.string(),
  embedded_title: z.string(),
  embedded_album: z.string(),
  embedded_error: z.string(),
  fingerprint_status: z.string(),
  fingerprint_error: z.string(),
  acoustid_status: z.string(),
  matched_artist: z.string(),
  matched_title: z.string(),
  matched_score: z.number(),
  matched_error: z.string(),
});

const moduleANetworkLrcStateSchema = z.object({
  display_status: z.string(),
  enabled: z.boolean(),
  updated_at: z.string(),
  last_search_at: z.string(),
  search_status: z.string(),
  lookup_error: z.string(),
  cached_candidates_count: z.number(),
  metadata_trace: moduleAMetadataTraceSchema,
  provider_groups: z.array(moduleALyricProviderGroupSchema),
  selected_candidate: moduleALyricCandidateSchema,
});

export const taskModuleADataSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  task_status: z.string(),
  duration_seconds: z.number().nullable().optional(),
  save_lyrics_port: z.number().int().optional(),
  module_a_status: z.string(),
  module_a_visualization: mediaAssetSchema,
  network_lrc_state: moduleANetworkLrcStateSchema,
});

export const taskModuleALyricsSearchSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  search_status: z.string(),
  search_mode: z.string().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
  suggest_manual_query: z.boolean().optional(),
  metadata_trace: moduleAMetadataTraceSchema,
  provider_groups: z.array(moduleALyricProviderGroupSchema).optional(),
  candidates: z.array(moduleALyricCandidateSchema),
});

export const taskModuleALyricDetailSchema = z.object({
  ok: z.boolean(),
  task_id: z.string(),
  candidate: moduleALyricCandidateSchema,
  synced_lyrics: z.string(),
  word_timed_lyrics: z.string(),
  translated_lyrics: z.string(),
  romanized_lyrics: z.string(),
});

export type TaskModuleAData = z.infer<typeof taskModuleADataSchema>;
export type TaskModuleALyricCandidate = z.infer<typeof moduleALyricCandidateSchema>;
export type TaskModuleALyricProviderGroup = z.infer<typeof moduleALyricProviderGroupSchema>;
export type TaskModuleAMetadataTrace = z.infer<typeof moduleAMetadataTraceSchema>;
export type TaskModuleALyricDetail = z.infer<typeof taskModuleALyricDetailSchema>;
