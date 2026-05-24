import { fetchJson } from "@/api/client";
import { appLogger } from "@/app/logger";
import { taskModuleADataSchema, taskModuleALyricDetailSchema, taskModuleALyricsSearchSchema } from "@/schemas/moduleA";
import { taskModuleBDataSchema } from "@/schemas/moduleB";
import {
  taskActionResponseSchema,
  taskDetailResponseSchema,
  taskListResponseSchema,
} from "@/schemas/tasks";
import { taskMonitorSnapshotSchema } from "@/schemas/monitor";
import { taskWebDataSchema } from "@/schemas/webData";

function buildQueryString(params: Record<string, string>): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    searchParams.set(key, value);
  }
  return searchParams.toString();
}

export const taskQueryKeys = {
  list: ["tasks"] as const,
  detail: (taskId: string) => ["task", taskId, "detail"] as const,
  snapshot: (taskId: string) => ["task", taskId, "snapshot"] as const,
  webData: (taskId: string) => ["task", taskId, "web-data"] as const,
  moduleA: (taskId: string) => ["task", taskId, "module-a"] as const,
  moduleB: (taskId: string) => ["task", taskId, "module-b"] as const,
};

export async function listTasks() {
  return fetchJson("/api/tasks", taskListResponseSchema);
}

export async function getTaskDetail(taskId: string) {
  return fetchJson(`/api/task?${buildQueryString({ task_id: taskId })}`, taskDetailResponseSchema);
}

export async function getTaskSnapshot(taskId: string) {
  return fetchJson(`/snapshot?${buildQueryString({ task_id: taskId })}`, taskMonitorSnapshotSchema);
}

export async function getTaskWebData(taskId: string) {
  return fetchJson(`/web-data?${buildQueryString({ task_id: taskId })}`, taskWebDataSchema);
}

export async function getTaskModuleBData(taskId: string) {
  return fetchJson(`/api/module-b?${buildQueryString({ task_id: taskId })}`, taskModuleBDataSchema);
}

export async function getTaskModuleAData(taskId: string) {
  return fetchJson(`/api/module-a?${buildQueryString({ task_id: taskId })}`, taskModuleADataSchema);
}

export async function createTask(input: {
  taskId: string;
  audioPath: string;
  configPath: string;
}) {
  appLogger.info("任务创建", "开始创建任务", { taskId: input.taskId });
  const payload = await fetchJson(
    `/api/task/create?${buildQueryString({
      task_id: input.taskId,
      audio_path: input.audioPath,
      config_path: input.configPath,
    })}`,
    taskActionResponseSchema,
  );
  appLogger.info("任务创建", "任务创建完成", { taskId: payload.task_id || input.taskId });
  return payload;
}

export async function renameTask(input: { oldTaskId: string; newTaskId: string }) {
  appLogger.info("任务详情", "开始提交任务改名", input);
  const payload = await fetchJson(
    `/api/task/rename?${buildQueryString({
      old_task_id: input.oldTaskId,
      new_task_id: input.newTaskId,
    })}`,
    taskActionResponseSchema,
  );
  appLogger.info("任务详情", "任务改名完成", { taskId: payload.task_id || input.newTaskId });
  return payload;
}

export async function copyTask(input: {
  sourceTaskId: string;
  newTaskId: string;
  audioPath: string;
  configPath: string;
}) {
  appLogger.info("任务详情", "开始复制任务", {
    sourceTaskId: input.sourceTaskId,
    newTaskId: input.newTaskId,
  });
  const payload = await fetchJson(
    `/api/task/copy?${buildQueryString({
      source_task_id: input.sourceTaskId,
      new_task_id: input.newTaskId,
      audio_path: input.audioPath,
      config_path: input.configPath,
    })}`,
    taskActionResponseSchema,
  );
  appLogger.info("任务详情", "任务复制完成", { taskId: payload.task_id || input.newTaskId });
  return payload;
}

export async function rerunTask(taskId: string) {
  appLogger.info("任务详情", "开始触发任务重跑", { taskId });
  const payload = await fetchJson(
    `/api/task/rerun?${buildQueryString({ task_id: taskId })}`,
    taskActionResponseSchema,
  );
  appLogger.info("任务详情", "任务重跑请求已提交", { taskId });
  return payload;
}

export async function searchTaskModuleALyrics(
  taskId: string,
  options?: { manualQuery?: string; manualArtist?: string; manualTitle?: string },
) {
  const manualArtist = options?.manualArtist?.trim() || "";
  const manualTitle = options?.manualTitle?.trim() || "";
  const manualQuery = options?.manualQuery?.trim()
    || (manualArtist && manualTitle ? `${manualArtist} - ${manualTitle}` : manualTitle);
  appLogger.info(
    "模块A",
    manualQuery ? "开始按歌曲名联网查找 lrc 歌词候选" : "开始自动联网查找 lrc 歌词候选",
    { taskId, manualQuery, manualArtist, manualTitle },
  );
  const query: Record<string, string> = { task_id: taskId };
  if (manualQuery) {
    query.manual_query = manualQuery;
  }
  return fetchJson(
    `/api/module-a/search-lyrics?${buildQueryString(query)}`,
    taskModuleALyricsSearchSchema,
  );
}

export function buildTaskModuleALyricsSearchSocketUrl(
  taskId: string,
  options?: { manualQuery?: string; manualArtist?: string; manualTitle?: string },
) {
  const manualArtist = options?.manualArtist?.trim() || "";
  const manualTitle = options?.manualTitle?.trim() || "";
  const manualQuery = options?.manualQuery?.trim()
    || (manualArtist && manualTitle ? `${manualArtist} - ${manualTitle}` : manualTitle);
  const configuredBaseUrl = import.meta.env.VITE_WS_BASE_URL?.trim();
  const baseUrl = configuredBaseUrl
    ? configuredBaseUrl.replace(/\/+$/, "")
    : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
  const searchParams = new URLSearchParams({ task_id: taskId });
  if (manualQuery) {
    searchParams.set("manual_query", manualQuery);
  }
  return `${baseUrl}/ws/module-a/search-lyrics?${searchParams.toString()}`;
}

export async function selectTaskModuleALyrics(
  taskId: string,
  candidateId: string,
  enable: boolean,
) {
  appLogger.info("模块A", "开始提交 lrc 歌词选择", { taskId, candidateId, enable });
  return fetchJson(
    `/api/module-a/select-lyrics?${buildQueryString({
      task_id: taskId,
      candidate_id: candidateId,
      enable: enable ? "1" : "0",
    })}`,
    taskActionResponseSchema,
  );
}

export async function getTaskModuleALyricDetail(taskId: string, candidateId: string) {
  return fetchJson(
    `/api/module-a/candidate-lyrics?${buildQueryString({
      task_id: taskId,
      candidate_id: candidateId,
    })}`,
    taskModuleALyricDetailSchema,
  );
}

export async function rerunModuleBRole(taskId: string, roleName: string, options?: { replaceRunning?: boolean }) {
  appLogger.info("模块B", "开始触发模块 B role 重跑", {
    taskId,
    roleName,
    replaceRunning: Boolean(options?.replaceRunning),
  });
  return fetchJson(
    `/api/module-b/rerun-role?${buildQueryString({
      task_id: taskId,
      role_name: roleName,
      replace_running: options?.replaceRunning ? "1" : "0",
    })}`,
    taskActionResponseSchema,
  );
}

export async function rerunModuleBRoleSegment(
  taskId: string,
  roleName: string,
  segmentId: string,
  options?: { replaceRunning?: boolean },
) {
  appLogger.info("模块B", "开始触发模块 B segment 重跑", {
    taskId,
    roleName,
    segmentId,
    replaceRunning: Boolean(options?.replaceRunning),
  });
  return fetchJson(
    `/api/module-b/rerun-role-segment?${buildQueryString({
      task_id: taskId,
      role_name: roleName,
      segment_id: segmentId,
      replace_running: options?.replaceRunning ? "1" : "0",
    })}`,
    taskActionResponseSchema,
  );
}
