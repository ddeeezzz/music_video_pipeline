function encodeTaskId(taskId: string): string {
  return encodeURIComponent(String(taskId).trim());
}

export const routes = {
  taskList: "/tasks",
  taskCreate: "/tasks/create",
  taskDetail(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}`;
  },
  taskMonitor(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/monitor`;
  },
  taskReview(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/review`;
  },
  taskModuleA(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/module-a`;
  },
  taskModuleB(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/module-b`;
  },
  taskModuleC(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/module-c`;
  },
  taskModuleD(taskId: string): string {
    return `/tasks/${encodeTaskId(taskId)}/module-d`;
  },
};
