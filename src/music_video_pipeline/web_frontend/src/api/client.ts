import { z } from "zod";

import { appLogger } from "@/app/logger";

function resolveBaseOrigin(): string {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }
  return window.location.origin;
}

function resolveUrl(path: string): URL {
  if (/^https?:\/\//i.test(path)) {
    return new URL(path);
  }
  return new URL(path, `${resolveBaseOrigin()}/`);
}

export async function fetchJson<Schema extends z.ZodTypeAny>(
  path: string,
  schema: Schema,
  init?: RequestInit,
): Promise<z.infer<Schema>> {
  const url = resolveUrl(path);
  const response = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers || {}),
    },
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }

  if (!response.ok) {
    const errorText =
      payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string"
        ? payload.error
        : `请求失败：${response.status}`;
    appLogger.error("任务接口", "接口请求失败", {
      path: url.pathname,
      status: response.status,
      error: errorText,
    });
    throw new Error(errorText);
  }

  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    appLogger.error("接口校验", "接口返回数据不符合预期", {
      path: url.pathname,
      issues: parsed.error.issues,
    });
    throw new Error(`接口数据校验失败：${url.pathname}`);
  }
  return parsed.data;
}
