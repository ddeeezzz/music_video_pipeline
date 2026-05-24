import { Tag } from "antd";

type TaskStatusTagProps = {
  status: string;
};

const STATUS_COLOR_MAP: Record<string, string> = {
  pending: "default",
  running: "processing",
  done: "success",
  failed: "error",
  not_found: "default",
  unknown: "default",
};

export function TaskStatusTag({ status }: TaskStatusTagProps) {
  const normalizedStatus = String(status || "unknown").trim().toLowerCase() || "unknown";
  return <Tag color={STATUS_COLOR_MAP[normalizedStatus] || "default"}>{normalizedStatus}</Tag>;
}
