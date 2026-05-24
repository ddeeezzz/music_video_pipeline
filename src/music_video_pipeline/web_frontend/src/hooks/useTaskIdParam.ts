import { useParams } from "react-router-dom";

export function useTaskIdParam(): string {
  const params = useParams();
  return String(params.taskId || "").trim();
}
