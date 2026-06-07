import type { RmsSeries } from "@/schemas/moduleAVisualization";
import { RmsPolyline } from "./RmsPolyline";

export function AccompanimentRmsLane({ data }: { data: RmsSeries | null }) {
  if (!data || !data.times || data.times.length === 0) {
    return <div style={{ padding: 4, fontSize: 11, color: "#8c959f" }}>无伴奏RMS数据</div>;
  }

  return <RmsPolyline times={data.times} values={data.values} trackHeight={48} />;
}
