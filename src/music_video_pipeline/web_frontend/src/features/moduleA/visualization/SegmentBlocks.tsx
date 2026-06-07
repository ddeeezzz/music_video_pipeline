import type { SegmentItem } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { useTooltip } from "./TooltipContainer";

function getLabelClass(label: string): string {
  const normalized = label.toLowerCase();
  if (normalized.includes("inst")) return "vis-seg--inst";
  if (normalized.includes("chorus")) return "vis-seg--chorus";
  if (normalized.includes("verse")) return "vis-seg--verse";
  if (normalized.includes("bridge")) return "vis-seg--bridge";
  if (normalized.includes("start")) return "vis-seg--start";
  if (normalized.includes("end")) return "vis-seg--end";
  return "vis-seg--default";
}

function buildSegmentTooltip(item: SegmentItem): string {
  return [
    `图层: ${item.layer || "-"}`,
    `segment_id: ${item.segment_id || item.id || "-"}`,
    `window_id: ${item.window_id || "-"}`,
    `big_segment_id: ${item.big_segment_id || "-"}`,
    `label: ${item.label || "-"}`,
    `role: ${item.role || "-"}`,
    `merge_action: ${item.merge_action || "-"}`,
    `source_segment_ids: ${(item.source_segment_ids || []).join(",") || "-"}`,
    `start: ${(item.start_time || 0).toFixed(3)}s`,
    `end: ${(item.end_time || 0).toFixed(3)}s`,
    `duration: ${(item.duration || 0).toFixed(3)}s`,
  ].join("\n");
}

export function SegmentBlocks({
  segments,
  clickToSeek = true,
}: {
  segments: SegmentItem[];
  clickToSeek?: boolean;
}) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const audioRef = useVisualizationStore((s) => s.audioRef);
  const setPlayheadTime = useVisualizationStore((s) => s.setPlayheadTime);
  const duration = useVisualizationStore((s) => s.duration);
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <>
      {segments.map((item, idx) => {
        const left = (item.start_time || 0) * pxPerSec;
        const endTime = item.end_time || item.start_time || 0;
        const startTime = item.start_time || 0;
        const width = Math.max(1, (endTime - startTime) * pxPerSec);
        const segText = item.segment_id || item.id || "";
        const displayText = item.display_text || item.label || "";
        const label = segText && displayText ? `${segText} | ${displayText}` : displayText || segText;

        return (
          <div
            key={item.id || `seg-${idx}`}
            className={`vis-seg ${getLabelClass(item.label)}`}
            style={{ left, width }}
            title={label.length > 40 ? label : undefined}
            onMouseEnter={(e) => {
              showTooltip(buildSegmentTooltip(item), e.clientX, e.clientY);
            }}
            onMouseMove={(e) => {
              showTooltip(buildSegmentTooltip(item), e.clientX, e.clientY);
            }}
            onMouseLeave={hideTooltip}
            onClick={() => {
              if (!clickToSeek) return;
              const seek = Math.max(0, Math.min(duration, item.start_time || 0));
              if (audioRef) {
                audioRef.currentTime = seek;
              }
              setPlayheadTime(seek);
            }}
          >
            {label}
          </div>
        );
      })}
    </>
  );
}
