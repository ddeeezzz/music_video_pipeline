import type { BeatItem } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { useTooltip } from "./TooltipContainer";

function buildBeatTooltip(beat: BeatItem): string {
  return [
    "图层: Beats",
    `time: ${(beat.time || 0).toFixed(3)}s`,
    `type: ${beat.type || "-"}`,
    `source: ${beat.source || "-"}`,
  ].join("\n");
}

export function BeatLines({ beats }: { beats: BeatItem[] }) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <>
      {beats.map((beat, idx) => {
        const left = (beat.time || 0) * pxPerSec;
        const isMajor = (beat.type || "").toLowerCase() === "major";

        return (
          <div
            key={beat.id || `beat-${idx}`}
            className={`vis-beat ${isMajor ? "vis-beat--major" : "vis-beat--minor"}`}
            style={{ left }}
            onMouseEnter={(e) => {
              showTooltip(buildBeatTooltip(beat), e.clientX, e.clientY);
            }}
            onMouseMove={(e) => {
              showTooltip(buildBeatTooltip(beat), e.clientX, e.clientY);
            }}
            onMouseLeave={hideTooltip}
          />
        );
      })}
    </>
  );
}
