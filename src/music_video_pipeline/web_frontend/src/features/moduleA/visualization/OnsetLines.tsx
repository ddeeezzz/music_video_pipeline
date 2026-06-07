import type { OnsetPointItem } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { useTooltip } from "./TooltipContainer";

function buildOnsetTooltip(item: OnsetPointItem, idx: number): string {
  return [
    "图层: Onset",
    `index: ${idx}`,
    `time: ${(item.time || 0).toFixed(3)}s`,
    `energy_raw: ${(item.energy_raw || 0).toFixed(4)}`,
    `energy_norm: ${(item.energy_norm || 0).toFixed(4)}`,
  ].join("\n");
}

export function OnsetLines({ onsets }: { onsets: OnsetPointItem[] }) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <>
      {onsets.map((item, idx) => {
        const left = (item.time || 0) * pxPerSec;
        const energyNorm = Math.max(0, Math.min(1, item.energy_norm || 0));
        const alpha = 0.15 + 0.8 * energyNorm;

        return (
          <div
            key={item.id || `onset-${idx}`}
            className="vis-onset"
            style={{
              left,
              backgroundColor: `rgba(179, 63, 98, ${alpha.toFixed(3)})`,
            }}
            onMouseEnter={(e) => {
              showTooltip(buildOnsetTooltip(item, idx), e.clientX, e.clientY);
            }}
            onMouseMove={(e) => {
              showTooltip(buildOnsetTooltip(item, idx), e.clientX, e.clientY);
            }}
            onMouseLeave={hideTooltip}
          />
        );
      })}
    </>
  );
}
