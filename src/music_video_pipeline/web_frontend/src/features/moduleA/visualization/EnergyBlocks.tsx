import type { EnergyItem } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { useTooltip } from "./TooltipContainer";

function buildEnergyTooltip(item: EnergyItem): string {
  return [
    "图层: Energy",
    `start: ${(item.start_time || 0).toFixed(3)}s`,
    `end: ${(item.end_time || 0).toFixed(3)}s`,
    `duration: ${(item.duration || 0).toFixed(3)}s`,
    `energy_level: ${item.energy_level}`,
    `trend: ${item.trend}`,
    `rhythm_tension: ${(item.rhythm_tension || 0).toFixed(3)}`,
  ].join("\n");
}

function getEnergyClass(level: string): string {
  switch (level) {
    case "high":
      return "vis-energy--high";
    case "mid":
      return "vis-energy--mid";
    case "low":
    default:
      return "vis-energy--low";
  }
}

export function EnergyBlocks({ energies }: { energies: EnergyItem[] }) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <>
      {energies.map((item, idx) => {
        const left = (item.start_time || 0) * pxPerSec;
        const endTime = item.end_time || item.start_time || 0;
        const width = Math.max(1, (endTime - (item.start_time || 0)) * pxPerSec);
        const label = `${item.energy_level} | ${item.trend}`;

        return (
          <div
            key={item.id || `energy-${idx}`}
            className={`vis-energy ${getEnergyClass(item.energy_level)}`}
            style={{ left, width }}
            onMouseEnter={(e) => {
              showTooltip(buildEnergyTooltip(item), e.clientX, e.clientY);
            }}
            onMouseMove={(e) => {
              showTooltip(buildEnergyTooltip(item), e.clientX, e.clientY);
            }}
            onMouseLeave={hideTooltip}
          >
            {label}
          </div>
        );
      })}
    </>
  );
}
