import { useMemo } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

export function TimelineAxis() {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const duration = useVisualizationStore((s) => s.duration);

  const ticks = useMemo(() => {
    const result: { x: number; major: boolean; label?: string }[] = [];
    const majorStep = duration <= 120 ? 5 : 10;
    const minorStep = majorStep / 2;

    for (let t = 0; t <= duration + 1e-6; t += minorStep) {
      const x = t * pxPerSec;
      const isMajor = Math.abs(t / majorStep - Math.round(t / majorStep)) < 1e-6;
      result.push({
        x,
        major: isMajor,
        label: isMajor ? `${Math.round(t)}s` : undefined,
      });
    }
    return result;
  }, [duration, pxPerSec]);

  const totalWidth = duration * pxPerSec;

  return (
    <div className="vis-axis" style={{ width: totalWidth }}>
      {ticks.map((tick, idx) => (
        <div
          key={idx}
          className={`vis-axis__tick ${tick.major ? "vis-axis__tick--major" : "vis-axis__tick--minor"}`}
          style={{ left: tick.x }}
        />
      ))}
      {ticks
        .filter((t) => t.label)
        .map((tick, idx) => (
          <div key={`lbl-${idx}`} className="vis-axis__label" style={{ left: tick.x }}>
            {tick.label}
          </div>
        ))}
    </div>
  );
}
