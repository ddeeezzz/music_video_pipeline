import { useEffect, useRef } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

export function PlayheadLine() {
  const playheadTime = useVisualizationStore((s) => s.playheadTime);
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const ref = useRef<HTMLDivElement>(null);

  const left = playheadTime * pxPerSec;

  useEffect(() => {
    if (ref.current) {
      ref.current.style.left = `${left}px`;
    }
  }, [left]);

  return <div ref={ref} className="vis-playhead" />;
}
