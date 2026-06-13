import { useEffect, useRef } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

const LANE_LABEL_WIDTH = 132;

export function PlayheadLine() {
  const playheadTime = useVisualizationStore((s) => s.playheadTime);
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const ref = useRef<HTMLDivElement>(null);

  // 加上标签列宽，与 track 内元素的坐标系对齐
  const left = LANE_LABEL_WIDTH + playheadTime * pxPerSec;

  useEffect(() => {
    if (ref.current) {
      ref.current.style.left = `${left}px`;
    }
  }, [left]);

  return <div ref={ref} className="vis-playhead" />;
}
