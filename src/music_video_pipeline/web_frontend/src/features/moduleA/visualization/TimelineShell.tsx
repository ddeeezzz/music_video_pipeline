import { useVisualizationStore } from "@/stores/visualizationStore";
import { useCallback, useRef } from "react";

const LANE_LABEL_WIDTH = 132;

export function TimelineShell({ children }: { children: React.ReactNode }) {
  const duration = useVisualizationStore((s) => s.duration);
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const setPlayheadTime = useVisualizationStore((s) => s.setPlayheadTime);
  const audioRef = useVisualizationStore((s) => s.audioRef);
  const scrollRef = useRef<HTMLDivElement>(null);

  const totalWidth = duration * pxPerSec;

  const handleTimelineClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement;
      if (target.closest(".vis-seg, .vis-lyric, .vis-energy, .vis-beat, .vis-onset, .vis-rms-line")) {
        return;
      }
      const rect = e.currentTarget.getBoundingClientRect();
      const scrollLeft = scrollRef.current?.scrollLeft ?? 0;
      // 减去标签列宽，使点击位置与 track 内元素坐标对齐
      const x = Math.max(0, e.clientX - rect.left + scrollLeft - LANE_LABEL_WIDTH);
      const seek = Math.max(0, Math.min(duration, x / pxPerSec));
      if (audioRef) {
        audioRef.currentTime = seek;
      }
      setPlayheadTime(seek);
    },
    [duration, pxPerSec, audioRef, setPlayheadTime],
  );

  return (
    <div className="vis-timeline-wrap">
      <div className="vis-timeline-scroll" ref={scrollRef}>
        <div
          className="vis-timeline-inner"
          style={{ width: totalWidth }}
          onClick={handleTimelineClick}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
