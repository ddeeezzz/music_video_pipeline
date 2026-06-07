import type { LyricItem } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { useTooltip } from "./TooltipContainer";

function buildLyricTooltip(item: LyricItem): string {
  return [
    "图层: Lyrics",
    `segment_id: ${item.segment_id || item.id || "-"}`,
    `text: ${item.text || "-"}`,
    `confidence: ${(item.confidence || 0).toFixed(3)}`,
    `start: ${(item.start_time || 0).toFixed(3)}s`,
    `end: ${(item.end_time || 0).toFixed(3)}s`,
    `duration: ${(item.duration || 0).toFixed(3)}s`,
  ].join("\n");
}

export function LyricBlocks({
  lyrics,
  clickToSeek = true,
}: {
  lyrics: LyricItem[];
  clickToSeek?: boolean;
}) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const audioRef = useVisualizationStore((s) => s.audioRef);
  const setPlayheadTime = useVisualizationStore((s) => s.setPlayheadTime);
  const duration = useVisualizationStore((s) => s.duration);
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <>
      {lyrics.map((item, idx) => {
        const left = (item.start_time || 0) * pxPerSec;
        const endTime = item.end_time || item.start_time || 0;
        const width = Math.max(20, (endTime - (item.start_time || 0)) * pxPerSec);
        const label = item.display_text || item.text || "";

        return (
          <div
            key={item.id || `lyric-${idx}`}
            className="vis-lyric"
            style={{ left, width }}
            onMouseEnter={(e) => {
              showTooltip(buildLyricTooltip(item), e.clientX, e.clientY);
            }}
            onMouseMove={(e) => {
              showTooltip(buildLyricTooltip(item), e.clientX, e.clientY);
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
