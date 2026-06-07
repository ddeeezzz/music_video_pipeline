import type { VocalPrecheckRms } from "@/schemas/moduleAVisualization";
import { useVisualizationStore } from "@/stores/visualizationStore";
import { RmsPolyline } from "./RmsPolyline";
import { useTooltip } from "./TooltipContainer";
import { useMemo } from "react";

function buildPrecheckTooltip(precheck: VocalPrecheckRms): string {
  return [
    "图层: 人声RMS预检",
    `should_skip_funasr: ${precheck.should_skip_funasr}`,
    `peak_rms: ${(precheck.peak_rms || 0).toFixed(6)}`,
    `active_ratio: ${(precheck.active_ratio || 0).toFixed(6)}`,
    `peak_threshold: ${(precheck.peak_threshold || 0).toFixed(6)}`,
    `active_ratio_threshold: ${(precheck.active_ratio_threshold || 0).toFixed(6)}`,
    `sample_source: ${precheck.sample_source || "-"}`,
    `sample_count: ${precheck.sample_count_kept || 0} / ${precheck.sample_count_raw || 0}`,
    `dynamic_gap_threshold: ${(precheck.dynamic_gap_threshold_seconds || 0).toFixed(3)}s`,
  ].join("\n");
}

export function PrecheckRmsLane({ data }: { data: VocalPrecheckRms | null }) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const duration = useVisualizationStore((s) => s.duration);
  const { showTooltip, hideTooltip } = useTooltip();

  if (!data || !data.times || data.times.length === 0) {
    return <div style={{ padding: 4, fontSize: 11, color: "#8c959f" }}>无预检数据</div>;
  }

  const maxVal = useMemo(() => {
    let m = 0;
    for (const v of data.values) if (v > m) m = v;
    return m || 1;
  }, [data.values]);

  const thresholdPx = 64 * ((data.peak_threshold || 0) / maxVal);
  const totalWidth = duration * pxPerSec;

  return (
    <>
      <RmsPolyline
        times={data.times}
        values={data.values}
        trackHeight={64}
        color="rgba(45, 110, 180, 0.5)"
        className="vis-rms-line--precheck"
      />
      {data.peak_threshold != null && data.peak_threshold > 0 && (
        <div
          className="vis-threshold-line"
          style={{
            bottom: thresholdPx,
            width: totalWidth,
          }}
          onMouseEnter={(e) => {
            showTooltip(buildPrecheckTooltip(data), e.clientX, e.clientY);
          }}
          onMouseMove={(e) => {
            showTooltip(buildPrecheckTooltip(data), e.clientX, e.clientY);
          }}
          onMouseLeave={hideTooltip}
        />
      )}
    </>
  );
}
