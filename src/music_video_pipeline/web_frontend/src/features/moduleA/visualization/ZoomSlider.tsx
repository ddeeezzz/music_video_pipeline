import { Slider } from "antd";
import { useVisualizationStore } from "@/stores/visualizationStore";

export function ZoomSlider() {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);
  const setPxPerSec = useVisualizationStore((s) => s.setPxPerSec);

  return (
    <div className="vis-controls__zoom">
      <div className="vis-controls__zoom-label">时间轴缩放：{pxPerSec} px/s</div>
      <Slider
        min={20}
        max={320}
        step={10}
        value={pxPerSec}
        onChange={setPxPerSec}
        tooltip={{ formatter: (v) => `${v} px/s` }}
      />
    </div>
  );
}
