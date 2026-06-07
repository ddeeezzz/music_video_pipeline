import { useMemo } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

interface RmsPolylineProps {
  times: number[];
  values: number[];
  /** 轨道高度（像素），用于线段长度计算 */
  trackHeight?: number;
  /** CSS 颜色 */
  color?: string;
  /** 额外的 CSS 类名 */
  className?: string;
}

/**
 * 用 CSS transform: rotate() 的 div 线段渲染 RMS 折线。
 * 与现有 HTML 可视化方案中的做法一致——每个相邻采样点
 * 之间生成一条绝对定位 + 旋转的细线，不使用 Canvas。
 */
export function RmsPolyline({
  times,
  values,
  trackHeight = 64,
  color = "rgba(179, 63, 98, 0.55)",
  className,
}: RmsPolylineProps) {
  const pxPerSec = useVisualizationStore((s) => s.pxPerSec);

  const maxVal = useMemo(() => {
    let m = 0;
    for (const v of values) {
      if (v > m) m = v;
    }
    return m || 1;
  }, [values]);

  const segments = useMemo(() => {
    const result: { left: number; width: number; angle: number; bottom: number }[] = [];
    for (let i = 0; i < times.length - 1; i++) {
      const x1 = (times[i] || 0) * pxPerSec;
      const y1 = ((values[i] || 0) / maxVal) * trackHeight;
      const x2 = (times[i + 1] || 0) * pxPerSec;
      const y2 = ((values[i + 1] || 0) / maxVal) * trackHeight;

      const dx = x2 - x1;
      const dy = y2 - y1;
      const length = Math.sqrt(dx * dx + dy * dy);
      if (length < 0.5) continue;

      const angle = (-Math.atan2(dy, dx) * 180) / Math.PI;
      result.push({
        left: x1,
        bottom: y1,
        width: length,
        angle,
      });
    }
    return result;
  }, [times, values, maxVal, trackHeight, pxPerSec]);

  return (
    <>
      {segments.map((seg, i) => (
        <div
          key={i}
          className={`vis-rms-line ${className ?? ""}`}
          style={{
            left: seg.left,
            bottom: seg.bottom,
            width: seg.width,
            transform: `rotate(${seg.angle.toFixed(2)}deg)`,
            backgroundColor: color,
          }}
        />
      ))}
    </>
  );
}
