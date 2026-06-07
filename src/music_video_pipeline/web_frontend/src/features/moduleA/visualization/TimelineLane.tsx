import type { ReactNode } from "react";

interface TimelineLaneProps {
  label: string;
  visible: boolean;
  tall?: boolean;
  children: ReactNode;
}

export function TimelineLane({ label, visible, tall, children }: TimelineLaneProps) {
  return (
    <div className={`vis-lane ${!visible ? "vis-lane--hidden" : ""} ${tall ? "vis-lane--tall" : ""}`}>
      <div className="vis-lane__label">{label}</div>
      <div className="vis-lane__track">{children}</div>
    </div>
  );
}
