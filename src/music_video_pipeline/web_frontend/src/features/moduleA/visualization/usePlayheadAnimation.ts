import { useEffect, useRef } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

export function usePlayheadAnimation() {
  const audioRef = useVisualizationStore((s) => s.audioRef);
  const setPlayheadTime = useVisualizationStore((s) => s.setPlayheadTime);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const el = audioRef;
    if (!el) return;

    const animate = () => {
      setPlayheadTime(el.currentTime);
      if (!el.paused && !el.ended) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };

    const onPlay = () => {
      rafRef.current = requestAnimationFrame(animate);
    };

    const onPause = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setPlayheadTime(el.currentTime);
    };

    const onSeeked = () => {
      setPlayheadTime(el.currentTime);
    };

    const onEnded = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setPlayheadTime(el.currentTime);
    };

    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("seeked", onSeeked);
    el.addEventListener("ended", onEnded);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("seeked", onSeeked);
      el.removeEventListener("ended", onEnded);
    };
  }, [audioRef, setPlayheadTime]);
}
