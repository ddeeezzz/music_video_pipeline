import { useEffect, useRef } from "react";
import { useVisualizationStore } from "@/stores/visualizationStore";

export function AudioPlayerShell({ audioUrl, available }: { audioUrl: string; available: boolean }) {
  const setAudioRef = useVisualizationStore((s) => s.setAudioRef);
  const setDuration = useVisualizationStore((s) => s.setDuration);
  const setPlayheadTime = useVisualizationStore((s) => s.setPlayheadTime);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const el = audioRef.current;
    if (el) {
      setAudioRef(el);
      const onLoaded = () => {
        if (el.duration && isFinite(el.duration)) {
          setDuration(el.duration);
        }
      };
      const onTimeUpdate = () => {
        if (el.paused) {
          setPlayheadTime(el.currentTime);
        }
      };
      el.addEventListener("loadedmetadata", onLoaded);
      el.addEventListener("timeupdate", onTimeUpdate);
      return () => {
        el.removeEventListener("loadedmetadata", onLoaded);
        el.removeEventListener("timeupdate", onTimeUpdate);
        setAudioRef(null);
      };
    }
  }, [setAudioRef, setDuration, setPlayheadTime]);

  if (!available) {
    return <div className="vis-no-audio">音频不可用，仅可查看时间轴。</div>;
  }

  return (
    <div className="vis-controls__audio">
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio ref={audioRef} src={audioUrl} controls preload="auto" />
    </div>
  );
}
