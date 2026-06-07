import { Alert, Empty, Skeleton } from "antd";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { taskQueryKeys, getModuleAVisualizationPayload } from "@/api/taskApi";
import { useVisualizationStore, type LayerKey } from "@/stores/visualizationStore";
import { SummaryCards } from "./SummaryCards";
import { AudioPlayerShell } from "./AudioPlayerShell";
import { ZoomSlider } from "./ZoomSlider";
import { LayerToggles } from "./LayerToggles";
import { TimelineShell } from "./TimelineShell";
import { TimelineAxis } from "./TimelineAxis";
import { TimelineLane } from "./TimelineLane";
import { SegmentBlocks } from "./SegmentBlocks";
import { BeatLines } from "./BeatLines";
import { OnsetLines } from "./OnsetLines";
import { LyricBlocks } from "./LyricBlocks";
import { EnergyBlocks } from "./EnergyBlocks";
import { PrecheckRmsLane } from "./PrecheckRmsLane";
import { AccompanimentRmsLane } from "./AccompanimentRmsLane";
import { PlayheadLine } from "./PlayheadLine";
import { TooltipProvider } from "./TooltipContainer";
import { usePlayheadAnimation } from "./usePlayheadAnimation";
import type { ModuleAVisualizationPayload } from "@/schemas/moduleAVisualization";
import "@/styles/visualization.css";

interface LayerDef {
  key: LayerKey;
  title: string;
  type: "segments" | "beats" | "lyrics" | "lyrics_attached" | "energy" | "onset" | "precheck" | "accompaniment_rms";
}

const LAYER_DEFS: LayerDef[] = [
  { key: "a0", title: "A0段 (stage_big_a0)", type: "segments" },
  { key: "al", title: "A1段 (stage_big_a1)", type: "segments" },
  { key: "b", title: "B段 (module_a_output.big_segments)", type: "segments" },
  { key: "s", title: "S段 (stage_segments_final)", type: "segments" },
  { key: "role", title: "窗口角色 (stage_windows_classified)", type: "segments" },
  { key: "beats", title: "Beats（最终节拍）", type: "beats" },
  { key: "lyrics", title: "Lyrics（全量分句）", type: "lyrics" },
  { key: "lyrics_attached", title: "Lyrics（挂载版）", type: "lyrics_attached" },
  { key: "energy", title: "Energy（能量特征）", type: "energy" },
  { key: "onset", title: "Onset（伴奏候选+能量）", type: "onset" },
  { key: "precheck", title: "人声RMS预检", type: "precheck" },
  { key: "accompaniment_rms", title: "伴奏RMS（no_vocals）", type: "accompaniment_rms" },
];

function renderLaneContent(def: LayerDef, payload: ModuleAVisualizationPayload) {
  switch (def.type) {
    case "segments": {
      let data = payload.a0_segments;
      if (def.key === "al") data = payload.al_segments;
      else if (def.key === "b") data = payload.b_segments;
      else if (def.key === "s") data = payload.s_segments;
      else if (def.key === "role") data = payload.content_roles;
      return <SegmentBlocks segments={data} />;
    }
    case "beats":
      return <BeatLines beats={payload.beats} />;
    case "lyrics":
      return <LyricBlocks lyrics={payload.lyric_units} />;
    case "lyrics_attached":
      return <LyricBlocks lyrics={payload.lyric_units_attached} />;
    case "energy":
      return <EnergyBlocks energies={payload.energy_features} />;
    case "onset":
      return <OnsetLines onsets={payload.onset_points} />;
    case "precheck":
      return <PrecheckRmsLane data={payload.vocal_precheck_rms} />;
    case "accompaniment_rms":
      return <AccompanimentRmsLane data={payload.accompaniment_rms} />;
  }
}

function VisualizationContent({ payload }: { payload: ModuleAVisualizationPayload }) {
  const layerVisibility = useVisualizationStore((s) => s.layerVisibility);
  const setDuration = useVisualizationStore((s) => s.setDuration);

  usePlayheadAnimation();

  useEffect(() => {
    setDuration(payload.duration_seconds || 0);
  }, [payload.duration_seconds, setDuration]);

  return (
    <div className="vis-shell">
      <SummaryCards payload={payload} />

      <div className="vis-controls">
        <div className="vis-controls__row">
          <AudioPlayerShell audioUrl={payload.audio_url} available={payload.audio_available} />
          <ZoomSlider />
        </div>
        <LayerToggles />
      </div>

      <TimelineShell>
        <TimelineAxis />
        {LAYER_DEFS.map((def) => (
          <TimelineLane
            key={def.key}
            label={def.title}
            visible={layerVisibility[def.key]}
            tall={def.type === "precheck" || def.type === "accompaniment_rms"}
          >
            {renderLaneContent(def, payload)}
          </TimelineLane>
        ))}
        <PlayheadLine />
      </TimelineShell>
    </div>
  );
}

export function ModuleAVisualization({ taskId }: { taskId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: taskQueryKeys.moduleAVisualization(taskId),
    queryFn: () => getModuleAVisualizationPayload(taskId),
    enabled: Boolean(taskId),
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <div style={{ padding: 24 }}>
        <Skeleton active paragraph={{ rows: 8 }} />
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        type="error"
        message="可视化数据加载失败"
        description={String(error)}
        showIcon
        style={{ margin: 16 }}
      />
    );
  }

  if (!data) {
    return <Empty description="暂无可视化数据" />;
  }

  return (
    <TooltipProvider>
      <VisualizationContent payload={data} />
    </TooltipProvider>
  );
}
