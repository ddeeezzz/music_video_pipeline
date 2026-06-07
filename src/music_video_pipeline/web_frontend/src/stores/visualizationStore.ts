import { create } from "zustand";

export type LayerKey =
  | "a0"
  | "al"
  | "b"
  | "s"
  | "role"
  | "beats"
  | "lyrics"
  | "lyrics_attached"
  | "energy"
  | "onset"
  | "precheck"
  | "accompaniment_rms";

type VisualizationState = {
  pxPerSec: number;
  layerVisibility: Record<LayerKey, boolean>;
  playheadTime: number;
  duration: number;
  audioRef: HTMLAudioElement | null;

  setPxPerSec: (value: number) => void;
  toggleLayer: (key: LayerKey) => void;
  setLayerVisibility: (key: LayerKey, visible: boolean) => void;
  setPlayheadTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setAudioRef: (el: HTMLAudioElement | null) => void;
};

const DEFAULT_VISIBILITY: Record<LayerKey, boolean> = {
  a0: true,
  al: true,
  b: true,
  s: true,
  role: true,
  beats: true,
  lyrics: true,
  lyrics_attached: true,
  energy: true,
  onset: true,
  precheck: true,
  accompaniment_rms: true,
};

export const useVisualizationStore = create<VisualizationState>((set) => ({
  pxPerSec: 90,
  layerVisibility: { ...DEFAULT_VISIBILITY },
  playheadTime: 0,
  duration: 0,
  audioRef: null,

  setPxPerSec: (value) => set({ pxPerSec: value }),
  toggleLayer: (key) =>
    set((state) => ({
      layerVisibility: {
        ...state.layerVisibility,
        [key]: !state.layerVisibility[key],
      },
    })),
  setLayerVisibility: (key, visible) =>
    set((state) => ({
      layerVisibility: {
        ...state.layerVisibility,
        [key]: visible,
      },
    })),
  setPlayheadTime: (time) => set({ playheadTime: time }),
  setDuration: (duration) => set({ duration }),
  setAudioRef: (el) => set({ audioRef: el }),
}));
