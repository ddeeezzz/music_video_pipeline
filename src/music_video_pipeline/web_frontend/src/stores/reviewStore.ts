import { create } from "zustand";

type ReviewState = {
  currentTime: number;
  duration: number;
  selectedSegmentId: string;
  promptModalSegmentId: string;
  setPlaybackState: (currentTime: number, duration: number) => void;
  setSelectedSegmentId: (segmentId: string) => void;
  openPromptModal: (segmentId: string) => void;
  closePromptModal: () => void;
};

export const useReviewStore = create<ReviewState>((set) => ({
  currentTime: 0,
  duration: 0,
  selectedSegmentId: "",
  promptModalSegmentId: "",
  setPlaybackState: (currentTime, duration) => set({ currentTime, duration }),
  setSelectedSegmentId: (segmentId) => set({ selectedSegmentId: segmentId }),
  openPromptModal: (segmentId) => set({ promptModalSegmentId: segmentId, selectedSegmentId: segmentId }),
  closePromptModal: () => set({ promptModalSegmentId: "" }),
}));
