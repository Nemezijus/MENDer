import { create } from 'zustand';

export const useModelArtifactStore = create((set) => ({
  artifact: null,

  setArtifact: (artifact) => set({ artifact }),

  clearArtifact: () => set({ artifact: null }),
}));
