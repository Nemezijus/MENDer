import { create } from 'zustand';

import { makeReset } from '../../../shared/state/storeFactories.js';

const initialState = {
  artifact: null,
  /** 'trained' | 'loaded' | null */
  source: null,
};

export const useModelArtifactStore = create((set) => ({
  ...initialState,

  /**
   * @param {any} artifact
   * @param {'trained' | 'loaded'} [source]
   */
  setArtifact: (artifact, source = 'trained') => set({ artifact, source }),

  clearArtifact: () => set({ artifact: null, source: null }),

  reset: makeReset(initialState, set),
}));
