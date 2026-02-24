import { create } from 'zustand';

import { makeReset } from '../../../shared/state/storeFactories.js';

/**
 * Training UI state that should persist across navigation.
 *
 * IMPORTANT: override-only.
 * - Any field left as `undefined` means "unset" and should fall back to schema defaults.
 * - `seed` is kept as a string because Mantine's NumberInput may emit ''.
 */

const INITIAL_STATE = {
  // --- split overrides ----------------------------------------------------
  splitMode: undefined, // undefined => schema default (holdout)
  trainFrac: undefined,
  nSplits: undefined,
  stratified: undefined,
  shuffle: undefined,
  seed: '',

  // --- shuffle-baseline (UI / eval override) ------------------------------
  useShuffleBaseline: false,
  nShuffles: 100,
};

export const useTrainingStore = create((set) => ({
  ...INITIAL_STATE,

  resetTraining: makeReset(INITIAL_STATE, set),

  // split
  setSplitMode: (splitMode) => set({ splitMode }),
  setTrainFrac: (trainFrac) => set({ trainFrac }),
  setNSplits: (nSplits) => set({ nSplits }),
  setStratified: (stratified) => set({ stratified }),
  setShuffle: (shuffle) => set({ shuffle }),
  setSeed: (seed) => {
    if (seed === '' || seed == null) return set({ seed: '' });
    return set({ seed: String(seed) });
  },

  // shuffle baseline
  setUseShuffleBaseline: (useShuffleBaseline) =>
    set({ useShuffleBaseline: !!useShuffleBaseline }),
  setNShuffles: (nShuffles) => set({ nShuffles: Number(nShuffles) || 0 }),
}));
