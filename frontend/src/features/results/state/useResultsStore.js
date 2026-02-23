import { create } from 'zustand';

export const useResultsStore = create((set) => ({
  // --- Result view selection (for ResultsPanel) ---
  activeResultKind: null,               // 'train' | 'regression' | 'cluster' | ...
  setActiveResultKind: (kind) => set({ activeResultKind: kind }),
  // --- Train a model -------------------
  trainResult: null,
  setTrainResult: (result) => set({ trainResult: result }),
  clearTrainResult: () => set({ trainResult: null }),

  // --- Learning curve ------------
  learningCurveResult: null,
  // Override-only: leave unset so backend defaults apply when omitted.
  learningCurveNSplits: undefined,
  // UI-only setting used for visualization cutoff.
  learningCurveWithinPct: 0.99,

  setLearningCurveResult: (result) => set({ learningCurveResult: result }),
  clearLearningCurveResult: () => set({ learningCurveResult: null }),

  setLearningCurveNSplits: (nSplits) =>
    set({
      learningCurveNSplits:
        nSplits === '' || nSplits == null ? undefined : Number(nSplits),
    }),

  setLearningCurveWithinPct: (pct) =>
    set({ learningCurveWithinPct: Number(pct) || 0.99 }),

  // --- Production apply -------------
  applyResult: null,        // ApplyModelResponse
  productionIsRunning: false,
  productionError: null,

  setApplyResult: (applyResult) => set({ applyResult }),
  clearApplyResult: () => set({ applyResult: null }),

  setProductionIsRunning: (isRunning) => set({ productionIsRunning: !!isRunning }),

  setProductionError: (error) => set({ productionError: error }),
}));
