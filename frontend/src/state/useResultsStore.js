import { create } from 'zustand';

export const useResultsStore = create((set) => ({
  // --- Train a model -------------------
  trainResult: null,
  setTrainResult: (result) => set({ trainResult: result }),
  clearTrainResult: () => set({ trainResult: null }),

  // --- Learning curve ------------
  learningCurveResult: null,
  learningCurveNSplits: 5,
  learningCurveWithinPct: 0.99,

  setLearningCurveResult: (result) => set({ learningCurveResult: result }),
  clearLearningCurveResult: () => set({ learningCurveResult: null }),

  setLearningCurveNSplits: (nSplits) =>
    set({ learningCurveNSplits: Number(nSplits) || 1 }),

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
