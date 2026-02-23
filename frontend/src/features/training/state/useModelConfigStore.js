import { create } from 'zustand';

/**
 * Per-panel model configs:
 * - train: RunModelPanel
 * - learningCurve: LearningCurvePanel
 * - validationCurve: ValidationCurvePanel
 * - gridSearch: GridSearchPanel
 * - randomSearch: RandomSearchPanel
 *
 * Each slice is independent so tuning changes don't affect training, etc.,
 * but they persist across navigation because this is a global store.
 */
export const useModelConfigStore = create((set) => ({
  train: null,
  learningCurve: null,
  validationCurve: null,
  gridSearch: null,
  randomSearch: null,

  setTrainModel: (model) => set({ train: model }),
  setLearningCurveModel: (model) => set({ learningCurve: model }),
  setValidationCurveModel: (model) => set({ validationCurve: model }),
  setGridSearchModel: (model) => set({ gridSearch: model }),
  setRandomSearchModel: (model) => set({ randomSearch: model }),
}));
