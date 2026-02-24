import { create } from 'zustand';

import { makeReset } from '../../../shared/state/storeFactories.js';

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
const INITIAL_STATE = {
  train: null,
  learningCurve: null,
  validationCurve: null,
  gridSearch: null,
  randomSearch: null,
};

export const useModelConfigStore = create((set) => ({
  ...INITIAL_STATE,

  // Optional convenience reset.
  resetModelConfigs: makeReset(INITIAL_STATE, set),

  setTrainModel: (model) => set({ train: model }),
  setLearningCurveModel: (model) => set({ learningCurve: model }),
  setValidationCurveModel: (model) => set({ validationCurve: model }),
  setGridSearchModel: (model) => set({ gridSearch: model }),
  setRandomSearchModel: (model) => set({ randomSearch: model }),
}));
