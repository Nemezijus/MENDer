import { create } from 'zustand';

const makeEmptyParam = () => ({
  paramName: '',
  values: [],
});

export const useTuningStore = create((set) => ({
  // Learning curve-specific UI state
  learningCurve: {
    // NOTE: overrides only; leave unset so backend/engine defaults apply.
    stratified: undefined,
    shuffle: undefined,
    seed: undefined,
    trainSizesCSV: '',
    nSteps: undefined,
    nJobs: undefined,
  },

  // Validation curve-specific UI state + last result
  validationCurve: {
    nSplits: undefined,
    stratified: undefined,
    shuffle: undefined,
    seed: undefined,
    nJobs: undefined,
    hyperParam: makeEmptyParam(),
    result: null,
  },

  // Grid search-specific UI state + last result
  gridSearch: {
    nSplits: undefined,
    stratified: undefined,
    shuffle: undefined,
    seed: undefined,
    nJobs: undefined,
    hyperParam1: makeEmptyParam(),
    hyperParam2: makeEmptyParam(),
    result: null,
  },

  // Randomized search-specific UI state + last result
  randomSearch: {
    nSplits: undefined,
    stratified: undefined,
    shuffle: undefined,
    seed: undefined,
    nJobs: undefined,
    nIter: undefined,
    hyperParam1: makeEmptyParam(),
    hyperParam2: makeEmptyParam(),
    result: null,
  },

  // Shallow-merge updates for each sub-slice
  setLearningCurve: (partial) =>
    set((state) => ({
      learningCurve: { ...state.learningCurve, ...partial },
    })),

  setValidationCurve: (partial) =>
    set((state) => ({
      validationCurve: { ...state.validationCurve, ...partial },
    })),

  setGridSearch: (partial) =>
    set((state) => ({
      gridSearch: { ...state.gridSearch, ...partial },
    })),

  setRandomSearch: (partial) =>
    set((state) => ({
      randomSearch: { ...state.randomSearch, ...partial },
    })),
}));
