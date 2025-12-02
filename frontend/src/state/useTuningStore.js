import { create } from 'zustand';

const makeEmptyParam = () => ({
  paramName: '',
  values: [],
});

export const useTuningStore = create((set) => ({
  // Learning curve-specific UI state
  learningCurve: {
    stratified: true,
    shuffle: true,
    seed: 42,
    trainSizesCSV: '',
    nSteps: 5,
    nJobs: 1,
  },

  // Validation curve-specific UI state + last result
  validationCurve: {
    nSplits: 5,
    stratified: true,
    shuffle: true,
    seed: 42,
    nJobs: 1,
    hyperParam: makeEmptyParam(),
    result: null,
  },

  // Grid search-specific UI state + last result
  gridSearch: {
    nSplits: 5,
    stratified: true,
    shuffle: true,
    seed: 42,
    nJobs: 1,
    hyperParam1: makeEmptyParam(),
    hyperParam2: makeEmptyParam(),
    result: null,
  },

  // Randomized search-specific UI state + last result
  randomSearch: {
    nSplits: 5,
    stratified: true,
    shuffle: true,
    seed: 42,
    nJobs: 1,
    nIter: 20,
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
