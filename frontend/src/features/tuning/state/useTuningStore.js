import { create } from 'zustand';
import { makeReset, makeShallowMergeSetter } from '../../../shared/state/storeFactories.js';

const makeEmptyParam = () => ({
  paramName: '',
  values: [],
});

const INITIAL_STATE = {
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
};

export const useTuningStore = create((set) => ({
  ...INITIAL_STATE,

  // Reset all tuning UI state.
  resetTuning: makeReset(INITIAL_STATE, set),

  // Shallow-merge updates for each sub-slice.
  setLearningCurve: makeShallowMergeSetter('learningCurve', set),
  setValidationCurve: makeShallowMergeSetter('validationCurve', set),
  setGridSearch: makeShallowMergeSetter('gridSearch', set),
  setRandomSearch: makeShallowMergeSetter('randomSearch', set),
}));
