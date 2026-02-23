import { create } from 'zustand';

import {
  makeShallowMergeSetter,
  makeNestedArrayItemRemover,
  makeNestedArrayItemUpdater,
} from '../../../shared/state/storeFactories.js';

/**
 * Ensemble config overrides only.
 *
 * IMPORTANT:
 *   This store must not hardcode Engine contract defaults or task rules.
 *   Any field left as `undefined` means "unset" and should fall back to
 *   `/api/v1/schema/defaults` in the UI, or be omitted from payloads so the
 *   Engine/Backend defaults apply.
 */

const baseSplit = () => ({
  splitMode: undefined,
  trainFrac: undefined,
  nSplits: undefined,
  stratified: undefined,
  shuffle: undefined,
  seed: undefined,
});

const makeVotingInitial = () => ({
  mode: 'simple',
  votingType: undefined,
  estimators: undefined,
  ...baseSplit(),
});

const makeBaggingInitial = () => ({
  mode: 'simple',
  problem_kind: undefined,
  base_estimator: undefined,

  n_estimators: undefined,
  max_samples: undefined,
  max_features: undefined,
  bootstrap: undefined,
  bootstrap_features: undefined,
  oob_score: undefined,
  warm_start: undefined,
  n_jobs: undefined,
  random_state: undefined,

  // Balanced bagging
  balanced: undefined,
  sampling_strategy: undefined,
  replacement: undefined,

  ...baseSplit(),
});

const makeAdaBoostInitial = () => ({
  mode: 'simple',
  problem_kind: undefined,
  base_estimator: undefined,

  n_estimators: undefined,
  learning_rate: undefined,
  algorithm: undefined, // e.g. 'SAMME' | 'SAMME.R' or undefined
  random_state: undefined,

  ...baseSplit(),
});

const makeXGBoostInitial = () => ({
  mode: 'simple',
  problem_kind: undefined,

  n_estimators: undefined,
  learning_rate: undefined,
  max_depth: undefined,
  subsample: undefined,
  colsample_bytree: undefined,

  reg_lambda: undefined,
  reg_alpha: undefined,

  min_child_weight: undefined,
  gamma: undefined,

  use_early_stopping: undefined,
  early_stopping_rounds: undefined,
  eval_set_fraction: undefined,

  n_jobs: undefined,
  random_state: undefined,

  ...baseSplit(),
});

export const useEnsembleStore = create((set) => ({
  voting: makeVotingInitial(),
  bagging: makeBaggingInitial(),
  adaboost: makeAdaBoostInitial(),
  xgboost: makeXGBoostInitial(),

  // ---- voting ----
  setVoting: makeShallowMergeSetter('voting', set),

  setVotingEstimators: (estimators) =>
    set((state) => ({ voting: { ...(state?.voting || {}), estimators } })),

  updateVotingEstimatorAt: makeNestedArrayItemUpdater('voting', 'estimators', set),

  removeVotingEstimatorAt: makeNestedArrayItemRemover('voting', 'estimators', set, {
    minLength: 2,
  }),

  resetVoting: () => set(() => ({ voting: makeVotingInitial() })),

  // ---- bagging ----
  setBagging: makeShallowMergeSetter('bagging', set),

  setBaggingBaseEstimator: (base_estimator) =>
    set((state) => ({ bagging: { ...(state?.bagging || {}), base_estimator } })),

  resetBagging: (_effectiveTask) => set(() => ({ bagging: makeBaggingInitial() })),

  // ---- adaboost ----
  setAdaBoost: makeShallowMergeSetter('adaboost', set),

  setAdaBoostBaseEstimator: (base_estimator) =>
    set((state) => ({ adaboost: { ...(state?.adaboost || {}), base_estimator } })),

  resetAdaBoost: (_effectiveTask) => set(() => ({ adaboost: makeAdaBoostInitial() })),

  // ---- xgboost ----
  setXGBoost: makeShallowMergeSetter('xgboost', set),

  resetXGBoost: (_effectiveTask) => set(() => ({ xgboost: makeXGBoostInitial() })),
}));
