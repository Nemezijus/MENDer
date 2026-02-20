import { create } from 'zustand';

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
  setVoting: (partial) =>
    set((state) => ({ voting: { ...state.voting, ...partial } })),

  setVotingEstimators: (estimators) =>
    set((state) => ({ voting: { ...state.voting, estimators } })),

  updateVotingEstimatorAt: (idx, patch) =>
    set((state) => {
      const cur = state.voting.estimators || [];
      const next = cur.map((s, i) => (i === idx ? { ...s, ...patch } : s));
      return { voting: { ...state.voting, estimators: next } };
    }),

  removeVotingEstimatorAt: (idx) =>
    set((state) => {
      const cur = state.voting.estimators || [];
      if (cur.length <= 2) return state;
      const next = cur.filter((_, i) => i !== idx);
      return { voting: { ...state.voting, estimators: next } };
    }),

  resetVoting: () => set(() => ({ voting: makeVotingInitial() })),

  // ---- bagging ----
  setBagging: (partial) =>
    set((state) => ({ bagging: { ...state.bagging, ...partial } })),

  setBaggingBaseEstimator: (base_estimator) =>
    set((state) => ({ bagging: { ...state.bagging, base_estimator } })),

  resetBagging: (_effectiveTask) => set(() => ({ bagging: makeBaggingInitial() })),

  // ---- adaboost ----
  setAdaBoost: (partial) =>
    set((state) => ({ adaboost: { ...state.adaboost, ...partial } })),

  setAdaBoostBaseEstimator: (base_estimator) =>
    set((state) => ({ adaboost: { ...state.adaboost, base_estimator } })),

  resetAdaBoost: (_effectiveTask) => set(() => ({ adaboost: makeAdaBoostInitial() })),

  // ---- xgboost ----
  setXGBoost: (partial) =>
    set((state) => ({ xgboost: { ...state.xgboost, ...partial } })),

  resetXGBoost: (_effectiveTask) => set(() => ({ xgboost: makeXGBoostInitial() })),
}));
