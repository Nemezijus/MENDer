import { create } from 'zustand';

const baseSplit = () => ({
  splitMode: 'holdout',
  trainFrac: 0.75,
  nSplits: 5,
  stratified: true,
  shuffle: true,
  seed: '',
});

const makeVotingInitial = () => ({
  mode: 'simple',
  votingType: 'hard',
  estimators: null,
  ...baseSplit(),
});

const makeBaggingInitial = () => ({
  mode: 'simple',
  problem_kind: 'classification',
  base_estimator: null,

  n_estimators: 10,
  max_samples: 1.0,
  max_features: 1.0,
  bootstrap: true,
  bootstrap_features: false,
  oob_score: false,
  warm_start: false,
  n_jobs: '',
  random_state: '',

  ...baseSplit(),
});

const makeAdaBoostInitial = () => ({
  mode: 'simple',
  problem_kind: 'classification',
  base_estimator: null,

  n_estimators: 50,
  learning_rate: 1.0,
  algorithm: '__default__', // '__default__' | 'SAMME' | 'SAMME.R'
  random_state: '',

  ...baseSplit(),
});

const makeXGBoostInitial = () => ({
  mode: 'simple',
  problem_kind: 'classification',

  n_estimators: 300,
  learning_rate: 0.1,
  max_depth: 6,
  subsample: 1.0,
  colsample_bytree: 1.0,

  reg_lambda: 1.0,
  reg_alpha: 0.0,

  min_child_weight: 1.0,
  gamma: 0.0,

  n_jobs: '',
  random_state: '',

  __hydrated: false,

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

  resetBagging: (effectiveTask) =>
    set(() => {
      const next = makeBaggingInitial();
      next.problem_kind = effectiveTask === 'regression' ? 'regression' : 'classification';
      if (effectiveTask === 'regression') next.stratified = false;
      return { bagging: next };
    }),

  // ---- adaboost ----
  setAdaBoost: (partial) =>
    set((state) => ({ adaboost: { ...state.adaboost, ...partial } })),

  setAdaBoostBaseEstimator: (base_estimator) =>
    set((state) => ({ adaboost: { ...state.adaboost, base_estimator } })),

  resetAdaBoost: (effectiveTask) =>
    set(() => {
      const next = makeAdaBoostInitial();
      next.problem_kind = effectiveTask === 'regression' ? 'regression' : 'classification';
      if (effectiveTask === 'regression') next.stratified = false;
      return { adaboost: next };
    }),

  // ---- xgboost ----
  setXGBoost: (partial) =>
    set((state) => ({ xgboost: { ...state.xgboost, ...partial } })),

  resetXGBoost: (effectiveTask) =>
    set(() => {
      const next = makeXGBoostInitial();
      next.problem_kind = effectiveTask === 'regression' ? 'regression' : 'classification';
      if (effectiveTask === 'regression') next.stratified = false;
      return { xgboost: next };
    }),
}));
