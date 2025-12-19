// frontend/src/state/useEnsembleStore.js
import { create } from 'zustand';

const makeVotingInitial = () => ({
  // UI mode
  mode: 'simple', // 'simple' | 'advanced'

  // Voting config
  votingType: 'hard', // 'hard' | 'soft' (ignored for regression on backend)

  // Estimators: [{ name, weight, model }]
  // We intentionally start as null so the panel can hydrate from schema defaults once available.
  estimators: null,

  // Split config
  splitMode: 'holdout', // 'holdout' | 'kfold'
  trainFrac: 0.75,
  nSplits: 5,
  stratified: true,
  shuffle: true,
  seed: '',
});

const makePlaceholderInitial = () => ({
  // For future tabs: keep a place to persist settings across navigation
  splitMode: 'holdout',
  trainFrac: 0.75,
  nSplits: 5,
  stratified: true,
  shuffle: true,
  seed: '',
});

export const useEnsembleStore = create((set) => ({
  voting: makeVotingInitial(),
  bagging: makePlaceholderInitial(),
  adaboost: makePlaceholderInitial(),
  xgboost: makePlaceholderInitial(),

  // ---- voting setters ------------------------------------------------------

  setVoting: (partial) =>
    set((state) => ({
      voting: { ...state.voting, ...partial },
    })),

  setVotingEstimators: (estimators) =>
    set((state) => ({
      voting: { ...state.voting, estimators },
    })),

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

  resetVoting: () =>
    set(() => ({
      voting: makeVotingInitial(),
    })),

  // ---- future tab setters (placeholders) -----------------------------------

  setBagging: (partial) =>
    set((state) => ({ bagging: { ...state.bagging, ...partial } })),

  setAdaBoost: (partial) =>
    set((state) => ({ adaboost: { ...state.adaboost, ...partial } })),

  setXGBoost: (partial) =>
    set((state) => ({ xgboost: { ...state.xgboost, ...partial } })),
}));
