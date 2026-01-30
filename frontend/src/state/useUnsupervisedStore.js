import { create } from 'zustand';

const initialState = {
  // modelling
  algo: null,
  model: null,
  fitScope: 'train_only',

  // evaluation
  // empty list => backend computes the default pack
  metrics: [],
  includeClusterProbabilities: false,

  // visualization/diagnostics
  embeddingMethod: 'pca',
  embeddingMaxPoints: 5000,
};

export const useUnsupervisedStore = create((set) => ({
  ...initialState,

  setAlgo: (algo) =>
    set((s) => ({
      algo,
      model: s.model ? { ...s.model, algo } : s.model,
    })),
  setModel: (model) =>
    set({
      model,
      algo: model?.algo ?? null,
    }),
  hydrateModel: (baseDefaults, prev) =>
    set(() => ({
      model: { ...(baseDefaults || {}), ...(prev || {}) },
      algo: (prev?.algo ?? baseDefaults?.algo ?? null) || null,
    })),
  setFitScope: (fitScope) => set({ fitScope }),

  setMetrics: (metrics) => set({ metrics }),
  setIncludeClusterProbabilities: (includeClusterProbabilities) =>
    set({ includeClusterProbabilities }),

  setEmbeddingMethod: (embeddingMethod) => set({ embeddingMethod }),
  setEmbeddingMaxPoints: (embeddingMaxPoints) => set({ embeddingMaxPoints }),

  reset: () => set({ ...initialState }),
}));
