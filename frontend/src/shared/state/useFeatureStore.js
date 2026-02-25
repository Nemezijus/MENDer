import { create } from 'zustand';

import { makeKeySetter, makeReset } from './storeFactories.js';

/**
 * Feature config overrides only.
 *
 * IMPORTANT:
 *   This store must not hardcode Engine contract defaults. Any field left as
 *   `undefined` means "unset" and should fall back to `/api/v1/schema/defaults`
 *   in the UI, or be omitted from the payload so the Engine/Backend defaults
 *   apply.
 */

const emptyState = {
  // --- feature method -----------------------------------------------------
  // 'none' | 'pca' | 'lda' | 'sfs'
  method: undefined,

  // PCA
  pca_n: undefined,
  pca_var: undefined,
  pca_whiten: undefined,

  // LDA
  lda_n: undefined,
  lda_solver: undefined,
  lda_shrinkage: undefined,
  lda_tol: undefined,

  // SFS
  sfs_k: undefined,
  sfs_direction: undefined,
  sfs_cv: undefined,
  sfs_n_jobs: undefined,
};

const hasOwn = (obj, key) => obj != null && Object.prototype.hasOwnProperty.call(obj, key);
const pick = (obj, key) => (hasOwn(obj, key) ? obj[key] : undefined);

export const useFeatureStore = create((set) => ({
  ...emptyState,

  // --- setters --------------------------------------------------------------
  setMethod: makeKeySetter('method', set),

  setPcaN: makeKeySetter('pca_n', set),
  setPcaVar: makeKeySetter('pca_var', set),
  setPcaWhiten: makeKeySetter('pca_whiten', set),

  setLdaN: makeKeySetter('lda_n', set),
  setLdaSolver: makeKeySetter('lda_solver', set),
  setLdaShrinkage: makeKeySetter('lda_shrinkage', set),
  setLdaTol: makeKeySetter('lda_tol', set),

  setSfsK: makeKeySetter('sfs_k', set),
  setSfsDirection: makeKeySetter('sfs_direction', set),
  setSfsCv: makeKeySetter('sfs_cv', set),
  setSfsNJobs: makeKeySetter('sfs_n_jobs', set),

  reset: makeReset(emptyState, set),

  // --- one-shot hydration from artifact.features ---------------------------
  // NOTE: do not apply fallbacks here; preserve artifact values only.
  setFromArtifact: (feat) =>
    set(() => ({
      ...emptyState,
      method: pick(feat, 'method'),

      pca_n: pick(feat, 'pca_n'),
      pca_var: pick(feat, 'pca_var'),
      pca_whiten: pick(feat, 'pca_whiten'),

      lda_n: pick(feat, 'lda_n'),
      lda_solver: pick(feat, 'lda_solver'),
      lda_shrinkage: pick(feat, 'lda_shrinkage'),
      lda_tol: pick(feat, 'lda_tol'),

      sfs_k: pick(feat, 'sfs_k'),
      sfs_direction: pick(feat, 'sfs_direction'),
      sfs_cv: pick(feat, 'sfs_cv'),
      sfs_n_jobs: pick(feat, 'sfs_n_jobs'),
    })),
}));
