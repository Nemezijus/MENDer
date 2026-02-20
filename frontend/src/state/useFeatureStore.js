import { create } from 'zustand';

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
  setMethod: (method) => set({ method }),

  setPcaN:       (pca_n)       => set({ pca_n }),
  setPcaVar:     (pca_var)     => set({ pca_var }),
  setPcaWhiten:  (pca_whiten)  => set({ pca_whiten }),

  setLdaN:         (lda_n)         => set({ lda_n }),
  setLdaSolver:    (lda_solver)    => set({ lda_solver }),
  setLdaShrinkage: (lda_shrinkage) => set({ lda_shrinkage }),
  setLdaTol:       (lda_tol)       => set({ lda_tol }),

  setSfsK:        (sfs_k)        => set({ sfs_k }),
  setSfsDirection:(sfs_direction)=> set({ sfs_direction }),
  setSfsCv:       (sfs_cv)       => set({ sfs_cv }),
  setSfsNJobs:    (sfs_n_jobs)   => set({ sfs_n_jobs }),

  reset: () => set({ ...emptyState }),

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
