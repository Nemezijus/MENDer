import { create } from 'zustand';

export const useFeatureStore = create((set) => ({
  // --- feature method -------------------------------------------------------
  // 'none' | 'pca' | 'lda' | 'sfs'
  method: 'none',

  // PCA
  pca_n: null,          // null or number
  pca_var: 0.95,        // number in (0,1]
  pca_whiten: false,

  // LDA
  lda_n: null,
  lda_solver: 'svd',    // 'svd' | 'lsqr' | 'eigen'
  lda_shrinkage: null,  // null or float (only for lsqr/eigen)
  lda_tol: 1e-4,

  // SFS
  sfs_k: 'auto',        // 'auto' or integer-like string
  sfs_direction: 'backward', // 'forward' | 'backward'
  sfs_cv: 5,
  sfs_n_jobs: null,

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

  // --- one-shot hydration from artifact.features ---------------------------
  setFromArtifact: (feat) =>
    set((prev) => {
      const m = feat?.method ?? 'none';

      // start from previous state and overwrite only what we need
      const next = { ...prev, method: m };

      if (m === 'pca') {
        next.pca_n = feat?.pca_n ?? null;
        next.pca_var = feat?.pca_var ?? 0.95;
        next.pca_whiten = !!feat?.pca_whiten;
        return next;
      }

      if (m === 'lda') {
        next.lda_n = feat?.lda_n ?? null;
        next.lda_solver = feat?.lda_solver ?? 'svd';
        next.lda_shrinkage = feat?.lda_shrinkage ?? null;
        next.lda_tol = feat?.lda_tol ?? 1e-4;
        return next;
      }

      if (m === 'sfs') {
        let k = feat?.sfs_k;
        if (k == null || k === '') k = 'auto';
        next.sfs_k = k;
        next.sfs_direction = feat?.sfs_direction ?? 'backward';
        next.sfs_cv = Number(feat?.sfs_cv ?? 5);
        next.sfs_n_jobs = feat?.sfs_n_jobs ?? null;
        return next;
      }

      // 'none' -> leave other fields as-is
      return next;
    }),
}));
