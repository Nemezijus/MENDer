// src/state/FeatureContext.jsx
import React, { createContext, useContext, useMemo, useState } from 'react';

const FeatureContext = createContext(null);

export function FeatureProvider({ children }) {

  const [method, setMethod] = useState('none'); // 'none' | 'pca' | 'lda' | 'sfs'

  // PCA
  const [pca_n, setPcaN] = useState(null);       // null or number
  const [pca_var, setPcaVar] = useState(0.95);   // number in (0,1]
  const [pca_whiten, setPcaWhiten] = useState(false);

  // LDA
  const [lda_n, setLdaN] = useState(null);
  const [lda_solver, setLdaSolver] = useState('svd'); // 'svd'|'lsqr'|'eigen'
  const [lda_shrinkage, setLdaShrinkage] = useState(null); // null or float (only for lsqr/eigen)
  const [lda_tol, setLdaTol] = useState(1e-4);

  // SFS
  const [sfs_k, setSfsK] = useState('auto'); // 'auto' or integer-like string
  const [sfs_direction, setSfsDirection] = useState('backward'); // 'forward'|'backward'
  const [sfs_cv, setSfsCv] = useState(5);
  const [sfs_n_jobs, setSfsNJobs] = useState(null);

  const value = useMemo(() => ({
    method, setMethod,
    pca_n, setPcaN,
    pca_var, setPcaVar,
    pca_whiten, setPcaWhiten,

    lda_n, setLdaN,
    lda_solver, setLdaSolver,
    lda_shrinkage, setLdaShrinkage,
    lda_tol, setLdaTol,

    sfs_k, setSfsK,
    sfs_direction, setSfsDirection,
    sfs_cv, setSfsCv,
    sfs_n_jobs, setSfsNJobs,
  }), [
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs
  ]);

  return <FeatureContext.Provider value={value}>{children}</FeatureContext.Provider>;
}

export function useFeatureCtx() {
  const ctx = useContext(FeatureContext);
  if (!ctx) throw new Error('useFeatureCtx must be used within FeatureProvider');
  return ctx;
}
