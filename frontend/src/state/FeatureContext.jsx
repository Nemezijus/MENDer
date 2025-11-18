import { createContext, useContext, useMemo, useState } from 'react';

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

  // --- one-shot hydration from artifact.features ---
  const setFromArtifact = (feat) => {
    const m = feat?.method ?? 'none';
    setMethod(m);

    if (m === 'pca') {
      setPcaN(feat?.pca_n ?? null);
      setPcaVar(feat?.pca_var ?? 0.95);
      setPcaWhiten(!!feat?.pca_whiten);
      return;
    }

    if (m === 'lda') {
      setLdaN(feat?.lda_n ?? null);
      setLdaSolver(feat?.lda_solver ?? 'svd');
      setLdaShrinkage(feat?.lda_shrinkage ?? null);
      setLdaTol(feat?.lda_tol ?? 1e-4);
      return;
    }

    if (m === 'sfs') {
      let k = feat?.sfs_k;
      if (k == null || k === '') k = 'auto';
      setSfsK(k);
      setSfsDirection(feat?.sfs_direction ?? 'backward');
      setSfsCv(Number(feat?.sfs_cv ?? 5));
      setSfsNJobs(feat?.sfs_n_jobs ?? null);
      return;
    }

    // 'none' -> leave other fields as-is
  };

  const value = useMemo(() => ({
    // state
    method, pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,

    // setters
    setMethod,
    setPcaN, setPcaVar, setPcaWhiten,
    setLdaN, setLdaSolver, setLdaShrinkage, setLdaTol,
    setSfsK, setSfsDirection, setSfsCv, setSfsNJobs,

    // hydrator
    setFromArtifact,
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
