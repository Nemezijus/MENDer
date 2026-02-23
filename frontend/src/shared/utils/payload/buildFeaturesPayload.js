import { compactPayload } from '../compactPayload.js';

/**
 * Build the "features" section from an override-only feature store snapshot.
 *
 * Supported methods:
 * - 'pca'
 * - 'lda'
 * - 'sfs'
 * - undefined / null => omitted (engine defaults apply)
 */
export function buildFeaturesPayload(featureCtx) {
  const m = featureCtx?.method;
  if (!m) return {};

  if (m === 'pca') {
    return compactPayload({
      method: 'pca',
      pca_n: featureCtx.pca_n,
      pca_var: featureCtx.pca_var,
      pca_whiten: featureCtx.pca_whiten,
    });
  }

  if (m === 'lda') {
    return compactPayload({
      method: 'lda',
      lda_n: featureCtx.lda_n,
      lda_solver: featureCtx.lda_solver,
      lda_shrinkage: featureCtx.lda_shrinkage,
      lda_tol: featureCtx.lda_tol,
    });
  }

  if (m === 'sfs') {
    return compactPayload({
      method: 'sfs',
      sfs_k: featureCtx.sfs_k,
      sfs_direction: featureCtx.sfs_direction,
      sfs_cv: featureCtx.sfs_cv,
      sfs_n_jobs: featureCtx.sfs_n_jobs,
    });
  }

  // Unknown method: pass through only the method discriminator.
  return compactPayload({ method: m });
}
