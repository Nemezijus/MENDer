import { compactPayload } from '../../../shared/utils/compactPayload.js';

/**
 * Build the unsupervised "eval" section.
 *
 * Convention:
 * - overrides only
 * - keep field names aligned with backend pydantic models
 */
export function buildUnsupervisedEvalPayload({
  metrics,
  includeClusterProbabilities,
  embeddingMethod,
} = {}) {
  return compactPayload({
    metrics,
    include_cluster_probabilities: includeClusterProbabilities,
    embedding_method: embeddingMethod,
  });
}
