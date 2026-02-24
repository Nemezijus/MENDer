import { Text } from '@mantine/core';

import { fmtNumber } from '../utils/artifactUtils.js';

export default function ArtifactStats({ artifact, isEnsemble }) {
  if (!artifact) return null;

  const nParameters =
    Object.prototype.hasOwnProperty.call(artifact, 'n_parameters') ? artifact.n_parameters : null;

  const extraStats =
    artifact && artifact.extra_stats && typeof artifact.extra_stats === 'object' ? artifact.extra_stats : {};

  const hasAnyExtraStat =
    extraStats.n_support_vectors != null ||
    extraStats.n_trees != null ||
    extraStats.total_tree_nodes != null ||
    extraStats.max_tree_depth != null ||
    extraStats.pca_n_components != null ||
    // Ensemble extras (optional)
    extraStats.ensemble_all_agree_rate != null ||
    extraStats.ensemble_pairwise_agreement != null ||
    extraStats.ensemble_tie_rate != null ||
    extraStats.ensemble_mean_margin != null ||
    extraStats.ensemble_best_estimator != null ||
    extraStats.ensemble_corrected_vs_best != null ||
    extraStats.ensemble_harmed_vs_best != null;

  return (
    <>
      <Text size="sm">
        <strong>Parameters:</strong> {nParameters != null ? fmtNumber(nParameters, 0) : 'not available'}
      </Text>

      {extraStats.n_support_vectors != null && (
        <Text size="sm">
          <strong>Support vectors:</strong> {fmtNumber(extraStats.n_support_vectors, 0)}
        </Text>
      )}

      {extraStats.n_trees != null && (
        <Text size="sm">
          <strong>Trees:</strong> {fmtNumber(extraStats.n_trees, 0)}
        </Text>
      )}

      {extraStats.total_tree_nodes != null && (
        <Text size="sm">
          <strong>Total tree nodes:</strong> {fmtNumber(extraStats.total_tree_nodes, 0)}
        </Text>
      )}

      {extraStats.max_tree_depth != null && (
        <Text size="sm">
          <strong>Max tree depth:</strong> {fmtNumber(extraStats.max_tree_depth, 0)}
        </Text>
      )}

      {isEnsemble && extraStats.ensemble_all_agree_rate != null && (
        <Text size="sm">
          <strong>All-agree rate:</strong> {fmtNumber(Number(extraStats.ensemble_all_agree_rate) * 100, 2)}%
        </Text>
      )}

      {isEnsemble && extraStats.ensemble_pairwise_agreement != null && (
        <Text size="sm">
          <strong>Avg pairwise agreement:</strong> {fmtNumber(Number(extraStats.ensemble_pairwise_agreement) * 100, 2)}%
        </Text>
      )}

      {isEnsemble && extraStats.ensemble_tie_rate != null && (
        <Text size="sm">
          <strong>Tie rate:</strong> {fmtNumber(Number(extraStats.ensemble_tie_rate) * 100, 2)}%
        </Text>
      )}

      {isEnsemble && extraStats.ensemble_mean_margin != null && (
        <Text size="sm">
          <strong>Mean vote margin:</strong> {fmtNumber(extraStats.ensemble_mean_margin, 3)}
        </Text>
      )}

      {isEnsemble && extraStats.ensemble_best_estimator != null && (
        <Text size="sm">
          <strong>Best estimator:</strong> {String(extraStats.ensemble_best_estimator)}
        </Text>
      )}

      {isEnsemble &&
        (extraStats.ensemble_corrected_vs_best != null || extraStats.ensemble_harmed_vs_best != null) && (
          <Text size="sm">
            <strong>Vs best:</strong> corrected {fmtNumber(extraStats.ensemble_corrected_vs_best, 0)}, harmed{' '}
            {fmtNumber(extraStats.ensemble_harmed_vs_best, 0)}
          </Text>
        )}

      {!hasAnyExtraStat && nParameters == null && (
        <Text size="sm" c="dimmed">
          Additional stats not available for this artifact.
        </Text>
      )}
    </>
  );
}
