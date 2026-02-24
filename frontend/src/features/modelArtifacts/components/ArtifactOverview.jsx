import { Text } from '@mantine/core';

import { fmtNumber, niceUnsupervisedMetricName } from '../utils/artifactUtils.js';

export default function ArtifactOverview({
  artifact,
  isUnsupervised,
  isKFold,
  kfoldTotalN,
  featuresText,
  ensembleSummary,
}) {
  if (!artifact) return null;

  return (
    <>
      {ensembleSummary ? <Text size="sm">{ensembleSummary}</Text> : null}

      <Text size="sm">
        <strong>Created:</strong>{' '}
        {artifact.created_at ? new Date(artifact.created_at).toLocaleString() : '—'}
      </Text>

      {isUnsupervised ? (
        <Text size="sm">
          <strong>Diagnostics:</strong> {niceUnsupervisedMetricName(artifact.metric_name)}
        </Text>
      ) : (
        <Text size="sm">
          <strong>Metric:</strong> {artifact.metric_name ?? '—'} ({fmtNumber(artifact.mean_score)} ±{' '}
          {fmtNumber(artifact.std_score)})
        </Text>
      )}

      <Text size="sm">
        <strong>Data:</strong>{' '}
        {isUnsupervised ? (
          <>
            train {fmtNumber(artifact.n_samples_train, 0)}, features {fmtNumber(artifact.n_features_in, 0)}
          </>
        ) : isKFold ? (
          <>
            train {fmtNumber(artifact.n_samples_train, 0)} / fold, test {fmtNumber(artifact.n_samples_test, 0)} / fold
            {kfoldTotalN != null ? ` (N=${fmtNumber(kfoldTotalN, 0)} OOF)` : ''}, features{' '}
            {fmtNumber(artifact.n_features_in, 0)}
          </>
        ) : (
          <>
            train {fmtNumber(artifact.n_samples_train, 0)}, test {fmtNumber(artifact.n_samples_test, 0)}, features{' '}
            {fmtNumber(artifact.n_features_in, 0)}
          </>
        )}
      </Text>

      {Array.isArray(artifact.classes) && artifact.classes.length > 0 && (
        <Text size="sm">
          <strong>Classes:</strong> {artifact.classes.join(', ')}
        </Text>
      )}

      <Text size="sm">
        <strong>Scaler:</strong> {artifact?.scale?.method ?? 'none'}
      </Text>

      <Text size="sm">
        <strong>Features:</strong> {featuresText}
      </Text>

      <Text size="sm">
        <strong>Split:</strong> {isUnsupervised ? '—' : artifact?.split?.mode ?? '—'}
        {isKFold ? ' (out-of-fold pooled)' : ''}
      </Text>
    </>
  );
}
