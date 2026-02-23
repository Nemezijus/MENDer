import { Alert, Text } from '@mantine/core';

export default function VotingWarnings({
  duplicateAlgosLabel,
  metricIsAllowed,
  metricOverride,
  effectiveTask,
  defaultMetricFromSchema,
  algoOptionsLength,
  effectiveSplitMode,
}) {
  return (
    <>
      {duplicateAlgosLabel ? (
        <Alert color="yellow" variant="light">
          <Text size="sm" fw={600}>
            Duplicate estimator types detected
          </Text>
          <Text size="sm">
            You selected: <strong>{duplicateAlgosLabel}</strong> more than once. If these are
            identical, this acts like implicit weighting. Prefer using explicit weights in Advanced mode.
          </Text>
        </Alert>
      ) : null}

      {!metricIsAllowed && metricOverride && (
        <Alert color="yellow" variant="light">
          <Text size="sm" fw={600}>
            Metric not available for this task
          </Text>
          <Text size="sm">
            The selected metric (<strong>{metricOverride}</strong>) is not listed for the current task.
            {effectiveTask === 'regression' && defaultMetricFromSchema
              ? ` Using '${defaultMetricFromSchema}' for this run.`
              : ' Please update Settings → Metric.'}
          </Text>
        </Alert>
      )}

      {algoOptionsLength === 0 && (
        <Alert color="yellow" variant="light">
          <Text size="sm" fw={600}>
            Schema defaults not loaded
          </Text>
          <Text size="sm">
            Voting ensemble needs backend schema defaults to list compatible algorithms. Please wait for
            <strong> /api/v1/schema/defaults</strong> to load.
          </Text>
        </Alert>
      )}

      {!effectiveSplitMode && (
        <Alert color="yellow" variant="light">
          <Text size="sm" fw={600}>
            Split defaults not available
          </Text>
          <Text size="sm">
            This panel relies on backend split defaults to choose a split strategy. Please wait for
            <strong> /api/v1/schema/defaults</strong> to load.
          </Text>
        </Alert>
      )}
    </>
  );
}
