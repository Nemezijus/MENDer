import { Card, Stack, Text, Alert } from '@mantine/core';

export default function TrainingDataSummaryCard({ inspectReport, effectiveTask }) {
  const ySum = inspectReport?.y_summary || null;
  const isClassification = effectiveTask === 'classification';
  const isRegression = effectiveTask === 'regression';

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Text fw={600} mb="xs">
        Summary
      </Text>

      {!inspectReport && (
        <Text size="sm" c="dimmed">
          Run Inspect to see data summary.
        </Text>
      )}

      {inspectReport && (
        <Stack gap={2}>
          <Text size="sm">n_samples: {inspectReport.n_samples}</Text>
          <Text size="sm">n_features: {inspectReport.n_features}</Text>
          <Text size="sm">
            task (inferred): {inspectReport.task_inferred || '—'}
          </Text>

          {/* Classification display */}
          {isClassification && (
            <>
              <Text size="sm">
                classes:{' '}
                {Array.isArray(inspectReport.classes)
                  ? inspectReport.classes.join(', ')
                  : String(inspectReport.classes)}
              </Text>
              <Text size="sm">
                n_classes:{' '}
                {Array.isArray(inspectReport.classes)
                  ? inspectReport.classes.length
                  : 0}
              </Text>
            </>
          )}

          {/* Regression display */}
          {isRegression && ySum && (
            <>
              <Text size="sm">
                y: n={ySum.n}, unique={ySum.n_unique}
              </Text>
              <Text size="sm">
                min/max: {ySum.min} / {ySum.max}
              </Text>
              <Text size="sm">
                mean±std: {ySum.mean} ± {ySum.std}
              </Text>
            </>
          )}

          <Text size="sm">
            missing total: {inspectReport.missingness?.total ?? 0}
          </Text>

          {inspectReport.suggestions?.recommend_pca && (
            <Alert color="blue" variant="light" mt="xs">
              <Text size="sm">
                Suggestion:{' '}
                {inspectReport.suggestions.reason || 'consider PCA'}
              </Text>
            </Alert>
          )}
        </Stack>
      )}
    </Card>
  );
}
