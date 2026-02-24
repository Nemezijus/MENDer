import { Badge, Group, Stack, Text, Tooltip } from '@mantine/core';

import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';

export function ModelSummary({ artifact }) {
  if (!artifact) {
    return (
      <Text size="sm" c="dimmed">
        No model loaded. Train a model or load a saved artifact to enable prediction.
      </Text>
    );
  }

  const algoId = artifact.model?.algo ?? 'unknown';
  const algoLabel = getAlgoLabel(algoId);
  const metricName = artifact.metric_name ?? artifact.eval?.metric ?? null;
  const metricValue = artifact.metric_value ?? artifact.mean_score ?? null;

  return (
    <Stack gap={4}>
      <Group gap="xs">
        <Text fw={500} size="sm">
          Current model:
        </Text>
        <Tooltip label={algoId} withArrow disabled={!algoId}>
          <Badge variant="light" size="sm">
            {algoLabel}
          </Badge>
        </Tooltip>
      </Group>
      {metricName && (
        <Text size="xs" c="dimmed">
          Trained metric: {metricName}
          {metricValue != null ? ` = ${Number(metricValue).toFixed(4)}` : ''}
        </Text>
      )}
      {artifact.created_at && (
        <Text size="xs" c="dimmed">
          Trained at: {artifact.created_at}
        </Text>
      )}
    </Stack>
  );
}
