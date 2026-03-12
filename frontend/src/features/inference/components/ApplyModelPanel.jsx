import { Alert, Button, Card, Group, Stack, Text } from '@mantine/core';

import { useProductionDataStore } from '../../dataFiles/state/useProductionDataStore.js';
import { useFilesConstraintsQuery } from '../../../shared/schema/useFilesConstraintsQuery.js';

import { useApplyModelRunner } from '../hooks/useApplyModelRunner.js';
import { ModelSummary } from './ModelSummary.jsx';
import { PredictionsPreview } from './PredictionsPreview.jsx';

export default function ApplyModelPanel() {
  const {
    artifact,
    hasModel,
    canRun,
    isRunning,
    error,
    applyResult,
    runPrediction,
    exportPredictionsCsv,
  } = useApplyModelRunner();

  const xPath = useProductionDataStore((s) => s.xPath);
  const yPath = useProductionDataStore((s) => s.yPath);
  const npzPath = useProductionDataStore((s) => s.npzPath);
  const xKey = useProductionDataStore((s) => s.xKey);
  const yKey = useProductionDataStore((s) => s.yKey);

  const { data: filesConstraints } = useFilesConstraintsQuery();
  const defaultXKey = filesConstraints?.data_default_keys?.x_key ?? 'X';
  const defaultYKey = filesConstraints?.data_default_keys?.y_key ?? 'y';
  const displayXKey = xKey?.trim() || defaultXKey;
  const displayYKey = yKey?.trim() || defaultYKey;

  return (
    <Card shadow="sm" padding="md" withBorder>
      <Stack gap="md">
        <Text fw={600} size="sm">
          Apply model to new data
        </Text>

        <ModelSummary artifact={artifact} />

        <Stack gap="xs">
          <Text size="xs" fw={500}>
            Production data
          </Text>
          <Text size="xs" c="dimmed">
            Features (X): {npzPath || xPath || 'not set'}
          </Text>
          <Text size="xs" c="dimmed">
            Labels (y, optional): {yPath || 'not set'}
          </Text>
          <Text size="xs" c="dimmed">
            Keys: X = {displayXKey}, y = {displayYKey}
          </Text>
        </Stack>

        {error && (
          <Alert color="red" variant="light">
            <Text size="xs">{error}</Text>
          </Alert>
        )}

        <Group justify="space-between">
          <Button size="xs" onClick={runPrediction} disabled={!canRun}>
            {isRunning ? 'Running…' : 'Run prediction'}
          </Button>
          <Button
            size="xs"
            variant="outline"
            onClick={exportPredictionsCsv}
            disabled={!applyResult || !hasModel}
          >
            Save predictions as CSV
          </Button>
        </Group>

        <PredictionsPreview applyResult={applyResult} />
      </Stack>
    </Card>
  );
}
