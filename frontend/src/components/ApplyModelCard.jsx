import {
  Card,
  Stack,
  Text,
  Group,
  Button,
  Alert,
  Table,
  Badge,
} from '@mantine/core';

import { useProductionDataStore } from '../state/useProductionDataStore.js';
import { useResultsStore } from '../state/useResultsStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import {
  applyModelToData,
  exportPredictions,
  saveBlobInteractive,
  downloadBlob,
} from '../api/models';

function ModelSummary({ artifact }) {
  if (!artifact) {
    return (
      <Text size="sm" c="dimmed">
        No model loaded. Train a model or load a saved artifact to enable prediction.
      </Text>
    );
  }

  const algo = artifact.model?.algo ?? 'unknown';
  const metricName = artifact.metric_name ?? artifact.eval?.metric ?? null;
  const metricValue = artifact.metric_value ?? artifact.mean_score ?? null;

  return (
    <Stack gap={4}>
      <Group gap="xs">
        <Text fw={500} size="sm">
          Current model:
        </Text>
        <Badge variant="light" size="sm">
          {algo}
        </Badge>
      </Group>
      {metricName && (
        <Text size="xs" c="dimmed">
          Trained metric: {metricName}
          {metricValue != null ? ` = ${metricValue.toFixed(4)}` : ''}
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

function PredictionsPreview({ applyResult }) {
  if (!applyResult) return null;

  const { n_samples, n_features, task, metric_name, metric_value, preview } =
    applyResult;

  return (
    <Stack gap="xs">
      <Text size="sm" fw={500}>
        Prediction summary
      </Text>
      <Text size="xs" c="dimmed">
        Samples: {n_samples} · Features: {n_features} · Task: {task}
      </Text>
      {metric_name && metric_value != null && (
        <Text size="xs" c="dimmed">
          {metric_name}: {metric_value.toFixed(4)} (on uploaded labels)
        </Text>
      )}
      {preview && preview.length > 0 && (
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Index</Table.Th>
              <Table.Th>y_pred</Table.Th>
              <Table.Th>y_true</Table.Th>
              <Table.Th>Residual</Table.Th>
              <Table.Th>Correct</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {preview.map((row) => (
              <Table.Tr key={row.index}>
                <Table.Td>{row.index}</Table.Td>
                <Table.Td>{String(row.y_pred)}</Table.Td>
                <Table.Td>
                  {row.y_true !== undefined && row.y_true !== null
                    ? String(row.y_true)
                    : '—'}
                </Table.Td>
                <Table.Td>
                  {row.residual !== undefined && row.residual !== null
                    ? row.residual.toFixed(4)
                    : '—'}
                </Table.Td>
                <Table.Td>
                  {typeof row.correct === 'boolean'
                    ? row.correct
                      ? '✓'
                      : '✗'
                    : '—'}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}
    </Stack>
  );
}

export default function ApplyModelCard() {
  const artifact = useModelArtifactStore((s) => s.artifact);

  // --- production data (Zustand) ---
  const xPath = useProductionDataStore((s) => s.xPath);
  const yPath = useProductionDataStore((s) => s.yPath);
  const npzPath = useProductionDataStore((s) => s.npzPath);
  const xKey = useProductionDataStore((s) => s.xKey);
  const yKey = useProductionDataStore((s) => s.yKey);

  const dataReady = useProductionDataStore(
    (s) => Boolean(s.xPath || s.npzPath)
  );

  // --- results / UI state (Zustand) ---
  const applyResult = useResultsStore((s) => s.applyResult);
  const setApplyResult = useResultsStore((s) => s.setApplyResult);

  const isRunning = useResultsStore((s) => s.productionIsRunning);
  const setIsRunning = useResultsStore((s) => s.setProductionIsRunning);

  const error = useResultsStore((s) => s.productionError);
  const setError = useResultsStore((s) => s.setProductionError);

  const hasModel = Boolean(artifact);
  const canRun = hasModel && dataReady && !isRunning;

  const buildDataPayload = () => ({
    // Mirrors DataInspectRequest fields
    x_path: xPath || null,
    y_path: yPath || null,
    npz_path: npzPath || null,
    x_key: xKey || 'X',
    y_key: yKey || 'y',
  });

  const handleRun = async () => {
    if (!artifact) return;
    setIsRunning(true);
    setError(null);
    try {
      const dataPayload = buildDataPayload();
      const resp = await applyModelToData({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        data: dataPayload,
      });
      setApplyResult(resp);
    } catch (e) {
      console.error(e);
      setError(e?.message || 'Prediction failed');
      setApplyResult(null);
    } finally {
      setIsRunning(false);
    }
  };

  const handleExport = async () => {
    if (!artifact) return;
    try {
      setError(null);
      const dataPayload = buildDataPayload();
      const suggestedName = 'predictions.csv';
      const { blob, filename } = await exportPredictions({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        data: dataPayload,
        filename: suggestedName,
      });

      const supported =
        typeof window !== 'undefined' && 'showSaveFilePicker' in window;
      if (supported) {
        await saveBlobInteractive(blob, filename);
      } else {
        downloadBlob(blob, filename);
      }
    } catch (e) {
      console.error(e);
      setError(e?.message || 'Export failed');
    }
  };

  return (
    <Card shadow="sm" padding="md" radius="md" withBorder>
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
            Keys: X = {xKey || 'X'}, y = {yKey || 'y'}
          </Text>
        </Stack>

        {error && (
          <Alert color="red" variant="light" radius="md">
            <Text size="xs">{error}</Text>
          </Alert>
        )}

        <Group justify="space-between">
          <Button size="xs" onClick={handleRun} disabled={!canRun}>
            {isRunning ? 'Running…' : 'Run prediction'}
          </Button>
          <Button
            size="xs"
            variant="outline"
            onClick={handleExport}
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
