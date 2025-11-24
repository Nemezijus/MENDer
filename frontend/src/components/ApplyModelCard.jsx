import React, { useCallback } from 'react';
import {
  Card,
  Stack,
  Text,
  Group,
  Button,
  FileInput,
  Alert,
  Table,
  Badge,
} from '@mantine/core';

import { useModelArtifact } from '../state/ModelArtifactContext.jsx';
import { useProductionDataCtx } from '../state/ProductionDataContext.jsx';
import { useProductionResultsCtx } from '../state/ProductionResultsContext.jsx';

import { uploadFile } from '../api/files';
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
  const { artifact } = useModelArtifact();
  const {
    xPath,
    setXPath,
    yPath,
    setYPath,
    npzPath,
    setNpzPath,
    xKey,
    yKey,
    dataReady,
  } = useProductionDataCtx();
  const {
    applyResult,
    setApplyResult,
    isRunning,
    setIsRunning,
    error,
    setError,
  } = useProductionResultsCtx();

  const hasModel = Boolean(artifact);
  const canRun = hasModel && dataReady && !isRunning;

  const handleUploadX = useCallback(
    async (file) => {
      if (!file) return;
      try {
        setError(null);
        const res = await uploadFile(file);
        // Assuming backend returns { path, original_name, ... }
        if (res.npz_path) {
          setNpzPath(res.npz_path);
          setXPath('');
        } else if (res.path) {
          setXPath(res.path);
          setNpzPath('');
        }
      } catch (e) {
        console.error(e);
        setError(e?.message || 'Failed to upload X file');
      }
    },
    [setXPath, setNpzPath, setError]
  );

  const handleUploadY = useCallback(
    async (file) => {
      if (!file) return;
      try {
        setError(null);
        const res = await uploadFile(file);
        if (res.path) {
          setYPath(res.path);
        }
      } catch (e) {
        console.error(e);
        setError(e?.message || 'Failed to upload y file');
      }
    },
    [setYPath, setError]
  );

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
          <FileInput
            label="Features (X)"
            placeholder="Upload features file"
            onChange={handleUploadX}
            disabled={!hasModel || isRunning}
            accept=".mat,.npz,.npy,.csv,.txt"
            size="xs"
          />
          {xPath || npzPath ? (
            <Text size="xs" c="dimmed">
              Saved X as: {npzPath || xPath}
            </Text>
          ) : null}
          <FileInput
            label="Labels (y, optional)"
            placeholder="Upload labels file"
            onChange={handleUploadY}
            disabled={!hasModel || isRunning}
            accept=".mat,.npz,.npy,.csv,.txt"
            size="xs"
          />
          {yPath ? (
            <Text size="xs" c="dimmed">
              Saved y as: {yPath}
            </Text>
          ) : null}
          <Text size="xs" c="dimmed">
            X is required. y is optional; if provided, an evaluation metric will be
            computed.
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
