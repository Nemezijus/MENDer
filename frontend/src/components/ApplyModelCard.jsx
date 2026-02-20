import {
  Card,
  Stack,
  Text,
  Group,
  Button,
  Alert,
  Table,
  Badge,
  ScrollArea,
  Tooltip,
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
import { useFilesConstraintsQuery } from '../state/useFilesConstraintsQuery.js';
import { compactPayload } from '../utils/compactPayload.js';

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

function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

function fmt3(v) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  return '—';
}

function pickPreviewColumns(rows) {
  if (!rows || rows.length === 0) return [];
  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const preferred = [
    'index',
    'trial_id',
    'y_true',
    'y_pred',
    'correct',
    'residual',
    'abs_error',
    'margin',
    'decoder_score',
  ];

  // Match the decoder table / export ordering: scores first, then probabilities.
  const scoreCols = [...keys].filter((k) => k.startsWith('score_')).sort();
  const pCols = [...keys].filter((k) => k.startsWith('p_')).sort();
  const rest = [...keys]
    .filter((k) => !preferred.includes(k) && !k.startsWith('p_') && !k.startsWith('score_'))
    .sort();

  const out = [];
  preferred.forEach((k) => keys.has(k) && out.push(k));
  out.push(...scoreCols, ...pCols, ...rest);
  return out;
}

function prettifyHeader(key) {
  if (!key) return '';
  if (key.startsWith('p_')) return `p=${key.slice(2)}`;
  if (key.startsWith('score_')) return `score=${key.slice(6)}`;
  return key.replaceAll('_', ' ');
}

function buildHeaderTooltip(key) {
  if (key === 'index') return 'Row index within the preview.';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth label/value for this sample (if provided).';
  if (key === 'y_pred') return 'Model-predicted label/value for this sample.';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'residual') return 'Residual = (y_pred − y_true). Only present when y_true is provided (regression).';
  if (key === 'abs_error') return 'Absolute error = |y_pred − y_true|. Only present when y_true is provided (regression).';
  if (key === 'decoder_score')
    return 'Binary decision value (decision_function). More positive typically means stronger evidence for the positive class.';
  if (key === 'margin')
    return 'Confidence proxy: usually (top score − runner-up score). Larger margin = more confident.';
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score (logit/decision value) for class ${c}. Higher score usually means higher probability.`;
  }
  return null;
}

function PredictionsPreview({ applyResult }) {
  if (!applyResult) return null;

  const { n_samples, n_features, task, metric_name, metric_value, preview, decoder_outputs } = applyResult;

  // Prefer decoder preview rows when available so the production preview matches
  // the decoder outputs/export columns (score_*, p_*, margin, etc.).
  const decoderPreview = Array.isArray(decoder_outputs?.preview_rows)
    ? decoder_outputs.preview_rows
    : [];
  const basePreview = Array.isArray(preview) ? preview : [];

  const rows = decoderPreview.length > 0 ? decoderPreview : basePreview;
  const columnsRaw = pickPreviewColumns(rows);

  // Hide columns that are entirely empty in the preview (common in classification: residual/abs_error).
  const alwaysKeep = new Set(['index', 'trial_id', 'y_true', 'y_pred', 'correct']);
  const columns = columnsRaw.filter((c) => {
    if (alwaysKeep.has(c)) return true;
    return rows.some((r) => {
      const v = r?.[c];
      return v !== null && v !== undefined && String(v) !== '';
    });
  });

  const renderCell = (col, value) => {
    if (value === null || value === undefined) return '—';

    if (col === 'correct') {
      const isTrue = value === true || value === 'true';
      return isTrue ? 'true' : 'false';
    }

    const num = parseNumber(value);
    if (num !== null) return fmt3(num);

    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
  };

  const stickyThStyle = {
    position: 'sticky',
    top: 0,
    zIndex: 2,
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
  };

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

      {rows.length > 0 && columns.length > 0 && (
        <ScrollArea h={240} type="auto">
          <Table withTableBorder={false} withColumnBorders={false} horizontalSpacing="xs" verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                {columns.map((c) => {
                  const tip = buildHeaderTooltip(c);
                  const label = prettifyHeader(c);
                  return (
                    <Table.Th key={c} style={stickyThStyle}>
                      {tip ? (
                        <Tooltip label={tip} multiline maw={260} withArrow>
                          <Text size="xs" fw={600} c="white">
                            {label}
                          </Text>
                        </Tooltip>
                      ) : (
                        <Text size="xs" fw={600} c="white">
                          {label}
                        </Text>
                      )}
                    </Table.Th>
                  );
                })}
              </Table.Tr>
            </Table.Thead>

            <Table.Tbody>
              {rows.map((row, idx) => {
                const isStriped = idx % 2 === 1;
                return (
                  <Table.Tr
                    key={row?.index ?? idx}
                    style={{
                      backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
                    }}
                  >
                    {columns.map((c) => {
                      const val = row?.[c];
                      const isCorrectCol = c === 'correct';
                      const isFalse = isCorrectCol && (val === false || val === 'false');

                      return (
                        <Table.Td
                          key={c}
                          style={{
                            textAlign: 'center',
                            backgroundColor: isFalse ? 'var(--mantine-color-red-1)' : undefined,
                          }}
                        >
                          <Text size="sm">{renderCell(c, val)}</Text>
                        </Table.Td>
                      );
                    })}
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
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

  const { data: filesConstraints } = useFilesConstraintsQuery();
  const defaultXKey = filesConstraints?.data_default_keys?.x_key ?? 'X';
  const defaultYKey = filesConstraints?.data_default_keys?.y_key ?? 'y';
  const displayXKey = xKey?.trim() || defaultXKey;
  const displayYKey = yKey?.trim() || defaultYKey;

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

  const buildDataPayload = () =>
    compactPayload({
      // Mirrors DataInspectRequest fields
      x_path: xPath || null,
      y_path: yPath || null,
      npz_path: npzPath || null,
      x_key: xKey?.trim() || undefined,
      y_key: yKey?.trim() || undefined,
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
            Keys: X = {displayXKey}, y = {displayYKey}
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
