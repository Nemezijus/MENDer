import {
  Card,
  Stack,
  Group,
  Text,
  Button,
  Divider,
  Alert,
  ScrollArea,
  Table,
  Code,
} from '@mantine/core';

import { exportDecoderOutputs, saveModel, saveBlobInteractive } from '../api/models.js';

function kvRows(obj) {
  if (!obj || typeof obj !== 'object') return [];
  return Object.entries(obj).map(([k, v]) => [String(k), v]);
}

function fmt(v) {
  if (v == null) return '';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    return Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-4)
      ? v.toExponential(4)
      : v.toFixed(4);
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) return JSON.stringify(v);
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function buildPreviewColumns(previewRows) {
  if (!Array.isArray(previewRows) || previewRows.length === 0) return ['index', 'cluster_id'];
  const keySet = new Set();
  // keep stable order
  ['index', 'cluster_id'].forEach((k) => keySet.add(k));
  for (const row of previewRows) {
    if (!row || typeof row !== 'object') continue;
    for (const k of Object.keys(row)) keySet.add(k);
  }
  return Array.from(keySet);
}

export default function UnsupervisedResultsPanel({ result, artifact }) {
  if (!result) return null;

  const metrics = result.metrics || {};
  const warnings = Array.isArray(result.warnings) ? result.warnings : [];
  const notes = Array.isArray(result.notes) ? result.notes : [];
  const clusterSummary = result.cluster_summary || {};
  const diagnostics = result.diagnostics || {};

  const previewRows = result.unsupervised_outputs?.preview_rows || [];
  const nTotal = result.unsupervised_outputs?.n_rows_total ?? null;
  const previewCols = buildPreviewColumns(previewRows);

  const canExport = !!artifact?.uid;

  const handleExportOutputs = async () => {
    if (!artifact?.uid) return;
    const { blob, filename } = await exportDecoderOutputs({
      artifactUid: artifact.uid,
      filename: `unsupervised_outputs_${artifact.uid}.csv`,
    });
    // prefer interactive save where possible
    const ok = await saveBlobInteractive(blob, filename);
    if (ok === false) return;
  };

  const handleSaveModel = async () => {
    if (!artifact?.uid) return;
    const { blob, filename } = await saveModel({
      artifactUid: artifact.uid,
      artifactMeta: artifact,
      filename: `model_${artifact.uid}.mend`,
    });
    const ok = await saveBlobInteractive(blob, filename);
    if (ok === false) return;
  };

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="xs">
          <Text fw={700} size="lg" align="center">
            Unsupervised Results
          </Text>

          <Group justify="space-between" wrap="wrap">
            <Text size="sm">
              Train samples: <b>{result.n_train}</b> &nbsp;|&nbsp; Features:{' '}
              <b>{result.n_features}</b>
              {result.n_apply != null ? (
                <>
                  &nbsp;|&nbsp; Apply samples: <b>{result.n_apply}</b>
                </>
              ) : null}
            </Text>

            <Group gap="xs">
              <Button size="xs" variant="light" onClick={handleSaveModel} disabled={!canExport}>
                Save model (.mend)
              </Button>
              <Button size="xs" variant="light" onClick={handleExportOutputs} disabled={!canExport}>
                Export outputs CSV
              </Button>
            </Group>
          </Group>
        </Stack>
      </Card>

      {(warnings.length > 0 || notes.length > 0) && (
        <Card withBorder shadow="sm" radius="md" padding="lg">
          <Stack gap="sm">
            {warnings.length > 0 && (
              <Alert color="yellow" title="Warnings">
                <Stack gap={4}>
                  {warnings.map((w, i) => (
                    <Text key={i} size="sm">
                      {w}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}
            {notes.length > 0 && (
              <Alert color="blue" title="Notes">
                <Stack gap={4}>
                  {notes.map((n, i) => (
                    <Text key={i} size="sm">
                      {n}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}
          </Stack>
        </Card>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Text fw={600}>Global metrics</Text>
          <Table striped highlightOnHover withTableBorder withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Metric</Table.Th>
                <Table.Th>Value</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {Object.keys(metrics).length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={2}>
                    <Text size="sm" c="dimmed">
                      No metrics returned.
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : (
                Object.entries(metrics).map(([k, v]) => (
                  <Table.Tr key={k}>
                    <Table.Td>
                      <Code>{k}</Code>
                    </Table.Td>
                    <Table.Td>{fmt(v)}</Table.Td>
                  </Table.Tr>
                ))
              )}
            </Table.Tbody>
          </Table>

          <Divider />

          <Text fw={600}>Cluster summary</Text>
          <Table striped highlightOnHover withTableBorder withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Field</Table.Th>
                <Table.Th>Value</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {kvRows(clusterSummary).map(([k, v]) => (
                <Table.Tr key={k}>
                  <Table.Td>
                    <Code>{k}</Code>
                  </Table.Td>
                  <Table.Td>{fmt(v)}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>

          <Divider />

          <Text fw={600}>Diagnostics</Text>
          <Table striped highlightOnHover withTableBorder withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Field</Table.Th>
                <Table.Th>Value</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {kvRows(diagnostics?.model_diagnostics || {}).map(([k, v]) => (
                <Table.Tr key={k}>
                  <Table.Td>
                    <Code>{k}</Code>
                  </Table.Td>
                  <Table.Td>{fmt(v)}</Table.Td>
                </Table.Tr>
              ))}
              {diagnostics?.embedding_2d ? (
                <Table.Tr>
                  <Table.Td>
                    <Code>embedding_2d</Code>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c="dimmed">
                      Present ({Array.isArray(diagnostics.embedding_2d) ? diagnostics.embedding_2d.length : 'n/a'} rows)
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : null}
            </Table.Tbody>
          </Table>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="sm">
          <Group justify="space-between">
            <Text fw={600}>Per-sample outputs preview</Text>
            <Text size="sm" c="dimmed">
              Showing {previewRows.length} / {nTotal ?? '...'}
            </Text>
          </Group>

          <ScrollArea h={360} type="auto">
            <Table striped highlightOnHover withTableBorder withColumnBorders>
              <Table.Thead>
                <Table.Tr>
                  {previewCols.map((c) => (
                    <Table.Th key={c}>{c}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {previewRows.map((r, i) => (
                  <Table.Tr key={i}>
                    {previewCols.map((c) => (
                      <Table.Td key={c}>{fmt(r?.[c])}</Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Stack>
      </Card>
    </Stack>
  );
}
