import { useMemo, useState } from 'react';
import {
  Card,
  Stack,
  Group,
  Text,
  Alert,
  Table,
  ScrollArea,
  Tooltip,
  Button,
  Divider,
} from '@mantine/core';

import { downloadBlob, exportDecoderOutputs } from '../api/models.js';

function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

function fmt3(v) {
  const num = parseNumber(v);
  if (num === null) return v == null ? '—' : String(v);
  if (Number.isInteger(num)) return String(num);
  return num.toFixed(3);
}

function fmtCell(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (Array.isArray(v) || (typeof v === 'object' && v !== null)) {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }
  return fmt3(v);
}

function titleCaseFromKey(key) {
  if (!key) return '';
  const map = {
    n_clusters: 'Number of clusters',
    n_noise: 'Noise points',
    noise_ratio: 'Noise ratio',
    cluster_sizes: 'Cluster sizes',
    inertia: 'Inertia',
    n_iter: 'Iterations',
    aic: 'AIC',
    bic: 'BIC',
    mean_log_likelihood: 'Mean log-likelihood',
    std_log_likelihood: 'Std log-likelihood',
    converged: 'Converged',
    lower_bound: 'Lower bound',
    embedding_2d: '2D embedding',
    silhouette: 'Silhouette',
    davies_bouldin: 'Davies–Bouldin',
    calinski_harabasz: 'Calinski–Harabasz',
  };
  if (Object.prototype.hasOwnProperty.call(map, key)) return map[key];
  return key
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function tooltipForKey(key) {
  const tips = {
    silhouette:
      'Silhouette score: how well-separated clusters are (higher is better). Only defined when there are at least 2 clusters and no degenerate cases.',
    davies_bouldin:
      'Davies–Bouldin index: average similarity between each cluster and its most similar one (lower is better).',
    calinski_harabasz:
      'Calinski–Harabasz index: ratio of between-cluster dispersion to within-cluster dispersion (higher is better).',
    n_clusters: 'Number of non-noise clusters. (Noise label -1 is excluded.)',
    n_noise: 'Number of points labeled as noise (-1), if the algorithm supports noise (e.g., DBSCAN).',
    noise_ratio: 'Fraction of points labeled as noise (-1).',
    cluster_sizes: 'Sample counts per cluster id.',
    inertia:
      'Sum of squared distances of samples to their closest cluster center (KMeans). Lower usually means tighter clusters.',
    aic: 'Akaike Information Criterion (Gaussian mixture models). Lower is better (relative, not absolute).',
    bic: 'Bayesian Information Criterion (Gaussian mixture models). Lower is better (relative, not absolute).',
    mean_log_likelihood:
      'Mean per-sample log-likelihood under the fitted model (mixture models). Higher is better.',
    std_log_likelihood: 'Standard deviation of per-sample log-likelihoods.',
    converged: 'Whether the estimator reports convergence.',
    lower_bound: 'Lower bound on the log-likelihood (mixture models).',
    n_iter: 'Number of iterations run by the estimator.',
    embedding_2d: 'A downsampled 2D embedding (PCA) for plotting.',
  };
  return tips[key] || null;
}

function buildPairsTableRows(pairs) {
  const rows = [];
  for (let i = 0; i < pairs.length; i += 2) {
    const a = pairs[i] || null;
    const b = pairs[i + 1] || null;
    rows.push([a, b]);
  }
  return rows;
}

function isEmptyCell(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === 'string') return v.trim() === '';
  return false;
}

function pickColumns(rows) {
  if (!rows || rows.length === 0) return [];

  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const nonEmptyKeys = new Set(
    [...keys].filter((k) => k === 'index' || rows.some((r) => !isEmptyCell(r?.[k]))),
  );

  const preferred = [
    'index',
    'cluster_id',
    'is_noise',
    'is_core',
    'distance_to_center',
    'max_membership_prob',
    'log_likelihood',
  ];

  const rest = [...nonEmptyKeys]
    .filter((k) => !preferred.includes(k))
    .sort();

  const out = [];
  preferred.forEach((k) => nonEmptyKeys.has(k) && out.push(k));
  out.push(...rest);
  return out;
}

function headerLabel(key) {
  const map = {
    index: 'Index',
    cluster_id: 'Cluster id',
    is_noise: 'Noise',
    is_core: 'Core',
    distance_to_center: 'Distance to center',
    max_membership_prob: 'Max membership prob.',
    log_likelihood: 'Log-likelihood',
  };
  if (Object.prototype.hasOwnProperty.call(map, key)) return map[key];
  return titleCaseFromKey(key);
}

function headerTooltip(key) {
  const map = {
    index: 'Row index within the preview.',
    cluster_id: 'Assigned cluster label for this sample.',
    is_noise: 'Whether this sample was labeled as noise (-1).',
    is_core: 'Whether this sample is a core point (DBSCAN).',
    distance_to_center: 'Distance to the nearest cluster center (KMeans).',
    max_membership_prob: 'Maximum soft membership probability (mixtures).',
    log_likelihood: 'Per-sample log-likelihood under the fitted model (mixtures).',
  };
  return map[key] || tooltipForKey(key);
}

export default function UnsupervisedResultsPanel({ trainResult }) {
  if (!trainResult) return null;

  const artifactUid = trainResult?.artifact?.uid || null;
  const metrics = trainResult.metrics || {};
  const warnings = Array.isArray(trainResult.warnings) ? trainResult.warnings : [];
  const notes = Array.isArray(trainResult.notes) ? trainResult.notes : [];

  const clusterSummary = trainResult.cluster_summary || {};
  const modelDiag = trainResult?.diagnostics?.model_diagnostics || {};
  const embedding2d = trainResult?.diagnostics?.embedding_2d || null;

  const mergedDiagnostics = useMemo(() => {
    const merged = {
      ...clusterSummary,
      ...modelDiag,
    };
    // Do not show redundant / noisy fields
    delete merged.label_summary;
    return merged;
  }, [clusterSummary, modelDiag]);

  const diagPairs = useMemo(() => {
    const pairs = [];
    Object.entries(mergedDiagnostics).forEach(([k, v]) => {
      if (v === null || v === undefined) return;
      pairs.push([k, v]);
    });
    if (embedding2d) {
      pairs.push(['embedding_2d', Array.isArray(embedding2d?.x) ? `${embedding2d.x.length} points` : 'Present']);
    }
    // Sort with a bit of intentional ordering
    const order = [
      'n_clusters',
      'n_noise',
      'noise_ratio',
      'cluster_sizes',
      'inertia',
      'n_iter',
      'converged',
      'lower_bound',
      'aic',
      'bic',
      'mean_log_likelihood',
      'std_log_likelihood',
      'embedding_2d',
    ];
    pairs.sort((a, b) => {
      const ia = order.indexOf(a[0]);
      const ib = order.indexOf(b[0]);
      if (ia !== -1 || ib !== -1) {
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      }
      return a[0].localeCompare(b[0]);
    });
    return pairs;
  }, [mergedDiagnostics, embedding2d]);

  const diagRows = useMemo(() => buildPairsTableRows(diagPairs), [diagPairs]);

  const previewRows = trainResult.unsupervised_outputs?.preview_rows || [];
  const nTotal = trainResult.unsupervised_outputs?.n_rows_total ?? null;

  const previewColumns = useMemo(() => pickColumns(previewRows), [previewRows]);

  const [exportErr, setExportErr] = useState(null);

  const handleExport = async () => {
    if (!artifactUid) return;
    setExportErr(null);
    try {
      const { blob, filename } = await exportDecoderOutputs({
        artifactUid,
        filename: `decoder_outputs_${artifactUid}.csv`,
      });
      downloadBlob(blob, filename);
    } catch (e) {
      setExportErr(e?.message || String(e));
    }
  };

  const metricPairs = useMemo(() => {
    const pairs = [];
    Object.entries(metrics).forEach(([k, v]) => {
      if (v === null || v === undefined) return;
      pairs.push([k, v]);
    });
    const order = ['silhouette', 'davies_bouldin', 'calinski_harabasz'];
    pairs.sort((a, b) => {
      const ia = order.indexOf(a[0]);
      const ib = order.indexOf(b[0]);
      if (ia !== -1 || ib !== -1) {
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      }
      return a[0].localeCompare(b[0]);
    });
    return pairs;
  }, [metrics]);

  return (
    <Stack gap="md">
      <Card withBorder radius="md" shadow="sm" padding="md">
        <Stack gap="sm">
          <Text fw={600} size="xl" align="center">
            Unsupervised diagnostics
          </Text>

          {(warnings.length > 0 || notes.length > 0) && (
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
          )}

          <Stack gap={0}>
            <Text fw={500} size="xl" align="center">
              Summary of metrics
            </Text>

            {metricPairs.length === 0 ? (
              <Text size="sm" c="dimmed">
                No global metrics were returned.
              </Text>
            ) : (
              metricPairs.map(([k, v]) => {
                const tip = tooltipForKey(k);
                const label = titleCaseFromKey(k);
                return (
                  <Text size="sm" key={k}>
                    {tip ? (
                      <Tooltip label={tip} multiline maw={360} withArrow>
                        <Text span fw={500} style={{ width: 'fit-content' }}>
                          {label}:{' '}
                        </Text>
                      </Tooltip>
                    ) : (
                      <Text span fw={500}>
                        {label}:{' '}
                      </Text>
                    )}
                    <Text span fw={700}>
                      {fmt3(v)}
                    </Text>
                  </Text>
                );
              })
            )}
          </Stack>

          <Divider />

          <Stack gap="xs">
            <Text fw={500} size="xl" align="center">
              Cluster summary &amp; diagnostics
            </Text>

            <Table
              withTableBorder={false}
              withColumnBorders={false}
              horizontalSpacing="xs"
              verticalSpacing="xs"
              style={{ tableLayout: 'fixed' }}
            >
              <Table.Tbody>
                {diagRows.length === 0 ? (
                  <Table.Tr>
                    <Table.Td colSpan={4}>
                      <Text size="sm" c="dimmed">
                        No diagnostics returned.
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ) : (
                  diagRows.map(([a, b], i) => (
                    <Table.Tr key={i}>
                      {[a, b].map((pair, j) => {
                        const k = pair?.[0] ?? null;
                        const val = pair?.[1];
                        const label = k ? titleCaseFromKey(k) : '';
                        const tip = k ? tooltipForKey(k) : null;

                        const labelCell = (
                          <Table.Td
                            key={`${i}-${j}-label`}
                            style={{ width: '35%', paddingLeft: j === 0 ? 0 : undefined }}
                          >
                            {k ? (
                              tip ? (
                                <Tooltip label={tip} multiline maw={360} withArrow>
                                  <Text size="sm" c="dimmed" style={{ width: 'fit-content' }}>
                                    {label}
                                  </Text>
                                </Tooltip>
                              ) : (
                                <Text size="sm" c="dimmed">
                                  {label}
                                </Text>
                              )
                            ) : (
                              <Text size="sm" c="dimmed">
                                {' '}
                              </Text>
                            )}
                          </Table.Td>
                        );

                        const valueCell = (
                          <Table.Td
                            key={`${i}-${j}-value`}
                            style={{ width: '15%', paddingRight: j === 1 ? 0 : undefined }}
                          >
                            {k ? (
                              <Text size="sm" fw={700}>
                                {fmtCell(val)}
                              </Text>
                            ) : (
                              <Text size="sm">{' '}</Text>
                            )}
                          </Table.Td>
                        );

                        return [labelCell, valueCell];
                      })}
                    </Table.Tr>
                  ))
                )}
              </Table.Tbody>
            </Table>
          </Stack>
        </Stack>
      </Card>

      <Card withBorder radius="md" shadow="sm" padding="md">
        <Stack gap="sm">
          <Group justify="space-between" wrap="wrap">
            <Text fw={600} size="xl">
              Decoder outputs
            </Text>
            <Group gap="xs" wrap="wrap">
              <Text size="sm" c="dimmed">
                Preview: {previewRows.length} / {nTotal ?? '...'}
              </Text>
              <Button size="xs" variant="light" onClick={handleExport} disabled={!artifactUid}>
                Export to CSV
              </Button>
            </Group>
          </Group>

          {exportErr ? (
            <Alert color="red" title="Export failed">
              <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                {exportErr}
              </Text>
            </Alert>
          ) : null}

          <ScrollArea h={360} type="auto">
            <Table striped highlightOnHover withTableBorder withColumnBorders>
              <Table.Thead>
                <Table.Tr>
                  {previewColumns.map((c) => {
                    const tip = headerTooltip(c);
                    const label = headerLabel(c);
                    return (
                      <Table.Th key={c} style={{ whiteSpace: 'nowrap' }}>
                        {tip ? (
                          <Tooltip label={tip} multiline maw={360} withArrow>
                            <Text size="xs" fw={600} style={{ width: 'fit-content' }}>
                              {label}
                            </Text>
                          </Tooltip>
                        ) : (
                          <Text size="xs" fw={600}>
                            {label}
                          </Text>
                        )}
                      </Table.Th>
                    );
                  })}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {previewRows.map((r, i) => (
                  <Table.Tr key={i}>
                    {previewColumns.map((c) => (
                      <Table.Td key={c}>{fmtCell(r?.[c])}</Table.Td>
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
