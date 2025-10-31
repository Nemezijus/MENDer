// src/components/CrossValPanel.jsx
import React, { useState } from 'react';
import {
  Card, Button, Select, Checkbox, NumberInput, Text,
  Stack, Group, Divider, Loader, Alert, Title, Box, Table
} from '@mantine/core';
import Plot from 'react-plotly.js';
import { useDataCtx } from '../state/DataContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import FeatureCard from './FeatureCard.jsx';
import { runCrossvalRequest } from '../api/cv';

export default function CrossValPanel() {
  // Shared data (from sidebar)
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();

  // Shared feature config (from FeatureContext)
  const {
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  // CV + pipeline params
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');
  const [nSplits, setNSplits] = useState(5);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42);

  // runtime
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null);

  async function handleRunCV() {
    if (!dataReady) { setErr('Load & inspect data first in the left sidebar.'); return; }
    setErr(null);
    setResult(null);
    setLoading(true);

    try {
      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: {
          mode: 'kfold',
          n_splits: nSplits,
          stratified,
          shuffle,
        },
        scale: { method: scaleMethod },
        features: {
          method,
          // PCA
          pca_n,
          pca_var,
          pca_whiten,
          // LDA
          lda_n,
          lda_solver,
          lda_shrinkage,
          lda_tol,
          // SFS
          sfs_k,
          sfs_direction,
          sfs_cv,
          sfs_n_jobs,
        },
        model: {
          algo: 'logreg',
          C: 1.0,
          penalty: 'l2',
          solver: 'lbfgs',
          max_iter: 1000,
          class_weight: null,
        },
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
        },
      };

      const data = await runCrossvalRequest(payload);
      setResult(data);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  function ResultView() {
    if (!result) return null;

    const folds = result.fold_scores || [];
    const idxs = folds.map((_, i) => i + 1);

    return (
      <Stack gap="md">
        <Text size="sm">
          <Text span fw={500}>Metric:</Text> {result.metric_name}
        </Text>
        <Text size="sm">
          <Text span fw={500}>Mean ± Std:</Text> {result.mean_score.toFixed(4)} ± {result.std_score.toFixed(4)}
        </Text>
        <Text size="sm">
          <Text span fw={500}>n_splits:</Text> {result.n_splits}
        </Text>

        {/* Fold scores table */}
        <Table striped highlightOnHover withTableBorder withColumnBorders maw={420}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Fold</Table.Th>
              <Table.Th>Score</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {folds.map((s, i) => (
              <Table.Tr key={i}>
                <Table.Td>{i + 1}</Table.Td>
                <Table.Td>{typeof s === 'number' ? s : JSON.stringify(s)}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>

        {/* Bar chart with mean line */}
        <Plot
          data={[
            { type: 'bar', x: idxs, y: folds, name: 'Fold score' },
            { type: 'scatter', mode: 'lines', x: [0, idxs.length + 1], y: [result.mean_score, result.mean_score], name: 'Mean' },
          ]}
          layout={{
            title: 'Fold scores',
            margin: { l: 40, r: 10, t: 40, b: 40 },
            xaxis: { title: 'Fold' },
            yaxis: { title: result.metric_name },
            autosize: true,
          }}
          style={{ width: '100%', height: 300 }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </Stack>
    );
  }

  return (
    <Stack gap="lg" maw={700}>
      <Title order={3}>Cross-Validation</Title>

      {err && (
        <Alert color="red" title="Error" variant="light">
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button size="xs" onClick={handleRunCV} loading={loading} disabled={!dataReady}>
              {loading ? 'Running…' : 'Run CV'}
            </Button>
          </Group>

          <Box w="100%" style={{ maxWidth: 460 }}>
            <Stack gap="sm">
              <Select
                label="Scale method"
                data={[
                  { value: 'standard', label: 'standard' },
                  { value: 'robust', label: 'robust' },
                  { value: 'minmax', label: 'minmax' },
                  { value: 'maxabs', label: 'maxabs' },
                  { value: 'quantile', label: 'quantile' },
                  { value: 'none', label: 'none' },
                ]}
                value={scaleMethod}
                onChange={setScaleMethod}
              />

              {/* Shared Feature card */}
              <FeatureCard title="Features" />

              <Select
                label="Metric"
                data={[
                  { value: 'accuracy', label: 'accuracy' },
                  { value: 'balanced_accuracy', label: 'balanced_accuracy' },
                  { value: 'f1_macro', label: 'f1_macro' },
                ]}
                value={metric}
                onChange={setMetric}
              />

              <NumberInput label="n_splits" min={2} max={20} step={1} value={nSplits} onChange={setNSplits} />
              <Checkbox label="Stratified" checked={stratified} onChange={(e) => setStratified(e.currentTarget.checked)} />
              <Checkbox label="Shuffle" checked={shuffle} onChange={(e) => setShuffle(e.currentTarget.checked)} />
              <NumberInput label="Seed" value={seed} onChange={setSeed} allowDecimal={false} disabled={!shuffle} />
            </Stack>
          </Box>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        {loading && (
          <Group align="center" gap="sm">
            <Loader size="sm" />
            <Text size="sm">Running cross-validation…</Text>
          </Group>
        )}
        {!loading && !result && (
          <Text size="sm" c="dimmed">Run CV to see fold scores.</Text>
        )}
        {!loading && result && <ResultView />}
      </Card>
    </Stack>
  );
}
