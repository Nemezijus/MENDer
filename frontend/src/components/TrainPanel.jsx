// src/components/TrainPanel.jsx
import React, { useState } from 'react';
import {
  Card, Button, Select, Checkbox, NumberInput, Text,
  Stack, Group, Divider, Loader, Alert, Title, Box
} from '@mantine/core';
import { useDataCtx } from '../state/DataContext.jsx';
import { runTrainRequest } from '../api/train';
import ConfusionTable from './ConfusionTable.jsx';

export default function TrainPanel() {
  // Shared data (from sidebar)
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();

  // Hold-out split + pipeline params (same as before)
  const [trainFrac, setTrainFrac] = useState(0.75);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [metric, setMetric] = useState('accuracy');
  const [seed, setSeed] = useState(42);

  const [scaleMethod, setScaleMethod] = useState('standard');
  const [featureMethod, setFeatureMethod] = useState('pca');

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null);

  async function handleRun() {
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
        split: { mode: 'holdout', train_frac: trainFrac, stratified, shuffle },
        scale: { method: scaleMethod },
        features: {
          method: featureMethod,
          pca_n: null, pca_var: 0.95, pca_whiten: false,
          lda_n: null, lda_solver: 'svd', lda_shrinkage: null, lda_tol: 1e-4,
          sfs_k: 'auto', sfs_direction: 'backward', sfs_cv: 5, sfs_n_jobs: null,
        },
        model: { algo: 'logreg', C: 1.0, penalty: 'l2', solver: 'lbfgs', max_iter: 1000, class_weight: null },
        eval: { metric, seed: seed === '' ? null : parseInt(seed, 10) },
      };

      const data = await runTrainRequest(payload);
      setResult(data);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="lg" maw={600}>
      <Title order={3}>Hold-out</Title>

      {err && <Alert color="red" title="Error" variant="light"><Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text></Alert>}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button size="xs" onClick={handleRun} loading={loading} disabled={!dataReady}>
              {loading ? 'Running…' : 'Run'}
            </Button>
          </Group>

          <Box w="100%" style={{ maxWidth: 420 }}>
            <Stack gap="sm">
              <NumberInput label="Train fraction" min={0.5} max={0.95} step={0.05} value={trainFrac} onChange={setTrainFrac} />
              <Checkbox label="Stratified" checked={stratified} onChange={(e) => setStratified(e.currentTarget.checked)} />
              <Checkbox label="Shuffle" checked={shuffle} onChange={(e) => setShuffle(e.currentTarget.checked)} />
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
              <NumberInput label="Seed" value={seed} onChange={setSeed} allowDecimal={false} disabled={!shuffle} />
              <Divider my="xs" />
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
              <Select
                label="Feature method"
                data={[
                  { value: 'none', label: 'none' },
                  { value: 'pca', label: 'pca' },
                  { value: 'lda', label: 'lda' },
                  { value: 'sfs', label: 'sfs' },
                ]}
                value={featureMethod}
                onChange={setFeatureMethod}
              />
            </Stack>
          </Box>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        {loading && <Group align="center" gap="sm"><Loader size="sm" /><Text size="sm">Running…</Text></Group>}
        {!loading && !result && <Text size="sm" c="dimmed">Run to see results.</Text>}
        {!loading && result && (
          <Stack gap="xs">
            <Text size="sm"><Text span fw={500}>Metric:</Text> {result.metric_name}</Text>
            <Text size="sm"><Text span fw={500}>Score:</Text> {result.metric_value?.toFixed?.(4) ?? result.metric_value}</Text>
            <Text size="sm"><Text span fw={500}>Train/Test:</Text> {result.n_train} / {result.n_test}</Text>
            {result.confusion?.matrix && result.confusion?.labels && (
              <>
                <Text fw={500} size="sm">Confusion matrix</Text>
                <ConfusionTable
                  labels={result.confusion.labels}
                  matrix={result.confusion.matrix}
                />
              </>
            )}
          </Stack>
        )}
      </Card>
    </Stack>
  );
}
