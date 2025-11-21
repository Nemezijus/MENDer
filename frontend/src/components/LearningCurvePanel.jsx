import { useEffect, useState } from 'react';
import { Card, Button, Text, Stack, Group, Divider, Alert, Title, Box, NumberInput } from '@mantine/core';

import { useDataCtx } from '../state/DataContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import FeatureCard from './FeatureCard.jsx';
import ScalingCard from './ScalingCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import MetricCard from './MetricCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';
import api from '../api/client';
import { useLearningCurveResultsCtx } from '../state/LearningCurveResultsContext.jsx';

// NEW: centralized defaults/enums/meta
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';

export default function LearningCurvePanel() {
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();

  const {
    method, pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  const { nSplits, setNSplits, withinPct, setWithinPct, setResult } = useLearningCurveResultsCtx();

  // centralized schema/defaults/enums
  const { loading: defsLoading, models, enums, getModelDefaults } = useSchemaDefaults();

  // split
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42);

  // scale/metric
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');

  // union model (hydrated from backend defaults)
  const [model, setModel] = useState(null);

  const [trainSizesCSV, setTrainSizesCSV] = useState('');
  const [nSteps, setNSteps] = useState(5);
  const [nJobs, setNJobs] = useState(1);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // initialize once defaults arrive
  useEffect(() => {
    if (!defsLoading && !model) {
      const init = getModelDefaults('logreg') || { algo: 'logreg' };
      setModel(init);
    }
  }, [defsLoading, getModelDefaults, model]);

  // on algo change, rehydrate from backend defaults while keeping edits
  useEffect(() => {
    if (!model) return;
    const base = getModelDefaults(model.algo) || { algo: model.algo };
    setModel((cur) => ({ ...base, ...cur }));
  }, [getModelDefaults, model?.algo]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the left sidebar.');
      return;
    }
    if (!model) return;

    setErr(null);
    setResult(null);
    setLoading(true);

    try {
      const train_sizes = trainSizesCSV
        ? trainSizesCSV.split(',').map(s => s.trim()).filter(Boolean)
            .map(x => (x.includes('.') ? parseFloat(x) : parseInt(x, 10)))
        : null;

      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: { mode: 'kfold', n_splits: nSplits, stratified, shuffle },
        scale: { method: scaleMethod },
        features: {
          method,
          pca_n, pca_var, pca_whiten,
          lda_n, lda_solver, lda_shrinkage, lda_tol,
          sfs_k: (sfs_k === '' || sfs_k == null) ? 'auto' : sfs_k,
          sfs_direction, sfs_cv, sfs_n_jobs,
        },
        model, // union payload as-is
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
        },
        train_sizes,
        n_steps: Number(nSteps),
        n_jobs: Number(nJobs),
      };

      const { data } = await api.post('/learning-curve', payload);
      setResult(data);
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e.message ?? String(e);
      const msg = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || !models || !model) {
    return null; // optionally render a skeleton/spinner
  }

  return (
    <Stack gap="md">
      <Title order={3}>Learning Curve</Title>

      {err && (
        <Alert color="red" variant="light">
          <Text fw={500}>Error</Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button size="xs" onClick={handleCompute} loading={loading} disabled={!dataReady}>
              {loading ? 'Computing…' : 'Compute'}
            </Button>
          </Group>

          <Box w="100%" style={{ maxWidth: 560 }}>
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
                nSplits={nSplits}
                onNSplitsChange={setNSplits}
                stratified={stratified}
                onStratifiedChange={setStratified}
                shuffle={shuffle}
                onShuffleChange={setShuffle}
                seed={seed}
                onSeedChange={setSeed}
              />

              <Divider my="xs" />

              <ScalingCard value={scaleMethod} onChange={setScaleMethod} />

              <FeatureCard title="Features" />

              <Divider my="xs" />

              <MetricCard value={metric} onChange={setMetric} />

              <Divider my="xs" />

              <ModelSelectionCard
                model={model}
                onChange={(next) => {
                  if (next?.algo && next.algo !== model.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setModel({ ...d, ...next });
                  } else {
                    setModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}   // <-- pass defaults + meta so filtering by task works
              />

              <Divider my="xs" />

              <NumberInput
                label="Steps (used if Train sizes empty)"
                min={2}
                max={50}
                step={1}
                value={nSteps}
                onChange={setNSteps}
              />
              <NumberInput
                label="n_jobs"
                min={1}
                step={1}
                value={nJobs}
                onChange={setNJobs}
              />
              <Text size="sm" c="dimmed">
                Optional Train sizes (CSV): fractions in (0,1] or absolute integers. Example:
                <Text span fw={500}> 0.1,0.3,0.5,0.7,1.0 </Text> or <Text span fw={500}> 50,100,200 </Text>
              </Text>
              <textarea
                style={{ width: '100%', minHeight: 70, fontFamily: 'inherit', fontSize: '0.9rem' }}
                placeholder="e.g. 0.1,0.3,0.5,0.7,1.0"
                value={trainSizesCSV}
                onChange={(e) => setTrainSizesCSV(e.currentTarget.value)}
              />

              <Divider my="xs" />

              <NumberInput
                label="Recommend the smallest train size achieving at least this fraction of the peak validation score"
                description="e.g., 0.99 = within 1% of peak"
                min={0.5}
                max={1.0}
                step={0.01}
                value={withinPct}
                onChange={setWithinPct}
                precision={2}
              />
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear in the right-hand “Learning Curve Results” panel.
            </Text>
          )}
        </Stack>
      </Card>
    </Stack>
  );
}
