import { useEffect, useState } from 'react';
import {
  Card,
  Button,
  Text,
  Stack,
  Group,
  Divider,
  Alert,
  Title,
  Box,
  NumberInput,
} from '@mantine/core';

import { useDataStore } from '../state/useDataStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useTuningStore } from '../state/useTuningStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';

import SplitOptionsCard from './SplitOptionsCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import { requestRandomSearch } from '../api/tuning';
import RandomSearchResultsPanel from './visualizations/RandomSearchResultsPanel.jsx';
import HyperparameterSelector from './helpers/HyperparameterSelector.jsx';

const EMPTY_PARAM = { paramName: '', values: [] };

export default function RandomSearchPanel() {
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;

  const {
    method,
    pca_n,
    pca_var,
    pca_whiten,
    lda_n,
    lda_solver,
    lda_shrinkage,
    lda_tol,
    sfs_k,
    sfs_direction,
    sfs_cv,
    sfs_n_jobs,
  } = useFeatureStore();

  const { loading: defsLoading, models, enums, getModelDefaults } = useSchemaDefaults();

  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  const rsState = useTuningStore((s) => s.randomSearch);
  const setRsState = useTuningStore((s) => s.setRandomSearch);

  const {
    nSplits,
    stratified,
    shuffle,
    seed,
    nJobs,
    nIter,
    result: randomResult,
  } = rsState;

  const hyperParam1 = rsState.hyperParam1 ?? EMPTY_PARAM;
  const hyperParam2 = rsState.hyperParam2 ?? EMPTY_PARAM;

  const handleHyperParam1Change = (next) =>
    setRsState({ hyperParam1: next });

  const handleHyperParam2Change = (next) =>
    setRsState({ hyperParam2: next });

  // Per-panel model config (random search slice)
  const rsModel = useModelConfigStore((s) => s.randomSearch);
  const setRsModel = useModelConfigStore((s) => s.setRandomSearchModel);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const taskInferred = inspectReport?.task_inferred || null;
  const defaultMetric = taskInferred === 'regression' ? 'r2' : 'accuracy';
  const effectiveMetric = metric || defaultMetric;

  useEffect(() => {
    if (!metric && taskInferred) {
      setMetric(defaultMetric);
    }
  }, [metric, taskInferred, defaultMetric, setMetric]);

  // Initialize RS model once
  useEffect(() => {
    if (!defsLoading && !rsModel) {
      const init = getModelDefaults('logreg') || { algo: 'logreg' };
      setRsModel(init);
    }
  }, [defsLoading, getModelDefaults, rsModel, setRsModel]);

  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the Data & files section.');
      return;
    }
    if (!rsModel) return;

    const p1 = (hyperParam1.paramName || '').trim();
    const p2 = (hyperParam2.paramName || '').trim();
    const v1 = Array.isArray(hyperParam1.values) ? hyperParam1.values : [];
    const v2 = Array.isArray(hyperParam2.values) ? hyperParam2.values : [];

    if (!p1 || !p2) {
      setErr('Please select exactly two hyperparameters to vary.');
      return;
    }
    if (p1 === p2) {
      setErr('The two hyperparameters must be different.');
      return;
    }
    if (v1.length < 2 || v2.length < 2) {
      setErr('Each hyperparameter must have at least two values.');
      return;
    }
    if (!nIter || nIter <= 0) {
      setErr('Please specify a positive number of iterations (n_iter).');
      return;
    }

    setErr(null);
    setRsState({ result: null });
    setLoading(true);

    try {
      const paramDistributions = {
        [p1]: v1,
        [p2]: v2,
      };

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
          pca_n,
          pca_var,
          pca_whiten,
          lda_n,
          lda_solver,
          lda_shrinkage,
          lda_tol,
          sfs_k: sfs_k === '' || sfs_k == null ? 'auto' : sfs_k,
          sfs_direction,
          sfs_cv,
          sfs_n_jobs,
        },
        model: rsModel,
        eval: {
          metric: effectiveMetric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
        },
        param_distributions: paramDistributions,
        n_iter: Number(nIter),
        n_jobs: Number(nJobs),
      };

      const data = await requestRandomSearch(payload);
      setRsState({ result: { ...data, metric_used: effectiveMetric } });
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e.message ?? String(e);
      const msg = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || !models || !rsModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Randomized search</Title>

      {err && (
        <Alert color="red" variant="light">
          <Text fw={500}>Error</Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
            {err}
          </Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button
              size="xs"
              onClick={handleCompute}
              loading={loading}
              disabled={!dataReady}
            >
              {loading ? 'Searching…' : 'Run randomized search'}
            </Button>
          </Group>

          <Box
            style={{
              margin: '0 auto',
              width: '100%',
            }}
          >
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
                nSplits={nSplits}
                onNSplitsChange={(value) => setRsState({ nSplits: value })}
                stratified={stratified}
                onStratifiedChange={(value) => setRsState({ stratified: value })}
                shuffle={shuffle}
                onShuffleChange={(value) => setRsState({ shuffle: value })}
                seed={seed}
                onSeedChange={(value) => setRsState({ seed: value })}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={rsModel}
                onChange={(next) => {
                  if (next?.algo && rsModel && next.algo !== rsModel.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setRsModel({ ...d, ...next });
                    setRsState({
                      hyperParam1: EMPTY_PARAM,
                      hyperParam2: EMPTY_PARAM,
                    });
                  } else {
                    setRsModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
              />

              <Divider my="xs" />

              <Stack gap="sm">
                <Text size="sm" fw={500}>
                  Select the 1st parameter to sample:
                </Text>
                <HyperparameterSelector
                  schema={models?.schema}
                  model={rsModel}
                  value={hyperParam1}
                  onChange={handleHyperParam1Change}
                  label="1st hyperparameter"
                />
              </Stack>

              <Stack gap="sm">
                <Text size="sm" fw={500}>
                  Select the 2nd parameter to sample:
                </Text>
                <HyperparameterSelector
                  schema={models?.schema}
                  model={rsModel}
                  value={hyperParam2}
                  onChange={handleHyperParam2Change}
                  label="2nd hyperparameter"
                />
              </Stack>

              <Group gap="md" align="flex-end">
                <Box style={{ maxWidth: 180 }}>
                  <NumberInput
                    label="n_iter (samples)"
                    min={1}
                    step={1}
                    value={nIter}
                    onChange={(value) => setRsState({ nIter: value })}
                  />
                </Box>
                <Box style={{ maxWidth: 180 }}>
                  <NumberInput
                    label="n_jobs"
                    min={1}
                    step={1}
                    value={nJobs}
                    onChange={(value) => setRsState({ nJobs: value })}
                  />
                </Box>
              </Group>
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear once the randomized search finishes.
            </Text>
          )}
        </Stack>
      </Card>

      <RandomSearchResultsPanel result={randomResult} />
    </Stack>
  );
}
