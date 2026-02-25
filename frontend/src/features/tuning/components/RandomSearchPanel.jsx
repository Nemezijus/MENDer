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

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useTuningStore } from '../state/useTuningStore.js';
import { useModelConfigStore } from '../../training/state/useModelConfigStore.js';
import { useTuningDefaultsQuery } from '../../../shared/schema/useTuningDefaultsQuery.js';

import { useEffectiveMetricForTask } from '../hooks/useEffectiveMetricForTask.js';
import { buildTuningCommonPayload } from '../utils/buildTuningCommonPayload.js';

import { compactPayload } from '../../../shared/utils/compactPayload.js';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../training/components/ModelSelectionCard.jsx';
import { requestRandomSearch } from '../api/tuningApi.js';
import RandomSearchResultsPanel from './results/RandomSearchResultsPanel.jsx';
import HyperparameterSelector from './helpers/HyperparameterSelector.jsx';

const EMPTY_PARAM = { paramName: '', values: [] };

function parseSeedForSearch({ shuffle, seed } = {}) {
  if (shuffle === false) return undefined;
  if (seed === '' || seed == null) return undefined;
  const n = parseInt(seed, 10);
  return Number.isFinite(n) ? n : undefined;
}

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

  const {
    data: tuningDefaults,
    isLoading: tuningLoading,
  } = useTuningDefaultsQuery();

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
  const { effectiveMetric } = useEffectiveMetricForTask({
    enums,
    taskInferred,
    metric,
    setMetric,
  });

  // Initialize RS model once
  useEffect(() => {
    if (!defsLoading && !rsModel) {
      const defaultAlgo = taskInferred === 'regression' ? 'linreg' : 'logreg';
      const init = getModelDefaults(defaultAlgo) || { algo: defaultAlgo };
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
    const defaultNIter = tuningDefaults?.random_search?.n_iter;
    const effectiveNIter = nIter ?? defaultNIter;
    if (!effectiveNIter || effectiveNIter <= 0) {
      setErr('Please specify a positive number of iterations (n_iter).');
      return;
    }

    setErr(null);
    setRsState({ result: null });
    setLoading(true);

    try {
      const parsedSeed = parseSeedForSearch({ shuffle, seed });
      const paramDistributions = {
        [p1]: v1,
        [p2]: v2,
      };
      const basePayload = buildTuningCommonPayload({
        data: { xPath, yPath, npzPath, xKey, yKey },
        features: {
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
        },
        scaleMethod,
        model: rsModel,
        split: { nSplits, stratified, shuffle, seed },
        evalMetric: effectiveMetric,
      });

      const defaultNJobs = tuningDefaults?.random_search?.n_jobs;
      const payload = compactPayload({
        ...basePayload,
        param_distributions: paramDistributions,
        // Send override only; omit if equal to backend default.
        n_iter:
          tuningDefaults?.random_search?.n_iter != null &&
          nIter === tuningDefaults.random_search.n_iter
            ? undefined
            : nIter,
        n_jobs: defaultNJobs != null && nJobs === defaultNJobs ? undefined : nJobs,
        // Make RandomizedSearchCV sampling reproducible when the user provides a seed.
        // This does not affect CV splitting; it only controls the parameter sampling.
        random_state: parsedSeed,
      });

      const data = await requestRandomSearch(payload);
      setRsState({ result: { ...data, metric_used: effectiveMetric } });
    } catch (e) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || tuningLoading || !models || !rsModel) {
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
                    value={nIter ?? tuningDefaults?.random_search?.n_iter}
                    onChange={(value) => {
                      const v = value === '' || value == null ? undefined : value;
                      const d = tuningDefaults?.random_search?.n_iter;
                      if (d != null && v === d) {
                        setRsState({ nIter: undefined });
                        return;
                      }
                      setRsState({ nIter: v });
                    }}
                  />
                </Box>
                <Box style={{ maxWidth: 180 }}>
                  <NumberInput
                    label="n_jobs"
                    min={1}
                    step={1}
                    value={nJobs ?? tuningDefaults?.random_search?.n_jobs}
                    onChange={(value) => {
                      const v = value === '' || value == null ? undefined : value;
                      const d = tuningDefaults?.random_search?.n_jobs;
                      if (d != null && v === d) {
                        setRsState({ nJobs: undefined });
                        return;
                      }
                      setRsState({ nJobs: v });
                    }}
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
