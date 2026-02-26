import { useEffect } from 'react';
import {
  Stack,
  Title,
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

import { requestRandomSearch } from '../api/tuningApi.js';
import RandomSearchResultsPanel from './results/RandomSearchResultsPanel.jsx';
import RandomSearchConfigPane from './randomSearch/RandomSearchConfigPane.jsx';
import TuningConfigCard from './common/TuningConfigCard.jsx';
import TuningErrorAlert from './common/TuningErrorAlert.jsx';
import { useTuningRunner } from '../hooks/useTuningRunner.js';

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

  const {
    loading: defsLoading,
    models,
    enums,
    scale: schemaScale,
    features: schemaFeatures,
    split: schemaSplit,
  } = useSchemaDefaults();

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

  const { loading, error, run } = useTuningRunner();

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
      setRsModel({ algo: defaultAlgo });
    }
  }, [defsLoading, rsModel, setRsModel, taskInferred]);

  function handleCompute() {
    const p1 = (hyperParam1.paramName || '').trim();
    const p2 = (hyperParam2.paramName || '').trim();
    const v1 = Array.isArray(hyperParam1.values) ? hyperParam1.values : [];
    const v2 = Array.isArray(hyperParam2.values) ? hyperParam2.values : [];

    const defaultNIter = tuningDefaults?.random_search?.n_iter;
    const defaultNJobs = tuningDefaults?.random_search?.n_jobs;
    const effectiveNIter = nIter ?? defaultNIter;

    return run({
      preflight: () => {
        if (!dataReady) return 'Load & inspect data first in the Data & files section.';
        if (!rsModel) return 'Select a model first.';
        if (!p1 || !p2) return 'Please select exactly two hyperparameters to vary.';
        if (p1 === p2) return 'The two hyperparameters must be different.';
        if (v1.length < 2 || v2.length < 2) {
          return 'Each hyperparameter must have at least two values.';
        }
        if (!effectiveNIter || effectiveNIter <= 0) {
          return 'Please specify a positive number of iterations (n_iter).';
        }
        return null;
      },
      onStart: () => setRsState({ result: null }),
      request: async () => {
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
          schemaDefaults: {
            scale: schemaScale,
            features: schemaFeatures,
            split: schemaSplit,
          },
        });

        const randomState = basePayload?.eval?.seed;

        const payload = compactPayload({
          ...basePayload,
          param_distributions: paramDistributions,
          n_iter:
            tuningDefaults?.random_search?.n_iter != null &&
            nIter === tuningDefaults.random_search.n_iter
              ? undefined
              : nIter,
          n_jobs: defaultNJobs != null && nJobs === defaultNJobs ? undefined : nJobs,
          random_state: randomState,
        });

        return requestRandomSearch(payload);
      },
      onSuccess: (data) => setRsState({ result: { ...data, metric_used: effectiveMetric } }),
    });
  }

  if (defsLoading || tuningLoading || !models || !rsModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Randomized search</Title>

      <TuningErrorAlert error={error} />

      <TuningConfigCard
        actionLabel="Run randomized search"
        actionLabelLoading="Searching…"
        onAction={handleCompute}
        loading={loading}
        disabled={!dataReady}
        loadingHint="Results will appear once the randomized search finishes."
      >
        <RandomSearchConfigPane
          nSplits={nSplits}
          onNSplitsChange={(value) => setRsState({ nSplits: value })}
          stratified={stratified}
          onStratifiedChange={(value) => setRsState({ stratified: value })}
          shuffle={shuffle}
          onShuffleChange={(value) => setRsState({ shuffle: value })}
          seed={seed}
          onSeedChange={(value) => setRsState({ seed: value })}

          model={rsModel}
          onModelChange={(next) => {
            if (next?.algo && rsModel && next.algo !== rsModel.algo) {
              setRsModel({ algo: next.algo });
              setRsState({ hyperParam1: EMPTY_PARAM, hyperParam2: EMPTY_PARAM });
            } else {
              setRsModel(next);
            }
          }}
          schema={models?.schema}
          enums={enums}
          models={models}

          hyperParam1={hyperParam1}
          onHyperParam1Change={handleHyperParam1Change}
          hyperParam2={hyperParam2}
          onHyperParam2Change={handleHyperParam2Change}

          nIterOverride={nIter}
          defaultNIter={tuningDefaults?.random_search?.n_iter}
          onNIterChangeOverride={(v) => setRsState({ nIter: v })}
          nJobsOverride={nJobs}
          defaultNJobs={tuningDefaults?.random_search?.n_jobs}
          onNJobsChangeOverride={(v) => setRsState({ nJobs: v })}
        />
      </TuningConfigCard>

      <RandomSearchResultsPanel result={randomResult} />
    </Stack>
  );
}
