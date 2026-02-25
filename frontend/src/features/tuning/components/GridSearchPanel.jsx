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

import { requestGridSearch } from '../api/tuningApi.js';
import GridSearchResultsPanel from './results/GridSearchResultsPanel.jsx';
import GridSearchConfigPane from './gridSearch/GridSearchConfigPane.jsx';
import TuningConfigCard from './common/TuningConfigCard.jsx';
import TuningErrorAlert from './common/TuningErrorAlert.jsx';
import { useTuningRunner } from '../hooks/useTuningRunner.js';

const EMPTY_PARAM = { paramName: '', values: [] };

export default function GridSearchPanel() {
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
    getModelDefaults,
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

  const gsState = useTuningStore((s) => s.gridSearch);
  const setGsState = useTuningStore((s) => s.setGridSearch);

  const {
    nSplits,
    stratified,
    shuffle,
    seed,
    nJobs,
    result: gridResult,
  } = gsState;

  // two hyperparameters to vary
  const hyperParam1 = gsState.hyperParam1 ?? EMPTY_PARAM;
  const hyperParam2 = gsState.hyperParam2 ?? EMPTY_PARAM;

  const handleHyperParam1Change = (next) =>
    setGsState({ hyperParam1: next });

  const handleHyperParam2Change = (next) =>
    setGsState({ hyperParam2: next });

  // Per-panel model config (grid search slice)
  const gsModel = useModelConfigStore((s) => s.gridSearch);
  const setGsModel = useModelConfigStore((s) => s.setGridSearchModel);

  const { loading, error, run } = useTuningRunner();
  const taskInferred = inspectReport?.task_inferred || null;
  const { effectiveMetric } = useEffectiveMetricForTask({
    enums,
    taskInferred,
    metric,
    setMetric,
  });

  // Initialize GS model once
  useEffect(() => {
    if (!defsLoading && !gsModel) {
      const defaultAlgo = taskInferred === 'regression' ? 'linreg' : 'logreg';
      const init = getModelDefaults(defaultAlgo) || { algo: defaultAlgo };
      setGsModel(init);
    }
  }, [defsLoading, getModelDefaults, gsModel, setGsModel]);

  function handleCompute() {
    const p1 = (hyperParam1.paramName || '').trim();
    const p2 = (hyperParam2.paramName || '').trim();
    const v1 = Array.isArray(hyperParam1.values) ? hyperParam1.values : [];
    const v2 = Array.isArray(hyperParam2.values) ? hyperParam2.values : [];

    return run({
      preflight: () => {
        if (!dataReady) return 'Load & inspect data first in the Data & files section.';
        if (!gsModel) return 'Select a model first.';
        if (!p1 || !p2) return 'Please select exactly two hyperparameters to vary.';
        if (p1 === p2) return 'The two hyperparameters must be different.';
        if (v1.length < 2 || v2.length < 2) {
          return 'Each hyperparameter must have at least two values.';
        }
        return null;
      },
      onStart: () => setGsState({ result: null }),
      request: async () => {
        const paramGrid = {
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
          model: gsModel,
          split: { nSplits, stratified, shuffle, seed },
          evalMetric: effectiveMetric,
          schemaDefaults: {
            scale: schemaScale,
            features: schemaFeatures,
            split: schemaSplit,
          },
        });

        const defaultNJobs = tuningDefaults?.grid_search?.n_jobs;
        const payload = compactPayload({
          ...basePayload,
          param_grid: paramGrid,
          n_jobs: defaultNJobs != null && nJobs === defaultNJobs ? undefined : nJobs,
        });

        return requestGridSearch(payload);
      },
      onSuccess: (data) => setGsState({ result: { ...data, metric_used: effectiveMetric } }),
    });
  }

  if (defsLoading || tuningLoading || !models || !gsModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Grid search</Title>

      <TuningErrorAlert error={error} />

      <TuningConfigCard
        actionLabel="Run grid search"
        actionLabelLoading="Searching…"
        onAction={handleCompute}
        loading={loading}
        disabled={!dataReady}
        loadingHint="Results will appear once the grid search finishes."
      >
        <GridSearchConfigPane
          nSplits={nSplits}
          onNSplitsChange={(value) => setGsState({ nSplits: value })}
          stratified={stratified}
          onStratifiedChange={(value) => setGsState({ stratified: value })}
          shuffle={shuffle}
          onShuffleChange={(value) => setGsState({ shuffle: value })}
          seed={seed}
          onSeedChange={(value) => setGsState({ seed: value })}

          model={gsModel}
          onModelChange={(next) => {
            if (next?.algo && gsModel && next.algo !== gsModel.algo) {
              const d = getModelDefaults(next.algo) || { algo: next.algo };
              setGsModel({ ...d, ...next });
              setGsState({ hyperParam1: EMPTY_PARAM, hyperParam2: EMPTY_PARAM });
            } else {
              setGsModel(next);
            }
          }}
          schema={models?.schema}
          enums={enums}
          models={models}

          hyperParam1={hyperParam1}
          onHyperParam1Change={handleHyperParam1Change}
          hyperParam2={hyperParam2}
          onHyperParam2Change={handleHyperParam2Change}

          nJobsOverride={nJobs}
          defaultNJobs={tuningDefaults?.grid_search?.n_jobs}
          onNJobsChangeOverride={(v) => setGsState({ nJobs: v })}
        />
      </TuningConfigCard>

      <GridSearchResultsPanel result={gridResult} />
    </Stack>
  );
}
