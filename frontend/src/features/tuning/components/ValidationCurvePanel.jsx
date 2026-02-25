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

import { requestValidationCurve } from '../api/tuningApi.js';
import ValidationCurveResultsPanel from './results/ValidationCurveResultsPanel.jsx';
import ValidationCurveConfigPane from './validationCurve/ValidationCurveConfigPane.jsx';
import TuningConfigCard from './common/TuningConfigCard.jsx';
import TuningErrorAlert from './common/TuningErrorAlert.jsx';
import { useTuningRunner } from '../hooks/useTuningRunner.js';

const EMPTY_PARAM = { paramName: '', values: [] };

export default function ValidationCurvePanel() {
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

  const { data: tuningDefaults, isLoading: tuningLoading } = useTuningDefaultsQuery();

  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  const vcState = useTuningStore((s) => s.validationCurve);
  const setVcState = useTuningStore((s) => s.setValidationCurve);

  const {
    nSplits,
    stratified,
    shuffle,
    seed,
    nJobs,
    result: validationResult,
  } = vcState;

  // Persisted hyperparameter selection (in tuning store)
  const hyperParam = vcState.hyperParam ?? EMPTY_PARAM;
  const handleHyperParamChange = (next) => setVcState({ hyperParam: next });

  const defaultNJobs = tuningDefaults?.validation_curve?.n_jobs;

  // Per-panel model config (validation curve slice)
  const vcModel = useModelConfigStore((s) => s.validationCurve);
  const setVcModel = useModelConfigStore((s) => s.setValidationCurveModel);

  const { loading, error, run } = useTuningRunner();

  const taskInferred = inspectReport?.task_inferred || null;
  const { effectiveMetric } = useEffectiveMetricForTask({
    enums,
    taskInferred,
    metric,
    setMetric,
  });

  // Initialize VC model once
  useEffect(() => {
    if (!defsLoading && !vcModel) {
      const defaultAlgo = taskInferred === 'regression' ? 'linreg' : 'logreg';
      const init = getModelDefaults(defaultAlgo) || { algo: defaultAlgo };
      setVcModel(init);
    }
  }, [defsLoading, getModelDefaults, taskInferred, vcModel, setVcModel]);

  function handleCompute() {
    const name = (hyperParam.paramName || '').trim();
    const paramRange = Array.isArray(hyperParam.values) ? hyperParam.values : [];

    return run({
      preflight: () => {
        if (!dataReady) return 'Load & inspect data first in the Data & files section.';
        if (!vcModel) return 'Select a model first.';
        if (!name) return 'Please select a hyperparameter to vary.';
        if (paramRange.length < 2) {
          return 'Please provide at least two values for the hyperparameter.';
        }
        return null;
      },
      onStart: () => setVcState({ result: null }),
      request: async () => {
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
          model: vcModel,
          split: { nSplits, stratified, shuffle, seed },
          evalMetric: effectiveMetric,
          schemaDefaults: {
            scale: schemaScale,
            features: schemaFeatures,
            split: schemaSplit,
          },
        });

        const payload = compactPayload({
          ...basePayload,
          param_name: name,
          param_range: paramRange,
          n_jobs: nJobs,
        });

        return requestValidationCurve(payload);
      },
      onSuccess: (data) => setVcState({ result: { ...data, metric_used: effectiveMetric } }),
    });
  }

  if (defsLoading || tuningLoading || !models || !vcModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Validation Curve</Title>

      <TuningErrorAlert error={error} />

      <TuningConfigCard
        actionLabel="Compute"
        actionLabelLoading="Computing…"
        onAction={handleCompute}
        loading={loading}
        disabled={!dataReady}
        loadingHint="Results will appear once the validation curve computation finishes."
      >
        <ValidationCurveConfigPane
          nSplits={nSplits}
          onNSplitsChange={(value) => setVcState({ nSplits: value })}
          stratified={stratified}
          onStratifiedChange={(value) => setVcState({ stratified: value })}
          shuffle={shuffle}
          onShuffleChange={(value) => setVcState({ shuffle: value })}
          seed={seed}
          onSeedChange={(value) => setVcState({ seed: value })}

          model={vcModel}
          onModelChange={(next) => {
            if (next?.algo && vcModel && next.algo !== vcModel.algo) {
              const d = getModelDefaults(next.algo) || { algo: next.algo };
              setVcModel({ ...d, ...next });
              setVcState({ hyperParam: EMPTY_PARAM });
            } else {
              setVcModel(next);
            }
          }}
          schema={models?.schema}
          enums={enums}
          models={models}

          hyperParam={hyperParam}
          onHyperParamChange={handleHyperParamChange}

          nJobsOverride={nJobs}
          defaultNJobs={defaultNJobs}
          onNJobsChangeOverride={(v) => setVcState({ nJobs: v })}
        />
      </TuningConfigCard>

      <ValidationCurveResultsPanel result={validationResult} nSplits={nSplits} />
    </Stack>
  );
}
