import { useEffect } from 'react';
import {
  Stack,
  Title,
} from '@mantine/core';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useTuningStore } from '../state/useTuningStore.js';
import { useModelConfigStore } from '../../training/state/useModelConfigStore.js';
import { useTuningDefaultsQuery } from '../../../shared/schema/useTuningDefaultsQuery.js';

import { useEffectiveMetricForTask } from '../hooks/useEffectiveMetricForTask.js';
import { buildTuningCommonPayload } from '../utils/buildTuningCommonPayload.js';
import { getDefaultAlgoForTask } from '../utils/getDefaultAlgoForTask.js';

import { compactPayload } from '../../../shared/utils/compactPayload.js';

import { requestLearningCurve } from '../api/tuningApi.js';
import LearningCurveResultsPanel from './results/LearningCurveResultsPanel.jsx';

import LearningCurveConfigPane from './learningCurve/LearningCurveConfigPane.jsx';
import TuningConfigCard from './common/TuningConfigCard.jsx';
import TuningErrorAlert from './common/TuningErrorAlert.jsx';
import { useTuningRunner } from '../hooks/useTuningRunner.js';

export default function LearningCurvePanel() {
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;
  const taskInferred = inspectReport?.task_inferred || null;

  const setLearningCurveResult = useResultsStore((s) => s.setLearningCurveResult);
  const learningCurveNSplits = useResultsStore((s) => s.learningCurveNSplits);
  const setLearningCurveNSplits = useResultsStore((s) => s.setLearningCurveNSplits);
  const learningCurveWithinPct = useResultsStore((s) => s.learningCurveWithinPct);
  const setLearningCurveWithinPct = useResultsStore((s) => s.setLearningCurveWithinPct);

  const lcState = useTuningStore((s) => s.learningCurve);
  const setLcState = useTuningStore((s) => s.setLearningCurve);

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

  // Per-panel model config (learning curve slice)
  const lcModel = useModelConfigStore((s) => s.learningCurve);
  const setLcModel = useModelConfigStore((s) => s.setLearningCurveModel);

  const { loading, error, run } = useTuningRunner();
  const { effectiveMetric } = useEffectiveMetricForTask({
    enums,
    taskInferred,
    metric,
    setMetric,
  });

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
    stratified,
    shuffle,
    seed,
    trainSizesCSV,
    nSteps,
    nJobs,
  } = lcState;

  const defaultNSteps = tuningDefaults?.learning_curve?.n_steps;
  const defaultNJobs = tuningDefaults?.learning_curve?.n_jobs;
  const effectiveNSteps = nSteps ?? defaultNSteps;
  const effectiveNJobs = nJobs ?? defaultNJobs;



  // Initialize LC model once
  useEffect(() => {
    if (!defsLoading && !lcModel && models) {
      const defaultAlgo = getDefaultAlgoForTask({
        models,
        task: taskInferred,
      });
      if (defaultAlgo) setLcModel({ algo: defaultAlgo });
    }
  }, [defsLoading, lcModel, setLcModel, taskInferred, models]);

  function handleCompute() {
    return run({
      preflight: () => {
        if (!dataReady) return 'Load & inspect data first in the Data & files section.';
        if (!lcModel) return 'Select a model first.';
        return null;
      },
      onStart: () => setLearningCurveResult(null),
      request: async () => {
        const train_sizes = trainSizesCSV
          ? trainSizesCSV
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean)
              .map((x) => (x.includes('.') ? parseFloat(x) : parseInt(x, 10)))
          : undefined;

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
          model: lcModel,
          split: {
            nSplits: learningCurveNSplits,
            stratified,
            shuffle,
            seed,
          },
          evalMetric: effectiveMetric,
          schemaDefaults: {
            scale: schemaScale,
            features: schemaFeatures,
            split: schemaSplit,
          },
        });

        const payload = compactPayload({
          ...basePayload,
          train_sizes,
          n_steps: nSteps,
          n_jobs: nJobs,
        });

        return requestLearningCurve(payload);
      },
      onSuccess: (data) => setLearningCurveResult({ ...data, metric_used: effectiveMetric }),
    });
  }

  if (defsLoading || tuningLoading || !models || !lcModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Learning Curve</Title>

      <TuningErrorAlert error={error} />

      <TuningConfigCard
        actionLabel="Compute"
        actionLabelLoading="Computing…"
        onAction={handleCompute}
        loading={loading}
        disabled={!dataReady}
        loadingHint="Results will appear below once the learning curve computation finishes."
      >
        <LearningCurveConfigPane
          nSplits={learningCurveNSplits}
          onNSplitsChange={setLearningCurveNSplits}
          stratified={stratified}
          onStratifiedChange={(value) => setLcState({ stratified: value })}
          shuffle={shuffle}
          onShuffleChange={(value) => setLcState({ shuffle: value })}
          seed={seed}
          onSeedChange={(value) => setLcState({ seed: value })}

          model={lcModel}
          onModelChange={(next) => {
            if (next?.algo && lcModel && next.algo !== lcModel.algo) {
              setLcModel({ algo: next.algo });
            } else {
              setLcModel(next);
            }
          }}
          schema={models?.schema}
          enums={enums}
          models={models}

          nStepsOverride={nSteps}
          defaultNSteps={defaultNSteps}
          onNStepsChangeOverride={(v) => setLcState({ nSteps: v })}
          nJobsOverride={nJobs}
          defaultNJobs={defaultNJobs}
          onNJobsChangeOverride={(v) => setLcState({ nJobs: v })}
          trainSizesCSV={trainSizesCSV}
          onTrainSizesCSVChange={(v) => setLcState({ trainSizesCSV: v })}

          withinPct={learningCurveWithinPct}
          onWithinPctChange={setLearningCurveWithinPct}
        />
      </TuningConfigCard>

      {/* Results are handled inside this panel now, same style as other tuning panels */}
      <LearningCurveResultsPanel />
    </Stack>
  );
}