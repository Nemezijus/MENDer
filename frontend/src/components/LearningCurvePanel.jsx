import { useEffect, useMemo, useState } from 'react';
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
import { useResultsStore } from '../state/useResultsStore.js';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useTuningStore } from '../state/useTuningStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';
import { useTuningDefaultsQuery } from '../state/useTuningDefaultsQuery.js';

import { compactPayload } from '../utils/compactPayload.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';

import { requestLearningCurve } from '../api/tuning';
import LearningCurveResultsPanel from './visualizations/LearningCurveResultsPanel.jsx';

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

  const { loading: defsLoading, models, enums, getModelDefaults } = useSchemaDefaults();

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

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const allowedMetrics = useMemo(() => {
    const mt = enums?.MetricByTask || null;
    if (taskInferred && mt && Array.isArray(mt[taskInferred])) {
      return mt[taskInferred].map(String);
    }
    if (Array.isArray(enums?.MetricName)) return enums.MetricName.map(String);
    return [];
  }, [enums, taskInferred]);

  // Use backend-provided task-specific metric ordering as the default.
  // Do not write this into state (stores are overrides-only).
  const defaultMetricFromSchema = allowedMetrics?.[0] ?? null;
  const effectiveMetric = metric ?? defaultMetricFromSchema;

  // If the user has an explicit metric override that doesn't belong to the
  // current task's allowed list, clear it.
  useEffect(() => {
    if (!metric) return;
    if (allowedMetrics.length > 0 && !allowedMetrics.includes(String(metric))) {
      setMetric(undefined);
    }
  }, [allowedMetrics, metric, setMetric]);

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
    if (!defsLoading && !lcModel) {
      const defaultAlgo = taskInferred === 'regression' ? 'linreg' : 'logreg';
      const init = getModelDefaults(defaultAlgo) || { algo: defaultAlgo };
      setLcModel(init);
    }
  }, [defsLoading, getModelDefaults, lcModel, setLcModel]);

  // When algo changes, rehydrate from backend defaults while preserving overrides
  useEffect(() => {
    if (!lcModel) return;
    const base = getModelDefaults(lcModel.algo) || { algo: lcModel.algo };
    const merged = { ...base, ...lcModel };
    setLcModel(merged);
  }, [getModelDefaults, lcModel?.algo]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the Data & files section.');
      return;
    }
    if (!lcModel) return;

    setErr(null);
    setLearningCurveResult(null);
    setLoading(true);

    try {
      const train_sizes = trainSizesCSV
        ? trainSizesCSV
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
            .map((x) => (x.includes('.') ? parseFloat(x) : parseInt(x, 10)))
        : undefined;

      const parsedSeed =
        shuffle === false
          ? undefined
          : seed === '' || seed == null
          ? undefined
          : parseInt(seed, 10);

      const payload = compactPayload({
        data: {
          x_path: npzPath ? undefined : xPath,
          y_path: npzPath ? undefined : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: {
          mode: 'kfold',
          n_splits: learningCurveNSplits,
          stratified,
          shuffle,
        },
        scale: {
          method: scaleMethod,
        },
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
        model: lcModel,
        eval: {
          metric: effectiveMetric,
          seed: parsedSeed,
        },
        train_sizes,
        n_steps: nSteps,
        n_jobs: nJobs,
      });

      const data = await requestLearningCurve(payload);
      setLearningCurveResult({ ...data, metric_used: effectiveMetric });
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e.message ?? String(e);
      const msg = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || tuningLoading || !models || !lcModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Learning Curve</Title>

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
              {loading ? 'Computing…' : 'Compute'}
            </Button>
          </Group>

          <Box w="100%" style={{ margin: '0 auto' }}>
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
                nSplits={learningCurveNSplits}
                onNSplitsChange={setLearningCurveNSplits}
                stratified={stratified}
                onStratifiedChange={(value) => setLcState({ stratified: value })}
                shuffle={shuffle}
                onShuffleChange={(value) => setLcState({ shuffle: value })}
                seed={seed}
                onSeedChange={(value) => setLcState({ seed: value })}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={lcModel}
                onChange={(next) => {
                  if (next?.algo && lcModel && next.algo !== lcModel.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setLcModel({ ...d, ...next });
                  } else {
                    setLcModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
              />

              <Divider my="xs" />

              <NumberInput
                label="Steps (used if Train sizes empty)"
                min={2}
                max={50}
                step={1}
                value={effectiveNSteps}
                onChange={(value) => {
                  const v = value === '' || value == null ? undefined : value;
                  if (defaultNSteps != null && v === defaultNSteps) {
                    setLcState({ nSteps: undefined });
                    return;
                  }
                  setLcState({ nSteps: v });
                }}
              />
              <NumberInput
                label="n_jobs"
                min={1}
                step={1}
                value={effectiveNJobs}
                onChange={(value) => {
                  const v = value === '' || value == null ? undefined : value;
                  if (defaultNJobs != null && v === defaultNJobs) {
                    setLcState({ nJobs: undefined });
                    return;
                  }
                  setLcState({ nJobs: v });
                }}
              />
              <Text size="sm" c="dimmed">
                Optional Train sizes (CSV): fractions in (0,1] or absolute integers.
                Example:
                <Text span fw={500}> 0.1,0.3,0.5,0.7,1.0 </Text> or{' '}
                <Text span fw={500}> 50,100,200 </Text>
              </Text>
              <textarea
                style={{
                  width: '100%',
                  minHeight: 70,
                  fontFamily: 'inherit',
                  fontSize: '0.9rem',
                }}
                placeholder="e.g. 0.1,0.3,0.5,0.7,1.0"
                value={trainSizesCSV}
                onChange={(e) =>
                  setLcState({ trainSizesCSV: e.currentTarget.value })
                }
              />

              <Divider my="xs" />

              <NumberInput
                label="Recommend the smallest train size achieving at least this fraction of the peak validation score"
                description="e.g., 0.99 = within 1% of peak"
                min={0.5}
                max={1.0}
                step={0.01}
                value={learningCurveWithinPct}
                onChange={setLearningCurveWithinPct}
                precision={2}
              />
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear below once the learning curve computation finishes.
            </Text>
          )}
        </Stack>
      </Card>

      {/* Results are handled inside this panel now, same style as other tuning panels */}
      <LearningCurveResultsPanel />
    </Stack>
  );
}