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
import { requestValidationCurve } from '../api/tuningApi.js';
import ValidationCurveResultsPanel from './results/ValidationCurveResultsPanel.jsx';
import HyperparameterSelector from './helpers/HyperparameterSelector.jsx';

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
  const effectiveNJobs = nJobs ?? defaultNJobs;

  // Per-panel model config (validation curve slice)
  const vcModel = useModelConfigStore((s) => s.validationCurve);
  const setVcModel = useModelConfigStore((s) => s.setValidationCurveModel);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

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

  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the Data & files section.');
      return;
    }
    if (!vcModel) return;

    const name = (hyperParam.paramName || '').trim();
    const paramRange = Array.isArray(hyperParam.values) ? hyperParam.values : [];

    if (!name) {
      setErr('Please select a hyperparameter to vary.');
      return;
    }
    if (paramRange.length < 2) {
      setErr('Please provide at least two values for the hyperparameter.');
      return;
    }

    setErr(null);
    setVcState({ result: null });
    setLoading(true);

    try {
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
        // overrides-only; omit when unset so backend request defaults apply
        n_jobs: nJobs,
      });

      const data = await requestValidationCurve(payload);
      setVcState({ result: { ...data, metric_used: effectiveMetric } });
    } catch (e) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || tuningLoading || !models || !vcModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Validation Curve</Title>

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
            <Button size="xs" onClick={handleCompute} loading={loading} disabled={!dataReady}>
              {loading ? 'Computing…' : 'Compute'}
            </Button>
          </Group>

          <Box style={{ margin: '0 auto', width: '100%' }}>
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
                nSplits={nSplits}
                onNSplitsChange={(value) => setVcState({ nSplits: value })}
                stratified={stratified}
                onStratifiedChange={(value) => setVcState({ stratified: value })}
                shuffle={shuffle}
                onShuffleChange={(value) => setVcState({ shuffle: value })}
                seed={seed}
                onSeedChange={(value) => setVcState({ seed: value })}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={vcModel}
                onChange={(next) => {
                  if (next?.algo && vcModel && next.algo !== vcModel.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setVcModel({ ...d, ...next });
                    // Reset hyperparameter selection when algo changes
                    setVcState({ hyperParam: EMPTY_PARAM });
                  } else {
                    setVcModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
              />

              <Divider my="xs" />

              <HyperparameterSelector
                schema={models?.schema}
                model={vcModel}
                value={hyperParam}
                onChange={handleHyperParamChange}
              />

              <Box style={{ maxWidth: 180 }}>
                <NumberInput
                  label="n_jobs"
                  min={1}
                  step={1}
                  value={effectiveNJobs}
                  onChange={(value) => {
                    const v = value === '' || value == null ? undefined : value;
                    if (defaultNJobs != null && v === defaultNJobs) {
                      setVcState({ nJobs: undefined });
                      return;
                    }
                    setVcState({ nJobs: v });
                  }}
                />
              </Box>
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear once the validation curve computation finishes.
            </Text>
          )}
        </Stack>
      </Card>

      <ValidationCurveResultsPanel result={validationResult} nSplits={nSplits} />
    </Stack>
  );
}
