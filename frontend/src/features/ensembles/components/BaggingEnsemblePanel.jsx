// frontend/src/components/ensembles/BaggingEnsemblePanel.jsx
import { useMemo, useState } from 'react';
import {
  Card,
  Stack,
  Group,
  Text,
  Button,
  Select,
  NumberInput,
  SegmentedControl,
  Divider,
  Alert,
  Box,
  ActionIcon,
  Switch,
} from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';

import { useDataStore } from '../../state/useDataStore.js';
import { useSettingsStore } from '../../../state/useSettingsStore.js';
import { useFeatureStore } from '../../../state/useFeatureStore.js';
import { useResultsStore } from '../../../state/useResultsStore.js';
import { useModelArtifactStore } from '../../state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import { compactPayload } from '../../utils/compactPayload.js';

import SplitOptionsCard from '../../../components/SplitOptionsCard.jsx';
import ModelSelectionCard from '../ModelSelectionCard.jsx';

import { runEnsembleTrainRequest } from '../api/ensemblesApi.js';

import EnsembleHelpText, { BaggingIntroText } from '../../../components/helpers/helpTexts/EnsembleHelpText.jsx';

import BaggingEnsembleClassificationResults from './BaggingEnsembleClassificationResults.jsx';
import BaggingEnsembleRegressionResults from './BaggingEnsembleRegressionResults.jsx';

/** ---------- helpers ---------- **/

function toErrorText(e) {
  if (typeof e === 'string') return e;
  const data = e?.response?.data;
  const detail = data?.detail ?? e?.detail;
  const pick = detail ?? data ?? e?.message ?? e;
  if (typeof pick === 'string') return pick;

  if (Array.isArray(pick)) {
    return pick
      .map((it) => {
        if (typeof it === 'string') return it;
        if (it && typeof it === 'object') {
          const loc = Array.isArray(it.loc) ? it.loc.join('.') : it.loc;
          return it.msg ? `${loc ? loc + ': ' : ''}${it.msg}` : JSON.stringify(it);
        }
        return String(it);
      })
      .join('\n');
  }

  try {
    return JSON.stringify(pick);
  } catch {
    return String(pick);
  }
}

// User-friendly names for the Algorithm dropdown.
// Values must remain the internal algo keys used by the backend.
const ALGO_LABELS = {
  // classifiers
  logreg: 'Logistic Regression',
  ridgeclf: 'Ridge Classifier',
  svm: 'SVM (RBF)',
  linsvm: 'Linear SVM',
  knn: 'kNN Classifier',
  tree: 'Decision Tree',
  forest: 'Random Forest',
  extratrees: 'Extra Trees',
  histgb: 'HistGradientBoosting',
  nb: 'Naive Bayes',

  // regressors
  linreg: 'Linear Regression',
  ridgereg: 'Ridge Regression',
  ridgecv: 'Ridge Regression (CV)',
  enet: 'Elastic Net',
  enetcv: 'Elastic Net (CV)',
  lasso: 'Lasso',
  lassocv: 'Lasso (CV)',
  bayridge: 'Bayesian Ridge',
  svr: 'SVR (RBF)',
  linsvr: 'Linear SVR',
  knnreg: 'kNN Regressor',
  treereg: 'Decision Tree Regressor',
  rfreg: 'Random Forest Regressor',
};

function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function algoKeyToLabel(key) {
  const k = String(key || '').toLowerCase();
  return ALGO_LABELS[k] || titleCase(k);
}

function normalizeNumberOrUndefined(v) {
  if (v === '' || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** ---------- component ---------- **/

export default function BaggingEnsemblePanel() {
  const inspectReport = useDataStore((s) => s.inspectReport);

  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  const fctx = useFeatureStore();
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);

  const {
    loading: defsLoading,
    models,
    enums,
    split,
    getModelDefaults,
    getCompatibleAlgos,
    getEnsembleDefaults,
  } = useSchemaDefaults();

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const trainResult = useResultsStore((s) => s.trainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  // ---- ensemble store slice ----
  const bagging = useEnsembleStore((s) => s.bagging);
  const setBagging = useEnsembleStore((s) => s.setBagging);
  const setBaggingBaseEstimator = useEnsembleStore((s) => s.setBaggingBaseEstimator);
  const resetBagging = useEnsembleStore((s) => s.resetBagging);

  // ---- local run state + help toggle ----
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const baggingDefaults = useMemo(
    () => getEnsembleDefaults?.('bagging') || null,
    [getEnsembleDefaults],
  );

  // ----------------- derived -----------------

  const compatibleAlgos = useMemo(
    () => getCompatibleAlgos?.(effectiveTask) || [],
    [getCompatibleAlgos, effectiveTask],
  );

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoKeyToLabel(a) })),
    [compatibleAlgos],
  );

  const samplingStrategyOptions = useMemo(
    () => [
      { value: 'auto', label: 'Auto (recommended)' },
      { value: 'majority', label: 'Majority' },
      { value: 'not minority', label: 'Not minority' },
      { value: 'not majority', label: 'Not majority' },
      { value: 'all', label: 'All' },
    ],
    [],
  );

  // ----------------- schema-driven defaults (display + payload) -----------------

  const allowedMetrics = useMemo(() => {
    if (!enums) return [];
    const metricByTask = enums.MetricByTask || null;
    if (metricByTask && effectiveTask && Array.isArray(metricByTask[effectiveTask])) {
      return metricByTask[effectiveTask].map(String);
    }
    if (Array.isArray(enums.MetricName)) return enums.MetricName.map(String);
    return [];
  }, [effectiveTask, enums]);

  // Use backend-provided task ordering as a suggestion without writing into state.
  // For regression we must avoid falling back to EvalModel.metric='accuracy'.
  const defaultMetricFromSchema = allowedMetrics?.[0] ?? undefined;
  const metricOverride = metric ? String(metric) : undefined;
  const metricIsAllowed =
    !metricOverride || allowedMetrics.length === 0 || allowedMetrics.includes(metricOverride);
  const metricForPayload = metricIsAllowed
    ? (metricOverride ?? (effectiveTask === 'regression' ? defaultMetricFromSchema : undefined))
    : (effectiveTask === 'regression' ? defaultMetricFromSchema : undefined);

  // Split defaults (schema-owned)
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;
  const defaultSplitMode = holdoutDefaults?.mode ?? kfoldDefaults?.mode ?? undefined;
  const effectiveSplitMode = bagging.splitMode ?? defaultSplitMode;

  // Base estimator: schema default (or first compatible) without mutating store.
  const defaultBaseAlgo =
    baggingDefaults?.base_estimator?.algo || (Array.isArray(compatibleAlgos) ? compatibleAlgos[0] : null);

  const defaultBaseEstimator = useMemo(() => {
    if (!defaultBaseAlgo) return undefined;
    return getModelDefaults?.(defaultBaseAlgo) || { algo: defaultBaseAlgo };
  }, [defaultBaseAlgo, getModelDefaults]);

  const effectiveBaseEstimator = bagging.base_estimator ?? defaultBaseEstimator;

  // Display values (defaults + overrides). Payload remains overrides-only.
  const dispNEstimators = bagging.n_estimators ?? baggingDefaults?.n_estimators;
  const dispMaxSamples = bagging.max_samples ?? baggingDefaults?.max_samples;
  const dispMaxFeatures = bagging.max_features ?? baggingDefaults?.max_features;
  const dispNJobs = bagging.n_jobs ?? baggingDefaults?.n_jobs;
  const dispRandomState = bagging.random_state ?? baggingDefaults?.random_state;

  const dispBootstrap = bagging.bootstrap ?? baggingDefaults?.bootstrap ?? false;
  const dispBootstrapFeatures = bagging.bootstrap_features ?? baggingDefaults?.bootstrap_features ?? false;
  const dispOobScore = bagging.oob_score ?? baggingDefaults?.oob_score ?? false;

  const dispBalanced = bagging.balanced ?? baggingDefaults?.balanced ?? false;
  const dispSamplingStrategy = bagging.sampling_strategy ?? baggingDefaults?.sampling_strategy;
  const dispReplacement = bagging.replacement ?? baggingDefaults?.replacement ?? false;

  const handleReset = () => {
    resetBagging(effectiveTask);
    setErr(null);
  };

  const buildPayload = () => {
    if (!effectiveSplitMode) {
      throw new Error('Schema defaults not loaded: split mode is unavailable.');
    }

    if (!effectiveBaseEstimator || !effectiveBaseEstimator.algo) {
      throw new Error('Base estimator is not selected.');
    }

    // DATA (override-only; empty-string keys are omitted)
    const data = compactPayload({
      x_path: npzPath ? undefined : xPath,
      y_path: npzPath ? undefined : yPath,
      npz_path: npzPath,
      x_key: xKey,
      y_key: yKey,
    });

    // SPLIT (override-only)
    const splitCfg = compactPayload(
      effectiveSplitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: bagging.nSplits,
            stratified: bagging.stratified,
            shuffle: bagging.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: bagging.trainFrac,
            stratified: bagging.stratified,
            shuffle: bagging.shuffle,
          },
    );

    // SCALE (override-only)
    const scale = compactPayload({ method: scaleMethod });

    // FEATURES (override-only)
    let features = {};
    const m = fctx?.method;
    if (m === 'pca') {
      features = compactPayload({
        method: m,
        pca_n: fctx.pca_n,
        pca_var: fctx.pca_var,
        pca_whiten: fctx.pca_whiten,
      });
    } else if (m === 'lda') {
      features = compactPayload({
        method: m,
        lda_n: fctx.lda_n,
        lda_solver: fctx.lda_solver,
        lda_shrinkage: fctx.lda_shrinkage,
        lda_tol: fctx.lda_tol,
      });
    } else if (m === 'sfs') {
      features = compactPayload({
        method: m,
        sfs_k: fctx.sfs_k,
        sfs_direction: fctx.sfs_direction,
        sfs_cv: fctx.sfs_cv,
        sfs_n_jobs: fctx.sfs_n_jobs,
      });
    } else {
      features = compactPayload({ method: m });
    }

    // EVAL (schema-driven metric; decoder defaults owned by engine)
    const seedInt =
      bagging.seed === '' || bagging.seed == null
        ? undefined
        : Number.parseInt(String(bagging.seed), 10);

    const evalCfg = compactPayload({
      metric: metricForPayload,
      seed: Number.isFinite(seedInt) ? seedInt : undefined,
    });

    const ensemble = compactPayload({
      kind: 'bagging',
      base_estimator: effectiveBaseEstimator,

      n_estimators: bagging.n_estimators,
      max_samples: bagging.max_samples,
      max_features: bagging.max_features,
      bootstrap: bagging.bootstrap,
      bootstrap_features: bagging.bootstrap_features,
      oob_score: bagging.oob_score,
      n_jobs: bagging.n_jobs,
      random_state: bagging.random_state,

      balanced: bagging.balanced,
      sampling_strategy: bagging.sampling_strategy,
      replacement: bagging.replacement,
    });

    return { data, split: splitCfg, scale, features, ensemble, eval: evalCfg };
  };

  const handleRun = async () => {
    setErr(null);

    if (!inspectReport || inspectReport?.n_samples <= 0) {
      setErr('No inspected training data. Please upload and inspect your data first.');
      return;
    }

    if (defsLoading) {
      setErr('Schema defaults are still loading. Please try again in a moment.');
      return;
    }

    if (algoOptions.length === 0) {
      setErr('Schema defaults not loaded: compatible algorithms are unavailable.');
      return;
    }

    if (!effectiveSplitMode) {
      setErr('Schema defaults not loaded: split defaults are unavailable.');
      return;
    }

    if (!effectiveBaseEstimator || !effectiveBaseEstimator.algo) {
      setErr('Please select a base estimator.');
      return;
    }

    setLoading(true);
    try {
      const payload = buildPayload();
      const result = await runEnsembleTrainRequest(payload);

      setTrainResult(result);
      setActiveResultKind('train');
      if (result?.artifact) setArtifact(result.artifact);

      setLoading(false);
    } catch (e) {
      setLoading(false);
      setErr(toErrorText(e));
    }
  };

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Text fw={700} size="lg">
              Bagging ensemble
            </Text>

            <Group gap="xs">
              <ActionIcon variant="subtle" onClick={handleReset} title="Reset to defaults">
                <IconRefresh size={18} />
              </ActionIcon>

              <SegmentedControl
                value={bagging.mode}
                onChange={(v) => setBagging({ mode: v })}
                data={[
                  { value: 'simple', label: 'Simple' },
                  { value: 'advanced', label: 'Advanced' },
                ]}
              />
            </Group>
          </Group>

          <Group justify="flex-end">
            <Button onClick={handleRun} loading={loading}>
              Train bagging ensemble
            </Button>
          </Group>

          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              {bagging.mode === 'simple' ? (
                <Select
                  label="Base estimator"
                  value={effectiveBaseEstimator?.algo || null}
                  onChange={(v) =>
                    setBaggingBaseEstimator(v ? getModelDefaults?.(v) || { algo: v } : undefined)
                  }
                  data={algoOptions}
                />
              ) : (
                <Box>
                  <ModelSelectionCard
                    model={effectiveBaseEstimator}
                    onChange={(next) => setBaggingBaseEstimator(next)}
                    schema={models?.schema}
                    enums={enums}
                    models={models}
                    showHelp={false}
                  />
                </Box>
              )}

              <NumberInput
                label="Number of estimators"
                min={1}
                step={1}
                value={dispNEstimators}
                onChange={(v) =>
                  setBagging({ n_estimators: v === '' || v == null ? undefined : v })
                }
                placeholder="default"
              />
            </Stack>

            <Box style={{ flex: 1, minWidth: 260 }}>
              <Stack justify="space-between" style={{ height: '100%' }} gap="xs">
                <Box>
                  <BaggingIntroText effectiveTask={effectiveTask} />
                </Box>

                <Group justify="flex-end">
                  <Button size="xs" variant="subtle" onClick={() => setShowHelp((p) => !p)}>
                    {showHelp ? 'Show less' : 'Show more'}
                  </Button>
                </Group>
              </Stack>
            </Box>
          </Group>

          {showHelp && (
            <Box>
              <EnsembleHelpText kind="bagging" effectiveTask={effectiveTask} mode={bagging.mode} />
            </Box>
          )}

          <Divider />

          {!metricIsAllowed && metricOverride && (
            <Alert color="yellow" variant="light">
              <Text size="sm" fw={600}>
                Metric not available for this task
              </Text>
              <Text size="sm">
                The selected metric (<strong>{metricOverride}</strong>) is not listed for the current task.
                {effectiveTask === 'regression' && defaultMetricFromSchema
                  ? ` Using '${defaultMetricFromSchema}' for this run.`
                  : ' Please update Settings → Metric.'}
              </Text>
            </Alert>
          )}

          {algoOptions.length === 0 && (
            <Alert color="yellow" variant="light">
              <Text size="sm" fw={600}>
                Schema defaults not loaded
              </Text>
              <Text size="sm">
                Bagging ensemble needs backend schema defaults to list compatible algorithms. Please wait for
                <strong> /api/v1/schema/defaults</strong> to load.
              </Text>
            </Alert>
          )}

          {!effectiveSplitMode && (
            <Alert color="yellow" variant="light">
              <Text size="sm" fw={600}>
                Split defaults not available
              </Text>
              <Text size="sm">
                This panel relies on backend split defaults to choose a split strategy. Please wait for
                <strong> /api/v1/schema/defaults</strong> to load.
              </Text>
            </Alert>
          )}

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Max samples (fraction)"
              step={0.1}
              min={0}
              max={1}
              value={dispMaxSamples}
              onChange={(v) => setBagging({ max_samples: v === '' || v == null ? undefined : v })}
              placeholder="default"
            />

            <NumberInput
              label="Max features (fraction)"
              step={0.1}
              min={0}
              max={1}
              value={dispMaxFeatures}
              onChange={(v) => setBagging({ max_features: v === '' || v == null ? undefined : v })}
              placeholder="default"
            />

            <NumberInput
              label="Number of jobs"
              step={1}
              value={dispNJobs}
              onChange={(v) => setBagging({ n_jobs: v === '' || v == null ? undefined : v })}
              placeholder="default"
            />

            <NumberInput
              label="Random state"
              step={1}
              value={dispRandomState}
              onChange={(v) => setBagging({ random_state: v === '' || v == null ? undefined : v })}
              placeholder="default"
            />
          </Group>

          <Group grow align="center" wrap="wrap">
            <Switch
              label="Bootstrap"
              checked={Boolean(dispBootstrap)}
              onChange={(e) => setBagging({ bootstrap: e.currentTarget.checked })}
            />
            <Switch
              label="Bootstrap features"
              checked={Boolean(dispBootstrapFeatures)}
              onChange={(e) => setBagging({ bootstrap_features: e.currentTarget.checked })}
            />
            <Switch
              label="Out-of-bag score"
              checked={Boolean(dispOobScore)}
              onChange={(e) => setBagging({ oob_score: e.currentTarget.checked })}
            />
            <Switch
              label="Balanced bagging"
              checked={Boolean(dispBalanced)}
              onChange={(e) => setBagging({ balanced: e.currentTarget.checked })}
            />
          </Group>

          {Boolean(dispBalanced) && (
            <Box mt="xs">
              <Group grow align="flex-end" wrap="wrap">
                <Select
                  label="Sampling strategy"
                  value={dispSamplingStrategy || null}
                  onChange={(v) => setBagging({ sampling_strategy: v || undefined })}
                  data={samplingStrategyOptions}
                  placeholder="default"
                />
                <Switch
                  label="Replacement"
                  checked={Boolean(dispReplacement)}
                  onChange={(e) => setBagging({ replacement: e.currentTarget.checked })}
                />
              </Group>
            </Box>
          )}
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={bagging.splitMode}
        onModeChange={(m) => setBagging({ splitMode: m })}
        trainFrac={bagging.trainFrac}
        onTrainFracChange={(v) => setBagging({ trainFrac: v })}
        nSplits={bagging.nSplits}
        onNSplitsChange={(v) => setBagging({ nSplits: v })}
        stratified={bagging.stratified}
        onStratifiedChange={(v) => setBagging({ stratified: v })}
        shuffle={bagging.shuffle}
        onShuffleChange={(v) => setBagging({ shuffle: v })}
        seed={bagging.seed}
        onSeedChange={(v) => setBagging({ seed: v })}
      />

      {trainResult?.ensemble_report?.kind === 'bagging' &&
        (trainResult.ensemble_report.task || 'classification') === 'regression' ? (
          <BaggingEnsembleRegressionResults report={trainResult.ensemble_report} />
        ) : (
          <BaggingEnsembleClassificationResults report={trainResult.ensemble_report} />
        )}

      {err && (
        <Alert color="red" variant="light">
          <Text fw={600}>Training failed</Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
            {err}
          </Text>
        </Alert>
      )}

      <Group justify="flex-end">
        <Button onClick={handleRun} loading={loading}>
          Train bagging ensemble
        </Button>
      </Group>

      <Alert color="blue" variant="light">
        <Text size="sm">
          This uses your current <strong>global</strong> Scaling / Metric / Features settings from the Settings
          section.
        </Text>
      </Alert>
    </Stack>
  );
}
