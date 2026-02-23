// frontend/src/components/ensembles/AdaBoostEnsemblePanel.jsx
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
} from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';

import { useDataStore } from '../../state/useDataStore.js';
import { useSettingsStore } from '../../state/useSettingsStore.js';
import { useFeatureStore } from '../../state/useFeatureStore.js';
import { useResultsStore } from '../../state/useResultsStore.js';
import { useModelArtifactStore } from '../../state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../../state/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../../state/useEnsembleStore.js';

import { compactPayload } from '../../utils/compactPayload.js';

import SplitOptionsCard from '../SplitOptionsCard.jsx';
import ModelSelectionCard from '../ModelSelectionCard.jsx';

import { runEnsembleTrainRequest } from '../../api/ensembles.js';

import EnsembleHelpText, {
  AdaBoostIntroText,
} from '../helpers/helpTexts/EnsembleHelpText.jsx';
import AdaBoostEnsembleClassificationResults from './AdaBoostEnsembleClassificationResults.jsx';
import AdaBoostEnsembleRegressionResults from './AdaBoostEnsembleRegressionResults.jsx';

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

/** ---------- component ---------- **/

export default function AdaBoostEnsemblePanel() {
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
    getModelMeta,
    getCompatibleAlgos,
    getEnsembleDefaults,
  } = useSchemaDefaults();

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const trainResult = useResultsStore((s) => s.trainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const adaboost = useEnsembleStore((s) => s.adaboost);
  const setAdaBoost = useEnsembleStore((s) => s.setAdaBoost);
  const setAdaBoostBaseEstimator = useEnsembleStore((s) => s.setAdaBoostBaseEstimator);
  const resetAdaBoost = useEnsembleStore((s) => s.resetAdaBoost);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const adaboostDefaults = useMemo(
    () => getEnsembleDefaults?.('adaboost') ?? null,
    [getEnsembleDefaults],
  );

  const compatibleAlgos = useMemo(
    () => getCompatibleAlgos?.(effectiveTask) || [],
    [getCompatibleAlgos, effectiveTask],
  );

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoKeyToLabel(a) })),
    [compatibleAlgos],
  );

  const algorithmOptions = useMemo(
    () => [
      { value: '__default__', label: 'default' },
      { value: 'SAMME', label: 'SAMME' },
      { value: 'SAMME.R', label: 'SAMME.R' },
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

  // For regression we must avoid defaulting to a classification metric.
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
  const effectiveSplitMode = adaboost.splitMode ?? defaultSplitMode;

  // Base estimator: schema default (or first compatible) without mutating store.
  const defaultBaseAlgo =
    adaboostDefaults?.base_estimator?.algo || (Array.isArray(compatibleAlgos) ? compatibleAlgos[0] : null);

  const defaultBaseEstimator = useMemo(() => {
    if (!defaultBaseAlgo) return undefined;
    return getModelDefaults?.(defaultBaseAlgo) || { algo: defaultBaseAlgo };
  }, [defaultBaseAlgo, getModelDefaults]);

  const effectiveBaseEstimator = adaboost.base_estimator ?? defaultBaseEstimator;
  const baseAlgo = effectiveBaseEstimator?.algo || null;

  // If the backend provides capability metadata, reflect it; do not enforce.
  const baseMeta = baseAlgo ? getModelMeta?.(baseAlgo) : null;
  const baseSupportsSampleWeight =
    baseMeta && typeof baseMeta.supports_sample_weight === 'boolean'
      ? baseMeta.supports_sample_weight
      : undefined;

  const showSampleWeightWarning = baseSupportsSampleWeight === false;

  // Display values (defaults + overrides). Payload remains overrides-only.
  const dispNEstimators = adaboost.n_estimators ?? adaboostDefaults?.n_estimators;
  const dispLearningRate = adaboost.learning_rate ?? adaboostDefaults?.learning_rate;
  const dispRandomState = adaboost.random_state ?? adaboostDefaults?.random_state;

  const dispAlgorithm =
    effectiveTask === 'regression'
      ? undefined
      : (adaboost.algorithm ?? adaboostDefaults?.algorithm ?? undefined);

  const handleReset = () => {
    resetAdaBoost(effectiveTask);
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
            n_splits: adaboost.nSplits,
            stratified: adaboost.stratified,
            shuffle: adaboost.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: adaboost.trainFrac,
            stratified: adaboost.stratified,
            shuffle: adaboost.shuffle,
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
      adaboost.seed === '' || adaboost.seed == null
        ? undefined
        : Number.parseInt(String(adaboost.seed), 10);

    const evalCfg = compactPayload({
      metric: metricForPayload,
      seed: Number.isFinite(seedInt) ? seedInt : undefined,
    });

    const algoForPayload =
      effectiveTask === 'regression'
        ? undefined
        : (adaboost.algorithm && adaboost.algorithm !== '__default__' ? adaboost.algorithm : undefined);

    const ensemble = compactPayload({
      kind: 'adaboost',
      base_estimator: effectiveBaseEstimator,
      n_estimators: adaboost.n_estimators,
      learning_rate: adaboost.learning_rate,
      algorithm: algoForPayload,
      random_state: adaboost.random_state,
    });

    return { data, split: splitCfg, scale, features, ensemble, eval: evalCfg };
  };

  const handleRun = async () => {
    // Reset results immediately so old results don't linger after a failed run
    setTrainResult(null);
    setActiveResultKind(null);
    setArtifact(null);

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
              AdaBoost ensemble
            </Text>

            <Group gap="xs">
              <ActionIcon variant="subtle" onClick={handleReset} title="Reset to defaults">
                <IconRefresh size={18} />
              </ActionIcon>

              <SegmentedControl
                value={adaboost.mode}
                onChange={(v) => setAdaBoost({ mode: v })}
                data={[
                  { value: 'simple', label: 'Simple' },
                  { value: 'advanced', label: 'Advanced' },
                ]}
              />
            </Group>
          </Group>

          <Group justify="flex-end">
            <Button onClick={handleRun} loading={loading}>
              Train AdaBoost ensemble
            </Button>
          </Group>

          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              {adaboost.mode === 'simple' ? (
                <Select
                  label="Base estimator"
                  value={effectiveBaseEstimator?.algo || null}
                  onChange={(v) =>
                    setAdaBoostBaseEstimator(v ? (getModelDefaults?.(v) || { algo: v }) : undefined)
                  }
                  data={algoOptions}
                />
              ) : (
                <Box>
                  <ModelSelectionCard
                    model={effectiveBaseEstimator}
                    onChange={(next) => setAdaBoostBaseEstimator(next)}
                    schema={models?.schema}
                    enums={enums}
                    models={models}
                    showHelp={false}
                  />
                </Box>
              )}

              {showSampleWeightWarning && (
                <Alert color="yellow" variant="light">
                  <Text fw={600}>Base estimator may be incompatible</Text>
                  <Text size="sm">
                    This base estimator is marked as not supporting <b>sample_weight</b>, which AdaBoost
                    typically requires. If training fails, choose another base estimator or use Bagging/Voting.
                  </Text>
                </Alert>
              )}

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
                    AdaBoost ensemble needs backend schema defaults to list compatible algorithms. Please wait for
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
                  label="Number of estimators"
                  min={1}
                  step={1}
                  value={dispNEstimators}
                  onChange={(v) => setAdaBoost({ n_estimators: v === '' || v == null ? undefined : v })}
                  placeholder="default"
                />

                <NumberInput
                  label="Learning rate"
                  step={0.1}
                  min={0}
                  value={dispLearningRate}
                  onChange={(v) => setAdaBoost({ learning_rate: v === '' || v == null ? undefined : v })}
                  placeholder="default"
                />
              </Group>
            </Stack>

            <Box style={{ flex: 1, minWidth: 260 }}>
              <Stack justify="space-between" style={{ height: '100%' }} gap="xs">
                <Box>
                  <AdaBoostIntroText effectiveTask={effectiveTask} />
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
              <EnsembleHelpText kind="adaboost" effectiveTask={effectiveTask} mode={adaboost.mode} />
            </Box>
          )}

          <Divider />

          <Group grow align="flex-end" wrap="wrap">
            <Select
              label="Algorithm (classification only)"
              value={dispAlgorithm ?? '__default__'}
              onChange={(v) => setAdaBoost({ algorithm: !v || v === '__default__' ? undefined : v })}
              data={algorithmOptions}
              disabled={effectiveTask === 'regression'}
            />

            <NumberInput
              label="Random state"
              step={1}
              value={dispRandomState}
              onChange={(v) => setAdaBoost({ random_state: v === '' || v == null ? undefined : v })}
              placeholder="default"
            />
          </Group>
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={adaboost.splitMode}
        onModeChange={(m) => setAdaBoost({ splitMode: m })}
        trainFrac={adaboost.trainFrac}
        onTrainFracChange={(v) => setAdaBoost({ trainFrac: v })}
        nSplits={adaboost.nSplits}
        onNSplitsChange={(v) => setAdaBoost({ nSplits: v })}
        stratified={adaboost.stratified}
        onStratifiedChange={(v) => setAdaBoost({ stratified: v })}
        shuffle={adaboost.shuffle}
        onShuffleChange={(v) => setAdaBoost({ shuffle: v })}
        seed={adaboost.seed}
        onSeedChange={(v) => setAdaBoost({ seed: v })}
      />

      {trainResult?.ensemble_report?.kind === 'adaboost' &&
        (trainResult.ensemble_report.task || 'classification') === 'regression' ? (
          <AdaBoostEnsembleRegressionResults report={trainResult.ensemble_report} />
        ) : (
          <AdaBoostEnsembleClassificationResults report={trainResult.ensemble_report} />
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
          Train AdaBoost ensemble
        </Button>
      </Group>

      <Alert color="blue" variant="light">
        <Text size="sm">
          This uses your current <strong>global</strong> Scaling / Metric / Features settings from the Settings section.
        </Text>
      </Alert>
    </Stack>
  );
}
