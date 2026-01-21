import { useEffect, useMemo, useRef, useState } from 'react';
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
import { useSettingsStore } from '../../state/useSettingsStore.js';
import { useFeatureStore } from '../../state/useFeatureStore.js';
import { useResultsStore } from '../../state/useResultsStore.js';
import { useModelArtifactStore } from '../../state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../../state/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../../state/useEnsembleStore.js';

import SplitOptionsCard from '../SplitOptionsCard.jsx';
import ModelSelectionCard from '../ModelSelectionCard.jsx';

import { runEnsembleTrainRequest } from '../../api/ensembles.js';

import EnsembleHelpText, { BaggingIntroText } from '../helpers/helpTexts/EnsembleHelpText.jsx';

import BaggingEnsembleClassificationResults from './BaggingEnsembleClassificationResults.jsx';
import BaggingEnsembleRegressionResults from './BaggingEnsembleRegressionResults.jsx';

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

function buildFeaturesPayload(fctx) {
  return {
    method: fctx.method,
    pca_n: fctx.pca_n,
    pca_var: fctx.pca_var,
    pca_whiten: fctx.pca_whiten,
    lda_n: fctx.lda_n,
    lda_solver: fctx.lda_solver,
    lda_shrinkage: fctx.lda_shrinkage,
    lda_tol: fctx.lda_tol,
    sfs_k: fctx.sfs_k,
    sfs_direction: fctx.sfs_direction,
    sfs_cv: fctx.sfs_cv,
    sfs_n_jobs: fctx.sfs_n_jobs,
  };
}

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
    getModelDefaults,
    getCompatibleAlgos,
    getEnsembleDefaults,
  } = useSchemaDefaults();

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const trainResult = useResultsStore((s) => s.trainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const bagging = useEnsembleStore((s) => s.bagging);
  const setBagging = useEnsembleStore((s) => s.setBagging);
  const setBaggingBaseEstimator = useEnsembleStore((s) => s.setBaggingBaseEstimator);
  const resetBagging = useEnsembleStore((s) => s.resetBagging);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const initializedRef = useRef(false);

  const compatibleAlgos = useMemo(() => {
    const list = getCompatibleAlgos?.(effectiveTask) || [];
    return list.length ? list : ['logreg', 'svm', 'tree', 'forest', 'knn', 'linreg'];
  }, [getCompatibleAlgos, effectiveTask]);

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

  useEffect(() => {
    if (initializedRef.current) return;
    if (defsLoading) return;

    if (bagging.base_estimator && bagging.base_estimator.algo) {
      initializedRef.current = true;
      return;
    }

    const algo = compatibleAlgos[0] || 'tree';
    setBaggingBaseEstimator(getModelDefaults?.(algo) || { algo });

    const defaults = getEnsembleDefaults?.('bagging') || null;
    if (defaults) {
      setBagging({
        n_estimators: defaults.n_estimators ?? bagging.n_estimators,
        max_samples: defaults.max_samples ?? bagging.max_samples,
        max_features: defaults.max_features ?? bagging.max_features,
        bootstrap: defaults.bootstrap ?? bagging.bootstrap,
        bootstrap_features: defaults.bootstrap_features ?? bagging.bootstrap_features,
        oob_score: defaults.oob_score ?? bagging.oob_score,
        n_jobs: defaults.n_jobs ?? bagging.n_jobs,
        random_state: defaults.random_state ?? bagging.random_state,

        // Balanced bagging (new)
        balanced: defaults.balanced ?? bagging.balanced ?? false,
        sampling_strategy: defaults.sampling_strategy ?? bagging.sampling_strategy ?? 'auto',
        replacement: defaults.replacement ?? bagging.replacement ?? false,
      });
    } else {
      // Ensure stable defaults even when schema defaults not loaded
      setBagging({
        balanced: bagging.balanced ?? false,
        sampling_strategy: bagging.sampling_strategy ?? 'auto',
        replacement: bagging.replacement ?? false,
      });
    }

    if (effectiveTask === 'regression') {
      if (bagging.stratified) setBagging({ stratified: false });
      if (bagging.balanced) setBagging({ balanced: false });
    }

    initializedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defsLoading, compatibleAlgos, getModelDefaults, getEnsembleDefaults]);

  const handleReset = () => {
    // Reset store values, then re-apply a default base estimator so the dropdown isn't empty
    initializedRef.current = false;
    resetBagging(effectiveTask);
    setErr(null);

    const algo = (getCompatibleAlgos?.(effectiveTask) || compatibleAlgos || [])[0] || 'tree';
    setBaggingBaseEstimator(getModelDefaults?.(algo) || { algo });
    initializedRef.current = true;
  };

  const buildPayload = () => {
    const data = {
      x_path: xPath || null,
      y_path: yPath || null,
      npz_path: npzPath || null,
      x_key: xKey || null,
      y_key: yKey || null,
    };

    const split =
      bagging.splitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: Number(bagging.nSplits) || 5,
            stratified: effectiveTask === 'regression' ? false : !!bagging.stratified,
            shuffle: !!bagging.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: Number(bagging.trainFrac) || 0.75,
            stratified: effectiveTask === 'regression' ? false : !!bagging.stratified,
            shuffle: !!bagging.shuffle,
          };

    const scale = { method: scaleMethod || 'standard' };
    const features = buildFeaturesPayload(fctx);

    const evalCfg = {
      metric: metric || (effectiveTask === 'regression' ? 'r2' : 'accuracy'),
      seed: bagging.seed === '' || bagging.seed == null ? null : Number(bagging.seed),
      n_shuffles: 0,
      progress_id: null,
      decoder: {
        enabled: true,
        include_decision_scores: true,
        include_probabilities: true,
        include_margin: true,
        positive_class_label: null,
        calibrate_probabilities: false,
        calibration_method: 'sigmoid',
        calibration_cv: 5,
        enable_export: true,
  },
    };

    const ensemble = {
      kind: 'bagging',
      problem_kind: effectiveTask === 'regression' ? 'regression' : 'classification',
      base_estimator: bagging.base_estimator,
      n_estimators: Number(bagging.n_estimators) || 10,
      max_samples: bagging.max_samples === '' ? null : bagging.max_samples,
      max_features: bagging.max_features === '' ? null : bagging.max_features,
      bootstrap: !!bagging.bootstrap,
      bootstrap_features: !!bagging.bootstrap_features,
      oob_score: !!bagging.oob_score,
      n_jobs: bagging.n_jobs === '' ? null : Number(bagging.n_jobs),
      random_state: bagging.random_state === '' ? null : Number(bagging.random_state),

      // Balanced bagging
      balanced: !!bagging.balanced,
      sampling_strategy: bagging.sampling_strategy || 'auto',
      replacement: !!bagging.replacement,
    };

    return { data, split, scale, features, ensemble, eval: evalCfg };
  };

  const handleRun = async () => {
    setErr(null);

    if (!inspectReport || inspectReport?.n_samples <= 0) {
      setErr('No inspected training data. Please upload and inspect your data first.');
      return;
    }

    if (!bagging.base_estimator || !bagging.base_estimator.algo) {
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
          {/* First row: left A+B stacked, right C help preview */}
          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              {bagging.mode === 'simple' ? (
                <Select
                  label="Base estimator"
                  value={bagging.base_estimator?.algo || null}
                  onChange={(v) =>
                    setBaggingBaseEstimator(getModelDefaults?.(v) || { algo: v || 'tree' })
                  }
                  data={algoOptions}
                />
              ) : (
                <Box>
                  <ModelSelectionCard
                    model={bagging.base_estimator}
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
                value={bagging.n_estimators}
                onChange={(v) => setBagging({ n_estimators: v })}
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

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Max samples (fraction)"
              step={0.1}
              min={0}
              max={1}
              value={bagging.max_samples}
              onChange={(v) => setBagging({ max_samples: v })}
              placeholder="default"
            />

            <NumberInput
              label="Max features (fraction)"
              step={0.1}
              min={0}
              max={1}
              value={bagging.max_features}
              onChange={(v) => setBagging({ max_features: v })}
              placeholder="default"
            />

            <NumberInput
              label="Number of jobs"
              step={1}
              value={bagging.n_jobs}
              onChange={(v) => setBagging({ n_jobs: v })}
              placeholder="default"
            />

            <NumberInput
              label="Random state"
              step={1}
              value={bagging.random_state}
              onChange={(v) => setBagging({ random_state: v })}
              placeholder="default"
            />
          </Group>

          <Group grow align="center" wrap="wrap">
            <Switch
              label="Bootstrap"
              checked={!!bagging.bootstrap}
              onChange={(e) => setBagging({ bootstrap: e.currentTarget.checked })}
            />
            <Switch
              label="Bootstrap features"
              checked={!!bagging.bootstrap_features}
              onChange={(e) => setBagging({ bootstrap_features: e.currentTarget.checked })}
            />
            <Switch
              label="Out-of-bag score"
              checked={!!bagging.oob_score}
              onChange={(e) => setBagging({ oob_score: e.currentTarget.checked })}
            />
            <Switch
              label="Balanced bagging"
              checked={!!bagging.balanced}
              disabled={effectiveTask === 'regression'}
              onChange={(e) => {
                const next = e.currentTarget.checked;
                // When enabling, ensure sensible defaults are present.
                setBagging({
                  balanced: next,
                  sampling_strategy: next ? bagging.sampling_strategy || 'auto' : bagging.sampling_strategy,
                  replacement: next ? !!bagging.replacement : bagging.replacement,
                });
              }}
            />
          </Group>

          {!!bagging.balanced && (
            <Box mt="xs">
              <Group grow align="flex-end" wrap="wrap">
                <Select
                  label="Sampling strategy"
                  value={bagging.sampling_strategy || 'auto'}
                  onChange={(v) => setBagging({ sampling_strategy: v || 'auto' })}
                  data={samplingStrategyOptions}
                  placeholder="auto"
                />
                <Switch
                  label="Replacement"
                  checked={!!bagging.replacement}
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
      {trainResult?.ensemble_report?.kind === 'bagging' && (
        (trainResult.ensemble_report.task || 'classification') === 'regression' ? (
          <BaggingEnsembleRegressionResults report={trainResult.ensemble_report} />
        ) : (
          <BaggingEnsembleClassificationResults report={trainResult.ensemble_report} />
        )
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
