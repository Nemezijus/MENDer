import { useEffect, useState } from 'react';
import {
  Card,
  Stack,
  Group,
  Text,
  Button,
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

import { runEnsembleTrainRequest } from '../../api/ensembles.js';

import EnsembleHelpText, { XGBoostIntroText } from '../helpers/helpTexts/EnsembleHelpText.jsx';
import XGBoostEnsembleResults from './XGBoostEnsembleResults.jsx';

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

function num(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function numOrNull(v) {
  if (v === '' || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export default function XGBoostEnsemblePanel() {
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

  const { getEnsembleDefaults } = useSchemaDefaults();

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const trainResult = useResultsStore((s) => s.trainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const xgb = useEnsembleStore((s) => s.xgboost);
  const setXGBoost = useEnsembleStore((s) => s.setXGBoost);
  const resetXGBoost = useEnsembleStore((s) => s.resetXGBoost);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    // one-time gentle hydrate from backend defaults if store still default-ish
    const defs = getEnsembleDefaults?.('xgboost') || null;
    if (!defs) return;

    // Only patch if user didn't touch values (heuristic: keep if already set)
    if (xgb.__hydrated) return;

    setXGBoost({
      n_estimators: defs.n_estimators ?? xgb.n_estimators,
      learning_rate: defs.learning_rate ?? xgb.learning_rate,
      max_depth: defs.max_depth ?? xgb.max_depth,
      subsample: defs.subsample ?? xgb.subsample,
      colsample_bytree: defs.colsample_bytree ?? xgb.colsample_bytree,
      reg_lambda: defs.reg_lambda ?? xgb.reg_lambda,
      reg_alpha: defs.reg_alpha ?? xgb.reg_alpha,
      min_child_weight: defs.min_child_weight ?? xgb.min_child_weight,
      gamma: defs.gamma ?? xgb.gamma,
      n_jobs: defs.n_jobs ?? xgb.n_jobs,
      random_state: defs.random_state ?? xgb.random_state,
      use_early_stopping: defs.use_early_stopping ?? xgb.use_early_stopping,
      early_stopping_rounds: defs.early_stopping_rounds ?? xgb.early_stopping_rounds,
      eval_set_fraction: defs.eval_set_fraction ?? xgb.eval_set_fraction,
      __hydrated: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getEnsembleDefaults]);

  const handleReset = () => {
    resetXGBoost(effectiveTask);
    setErr(null);
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
      xgb.splitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: Number(xgb.nSplits) || 5,
            stratified: effectiveTask === 'regression' ? false : !!xgb.stratified,
            shuffle: !!xgb.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: Number(xgb.trainFrac) || 0.75,
            stratified: effectiveTask === 'regression' ? false : !!xgb.stratified,
            shuffle: !!xgb.shuffle,
          };

    const scale = { method: scaleMethod || 'standard' };
    const features = buildFeaturesPayload(fctx);

    const evalCfg = {
      metric: metric || (effectiveTask === 'regression' ? 'r2' : 'accuracy'),
      seed: xgb.seed === '' || xgb.seed == null ? null : Number(xgb.seed),
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
      kind: 'xgboost',
      problem_kind: effectiveTask === 'regression' ? 'regression' : 'classification',

      n_estimators: num(xgb.n_estimators, 300),
      learning_rate: num(xgb.learning_rate, 0.1),
      max_depth: num(xgb.max_depth, 6),
      subsample: num(xgb.subsample, 1.0),
      colsample_bytree: num(xgb.colsample_bytree, 1.0),

      reg_lambda: num(xgb.reg_lambda, 1.0),
      reg_alpha: num(xgb.reg_alpha, 0.0),

      min_child_weight: num(xgb.min_child_weight, 1.0),
      gamma: num(xgb.gamma, 0.0),

      use_early_stopping: !!xgb.use_early_stopping,
      early_stopping_rounds: numOrNull(xgb.early_stopping_rounds),
      eval_set_fraction: num(xgb.eval_set_fraction, 0.2),

      n_jobs: numOrNull(xgb.n_jobs),
      random_state: numOrNull(xgb.random_state),
    };

    return { data, split, scale, features, ensemble, eval: evalCfg };
  };

  const handleRun = async () => {
    setErr(null);

    if (!inspectReport || inspectReport?.n_samples <= 0) {
      setErr('No inspected training data. Please upload and inspect your data first.');
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
              XGBoost ensemble
            </Text>

            <Group gap="xs">
              <ActionIcon variant="subtle" onClick={handleReset} title="Reset to defaults">
                <IconRefresh size={18} />
              </ActionIcon>

              <SegmentedControl
                value={xgb.mode}
                onChange={(v) => setXGBoost({ mode: v })}
                data={[
                  { value: 'simple', label: 'Simple' },
                  { value: 'advanced', label: 'Advanced' },
                ]}
              />
            </Group>
          </Group>

          <Group justify="flex-end">
            <Button onClick={handleRun} loading={loading}>
              Train XGBoost ensemble
            </Button>
          </Group>

          {/* First row: left parameters, right help preview */}
          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              {/* Core */}
              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="Estimators"
                  min={1}
                  step={10}
                  value={xgb.n_estimators}
                  onChange={(v) => setXGBoost({ n_estimators: v })}
                />
                <NumberInput
                  label="Learning rate"
                  step={0.01}
                  min={0}
                  value={xgb.learning_rate}
                  onChange={(v) => setXGBoost({ learning_rate: v })}
                />
                <NumberInput
                  label="Max depth"
                  step={1}
                  min={1}
                  value={xgb.max_depth}
                  onChange={(v) => setXGBoost({ max_depth: v })}
                />
              </Group>

              {/* Sampling */}
              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="Subsample"
                  step={0.05}
                  min={0}
                  max={1}
                  value={xgb.subsample}
                  onChange={(v) => setXGBoost({ subsample: v })}
                />
                <NumberInput
                  label="Column sample by tree"
                  step={0.05}
                  min={0}
                  max={1}
                  value={xgb.colsample_bytree}
                  onChange={(v) => setXGBoost({ colsample_bytree: v })}
                />
              </Group>
            </Stack>

            <Box style={{ flex: 1, minWidth: 260 }}>
              <Stack justify="space-between" style={{ height: '100%' }} gap="xs">
                <Box>
                  <XGBoostIntroText effectiveTask={effectiveTask} />
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
              <EnsembleHelpText kind="xgboost" effectiveTask={effectiveTask} mode={xgb.mode} />
            </Box>
          )}

          <Divider />

          {/* Advanced-only: Regularization / constraints + training diagnostics */}
          {xgb.mode === 'advanced' && (
            <Stack gap="sm">
              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="L2 (λ)"
                  step={0.1}
                  min={0}
                  value={xgb.reg_lambda}
                  onChange={(v) => setXGBoost({ reg_lambda: v })}
                />
                <NumberInput
                  label="L1 (α)"
                  step={0.1}
                  min={0}
                  value={xgb.reg_alpha}
                  onChange={(v) => setXGBoost({ reg_alpha: v })}
                />
                <NumberInput
                  label="Min child weight"
                  step={0.5}
                  min={0}
                  value={xgb.min_child_weight}
                  onChange={(v) => setXGBoost({ min_child_weight: v })}
                />
                <NumberInput
                  label="Gamma"
                  step={0.1}
                  min={0}
                  value={xgb.gamma}
                  onChange={(v) => setXGBoost({ gamma: v })}
                />
              </Group>

              <Divider />

              <Group align="flex-end" wrap="wrap" gap="md">
                <Switch
                  label="Use early stopping"
                  checked={!!xgb.use_early_stopping}
                  onChange={(e) => setXGBoost({ use_early_stopping: e.currentTarget.checked })}
                />

                <NumberInput
                  label="Early stopping rounds (patience)"
                  min={1}
                  step={1}
                  value={xgb.early_stopping_rounds}
                  onChange={(v) => setXGBoost({ early_stopping_rounds: v })}
                  placeholder="auto"
                  disabled={!xgb.use_early_stopping}
                />

                <NumberInput
                  label="Eval set fraction"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  precision={2}
                  value={xgb.eval_set_fraction}
                  onChange={(v) => setXGBoost({ eval_set_fraction: v })}
                  placeholder="0.20"
                  disabled={!xgb.use_early_stopping}
                />
              </Group>

              <Text size="xs" c="dimmed">
                Early stopping uses an internal validation split (from training data) to select the best
                boosting round and enable learning curves. This does not change your external train/test split.
              </Text>
            </Stack>
          )}

          {/* Misc */}
          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Number of jobs"
              step={1}
              value={xgb.n_jobs}
              onChange={(v) => setXGBoost({ n_jobs: v })}
              placeholder="default"
            />
            <NumberInput
              label="Random state"
              step={1}
              value={xgb.random_state}
              onChange={(v) => setXGBoost({ random_state: v })}
              placeholder="default"
            />
          </Group>
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={xgb.splitMode}
        onModeChange={(m) => setXGBoost({ splitMode: m })}
        trainFrac={xgb.trainFrac}
        onTrainFracChange={(v) => setXGBoost({ trainFrac: v })}
        nSplits={xgb.nSplits}
        onNSplitsChange={(v) => setXGBoost({ nSplits: v })}
        stratified={xgb.stratified}
        onStratifiedChange={(v) => setXGBoost({ stratified: v })}
        shuffle={xgb.shuffle}
        onShuffleChange={(v) => setXGBoost({ shuffle: v })}
        seed={xgb.seed}
        onSeedChange={(v) => setXGBoost({ seed: v })}
      />
      {trainResult?.ensemble_report?.kind === 'xgboost' && (
        <XGBoostEnsembleResults report={trainResult.ensemble_report} />
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
          Train XGBoost ensemble
        </Button>
      </Group>

      <Alert color="blue" variant="light">
        <Text size="sm">
          This uses your current <strong>global</strong> Scaling / Metric / Features settings from the
          Settings section.
        </Text>
      </Alert>
    </Stack>
  );
}
