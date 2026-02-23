import { useState } from 'react';
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
import { compactPayload } from '../../utils/compactPayload.js';

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

function numOrUndef(v) {
  if (v === '' || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function intOrUndef(v) {
  const n = numOrUndef(v);
  if (n === undefined) return undefined;
  return Math.trunc(n);
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
  const xgbDefaults = getEnsembleDefaults?.('xgboost') ?? null;

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

  const handleReset = () => {
    resetXGBoost(effectiveTask);
    setErr(null);
  };

  const buildPayload = () => {
    const data = {
      x_path: xPath || null,
      y_path: yPath || null,
      npz_path: npzPath || null,
      // empty string means "unset" (backend defaults to X / y)
      x_key: xKey,
      y_key: yKey,
    };

    // The run contract requires a split object; if user did not pick a mode,
    // we follow the UI default (holdout).
    const splitMode = xgb.splitMode ?? 'holdout';
    const split =
      splitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: intOrUndef(xgb.nSplits),
            stratified: xgb.stratified,
            shuffle: xgb.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: numOrUndef(xgb.trainFrac),
            stratified: xgb.stratified,
            shuffle: xgb.shuffle,
          };

    // Settings / features / eval are owned by backend defaults when unset.
    const scale = { method: scaleMethod };
    const features = buildFeaturesPayload(fctx);
    const evalCfg = {
      metric,
      seed: intOrUndef(xgb.seed),
      // decoder defaults are engine-owned
    };

    const ensemble = {
      kind: 'xgboost',
      // Only override when task implies regression; otherwise backend defaults apply.
      problem_kind: effectiveTask === 'regression' ? 'regression' : undefined,

      n_estimators: intOrUndef(xgb.n_estimators),
      learning_rate: numOrUndef(xgb.learning_rate),
      max_depth: intOrUndef(xgb.max_depth),
      subsample: numOrUndef(xgb.subsample),
      colsample_bytree: numOrUndef(xgb.colsample_bytree),

      reg_lambda: numOrUndef(xgb.reg_lambda),
      reg_alpha: numOrUndef(xgb.reg_alpha),

      min_child_weight: numOrUndef(xgb.min_child_weight),
      gamma: numOrUndef(xgb.gamma),

      use_early_stopping: xgb.use_early_stopping,
      early_stopping_rounds: intOrUndef(xgb.early_stopping_rounds),
      eval_set_fraction: numOrUndef(xgb.eval_set_fraction),

      n_jobs: intOrUndef(xgb.n_jobs),
      random_state: intOrUndef(xgb.random_state),
    };

    return compactPayload({ data, split, scale, features, ensemble, eval: evalCfg });
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
                  value={xgb.n_estimators ?? xgbDefaults?.n_estimators ?? undefined}
                  onChange={(v) => setXGBoost({ n_estimators: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Learning rate"
                  step={0.01}
                  min={0}
                  value={xgb.learning_rate ?? xgbDefaults?.learning_rate ?? undefined}
                  onChange={(v) => setXGBoost({ learning_rate: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Max depth"
                  step={1}
                  min={1}
                  value={xgb.max_depth ?? xgbDefaults?.max_depth ?? undefined}
                  onChange={(v) => setXGBoost({ max_depth: v === '' ? undefined : v })}
                />
              </Group>

              {/* Sampling */}
              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="Subsample"
                  step={0.05}
                  min={0}
                  max={1}
                  value={xgb.subsample ?? xgbDefaults?.subsample ?? undefined}
                  onChange={(v) => setXGBoost({ subsample: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Column sample by tree"
                  step={0.05}
                  min={0}
                  max={1}
                  value={xgb.colsample_bytree ?? xgbDefaults?.colsample_bytree ?? undefined}
                  onChange={(v) => setXGBoost({ colsample_bytree: v === '' ? undefined : v })}
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
                  value={xgb.reg_lambda ?? xgbDefaults?.reg_lambda ?? undefined}
                  onChange={(v) => setXGBoost({ reg_lambda: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="L1 (α)"
                  step={0.1}
                  min={0}
                  value={xgb.reg_alpha ?? xgbDefaults?.reg_alpha ?? undefined}
                  onChange={(v) => setXGBoost({ reg_alpha: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Min child weight"
                  step={0.5}
                  min={0}
                  value={xgb.min_child_weight ?? xgbDefaults?.min_child_weight ?? undefined}
                  onChange={(v) => setXGBoost({ min_child_weight: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Gamma"
                  step={0.1}
                  min={0}
                  value={xgb.gamma ?? xgbDefaults?.gamma ?? undefined}
                  onChange={(v) => setXGBoost({ gamma: v === '' ? undefined : v })}
                />
              </Group>

              <Divider />

              <Group align="flex-end" wrap="wrap" gap="md">
                {(() => {
                  const effectiveUseEarly =
                    xgb.use_early_stopping ?? xgbDefaults?.use_early_stopping ?? true;
                  const effectiveEvalFrac =
                    xgb.eval_set_fraction ?? xgbDefaults?.eval_set_fraction ?? undefined;
                  return (
                    <>
                <Switch
                  label="Use early stopping"
                  checked={Boolean(effectiveUseEarly)}
                  onChange={(e) =>
                    setXGBoost({ use_early_stopping: e.currentTarget.checked })
                  }
                />

                <NumberInput
                  label="Early stopping rounds (patience)"
                  min={1}
                  step={1}
                  value={
                    xgb.early_stopping_rounds ??
                    xgbDefaults?.early_stopping_rounds ??
                    undefined
                  }
                  onChange={(v) =>
                    setXGBoost({ early_stopping_rounds: v === '' ? undefined : v })
                  }
                  placeholder="auto"
                  disabled={!effectiveUseEarly}
                />

                <NumberInput
                  label="Eval set fraction"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  precision={2}
                  value={effectiveEvalFrac}
                  onChange={(v) =>
                    setXGBoost({ eval_set_fraction: v === '' ? undefined : v })
                  }
                  placeholder="0.20"
                  disabled={!effectiveUseEarly}
                />
                    </>
                  );
                })()}
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
              value={xgb.n_jobs ?? xgbDefaults?.n_jobs ?? undefined}
              onChange={(v) => setXGBoost({ n_jobs: v === '' ? undefined : v })}
              placeholder="default"
            />
            <NumberInput
              label="Random state"
              step={1}
              value={xgb.random_state ?? xgbDefaults?.random_state ?? undefined}
              onChange={(v) => setXGBoost({ random_state: v === '' ? undefined : v })}
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
