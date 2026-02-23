import { useMemo, useState } from 'react';
import {
  Card,
  Stack,
  Group,
  Text,
  Button,
  NumberInput,
  Divider,
  Box,
  Switch,
} from '@mantine/core';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

import EnsembleHelpText, {
  XGBoostIntroText,
} from '../../../shared/content/help/EnsembleHelpText.jsx';
import XGBoostEnsembleResults from './XGBoostEnsembleResults.jsx';

import EnsemblePanelHeader from './common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from './common/EnsembleErrorAlert.jsx';

import { useEnsembleTrainRunner } from '../hooks/useEnsembleTrainRunner.js';
import { buildCommonEnsemblePayload, buildEnsembleTrainPayload } from '../utils/payload.js';
import { getAllowedMetrics, resolveMetricForPayload } from '../utils/metric.js';
import { intOrUndef, numOrUndef } from '../utils/coerce.js';

export default function XGBoostEnsemblePanel() {
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

  const { enums, split, getEnsembleDefaults } = useSchemaDefaults();
  const xgbDefaults = getEnsembleDefaults?.('xgboost') ?? null;

  const trainResult = useResultsStore((s) => s.trainResult);

  const xgb = useEnsembleStore((s) => s.xgboost);
  const setXGBoost = useEnsembleStore((s) => s.setXGBoost);
  const resetXGBoost = useEnsembleStore((s) => s.resetXGBoost);

  const [showHelp, setShowHelp] = useState(false);

  const { loading, err, setErr, runTrain } = useEnsembleTrainRunner();

  const allowedMetrics = useMemo(
    () => getAllowedMetrics(enums, effectiveTask),
    [enums, effectiveTask],
  );

  const metricForPayload = useMemo(
    () => resolveMetricForPayload({ metric, effectiveTask, allowedMetrics }),
    [metric, effectiveTask, allowedMetrics],
  );

  // The run contract requires a split object; if user did not pick a mode,
  // follow the UI default (holdout).
  const splitMode = xgb.splitMode ?? 'holdout';

  // Split defaults (schema-owned)
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;
  const defaultTrainFrac = holdoutDefaults?.train_frac ?? undefined;
  const defaultNSplits = kfoldDefaults?.n_splits ?? undefined;

  const handleReset = () => {
    resetXGBoost(effectiveTask);
    setErr(null);
  };

  const buildPayload = () => {
    const common = buildCommonEnsemblePayload({
      dataInputs: { xPath, yPath, npzPath, xKey, yKey },
      splitInputs:
        splitMode === 'kfold'
          ? {
              mode: 'kfold',
              nSplits: intOrUndef(xgb.nSplits ?? defaultNSplits),
              stratified: xgb.stratified,
              shuffle: xgb.shuffle,
            }
          : {
              mode: 'holdout',
              trainFrac: numOrUndef(xgb.trainFrac ?? defaultTrainFrac),
              stratified: xgb.stratified,
              shuffle: xgb.shuffle,
            },
      scaleMethod,
      featureCtx: fctx,
      evalInputs: {
        metric: metricForPayload,
        seed: intOrUndef(xgb.seed),
      },
    });

    const ensemble = {
      kind: 'xgboost',
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

    return buildEnsembleTrainPayload({ common, ensemble });
  };

  const handleRun = async () => {
    setErr(null);

    if (effectiveTask === 'regression' && !metricForPayload) {
      setErr('No metric selected. Please choose a regression metric in Settings → Metric.');
      return;
    }

    await runTrain({ buildPayload });
  };

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <EnsemblePanelHeader
            title="XGBoost ensemble"
            mode={xgb.mode}
            onModeChange={(v) => setXGBoost({ mode: v })}
            onReset={handleReset}
          />

          <Group justify="flex-end">
            <Button onClick={handleRun} loading={loading}>
              Train XGBoost ensemble
            </Button>
          </Group>

          <EnsembleErrorAlert error={err} />

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
                  min={0.0001}
                  step={0.05}
                  value={xgb.learning_rate ?? xgbDefaults?.learning_rate ?? undefined}
                  onChange={(v) => setXGBoost({ learning_rate: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Max depth"
                  min={1}
                  step={1}
                  value={xgb.max_depth ?? xgbDefaults?.max_depth ?? undefined}
                  onChange={(v) => setXGBoost({ max_depth: v === '' ? undefined : v })}
                />
              </Group>

              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="Subsample"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={xgb.subsample ?? xgbDefaults?.subsample ?? undefined}
                  onChange={(v) => setXGBoost({ subsample: v === '' ? undefined : v })}
                />
                <NumberInput
                  label="Colsample (bytree)"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={xgb.colsample_bytree ?? xgbDefaults?.colsample_bytree ?? undefined}
                  onChange={(v) => setXGBoost({ colsample_bytree: v === '' ? undefined : v })}
                />
              </Group>

              {xgb.mode === 'advanced' && (
                <>
                  <Divider my="xs" label="Regularization" labelPosition="center" />

                  <Group grow align="flex-end" wrap="wrap">
                    <NumberInput
                      label="Lambda (L2)"
                      min={0}
                      step={0.1}
                      value={xgb.reg_lambda ?? xgbDefaults?.reg_lambda ?? undefined}
                      onChange={(v) => setXGBoost({ reg_lambda: v === '' ? undefined : v })}
                    />
                    <NumberInput
                      label="Alpha (L1)"
                      min={0}
                      step={0.1}
                      value={xgb.reg_alpha ?? xgbDefaults?.reg_alpha ?? undefined}
                      onChange={(v) => setXGBoost({ reg_alpha: v === '' ? undefined : v })}
                    />
                  </Group>

                  <Group grow align="flex-end" wrap="wrap">
                    <NumberInput
                      label="Min child weight"
                      min={0}
                      step={0.5}
                      value={xgb.min_child_weight ?? xgbDefaults?.min_child_weight ?? undefined}
                      onChange={(v) =>
                        setXGBoost({ min_child_weight: v === '' ? undefined : v })
                      }
                    />
                    <NumberInput
                      label="Gamma"
                      min={0}
                      step={0.1}
                      value={xgb.gamma ?? xgbDefaults?.gamma ?? undefined}
                      onChange={(v) => setXGBoost({ gamma: v === '' ? undefined : v })}
                    />
                  </Group>

                  <Divider my="xs" label="Early stopping" labelPosition="center" />

                  <Switch
                    label="Use early stopping"
                    checked={Boolean(xgb.use_early_stopping ?? xgbDefaults?.use_early_stopping)}
                    onChange={(e) =>
                      setXGBoost({ use_early_stopping: e.currentTarget.checked })
                    }
                  />

                  <Group grow align="flex-end" wrap="wrap">
                    <NumberInput
                      label="Early stopping rounds"
                      min={1}
                      step={1}
                      value={
                        xgb.early_stopping_rounds ??
                        xgbDefaults?.early_stopping_rounds ??
                        undefined
                      }
                      onChange={(v) =>
                        setXGBoost({
                          early_stopping_rounds: v === '' ? undefined : v,
                        })
                      }
                      disabled={!Boolean(xgb.use_early_stopping)}
                    />

                    <NumberInput
                      label="Eval set fraction"
                      min={0.05}
                      max={0.5}
                      step={0.05}
                      value={xgb.eval_set_fraction ?? xgbDefaults?.eval_set_fraction ?? undefined}
                      onChange={(v) =>
                        setXGBoost({
                          eval_set_fraction: v === '' ? undefined : v,
                        })
                      }
                      disabled={!Boolean(xgb.use_early_stopping)}
                    />
                  </Group>

                  <Divider my="xs" label="Other" labelPosition="center" />

                  <Group grow align="flex-end" wrap="wrap">
                    <NumberInput
                      label="n_jobs"
                      min={-1}
                      step={1}
                      value={xgb.n_jobs ?? xgbDefaults?.n_jobs ?? undefined}
                      onChange={(v) => setXGBoost({ n_jobs: v === '' ? undefined : v })}
                    />
                    <NumberInput
                      label="Random state"
                      min={0}
                      step={1}
                      value={xgb.random_state ?? xgbDefaults?.random_state ?? undefined}
                      onChange={(v) =>
                        setXGBoost({ random_state: v === '' ? undefined : v })
                      }
                    />
                  </Group>
                </>
              )}

              <SplitOptionsCard
                title="Data split"
                allowedModes={['holdout', 'kfold']}
                mode={xgb.splitMode}
                onModeChange={(v) => setXGBoost({ splitMode: v })}
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
            </Stack>

            <Box style={{ flex: 1, minWidth: 260 }}>
              <Stack gap="xs">
                <XGBoostIntroText />
                <Button size="xs" variant="subtle" onClick={() => setShowHelp((p) => !p)}>
                  {showHelp ? 'Show less' : 'Show more'}
                </Button>
                {showHelp && <EnsembleHelpText kind="xgboost" />}
              </Stack>
            </Box>
          </Group>
        </Stack>
      </Card>

      {trainResult && (
        <Card withBorder shadow="sm" radius="md" padding="lg">
          <Stack gap="md">
            <Text fw={700} size="lg">
              Results
            </Text>
            <XGBoostEnsembleResults result={trainResult} />
          </Stack>
        </Card>
      )}
    </Stack>
  );
}
