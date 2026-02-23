import { useMemo, useState } from 'react';
import { Card, Stack, Group, Text, Button } from '@mantine/core';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import XGBoostEnsembleResults from './XGBoostEnsembleResults.jsx';

import EnsemblePanelHeader from './common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from './common/EnsembleErrorAlert.jsx';

import XGBoostConfigPane from './xgboost/XGBoostConfigPane.jsx';
import XGBoostHelpPane from './xgboost/XGBoostHelpPane.jsx';

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
            <XGBoostConfigPane
              mode={xgb.mode}
              xgb={xgb}
              xgbDefaults={xgbDefaults}
              setXGBoost={setXGBoost}
            />

            <XGBoostHelpPane
              showHelp={showHelp}
              onToggleHelp={() => setShowHelp((p) => !p)}
            />
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
