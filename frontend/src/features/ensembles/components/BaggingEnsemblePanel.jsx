import { useMemo, useState } from 'react';
import { Card, Stack, Group, Button, Divider, Alert, Text } from '@mantine/core';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import BaggingEnsembleClassificationResults from './BaggingEnsembleClassificationResults.jsx';
import BaggingEnsembleRegressionResults from './BaggingEnsembleRegressionResults.jsx';

import EnsemblePanelHeader from './common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from './common/EnsembleErrorAlert.jsx';

import BaggingConfigPane from './bagging/BaggingConfigPane.jsx';
import BaggingHelpPane from './bagging/BaggingHelpPane.jsx';

import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';

import { useEnsembleTrainRunner } from '../hooks/useEnsembleTrainRunner.js';
import { buildCommonEnsemblePayload, buildEnsembleTrainPayload } from '../utils/payload.js';
import { getAllowedMetrics, resolveMetricForPayload } from '../utils/metric.js';
import { intOrUndef, numOrUndef } from '../utils/coerce.js';

function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function algoLabelWithFallback(key) {
  const k = String(key || '');
  const lbl = getAlgoLabel(k);
  return lbl === k ? titleCase(k) : lbl;
}

export default function BaggingEnsemblePanel() {
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);

  const effectiveTask = useDataStore((s) => s.taskSelected || s.inspectReport?.task_inferred || null);

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

  const trainResult = useResultsStore((s) => s.trainResult);

  // ---- ensemble store slice ----
  const bagging = useEnsembleStore((s) => s.bagging);
  const setBagging = useEnsembleStore((s) => s.setBagging);
  const setBaggingBaseEstimator = useEnsembleStore((s) => s.setBaggingBaseEstimator);
  const resetBagging = useEnsembleStore((s) => s.resetBagging);

  // ---- local help toggle ----
  const [showHelp, setShowHelp] = useState(false);

  // ---- shared run state ----
  const { loading, err, setErr, runTrain } = useEnsembleTrainRunner();

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
    () => compatibleAlgos.map((a) => ({ value: a, label: algoLabelWithFallback(a) })),
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

  const allowedMetrics = useMemo(() => getAllowedMetrics(enums, effectiveTask), [enums, effectiveTask]);

  const metricForPayload = useMemo(
    () => resolveMetricForPayload({ metric, effectiveTask, allowedMetrics }),
    [metric, effectiveTask, allowedMetrics],
  );

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

    const common = buildCommonEnsemblePayload({
      dataInputs: { xPath, yPath, npzPath, xKey, yKey },
      splitInputs:
        effectiveSplitMode === 'kfold'
          ? {
              mode: 'kfold',
              nSplits: intOrUndef(bagging.nSplits),
              stratified: bagging.stratified,
              shuffle: bagging.shuffle,
            }
          : {
              mode: 'holdout',
              trainFrac: numOrUndef(bagging.trainFrac),
              stratified: bagging.stratified,
              shuffle: bagging.shuffle,
            },
      scaleMethod,
      featureCtx: fctx,
      evalInputs: {
        metric: metricForPayload,
        seed: intOrUndef(bagging.seed),
      },
    });

    const ensemble = {
      kind: 'bagging',
      base_estimator: effectiveBaseEstimator,

      n_estimators: intOrUndef(bagging.n_estimators),
      max_samples: numOrUndef(bagging.max_samples),
      max_features: numOrUndef(bagging.max_features),
      bootstrap: bagging.bootstrap,
      bootstrap_features: bagging.bootstrap_features,
      oob_score: bagging.oob_score,
      n_jobs: intOrUndef(bagging.n_jobs),
      random_state: intOrUndef(bagging.random_state),

      balanced: bagging.balanced,
      sampling_strategy: bagging.sampling_strategy,
      replacement: bagging.replacement,
    };

    return buildEnsembleTrainPayload({ common, ensemble });
  };

  const handleRun = async () => {
    setErr(null);

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

    if (effectiveTask === 'regression' && !metricForPayload) {
      setErr('No metric selected. Please choose a regression metric in Settings → Metric.');
      return;
    }

    await runTrain({ buildPayload });
  };

  const baggingReport = trainResult?.ensemble_report?.kind === 'bagging' ? trainResult.ensemble_report : null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <EnsemblePanelHeader
            title="Bagging ensemble"
            mode={bagging.mode}
            onModeChange={(v) => setBagging({ mode: v })}
            onReset={handleReset}
          />

          <Group justify="flex-end">
            <Button
              onClick={handleRun}
              loading={loading}
              disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
            >
              Train bagging ensemble
            </Button>
          </Group>

          <EnsembleErrorAlert error={err} />

          <BaggingConfigPane
            mode={bagging.mode}
            algoOptions={algoOptions}
            compatibleAlgos={compatibleAlgos}
            models={models}
            effectiveTask={effectiveTask}
            effectiveBaseEstimator={effectiveBaseEstimator}
            getModelDefaults={getModelDefaults}
            onBaseEstimatorChange={setBaggingBaseEstimator}
            onBaseEstimatorConfigChange={(next) => setBaggingBaseEstimator(next)}
            bagging={bagging}
            baggingDefaults={baggingDefaults}
            samplingStrategyOptions={samplingStrategyOptions}
            metricForPayload={metricForPayload}
            setBagging={setBagging}
          />

          <Divider my="xs" />

          <BaggingHelpPane showHelp={showHelp} onToggleHelp={() => setShowHelp((p) => !p)} />
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={effectiveSplitMode}
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

      {baggingReport?.task === 'classification' && (
        <BaggingEnsembleClassificationResults report={baggingReport} />
      )}

      {baggingReport?.task === 'regression' && <BaggingEnsembleRegressionResults report={baggingReport} />}

      <Group justify="flex-end">
        <Button
          onClick={handleRun}
          loading={loading}
          disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
        >
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
