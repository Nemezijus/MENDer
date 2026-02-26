import { useMemo, useState } from 'react';
import { Card, Stack, Group, Button, Divider, Alert, Text } from '@mantine/core';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import AdaBoostEnsembleClassificationResults from './AdaBoostEnsembleClassificationResults.jsx';
import AdaBoostEnsembleRegressionResults from './AdaBoostEnsembleRegressionResults.jsx';

import EnsemblePanelHeader from './common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from './common/EnsembleErrorAlert.jsx';

import AdaBoostConfigPane from './adaboost/AdaBoostConfigPane.jsx';
import AdaBoostHelpPane from './adaboost/AdaBoostHelpPane.jsx';

import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';

import { useEnsembleTrainRunner } from '../hooks/useEnsembleTrainRunner.js';
import { buildCommonEnsemblePayload, buildEnsembleTrainPayload } from '../utils/payload.js';
import { getAllowedMetrics, resolveMetricForPayload } from '../utils/metric.js';
import { intOrUndef, numOrUndef } from '../utils/coerce.js';

import { titleCase } from '../utils/resultsFormat.js';

function algoLabelWithFallback(key) {
  const k = String(key || '');
  const lbl = getAlgoLabel(k);
  return lbl === k ? titleCase(k) : lbl;
}

export default function AdaBoostEnsemblePanel() {
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
    getModelMeta,
    getCompatibleAlgos,
    getEnsembleDefaults,
  } = useSchemaDefaults();

  const trainResult = useResultsStore((s) => s.trainResult);

  // ---- ensemble store slice ----
  const adaboost = useEnsembleStore((s) => s.adaboost);
  const setAdaBoost = useEnsembleStore((s) => s.setAdaBoost);
  const setAdaBoostBaseEstimator = useEnsembleStore((s) => s.setAdaBoostBaseEstimator);
  const resetAdaBoost = useEnsembleStore((s) => s.resetAdaBoost);

  // ---- local help toggle ----
  const [showHelp, setShowHelp] = useState(false);

  // ---- shared run state ----
  const { loading, err, setErr, runTrain } = useEnsembleTrainRunner();

  const adaboostDefaults = useMemo(() => getEnsembleDefaults?.('adaboost') || null, [getEnsembleDefaults]);

  // ----------------- derived -----------------

  const compatibleAlgos = useMemo(
    () => getCompatibleAlgos?.(effectiveTask) || [],
    [getCompatibleAlgos, effectiveTask],
  );

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoLabelWithFallback(a) })),
    [compatibleAlgos],
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
  const effectiveSplitMode = adaboost.splitMode ?? defaultSplitMode;

  // Base estimator: schema default (or first compatible) without mutating store.
  const defaultBaseAlgo =
    adaboostDefaults?.base_estimator?.algo || (Array.isArray(compatibleAlgos) ? compatibleAlgos[0] : null);

  const defaultBaseEstimator = useMemo(() => {
    if (!defaultBaseAlgo) return undefined;
    return { algo: defaultBaseAlgo };
  }, [defaultBaseAlgo]);

  const effectiveBaseEstimator = adaboost.base_estimator ?? defaultBaseEstimator;
  const baseAlgo = effectiveBaseEstimator?.algo || null;

  // If the backend provides capability metadata, reflect it; do not enforce.
  const baseMeta = baseAlgo ? getModelMeta?.(baseAlgo) : null;
  const baseSupportsSampleWeight =
    baseMeta && typeof baseMeta.supports_sample_weight === 'boolean' ? baseMeta.supports_sample_weight : undefined;

  const showSampleWeightWarning = baseSupportsSampleWeight === false;

  const handleReset = () => {
    resetAdaBoost(effectiveTask);
    setErr(null);
  };

  const algorithmOptions = useMemo(() => {
    if (effectiveTask === 'regression') return [];
    // Backend accepts SAMME / SAMME.R; keep UI forgiving.
    return [
      { value: '__default__', label: 'Default (backend)' },
      { value: 'SAMME', label: 'SAMME' },
      { value: 'SAMME.R', label: 'SAMME.R' },
    ];
  }, [effectiveTask]);

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
              nSplits: intOrUndef(adaboost.nSplits),
              stratified: adaboost.stratified,
              shuffle: adaboost.shuffle,
            }
          : {
              mode: 'holdout',
              trainFrac: numOrUndef(adaboost.trainFrac),
              stratified: adaboost.stratified,
              shuffle: adaboost.shuffle,
            },
      scaleMethod,
      featureCtx: fctx,
      evalInputs: {
        metric: metricForPayload,
        seed: intOrUndef(adaboost.seed),
      },
    });

    const algoForPayload =
      effectiveTask === 'regression'
        ? undefined
        : adaboost.algorithm && adaboost.algorithm !== '__default__'
          ? adaboost.algorithm
          : undefined;

    const ensemble = {
      kind: 'adaboost',
      base_estimator: effectiveBaseEstimator,
      n_estimators: intOrUndef(adaboost.n_estimators),
      learning_rate: numOrUndef(adaboost.learning_rate),
      algorithm: algoForPayload,
      random_state: intOrUndef(adaboost.random_state),
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

    await runTrain({ buildPayload, clearPrevious: true });
  };

  const adaboostReport = trainResult?.ensemble_report?.kind === 'adaboost' ? trainResult.ensemble_report : null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <EnsemblePanelHeader
            title="AdaBoost ensemble"
            mode={adaboost.mode}
            onModeChange={(v) => setAdaBoost({ mode: v })}
            onReset={handleReset}
          />

          <Group justify="flex-end">
            <Button
              onClick={handleRun}
              loading={loading}
              disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
            >
              Train AdaBoost ensemble
            </Button>
          </Group>

          <EnsembleErrorAlert error={err} />

          <AdaBoostConfigPane
            mode={adaboost.mode}
            algoOptions={algoOptions}
            compatibleAlgos={compatibleAlgos}
            models={models}
            enums={enums}
            effectiveTask={effectiveTask}
            effectiveBaseEstimator={effectiveBaseEstimator}
            onBaseEstimatorChange={setAdaBoostBaseEstimator}
            onBaseEstimatorConfigChange={(next) => setAdaBoostBaseEstimator(next)}
            showSampleWeightWarning={showSampleWeightWarning}
            algorithmOptions={algorithmOptions}
            adaboost={adaboost}
            adaboostDefaults={adaboostDefaults}
            metricForPayload={metricForPayload}
            setAdaBoost={setAdaBoost}
          />

          <Divider my="xs" />

          <AdaBoostHelpPane showHelp={showHelp} onToggleHelp={() => setShowHelp((p) => !p)} />
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={effectiveSplitMode}
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

      {adaboostReport?.task === 'classification' && (
        <AdaBoostEnsembleClassificationResults report={adaboostReport} />
      )}

      {adaboostReport?.task === 'regression' && <AdaBoostEnsembleRegressionResults report={adaboostReport} />}

      <Group justify="flex-end">
        <Button
          onClick={handleRun}
          loading={loading}
          disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
        >
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
