import { useEffect, useMemo, useRef, useState } from 'react';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';

import { useEnsembleTrainRunner } from './useEnsembleTrainRunner.js';
import { buildCommonEnsemblePayload, buildEnsembleTrainPayload } from '../utils/payload.js';
import { getAllowedMetrics, resolveMetricForPayload } from '../utils/metric.js';
import { intOrUndef, numOrUndef } from '../utils/coerce.js';
import { titleCase } from '../utils/resultsFormat.js';
import { dedupeWarning, normalizeWeight } from '../utils/voting.js';

function algoLabelWithFallback(key) {
  const k = String(key || '');
  const lbl = getAlgoLabel(k);
  return lbl === k ? titleCase(k) : lbl;
}

export function useVotingEnsembleController() {
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
  const voting = useEnsembleStore((s) => s.voting);
  const setVoting = useEnsembleStore((s) => s.setVoting);
  const setVotingEstimators = useEnsembleStore((s) => s.setVotingEstimators);
  const updateVotingEstimatorAt = useEnsembleStore((s) => s.updateVotingEstimatorAt);
  const removeVotingEstimatorAt = useEnsembleStore((s) => s.removeVotingEstimatorAt);
  const resetVoting = useEnsembleStore((s) => s.resetVoting);

  const { loading, err, setErr, runTrain } = useEnsembleTrainRunner();

  const [showHelp, setShowHelp] = useState(false);

  const initializedRef = useRef(false);

  // ----------------- derived -----------------

  const compatibleAlgos = useMemo(
    () => getCompatibleAlgos?.(effectiveTask) || [],
    [getCompatibleAlgos, effectiveTask],
  );

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoLabelWithFallback(a) })),
    [compatibleAlgos],
  );

  const estimators = Array.isArray(voting.estimators) ? voting.estimators : [];

  const duplicateAlgos = useMemo(() => dedupeWarning(voting.estimators), [voting.estimators]);

  const duplicateAlgosLabel =
    duplicateAlgos && duplicateAlgos.length > 0 ? duplicateAlgos.map(algoLabelWithFallback).join(', ') : '';

  // ----------------- schema-driven defaults (display + payload) -----------------

  const allowedMetrics = useMemo(() => getAllowedMetrics(enums, effectiveTask), [enums, effectiveTask]);

  const defaultMetricFromSchema = allowedMetrics?.[0] ?? undefined;
  const metricOverride = metric ? String(metric) : undefined;
  const metricIsAllowed = !metricOverride || allowedMetrics.length === 0 || allowedMetrics.includes(metricOverride);

  const metricForPayload = useMemo(
    () => resolveMetricForPayload({ metric, effectiveTask, allowedMetrics }),
    [metric, effectiveTask, allowedMetrics],
  );

  // Split defaults (schema-owned)
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;
  const defaultSplitMode = holdoutDefaults?.mode ?? kfoldDefaults?.mode ?? undefined;
  const effectiveSplitMode = voting.splitMode ?? defaultSplitMode;

  useEffect(() => {
    if (initializedRef.current) return;
    if (defsLoading) return;

    if (Array.isArray(voting.estimators) && voting.estimators.length >= 2) {
      initializedRef.current = true;
      return;
    }

    const ensembleDefaults = getEnsembleDefaults?.('voting') || null;
    const defaultsFromSchema = ensembleDefaults?.estimators || null;

    const firstAlgo = Array.isArray(compatibleAlgos) ? compatibleAlgos[0] : null;
    const fallbackModel = firstAlgo ? getModelDefaults?.(firstAlgo) || { algo: firstAlgo } : null;

    if (Array.isArray(defaultsFromSchema) && defaultsFromSchema.length >= 2) {
      setVotingEstimators(
        defaultsFromSchema.slice(0, 5).map((s) => ({
          name: s?.name ?? '',
          weight: s?.weight ?? '',
          model: s?.model ?? fallbackModel ?? undefined,
        })),
      );
      initializedRef.current = true;
      return;
    }

    // No explicit estimator list in schema defaults: pick the first compatible algos.
    if (Array.isArray(compatibleAlgos) && compatibleAlgos.length >= 2) {
      const base = compatibleAlgos.slice(0, 3);
      const init = base.slice(0, 3).map((algo) => ({
        name: '',
        weight: '',
        model: getModelDefaults?.(algo) || { algo },
      }));
      if (init.length >= 2) {
        setVotingEstimators(init);
        initializedRef.current = true;
        return;
      }
    }

    initializedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defsLoading, compatibleAlgos, getModelDefaults, getEnsembleDefaults]);

  // ----------------- actions -----------------

  const resetToDefaults = () => {
    resetVoting();
    initializedRef.current = false;
    setErr(null);
  };

  const clampEstimatorCount = (n) => {
    const target = Math.max(2, Number(n) || 2);

    const cur = Array.isArray(voting.estimators) ? voting.estimators : [];
    if (cur.length === target) return;

    if (cur.length > target) {
      setVotingEstimators(cur.slice(0, target));
      return;
    }

    const needed = target - cur.length;
    const extras = [];
    for (let i = 0; i < needed; i++) {
      const algo = compatibleAlgos[Math.min(cur.length + i, compatibleAlgos.length - 1)];
      if (!algo) break;
      extras.push({
        name: '',
        weight: '',
        model: getModelDefaults?.(algo) || { algo },
      });
    }
    if (extras.length) setVotingEstimators([...cur, ...extras]);
  };

  const addEstimator = () => {
    const cur = Array.isArray(voting.estimators) ? voting.estimators : [];

    const algo = compatibleAlgos[Math.min(cur.length, compatibleAlgos.length - 1)];
    if (!algo) {
      setErr('No compatible algorithms available. Ensure schema defaults are loaded.');
      return;
    }

    setVotingEstimators([
      ...cur,
      {
        name: '',
        weight: '',
        model: getModelDefaults?.(algo) || { algo },
      },
    ]);
  };

  const updateEstimatorAlgoSimple = (idx, algo) => {
    if (!algo) return;
    const base = getModelDefaults?.(algo) || { algo };
    updateVotingEstimatorAt(idx, { model: base });
  };

  const buildPayload = () => {
    if (!effectiveSplitMode) {
      throw new Error('Schema defaults not loaded: split mode is unavailable.');
    }

    const common = buildCommonEnsemblePayload({
      dataInputs: { xPath, yPath, npzPath, xKey, yKey },
      splitInputs:
        effectiveSplitMode === 'kfold'
          ? {
              mode: 'kfold',
              nSplits: intOrUndef(voting.nSplits),
              stratified: voting.stratified,
              shuffle: voting.shuffle,
            }
          : {
              mode: 'holdout',
              trainFrac: numOrUndef(voting.trainFrac),
              stratified: voting.stratified,
              shuffle: voting.shuffle,
            },
      scaleMethod,
      featureCtx: fctx,
      evalInputs: {
        metric: metricForPayload,
        seed: intOrUndef(voting.seed),
      },
    });

    const payloadEstimators = (voting.estimators || []).map((s) => {
      const nm = s?.name ? String(s.name).trim() : '';
      return {
        model: s?.model,
        name: nm || undefined,
        weight: normalizeWeight(s?.weight),
      };
    });

    const ensemble = {
      kind: 'voting',
      voting: voting.votingType,
      estimators: payloadEstimators,
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

    const cur = Array.isArray(voting.estimators) ? voting.estimators : [];
    if (cur.length < 2) {
      setErr('Voting ensemble requires at least 2 estimators.');
      return;
    }

    // Regression needs a regression metric; do not fall back to EvalModel.metric='accuracy'.
    if (effectiveTask === 'regression' && !metricForPayload) {
      setErr('No metric selected. Please choose a regression metric in Settings → Metric.');
      return;
    }

    for (let i = 0; i < cur.length; i++) {
      if (!cur[i]?.model?.algo) {
        setErr(`Estimator ${i + 1} is missing a model selection.`);
        return;
      }
    }

    await runTrain({ buildPayload });
  };

  const ensembleDefaults = getEnsembleDefaults?.('voting') || null;
  const defaultVotingType = ensembleDefaults?.voting;
  const effectiveVotingType = voting.votingType ?? defaultVotingType ?? null;

  const handleVotingTypeChange = (v) => {
    const next = v ? String(v) : undefined;
    if (defaultVotingType != null && next === String(defaultVotingType)) {
      setVoting({ votingType: undefined });
    } else {
      setVoting({ votingType: next });
    }
  };

  const trainDisabled = defsLoading || algoOptions.length === 0 || !effectiveSplitMode;

  return {
    // state
    voting,
    setVoting,
    models,
    enums,
    trainResult,

    // flags
    defsLoading,
    loading,
    trainDisabled,

    // derived
    effectiveTask,
    algoOptions,
    effectiveSplitMode,
    estimators,

    duplicateAlgosLabel,
    metricIsAllowed,
    metricOverride,
    defaultMetricFromSchema,

    effectiveVotingType,

    // help UI
    showHelp,
    toggleHelp: () => setShowHelp((p) => !p),

    // actions
    err,
    resetToDefaults,
    handleRun,
    clampEstimatorCount,
    addEstimator,
    updateEstimatorAlgoSimple,
    updateVotingEstimatorAt,
    removeVotingEstimatorAt,
    handleVotingTypeChange,
  };
}
