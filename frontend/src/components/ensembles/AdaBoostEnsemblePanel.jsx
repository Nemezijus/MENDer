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

import EnsembleHelpText, {
  AdaBoostIntroText,
} from '../helpers/helpTexts/EnsembleHelpText.jsx';
import AdaBoostEnsembleResults from './AdaBoostEnsembleResults.jsx';

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

function numOrNull(v) {
  if (v === '' || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

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
    getModelDefaults,
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

  const initializedRef = useRef(false);

  const compatibleAlgos = useMemo(() => {
    const list = getCompatibleAlgos?.(effectiveTask) || [];
    return list.length ? list : ['tree', 'logreg', 'svm', 'forest', 'knn', 'linreg'];
  }, [getCompatibleAlgos, effectiveTask]);

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: a })),
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

  const baseAlgo = adaboost.base_estimator?.algo || null;
  const isKnnBase = baseAlgo === 'knn';

  useEffect(() => {
    if (initializedRef.current) return;
    if (defsLoading) return;

    // base estimator
    if (!adaboost.base_estimator) {
      const algo = compatibleAlgos[0] || 'tree';
      setAdaBoostBaseEstimator(getModelDefaults?.(algo) || { algo });
    }

    // hydrate defaults from backend schema if available
    const defs = getEnsembleDefaults?.('adaboost') || null;
    if (defs) {
      setAdaBoost({
        n_estimators: defs.n_estimators ?? adaboost.n_estimators,
        learning_rate: defs.learning_rate ?? adaboost.learning_rate,
        algorithm: defs.algorithm ?? adaboost.algorithm,
        random_state: defs.random_state ?? adaboost.random_state,
      });
    }

    // regression guardrail
    if (effectiveTask === 'regression' && adaboost.stratified) {
      setAdaBoost({ stratified: false });
    }
    initializedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defsLoading, compatibleAlgos, getModelDefaults, getEnsembleDefaults, effectiveTask]);

  const handleReset = () => {
    resetAdaBoost(effectiveTask);
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
      adaboost.splitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: Number(adaboost.nSplits) || 5,
            stratified: effectiveTask === 'regression' ? false : !!adaboost.stratified,
            shuffle: !!adaboost.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: Number(adaboost.trainFrac) || 0.75,
            stratified: effectiveTask === 'regression' ? false : !!adaboost.stratified,
            shuffle: !!adaboost.shuffle,
          };

    const scale = { method: scaleMethod || 'standard' };
    const features = buildFeaturesPayload(fctx);

    const evalCfg = {
      metric: metric || (effectiveTask === 'regression' ? 'r2' : 'accuracy'),
      seed: adaboost.seed === '' || adaboost.seed == null ? null : Number(adaboost.seed),
      n_shuffles: 0,
      progress_id: null,
    };

    const ensemble = {
      kind: 'adaboost',
      problem_kind: effectiveTask === 'regression' ? 'regression' : 'classification',
      base_estimator: adaboost.base_estimator,
      n_estimators: Number(adaboost.n_estimators) || 50,
      learning_rate: Number(adaboost.learning_rate) || 1.0,
      algorithm:
        effectiveTask === 'regression'
          ? null
          : adaboost.algorithm === '__default__'
          ? null
          : adaboost.algorithm || null,
      random_state: numOrNull(adaboost.random_state),
    };

    return { data, split, scale, features, ensemble, eval: evalCfg };
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

    if (!adaboost.base_estimator?.algo) {
      setErr('Please select a base estimator.');
      return;
    }

    // Guardrail: KNN can't be used for AdaBoost (no sample_weight support)
    if (adaboost.base_estimator?.algo === 'knn') {
      setErr(
        "KNN can't be used as an AdaBoost base estimator because it doesn't support sample_weight. Choose another base estimator (e.g., tree/logreg/svm) or use Bagging/Voting for KNN.",
      );
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

          {/* First row: left settings stack, right help preview */}
          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              {adaboost.mode === 'simple' ? (
                <Select
                  label="Base estimator"
                  value={adaboost.base_estimator?.algo || null}
                  onChange={(v) =>
                    setAdaBoostBaseEstimator(getModelDefaults?.(v) || { algo: v || 'tree' })
                  }
                  data={algoOptions}
                />
              ) : (
                <Box>
                  <ModelSelectionCard
                    model={adaboost.base_estimator}
                    onChange={(next) => setAdaBoostBaseEstimator(next)}
                    schema={models?.schema}
                    enums={enums}
                    models={models}
                    showHelp={false}
                  />
                </Box>
              )}

              {isKnnBase && (
                <Alert color="yellow" variant="light">
                  <Text fw={600}>KNN is not supported for AdaBoost</Text>
                  <Text size="sm">
                    AdaBoost requires base estimators to support <b>sample_weight</b>. KNN does not,
                    so training will fail. Use Bagging/Voting if you want KNN in an ensemble.
                  </Text>
                </Alert>
              )}

              <Group grow align="flex-end" wrap="wrap">
                <NumberInput
                  label="Number of estimators"
                  min={1}
                  step={1}
                  value={adaboost.n_estimators}
                  onChange={(v) => setAdaBoost({ n_estimators: v })}
                />

                <NumberInput
                  label="Learning rate"
                  step={0.1}
                  min={0}
                  value={adaboost.learning_rate}
                  onChange={(v) => setAdaBoost({ learning_rate: v })}
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
              value={adaboost.algorithm ?? '__default__'}
              onChange={(v) => setAdaBoost({ algorithm: v || '__default__' })}
              data={algorithmOptions}
              disabled={effectiveTask === 'regression'}
            />

            <NumberInput
              label="Random state"
              step={1}
              value={adaboost.random_state}
              onChange={(v) => setAdaBoost({ random_state: v })}
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

      {trainResult?.ensemble_report?.kind === 'adaboost' && (
        <AdaBoostEnsembleResults report={trainResult.ensemble_report} />
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
