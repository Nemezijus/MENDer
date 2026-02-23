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
  TextInput,
} from '@mantine/core';
import { IconPlus, IconTrash, IconRefresh } from '@tabler/icons-react';

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
  VotingIntroText,
} from '../helpers/helpTexts/EnsembleHelpText.jsx';

import VotingEnsembleClassificationResults from './VotingEnsembleClassificationResults.jsx';
import VotingEnsembleRegressionResults from './VotingEnsembleRegressionResults.jsx';

/** ---------- helpers ---------- **/

// User-friendly names for the Algorithm dropdown.
// Values must remain the internal algo keys used by the backend.
const ALGO_LABELS = {
  // -------- classifiers --------
  logreg: 'Logistic Regression',
  ridge: 'Ridge Classifier',
  sgd: 'SGD Classifier',
  svm: 'Support Vector Machine (SVC)',
  tree: 'Decision Tree',
  forest: 'Random Forest',
  extratrees: 'Extra Trees Classifier',
  hgb: 'Histogram Gradient Boosting',
  knn: 'k-Nearest Neighbors',
  gnb: 'Gaussian Naive Bayes',

  // -------- regressors --------
  linreg: 'Linear Regression',
  ridgereg: 'Ridge Regression',
  ridgecv: 'Ridge Regression (CV)',
  enet: 'Elastic Net',
  enetcv: 'Elastic Net (CV)',
  lasso: 'Lasso',
  lassocv: 'Lasso (CV)',
  bayridge: 'Bayesian Ridge',
  svr: 'Support Vector Regression (SVR)',
  linsvr: 'Linear SVR',
  knnreg: 'k-Nearest Neighbors Regressor',
  treereg: 'Decision Tree Regressor',
  rfreg: 'Random Forest Regressor',
};

function algoKeyToLabel(algo) {
  if (!algo) return '';
  return ALGO_LABELS[algo] ?? String(algo);
}

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

function dedupeWarning(estimators) {
  const algos = (estimators || []).map((s) => s?.model?.algo).filter(Boolean);
  const set = new Set();
  const dup = new Set();
  for (const a of algos) {
    if (set.has(a)) dup.add(a);
    set.add(a);
  }
  return dup.size ? Array.from(dup) : null;
}

function normalizeWeight(v) {
  if (v === '' || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** ---------- component ---------- **/

export default function VotingEnsemblePanel() {
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
    getCompatibleAlgos,
    getEnsembleDefaults,
  } = useSchemaDefaults();

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const trainResult = useResultsStore((s) => s.trainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  // ---- ensemble store slice ----
  const voting = useEnsembleStore((s) => s.voting);
  const setVoting = useEnsembleStore((s) => s.setVoting);
  const setVotingEstimators = useEnsembleStore((s) => s.setVotingEstimators);
  const updateVotingEstimatorAt = useEnsembleStore((s) => s.updateVotingEstimatorAt);
  const removeVotingEstimatorAt = useEnsembleStore((s) => s.removeVotingEstimatorAt);
  const resetVoting = useEnsembleStore((s) => s.resetVoting);

  // ---- local run state + help toggle ----
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const initializedRef = useRef(false);

  // ----------------- derived -----------------

  const compatibleAlgos = useMemo(
    () => getCompatibleAlgos?.(effectiveTask) || [],
    [getCompatibleAlgos, effectiveTask],
  );

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoKeyToLabel(a) })),
    [compatibleAlgos],
  );

  const duplicateAlgos = useMemo(() => dedupeWarning(voting.estimators), [voting.estimators]);

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

  // Use backend-provided task ordering as a suggestion without writing into state.
  // For regression we must avoid falling back to EvalModel.metric='accuracy'.
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
    const fallbackModel = firstAlgo
      ? getModelDefaults?.(firstAlgo) || { algo: firstAlgo }
      : null;

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

    // DATA (override-only; empty-string keys are omitted)
    const data = compactPayload({
      x_path: npzPath ? undefined : xPath,
      y_path: npzPath ? undefined : yPath,
      npz_path: npzPath,
      x_key: xKey,
      y_key: yKey,
    });

    // SPLIT (override-only; if mode is unset, let backend defaults apply)
    const splitCfg = compactPayload(
      effectiveSplitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: voting.nSplits,
            stratified: voting.stratified,
            shuffle: voting.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: voting.trainFrac,
            stratified: voting.stratified,
            shuffle: voting.shuffle,
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
      voting.seed === '' || voting.seed == null
        ? undefined
        : Number.parseInt(String(voting.seed), 10);

    const evalCfg = compactPayload({
      metric: metricForPayload,
      seed: Number.isFinite(seedInt) ? seedInt : undefined,
    });

    // ESTIMATORS
    const estimators = (voting.estimators || []).map((s) => {
      const nm = s?.name ? String(s.name).trim() : '';
      return compactPayload({
        model: s?.model,
        name: nm || undefined,
        weight: normalizeWeight(s?.weight),
      });
    });

    const ensemble = compactPayload({
      kind: 'voting',
      voting: voting.votingType,
      estimators,
    });

    return { data, split: splitCfg, scale, features, ensemble, eval: evalCfg };
  };

  const handleRun = async () => {
    setErr(null);

    if (!inspectReport || inspectReport?.n_samples <= 0) {
      setErr('No inspected training data. Please upload and inspect your data first.');
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

  const renderSimpleEstimatorRow = (s, idx) => (
  <Group key={idx} align="flex-end" wrap="nowrap">
    <Select
      style={{ flex: 1, minWidth: 180, maxWidth: 360 }}
      label={`Estimator ${idx + 1}`}
      value={s?.model?.algo || null}
      onChange={(v) => updateEstimatorAlgoSimple(idx, v || s?.model?.algo)}
      data={algoOptions}
    />

    <ActionIcon
      variant="subtle"
      color="red"
      onClick={() => removeVotingEstimatorAt(idx)}
      disabled={estimators.length <= 2}
      title="Remove estimator"
    >
      <IconTrash size={18} />
    </ActionIcon>
  </Group>
);

  // ----------------- render -----------------

  const estimators = Array.isArray(voting.estimators) ? voting.estimators : [];

  const ensembleDefaults = getEnsembleDefaults?.('voting') || null;
  const defaultVotingType = ensembleDefaults?.voting;
  const effectiveVotingType = voting.votingType ?? defaultVotingType ?? null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Text fw={700} size="lg">
              Voting ensemble
            </Text>

            <Group gap="xs">
              <ActionIcon variant="subtle" onClick={resetToDefaults} title="Reset to defaults">
                <IconRefresh size={18} />
              </ActionIcon>

              <SegmentedControl
                value={voting.mode}
                onChange={(v) => setVoting({ mode: v })}
                data={[
                  { value: 'simple', label: 'Simple' },
                  { value: 'advanced', label: 'Advanced' },
                ]}
              />
            </Group>
          </Group>
          <Group justify="flex-end">
            <Button
              onClick={handleRun}
              loading={loading}
              disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
            >
              Train voting ensemble
            </Button>
          </Group>

          {/* First row: left A+B stacked, right C help preview */}
          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            {/* Left: A and B stacked */}
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              <Select
                label="Voting type"
                value={effectiveVotingType}
                onChange={(v) => {
                  const next = v ? String(v) : undefined;
                  if (defaultVotingType != null && next === String(defaultVotingType)) {
                    setVoting({ votingType: undefined });
                  } else {
                    setVoting({ votingType: next });
                  }
                }}
                data={[
                  { value: 'hard', label: 'Hard (labels)' },
                  { value: 'soft', label: 'Soft (probabilities)' },
                ]}
                disabled={effectiveTask === 'regression'}
                description={
                  effectiveTask === 'regression'
                    ? 'VotingRegressor is used for regression; voting type is ignored.'
                    : 'Soft voting requires all estimators to support predict_proba.'
                }
              />

              {voting.mode === 'simple' ? (
                <NumberInput
                  label="Number of models"
                  min={2}
                  step={1}
                  value={estimators.length}
                  onChange={clampEstimatorCount}
                  disabled={algoOptions.length === 0}
                />
              ) : (
                <Button
                  leftSection={<IconPlus size={16} />}
                  variant="light"
                  onClick={addEstimator}
                  disabled={algoOptions.length === 0}
                >
                  Add estimator
                </Button>
              )}
            </Stack>

            {/* Right: C help preview (same height as left stack) */}
            <Box style={{ flex: 1, minWidth: 260 }}>
                <Stack justify="space-between" style={{ height: '100%' }} gap="xs">
                    <Box>
                    <VotingIntroText effectiveTask={effectiveTask} votingType={voting.votingType} />
                    </Box>

                    <Group justify="flex-end">
                    <Button size="xs" variant="subtle" onClick={() => setShowHelp((p) => !p)}>
                        {showHelp ? 'Show less' : 'Show more'}
                    </Button>
                    </Group>
                </Stack>
            </Box>
          </Group>

          {/* Expanded help block between first row and estimator list */}
          {showHelp && (
            <Box>
              <EnsembleHelpText
                kind="voting"
                effectiveTask={effectiveTask}
                votingType={voting.votingType}
                mode={voting.mode}
              />
            </Box>
          )}

          {duplicateAlgos && duplicateAlgos.length > 0 && (
            <Alert color="yellow" variant="light">
              <Text size="sm" fw={600}>
                Duplicate estimator types detected
              </Text>
              <Text size="sm">
                You selected: <strong>{duplicateAlgos.map(algoKeyToLabel).join(', ')}</strong> more than once. If these are
                identical, this acts like implicit weighting. Prefer using explicit weights in Advanced mode.
              </Text>
            </Alert>
          )}

          <Divider />

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
                Voting ensemble needs backend schema defaults to list compatible algorithms. Please wait for
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

          {/* D row: estimator list */}
          {voting.mode === 'simple' && (
            <Stack gap="sm">
              <Text size="sm" c="dimmed">
                Simple mode uses each model’s default hyperparameters. Switch to Advanced to edit parameters
                and set weights.
              </Text>

              
                <Group align="flex-start" wrap="nowrap" gap="md">
                    <Stack style={{ flex: 1 }} gap="sm">
                        {estimators
                        .map((s, idx) => ({ s, idx }))
                        .filter((x) => x.idx % 2 === 0)
                        .map(({ s, idx }) => renderSimpleEstimatorRow(s, idx))}
                    </Stack>

                    <Stack style={{ flex: 1 }} gap="sm">
                        {estimators
                        .map((s, idx) => ({ s, idx }))
                        .filter((x) => x.idx % 2 === 1)
                        .map(({ s, idx }) => renderSimpleEstimatorRow(s, idx))}
                    </Stack>
                </Group>
              
            </Stack>
          )}

          {voting.mode === 'advanced' && (
            <Stack gap="md">
              <Text size="sm" c="dimmed">
                Advanced mode lets you tune each estimator and optionally assign weights.
              </Text>

              {estimators.map((s, idx) => (
                <Card key={idx} withBorder radius="md" p="md">
                  <Stack gap="sm">
                    <Group justify="space-between" align="center">
                      <Text fw={600}>Estimator {idx + 1}</Text>

                      <ActionIcon
                        variant="subtle"
                        color="red"
                        onClick={() => removeVotingEstimatorAt(idx)}
                        disabled={estimators.length <= 2}
                        title="Remove estimator"
                      >
                        <IconTrash size={18} />
                      </ActionIcon>
                    </Group>

                    <Group grow align="flex-end" wrap="wrap">
                      <TextInput
                        label="Name (optional)"
                        placeholder="auto"
                        value={s?.name ?? ''}
                        onChange={(e) =>
                          updateVotingEstimatorAt(idx, { name: e.currentTarget.value })
                        }
                      />

                      <NumberInput
                        label="Weight (optional)"
                        value={s?.weight ?? ''}
                        onChange={(v) => updateVotingEstimatorAt(idx, { weight: v })}
                        step={0.5}
                        min={0}
                      />
                    </Group>

                    <Box>
                      <ModelSelectionCard
                        model={s?.model}
                        onChange={(next) => updateVotingEstimatorAt(idx, { model: next })}
                        schema={models?.schema}
                        enums={enums}
                        models={models}
                        showHelp={false}
                      />
                    </Box>
                  </Stack>
                </Card>
              ))}
            </Stack>
          )}
        </Stack>
      </Card>

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={effectiveSplitMode}
        onModeChange={(m) => setVoting({ splitMode: m })}
        trainFrac={voting.trainFrac}
        onTrainFracChange={(v) => setVoting({ trainFrac: v })}
        nSplits={voting.nSplits}
        onNSplitsChange={(v) => setVoting({ nSplits: v })}
        stratified={voting.stratified}
        onStratifiedChange={(v) => setVoting({ stratified: v })}
        shuffle={voting.shuffle}
        onShuffleChange={(v) => setVoting({ shuffle: v })}
        seed={voting.seed}
        onSeedChange={(v) => setVoting({ seed: v })}
      />
      {trainResult?.ensemble_report?.kind === 'voting' &&
      trainResult.ensemble_report.task === 'classification' && (
        <VotingEnsembleClassificationResults report={trainResult.ensemble_report} />
      )}

      {trainResult?.ensemble_report?.kind === 'voting' &&
      trainResult.ensemble_report.task === 'regression' && (
        <VotingEnsembleRegressionResults report={trainResult.ensemble_report} />
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
        <Button
          onClick={handleRun}
          loading={loading}
          disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
        >
          Train voting ensemble
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
