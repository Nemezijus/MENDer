// frontend/src/components/ensembles/VotingEnsemblePanel.jsx
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

function normalizeWeight(v) {
  if (v === '' || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
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
  const setMetric = useSettingsStore((s) => s.setMetric);

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

  // ---- ensemble store slice ----
  const voting = useEnsembleStore((s) => s.voting);
  const setVoting = useEnsembleStore((s) => s.setVoting);
  const setVotingEstimators = useEnsembleStore((s) => s.setVotingEstimators);
  const updateVotingEstimatorAt = useEnsembleStore((s) => s.updateVotingEstimatorAt);
  const removeVotingEstimatorAt = useEnsembleStore((s) => s.removeVotingEstimatorAt);

  // ---- local run state + help toggle ----
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const initializedRef = useRef(false);

  // ----------------- derived -----------------

  const compatibleAlgos = useMemo(() => {
    const list = getCompatibleAlgos?.(effectiveTask) || [];
    return list.length ? list : ['logreg', 'svm', 'tree', 'forest', 'knn', 'linreg'];
  }, [getCompatibleAlgos, effectiveTask]);

  const algoOptions = useMemo(
    () => compatibleAlgos.map((a) => ({ value: a, label: algoKeyToLabel(a) })),
    [compatibleAlgos],
  );

  const duplicateAlgos = useMemo(() => dedupeWarning(voting.estimators), [voting.estimators]);

  // ----------------- guardrails -----------------

  useEffect(() => {
    if (!enums) return;

    const metricByTask = enums.MetricByTask || null;

    let rawList;
    if (metricByTask && effectiveTask && Array.isArray(metricByTask[effectiveTask])) {
      rawList = metricByTask[effectiveTask];
    } else if (Array.isArray(enums.MetricName)) {
      rawList = enums.MetricName;
    } else {
      rawList = ['accuracy', 'balanced_accuracy', 'f1_macro'];
    }

    if (!rawList || rawList.length === 0) return;

    if (!metric || !rawList.includes(metric)) {
      const first = rawList[0];
      if (first != null) setMetric(String(first));
    }
  }, [effectiveTask, enums, metric, setMetric]);

  useEffect(() => {
    if (initializedRef.current) return;
    if (defsLoading) return;

    if (Array.isArray(voting.estimators) && voting.estimators.length >= 2) {
      initializedRef.current = true;
      return;
    }

    const pick = compatibleAlgos.slice(0, 3);
    const base = pick.length ? pick : ['logreg', 'tree', 'svm'];

    const init = base.slice(0, 3).map((algo) => ({
      name: '',
      weight: '',
      model: getModelDefaults?.(algo) || { algo },
    }));

    if (init.length < 2) {
      init.push({
        name: '',
        weight: '',
        model: getModelDefaults?.(compatibleAlgos[0]) || { algo: compatibleAlgos[0] || 'logreg' },
      });
    }

    setVotingEstimators(init);

    if (effectiveTask === 'regression' && voting.stratified) {
      setVoting({ stratified: false });
    }

    initializedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defsLoading, compatibleAlgos, getModelDefaults]);

  useEffect(() => {
    if (effectiveTask === 'regression' && voting.stratified) {
      setVoting({ stratified: false });
    }
  }, [effectiveTask, voting.stratified, setVoting]);

  // ----------------- actions -----------------

  const resetToDefaults = () => {
    const ensembleDefaults = getEnsembleDefaults?.('voting') || null;

    const nextVotingType = ensembleDefaults?.voting || 'hard';

    setVoting({
      mode: 'simple',
      votingType: nextVotingType,
      splitMode: 'holdout',
      trainFrac: 0.75,
      nSplits: 5,
      stratified: effectiveTask !== 'regression',
      shuffle: true,
      seed: '',
    });

    const defaultsFromBackend = ensembleDefaults?.estimators || null;

    if (Array.isArray(defaultsFromBackend) && defaultsFromBackend.length >= 2) {
      setVotingEstimators(
        defaultsFromBackend.slice(0, 5).map((s) => ({
          name: s?.name ?? '',
          weight: s?.weight ?? '',
          model: s?.model ?? { algo: 'logreg' },
        })),
      );
    } else {
      const base = (compatibleAlgos.length ? compatibleAlgos : ['logreg', 'tree', 'svm']).slice(0, 3);
      setVotingEstimators(
        base.slice(0, 3).map((algo) => ({
          name: '',
          weight: '',
          model: getModelDefaults?.(algo) || { algo },
        })),
      );
    }

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
      const algo =
        compatibleAlgos[Math.min(cur.length + i, compatibleAlgos.length - 1)] || 'logreg';
      extras.push({
        name: '',
        weight: '',
        model: getModelDefaults?.(algo) || { algo },
      });
    }
    setVotingEstimators([...cur, ...extras]);
  };

  const addEstimator = () => {
    const cur = Array.isArray(voting.estimators) ? voting.estimators : [];

    const algo = compatibleAlgos[Math.min(cur.length, compatibleAlgos.length - 1)] || 'logreg';

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
    const base = getModelDefaults?.(algo) || { algo };
    updateVotingEstimatorAt(idx, { model: base });
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
      voting.splitMode === 'kfold'
        ? {
            mode: 'kfold',
            n_splits: Number(voting.nSplits) || 5,
            stratified: !!voting.stratified,
            shuffle: !!voting.shuffle,
          }
        : {
            mode: 'holdout',
            train_frac: Number(voting.trainFrac) || 0.75,
            stratified: !!voting.stratified,
            shuffle: !!voting.shuffle,
          };

    const scale = { method: scaleMethod || 'standard' };
    const features = buildFeaturesPayload(fctx);

    const evalCfg = {
      metric: metric || 'accuracy',
      seed: voting.seed === '' || voting.seed == null ? null : Number(voting.seed),
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

    const estimators = (voting.estimators || []).map((s) => ({
      model: s.model,
      name: s.name ? String(s.name).trim() || null : null,
      weight: normalizeWeight(s.weight),
    }));

    const ensemble = {
      kind: 'voting',
      voting: voting.votingType,
      flatten_transform: true,
      estimators,
    };

    return { data, split, scale, features, ensemble, eval: evalCfg };
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
      onChange={(v) => updateEstimatorAlgoSimple(idx, v || 'logreg')}
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
            <Button onClick={handleRun} loading={loading}>
                Train voting ensemble
            </Button>
        </Group>

          {/* First row: left A+B stacked, right C help preview */}
          <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
            {/* Left: A and B stacked */}
            <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
              <Select
                label="Voting type"
                value={voting.votingType}
                onChange={(v) => setVoting({ votingType: v || 'hard' })}
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
                />
              ) : (
                <Button
                  leftSection={<IconPlus size={16} />}
                  variant="light"
                  onClick={addEstimator}
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
        mode={voting.splitMode}
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
        <Button onClick={handleRun} loading={loading}>
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
