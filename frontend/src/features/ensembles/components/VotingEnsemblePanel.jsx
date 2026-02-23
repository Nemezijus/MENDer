import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Card,
  Stack,
  Group,
  Text,
  Button,
  Select,
  NumberInput,
  Divider,
  Alert,
  Box,
  ActionIcon,
  TextInput,
} from '@mantine/core';
import { IconPlus, IconTrash } from '@tabler/icons-react';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useEnsembleStore } from '../state/useEnsembleStore.js';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../training/components/ModelSelectionCard.jsx';

import EnsembleHelpText, {
  VotingIntroText,
} from '../../../shared/content/help/EnsembleHelpText.jsx';

import VotingEnsembleClassificationResults from './VotingEnsembleClassificationResults.jsx';
import VotingEnsembleRegressionResults from './VotingEnsembleRegressionResults.jsx';

import EnsemblePanelHeader from './common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from './common/EnsembleErrorAlert.jsx';

import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';

import { useEnsembleTrainRunner } from '../hooks/useEnsembleTrainRunner.js';
import { buildCommonEnsemblePayload, buildEnsembleTrainPayload } from '../utils/payload.js';
import { getAllowedMetrics, resolveMetricForPayload } from '../utils/metric.js';
import { intOrUndef, numOrUndef } from '../utils/coerce.js';
import { dedupeWarning, normalizeWeight } from '../utils/voting.js';

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

export default function VotingEnsemblePanel() {
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

  const duplicateAlgos = useMemo(
    () => dedupeWarning(voting.estimators),
    [voting.estimators],
  );

  // ----------------- schema-driven defaults (display + payload) -----------------

  const allowedMetrics = useMemo(
    () => getAllowedMetrics(enums, effectiveTask),
    [enums, effectiveTask],
  );

  const defaultMetricFromSchema = allowedMetrics?.[0] ?? undefined;
  const metricOverride = metric ? String(metric) : undefined;
  const metricIsAllowed =
    !metricOverride || allowedMetrics.length === 0 || allowedMetrics.includes(metricOverride);

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

  const ensembleDefaults = getEnsembleDefaults?.('voting') || null;
  const defaultVotingType = ensembleDefaults?.voting;
  const effectiveVotingType = voting.votingType ?? defaultVotingType ?? null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <EnsemblePanelHeader
            title="Voting ensemble"
            mode={voting.mode}
            onModeChange={(v) => setVoting({ mode: v })}
            onReset={resetToDefaults}
          />

          <Group justify="flex-end">
            <Button
              onClick={handleRun}
              loading={loading}
              disabled={defsLoading || algoOptions.length === 0 || !effectiveSplitMode}
            >
              Train voting ensemble
            </Button>
          </Group>

          <EnsembleErrorAlert error={err} />

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
                  <VotingIntroText
                    effectiveTask={effectiveTask}
                    votingType={voting.votingType}
                  />
                </Box>

                <Group justify="flex-end">
                  <Button
                    size="xs"
                    variant="subtle"
                    onClick={() => setShowHelp((p) => !p)}
                  >
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
                You selected: <strong>{duplicateAlgos.map(algoLabelWithFallback).join(', ')}</strong> more than once. If these are
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
