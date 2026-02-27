import { useEffect, useMemo, useRef } from 'react';
import {
  Card, Button, Text,
  Stack, Group, Divider, Alert, Title, Box, Progress
} from '@mantine/core';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';
import { useTrainingStore } from '../state/useTrainingStore.js';
import { getDefaultSplitMode } from '../../../shared/utils/splitMode.js';

import { useSingleModelTrainer } from '../hooks/useSingleModelTrainer.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import ShuffleLabelsCard from './ShuffleLabelsCard.jsx';
import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

/** ---------- component ---------- **/

export default function SingleModelTrainingPanel() {
  const {
    loading: defsLoading,
    models,
    enums,
    split,
    eval: evalCfg,
    getCompatibleAlgos,
  } = useSchemaDefaults();

  // SPLIT (override-only; persists across navigation)
  const splitMode = useTrainingStore((s) => s.splitMode);
  const setSplitMode = useTrainingStore((s) => s.setSplitMode);
  const trainFrac = useTrainingStore((s) => s.trainFrac);
  const setTrainFrac = useTrainingStore((s) => s.setTrainFrac);
  const nSplits = useTrainingStore((s) => s.nSplits);
  const setNSplits = useTrainingStore((s) => s.setNSplits);
  const stratified = useTrainingStore((s) => s.stratified);
  const setStratified = useTrainingStore((s) => s.setStratified);
  const shuffle = useTrainingStore((s) => s.shuffle);
  const setShuffle = useTrainingStore((s) => s.setShuffle);
  const seed = useTrainingStore((s) => s.seed);
  const setSeed = useTrainingStore((s) => s.setSeed);

  // Shuffle baseline (also persists)
  const useShuffleBaseline = useTrainingStore((s) => s.useShuffleBaseline);
  const setUseShuffleBaseline = useTrainingStore((s) => s.setUseShuffleBaseline);
  const nShuffles = useTrainingStore((s) => s.nShuffles);
  const setNShuffles = useTrainingStore((s) => s.setNShuffles);

  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  // Per-panel model config (train slice)
  const trainModel = useModelConfigStore((s) => s.train);
  const setTrainModel = useModelConfigStore((s) => s.setTrainModel);

  // Runner (request + progress)
  const {
    dataReady,
    isRunning,
    error,
    setError,
    progress,
    progressLabel,
    runTraining,
  } = useSingleModelTrainer();

  // Zustand results store: train result slice
  const trainResult = useResultsStore((s) => s.trainResult);
  const clearTrainResult = useResultsStore((s) => s.clearTrainResult);

  // Model artifact store (Zustand)
  const artifact = useModelArtifactStore((s) => s.artifact);
  const artifactSource = useModelArtifactStore((s) => s.source);

  const lastHydratedUid = useRef(null);

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  // ---------------------------------------------------------------------
  // Split defaults (schema-owned) — computed early so they can be referenced
  // safely from effects declared below.
  // ---------------------------------------------------------------------
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;
  const defaultSplitMode = getDefaultSplitMode({ split, allowedModes: ['holdout', 'kfold'] });
  const effectiveMode = splitMode ?? defaultSplitMode;
  const defaultStratified =
    effectiveMode === 'kfold' ? kfoldDefaults?.stratified : holdoutDefaults?.stratified;
  const defaultShuffle =
    effectiveMode === 'kfold' ? kfoldDefaults?.shuffle : holdoutDefaults?.shuffle;
  // NOTE: effective shuffle (for payload seed rules) is handled inside the trainer hook.

  // ---------------------------------------------------------------------
  // Shuffle-baseline defaults (schema-owned)
  // ---------------------------------------------------------------------
  const evalDefaults = evalCfg?.defaults ?? null;
  const shuffleBaselineDefaults = evalDefaults?.shuffle_baseline ?? null;
  const defaultShuffleBaselineEnabled =
    shuffleBaselineDefaults?.enabled ?? (Number(evalDefaults?.n_shuffles ?? 0) > 0);
  const defaultNShuffles = shuffleBaselineDefaults?.n_shuffles ?? null;

  const effectiveShuffleBaselineEnabled =
    useShuffleBaseline ?? Boolean(defaultShuffleBaselineEnabled);
  const effectiveNShuffles = nShuffles ?? defaultNShuffles;


  // Initialize train model once defaults arrive
  useEffect(() => {
    if (!defsLoading && !trainModel) {
      setTrainModel({ algo: 'logreg' });
    }
  }, [defsLoading, trainModel, setTrainModel]);

  useEffect(() => {
    if (!trainModel) return;
    if (!models) return; // schema/defaults not ready yet

    const currentAlgo = trainModel.algo;
    const compat = getCompatibleAlgos(effectiveTask); // [] if none / unknown task

    if (!compat || compat.length === 0) {
      // Nothing to filter against; leave as is
      return;
    }

    // If current algo is still okay, do nothing
    if (currentAlgo && compat.includes(currentAlgo)) {
      return;
    }

    // Otherwise, pick the first compatible algo and reset to its defaults
    const nextAlgo = compat[0];
    setTrainModel({ algo: nextAlgo });
  }, [
    effectiveTask,
    trainModel,
    models,
    getCompatibleAlgos,
    setTrainModel,
  ]);

  // Hydrate from artifact into train model + split/eval settings.
  // IMPORTANT:
  // - The training run itself will set artifactSource='trained'. We must NOT copy
  //   engine-resolved defaults into override state (that would bloat payloads and
  //   turn "Default: …" placeholders into explicit overrides).
  // - Only hydrate when the user explicitly loaded an artifact (artifactSource='loaded').
  // - Even when loaded, store overrides only: strip keys that match schema defaults.
  useEffect(() => {
    const uid = artifact?.uid;
    if (!uid || !trainModel) return;
    if (lastHydratedUid.current === uid) return;

    if (artifactSource !== 'loaded') return;
    lastHydratedUid.current = uid;

    if (artifact?.model && typeof artifact.model === 'object') {
      const am = artifact.model;
      const algo = am?.algo;
      const d = (algo && models?.defaults && models.defaults[algo]) ? models.defaults[algo] : null;

      const same = (a, b) => {
        if (a === b) return true;
        if (a == null || b == null) return a == null && b == null;
        if (Array.isArray(a) && Array.isArray(b)) {
          if (a.length !== b.length) return false;
          for (let i = 0; i < a.length; i += 1) if (!same(a[i], b[i])) return false;
          return true;
        }
        if (typeof a === 'object' && typeof b === 'object') {
          const ka = Object.keys(a);
          const kb = Object.keys(b);
          if (ka.length !== kb.length) return false;
          for (const k of ka) {
            if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
            if (!same(a[k], b[k])) return false;
          }
          return true;
        }
        return false;
      };

      // Keep algo always; keep only keys that differ from defaults.
      const stripped = { algo };
      for (const [k, v] of Object.entries(am)) {
        if (k === 'algo') continue;
        if (!d || !Object.prototype.hasOwnProperty.call(d, k)) {
          stripped[k] = v;
          continue;
        }
        if (!same(v, d[k])) stripped[k] = v;
      }
      setTrainModel(stripped);
    }

    const artSplit = artifact?.split || {};
    const inferredMode = artSplit?.mode || ('n_splits' in artSplit ? 'kfold' : undefined);
    const resolvedMode = inferredMode ?? defaultSplitMode;
    setSplitMode(resolvedMode === defaultSplitMode ? undefined : resolvedMode);

    if (resolvedMode === 'kfold' || 'n_splits' in artSplit) {
      if (artSplit?.n_splits != null) {
        const vv = Number(artSplit.n_splits);
        const d = kfoldDefaults?.n_splits;
        if (d != null && vv === Number(d)) setNSplits(undefined);
        else setNSplits(vv);
      }
    } else {
      if (artSplit?.train_frac != null) {
        const vv = Number(artSplit.train_frac);
        const d = holdoutDefaults?.train_frac;
        if (d != null && Math.abs(vv - Number(d)) < 1e-12) setTrainFrac(undefined);
        else setTrainFrac(vv);
      }
    }
    if (artSplit?.stratified != null) {
      const vv = Boolean(artSplit.stratified);
      const d =
        resolvedMode === 'kfold' ? kfoldDefaults?.stratified : holdoutDefaults?.stratified;
      if (d != null && vv === Boolean(d)) setStratified(undefined);
      else setStratified(vv);
    }
    if (artSplit?.shuffle != null) {
      const vv = Boolean(artSplit.shuffle);
      const d = resolvedMode === 'kfold' ? kfoldDefaults?.shuffle : holdoutDefaults?.shuffle;
      if (d != null && vv === Boolean(d)) setShuffle(undefined);
      else setShuffle(vv);
    }

    const ev = artifact?.eval || {};
    if ('seed' in ev) setSeed(ev.seed == null ? '' : String(ev.seed));

    const resultUid = trainResult?.artifact?.uid;
    if (!resultUid || resultUid !== uid) {
      clearTrainResult();
    }
  }, [
    artifact,
    artifactSource,
    trainModel,
    trainResult,
    clearTrainResult,
    defaultSplitMode,
    holdoutDefaults?.train_frac,
    holdoutDefaults?.shuffle,
    holdoutDefaults?.stratified,
    kfoldDefaults?.n_splits,
    kfoldDefaults?.shuffle,
    kfoldDefaults?.stratified,
    models?.defaults,
    setTrainModel,
    setSplitMode,
    setNSplits,
    setTrainFrac,
    setStratified,
    setShuffle,
    setSeed,
  ]);

  // Sanitize overrides: if user has an override that is no longer valid for this task, clear it.
  // IMPORTANT: we never auto-assign defaults here; schema/engine owns defaults.
  const allowedMetrics = useMemo(() => {
    if (!enums) return null;
    const metricByTask = enums.MetricByTask || null;
    if (metricByTask && effectiveTask && Array.isArray(metricByTask[effectiveTask])) {
      return metricByTask[effectiveTask].map(String);
    }
    if (Array.isArray(enums.MetricName)) return enums.MetricName.map(String);
    return null;
  }, [effectiveTask, enums]);

  useEffect(() => {
    if (!metric) return;
    if (!allowedMetrics || allowedMetrics.length === 0) return;
    if (!allowedMetrics.includes(String(metric))) {
      setMetric(undefined);
    }
  }, [allowedMetrics, metric, setMetric]);


  if (defsLoading || !models || !trainModel) {
    return null; // optionally render a skeleton
  }

  return (
    <Stack gap="lg" maw={760}>
      <Title order={3}>Run a Model</Title>

      {error && (
        <Alert color="red" title="Error" variant="light">
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{error}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button
              size="xs"
              onClick={() => {
                setError(null);
                runTraining();
              }}
              loading={isRunning}
              disabled={!dataReady}
            >
              {isRunning ? 'Running…' : 'Run'}
            </Button>
          </Group>

          {isRunning && effectiveShuffleBaselineEnabled && Number(effectiveNShuffles) > 0 && (
            <Stack gap={4}>
              <Group justify="space-between">
                <Text size="xs" c="dimmed">{progressLabel || 'Running…'}</Text>
                <Text size="xs" c="dimmed">{progress}%</Text>
              </Group>
              <Progress value={progress} />
            </Stack>
          )}

          {/* Centered configuration stack inside the card */}
          <Box
            style={{
              
              margin: '0 auto',
              width: '100%',
            }}
          >
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['holdout', 'kfold']}
                mode={splitMode}
                onModeChange={(v) => {
                  // Keep override-only by clearing when selecting the schema default mode.
                  const next = v || undefined;
                  setSplitMode(next && next === defaultSplitMode ? undefined : next);
                  // Also clear per-mode overrides when switching modes.
                  setTrainFrac(undefined);
                  setNSplits(undefined);
                  setStratified(undefined);
                  setShuffle(undefined);
                  setSeed('');
                }}
                trainFrac={trainFrac}
                onTrainFracChange={(v) => {
                  const vv = v === '' || v == null ? undefined : Number(v);
                  const d = holdoutDefaults?.train_frac;
                  if (d != null && vv != null && Math.abs(vv - Number(d)) < 1e-12) {
                    setTrainFrac(undefined);
                  } else {
                    setTrainFrac(vv);
                  }
                }}
                nSplits={nSplits}
                onNSplitsChange={(v) => {
                  const vv = v === '' || v == null ? undefined : Number(v);
                  const d = kfoldDefaults?.n_splits;
                  if (d != null && vv != null && vv === Number(d)) {
                    setNSplits(undefined);
                  } else {
                    setNSplits(vv);
                  }
                }}
                stratified={stratified}
                onStratifiedChange={(v) => {
                  const d = defaultStratified;
                  if (d != null && v === Boolean(d)) setStratified(undefined);
                  else setStratified(Boolean(v));
                }}
                shuffle={shuffle}
                onShuffleChange={(v) => {
                  const d = defaultShuffle;
                  if (d != null && v === Boolean(d)) setShuffle(undefined);
                  else setShuffle(Boolean(v));
                }}
                seed={seed}
                onSeedChange={setSeed}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={trainModel}
                onChange={setTrainModel}
                schema={models?.schema}
                enums={enums}
                models={models}
                showHelp={true}
              />

              <Divider my="xs" />

              <ShuffleLabelsCard
                checked={effectiveShuffleBaselineEnabled}
                onCheckedChange={(checked) => {
                  const d = Boolean(defaultShuffleBaselineEnabled);
                  if (checked === d) setUseShuffleBaseline(undefined);
                  else setUseShuffleBaseline(Boolean(checked));
                }}
                nShuffles={nShuffles}
                nShufflesPlaceholder={
                  defaultNShuffles != null ? `Default: ${defaultNShuffles}` : undefined
                }
                nShufflesDescription={
                  defaultNShuffles == null
                    ? 'Schema did not provide a default number of shuffles.'
                    : undefined
                }
                onNShufflesChange={(v) => {
                  const vv = v === '' || v == null ? undefined : Number(v);
                  const d = defaultNShuffles;
                  if (vv == null) setNShuffles(undefined);
                  else if (d != null && vv === Number(d)) setNShuffles(undefined);
                  else setNShuffles(vv);
                }}
              />
            </Stack>
          </Box>
        </Stack>
      </Card>
    </Stack>
  );
}
