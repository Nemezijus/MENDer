import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Card, Button, Text,
  Stack, Group, Divider, Alert, Title, Box, Progress
} from '@mantine/core';

import { useDataStore } from '../state/useDataStore.js';
import { useResultsStore } from '../state/useResultsStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import ShuffleLabelsCard from './ShuffleLabelsCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';

import { runTrainRequest } from '../api/train';
import { fetchProgress } from '../api/progress';
import { compactPayload } from '../utils/compactPayload.js';

/** ---------- helpers ---------- **/

function uuidv4() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function toErrorText(e) {
  if (typeof e === 'string') return e;
  const data = e?.response?.data;
  const detail = data?.detail ?? e?.detail;
  const pick = detail ?? data ?? e?.message ?? e;
  if (typeof pick === 'string') return pick;
  if (Array.isArray(pick)) {
    return pick.map((it) => {
      if (typeof it === 'string') return it;
      if (it && typeof it === 'object') {
        const loc = Array.isArray(it.loc) ? it.loc.join('.') : it.loc;
        return it.msg ? `${loc ? loc + ': ' : ''}${it.msg}` : JSON.stringify(it);
      }
      return String(it);
    }).join('\n');
  }
  if (pick && typeof pick === 'object' && ('msg' in pick || 'type' in pick || 'loc' in pick)) {
    const loc = Array.isArray(pick.loc) ? it.loc.join('.') : pick.loc;
    return `${loc ? loc + ': ' : ''}${pick.msg || pick.type || 'Validation error'}`;
  }
  try { return JSON.stringify(pick); } catch { return String(pick); }
}

/** ---------- component ---------- **/

export default function RunModelPanel() {
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;
  const fctx = useFeatureStore();

  const { loading: defsLoading, models, enums, split, getModelDefaults, getCompatibleAlgos } = useSchemaDefaults();

  // SPLIT / SCALE / METRIC (override-only; schema owns defaults)
  const [splitMode, setSplitMode] = useState(undefined);
  const [trainFrac, setTrainFrac] = useState(undefined);
  const [nSplits, setNSplits] = useState(undefined);
  const [stratified, setStratified] = useState(undefined);
  const [shuffle, setShuffle] = useState(undefined);
  const [seed, setSeed] = useState('');
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  // Per-panel model config (train slice)
  const trainModel = useModelConfigStore((s) => s.train);
  const setTrainModel = useModelConfigStore((s) => s.setTrainModel);

  // Results / progress (local UI)
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [useShuffleBaseline, setUseShuffleBaseline] = useState(false);
  const [nShuffles, setNShuffles] = useState(100);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const pollStopRef = useRef(false);

  // Zustand results store: train result slice
  const trainResult = useResultsStore((s) => s.trainResult);
  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const clearTrainResult = useResultsStore((s) => s.clearTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);

  // Model artifact store (Zustand)
  const artifact = useModelArtifactStore((s) => s.artifact);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const clearArtifact = useModelArtifactStore((s) => s.clearArtifact);

  const lastHydratedUid = useRef(null);

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );


  useEffect(() => () => { pollStopRef.current = true; }, []);

  // Initialize train model once defaults arrive
  useEffect(() => {
    if (!defsLoading && !trainModel) {
      const init = getModelDefaults('logreg') || { algo: 'logreg' };
      setTrainModel(init);
    }
  }, [defsLoading, getModelDefaults, trainModel, setTrainModel]);

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
    const defaults = getModelDefaults(nextAlgo) || { algo: nextAlgo };
    setTrainModel(defaults);
  }, [
    effectiveTask,
    trainModel,
    models,
    getCompatibleAlgos,
    getModelDefaults,
    setTrainModel,
  ]);

  // Hydrate from artifact into train model + split/eval settings
  useEffect(() => {
    const uid = artifact?.uid;
    if (!uid || !trainModel) return;
    if (lastHydratedUid.current === uid) return;
    lastHydratedUid.current = uid;

    if (artifact?.model && typeof artifact.model === 'object') {
      // union payload straight in
      setTrainModel(artifact.model);
    }

    const split = artifact?.split || {};
    const mode = split?.mode || ('n_splits' in split ? 'kfold' : 'holdout');
    setSplitMode(mode === 'kfold' ? 'kfold' : 'holdout');

    if (mode === 'kfold' || 'n_splits' in split) {
      if (split?.n_splits != null) setNSplits(Number(split.n_splits));
    } else {
      if (split?.train_frac != null) setTrainFrac(Number(split.train_frac));
    }
    if (split?.stratified != null) setStratified(!!split.stratified);
    if (split?.shuffle != null) setShuffle(!!split.shuffle);

    const ev = artifact?.eval || {};
    if ('seed' in ev) {
      setSeed(ev.seed == null ? '' : String(ev.seed));
    }

    const resultUid = trainResult?.artifact?.uid;
    if (!resultUid || resultUid !== uid) {
      clearTrainResult();
    }
  }, [
    artifact,
    trainModel,
    trainResult,
    clearTrainResult,
  ]);

  // When algo changes, rehydrate from backend defaults and preserve user edits
  useEffect(() => {
    if (!trainModel) return;
    const base = getModelDefaults(trainModel.algo) || { algo: trainModel.algo };
    const merged = { ...base, ...trainModel };
    setTrainModel(merged);
  }, [getModelDefaults, trainModel?.algo]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Split defaults (for clearing overrides when user selects defaults)
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;
  const effectiveMode = splitMode ?? 'holdout';
  const defaultStratified =
    effectiveMode === 'kfold' ? kfoldDefaults?.stratified : holdoutDefaults?.stratified;
  const defaultShuffle =
    effectiveMode === 'kfold' ? kfoldDefaults?.shuffle : holdoutDefaults?.shuffle;
  const effectiveShuffle = shuffle ?? defaultShuffle;

  function startProgressPolling(progressId) {
    pollStopRef.current = false;
    async function tick() {
      if (pollStopRef.current) return;
      try {
        const rec = await fetchProgress(progressId);
        const pct = Math.round(rec.percent || 0);
        setProgress(pct);
        setProgressLabel(rec.label || '');
        let delay = 250;
        if (pct >= 90 && pct < 98) delay = 500;
        else if (pct >= 98 && !rec.done) delay = 1000;
        if (!rec.done) setTimeout(tick, delay);
        else pollStopRef.current = true;
      } catch {
        setTimeout(tick, 300);
      }
    }
    tick();
  }
  function stopProgressPolling() { pollStopRef.current = true; }

  async function handleRun() {
    if (!dataReady) {
      setErr('Load & inspect data first in the left sidebar.');
      return;
    }
    if (!trainModel) return;

    setErr(null);
    clearTrainResult();
    clearArtifact();
    setLoading(true);

    const wantProgress = useShuffleBaseline && Number(nShuffles) > 0;
    const progressId = wantProgress ? uuidv4() : null;
    if (wantProgress) {
      setProgress(0);
      setProgressLabel(`Shuffling 0/${nShuffles}…`);
      startProgressPolling(progressId);
    } else {
      stopProgressPolling();
      setProgress(0);
      setProgressLabel('');
    }

    try {
      // IMPORTANT: Overrides-only payload.
      // - Engine owns defaults via /schema/defaults.
      // - Backend owns IO defaults (e.g. x_key/y_key, upload constraints).
      // - We omit unset fields, so defaults apply naturally.

      // DATA
      const dataPayload = compactPayload({
        x_path: npzPath ? undefined : xPath,
        y_path: npzPath ? undefined : yPath,
        npz_path: npzPath,
        x_key: xKey,
        y_key: yKey,
      });

      // SCALE (store is override-only)
      const scalePayload = compactPayload({
        method: scaleMethod,
      });

      // FEATURES (override-only store)
      let featuresPayload = {};
      const m = fctx?.method;
      if (m === 'pca') {
        featuresPayload = compactPayload({
          method: m,
          pca_n: fctx.pca_n,
          pca_var: fctx.pca_var,
          pca_whiten: fctx.pca_whiten,
        });
      } else if (m === 'lda') {
        featuresPayload = compactPayload({
          method: m,
          lda_n: fctx.lda_n,
          lda_solver: fctx.lda_solver,
          lda_shrinkage: fctx.lda_shrinkage,
          lda_tol: fctx.lda_tol,
        });
      } else if (m === 'sfs') {
        featuresPayload = compactPayload({
          method: m,
          sfs_k: fctx.sfs_k,
          sfs_direction: fctx.sfs_direction,
          sfs_cv: fctx.sfs_cv,
          sfs_n_jobs: fctx.sfs_n_jobs,
        });
      } else {
        // method is unset => omit entirely and let engine default to 'none'
        featuresPayload = compactPayload({
          method: m,
        });
      }

      // EVAL (override-only; decoder is left to engine default)
      const seedInt = seed === '' ? undefined : Number.parseInt(String(seed), 10);
      const evalPayload = compactPayload({
        metric,
        seed: Boolean(effectiveShuffle) && Number.isFinite(seedInt) ? seedInt : undefined,
        n_shuffles: wantProgress ? Number(nShuffles) : undefined,
        progress_id: wantProgress ? progressId : undefined,
      });

      // SPLIT (override-only; always include mode discriminator)
      const splitPayload = compactPayload(
        effectiveMode === 'holdout'
          ? {
              mode: 'holdout',
              train_frac: trainFrac,
              stratified,
              shuffle,
            }
          : {
              mode: 'kfold',
              n_splits: nSplits,
              stratified,
              shuffle,
            },
      );

      const payload = {
        data: dataPayload,
        // These top-level sections are required by the request model; empty objects are OK.
        scale: scalePayload,
        features: featuresPayload,
        model: trainModel,
        eval: evalPayload,
        split: splitPayload,
      };

      const data = await runTrainRequest(payload);
      
      setTrainResult(data);
      setActiveResultKind('train');
      if (data?.artifact) setArtifact(data.artifact);
    } catch (e) {
      setErr(toErrorText(e));
      setProgress(0);
      setProgressLabel('');
    } finally {
      stopProgressPolling();
      setLoading(false);
    }
  }

  if (defsLoading || !models || !trainModel) {
    return null; // optionally render a skeleton
  }

  return (
    <Stack gap="lg" maw={760}>
      <Title order={3}>Run a Model</Title>

      {err && (
        <Alert color="red" title="Error" variant="light">
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button size="xs" onClick={handleRun} loading={loading} disabled={!dataReady}>
              {loading ? 'Running…' : 'Run'}
            </Button>
          </Group>

          {loading && useShuffleBaseline && Number(nShuffles) > 0 && (
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
                  // Default mode is holdout; keep override-only by clearing when selecting default.
                  setSplitMode(v === 'holdout' ? undefined : v);
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
                onChange={(next) => {
                  if (next?.algo && trainModel && next.algo !== trainModel.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setTrainModel({ ...d, ...next });
                  } else {
                    setTrainModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
                showHelp={true}
              />

              <Divider my="xs" />

              <ShuffleLabelsCard
                checked={useShuffleBaseline}
                onCheckedChange={setUseShuffleBaseline}
                nShuffles={nShuffles}
                onNShufflesChange={setNShuffles}
              />
            </Stack>
          </Box>
        </Stack>
      </Card>
    </Stack>
  );
}
