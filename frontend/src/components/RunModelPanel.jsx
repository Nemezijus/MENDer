import { useEffect, useRef, useState } from 'react';
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

// import FeatureCard from './FeatureCard.jsx';
// import ScalingCard from './ScalingCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import ShuffleLabelsCard from './ShuffleLabelsCard.jsx';
// import MetricCard from './MetricCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';

import { runTrainRequest } from '../api/train';
import { fetchProgress } from '../api/progress';

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

  // Centralized schema/defaults/enums
  const { loading: defsLoading, models, enums, getModelDefaults } = useSchemaDefaults();
  
  // SPLIT / SCALE / METRIC
  const [splitMode, setSplitMode] = useState('holdout');
  const [trainFrac, setTrainFrac] = useState(0.75);
  const [nSplits, setNSplits] = useState(5);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState('');
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const setScaleMethod = useSettingsStore((s) => s.setScaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  // Union model state (hydrated from backend defaults)
  const [model, setModel] = useState(null);

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

  // Model artifact store (Zustand)
  const artifact = useModelArtifactStore((s) => s.artifact);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const clearArtifact = useModelArtifactStore((s) => s.clearArtifact);

  const lastHydratedUid = useRef(null);

  useEffect(() => () => { pollStopRef.current = true; }, []);

  // Initialize model once defaults arrive
  useEffect(() => {
    if (!defsLoading && !model) {
      const init = getModelDefaults('logreg') || { algo: 'logreg' };
      setModel(init);
    }
  }, [defsLoading, getModelDefaults, model]);

  // hydrate from artifact
useEffect(() => {
  const uid = artifact?.uid;
  if (!uid || !model) return;
  if (lastHydratedUid.current === uid) return;
  lastHydratedUid.current = uid;

  // Hydrate model hyperparameters from the artifact,
  // but DO NOT override global scaling / metric / feature settings.
  if (artifact?.model && typeof artifact.model === 'object') {
    setModel(artifact.model); // union payload straight in
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
  model,
  trainResult,
  clearTrainResult,
  setSplitMode,
  setNSplits,
  setTrainFrac,
  setStratified,
  setShuffle,
  setSeed,
]);

  // When algo changes, rehydrate from backend defaults and preserve user edits
  useEffect(() => {
    if (!model) return;
    const base = getModelDefaults(model.algo) || { algo: model.algo };
    setModel((cur) => ({ ...base, ...cur }));
  }, [getModelDefaults, model?.algo]); // eslint-disable-line react-hooks/exhaustive-deps

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
    if (!model) return;
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
      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        scale: { method: scaleMethod },
        features: (() => {
          const method = fctx?.method || 'none';
          if (method === 'pca') {
            return {
              method,
              pca_n: fctx.pca_n ?? null,
              pca_var: fctx.pca_var ?? 0.95,
              pca_whiten: !!fctx.pca_whiten,
            };
          }
          if (method === 'lda') {
            return {
              method,
              lda_n: fctx.lda_n ?? null,
              lda_solver: fctx.lda_solver ?? 'svd',
              lda_shrinkage: fctx.lda_shrinkage ?? null,
              lda_tol: fctx.lda_tol ?? 1e-4,
            };
          }
          if (method === 'sfs') {
            let k = fctx.sfs_k;
            if (k === '' || k == null) k = 'auto';
            return {
              method,
              sfs_k: k,
              sfs_direction: fctx.sfs_direction ?? 'forward',
              sfs_cv: fctx.sfs_cv ?? 5,
              sfs_n_jobs: fctx.sfs_n_jobs ?? null,
            };
          }
          return { method: 'none' };
        })(),
        model, // ← union payload as-is
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
          n_shuffles: wantProgress ? Number(nShuffles) : 0,
          ...(wantProgress ? { progress_id: progressId } : {}),
        },
        split:
          splitMode === 'holdout'
            ? { mode: 'holdout', train_frac: trainFrac, stratified, shuffle }
            : { mode: 'kfold', n_splits: nSplits, stratified, shuffle },
      };

      const data = await runTrainRequest(payload);
      setTrainResult(data);
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

  if (defsLoading || !models || !model) {
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
              maxWidth: 520,
              margin: '0 auto',
              width: '100%',
            }}
          >
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['holdout', 'kfold']}
                mode={splitMode}
                onModeChange={setSplitMode}
                trainFrac={trainFrac}
                onTrainFracChange={setTrainFrac}
                nSplits={nSplits}
                onNSplitsChange={setNSplits}
                stratified={stratified}
                onStratifiedChange={setStratified}
                shuffle={shuffle}
                onShuffleChange={setShuffle}
                seed={seed}
                onSeedChange={setSeed}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={model}
                onChange={(next) => {
                  if (next?.algo && next.algo !== model.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setModel({ ...d, ...next });
                  } else {
                    setModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
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
