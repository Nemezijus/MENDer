import { useEffect, useRef, useState } from 'react';
import {
  Card, Button, Text,
  Stack, Group, Divider, Alert, Title, Box, Progress
} from '@mantine/core';

import { useDataCtx } from '../state/DataContext.jsx';
import { useRunModelResultsCtx } from '../state/RunModelResultsContext.jsx';
import { useModelArtifact } from '../state/ModelArtifactContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';

import FeatureCard from './FeatureCard.jsx';
import ScalingCard from './ScalingCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import ShuffleLabelsCard from './ShuffleLabelsCard.jsx';
import MetricCard from './MetricCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';

import { runTrainRequest } from '../api/train';
import { fetchProgress } from '../api/progress';
import { getModelSchema } from '../api/schema';

// --- helpers ---------------------------------------------------------------

function uuidv4() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// stringify any error-like into readable text
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
    const loc = Array.isArray(pick.loc) ? pick.loc.join('.') : pick.loc;
    return `${loc ? loc + ': ' : ''}${pick.msg || pick.type || 'Validation error'}`;
  }
  try { return JSON.stringify(pick); } catch { return String(pick); }
}

// Build a method-specific, JSON-clean features payload from FeatureContext.
// Only include keys for the chosen method to avoid Pydantic complaints.
function featuresFromContext(fctx) {
  const method = fctx?.method || 'none';
  if (method === 'pca') {
    return {
      method: 'pca',
      pca_n: fctx.pca_n ?? null,
      pca_var: fctx.pca_var ?? 0.95,
      pca_whiten: !!fctx.pca_whiten,
    };
  }
  if (method === 'lda') {
    return {
      method: 'lda',
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
      method: 'sfs',
      sfs_k: k, // int | 'auto'
      sfs_direction: fctx.sfs_direction ?? 'forward',
      sfs_cv: fctx.sfs_cv ?? 5,
      sfs_n_jobs: fctx.sfs_n_jobs ?? null,
    };
  }
  return { method: 'none' };
}

// --- component -------------------------------------------------------------

export default function RunModelPanel() {
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();
  const fctx = useFeatureCtx();

  const [modelSchema, setModelSchema] = useState(null);
  const [defaultsApplied, setDefaultsApplied] = useState(false);

  // SPLIT / SCALE / METRIC
  const [splitMode, setSplitMode] = useState('holdout');
  const [trainFrac, setTrainFrac] = useState(0.75);
  const [nSplits, setNSplits] = useState(5);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState('');

  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');

  // >>> flat model state mirroring backend ModelModel <<<
  const [model, setModel] = useState(() => ({
    algo: 'logreg',

    // Logistic Regression
    C: 1.0,
    penalty: 'l2',
    solver: 'lbfgs',
    max_iter: 1000,
    class_weight: null,
    l1_ratio: 0.5,

    // SVM (SVC)
    svm_C: 1.0,
    svm_kernel: 'rbf',
    svm_degree: 3,
    svm_gamma: 'scale', // 'scale' | 'auto' | number
    svm_coef0: 0.0,
    svm_shrinking: true,
    svm_probability: false,
    svm_tol: 1e-3,
    svm_cache_size: 200.0,
    svm_class_weight: null,
    svm_max_iter: -1,
    svm_decision_function_shape: 'ovr',
    svm_break_ties: false,

    // DecisionTreeClassifier
    tree_criterion: 'gini',
    tree_splitter: 'best',
    tree_max_depth: null,
    tree_min_samples_split: 2,
    tree_min_samples_leaf: 1,
    tree_min_weight_fraction_leaf: 0.0,
    tree_max_features: null, // null | 'sqrt' | 'log2' | number
    tree_max_leaf_nodes: null,
    tree_min_impurity_decrease: 0.0,
    tree_class_weight: null,
    tree_ccp_alpha: 0.0,

    // RandomForestClassifier
    forest_n_estimators: 100,
    forest_criterion: 'gini',
    forest_max_depth: null,
    forest_min_samples_split: 2,
    forest_min_samples_leaf: 1,
    forest_min_weight_fraction_leaf: 0.0,
    forest_max_features: 'sqrt', // 'sqrt' | 'log2' | number | null
    forest_max_leaf_nodes: null,
    forest_min_impurity_decrease: 0.0,
    forest_bootstrap: true,
    forest_oob_score: false,
    forest_n_jobs: null,
    forest_random_state: null,
    forest_warm_start: false,
    forest_class_weight: null,
    forest_ccp_alpha: 0.0,
    forest_max_samples: null,

    // KNN
    knn_n_neighbors: 5,
    knn_weights: 'uniform',
    knn_algorithm: 'auto',
    knn_leaf_size: 30,
    knn_p: 2,
    knn_metric: 'minkowski',
    knn_n_jobs: null,
  }));

  // Local UI state (progress/errors)
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const [useShuffleBaseline, setUseShuffleBaseline] = useState(false);
  const [nShuffles, setNShuffles] = useState(100);

  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const pollStopRef = useRef(false);
  const lastPercentRef = useRef(0);

  const { setResult } = useRunModelResultsCtx();
  const { artifact, setArtifact, clearArtifact } = useModelArtifact();
  const lastHydratedUid = useRef(null);

  useEffect(() => () => { pollStopRef.current = true; }, []);
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { schema, defaults } = await getModelSchema();
        if (!alive) return;
        setModelSchema(schema);

        // Apply backend defaults to the flat model only once (don’t overwrite user edits)
        if (!defaultsApplied && defaults && typeof defaults === 'object') {
          // The backend defaults already use flat keys (per Pydantic ModelModel)
          // Make sure we keep our current algo if user changed it quickly
          setModel((cur) => ({ ...defaults, algo: cur.algo ?? defaults.algo ?? 'logreg' }));
          setDefaultsApplied(true);
        }
      } catch {
        // ignore; UI will use hardcoded fallbacks for enums
      }
    })();
    return () => { alive = false; };
  }, [defaultsApplied]);

  useEffect(() => {
    const uid = artifact?.uid;
    if (!uid) return;

    if (lastHydratedUid.current === uid) return;
    lastHydratedUid.current = uid;

    // 1) MODEL (already flat)
    if (artifact?.model && typeof artifact.model === 'object') {
      setModel((cur) => ({ ...cur, ...artifact.model }));
    }

    // 2) FEATURES (via context helper)
    if (artifact?.features && typeof artifact.features === 'object') {
      fctx.setFromArtifact(artifact.features);
    }

    // 3) SCALE
    const scaleMethodFromArt = artifact?.scale?.method;
    if (scaleMethodFromArt) setScaleMethod(scaleMethodFromArt);

    // 4) SPLIT
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

    // 5) EVAL
    const ev = artifact?.eval || {};
    if (ev?.metric) setMetric(ev.metric);
    if ('seed' in ev) {
      setSeed(ev.seed == null ? '' : String(ev.seed));
    }

    // 6) Clear current results (avoid stale numbers from a different config)
    setResult(null);
  }, [artifact, fctx, setMetric, setNSplits, setSeed, setSplitMode, setStratified, setTrainFrac, setShuffle, setScaleMethod, setModel, setResult]);

  function startProgressPolling(progressId) {
    pollStopRef.current = false;
    lastPercentRef.current = 0;
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
        lastPercentRef.current = pct;
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
    setErr(null);
    setResult(null);
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
        features: featuresFromContext(fctx),
        model,
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
      setResult(data);
      if (data?.artifact) setArtifact(data.artifact);
    } catch (e) {
      const msg =
        toErrorText(e?.response?.data?.detail) ||
        toErrorText(e?.response?.data) ||
        toErrorText(e);
      setErr(msg);
      setProgress(0);
      setProgressLabel('');
    } finally {
      stopProgressPolling();
      setLoading(false);
    }
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

          <Box w="100%" style={{ maxWidth: 520 }}>
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

              <ScalingCard value={scaleMethod} onChange={setScaleMethod} />

              {/* Restored: Features section */}
              <FeatureCard />

              <Divider my="xs" />

              <ModelSelectionCard model={model} onChange={setModel} schema={modelSchema} />

              <Divider my="xs" />

              <MetricCard value={metric} onChange={setMetric} />
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
