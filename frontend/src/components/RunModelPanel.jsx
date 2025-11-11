import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Card, Button, Select, Checkbox, NumberInput, Text,
  Stack, Group, Divider, Loader, Alert, Title, Box, Table, Progress, Tooltip
} from '@mantine/core';
import Plot from 'react-plotly.js';

import { useDataCtx } from '../state/DataContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import FeatureCard from './FeatureCard.jsx';
import ModelCard from './ModelCard.jsx';
import { runModelRequest } from '../api/runModel';
import { fetchProgress } from '../api/progress';

function uuidv4() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export default function RunModelPanel() {
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();
  const {
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  const [splitMode, setSplitMode] = useState('holdout');
  const [trainFrac, setTrainFrac] = useState(0.75);
  const [nSplits, setNSplits] = useState(5);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42);
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');

  const [model, setModel] = useState({
    algo: 'logreg',
    logreg: { C: 1.0, penalty: 'l2', solver: 'lbfgs', max_iter: 1000, class_weight: null, l1_ratio: 0.5 },
    svm: {
      kernel: 'rbf', C: 1.0, degree: 3, gammaMode: 'scale', gammaValue: 0.1, coef0: 0.0,
      shrinking: true, probability: false, tol: 0.001, cache_size: 200.0, class_weight: null,
      max_iter: -1, decision_function_shape: 'ovr', break_ties: false,
    },
    tree: {
      criterion: 'gini', splitter: 'best', max_depth: null, min_samples_split: 2, min_samples_leaf: 1,
      min_weight_fraction_leaf: 0.0, max_featuresMode: 'auto', max_featuresValue: null, random_state: null,
      max_leaf_nodes: null, min_impurity_decrease: 0.0, class_weight: null, ccp_alpha: 0.0,
    },
    forest: {
      n_estimators: 100, criterion: 'gini', max_depth: null, min_samples_split: 2, min_samples_leaf: 1,
      min_weight_fraction_leaf: 0.0, max_featuresMode: 'sqrt', max_featuresValue: null, max_leaf_nodes: null,
      min_impurity_decrease: 0.0, bootstrap: true, oob_score: false, n_jobs: null, random_state: null,
      warm_start: false, class_weight: null, ccp_alpha: 0.0, max_samples: null,
    },
    knn: { n_neighbors: 5, weights: 'uniform', algorithm: 'auto', leaf_size: 30, p: 2, metric: 'minkowski', n_jobs: null },
  });

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null);

  const [useShuffleBaseline, setUseShuffleBaseline] = useState(false);
  const [nShuffles, setNShuffles] = useState(100);

  // Progress polling (self-scheduling; stops immediately on done)
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const pollStopRef = useRef(false);
  const lastPercentRef = useRef(0);

  useEffect(() => () => { pollStopRef.current = true; }, []);

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

        // Adaptive backoff to reduce log noise
        let delay = 250;
        if (pct >= 90 && pct < 98) delay = 500;
        else if (pct >= 98 && !rec.done) delay = 1000;

        lastPercentRef.current = pct;

        if (!rec.done) {
          setTimeout(tick, delay);
        } else {
          pollStopRef.current = true;
        }
      } catch {
        // Progress record might not exist yet—retry soon
        setTimeout(tick, 300);
      }
    }
    tick();
  }

  function stopProgressPolling() {
    pollStopRef.current = true;
  }

  const maxFeaturesToValue = (mode, val) => {
    if (mode === 'auto' || mode === 'sqrt' || mode === 'log2') return mode;
    if (mode === 'all') return null;
    if (mode === 'custom') {
      if (val === '' || val == null) return null;
      const n = Number(val);
      return Number.isNaN(n) ? null : n;
    }
    return null;
  };

  const modelPayload = useMemo(() => {
    const m = model;
    if (m.algo === 'logreg') {
      const lr = m.logreg;
      return {
        algo: 'logreg',
        C: Number(lr.C),
        penalty: lr.penalty,
        solver: lr.solver,
        max_iter: Number(lr.max_iter),
        class_weight: lr.class_weight,
        ...(lr.penalty === 'elasticnet' ? { l1_ratio: Number(lr.l1_ratio ?? 0.5) } : {}),
      };
    }
    if (m.algo === 'svm') {
      const s = m.svm;
      return {
        algo: 'svm',
        kernel: s.kernel,
        C: Number(s.C),
        degree: Number(s.degree),
        gamma: s.gammaMode === 'value' ? Number(s.gammaValue) : s.gammaMode,
        coef0: Number(s.coef0),
        shrinking: Boolean(s.shrinking),
        probability: Boolean(s.probability),
        tol: Number(s.tol),
        cache_size: Number(s.cache_size),
        class_weight: s.class_weight,
        max_iter: Number(s.max_iter),
        decision_function_shape: s.decision_function_shape,
        break_ties: Boolean(s.break_ties),
      };
    }
    if (m.algo === 'tree') {
      const t = m.tree;
      return {
        algo: 'tree',
        criterion: t.criterion,
        splitter: t.splitter,
        max_depth: t.max_depth == null || t.max_depth === '' ? null : Number(t.max_depth),
        min_samples_split: Number(t.min_samples_split),
        min_samples_leaf: Number(t.min_samples_leaf),
        min_weight_fraction_leaf: Number(t.min_weight_fraction_leaf),
        max_features: maxFeaturesToValue(t.max_featuresMode, t.max_featuresValue),
        random_state: t.random_state == null || t.random_state === '' ? null : Number(t.random_state),
        max_leaf_nodes: t.max_leaf_nodes == null || t.max_leaf_nodes === '' ? null : Number(t.max_leaf_nodes),
        min_impurity_decrease: Number(t.min_impurity_decrease),
        class_weight: t.class_weight,
        ccp_alpha: Number(t.ccp_alpha),
      };
    }
    if (m.algo === 'forest') {
      const f = m.forest;
      return {
        algo: 'forest',
        n_estimators: Number(f.n_estimators),
        criterion: f.criterion,
        max_depth: f.max_depth == null || f.max_depth === '' ? null : Number(f.max_depth),
        min_samples_split: Number(f.min_samples_split),
        min_samples_leaf: Number(f.min_samples_leaf),
        min_weight_fraction_leaf: Number(f.min_weight_fraction_leaf),
        max_features: maxFeaturesToValue(f.max_featuresMode, f.max_featuresValue),
        max_leaf_nodes: f.max_leaf_nodes == null || f.max_leaf_nodes === '' ? null : Number(f.max_leaf_nodes),
        min_impurity_decrease: Number(f.min_impurity_decrease),
        bootstrap: Boolean(f.bootstrap),
        oob_score: Boolean(f.oob_score),
        n_jobs: f.n_jobs == null || f.n_jobs === '' ? null : Number(f.n_jobs),
        random_state: f.random_state == null || f.random_state === '' ? null : Number(f.random_state),
        warm_start: Boolean(f.warm_start),
        class_weight: f.class_weight,
        ccp_alpha: Number(f.ccp_alpha),
        max_samples: f.max_samples == null || f.max_samples === '' ? null : Number(f.max_samples),
      };
    }
    if (m.algo === 'knn') {
      const k = m.knn;
      return {
        algo: 'knn',
        n_neighbors: Number(k.n_neighbors),
        weights: k.weights,
        algorithm: k.algorithm,
        leaf_size: Number(k.leaf_size),
        p: Number(k.p),
        metric: k.metric,
        n_jobs: k.n_jobs == null || k.n_jobs === '' ? null : Number(k.n_jobs),
      };
    }
    return { algo: 'logreg', C: 1.0, penalty: 'l2', solver: 'lbfgs', max_iter: 1000, class_weight: null };
  }, [model]);

  async function handleRun() {
    if (!dataReady) {
      setErr('Load & inspect data first in the left sidebar.');
      return;
    }
    setErr(null);
    setResult(null);
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
      const base = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        scale: { method: scaleMethod },
        features: {
          method,
          pca_n, pca_var, pca_whiten,
          lda_n, lda_solver, lda_shrinkage, lda_tol,
          sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
        },
        model: modelPayload,
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
          n_shuffles: wantProgress ? Number(nShuffles) : 0,
          ...(wantProgress ? { progress_id: progressId } : {}),
        },
      };

      const payload =
        splitMode === 'holdout'
          ? { ...base, split: { mode: 'holdout', train_frac: trainFrac, stratified, shuffle } }
          : { ...base, split: { mode: 'kfold', n_splits: nSplits, stratified, shuffle } };

      const data = await runModelRequest(payload);
      setResult(data);
      // Polling stops automatically when the baseline finalizes; no need to force 100%.
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setProgress(0);
      setProgressLabel('');
    } finally {
      stopProgressPolling();
      setLoading(false);
    }
  }

  function ResultView() {
    if (!result) return null;
    const isCV = Array.isArray(result.fold_scores);

    if (!isCV) {
      return (
        <Stack gap="xs">
          <Text size="sm"><Text span fw={500}>Metric:</Text> {result.metric_name}</Text>
          <Text size="sm"><Text span fw={500}>Score:</Text> {typeof result.metric_value === 'number' ? result.metric_value.toFixed(4) : result.metric_value}</Text>
          <Text size="sm"><Text span fw={500}>Train/Test:</Text> {result.n_train} / {result.n_test}</Text>
          {result.confusion?.matrix && result.confusion?.labels && (
            <>
              <Text fw={500} size="sm">Confusion matrix</Text>
              <Table striped withTableBorder withColumnBorders maw={460}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th></Table.Th>
                    {result.confusion.labels.map((lbl, j) => (
                      <Table.Th key={`pred-${j}`}>
                        <Text size="sm" fw={500}>Pred {String(lbl)}</Text>
                      </Table.Th>
                    ))}
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {result.confusion.matrix.map((row, i) => (
                    <Table.Tr key={i}>
                      <Table.Td>
                        <Text size="sm" fw={500}>True {String(result.confusion.labels[i])}</Text>
                      </Table.Td>
                      {row.map((v, j) => <Table.Td key={j}>{v}</Table.Td>)}
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>

              {Array.isArray(result.shuffled_scores) && result.shuffled_scores.length > 0 && (
                <>
                  <Text fw={500} size="sm" mt="sm">Shuffle-label baseline</Text>
                  <Plot
                    data={[ { type: 'histogram', x: result.shuffled_scores, opacity: 0.75, name: 'Shuffled' } ]}
                    layout={{
                      bargap: 0.05,
                      xaxis: { title: result.metric_name },
                      yaxis: { title: 'Count' },
                      shapes: [
                        { type: 'line', x0: result.metric_value, x1: result.metric_value, y0: 0, y1: 1, yref: 'paper', line: { width: 2 } }
                      ],
                      annotations: result.p_value != null ? [{
                        x: result.metric_value, y: 1, yref: 'paper',
                        text: `real = ${typeof result.metric_value === 'number' ? result.metric_value.toFixed(4) : result.metric_value}${result.p_value != null ? ` · p≈${Number(result.p_value).toFixed(4)}` : ''}`,
                        showarrow: false, xanchor: 'left', align: 'left'
                      }] : [],
                      margin: { t: 20, r: 10, b: 40, l: 50 }, height: 260
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: '100%', maxWidth: 520 }}
                  />
                </>
              )}
            </>
          )}
          {Array.isArray(result.notes) && result.notes.length > 0 && (
            <>
              <Text fw={500} size="sm" mt="sm">Notes</Text>
              <ul style={{ marginTop: 4 }}>
                {result.notes.map((n, i) => <li key={i}><Text size="sm">{n}</Text></li>)}
              </ul>
            </>
          )}
        </Stack>
      );
    }

    // CV view
    const folds = result.fold_scores || [];
    const idxs = folds.map((_, i) => i + 1);

    return (
      <Stack gap="xs">
        <Text size="sm"><Text span fw={500}>Metric:</Text> {result.metric_name}</Text>
        <Text size="sm">
          <Text span fw={500}>Score:</Text> mean {typeof result.mean_score === 'number' ? result.mean_score.toFixed(4) : result.mean_score} ± {typeof result.std_score === 'number' ? result.std_score.toFixed(4) : result.std_score}
        </Text>

        <Table withTableBorder withColumnBorders maw={460} striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>#</Table.Th>
              <Table.Th>score</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {folds.map((s, i) => (
              <Table.Tr key={i}>
                <Table.Td>{i + 1}</Table.Td>
                <Table.Td>{typeof s === 'number' ? s : JSON.stringify(s)}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>

        <Plot
          data={[
            { type: 'bar', x: idxs, y: folds, name: 'Fold score' },
            { type: 'scatter', mode: 'lines', x: [0, idxs.length + 1], y: [result.mean_score, result.mean_score], name: 'Mean' },
          ]}
          layout={{ title: 'Fold scores', margin: { l: 40, r: 10, b: 40, t: 20 }, xaxis: { title: 'Fold' }, yaxis: { title: result.metric_name }, autosize: true }}
          style={{ width: '100%', height: 300 }}
          config={{ displayModeBar: false, responsive: true }}
        />

        {Array.isArray(result.shuffled_scores) && result.shuffled_scores.length > 0 && (
          <>
            <Divider my="xs" />
            <Text fw={500} size="sm">Shuffle-label baseline (CV mean)</Text>
            <Plot
              data={[ { type: 'histogram', x: result.shuffled_scores, opacity: 0.75, name: 'Shuffled' } ]}
              layout={{
                bargap: 0.05,
                xaxis: { title: result.metric_name },
                yaxis: { title: 'Count' },
                shapes: [
                  { type: 'line', x0: result.mean_score, x1: result.mean_score, y0: 0, y1: 1, yref: 'paper', line: { width: 2 } }
                ],
                annotations: result.p_value != null ? [{
                  x: result.mean_score, y: 1, yref: 'paper',
                  text: `real mean = ${typeof result.mean_score === 'number' ? result.mean_score.toFixed(4) : result.mean_score}${result.p_value != null ? ` · p≈${Number(result.p_value).toFixed(4)}` : ''}`,
                  showarrow: false, xanchor: 'left', align: 'left'
                }] : [],
                margin: { t: 20, r: 10, b: 40, l: 50 }, height: 260
              }}
              config={{ displayModeBar: false }}
              style={{ width: '100%', maxWidth: 520 }}
            />
          </>
        )}
      </Stack>
    );
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

          {/* Only show progress when shuffling (true backend progress available) */}
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
              <Select
                label="Split strategy"
                data={[
                  { value: 'holdout', label: 'Hold-out' },
                  { value: 'kfold', label: 'K-fold cross-validation' },
                ]}
                value={splitMode}
                onChange={(v) => setSplitMode(v || 'holdout')}
              />

              {splitMode === 'holdout'
                ? <NumberInput label="Train fraction" min={0.5} max={0.95} step={0.05} value={trainFrac} onChange={setTrainFrac} />
                : <NumberInput label="n_splits" min={2} max={20} step={1} value={nSplits} onChange={setNSplits} />
              }

              <Group grow>
                <Checkbox label="Stratified" checked={stratified} onChange={(e) => setStratified(e.currentTarget.checked)} />
                <Checkbox label="Shuffle split" checked={shuffle} onChange={(e) => setShuffle(e.currentTarget.checked)} />
              </Group>
              <NumberInput label="Seed (used if shuffle split)" value={seed} onChange={setSeed} />

              <Divider my="xs" />

              <Select
                label="Scaling"
                data={[
                  { value: 'none', label: 'None' },
                  { value: 'standard', label: 'StandardScaler' },
                  { value: 'robust', label: 'RobustScaler' },
                  { value: 'minmax', label: 'MinMaxScaler' },
                  { value: 'maxabs', label: 'MaxAbsScaler' },
                  { value: 'quantile', label: 'QuantileTransformer' },
                ]}
                value={scaleMethod}
                onChange={setScaleMethod}
              />

              <FeatureCard title="Features" />
              <Divider my="xs" />

              <Select
                label="Metric"
                data={[
                  { value: 'accuracy', label: 'accuracy' },
                  { value: 'balanced_accuracy', label: 'balanced_accuracy' },
                  { value: 'f1_macro', label: 'f1_macro' },
                ]}
                value={metric}
                onChange={setMetric}
              />
              <Tooltip label="Perform label shuffling to create a baseline distribution of scores for comparison. This does not affect the main model training.">
                <Checkbox
                  label="Shuffle labels for control"
                  checked={useShuffleBaseline}
                  onChange={(e) => setUseShuffleBaseline(e.currentTarget.checked)}
                />
              </Tooltip>
              {useShuffleBaseline && (
                <NumberInput
                  label="Number of shuffles"
                  min={10}
                  max={5000}
                  step={10}
                  value={nShuffles}
                  onChange={setNShuffles}
                />
              )}

              <Divider my="xs" />
              <ModelCard value={model} onChange={setModel} />
            </Stack>
          </Box>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        {loading && (
          <Group align="center" gap="sm">
            <Loader size="sm" />
            <Text size="sm">Crunching numbers…</Text>
          </Group>
        )}
        {!loading && !result && <Text size="sm" c="dimmed">Run to see results.</Text>}
        {!loading && result && <ResultView />}
      </Card>
    </Stack>
  );
}
