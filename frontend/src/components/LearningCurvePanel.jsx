import { useMemo, useState } from 'react';
import {
  Card, Button, Checkbox, NumberInput, Text,
  Stack, Group, Divider, Loader, Alert, Title, Box, useMantineTheme
} from '@mantine/core';
import Plot from 'react-plotly.js';

import { useDataCtx } from '../state/DataContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import FeatureCard from './FeatureCard.jsx';
import ScalingCard from './ScalingCard.jsx';
import ModelCard from './ModelCard.jsx';
import MetricCard from './MetricCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';
import api from '../api/client';

// A tiny helper identical to RunModelPanel's logic to normalize max_features UI → value
function maxFeaturesToValue(mode, val) {
  if (!mode || mode === 'none') return null;
  if (mode === 'sqrt' || mode === 'log2') return mode;
  if (mode === 'int' || mode === 'float') return Number(val);
  return null;
}

export default function LearningCurvePanel() {

const theme = useMantineTheme();
const isDark = theme.colorScheme === 'dark';
const textColor = isDark ? theme.colors.gray[2] : theme.black;
const gridColor = isDark ? theme.colors.dark[4] : '#e0e0e0';
const axisColor = isDark ? theme.colors.dark[2] : '#222';
  // Shared data (from sidebar)
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();

  // Shared features (from context)
  const {
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  // K-fold params (LC is always k-fold)
  const [nSplits, setNSplits] = useState(5);

  // common split params
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42); // enabled only if shuffle

  // Scaling + Metric
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');

  // Model state (same shape used by ModelCard)
  const [model, setModel] = useState({
    algo: 'logreg',
    logreg: {
      C: 1.0,
      penalty: 'l2',
      solver: 'lbfgs',
      max_iter: 1000,
      class_weight: null,
      l1_ratio: 0.5,
    },
    svm: {
      kernel: 'rbf',
      C: 1.0,
      degree: 3,
      gammaMode: 'scale',     // 'scale' | 'auto' | 'numeric'
      gammaValue: 0.1,        // used only when gammaMode==='numeric'
      coef0: 0.0,
      shrinking: true,
      probability: false,
      tol: 0.001,
      cache_size: 200.0,
      class_weight: null,     // null | 'balanced'
      max_iter: -1,
      decision_function_shape: 'ovr',
      break_ties: false,
    },
    tree: {
      criterion: 'gini', splitter: 'best',
      max_depth: null, min_samples_split: 2, min_samples_leaf: 1,
      min_weight_fraction_leaf: 0.0,
      max_features_mode: 'none', max_features_value: null, // 'none'|'sqrt'|'log2'|'int'|'float'
      max_leaf_nodes: null, min_impurity_decrease: 0.0,
      class_weight: null,
      ccp_alpha: 0.0,
    },
    forest: {
      n_estimators: 100, criterion: 'gini',
      max_depth: null, min_samples_split: 2, min_samples_leaf: 1,
      min_weight_fraction_leaf: 0.0,
      max_features_mode: 'sqrt', max_features_value: null, // default to 'sqrt' like sklearn
      max_leaf_nodes: null, min_impurity_decrease: 0.0,
      bootstrap: true, oob_score: false, n_jobs: null,
      class_weight: null, ccp_alpha: 0.0, warm_start: false,
    },
    knn: {
      n_neighbors: 5, weights: 'uniform', algorithm: 'auto',
      leaf_size: 30, p: 2, metric: 'minkowski', n_jobs: null,
    },
  });

  // Learning-curve specific
  const [trainSizesCSV, setTrainSizesCSV] = useState('');
  const [nSteps, setNSteps] = useState(5);
  const [nJobs, setNJobs] = useState(1);

  // “best size” threshold (as fraction of peak validation)
  const [withinPct, setWithinPct] = useState(0.99); // 99% by default

  // runtime
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null); // { train_sizes, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std }

  // Convert model UI state → backend ModelModel (same logic as RunModelPanel)
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
      const gamma = s.gammaMode === 'numeric' ? Number(s.gammaValue) : (s.gammaMode || 'scale');
      return {
        algo: 'svm',
        svm_C: Number(s.C),
        svm_kernel: s.kernel,
        svm_degree: Number(s.degree),
        svm_gamma: gamma,
        svm_coef0: Number(s.coef0),
        svm_shrinking: !!s.shrinking,
        svm_probability: !!s.probability,
        svm_tol: Number(s.tol),
        svm_cache_size: Number(s.cache_size),
        svm_class_weight: s.class_weight,
        svm_max_iter: Number(s.max_iter),
        svm_decision_function_shape: s.decision_function_shape,
        svm_break_ties: !!s.break_ties,
      };
    }
    if (m.algo === 'tree') {
      const t = m.tree;
      const mf = maxFeaturesToValue(t.max_features_mode, t.max_features_value);
      return {
        algo: 'tree',
        tree_criterion: t.criterion,
        tree_splitter: t.splitter,
        tree_max_depth: t.max_depth == null ? null : Number(t.max_depth),
        tree_min_samples_split: Number(t.min_samples_split),
        tree_min_samples_leaf: Number(t.min_samples_leaf),
        tree_min_weight_fraction_leaf: Number(t.min_weight_fraction_leaf),
        tree_max_features: mf,
        tree_max_leaf_nodes: t.max_leaf_nodes == null ? null : Number(t.max_leaf_nodes),
        tree_min_impurity_decrease: Number(t.min_impurity_decrease),
        tree_class_weight: t.class_weight,
        tree_ccp_alpha: Number(t.ccp_alpha),
      };
    }
    if (m.algo === 'forest') {
      const f = m.forest;
      const mf = maxFeaturesToValue(f.max_features_mode, f.max_features_value);
      return {
        algo: 'forest',
        rf_n_estimators: Number(f.n_estimators),
        rf_criterion: f.criterion,
        rf_max_depth: f.max_depth == null ? null : Number(f.max_depth),
        rf_min_samples_split: Number(f.min_samples_split),
        rf_min_samples_leaf: Number(f.min_samples_leaf),
        rf_min_weight_fraction_leaf: Number(f.min_weight_fraction_leaf),
        rf_max_features: mf,
        rf_max_leaf_nodes: f.max_leaf_nodes == null ? null : Number(f.max_leaf_nodes),
        rf_min_impurity_decrease: Number(f.min_impurity_decrease),
        rf_bootstrap: !!f.bootstrap,
        rf_oob_score: !!f.oob_score,
        rf_n_jobs: f.n_jobs == null ? null : Number(f.n_jobs),
        rf_class_weight: f.class_weight,
        rf_ccp_alpha: Number(f.ccp_alpha),
        rf_warm_start: !!f.warm_start,
      };
    }
    // KNN
    const k = m.knn;
    return {
      algo: 'knn',
      knn_n_neighbors: Number(k.n_neighbors),
      knn_weights: k.weights,
      knn_algorithm: k.algorithm,
      knn_leaf_size: Number(k.leaf_size),
      knn_p: Number(k.p),
      knn_metric: k.metric,
      knn_n_jobs: k.n_jobs == null ? null : Number(k.n_jobs),
    };
  }, [model]);

  // Build LC request payload & call the API
  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the left sidebar.');
      return;
    }
    setErr(null);
    setResult(null);
    setLoading(true);

    try {
      const train_sizes = trainSizesCSV
        ? trainSizesCSV
            .split(',')
            .map(s => s.trim())
            .filter(Boolean)
            .map(x => (x.includes('.') ? parseFloat(x) : parseInt(x, 10)))
        : null;

      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: { mode: 'kfold', n_splits: nSplits, stratified, shuffle },
        scale: { method: scaleMethod },
        features: {
          method,
          pca_n, pca_var, pca_whiten,
          lda_n, lda_solver, lda_shrinkage, lda_tol,
          sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
        },
        model: modelPayload,
        eval: { metric, seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null },
        // LC-specific
        train_sizes,
        n_steps: Number(nSteps),
        n_jobs: Number(nJobs),
      };

      // POST /api/v1/learning-curve
      const { data } = await api.post('/learning-curve', payload);
      setResult(data);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  // Compute SEM bands, best size (peak), and minimal size within threshold
  const analytics = useMemo(() => {
    if (!result) return null;
    const xs = result.train_sizes; // absolute sample counts
    const trainMean = result.train_scores_mean;
    const trainStd  = result.train_scores_std;
    const valMean   = result.val_scores_mean;
    const valStd    = result.val_scores_std;

    const n = Math.max(1, Number(nSplits)); // folds
    const trainSEM = trainStd.map(s => s / Math.sqrt(n));
    const valSEM   = valStd.map(s => s / Math.sqrt(n));

    // Peak validation
    let bestIdx = 0;
    for (let i = 1; i < valMean.length; i++) {
      if (valMean[i] > valMean[bestIdx]) bestIdx = i;
    }
    const best = {
      size: xs[bestIdx],
      val: valMean[bestIdx],
      train: trainMean[bestIdx],
      idx: bestIdx,
    };

    // Minimal size that reaches >= withinPct * peak
    const cutoff = withinPct * best.val;
    let minimalIdx = bestIdx;
    for (let i = 0; i < valMean.length; i++) {
      if (valMean[i] >= cutoff) { minimalIdx = i; break; }
    }
    const minimal = {
      size: xs[minimalIdx],
      val: valMean[minimalIdx],
      train: trainMean[minimalIdx],
      idx: minimalIdx,
      cutoff,
    };

    return { xs, trainMean, valMean, trainSEM, valSEM, best, minimal };
  }, [result, nSplits, withinPct]);

  // Plotly traces: mean lines + SEM shaded areas + vertical line for recommended size
  const plotTraces = useMemo(() => {
    if (!analytics) return [];

    const { xs, trainMean, valMean, trainSEM, valSEM, minimal } = analytics;

    // Build shaded area by adding lower then upper with fill='tonexty'
    const lower = (arr, sem) => arr.map((v, i) => v - sem[i]);
    const upper = (arr, sem) => arr.map((v, i) => v + sem[i]);

    const trainLower = {
      x: xs, y: lower(trainMean, trainSEM),
      name: 'Train (−SEM)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainUpper = {
      x: xs, y: upper(trainMean, trainSEM),
      name: 'Train (SEM area)',
      fill: 'tonexty',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainLine = {
      x: xs, y: trainMean,
      name: 'Train (mean)',
      type: 'scatter',
      mode: 'lines+markers',
      hovertemplate: 'Train size: %{x}<br>Train acc: %{y:.3f}<extra></extra>',
    };

    const valLower = {
      x: xs, y: lower(valMean, valSEM),
      name: 'Validation (−SEM)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const valUpper = {
      x: xs, y: upper(valMean, valSEM),
      name: 'Validation (SEM area)',
      fill: 'tonexty',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const valLine = {
      x: xs, y: valMean,
      name: 'Validation (mean)',
      type: 'scatter',
      mode: 'lines+markers',
      hovertemplate: 'Train size: %{x}<br>Val acc: %{y:.3f}<extra></extra>',
    };

    const vLine = {
      x: [analytics.minimal.size, analytics.minimal.size],
      y: [0, 1],
      name: `Recommended size`,
      mode: 'lines',
      line: { dash: 'dash' },
      hoverinfo: 'skip',
      showlegend: true,
    };

    return [trainLower, trainUpper, trainLine, valLower, valUpper, valLine];
  }, [analytics]);

  return (
    <Stack gap="md">
      <Title order={3}>Learning Curve</Title>

      {err && (
        <Alert color="red" variant="light">
          <Text fw={500}>Error</Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button size="xs" onClick={handleCompute} loading={loading} disabled={!dataReady}>
              {loading ? 'Computing…' : 'Compute'}
            </Button>
          </Group>

          <Box w="100%" style={{ maxWidth: 560 }}>
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
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

              {/* Scaling */}
              <ScalingCard 
                value={scaleMethod} 
                onChange={setScaleMethod} 
              />

              {/* Features */}
              <FeatureCard 
                title="Features" 
              />

              <Divider my="xs" />

              {/* Metric */}
              <MetricCard
                value={metric}
                onChange={setMetric}
              />

              <Divider my="xs" />

              {/* Model */}
              <ModelCard value={model} onChange={setModel} />

              <Divider my="xs" />

              {/* Learning Curve specific */}
              <NumberInput label="Steps (used if Train sizes empty)" min={2} max={50} step={1} value={nSteps} onChange={setNSteps} />
              <NumberInput label="n_jobs" min={1} step={1} value={nJobs} onChange={setNJobs} />
              <Text size="sm" c="dimmed">
                Optional Train sizes (CSV): fractions in (0,1] or absolute integers. Example:
                <Text span fw={500}> 0.1,0.3,0.5,0.7,1.0 </Text> or <Text span fw={500}> 50,100,200 </Text>
              </Text>
              <textarea
                style={{ width: '100%', minHeight: 70, fontFamily: 'inherit', fontSize: '0.9rem' }}
                placeholder="e.g. 0.1,0.3,0.5,0.7,1.0"
                value={trainSizesCSV}
                onChange={(e) => setTrainSizesCSV(e.currentTarget.value)}
              />

              <Divider my="xs" />

              <NumberInput
                label="Recommend the smallest train size achieving at least this fraction of the peak validation score"
                description="e.g., 0.99 = within 1% of peak"
                min={0.5} max={1.0} step={0.01}
                value={withinPct}
                onChange={setWithinPct}
                precision={2}
              />
            </Stack>
          </Box>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        {loading && (
          <Group align="center" gap="sm">
            <Loader size="sm" />
            <Text size="sm">Computing…</Text>
          </Group>
        )}
        {!loading && !result && (
          <Text size="sm" c="dimmed">Run to see results.</Text>
        )}
        {!loading && result && (
          <>
            <Plot
                data={plotTraces}
                layout={{
                    title: { text: 'Learning Curve — Accuracy vs. Training Set Size', font: { color: textColor } },
                    font: { color: textColor },
                    xaxis: {
                    title: { text: 'Training size (samples)' },
                    tickfont: { color: textColor },
                    titlefont: { color: textColor },
                    gridcolor: gridColor,
                    linecolor: axisColor,
                    mirror: true,
                    automargin: true,
                    },
                    yaxis: {
                    title: { text: 'Accuracy (mean ± SEM)' },
                    tickfont: { color: textColor },
                    titlefont: { color: textColor },
                    gridcolor: gridColor,
                    linecolor: axisColor,
                    mirror: true,
                    automargin: true,
                    range: [0, 1.1],
                    },
                    legend: { orientation: 'h', x: 0, y: 1.12, font: { color: textColor } },
                    margin: { l: 70, r: 20, t: 60, b: 80 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    autosize: true,
                }}
                config={{ displaylogo: false, responsive: true }}
                style={{ width: '100%', height: '460px' }}
                useResizeHandler
                />


            {analytics && (
              <Box mt="sm">
                <Text size="sm">
                  <Text span fw={600}>Peak validation</Text>: size <Text span fw={600}>{analytics.best.size}</Text>,
                  val = <Text span fw={600}>{analytics.best.val.toFixed(3)}</Text>, train = {analytics.best.train.toFixed(3)}
                </Text>
                <Text size="sm">
                  <Text span fw={600}>Recommended (≥ {(withinPct * 100).toFixed(0)}% of peak)</Text>: size <Text span fw={600}>{analytics.minimal.size}</Text>,
                  val = <Text span fw={600}>{analytics.minimal.val.toFixed(3)}</Text>, train = {analytics.minimal.train.toFixed(3)}
                </Text>
              </Box>
            )}
          </>
        )}
      </Card>
    </Stack>
  );
}
