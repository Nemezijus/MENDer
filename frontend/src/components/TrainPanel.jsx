// src/components/TrainPanel.jsx
import React, { useState } from 'react';
import {
  Card, Button, Select, Checkbox, NumberInput, Text,
  Stack, Group, Divider, Loader, Alert, Title, Box
} from '@mantine/core';
import { useDataCtx } from '../state/DataContext.jsx';
import { runTrainRequest } from '../api/train';
import ConfusionTable from './ConfusionTable.jsx';
import FeatureCard from './FeatureCard.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import ModelCard from './ModelCard.jsx';

export default function TrainPanel() {
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();
  const {
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  // Model state (now includes l1_ratio default)
  const [model, setModel] = useState({
    algo: 'logreg',
    logreg: {
      C: 1.0, penalty: 'l2', solver: 'lbfgs', max_iter: 1000, class_weight: null, l1_ratio: 0.5,
    },
    svm: {
      kernel: 'rbf', C: 1.0, degree: 3, gammaMode: 'scale', gammaValue: 0.1, coef0: 0.0,
      shrinking: true, probability: false, tol: 0.001, cache_size: 200.0, class_weight: null,
      max_iter: -1, decision_function_shape: 'ovr', break_ties: false,
    },
    tree: {
      criterion: 'gini', splitter: 'best', max_depth: null,
      min_samples_split: 2, min_samples_leaf: 1, min_weight_fraction_leaf: 0,
      max_features_mode: 'none', max_features_value: null,
      max_leaf_nodes: null, min_impurity_decrease: 0, class_weight: null, ccp_alpha: 0,
    },
    forest: {
      n_estimators: 100, criterion: 'gini', max_depth: null,
      min_samples_split: 2, min_samples_leaf: 1, min_weight_fraction_leaf: 0,
      max_features_mode: 'sqrt', max_features_value: null,
      max_leaf_nodes: null, min_impurity_decrease: 0,
      bootstrap: true, oob_score: false, n_jobs: null,
      class_weight: null, ccp_alpha: 0, warm_start: false,
    },
  });

  const maxFeaturesToValue = (mode, val) => {
    if (!mode || mode === 'none') return null;
    if (mode === 'sqrt' || mode === 'log2') return mode;
    if (mode === 'int' || mode === 'float') return Number(val);
    return null;
  };
  // Hold-out split + pipeline params
  const [trainFrac, setTrainFrac] = useState(0.75);
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [metric, setMetric] = useState('accuracy');
  const [seed, setSeed] = useState(42);
  const [scaleMethod, setScaleMethod] = useState('standard');

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null);

  function modelToPayload(m) {
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
    // forest
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

  async function handleRun() {
    if (!dataReady) { setErr('Load & inspect data first in the left sidebar.'); return; }
    setErr(null);
    setResult(null);
    setLoading(true);

    try {
      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: { mode: 'holdout', train_frac: trainFrac, stratified, shuffle },
        scale: { method: scaleMethod },
        features: {
          method,
          pca_n, pca_var, pca_whiten,
          lda_n, lda_solver, lda_shrinkage, lda_tol,
          sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
        },
        model: modelToPayload(model),
        eval: { metric, seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null },
      };

      const data = await runTrainRequest(payload);
      setResult(data);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="lg" maw={600}>
      <Title order={3}>Hold-out</Title>

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

          <Box w="100%" style={{ maxWidth: 460 }}>
            <Stack gap="sm">
              <NumberInput label="Train fraction" min={0.5} max={0.95} step={0.05} value={trainFrac} onChange={setTrainFrac} />
              <Checkbox label="Stratified" checked={stratified} onChange={(e) => setStratified(e.currentTarget.checked)} />
              <Checkbox label="Shuffle" checked={shuffle} onChange={(e) => setShuffle(e.currentTarget.checked)} />
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
              <NumberInput label="Seed" value={seed} onChange={setSeed} allowDecimal={false} disabled={!shuffle} />

              <Divider my="xs" />

              <Select
                label="Scale method"
                data={[
                  { value: 'standard', label: 'standard' },
                  { value: 'robust', label: 'robust' },
                  { value: 'minmax', label: 'minmax' },
                  { value: 'maxabs', label: 'maxabs' },
                  { value: 'quantile', label: 'quantile' },
                  { value: 'none', label: 'none' },
                ]}
                value={scaleMethod}
                onChange={setScaleMethod}
              />

              <FeatureCard title="Features" />

              <Divider my="xs" />

              <ModelCard value={model} onChange={setModel} />
            </Stack>
          </Box>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        {loading && <Group align="center" gap="sm"><Loader size="sm" /><Text size="sm">Running…</Text></Group>}
        {!loading && !result && <Text size="sm" c="dimmed">Run to see results.</Text>}
        {!loading && result && (
          <Stack gap="xs">
            <Text size="sm"><Text span fw={500}>Metric:</Text> {result.metric_name}</Text>
            <Text size="sm"><Text span fw={500}>Score:</Text> {result.metric_value?.toFixed?.(4) ?? result.metric_value}</Text>
            <Text size="sm"><Text span fw={500}>Train/Test:</Text> {result.n_train} / {result.n_test}</Text>
            {result.confusion?.matrix && result.confusion?.labels && (
              <>
                <Text fw={500} size="sm">Confusion matrix</Text>
                <ConfusionTable labels={result.confusion.labels} matrix={result.confusion.matrix} />
              </>
            )}
          </Stack>
        )}
      </Card>
    </Stack>
  );
}
