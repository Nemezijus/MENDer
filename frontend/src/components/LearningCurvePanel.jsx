import { useEffect, useMemo, useState } from 'react';
import {
  Card, Button, Text,
  Stack, Group, Divider, Alert, Title, Box, NumberInput
} from '@mantine/core';

import { useDataCtx } from '../state/DataContext.jsx';
import { useFeatureCtx } from '../state/FeatureContext.jsx';
import FeatureCard from './FeatureCard.jsx';
import ScalingCard from './ScalingCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import MetricCard from './MetricCard.jsx';
import SplitOptionsCard from './SplitOptionsCard.jsx';
import api from '../api/client';
import { useLearningCurveResultsCtx } from '../state/LearningCurveResultsContext.jsx';

export default function LearningCurvePanel() {
  const { xPath, yPath, npzPath, xKey, yKey, dataReady } = useDataCtx();

  const {
    method,
    pca_n, pca_var, pca_whiten,
    lda_n, lda_solver, lda_shrinkage, lda_tol,
    sfs_k, sfs_direction, sfs_cv, sfs_n_jobs,
  } = useFeatureCtx();

  // From LC results context (shared with right column)
  const {
    nSplits,
    setNSplits,
    withinPct,
    setWithinPct,
    setResult,
  } = useLearningCurveResultsCtx();

  // common split params
  const [stratified, setStratified] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42); // enabled only if shuffle

  // Scaling + Metric
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [metric, setMetric] = useState('accuracy');

  // --- NEW: model schema + flat model state (schema-driven) ----------------
  const [modelSchema, setModelSchema] = useState(null);

  // Flat model: keys match backend ModelModel (shared_schemas.model_configs)
  const [model, setModel] = useState({
    algo: 'logreg',
    // logreg
    C: 1.0,
    penalty: 'l2',
    solver: 'lbfgs',
    max_iter: 1000,
    class_weight: null,
    l1_ratio: 0.5,

    // svm
    svm_kernel: 'rbf',
    svm_C: 1.0,
    svm_degree: 3,
    svm_gamma: 'scale', // or 'auto' or numeric
    svm_coef0: 0.0,
    svm_shrinking: true,
    svm_probability: false,
    svm_tol: 1e-3,
    svm_cache_size: 200.0,
    svm_class_weight: null,
    svm_max_iter: -1,
    svm_decision_function_shape: 'ovr',
    svm_break_ties: false,

    // tree
    tree_criterion: 'gini',
    tree_splitter: 'best',
    tree_max_depth: null,
    tree_min_samples_split: 2,
    tree_min_samples_leaf: 1,
    tree_min_weight_fraction_leaf: 0.0,
    tree_max_features: null, // 'sqrt' | 'log2' | number | null
    tree_max_leaf_nodes: null,
    tree_min_impurity_decrease: 0.0,
    tree_class_weight: null,
    tree_ccp_alpha: 0.0,

    // forest
    forest_n_estimators: 100,
    forest_criterion: 'gini',
    forest_max_depth: null,
    forest_min_samples_split: 2,
    forest_min_samples_leaf: 1,
    forest_min_weight_fraction_leaf: 0.0,
    forest_max_features: 'sqrt',
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

    // knn
    knn_n_neighbors: 5,
    knn_weights: 'uniform',
    knn_algorithm: 'auto',
    knn_leaf_size: 30,
    knn_p: 2,
    knn_metric: 'minkowski',
    knn_n_jobs: null,
  });

  // Fetch schema/defaults to keep enums/defaults in sync with backend
  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const { data } = await api.get('/schema/model'); // served at /api/v1/schema/model
        if (!cancel) {
          setModelSchema(data?.schema ?? null);
          if (data?.defaults) {
            // Merge defaults over current model (preserve user changes if any)
            setModel((prev) => ({ ...data.defaults, ...prev }));
          }
        }
      } catch {
        // schema fetch optional; UI has fallbacks
      }
    })();
    return () => { cancel = true; };
  }, []);

  const [trainSizesCSV, setTrainSizesCSV] = useState('');
  const [nSteps, setNSteps] = useState(5);
  const [nJobs, setNJobs] = useState(1);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // With flat state, payload can be the model as-is (minor sanitization)
  const modelPayload = useMemo(() => {
    const m = model;

    // Ensure numeric-ish fields are numbers where appropriate
    const num = (v) => (v === '' || v == null ? v : Number(v));

    // SVM gamma can be 'scale' | 'auto' | number
    const svm_gamma = (typeof m.svm_gamma === 'number')
      ? Number(m.svm_gamma)
      : (m.svm_gamma ?? 'scale');

    return {
      algo: m.algo,

      // logreg
      C: num(m.C),
      penalty: m.penalty,
      solver: m.solver,
      max_iter: num(m.max_iter),
      class_weight: m.class_weight,
      ...(m.penalty === 'elasticnet' ? { l1_ratio: num(m.l1_ratio ?? 0.5) } : {}),

      // svm
      svm_C: num(m.svm_C),
      svm_kernel: m.svm_kernel,
      svm_degree: num(m.svm_degree),
      svm_gamma,
      svm_coef0: num(m.svm_coef0),
      svm_shrinking: !!m.svm_shrinking,
      svm_probability: !!m.svm_probability,
      svm_tol: num(m.svm_tol),
      svm_cache_size: num(m.svm_cache_size),
      svm_class_weight: m.svm_class_weight,
      svm_max_iter: num(m.svm_max_iter),
      svm_decision_function_shape: m.svm_decision_function_shape,
      svm_break_ties: !!m.svm_break_ties,

      // tree
      tree_criterion: m.tree_criterion,
      tree_splitter: m.tree_splitter,
      tree_max_depth: m.tree_max_depth == null ? null : num(m.tree_max_depth),
      tree_min_samples_split: num(m.tree_min_samples_split),
      tree_min_samples_leaf: num(m.tree_min_samples_leaf),
      tree_min_weight_fraction_leaf: num(m.tree_min_weight_fraction_leaf),
      tree_max_features: m.tree_max_features,
      tree_max_leaf_nodes: m.tree_max_leaf_nodes == null ? null : num(m.tree_max_leaf_nodes),
      tree_min_impurity_decrease: num(m.tree_min_impurity_decrease),
      tree_class_weight: m.tree_class_weight,
      tree_ccp_alpha: num(m.tree_ccp_alpha),

      // forest (all with forest_* prefix)
      forest_n_estimators: num(m.forest_n_estimators),
      forest_criterion: m.forest_criterion,
      forest_max_depth: m.forest_max_depth == null ? null : num(m.forest_max_depth),
      forest_min_samples_split: num(m.forest_min_samples_split),
      forest_min_samples_leaf: num(m.forest_min_samples_leaf),
      forest_min_weight_fraction_leaf: num(m.forest_min_weight_fraction_leaf),
      forest_max_features: m.forest_max_features,
      forest_max_leaf_nodes: m.forest_max_leaf_nodes == null ? null : num(m.forest_max_leaf_nodes),
      forest_min_impurity_decrease: num(m.forest_min_impurity_decrease),
      forest_bootstrap: !!m.forest_bootstrap,
      forest_oob_score: !!m.forest_oob_score,
      forest_n_jobs: m.forest_n_jobs == null ? null : num(m.forest_n_jobs),
      forest_random_state: m.forest_random_state == null ? null : num(m.forest_random_state),
      forest_warm_start: !!m.forest_warm_start,
      forest_class_weight: m.forest_class_weight,
      forest_ccp_alpha: num(m.forest_ccp_alpha),
      forest_max_samples: m.forest_max_samples == null ? null : num(m.forest_max_samples),

      // knn
      knn_n_neighbors: num(m.knn_n_neighbors),
      knn_weights: m.knn_weights,
      knn_algorithm: m.knn_algorithm,
      knn_leaf_size: num(m.knn_leaf_size),
      knn_p: num(m.knn_p),
      knn_metric: m.knn_metric,
      knn_n_jobs: m.knn_n_jobs == null ? null : num(m.knn_n_jobs),
    };
  }, [model]);

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
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
        },
        train_sizes,
        n_steps: Number(nSteps),
        n_jobs: Number(nJobs),
      };

      const { data } = await api.post('/learning-curve', payload);
      setResult(data);
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e.message ?? String(e);
      const msg = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

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

              <ScalingCard
                value={scaleMethod}
                onChange={setScaleMethod}
              />

              <FeatureCard title="Features" />

              <Divider my="xs" />

              <MetricCard
                value={metric}
                onChange={setMetric}
              />

              <Divider my="xs" />

              <ModelSelectionCard model={model} onChange={setModel} schema={modelSchema?.schema ? modelSchema.schema : modelSchema} />

              <Divider my="xs" />

              <NumberInput
                label="Steps (used if Train sizes empty)"
                min={2}
                max={50}
                step={1}
                value={nSteps}
                onChange={setNSteps}
              />
              <NumberInput
                label="n_jobs"
                min={1}
                step={1}
                value={nJobs}
                onChange={setNJobs}
              />
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
                min={0.5}
                max={1.0}
                step={0.01}
                value={withinPct}
                onChange={setWithinPct}
                precision={2}
              />
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear in the right-hand “Learning Curve Results” panel.
            </Text>
          )}
        </Stack>
      </Card>
    </Stack>
  );
}
