import { useEffect, useMemo, useState } from 'react';
import {
  Stack,
  Card,
  Text,
  Group,
  Button,
  Alert,
  Select,
  MultiSelect,
  Checkbox,
  NumberInput,
  Divider,
} from '@mantine/core';

import TrainingDataUploadCard from './TrainingDataUploadCard.jsx';
import ScalingCard from './ScalingCard.jsx';
import FeatureCard from './FeatureCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import UnsupervisedResultsPanel from './UnsupervisedResultsPanel.jsx';

import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';
import { useDataStore } from '../state/useDataStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useUnsupervisedStore } from '../state/useUnsupervisedStore.js';

import { runUnsupervisedTrainRequest } from '../api/unsupervised.js';

function cloneDefaults(obj) {
  if (!obj) return obj;
  if (typeof structuredClone === 'function') return structuredClone(obj);
  return JSON.parse(JSON.stringify(obj));
}

export default function UnsupervisedTrainingPanel() {
  const schema = useSchemaDefaults();

  const { xPath, yPath, npzPath, xKey, yKey } = useDataStore((s) => ({
    xPath: s.xPath,
    yPath: s.yPath,
    npzPath: s.npzPath,
    xKey: s.xKey,
    yKey: s.yKey,
  }));

  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const setScaleMethod = useSettingsStore((s) => s.setScaleMethod);

  const features = useFeatureStore((s) => s);

  const { algo, setAlgo, fitScope, setFitScope, metrics, setMetrics, includeClusterProbabilities, setIncludeClusterProbabilities, embeddingMethod, setEmbeddingMethod, embeddingMaxPoints, setEmbeddingMaxPoints } =
    useUnsupervisedStore((s) => ({
      algo: s.algo,
      setAlgo: s.setAlgo,
      fitScope: s.fitScope,
      setFitScope: s.setFitScope,
      metrics: s.metrics,
      setMetrics: s.setMetrics,
      includeClusterProbabilities: s.includeClusterProbabilities,
      setIncludeClusterProbabilities: s.setIncludeClusterProbabilities,
      embeddingMethod: s.embeddingMethod,
      setEmbeddingMethod: s.setEmbeddingMethod,
      embeddingMaxPoints: s.embeddingMaxPoints,
      setEmbeddingMaxPoints: s.setEmbeddingMaxPoints,
    }));

  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const artifact = useModelArtifactStore((s) => s.artifact);

  const [model, setModel] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleModelChange = (next) => {
    setModel(next);
    const nextAlgo = next?.algo ?? null;
    if (nextAlgo && nextAlgo !== algo) setAlgo(nextAlgo);
  };

  // Build candidate algorithms from backend metadata
  const unsupAlgos = useMemo(() => {
    const list = schema.getCompatibleAlgos?.('unsupervised') || [];
    return list;
  }, [schema]);

  // Initialize default algo + model config when schema defaults arrive
  useEffect(() => {
    if (algo) return;
    if (!unsupAlgos || unsupAlgos.length === 0) return;
    const first = unsupAlgos[0];
    setAlgo(first);
    const def = schema.getModelDefaults?.(first);
    setModel(def ? cloneDefaults(def) : { algo: first });
  }, [algo, unsupAlgos, schema, setAlgo]);

  // When algo changes via the store, keep local model in sync (reset to defaults)
  useEffect(() => {
    if (!algo) return;
    if (model?.algo === algo) return;
    const def = schema.getModelDefaults?.(algo);
    setModel(def ? cloneDefaults(def) : { algo });
  }, [algo]);

  const metricOptions = (schema.enums?.UnsupervisedMetricName || ['silhouette', 'davies_bouldin', 'calinski_harabasz']).map(
    (m) => ({ value: String(m), label: String(m) }),
  );

  const fitScopeOptions = (schema.enums?.FitScopeName || ['train_only', 'train_and_predict']).map((v) => ({
    value: String(v),
    label: String(v),
  }));

  const canRun = !!(xPath || npzPath) && !!model?.algo && !schema.loading;

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        task: 'unsupervised',
        data: {
          x_path: xPath || null,
          y_path: yPath || null,
          npz_path: npzPath || null,
          x_key: xKey || null,
          y_key: yKey || null,
        },
        apply: null,
        fit_scope: fitScope,
        scale: {
          method: scaleMethod,
        },
        features: {
          method: features.method,
          pca_n: features.pca_n,
          pca_var: features.pca_var,
          pca_whiten: features.pca_whiten,
          lda_n: features.lda_n,
          lda_solver: features.lda_solver,
          lda_shrinkage: features.lda_shrinkage,
          lda_tol: features.lda_tol,
          lda_priors: features.lda_priors,
          sfs_k: features.sfs_k,
          sfs_direction: features.sfs_direction,
          sfs_cv: features.sfs_cv,
          sfs_n_jobs: features.sfs_n_jobs,
        },
        model,
        eval: {
          metrics: metrics || [],
          seed: null,
          compute_embedding_2d: true,
          embedding_method: embeddingMethod,
          per_sample_outputs: true,
          include_cluster_probabilities: !!includeClusterProbabilities,
        },
        use_y_for_external_metrics: false,
        external_metrics: [],
      };

      const resp = await runUnsupervisedTrainRequest(payload);
      setResult(resp);
      if (resp?.artifact) setArtifact(resp.artifact);
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Stack gap="md">
      <TrainingDataUploadCard />

      {yPath ? (
        <Alert color="blue" title="Note">
          A <b>y</b> file was provided, but unsupervised training ignores it. If you want to evaluate clusters against known labels, we will add that later.
        </Alert>
      ) : null}

      <ScalingCard value={scaleMethod} onChange={setScaleMethod} title="Scaling" />

      <FeatureCard />

      <ModelSelectionCard
        model={model}
        onChange={handleModelChange}
        schema={schema.models?.schema}
        enums={schema.enums}
        models={schema.models}
        taskOverride="unsupervised"
        showHelp
      />

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Text fw={700} size="lg" align="center">
            Unsupervised settings
          </Text>

          <Group grow align="flex-start" wrap="nowrap" gap="xl">
            <Stack gap="sm" style={{ flex: 1, minWidth: 0 }}>
              <Select
                label="Fit scope"
                data={fitScopeOptions}
                value={fitScope}
                onChange={(v) => v && setFitScope(v)}
              />
              <MultiSelect
                label="Metrics (leave empty for default pack)"
                data={metricOptions}
                value={metrics || []}
                onChange={setMetrics}
                searchable
                clearable
              />
              <Checkbox
                label="Include cluster probabilities (only if supported)"
                checked={!!includeClusterProbabilities}
                onChange={(e) => setIncludeClusterProbabilities(e.currentTarget.checked)}
              />
            </Stack>

            <Stack gap="sm" style={{ flex: 1, minWidth: 220 }}>
              <Text size="sm" c="dimmed">
                Diagnostics
              </Text>
              <Select
                label="Embedding method"
                data={[{ value: 'pca', label: 'pca' }]}
                value={embeddingMethod}
                onChange={(v) => v && setEmbeddingMethod(v)}
              />
              <NumberInput
                label="Embedding max points"
                value={embeddingMaxPoints}
                onChange={setEmbeddingMaxPoints}
                allowDecimal={false}
                min={100}
              />
            </Stack>
          </Group>

          <Divider />

          <Group justify="space-between" wrap="wrap">
            <Button onClick={handleRun} disabled={!canRun} loading={isRunning}>
              Train unsupervised model
            </Button>
            {algo ? (
              <Text size="sm" c="dimmed">
                Selected algorithm: <b>{algo}</b>
              </Text>
            ) : null}
          </Group>

          {error ? (
            <Alert color="red" title="Error">
              {error}
            </Alert>
          ) : null}
        </Stack>
      </Card>

      <UnsupervisedResultsPanel result={result} artifact={artifact} />
    </Stack>
  );
}
