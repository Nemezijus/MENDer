import { useEffect, useMemo, useState } from 'react';
import { Stack, Card, Text, Group, Button, Alert, Divider, Code } from '@mantine/core';
import { useShallow } from 'zustand/react/shallow';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import UnsupervisedResultsPanel from './UnsupervisedResultsPanel.jsx';

import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';
import { useDataStore } from '../state/useDataStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';

import { runUnsupervisedTrainRequest } from '../api/unsupervised.js';

function cloneDefaults(obj) {
  if (!obj) return obj;
  if (typeof structuredClone === 'function') return structuredClone(obj);
  return JSON.parse(JSON.stringify(obj));
}

function formatAxiosError(e) {
  const detail = e?.response?.data?.detail ?? e?.response?.data?.message ?? e?.response?.data;
  if (detail) {
    if (typeof detail === 'string') return detail;
    try {
      return JSON.stringify(detail, null, 2);
    } catch {
      return String(detail);
    }
  }
  return e?.message || String(e);
}

export default function UnsupervisedTrainingPanel() {
  const schema = useSchemaDefaults();

  // Data selections come from the global Upload tab
  const { xPath, yPath, npzPath, xKey, yKey } = useDataStore(
    useShallow((s) => ({
      xPath: s.xPath,
      yPath: s.yPath,
      npzPath: s.npzPath,
      xKey: s.xKey,
      yKey: s.yKey,
    })),
  );

  // Global Settings
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const features = useFeatureStore((s) => s); // contains features.method and associated params

  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const artifact = useModelArtifactStore((s) => s.artifact);

  const [model, setModel] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleModelChange = (next) => {
    setModel(next);
  };

  // Available unsupervised algorithms from backend meta
  const unsupAlgos = useMemo(() => {
    return schema.getCompatibleAlgos?.('unsupervised') || [];
  }, [schema]);

  // Initialize model defaults when schema loads
  useEffect(() => {
    if (model?.algo) return;
    if (!unsupAlgos || unsupAlgos.length === 0) return;
    const first = unsupAlgos[0];
    const def = schema.getModelDefaults?.(first);
    setModel(def ? cloneDefaults(def) : { algo: first });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unsupAlgos, schema]);

  const hasX = !!(xPath || npzPath);
  const canRun = hasX && !!model?.algo && !schema.loading && !isRunning;

  // LDA and SFS require y (supervised)
  const needsY = features?.method === 'lda' || features?.method === 'sfs';
  const willOverrideFeatures = needsY && !yPath;

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const safeFeatures = {
        method: willOverrideFeatures ? 'none' : features.method,
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
      };

      const payload = {
        task: 'unsupervised',
        data: {
          x_path: xPath || null,
          y_path: yPath || null,     // ignored by backend for unsupervised
          npz_path: npzPath || null,
          x_key: xKey || null,
          y_key: yKey || null,
        },
        // we train here; applying to production happens via Predictions tab
        fit_scope: 'train_only',
        scale: { method: scaleMethod },
        features: safeFeatures,
        model,
        eval: {
          // UI does not select metrics; backend computes default pack + model diagnostics
          metrics: [],
          compute_embedding_2d: true,
          embedding_method: 'pca',
          per_sample_outputs: true,
          include_cluster_probabilities: false,
        },
        use_y_for_external_metrics: false,
        external_metrics: [],
      };

      const resp = await runUnsupervisedTrainRequest(payload);
      setResult(resp);
      if (resp?.artifact) setArtifact(resp.artifact);
    } catch (e) {
      setError(formatAxiosError(e));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="xs">
          <Text fw={700} size="lg">
            Unsupervised learning
          </Text>
          <Text size="sm" c="dimmed">
            Uses the training data and settings you already configured in <b>Upload data &amp; models</b> and <b>Settings</b>.
            This panel trains a clustering model and reports diagnostics. Use <b>Predictions</b> to apply a saved model to production data.
          </Text>
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="sm">
          <Text fw={700}>Current data</Text>

          {!hasX ? (
            <Alert color="red" title="Missing X">
              Select a Feature matrix (X) in <b>Upload data &amp; models</b>.
            </Alert>
          ) : (
            <Stack gap={6}>
              <Text size="sm">
                X: <Code>{npzPath || xPath}</Code>
              </Text>
              {yPath ? (
                <Alert color="blue" title="Note">
                  A <b>y</b> file is present, but unsupervised training ignores it.
                </Alert>
              ) : null}
            </Stack>
          )}
        </Stack>
      </Card>

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="sm">
          <Text fw={700}>Using global Settings</Text>
          <Text size="sm">
            Scaling: <Code>{scaleMethod}</Code>
          </Text>
          <Text size="sm">
            Features: <Code>{features?.method}</Code>
          </Text>

          {willOverrideFeatures ? (
            <Alert color="yellow" title="Feature method overridden">
              Your current Features method is <b>{features.method}</b>, which requires <b>y</b>.
              Since you are training unsupervised (X-only), this run will use <Code>none</Code> for Features.
              Change this in the <b>Settings</b> tab if you want a different unsupervised-compatible reduction (e.g., PCA).
            </Alert>
          ) : null}

          <Text size="sm" c="dimmed">
            Metrics are not selected here — the backend computes the default unsupervised metric pack (when applicable) and model-specific diagnostics automatically.
          </Text>
        </Stack>
      </Card>

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
          <Group justify="space-between" wrap="wrap">
            <Button onClick={handleRun} disabled={!canRun} loading={isRunning}>
              Train unsupervised model
            </Button>
            {model?.algo ? (
              <Text size="sm" c="dimmed">
                Selected algorithm: <b>{model.algo}</b>
              </Text>
            ) : null}
          </Group>

          <Divider />

          {error ? (
            <Alert color="red" title="Error">
              <Code block>{error}</Code>
            </Alert>
          ) : null}
        </Stack>
      </Card>

      <UnsupervisedResultsPanel result={result} artifact={artifact} />
    </Stack>
  );
}
