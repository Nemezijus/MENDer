import { useEffect, useMemo, useState } from 'react';
import { Stack, Card, Text, Group, Button, Alert, Title } from '@mantine/core';
import { useShallow } from 'zustand/react/shallow';

import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';
import { useDataStore } from '../state/useDataStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useResultsStore } from '../state/useResultsStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
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
  const features = useFeatureStore((s) => s);

  // Results + artifact persistence
  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const [model, setModel] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

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

  // Keep model hydrated with backend defaults for current algo
  useEffect(() => {
    if (!model?.algo) return;
    const base = schema.getModelDefaults?.(model.algo) || { algo: model.algo };
    setModel((prev) => ({ ...cloneDefaults(base), ...(prev || {}) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema.getModelDefaults, model?.algo]);

  const hasX = !!(xPath || npzPath);
  const canRun = hasX && !!model?.algo && !schema.loading && !isRunning;

  // LDA and SFS require y (supervised). For X-only unsupervised, override to 'none'.
  const needsY = features?.method === 'lda' || features?.method === 'sfs';
  const willOverrideFeatures = needsY && !yPath;

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);

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
          y_path: yPath || null, // ignored by backend for unsupervised
          npz_path: npzPath || null,
          x_key: xKey || null,
          y_key: yKey || null,
        },
        // Applying to production is done via the Predictions tab
        fit_scope: 'train_only',
        scale: { method: scaleMethod },
        features: safeFeatures,
        model,
        eval: {
          // Empty list => backend computes default pack (and model-specific diagnostics)
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
      setTrainResult(resp);
      setActiveResultKind('train');
      if (resp?.artifact) setArtifact(resp.artifact);
    } catch (e) {
      setError(formatAxiosError(e));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Stack gap="md">
      <Title order={3}>Train an unsupervised Model</Title>

      <Card withBorder shadow="sm" radius="md" padding="md">
        <Stack gap="sm">
          <Group justify="space-between" align="flex-start" wrap="wrap">
            <Text fw={600}>Configuration</Text>
            <Button size="xs" onClick={handleRun} disabled={!canRun} loading={isRunning}>
              Run
            </Button>
          </Group>

          {!hasX ? (
            <Alert color="red" title="Missing Feature matrix (X)">
              Select a Feature matrix (X) in <b>Upload data &amp; models</b>.
            </Alert>
          ) : null}

          {willOverrideFeatures ? (
            <Alert color="yellow" title="Features overridden for X-only unsupervised">
              Your current Features method (<b>{features.method}</b>) requires <b>y</b>. Since this run is X-only,
              Features will be treated as <b>None</b>. You can change this in the <b>Settings</b> tab (e.g., use PCA).
            </Alert>
          ) : null}

          {error ? (
            <Alert color="red" title="Error">
              <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                {error}
              </Text>
            </Alert>
          ) : null}

          <ModelSelectionCard
            model={model}
            onChange={setModel}
            schema={schema.models?.schema}
            enums={schema.enums}
            models={schema.models}
            taskOverride="unsupervised"
            showHelp
          />
        </Stack>
      </Card>

      <Text size="sm" c="dimmed">
        This uses your current global Scaling / Metric / Features settings from the Settings section.
      </Text>
    </Stack>
  );
}
