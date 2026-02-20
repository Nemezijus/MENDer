import { useEffect, useMemo, useState } from 'react';
import { Stack, Card, Text, Group, Button, Alert, Title } from '@mantine/core';
import { useShallow } from 'zustand/react/shallow';

import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';
import { useDataStore } from '../state/useDataStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useResultsStore } from '../state/useResultsStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useUnsupervisedStore } from '../state/useUnsupervisedStore.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import { runUnsupervisedTrainRequest } from '../api/unsupervised.js';
import { compactPayload } from '../utils/compactPayload.js';

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
  const {
    method: featureMethod,
    pca_n,
    pca_var,
    pca_whiten,
    lda_n,
    lda_solver,
    lda_shrinkage,
    lda_tol,
    sfs_k,
    sfs_direction,
    sfs_cv,
    sfs_n_jobs,
    setMethod,
  } = useFeatureStore(
    useShallow((s) => ({
      method: s.method,
      pca_n: s.pca_n,
      pca_var: s.pca_var,
      pca_whiten: s.pca_whiten,
      lda_n: s.lda_n,
      lda_solver: s.lda_solver,
      lda_shrinkage: s.lda_shrinkage,
      lda_tol: s.lda_tol,
      sfs_k: s.sfs_k,
      sfs_direction: s.sfs_direction,
      sfs_cv: s.sfs_cv,
      sfs_n_jobs: s.sfs_n_jobs,
      setMethod: s.setMethod,
    })),
  );

  // Results + artifact persistence
  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const {
    model,
    algo,
    fitScope,
    metrics,
    includeClusterProbabilities,
    embeddingMethod,
    setModel,
    hydrateModel,
  } = useUnsupervisedStore(
    useShallow((s) => ({
      model: s.model,
      algo: s.algo,
      fitScope: s.fitScope,
      metrics: s.metrics,
      includeClusterProbabilities: s.includeClusterProbabilities,
      embeddingMethod: s.embeddingMethod,
      setModel: s.setModel,
      hydrateModel: s.hydrateModel,
    })),
  );
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [clearedFeature, setClearedFeature] = useState(null);

  // Available unsupervised algorithms from backend meta
  const unsupAlgos = useMemo(() => {
    return schema.getCompatibleAlgos?.('unsupervised') || [];
  }, [schema]);

  const shallowEqual = (a, b) => {
    if (a === b) return true;
    if (!a || !b) return false;
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (let i = 0; i < ka.length; i += 1) {
      const k = ka[i];
      if (a[k] !== b[k]) return false;
    }
    return true;
  };

  // Initialize model defaults once (and keep selections across navigation).
  useEffect(() => {
    if (model?.algo) return;
    if (!unsupAlgos || unsupAlgos.length === 0) return;
    const preferredAlgo = algo || unsupAlgos[0];
    const def = schema.getModelDefaults?.(preferredAlgo);
    const base = def ? cloneDefaults(def) : { algo: preferredAlgo };
    hydrateModel(base, model);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unsupAlgos, schema, algo]);

  // Keep model hydrated with backend defaults for current algo (without resetting user tweaks).
  useEffect(() => {
    if (!model?.algo) return;
    const base = schema.getModelDefaults?.(model.algo) || { algo: model.algo };
    const merged = { ...cloneDefaults(base), ...(model || {}) };
    if (!shallowEqual(merged, model)) setModel(merged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema, model?.algo]);

  const hasX = !!(xPath || npzPath);
  const canRun = hasX && !!model?.algo && !schema.loading && !isRunning;

  // IMPORTANT: the Engine unsupervised pipeline fits on X only (y is ignored), so
  // supervised feature methods (LDA/SFS) are not supported here.
  // Cleaner behavior: if a user has an override set to an incompatible method, clear it.
  useEffect(() => {
    const isSupervisedOnly = featureMethod === 'lda' || featureMethod === 'sfs';
    if (!isSupervisedOnly) return;
    setMethod(undefined);
    setClearedFeature(featureMethod);
  }, [featureMethod, setMethod]);

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);

    try {
      const dataPayload = compactPayload({
        x_path: xPath || null,
        y_path: yPath || null,
        npz_path: npzPath || null,
        x_key: xKey?.trim() || undefined,
        y_key: yKey?.trim() || undefined,
      });

      const scalePayload = compactPayload({ method: scaleMethod });

      const featuresPayload = compactPayload({
        method: featureMethod,
        pca_n,
        pca_var,
        pca_whiten,
        lda_n,
        lda_solver,
        lda_shrinkage,
        lda_tol,
        sfs_k,
        sfs_direction,
        sfs_cv,
        sfs_n_jobs,
      });

      const evalPayload = compactPayload({
        metrics: Array.isArray(metrics) ? metrics : metrics,
        include_cluster_probabilities: includeClusterProbabilities,
        embedding_method: embeddingMethod,
      });

      // Overrides-only: omit unset fields; keep required objects (scale/features/eval)
      const payload = {
        task: 'unsupervised',
        data: dataPayload,
        ...(fitScope !== undefined ? { fit_scope: fitScope } : {}),
        scale: scalePayload,
        features: featuresPayload,
        model,
        eval: evalPayload,
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

          {clearedFeature ? (
            <Alert color="yellow" title="Incompatible Features method cleared">
              The Features method <b>{clearedFeature}</b> is supervised-only and is not supported for unsupervised
              training (the Engine fits on <b>X only</b>). The override was cleared. Use <b>PCA</b> or <b>None</b> in the
              <b> Settings</b> tab.
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
        This uses your current global Scaling / Features settings from the Settings section.
      </Text>
    </Stack>
  );
}
