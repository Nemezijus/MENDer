import { useEffect, useRef, useState } from 'react';
import { Card, Text, Group, Stack, Divider, Badge, Button, Tooltip } from '@mantine/core';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useDataStore } from '../state/useDataStore.js';
import { saveModel, loadModel, saveBlobInteractive } from '../api/models';

function fmt(val, digits = 4) {
  if (val == null || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

function friendlyClassName(classPath) {
  if (!classPath) return 'unknown';
  if (classPath === 'builtins.str') return 'none';
  const parts = classPath.split('.');
  return parts[parts.length - 1] || classPath;
}

function HyperparamSummary({ modelCfg }) {
  if (!modelCfg || typeof modelCfg !== 'object') {
    return <Text size="sm" c="dimmed">No hyperparameters recorded.</Text>;
  }

  const entries = Object.entries(modelCfg)
    .filter(([k, v]) => k !== 'algo' && v !== undefined && v !== null)
    .slice(0, 6);

  if (!entries.length) {
    return <Text size="sm" c="dimmed">No hyperparameters recorded.</Text>;
  }

  return (
    <Text size="sm">
      {entries.map(([k, v], idx) => (
        <span key={k}>
          {k}={String(v)}
          {idx < entries.length - 1 ? ', ' : ''}
        </span>
      ))}
    </Text>
  );
}

export default function ModelCard() {
  const artifact = useModelArtifactStore((s) => s.artifact);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const clearArtifact = useModelArtifactStore((s) => s.clearArtifact);

  const inspectReport = useDataStore((s) => s.inspectReport);
  const taskSelected = useDataStore((s) => s.taskSelected);
  const effectiveTask = taskSelected || inspectReport?.task_inferred || null;

  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [info, setInfo] = useState(null); // only for transient messages (save/load)

  // 'none' | 'trained' | 'loaded' | 'incompatible'
  const [status, setStatus] = useState('none');

  const fileInputRef = useRef(null);
  const lastUidRef = useRef(null);
  const justLoadedRef = useRef(false); // set only in onFileChosen

  /** ---------- compatibility checks ---------- **/
  let compatible = true;
  let compatReason = '';

  if (artifact && inspectReport) {
    if (artifact.kind && effectiveTask && artifact.kind !== effectiveTask) {
      compatible = false;
      compatReason =
        compatReason ||
        `Task mismatch: model is "${artifact.kind}", data is "${effectiveTask}".`;
    }

    if (
      artifact.n_features_in != null &&
      inspectReport.n_features != null &&
      artifact.n_features_in !== inspectReport.n_features
    ) {
      compatible = false;
      const r = `Feature count mismatch: model expects ${artifact.n_features_in}, data has ${inspectReport.n_features}.`;
      compatReason = compatReason ? `${compatReason} ${r}` : r;
    }

    if (
      artifact.kind === 'classification' &&
      Array.isArray(artifact.classes) &&
      Array.isArray(inspectReport.classes) &&
      artifact.classes.length !== inspectReport.classes.length
    ) {
      compatible = false;
      const r = `Number of classes changed: model trained on ${artifact.classes.length}, data has ${inspectReport.classes.length}.`;
      compatReason = compatReason ? `${compatReason} ${r}` : r;
    }
  }

  const visualStatus = !artifact
    ? 'none'
    : !compatible
    ? 'incompatible'
    : status;

  const borderColor =
    visualStatus === 'trained'
      ? 'var(--mantine-color-blue-4)'
      : visualStatus === 'loaded'
      ? 'var(--mantine-color-green-4)'
      : visualStatus === 'incompatible'
      ? 'var(--mantine-color-red-4)'
      : undefined;

  const statusColor =
    visualStatus === 'trained'
      ? 'blue'
      : visualStatus === 'loaded'
      ? 'green'
      : visualStatus === 'incompatible'
      ? 'red'
      : 'dimmed';

  /** ---------- detect artifact changes (train vs load) ---------- **/
  useEffect(() => {
    if (!artifact) {
      setStatus('none');
      setInfo(null);
      lastUidRef.current = null;
      return;
    }

    if (artifact.uid && artifact.uid !== lastUidRef.current) {
      if (justLoadedRef.current) {
        // We just loaded from disk, onFileChosen already set status + info
        justLoadedRef.current = false;
      } else {
        // Artifact changed due to a fresh training run
        setStatus('trained');
        // no info message here, to avoid duplicate text
        setInfo(null);
      }
      lastUidRef.current = artifact.uid;
    }
  }, [artifact]);

  /** ---------- actions ---------- **/
  async function onSave() {
    if (!artifact) return;
    setErr(null);
    setInfo(null);
    setSaving(true);
    try {
      const suggested = `model-${artifact?.model?.algo || 'unknown'}-${artifact.uid?.slice(0, 8) || 'artifact'}.mend`;
      const { blob, filename } = await saveModel({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        filename: suggested,
      });
      const usedInteractive = await saveBlobInteractive(blob, filename);
      if (!usedInteractive) {
        setInfo(`Saved to your browser's default Downloads as "${filename}".`);
      }
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  }

  function onClickLoad() {
    setErr(null);
    setInfo(null);
    fileInputRef.current?.click();
  }

  async function onFileChosen(ev) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setErr(null);
    try {
      const { artifact: loadedMeta } = await loadModel(file);
      justLoadedRef.current = true;
      setArtifact(loadedMeta);
      setStatus('loaded');
      setInfo(`Loaded model "${file.name}".`);
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
      ev.target.value = '';
    }
  }

  /** ---------- status text ---------- **/
  let statusText = '';
  if (!artifact) {
    statusText = 'No model yet. Run training or load a saved model.';
  } else if (visualStatus === 'incompatible') {
    statusText =
      compatReason ||
      'Model is not compatible with the currently loaded data.';
  } else if (visualStatus === 'trained') {
    statusText = 'Model trained on current data.';
  } else if (visualStatus === 'loaded') {
    statusText = 'Model loaded from file.';
  } else {
    statusText = 'Model present.';
  }

  // Convenience for stats; tolerate old artifacts that don't have these fields.
  const nParameters =
    artifact && Object.prototype.hasOwnProperty.call(artifact, 'n_parameters')
      ? artifact.n_parameters
      : null;
  const extraStats =
    artifact && artifact.extra_stats && typeof artifact.extra_stats === 'object'
      ? artifact.extra_stats
      : {};

  const hasAnyExtraStat =
    extraStats.n_support_vectors != null ||
    extraStats.n_trees != null ||
    extraStats.total_tree_nodes != null ||
    extraStats.max_tree_depth != null ||
    extraStats.pca_n_components != null;

  // Feature section: show PCA components if we can
  const featureCfg = artifact?.features ?? null;
  const featuresMethod = featureCfg?.method || 'none';

  let featuresText = featuresMethod;

  if (featuresMethod === 'pca') {
    // 1) Actual fitted components (if backend ever adds this into extra_stats)
    const pcaNFromStats =
      extraStats && Object.prototype.hasOwnProperty.call(extraStats, 'pca_n_components')
        ? extraStats.pca_n_components
        : null;

    // 2) Configured pca_n (manual)
    const pcaNFromCfg =
      featureCfg && Object.prototype.hasOwnProperty.call(featureCfg, 'pca_n')
        ? featureCfg.pca_n
        : null;

    // 3) Variance-based PCA
    const pcaVarFromCfg =
      featureCfg && Object.prototype.hasOwnProperty.call(featureCfg, 'pca_var')
        ? featureCfg.pca_var
        : null;

    const nComp = pcaNFromStats != null ? pcaNFromStats : pcaNFromCfg;

    if (nComp != null) {
      featuresText = `pca (${nComp} components)`;
    } else if (pcaVarFromCfg != null) {
      featuresText = `pca (var=${pcaVarFromCfg})`;
    } else {
      featuresText = 'pca';
    }
  }

  return (
    <Card
      withBorder
      radius="md"
      shadow="sm"
      padding="md"
      style={borderColor ? { borderColor } : undefined}
    >
      <Stack gap="sm">
        <Text fw={600} ta="center">
          Model
        </Text>

        {/* Status line */}
        <Text size="xs" c={statusColor} style={{ whiteSpace: 'pre-wrap' }}>
          {statusText}
        </Text>

        {/* Transient info (save/load) */}
        {info && (
          <Text size="xs" c="teal" style={{ whiteSpace: 'pre-wrap' }}>
            {info}
          </Text>
        )}

        {err && (
          <Text size="xs" c="red" style={{ whiteSpace: 'pre-wrap' }}>
            {err}
          </Text>
        )}

        {!artifact ? (
          <Text size="sm" c="dimmed">
            Once you train or load a model, details will appear here.
          </Text>
        ) : (
          <Stack gap="xs">
            <Group gap="sm">
              <Badge variant="light">
                {artifact?.model?.algo ?? 'unknown'}
              </Badge>
              {artifact?.kind && (
                <Badge color="gray" variant="light">
                  {artifact.kind}
                </Badge>
              )}
              {artifact?.n_splits ? (
                <Badge color="grape" variant="light">
                  {artifact.n_splits} splits
                </Badge>
              ) : null}
            </Group>

            <Text size="sm">
              <strong>Created:</strong>{' '}
              {artifact.created_at
                ? new Date(artifact.created_at).toLocaleString()
                : '—'}
            </Text>

            <Text size="sm">
              <strong>Metric:</strong>{' '}
              {artifact.metric_name ?? '—'} (
              {fmt(artifact.mean_score)} ± {fmt(artifact.std_score)})
            </Text>

            <Text size="sm">
              <strong>Data:</strong>{' '}
              train {fmt(artifact.n_samples_train, 0)}, test{' '}
              {fmt(artifact.n_samples_test, 0)}, features{' '}
              {fmt(artifact.n_features_in, 0)}
            </Text>

            {Array.isArray(artifact.classes) && artifact.classes.length > 0 && (
              <Text size="sm">
                <strong>Classes:</strong> {artifact.classes.join(', ')}
              </Text>
            )}

            <Text size="sm">
              <strong>Scaler:</strong> {artifact?.scale?.method ?? 'none'}
            </Text>

            <Text size="sm">
              <strong>Features:</strong> {featuresText}
            </Text>

            <Text size="sm">
              <strong>Split:</strong> {artifact?.split?.mode ?? '—'}
            </Text>

            {/* Model stats */}
            <Divider my="xs" />
            <Text size="sm" fw={500}>
              Model stats
            </Text>
            <Text size="sm">
              <strong>Parameters:</strong>{' '}
              {nParameters != null ? fmt(nParameters, 0) : 'not available'}
            </Text>
            {extraStats.n_support_vectors != null && (
              <Text size="sm">
                <strong>Support vectors:</strong>{' '}
                {fmt(extraStats.n_support_vectors, 0)}
              </Text>
            )}
            {extraStats.n_trees != null && (
              <Text size="sm">
                <strong>Trees:</strong> {fmt(extraStats.n_trees, 0)}
              </Text>
            )}
            {extraStats.total_tree_nodes != null && (
              <Text size="sm">
                <strong>Total tree nodes:</strong>{' '}
                {fmt(extraStats.total_tree_nodes, 0)}
              </Text>
            )}
            {extraStats.max_tree_depth != null && (
              <Text size="sm">
                <strong>Max tree depth:</strong>{' '}
                {fmt(extraStats.max_tree_depth, 0)}
              </Text>
            )}
            {!hasAnyExtraStat && nParameters == null && (
              <Text size="sm" c="dimmed">
                Additional stats not available for this artifact.
              </Text>
            )}

            {/* Hyperparameters */}
            {artifact?.model && (
              <>
                <Divider my="xs" />
                <Text size="sm" fw={500}>
                  Hyperparameters
                </Text>
                <HyperparamSummary modelCfg={artifact.model} />
              </>
            )}

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              Pipeline
            </Text>
            {Array.isArray(artifact.pipeline) &&
            artifact.pipeline.length > 0 ? (
              <ul style={{ margin: 0, paddingInlineStart: 18 }}>
                {artifact.pipeline.map((s, i) => (
                  <li key={`${s.name}-${i}`}>
                    <Text size="sm">
                      <strong>{s.name}</strong>: {friendlyClassName(s.class_path)}
                    </Text>
                  </li>
                ))}
              </ul>
            ) : (
              <Text size="sm" c="dimmed">
                No pipeline details.
              </Text>
            )}
          </Stack>
        )}

        {/* Buttons at the bottom */}
        <Group justify="flex-end" mt="sm">
          <Tooltip
            label={
              artifact
                ? 'Save to a chosen location'
                : 'Run or load a model first'
            }
          >
            <Button
              size="xs"
              variant="light"
              onClick={onSave}
              disabled={!artifact}
              loading={saving}
            >
              Save model
            </Button>
          </Tooltip>
          <Button size="xs" variant="light" onClick={onClickLoad} loading={loading}>
            Load model
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".mend,application/octet-stream"
            style={{ display: 'none' }}
            onChange={onFileChosen}
          />
        </Group>
      </Stack>
    </Card>
  );
}
