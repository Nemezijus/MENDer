// frontend/src/components/ModelCard.jsx
import { useEffect, useRef, useState } from 'react';
import { Card, Text, Group, Stack, Divider, Badge, Button, Tooltip } from '@mantine/core';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useDataStore } from '../state/useDataStore.js';
import { saveModel, saveBlobInteractive } from '../api/models';

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

function inferEnsembleKindFromClassPath(classPath) {
  if (!classPath) return null;
  const name = friendlyClassName(classPath);

  if (name === 'VotingClassifier' || name === 'VotingRegressor') return 'voting';
  if (name === 'BaggingClassifier' || name === 'BaggingRegressor') return 'bagging';
  if (name === 'AdaBoostClassifier' || name === 'AdaBoostRegressor') return 'adaboost';
  if (name === 'XGBClassifier' || name === 'XGBRegressor') return 'xgboost';

  return null;
}

function inferPrimaryLabel(artifact) {
  const algo = artifact?.model?.algo ?? null;

  // If artifact explicitly records an ensemble, prefer ensemble_kind.
  if (algo === 'ensemble') {
    const kind =
      artifact?.model?.ensemble_kind ??
      artifact?.model?.ensemble?.kind ??
      inferEnsembleKindFromClassPath(
        Array.isArray(artifact?.pipeline) && artifact.pipeline.length
          ? artifact.pipeline[artifact.pipeline.length - 1]?.class_path
          : null
      ) ??
      'ensemble';

    return { label: kind, isEnsemble: true, ensembleKind: kind };
  }

  // Normal single model: show algo from config
  if (algo) return { label: algo, isEnsemble: false, ensembleKind: null };

  // Fall back to pipeline last step class name
  const lastStep =
    Array.isArray(artifact?.pipeline) && artifact.pipeline.length
      ? artifact.pipeline[artifact.pipeline.length - 1]
      : null;

  const ensembleKind = inferEnsembleKindFromClassPath(lastStep?.class_path);
  if (ensembleKind) return { label: ensembleKind, isEnsemble: true, ensembleKind };

  // If not recognized, still show last estimator class (better than 'unknown')
  const lastCls = lastStep?.class_path ? friendlyClassName(lastStep.class_path) : null;
  if (lastCls) return { label: lastCls, isEnsemble: true, ensembleKind: null };

  return { label: 'unknown', isEnsemble: false, ensembleKind: null };
}

function HyperparamSummary({ modelCfg }) {
  if (!modelCfg || typeof modelCfg !== 'object') {
    return <Text size="sm" c="dimmed">No hyperparameters recorded.</Text>;
  }

  const entries = Object.entries(modelCfg)
    .filter(([k, v]) => !['algo', 'ensemble_kind', 'ensemble'].includes(k) && v !== undefined && v !== null)
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

function StepParamSummary({ step }) {
  const params = step?.params && typeof step.params === 'object' ? step.params : null;
  if (!params) return <Text size="sm" c="dimmed">No estimator parameters recorded.</Text>;

  const entries = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .slice(0, 6);

  if (!entries.length) {
    return <Text size="sm" c="dimmed">No estimator parameters recorded.</Text>;
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
  const inferredTaskRaw = inspectReport?.task_inferred || null;
  const inferredTask = inferredTaskRaw === 'clustering' ? 'unsupervised' : inferredTaskRaw;
  const effectiveTask = taskSelected || inferredTask || null;

  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const [info, setInfo] = useState(null);

  // 'none' | 'trained' | 'loaded' | 'incompatible'
  const [status, setStatus] = useState('none');
  const lastUidRef = useRef(null);

  /** ---------- compatibility checks ---------- **/
  let compatible = true;
  let compatReason = '';

  if (artifact && inspectReport) {
    const artifactKind = artifact.kind === 'clustering' ? 'unsupervised' : artifact.kind;
    if (artifactKind && effectiveTask && artifactKind !== effectiveTask) {
      compatible = false;
      compatReason =
        compatReason ||
        `Task mismatch: model is "${artifactKind}", data is "${effectiveTask}".`;
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

  const visualStatus = !artifact ? 'none' : !compatible ? 'incompatible' : status;

  const splitMode = artifact?.split?.mode ?? null;
  const isKFold = splitMode === 'kfold';
  const nSplits = artifact?.n_splits ?? null;

  // Under k-fold, the artifact stores per-fold train/test sizes.
  // Summing them yields an approximate total N for out-of-fold evaluation.
  const kfoldTotalN =
    isKFold && Number.isFinite(Number(artifact?.n_samples_train)) && Number.isFinite(Number(artifact?.n_samples_test))
      ? Number(artifact.n_samples_train) + Number(artifact.n_samples_test)
      : null;

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

  /** ---------- detect artifact changes ---------- **/
  useEffect(() => {
    if (!artifact) {
      setStatus('none');
      setInfo(null);
      lastUidRef.current = null;
      return;
    }

    if (artifact.uid && artifact.uid !== lastUidRef.current) {
      setStatus('trained');
      setInfo(null);
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
      const inferred = inferPrimaryLabel(artifact);
      const nameForFile = inferred?.label || 'unknown';
      const suggested = `model-${nameForFile}-${artifact.uid?.slice(0, 8) || 'artifact'}.mend`;

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

  /** ---------- status text ---------- **/
  let statusText = '';
  if (!artifact) {
    statusText = 'No model yet. Run training or load a saved model.';
  } else if (visualStatus === 'incompatible') {
    statusText = compatReason || 'Model is not compatible with the currently loaded data.';
  } else if (visualStatus === 'trained') {
    statusText = 'Model trained on current data.';
  } else if (visualStatus === 'loaded') {
    statusText = 'Model loaded from file.';
  } else {
    statusText = 'Model present.';
  }

  const nParameters =
    artifact && Object.prototype.hasOwnProperty.call(artifact, 'n_parameters')
      ? artifact.n_parameters
      : null;

  const extraStats =
    artifact && artifact.extra_stats && typeof artifact.extra_stats === 'object'
      ? artifact.extra_stats
      : {};

  const inferred = inferPrimaryLabel(artifact);
  const primaryLabel = inferred.label;
  const isEnsemble = inferred.isEnsemble;

  const ensembleCfg = artifact?.model?.algo === 'ensemble' ? artifact?.model?.ensemble : null;
  const ensembleKind =
    artifact?.model?.algo === 'ensemble'
      ? artifact?.model?.ensemble_kind ?? ensembleCfg?.kind ?? inferred.ensembleKind
      : null;

  const ensembleVoting = ensembleCfg?.voting ?? null;
  const ensembleNEstimators =
    Array.isArray(ensembleCfg?.estimators) ? ensembleCfg.estimators.length : null;

  const hasAnyExtraStat =
    extraStats.n_support_vectors != null ||
    extraStats.n_trees != null ||
    extraStats.total_tree_nodes != null ||
    extraStats.max_tree_depth != null ||
    extraStats.pca_n_components != null ||
    // Ensemble extras (optional; only if you merged them into artifact.extra_stats)
    extraStats.ensemble_all_agree_rate != null ||
    extraStats.ensemble_pairwise_agreement != null ||
    extraStats.ensemble_tie_rate != null ||
    extraStats.ensemble_mean_margin != null ||
    extraStats.ensemble_best_estimator != null ||
    extraStats.ensemble_corrected_vs_best != null ||
    extraStats.ensemble_harmed_vs_best != null;

  // Feature section: show PCA components if we can
  const featureCfg = artifact?.features ?? null;
  const featuresMethod = featureCfg?.method || 'none';

  let featuresText = featuresMethod;

  if (featuresMethod === 'pca') {
    const pcaNFromStats =
      extraStats && Object.prototype.hasOwnProperty.call(extraStats, 'pca_n_components')
        ? extraStats.pca_n_components
        : null;

    const pcaNFromCfg =
      featureCfg && Object.prototype.hasOwnProperty.call(featureCfg, 'pca_n')
        ? featureCfg.pca_n
        : null;

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

  const lastPipelineStep =
    Array.isArray(artifact?.pipeline) && artifact.pipeline.length
      ? artifact.pipeline[artifact.pipeline.length - 1]
      : null;

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

        <Text size="xs" c={statusColor} style={{ whiteSpace: 'pre-wrap' }}>
          {statusText}
        </Text>

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
              <Badge variant="light">{primaryLabel}</Badge>

              {isEnsemble && (
                <Badge color="cyan" variant="light">
                  ensemble
                </Badge>
              )}

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
              {isKFold ? (
                <Tooltip
                  label="Out-of-fold evaluation (pooled across folds): each sample is predicted by a model that did not train on it."
                  withArrow
                >
                  <Badge color="teal" variant="light">
                    OOF
                  </Badge>
                </Tooltip>
              ) : null}
            </Group>

            {isEnsemble && ensembleKind && (
              <Text size="sm">
                <strong>Ensemble:</strong> {ensembleKind}
                {ensembleVoting ? ` (${ensembleVoting})` : ''}
                {ensembleNEstimators != null ? ` • ${ensembleNEstimators} estimators` : ''}
              </Text>
            )}

            <Text size="sm">
              <strong>Created:</strong>{' '}
              {artifact.created_at ? new Date(artifact.created_at).toLocaleString() : '—'}
            </Text>

            <Text size="sm">
              <strong>Metric:</strong>{' '}
              {artifact.metric_name ?? '—'} ({fmt(artifact.mean_score)} ± {fmt(artifact.std_score)})
            </Text>

            <Text size="sm">
              <strong>Data:</strong>{' '}
              {isKFold ? (
                <>
                  train {fmt(artifact.n_samples_train, 0)} / fold, test {fmt(artifact.n_samples_test, 0)} / fold
                  {kfoldTotalN != null ? ` (N=${fmt(kfoldTotalN, 0)} OOF)` : ''}, features {fmt(artifact.n_features_in, 0)}
                </>
              ) : (
                <>
                  train {fmt(artifact.n_samples_train, 0)}, test {fmt(artifact.n_samples_test, 0)}, features {fmt(artifact.n_features_in, 0)}
                </>
              )}
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
              {isKFold ? ' (out-of-fold pooled)' : ''}
            </Text>

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              Model stats
            </Text>

            <Text size="sm">
              <strong>Parameters:</strong> {nParameters != null ? fmt(nParameters, 0) : 'not available'}
            </Text>

            {extraStats.n_support_vectors != null && (
              <Text size="sm">
                <strong>Support vectors:</strong> {fmt(extraStats.n_support_vectors, 0)}
              </Text>
            )}

            {extraStats.n_trees != null && (
              <Text size="sm">
                <strong>Trees:</strong> {fmt(extraStats.n_trees, 0)}
              </Text>
            )}

            {extraStats.total_tree_nodes != null && (
              <Text size="sm">
                <strong>Total tree nodes:</strong> {fmt(extraStats.total_tree_nodes, 0)}
              </Text>
            )}

            {extraStats.max_tree_depth != null && (
              <Text size="sm">
                <strong>Max tree depth:</strong> {fmt(extraStats.max_tree_depth, 0)}
              </Text>
            )}

            {/* Optional ensemble extras (only show if present) */}
            {isEnsemble && extraStats.ensemble_all_agree_rate != null && (
              <Text size="sm">
                <strong>All-agree rate:</strong> {fmt(Number(extraStats.ensemble_all_agree_rate) * 100, 2)}%
              </Text>
            )}

            {isEnsemble && extraStats.ensemble_pairwise_agreement != null && (
              <Text size="sm">
                <strong>Avg pairwise agreement:</strong>{' '}
                {fmt(Number(extraStats.ensemble_pairwise_agreement) * 100, 2)}%
              </Text>
            )}

            {isEnsemble && extraStats.ensemble_tie_rate != null && (
              <Text size="sm">
                <strong>Tie rate:</strong> {fmt(Number(extraStats.ensemble_tie_rate) * 100, 2)}%
              </Text>
            )}

            {isEnsemble && extraStats.ensemble_mean_margin != null && (
              <Text size="sm">
                <strong>Mean vote margin:</strong> {fmt(extraStats.ensemble_mean_margin, 3)}
              </Text>
            )}

            {isEnsemble && extraStats.ensemble_best_estimator != null && (
              <Text size="sm">
                <strong>Best estimator:</strong> {String(extraStats.ensemble_best_estimator)}
              </Text>
            )}

            {isEnsemble &&
              (extraStats.ensemble_corrected_vs_best != null || extraStats.ensemble_harmed_vs_best != null) && (
                <Text size="sm">
                  <strong>Vs best:</strong>{' '}
                  corrected {fmt(extraStats.ensemble_corrected_vs_best, 0)}, harmed {fmt(extraStats.ensemble_harmed_vs_best, 0)}
                </Text>
              )}

            {!hasAnyExtraStat && nParameters == null && (
              <Text size="sm" c="dimmed">
                Additional stats not available for this artifact.
              </Text>
            )}

            <Divider my="xs" />
            <Text size="sm" fw={500}>
              {artifact?.model ? 'Hyperparameters' : 'Estimator parameters'}
            </Text>

            {artifact?.model ? <HyperparamSummary modelCfg={artifact.model} /> : <StepParamSummary step={lastPipelineStep} />}

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              Pipeline
            </Text>

            {Array.isArray(artifact.pipeline) && artifact.pipeline.length > 0 ? (
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

        <Group justify="flex-end" mt="sm">
          <Tooltip label={artifact ? 'Save to a chosen location' : 'Run or load a model first'}>
            <Button size="xs" variant="light" onClick={onSave} disabled={!artifact} loading={saving}>
              Save model
            </Button>
          </Tooltip>
        </Group>
      </Stack>
    </Card>
  );
}
