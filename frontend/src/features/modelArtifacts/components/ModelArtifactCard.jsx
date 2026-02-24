import { useMemo, useState } from 'react';
import { Card, Text, Stack, Divider, Group, Button, Tooltip } from '@mantine/core';

import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { saveModel } from '../api/modelsApi.js';
import { downloadBlob, saveBlobInteractive } from '../../../shared/utils/download.js';
import { toErrorText } from '../../../shared/utils/errors.js';

import ArtifactBadges from './ArtifactBadges.jsx';
import ArtifactOverview from './ArtifactOverview.jsx';
import ArtifactStats from './ArtifactStats.jsx';
import ArtifactParams from './ArtifactParams.jsx';
import ArtifactPipeline from './ArtifactPipeline.jsx';

import {
  computeCompatibility,
  computeFeaturesText,
  inferPrimaryLabel,
} from '../utils/artifactUtils.js';

function getStatusColors(visualStatus) {
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

  return { borderColor, statusColor };
}

function makeSuggestedFilename(artifact, inferred) {
  const base = String(inferred?.raw || inferred?.label || 'model')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  const uid8 = artifact?.uid?.slice(0, 8) || 'artifact';
  return `model-${base || 'model'}-${uid8}.mend`;
}

export default function ModelArtifactCard() {
  const artifact = useModelArtifactStore((s) => s.artifact);
  const source = useModelArtifactStore((s) => s.source);

  const inspectReport = useDataStore((s) => s.inspectReport);
  const taskSelected = useDataStore((s) => s.taskSelected);

  const inferredTaskRaw = inspectReport?.task_inferred || null;
  const inferredTask = inferredTaskRaw === 'clustering' ? 'unsupervised' : inferredTaskRaw;
  const effectiveTask = taskSelected || inferredTask || null;

  const isUnsupervised =
    (artifact?.kind === 'clustering' ? 'unsupervised' : artifact?.kind) === 'unsupervised';

  const { compatible, reason: compatReason } = useMemo(() => {
    return computeCompatibility({ artifact, inspectReport, effectiveTask });
  }, [artifact, effectiveTask, inspectReport]);

  const visualStatus = !artifact
    ? 'none'
    : !compatible
      ? 'incompatible'
      : source === 'loaded'
        ? 'loaded'
        : 'trained';

  const { borderColor, statusColor } = getStatusColors(visualStatus);

  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const [info, setInfo] = useState(null);

  const splitMode = artifact?.split?.mode ?? null;
  const isKFold = splitMode === 'kfold';
  const nSplits = artifact?.n_splits ?? null;

  const kfoldTotalN =
    isKFold && Number.isFinite(Number(artifact?.n_samples_train)) && Number.isFinite(Number(artifact?.n_samples_test))
      ? Number(artifact.n_samples_train) + Number(artifact.n_samples_test)
      : null;

  const featuresText = useMemo(() => computeFeaturesText(artifact), [artifact]);

  const inferred = useMemo(() => inferPrimaryLabel(artifact), [artifact]);
  const primaryLabel = inferred.label;
  const isEnsemble = inferred.isEnsemble;

  const ensembleCfg = artifact?.model?.algo === 'ensemble' ? artifact?.model?.ensemble : null;
  const ensembleKind =
    artifact?.model?.algo === 'ensemble'
      ? artifact?.model?.ensemble_kind ?? ensembleCfg?.kind ?? inferred.ensembleKind
      : null;
  const ensembleVoting = ensembleCfg?.voting ?? null;
  const ensembleNEstimators = Array.isArray(ensembleCfg?.estimators) ? ensembleCfg.estimators.length : null;

  const ensembleSummary =
    isEnsemble && ensembleKind
      ? `Ensemble: ${String(ensembleKind)}${ensembleVoting ? ` (${ensembleVoting})` : ''}${
          ensembleNEstimators != null ? ` • ${ensembleNEstimators} estimators` : ''
        }`
      : null;

  const lastPipelineStep =
    Array.isArray(artifact?.pipeline) && artifact.pipeline.length
      ? artifact.pipeline[artifact.pipeline.length - 1]
      : null;

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

  async function onSave() {
    if (!artifact) return;
    setErr(null);
    setInfo(null);
    setSaving(true);
    try {
      const suggested = makeSuggestedFilename(artifact, inferred);
      const { blob, filename } = await saveModel({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        filename: suggested,
      });

      const supported = typeof window !== 'undefined' && 'showSaveFilePicker' in window;
      if (supported) {
        const ok = await saveBlobInteractive(blob, filename);
        if (ok === false) setInfo('Save canceled.');
      } else {
        downloadBlob(blob, filename);
        setInfo(`Saved to your browser's default Downloads as "${filename}".`);
      }
    } catch (e) {
      setErr(toErrorText(e) || 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card withBorder radius="md" shadow="sm" padding="md" style={borderColor ? { borderColor } : undefined}>
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
            <ArtifactBadges
              primaryLabel={primaryLabel}
              isEnsemble={isEnsemble}
              kind={artifact?.kind}
              nSplits={nSplits}
              isKFold={isKFold}
            />

            <ArtifactOverview
              artifact={artifact}
              isUnsupervised={isUnsupervised}
              isKFold={isKFold}
              kfoldTotalN={kfoldTotalN}
              featuresText={featuresText}
              ensembleSummary={ensembleSummary}
            />

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              Model stats
            </Text>
            <ArtifactStats artifact={artifact} isEnsemble={isEnsemble} />

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              {artifact?.model ? 'Hyperparameters' : 'Estimator parameters'}
            </Text>
            <ArtifactParams artifact={artifact} lastPipelineStep={lastPipelineStep} />

            <Divider my="xs" />

            <Text size="sm" fw={500}>
              Pipeline
            </Text>
            <ArtifactPipeline artifact={artifact} />
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
