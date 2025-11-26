import { useRef, useState } from 'react';
import { Card, Text, Group, Stack, Divider, Badge, Button, Tooltip } from '@mantine/core';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { saveModel, loadModel, saveBlobInteractive } from '../api/models';

function fmt(val, digits = 4) {
  if (val == null || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

export default function ModelCard() {
  const artifact = useModelArtifactStore((s) => s.artifact);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const clearArtifact = useModelArtifactStore((s) => s.clearArtifact);

  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [info, setInfo] = useState(null);   // NEW: tiny success/fallback message

  const fileInputRef = useRef(null);

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
    try {
      const { artifact: loadedMeta } = await loadModel(file);
      setArtifact(loadedMeta);
      setInfo(`Loaded model "${file.name}".`);
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
      ev.target.value = '';
    }
  }

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Group justify="space-between" align="center" mb="xs">
        <Text fw={600}>Model</Text>
        <Group gap="xs">
          <Tooltip label={artifact ? 'Save to a chosen location' : 'Run or load a model first'}>
            <Button size="xs" variant="light" onClick={onSave} disabled={!artifact} loading={saving}>
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
      </Group>

      {err && (
        <Text size="xs" c="red" mb="xs" style={{ whiteSpace: 'pre-wrap' }}>
          {err}
        </Text>
      )}
      {info && (
        <Text size="xs" c="teal" mb="xs" style={{ whiteSpace: 'pre-wrap' }}>
          {info}
        </Text>
      )}

      {!artifact ? (
        <Text size="sm" c="dimmed">
          No model yet. Run training or load a saved model.
        </Text>
      ) : (
        <Stack gap="xs">
          <Group gap="sm">
            <Badge variant="light">{artifact?.model?.algo ?? 'unknown'}</Badge>
            {artifact?.kind && <Badge color="gray" variant="light">{artifact.kind}</Badge>}
            {artifact?.n_splits ? <Badge color="grape" variant="light">{artifact.n_splits} splits</Badge> : null}
          </Group>

          <Text size="sm">
            <strong>Created:</strong>{' '}
            {artifact.created_at ? new Date(artifact.created_at).toLocaleString() : '—'}
          </Text>

          <Text size="sm">
            <strong>Metric:</strong>{' '}
            {artifact.metric_name ?? '—'}{' '}
            ({fmt(artifact.mean_score)} ± {fmt(artifact.std_score)})
          </Text>

          <Text size="sm">
            <strong>Data:</strong>{' '}
            train {fmt(artifact.n_samples_train, 0)}, test {fmt(artifact.n_samples_test, 0)}, features {fmt(artifact.n_features_in, 0)}
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
            <strong>Features:</strong> {artifact?.features?.method ?? 'none'}
          </Text>

          <Text size="sm">
            <strong>Split:</strong> {artifact?.split?.mode ?? '—'}
          </Text>

          <Divider my="xs" />

          <Text size="sm" fw={500}>Pipeline</Text>
          {Array.isArray(artifact.pipeline) && artifact.pipeline.length > 0 ? (
            <ul style={{ margin: 0, paddingInlineStart: 18 }}>
              {artifact.pipeline.map((s, i) => (
                <li key={`${s.name}-${i}`}>
                  <strong>{s.name}</strong>{' '}
                  <Text component="span" c="dimmed">
                    — {s.class_path || 'unknown'}
                  </Text>
                </li>
              ))}
            </ul>
          ) : (
            <Text size="sm" c="dimmed">No pipeline details.</Text>
          )}
        </Stack>
      )}
    </Card>
  );
}
