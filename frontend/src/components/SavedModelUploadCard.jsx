// frontend/src/components/ModelLoadCard.jsx
import { useRef, useState } from 'react';
import { Card, Stack, Text, Button } from '@mantine/core';

import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { loadModel } from '../api/models';

export default function ModelLoadCard() {
  const fileInputRef = useRef(null);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState(null);
  const [err, setErr] = useState(null);

  async function onFileChosen(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setInfo(null);
    setErr(null);
    try {
      const { artifact: loadedMeta } = await loadModel(file);
      setArtifact(loadedMeta);
      setInfo(`Loaded model "${file.name}".`);
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
      event.target.value = '';
    }
  }

  function onClickLoad() {
    setInfo(null);
    setErr(null);
    fileInputRef.current?.click();
  }

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>Load a saved model</Text>
        <Text size="sm" c="dimmed">
          Load a previously saved model (.mend file) to inspect it and use it for results or predictions.
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

        <Button size="xs" onClick={onClickLoad} loading={loading}>
          Load model from file…
        </Button>

        <input
          type="file"
          accept=".mend"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={onFileChosen}
        />
      </Stack>
    </Card>
  );
}
