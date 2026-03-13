import { useRef, useState } from 'react';
import { Card, Stack, Text, Button } from '@mantine/core';

import '../styles/artifactPanel.css';

import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { loadModel } from '../api/modelsApi.js';
import { toErrorText } from '../../../shared/utils/errors.js';

export default function SavedModelUploadCard() {
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
      setArtifact(loadedMeta, 'loaded');
      setInfo(`Loaded model "${file.name}".`);
    } catch (e) {
      setErr(toErrorText(e) || 'Load failed');
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
    <Card withBorder shadow="sm" padding="lg" className="specialCard">
      <Stack gap="sm">
        <Text fw={500}>Saved model</Text>
        <Text size="sm" c="dimmed">
          Load a previously saved model (.mend file) to inspect it and use it for results or predictions.
        </Text>

        {info && (
          <Text size="xs" c="teal" className="artifactPreWrap">
            {info}
          </Text>
        )}

        {err && (
          <Text size="xs" c="red" className="artifactPreWrap">
            {err}
          </Text>
        )}

        <Button size="xs" onClick={onClickLoad} loading={loading}>
          Load model from file…
        </Button>

        <input
          type="file"
          hidden
          accept=".mend"
          ref={fileInputRef}
          onChange={onFileChosen}
        />
      </Stack>
    </Card>
  );
}
