import { useState } from 'react';
import { Card, Stack, Text, Button, FileInput, Group, Alert } from '@mantine/core';

import { useProductionDataStore } from '../state/useProductionDataStore.js';
import { uploadFile } from '../api/files';

export default function ProductionDataUploadCard() {
  const {
    setXPath,
    setYPath,
    setNpzPath,
    reset: resetProductionData,
  } = useProductionDataStore();

  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState(null);

  // local file selections for showing filenames in the inputs
  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  async function handleUpload(file, target) {
    if (!file) return;
    setUploading(true);
    setErr(null);
    try {
      const res = await uploadFile(file);
      const npzPath = res?.npz_path || null;
      const path = res?.path || null;

      if (target === 'x') {
        if (npzPath) {
          setNpzPath(npzPath);
          setXPath('');
        } else if (path) {
          setXPath(path);
          setNpzPath('');
        }
      } else if (target === 'y') {
        if (path) {
          setYPath(path);
        }
      }
    } catch (e) {
      setErr(e?.message || 'Failed to upload production data');
    } finally {
      setUploading(false);
    }
  }

  function handleClear() {
    setXLocalFile(null);
    setYLocalFile(null);
    resetProductionData();
  }

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={600}>Production data</Text>
        <Text size="sm" c="dimmed">
          Upload feature matrix and optional labels for running predictions.
        </Text>

        <Stack grow gap="sm">
          <FileInput
            label="Production features (X)"
            placeholder="Pick file (optional)"
            value={xLocalFile}
            onChange={(file) => {
              setXLocalFile(file);
              handleUpload(file, 'x');
            }}
            disabled={uploading}
            accept=".mat,.npz,.npy,.csv,.txt"
            clearable
          />
          <FileInput
            label="Production labels (y, optional)"
            placeholder="Pick file (optional)"
            value={yLocalFile}
            onChange={(file) => {
              setYLocalFile(file);
              handleUpload(file, 'y');
            }}
            disabled={uploading}
            accept=".mat,.npz,.npy,.csv,.txt"
            clearable
          />
        </Stack>

        <Text size="xs" c="dimmed">
          X is required to run predictions. y is optional; if provided, an evaluation metric will be computed.
        </Text>

        {err && (
          <Alert color="red" variant="light">
            <Text size="xs">{err}</Text>
          </Alert>
        )}

        <Group justify="flex-end">
          <Button
            size="xs"
            variant="subtle"
            color="gray"
            onClick={handleClear}
            disabled={uploading}
          >
            Clear production data
          </Button>
        </Group>
      </Stack>
    </Card>
  );
}
