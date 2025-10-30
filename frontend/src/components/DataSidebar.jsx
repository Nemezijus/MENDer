// src/components/DataSidebar.jsx
import React, { useRef, useState } from 'react';
import { Card, Stack, Text, TextInput, Button, Divider, Alert, Group, Badge, FileInput } from '@mantine/core';
import { useDataCtx } from '../state/DataContext.jsx';
import { inspectData, uploadData } from '../api/data';

export default function DataSidebar() {
  const {
    xPath, setXPath,
    yPath, setYPath,
    npzPath, setNPZPath,
    xKey, setXKey,
    yKey, setYKey,
    inspectReport, setInspectReport,
    dataReady,
  } = useDataCtx();

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // file inputs
  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
      const payload = {
        x_path: npzPath ? null : xPath,
        y_path: npzPath ? null : yPath,
        npz_path: npzPath,
        x_key: xKey,
        y_key: yKey,
      };
      const report = await inspectData(payload);
      setInspectReport(report);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

async function handleUpload() {
  setErr(null);
  setLoading(true);
  try {
    const form = new FormData();
    if (xLocalFile) form.append('x_file', xLocalFile);
    if (yLocalFile) form.append('y_file', yLocalFile);

    const data = await uploadData(form);
    const saved = data?.saved || {};

    // update paths/keys from server
    setXPath(saved.x_path || '');
    setYPath(saved.y_path || '');
    setNPZPath(saved.npz_path || null);
    setXKey(saved.x_key || 'X');
    setYKey(saved.y_key || 'y');

    // now immediately inspect using those values
    const inspectPayload = {
      x_path: saved.npz_path ? null : (saved.x_path || ''),
      y_path: saved.npz_path ? null : (saved.y_path || ''),
      npz_path: saved.npz_path || null,
      x_key: saved.x_key || 'X',
      y_key: saved.y_key || 'y',
    };
    const report = await inspectData(inspectPayload);
    setInspectReport(report);
  } catch (e) {
    const msg = e?.response?.data?.detail || e.message || String(e);
    setErr(msg);
    setInspectReport(null);
  } finally {
    setLoading(false);
  }
}

  return (
    <Stack gap="md" w={320}>
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="sm">
          <Group justify="space-between" align="center">
            <Text fw={600}>Data</Text>
            {dataReady ? <Badge color="green">Ready</Badge> : <Badge color="gray">Not loaded</Badge>}
          </Group>

          {/* Local filesystem paths (defaults kept) */}
          <TextInput label="X path (.mat or .npz)" value={xPath} onChange={(e) => setXPath(e.currentTarget.value)} disabled={!!npzPath} />
          <TextInput label="y path (.mat or .npz)" value={yPath} onChange={(e) => setYPath(e.currentTarget.value)} disabled={!!npzPath} />
          <TextInput label="npz path (X,y)" value={npzPath || ''} onChange={(e) => setNPZPath(e.currentTarget.value || null)} />

          <Group grow>
            <TextInput label="X key" value={xKey} onChange={(e) => setXKey(e.currentTarget.value)} />
            <TextInput label="y key" value={yKey} onChange={(e) => setYKey(e.currentTarget.value)} />
          </Group>

          <Group gap="xs">
            <Button size="xs" onClick={handleInspect} loading={loading}>Inspect</Button>
          </Group>

          <Divider my="xs" label="or upload files" labelPosition="center" />

          {/* Upload area */}
          <FileInput label="Upload X (.mat or .npz)" placeholder="Pick file (optional)" value={xLocalFile} onChange={setXLocalFile} clearable />
          <FileInput label="Upload y (.mat or .npz)" placeholder="Pick file (optional)" value={yLocalFile} onChange={setYLocalFile} clearable />
          <Group gap="xs">
            <Button size="xs" variant="light" onClick={handleUpload} loading={loading}>Upload & Inspect</Button>
          </Group>

          {err && (
            <Alert color="red" variant="light" title="Error">
              <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{err}</Text>
            </Alert>
          )}
        </Stack>
      </Card>

      {/* Quick summary */}
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Text fw={600} mb="xs">Summary</Text>
        {!inspectReport && <Text size="sm" c="dimmed">Run Inspect to see data summary.</Text>}
        {inspectReport && (
          <Stack gap={2}>
            <Text size="sm">n_samples: {inspectReport.n_samples}</Text>
            <Text size="sm">n_features: {inspectReport.n_features}</Text>
            <Text size="sm">classes: {Array.isArray(inspectReport.classes) ? inspectReport.classes.join(', ') : String(inspectReport.classes)}</Text>
            <Text size="sm">missing total: {inspectReport.missingness?.total ?? 0}</Text>
            {inspectReport.suggestions?.recommend_pca && (
              <Alert color="blue" variant="light" mt="xs">
                <Text size="sm">Suggestion: {inspectReport.suggestions.reason || 'consider PCA'}</Text>
              </Alert>
            )}
          </Stack>
        )}
      </Card>
    </Stack>
  );
}
