// src/components/DataSidebar.jsx
import React, { useState } from 'react';
import { Card, Stack, Text, TextInput, Button, Divider, Alert, Group, Badge, FileInput } from '@mantine/core';
import { useDataCtx } from '../state/DataContext.jsx';
import { inspectData } from '../api/data';
import { uploadFile } from '../api/files'; // NEW

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

  // local file selections for upload
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

  async function handleUploadAndInspect() {
    setErr(null);
    setLoading(true);
    try {
      // Upload X if provided
      let newXPath = xPath;
      if (xLocalFile) {
        const up = await uploadFile(xLocalFile);  // { path, original_name }
        newXPath = up.path;
        setXPath(newXPath);
        setNPZPath(null); // if we upload X/Y separately, clear npz
      }

      // Upload y if provided
      let newYPath = yPath;
      if (yLocalFile) {
        const up = await uploadFile(yLocalFile);
        newYPath = up.path;
        setYPath(newYPath);
        setNPZPath(null);
      }

      // Build inspect payload from whatever we have now
      const inspectPayload = {
        x_path: npzPath ? null : newXPath,
        y_path: npzPath ? null : newYPath,
        npz_path: npzPath || null,
        x_key: xKey || 'X',
        y_key: yKey || 'y',
      };

      const report = await inspectData(inspectPayload);
      setInspectReport(report);
    } catch (e) {
  const status = e?.response?.status;
  const msg = e?.response?.data?.detail || e.message || String(e);
  if (status === 404 && /files\/upload/.test(e?.config?.url || '')) {
    // Upload router not found — tell the user they can use manual paths in dev
    setErr("Upload API not found. In dev, either start the backend with the files router or use the manual X/y path fields above.");
  } else {
    setErr(msg);
  }
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

          {/* Manual container paths (if not uploading) */}
          <TextInput label="X path (.mat / .npz)" value={xPath} onChange={(e) => setXPath(e.currentTarget.value)} disabled={!!npzPath} />
          <TextInput label="y path (.mat / .npz)" value={yPath} onChange={(e) => setYPath(e.currentTarget.value)} disabled={!!npzPath} />
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
          <FileInput label="Upload X (.mat / .npz / .npy)" placeholder="Pick file (optional)" value={xLocalFile} onChange={setXLocalFile} accept=".mat,.npz,.npy" clearable />
          <FileInput label="Upload y (.mat / .npz / .npy)" placeholder="Pick file (optional)" value={yLocalFile} onChange={setYLocalFile} accept=".mat,.npz,.npy" clearable />
          <Group gap="xs">
            <Button size="xs" variant="light" onClick={handleUploadAndInspect} loading={loading}>Upload & Inspect</Button>
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
