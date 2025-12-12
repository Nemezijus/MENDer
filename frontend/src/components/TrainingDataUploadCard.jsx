import { useState } from 'react';
import {
  Card,
  Stack,
  Text,
  TextInput,
  Button,
  Divider,
  Alert,
  Group,
  Badge,
  FileInput,
  Select,
} from '@mantine/core';
import { useDataStore } from '../state/useDataStore.js';
import { useInspectDataMutation } from '../state/useInspectDataMutation.js';
import { uploadFile } from '../api/files.js';
import TrainingDataSummaryCard from './helpers/TrainingDataSummaryCard.jsx';

export default function TrainingDataUploadCard() {
  const xPath = useDataStore((s) => s.xPath);
  const setXPath = useDataStore((s) => s.setXPath);
  const yPath = useDataStore((s) => s.yPath);
  const setYPath = useDataStore((s) => s.setYPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const setNPZPath = useDataStore((s) => s.setNPZPath);
  const xKey = useDataStore((s) => s.xKey);
  const setXKey = useDataStore((s) => s.setXKey);
  const yKey = useDataStore((s) => s.yKey);
  const setYKey = useDataStore((s) => s.setYKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const setInspectReport = useDataStore((s) => s.setInspectReport);
  const taskSelected = useDataStore((s) => s.taskSelected);
  const setTaskSelected = useDataStore((s) => s.setTaskSelected);

  // derived values (used to live in DataContext)
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;
  const taskInferred = inspectReport?.task_inferred || null;
  const effectiveTask = taskSelected || taskInferred || null;

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // local file selections for upload (for showing filenames while component is mounted)
  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  const inspectMutation = useInspectDataMutation();

  function basename(path) {
    if (!path) return '';
    const parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1] || path;
  }

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
      const report = await inspectMutation.mutateAsync(payload);
      setInspectReport(report);
      // We still let the user explicitly override task; no auto-set here.
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
        const up = await uploadFile(xLocalFile); // { path, original_name }
        newXPath = up.path;
        setXPath(newXPath);
        setNPZPath(null); // clear npz when providing X/Y separately
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

      const report = await inspectMutation.mutateAsync(inspectPayload);
      setInspectReport(report);
    } catch (e) {
      const status = e?.response?.status;
      const msg = e?.response?.data?.detail || e.message || String(e);
      if (status === 404 && /files\/upload/.test(e?.config?.url || '')) {
        setErr(
          'Upload API not found. In dev, either start the backend with the files router or use the manual X/y path fields above.',
        );
      } else {
        setErr(msg);
      }
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

  // Filenames to show after navigating away & back
  const xUploadedName = xLocalFile?.name || basename(xPath);
  const yUploadedName = yLocalFile?.name || basename(yPath);

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="sm">
          <Group justify="space-between" align="center">
            <Text fw={600}>Training data</Text>
            {dataReady ? (
              <Badge color="green">Ready</Badge>
            ) : (
              <Badge color="gray">Not loaded</Badge>
            )}
          </Group>

          {/* Manual container paths (if not uploading) */}
          <TextInput
            label="Feature matrix (X) path (.mat / .npz)"
            value={xPath}
            onChange={(e) => setXPath(e.currentTarget.value)}
            disabled={!!npzPath}
          />
          <TextInput
            label="Label vector (y) path (.mat / .npz)"
            value={yPath}
            onChange={(e) => setYPath(e.currentTarget.value)}
            disabled={!!npzPath}
          />
          <TextInput
            label="npz path (X,y)"
            value={npzPath || ''}
            onChange={(e) => setNPZPath(e.currentTarget.value || null)}
          />

          <Group grow>
            <TextInput
              label="X key"
              value={xKey}
              onChange={(e) => setXKey(e.currentTarget.value)}
            />
            <TextInput
              label="y key"
              value={yKey}
              onChange={(e) => setYKey(e.currentTarget.value)}
            />
          </Group>

          {/* Task selector (inferred + override) */}
          <Select
            label="Task"
            data={[
              { value: 'classification', label: 'classification' },
              { value: 'regression', label: 'regression' },
            ]}
            value={taskSelected || taskInferred || null}
            placeholder={taskInferred ? `inferred: ${taskInferred}` : 'select'}
            onChange={(v) => setTaskSelected(v)}
            clearable
          />

          <Group gap="xs">
            <Button size="xs" onClick={handleInspect} loading={loading}>
              Inspect
            </Button>
          </Group>

          <Divider my="xs" label="or upload files" labelPosition="center" />

          {/* Upload area */}
          <FileInput
            label="Upload Feature Matrix (X) (.mat / .npz)"
            placeholder={xUploadedName || 'Pick file (optional)'}
            value={xLocalFile}
            onChange={setXLocalFile}
            accept=".mat,npz,npy"
            clearable
          />
          <FileInput
            label="Upload Label vector (y) (.mat / .npz)"
            placeholder={yUploadedName || 'Pick file (optional)'}
            value={yLocalFile}
            onChange={setYLocalFile}
            accept=".mat,npz,npy"
            clearable
          />
          <Group gap="xs">
            <Button
              size="xs"
              variant="light"
              onClick={handleUploadAndInspect}
              loading={loading}
            >
              Upload & Inspect
            </Button>
          </Group>

          {err && (
            <Alert color="red" variant="light" title="Error">
              <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                {err}
              </Text>
            </Alert>
          )}
        </Stack>
      </Card>

      <TrainingDataSummaryCard
        inspectReport={inspectReport}
        effectiveTask={effectiveTask}
      />
    </Stack>
  );
}
