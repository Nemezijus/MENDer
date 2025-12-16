import { useState } from 'react';
import {
  Card,
  Stack,
  Text,
  TextInput,
  Button,
  Group,
  Badge,
  Tabs,
  Box,
  Alert,
  FileButton,
} from '@mantine/core';

import { useProductionDataStore } from '../state/useProductionDataStore.js';
import { useInspectProductionDataMutation } from '../state/useInspectDataMutation.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';
import { uploadFile } from '../api/files';

import DataSummaryCard from './helpers/DataSummaryCard.jsx';

import {
  ProductionDataIntroText,
  ProductionIndividualFilesText,
  ProductionCompoundFileText,
} from './helpers/helpTexts/DataFilesHelpTexts.jsx';

// -------------------------
// Helpers
// -------------------------
function displayLocalFilePath(file) {
  if (!file) return '';
  const rel =
    file.webkitRelativePath && file.webkitRelativePath.length > 0
      ? file.webkitRelativePath
      : file.name;
  return `local://${rel}`;
}

// ============================================================
// Tab 1: Individual files
// ============================================================
function IndividualFilesTab({
  setXPath,
  setYPath,
  setNpzPath,
  onInspect,
  modelArtifact,
}) {
  const inspectMutation = useInspectProductionDataMutation();

  // local-only state (dies when tab unmounts)
  const [xPath, setXPathLocal] = useState('');
  const [yPath, setYPathLocal] = useState('');
  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState(null);

  async function handleUploadAndInspect() {
    setErr(null);
    setUploading(true);
    try {
      let resolvedX = xPath?.trim() || null;
      let resolvedY = yPath?.trim() || null;

      if (xLocalFile) {
        const res = await uploadFile(xLocalFile);
        const npzPathFromApi = res?.npz_path || null;
        const pathFromApi = res?.path || null;

        // Preserve existing behavior: if API returns an npz_path, treat as compound.
        if (npzPathFromApi) {
          setNpzPath(npzPathFromApi);
          setXPath('');
          setYPath('');
          onInspect?.(null);
          return; // switched to compound mode
        }

        if (pathFromApi) {
          resolvedX = pathFromApi;
          setXPathLocal(pathFromApi);
        }
      }

      if (yLocalFile) {
        const res = await uploadFile(yLocalFile);
        const pathFromApi = res?.path || null;
        if (pathFromApi) {
          resolvedY = pathFromApi;
          setYPathLocal(pathFromApi);
        }
      }

      // Individual mode clears compound
      setNpzPath('');
      setXPath(resolvedX || '');
      setYPath(resolvedY || '');

      // Inspect (labels optional)
      const payload = {
        x_path: resolvedX,
        y_path: resolvedY || null,
        npz_path: null,
        x_key: 'X',
        y_key: 'y',
        expected_n_features: modelArtifact?.n_features_in ?? null,
      };

      const report = await inspectMutation.mutateAsync(payload);
      onInspect?.(report);
    } catch (e) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        'Failed to upload/inspect production data';
      setErr(msg);
      onInspect?.(null);
    } finally {
      setUploading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionIndividualFilesText />

      <TextInput
        label="Production features (X)"
        placeholder="Paste file path"
        value={xPath}
        onChange={(e) => setXPathLocal(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setXLocalFile(file);
              if (file) setXPathLocal(displayLocalFilePath(file));
            }}
            accept=".mat,.npz,.npy,.csv,.txt"
          >
            {(props) => (
              <Button {...props} size="xs" variant="light">
                Browse
              </Button>
            )}
          </FileButton>
        }
        disabled={uploading}
      />

      <TextInput
        label="Production labels (y, optional)"
        placeholder="Paste file path"
        value={yPath}
        onChange={(e) => setYPathLocal(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setYLocalFile(file);
              if (file) setYPathLocal(displayLocalFilePath(file));
            }}
            accept=".mat,.npz,.npy,.csv,.txt"
          >
            {(props) => (
              <Button {...props} size="xs" variant="light">
                Browse
              </Button>
            )}
          </FileButton>
        }
        disabled={uploading}
      />

      <Group gap="xs">
        <Button size="xs" onClick={handleUploadAndInspect} loading={uploading}>
          Upload &amp; Inspect
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
  );
}

// ============================================================
// Tab 2: Compound file
// ============================================================
function CompoundFileTab({
  setXPath,
  setYPath,
  setNpzPath,
  onInspect,
  modelArtifact,
}) {
  const inspectMutation = useInspectProductionDataMutation();

  const [npzPath, setNpzPathLocal] = useState('');
  const [npzLocalFile, setNpzLocalFile] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState(null);

  async function handleUploadAndInspect() {
    setErr(null);
    setUploading(true);
    try {
      let resolved = npzPath?.trim() || null;

      if (npzLocalFile) {
        const res = await uploadFile(npzLocalFile);
        const npzPathFromApi = res?.npz_path || null;
        const pathFromApi = res?.path || null;

        // Preserve existing behavior: prefer npz_path if backend provides it.
        resolved = npzPathFromApi || pathFromApi || resolved;
        if (resolved) setNpzPathLocal(resolved);
      }

      // Compound mode clears separate X/Y
      setXPath('');
      setYPath('');
      setNpzPath(resolved || '');

      // Inspect (labels optional; depends on file contents / keys)
      const payload = {
        x_path: null,
        y_path: null,
        npz_path: resolved,
        x_key: 'X',
        y_key: 'y',
        expected_n_features: modelArtifact?.n_features_in ?? null,
      };

      const report = await inspectMutation.mutateAsync(payload);
      onInspect?.(report);
    } catch (e) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        'Failed to upload/inspect production data';
      setErr(msg);
      onInspect?.(null);
    } finally {
      setUploading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionCompoundFileText />

      <TextInput
        label="Production compound file (.npz)"
        placeholder="Paste file path"
        value={npzPath}
        onChange={(e) => setNpzPathLocal(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setNpzLocalFile(file);
              if (file) setNpzPathLocal(displayLocalFilePath(file));
            }}
            accept=".npz,.mat"
          >
            {(props) => (
              <Button {...props} size="xs" variant="light">
                Browse
              </Button>
            )}
          </FileButton>
        }
        disabled={uploading}
      />

      <Group gap="xs">
        <Button size="xs" onClick={handleUploadAndInspect} loading={uploading}>
          Upload &amp; Inspect
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
  );
}

// ============================================================
// Main component
// ============================================================
export default function ProductionDataUploadCard() {
  const {
    setXPath,
    setYPath,
    setNpzPath,
    xPath,
    yPath,
    npzPath,
    inspectReport,
    setInspectReport,
  } = useProductionDataStore();

  // Current model artifact (for compatibility warning in summary)
  const modelArtifact = useModelArtifactStore((s) =>
    s?.artifact || s?.activeArtifact || s?.modelArtifact || null
  );

  const ready = !!(npzPath || xPath);
  const effectiveTask = inspectReport?.task_inferred || null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          {/* Center title, badge on right */}
          <Group justify="space-between" align="center">
            <Box style={{ width: 90 }} />
            <Text fw={700} size="lg" align="center" style={{ flex: 1 }}>
              Production data
            </Text>
            {ready ? (
              <Badge color="green">Ready</Badge>
            ) : (
              <Badge color="gray">Not loaded</Badge>
            )}
          </Group>

          <ProductionDataIntroText />

          <Tabs defaultValue="individual" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="individual">Individual files</Tabs.Tab>
              <Tabs.Tab value="compound">Compound file</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="individual" pt="md">
              <IndividualFilesTab
                setXPath={setXPath}
                setYPath={setYPath}
                setNpzPath={setNpzPath}
                onInspect={setInspectReport}
                modelArtifact={modelArtifact}
              />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <CompoundFileTab
                setXPath={setXPath}
                setYPath={setYPath}
                setNpzPath={setNpzPath}
                onInspect={setInspectReport}
                modelArtifact={modelArtifact}
              />
            </Tabs.Panel>
          </Tabs>

          <Text size="xs" c="dimmed">
            X (or a compound file containing X) is required to run predictions. y is optional.
          </Text>
        </Stack>
      </Card>

      <DataSummaryCard
        inspectReport={inspectReport}
        effectiveTask={effectiveTask}
        xPath={xPath}
        yPath={yPath}
        npzPath={npzPath}
        showSuggestion={false}
        modelArtifact={modelArtifact}
      />
    </Stack>
  );
}
