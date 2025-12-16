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
  Tabs,
  Box,
  Select,
  FileButton,
} from '@mantine/core';

import { useDataStore } from '../state/useDataStore.js';
import { useInspectDataMutation } from '../state/useInspectDataMutation.js';
import { uploadFile } from '../api/files';

import DataSummaryCard from './helpers/DataSummaryCard.jsx';
import {
  TrainingDataIntroText,
  TrainingIndividualFilesText,
  TrainingCompoundFileText,
} from './helpers/helpTexts/DataFilesHelpTexts.jsx';

// -------------------------
// Helpers
// -------------------------
function displayLocalFilePath(file) {
  if (!file) return '';
  // Browsers do NOT expose real absolute local file paths for security.
  // Use a friendly pseudo-path; if directory upload is used, webkitRelativePath may exist.
  const rel =
    file.webkitRelativePath && file.webkitRelativePath.length > 0
      ? file.webkitRelativePath
      : file.name;
  return `local://${rel}`;
}

// ============================================================
// Tab 1: Individual files (features & labels)
// ============================================================
function IndividualFilesTab({
  setInspectReportGlobal,
  setXPathGlobal,
  setYPathGlobal,
  setNPZPathGlobal,
  xKeyGlobal,
  yKeyGlobal,
}) {
  const inspectMutation = useInspectDataMutation();

  // Local-only state (dies when tab unmounts)
  const [xPath, setXPath] = useState('');
  const [yPath, setYPath] = useState('');
  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
      // Upload if local files selected; otherwise use typed paths
      let resolvedXPath = xPath?.trim() || null;
      let resolvedYPath = yPath?.trim() || null;

      if (xLocalFile) {
        const up = await uploadFile(xLocalFile); // { path, original_name }
        resolvedXPath = up.path;
        setXPath(up.path); // replace pseudo-path with server path
      }

      if (yLocalFile) {
        const up = await uploadFile(yLocalFile);
        resolvedYPath = up.path;
        setYPath(up.path);
      }

      const payload = {
        x_path: resolvedXPath,
        y_path: resolvedYPath,
        npz_path: null,
        // keys are ignored for separate-file loads but we keep them consistent
        x_key: xKeyGlobal || 'X',
        y_key: yKeyGlobal || 'y',
      };

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);
      setXPathGlobal(resolvedXPath);
      setYPathGlobal(resolvedYPath);
      setNPZPathGlobal(null);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setInspectReportGlobal(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <TrainingIndividualFilesText />

      <TextInput
        label="Feature matrix (X)"
        placeholder="Paste file path"
        value={xPath}
        onChange={(e) => setXPath(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setXLocalFile(file);
              if (file) setXPath(displayLocalFilePath(file));
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
      />

      <TextInput
        label="Label vector (y)"
        placeholder="Paste file path"
        value={yPath}
        onChange={(e) => setYPath(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setYLocalFile(file);
              if (file) setYPath(displayLocalFilePath(file));
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
      />

      <Group gap="xs">
        <Button size="xs" onClick={handleInspect} loading={loading}>
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
  setInspectReportGlobal,
  setXPathGlobal,
  setYPathGlobal,
  setNPZPathGlobal,
  xKey,
  yKey,
  setXKey,
  setYKey,
}) {
  const inspectMutation = useInspectDataMutation();

  const [npzPath, setNPZPath] = useState('');
  const [npzLocalFile, setNPZLocalFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
      let resolvedNPZPath = npzPath?.trim() || null;

      if (npzLocalFile) {
        const up = await uploadFile(npzLocalFile);
        resolvedNPZPath = up.path;
        setNPZPath(up.path);
      }

      const payload = {
        x_path: null,
        y_path: null,
        npz_path: resolvedNPZPath,
        x_key: xKey || 'X',
        y_key: yKey || 'y',
      };

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);
      setNPZPathGlobal(resolvedNPZPath);
      setXPathGlobal(null);
      setYPathGlobal(null);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setInspectReportGlobal(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <TrainingCompoundFileText />

      <TextInput
        label="Compound dataset (.npz)"
        placeholder="Paste file path"
        value={npzPath}
        onChange={(e) => setNPZPath(e.currentTarget.value)}
        rightSectionWidth={86}
        rightSection={
          <FileButton
            onChange={(file) => {
              setNPZLocalFile(file);
              if (file) setNPZPath(displayLocalFilePath(file));
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
      />

      <Group grow align="flex-start">
        <TextInput
          label="X key (features)"
          value={xKey || ''}
          onChange={(e) => setXKey(e.currentTarget.value)}
          placeholder="X"
        />
        <TextInput
          label="y key (labels)"
          value={yKey || ''}
          onChange={(e) => setYKey(e.currentTarget.value)}
          placeholder="y"
        />
      </Group>

      <Group gap="xs">
        <Button size="xs" onClick={handleInspect} loading={loading}>
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
// Main card: TrainingDataUploadCard
// ============================================================
export default function TrainingDataUploadCard() {
  const inspectReport = useDataStore((s) => s.inspectReport);
  const setInspectReport = useDataStore((s) => s.setInspectReport);

  const taskSelected = useDataStore((s) => s.taskSelected);
  const setTaskSelected = useDataStore((s) => s.setTaskSelected);

  const xKey = useDataStore((s) => s.xKey);
  const setXKey = useDataStore((s) => s.setXKey);
  const yKey = useDataStore((s) => s.yKey);
  const setYKey = useDataStore((s) => s.setYKey);

  const setXPath = useDataStore((s) => s.setXPath);
  const setYPath = useDataStore((s) => s.setYPath);
  const setNPZPath = useDataStore((s) => s.setNPZPath);

  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);

  const dataReady = !!inspectReport && (inspectReport?.n_samples ?? 0) > 0;
  const taskInferred = inspectReport?.task_inferred || null;
  const effectiveTask = taskSelected || taskInferred || null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          {/* Center title, badge on right */}
          <Group justify="space-between" align="center">
            <Box style={{ width: 90 }} /> {/* left spacer */}
            <Text fw={700} size="lg" align="center" style={{ flex: 1 }}>
              Training data
            </Text>
            {dataReady ? (
              <Badge color="green">Ready</Badge>
            ) : (
              <Badge color="gray">Not loaded</Badge>
            )}
          </Group>

          {/* Intro text */}
          <TrainingDataIntroText />

          <Tabs defaultValue="individual" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="individual">Individual files</Tabs.Tab>
              <Tabs.Tab value="compound">Compound file</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="individual" pt="md">
              <IndividualFilesTab
                setInspectReportGlobal={setInspectReport}
                setXPathGlobal={setXPath}
                setYPathGlobal={setYPath}
                setNPZPathGlobal={setNPZPath}
                xKeyGlobal={xKey}
                yKeyGlobal={yKey}
              />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <CompoundFileTab
                setInspectReportGlobal={setInspectReport}
                setXPathGlobal={setXPath}
                setYPathGlobal={setYPath}
                setNPZPathGlobal={setNPZPath}
                xKey={xKey}
                yKey={yKey}
                setXKey={setXKey}
                setYKey={setYKey}
              />
            </Tabs.Panel>
          </Tabs>

          {/* Task suggestion under tabs */}
          <Select
            label="Task (suggestion)"
            description="Overrides inferred task. You can leave it empty."
            data={[
              { value: 'classification', label: 'classification' },
              { value: 'regression', label: 'regression' },
            ]}
            value={taskSelected || null}
            placeholder={taskInferred ? `inferred: ${taskInferred}` : 'leave empty'}
            onChange={(v) => setTaskSelected(v)}
            clearable
          />
        </Stack>
      </Card>

      {/* Summary underneath */}
      <DataSummaryCard
        inspectReport={inspectReport}
        effectiveTask={effectiveTask}
        xPath={xPath}
        yPath={yPath}
        npzPath={npzPath}
      />
    </Stack>
  );
}
