import { useEffect, useState } from 'react';
import {
  Card,
  Stack,
  Text,
  TextInput,
  Button,
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
  const rel =
    file.webkitRelativePath && file.webkitRelativePath.length > 0
      ? file.webkitRelativePath
      : file.name;
  return `local://${rel}`;
}

function shortHashFromSavedName(savedName, n = 7) {
  if (!savedName) return '';
  const dot = savedName.lastIndexOf('.');
  const base = dot >= 0 ? savedName.slice(0, dot) : savedName;
  return base.slice(0, n);
}

function formatDisplayNameFromUpload(up) {
  if (!up) return '';
  const sh = shortHashFromSavedName(up.saved_name, 7);
  const name = up.original_name || up.saved_name;
  return sh ? `[${sh}] ${name}` : name;
}

function StoredAsLine({ uploadInfo }) {
  if (!uploadInfo?.saved_name) return null;
  const sh = shortHashFromSavedName(uploadInfo.saved_name, 7);
  return (
    <Text size="xs" c="dimmed">
      Stored as: [{sh}] {uploadInfo.saved_name}
    </Text>
  );
}

// ============================================================
// Tab 1: Individual files
// ============================================================
function IndividualFilesTab({
  setInspectReportGlobal,
  setXPathGlobal,
  setYPathGlobal,
  setNpzPathGlobal,
  xKeyGlobal,
  yKeyGlobal,
  setXDisplayGlobal,
  setYDisplayGlobal,

  // NEW: initial values from store (dev quick-start)
  initialXPath,
  initialYPath,
  initialXDisplay,
  initialYDisplay,
}) {
  const inspectMutation = useInspectDataMutation();

  const [xPathDisplay, setXPathDisplay] = useState('');
  const [yPathDisplay, setYPathDisplay] = useState('');
  const [xBackendPath, setXBackendPath] = useState(null);
  const [yBackendPath, setYBackendPath] = useState(null);

  const [xLocalFile, setXLocalFile] = useState(null);
  const [yLocalFile, setYLocalFile] = useState(null);

  const [xUploadInfo, setXUploadInfo] = useState(null);
  const [yUploadInfo, setYUploadInfo] = useState(null);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // Autofill once from store:
  // - prefer friendly display label if present
  // - otherwise use the raw default path (dev quick-start)
  useEffect(() => {
    if (!xPathDisplay) {
      const v = (initialXDisplay || initialXPath || '').trim();
      if (v) {
        setXPathDisplay(v);
        // only set backendPath for raw paths; if it's a friendly "[hash] name" label, keep backendPath null
        setXBackendPath(initialXPath ? initialXPath : null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialXPath, initialXDisplay]);

  useEffect(() => {
    if (!yPathDisplay) {
      const v = (initialYDisplay || initialYPath || '').trim();
      if (v) {
        setYPathDisplay(v);
        setYBackendPath(initialYPath ? initialYPath : null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialYPath, initialYDisplay]);

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
      // If user typed a path, we rely on xBackendPath/yBackendPath.
      // If they browsed a file, we upload and use returned path.
      let resolvedXPath = xBackendPath || xPathDisplay?.trim() || null;
      let resolvedYPath = yBackendPath || yPathDisplay?.trim() || null;

      if (xLocalFile) {
        const up = await uploadFile(xLocalFile);
        resolvedXPath = up.path;
        setXBackendPath(up.path);
        setXUploadInfo(up);

        const disp = formatDisplayNameFromUpload(up);
        setXPathDisplay(disp);
        setXDisplayGlobal(disp);
      } else {
        // If they typed a path, keep the display as-is (path), and persist it as display too.
        setXDisplayGlobal(xPathDisplay?.trim() || '');
      }

      if (yLocalFile) {
        const up = await uploadFile(yLocalFile);
        resolvedYPath = up.path;
        setYBackendPath(up.path);
        setYUploadInfo(up);

        const disp = formatDisplayNameFromUpload(up);
        setYPathDisplay(disp);
        setYDisplayGlobal(disp);
      } else {
        setYDisplayGlobal(yPathDisplay?.trim() || '');
      }

      const payload = {
        x_path: resolvedXPath,
        y_path: resolvedYPath,
        npz_path: null,
        x_key: xKeyGlobal || 'X',
        y_key: yKeyGlobal || 'y',
      };

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);

      // Keep store in sync with backend paths (for dev quick-start + for summary + for later runs)
      setXPathGlobal(resolvedXPath);
      setYPathGlobal(resolvedYPath);
      setNpzPathGlobal(null);
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

      <Stack gap={4}>
        <TextInput
          label="Feature matrix (X)"
          placeholder="Paste file path"
          value={xPathDisplay}
          onChange={(e) => {
            const v = e.currentTarget.value;
            setXPathDisplay(v);

            // Treat typed values as backend paths (dev quick-start is this case)
            setXBackendPath(v?.trim() || null);

            // Clear file-upload state when user types
            setXLocalFile(null);
            setXUploadInfo(null);
          }}
          rightSectionWidth={86}
          rightSection={
            <FileButton
              onChange={(file) => {
                setXLocalFile(file);
                setXUploadInfo(null);
                setXBackendPath(null);
                if (file) setXPathDisplay(displayLocalFilePath(file));
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
        <StoredAsLine uploadInfo={xUploadInfo} />
      </Stack>

      <Stack gap={4}>
        <TextInput
          label="Label vector (y)"
          placeholder="Paste file path"
          value={yPathDisplay}
          onChange={(e) => {
            const v = e.currentTarget.value;
            setYPathDisplay(v);
            setYBackendPath(v?.trim() || null);
            setYLocalFile(null);
            setYUploadInfo(null);
          }}
          rightSectionWidth={86}
          rightSection={
            <FileButton
              onChange={(file) => {
                setYLocalFile(file);
                setYUploadInfo(null);
                setYBackendPath(null);
                if (file) setYPathDisplay(displayLocalFilePath(file));
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
        <StoredAsLine uploadInfo={yUploadInfo} />
      </Stack>

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
  setNpzPathGlobal,
  xKey,
  yKey,
  setXKey,
  setYKey,
  setNpzDisplayGlobal,
}) {
  const inspectMutation = useInspectDataMutation();

  const [npzPathDisplay, setNpzPathDisplay] = useState('');
  const [npzBackendPath, setNpzBackendPath] = useState(null);
  const [npzLocalFile, setNpzLocalFile] = useState(null);
  const [npzUploadInfo, setNpzUploadInfo] = useState(null);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
      let resolvedNpzPath = npzBackendPath || npzPathDisplay?.trim() || null;

      if (npzLocalFile) {
        const up = await uploadFile(npzLocalFile);
        resolvedNpzPath = up.path;
        setNpzBackendPath(up.path);
        setNpzUploadInfo(up);

        const disp = formatDisplayNameFromUpload(up);
        setNpzPathDisplay(disp);
        setNpzDisplayGlobal(disp);
      } else {
        setNpzDisplayGlobal(npzPathDisplay?.trim() || '');
      }

      const payload = {
        x_path: null,
        y_path: null,
        npz_path: resolvedNpzPath,
        x_key: xKey || 'X',
        y_key: yKey || 'y',
      };

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);
      setNpzPathGlobal(resolvedNpzPath);
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

      <Stack gap={4}>
        <TextInput
          label="Compound dataset (.npz)"
          placeholder="Paste file path"
          value={npzPathDisplay}
          onChange={(e) => {
            const v = e.currentTarget.value;
            setNpzPathDisplay(v);
            setNpzBackendPath(v?.trim() || null);
            setNpzLocalFile(null);
            setNpzUploadInfo(null);
          }}
          rightSectionWidth={86}
          rightSection={
            <FileButton
              onChange={(file) => {
                setNpzLocalFile(file);
                setNpzUploadInfo(null);
                setNpzBackendPath(null);
                if (file) setNpzPathDisplay(displayLocalFilePath(file));
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
        <StoredAsLine uploadInfo={npzUploadInfo} />
      </Stack>

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
// Main card
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
  const setNpzPath = useDataStore((s) => s.setNpzPath);

  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);

  // persisted display
  const xDisplay = useDataStore((s) => s.xDisplay);
  const yDisplay = useDataStore((s) => s.yDisplay);
  const npzDisplay = useDataStore((s) => s.npzDisplay);
  const setXDisplay = useDataStore((s) => s.setXDisplay);
  const setYDisplay = useDataStore((s) => s.setYDisplay);
  const setNpzDisplay = useDataStore((s) => s.setNpzDisplay);

  const dataReady = !!inspectReport && (inspectReport?.n_samples ?? 0) > 0;
  const taskInferred = inspectReport?.task_inferred || null;
  const effectiveTask = taskSelected || taskInferred || null;

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Box style={{ width: 90 }} />
            <Text fw={700} size="lg" align="center" style={{ flex: 1 }}>
              Training data
            </Text>
            {dataReady ? (
              <Badge color="green">Ready</Badge>
            ) : (
              <Badge color="gray">Not loaded</Badge>
            )}
          </Group>

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
                setNpzPathGlobal={setNpzPath}
                xKeyGlobal={xKey}
                yKeyGlobal={yKey}
                setXDisplayGlobal={setXDisplay}
                setYDisplayGlobal={setYDisplay}
                initialXPath={xPath}
                initialYPath={yPath}
                initialXDisplay={xDisplay}
                initialYDisplay={yDisplay}
              />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <CompoundFileTab
                setInspectReportGlobal={setInspectReport}
                setXPathGlobal={setXPath}
                setYPathGlobal={setYPath}
                setNpzPathGlobal={setNpzPath}
                xKey={xKey}
                yKey={yKey}
                setXKey={setXKey}
                setYKey={setYKey}
                setNpzDisplayGlobal={setNpzDisplay}
              />
            </Tabs.Panel>
          </Tabs>

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

      <DataSummaryCard
        inspectReport={inspectReport}
        effectiveTask={effectiveTask}
        xPath={xPath}
        yPath={yPath}
        npzPath={npzPath}
        xDisplay={xDisplay}
        yDisplay={yDisplay}
        npzDisplay={npzDisplay}
      />
    </Stack>
  );
}
