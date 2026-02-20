import { useState } from 'react';
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
  FileButton,
} from '@mantine/core';

import { uploadFile } from '../api/files';
import { useFilesConstraintsQuery } from '../state/useFilesConstraintsQuery.js';
import { compactPayload } from '../utils/compactPayload.js';
import { useInspectProductionDataMutation } from '../state/useInspectDataMutation.js';
import { useProductionDataStore } from '../state/useProductionDataStore.js';
import { useModelArtifactStore } from '../state/useModelArtifactStore.js';

import DataSummaryCard from './helpers/DataSummaryCard.jsx';
import {
  ProductionDataIntroText,
  ProductionIndividualFilesText,
  ProductionCompoundFileText,
} from './helpers/helpTexts/DataFilesHelpTexts.jsx';

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

function IndividualFilesTab({ acceptExts, setXDisplayGlobal, setYDisplayGlobal, modelArtifact }) {
  const inspectMutation = useInspectProductionDataMutation();

  const setInspectReport = useProductionDataStore((s) => s.setInspectReport);
  const setXPath = useProductionDataStore((s) => s.setXPath);
  const setYPath = useProductionDataStore((s) => s.setYPath);
  const setNpzPath = useProductionDataStore((s) => s.setNpzPath);

  const xKey = useProductionDataStore((s) => s.xKey);
  const yKey = useProductionDataStore((s) => s.yKey);

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

  async function handleInspect() {
    setErr(null);
    setLoading(true);
    try {
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

      const payload = compactPayload({
        x_path: resolvedXPath,
        y_path: resolvedYPath,
        npz_path: null,
        x_key: xKey?.trim() || undefined,
        y_key: yKey?.trim() || undefined,
        expected_n_features: modelArtifact?.n_features_in ?? null,
      });

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReport(report);
      setXPath(resolvedXPath);
      setYPath(resolvedYPath);
      setNpzPath(null);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionIndividualFilesText />

      <Stack gap={4}>
        <TextInput
          label="Feature matrix (X)"
          placeholder="Paste file path"
          value={xPathDisplay}
          onChange={(e) => {
            const v = e.currentTarget.value;
            setXPathDisplay(v);
            setXBackendPath(v?.trim() || null);
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
              accept={acceptExts}
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
          label="Label vector (y) (optional)"
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
              accept={acceptExts}
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
        <Badge color="gray">Labels optional</Badge>
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

function CompoundFileTab({ acceptExts, defaultXKey, defaultYKey, setNpzDisplayGlobal, modelArtifact }) {
  const inspectMutation = useInspectProductionDataMutation();

  const setInspectReport = useProductionDataStore((s) => s.setInspectReport);
  const setXPath = useProductionDataStore((s) => s.setXPath);
  const setYPath = useProductionDataStore((s) => s.setYPath);
  const setNpzPath = useProductionDataStore((s) => s.setNpzPath);

  const xKey = useProductionDataStore((s) => s.xKey);
  const yKey = useProductionDataStore((s) => s.yKey);
  const setXKey = useProductionDataStore((s) => s.setXKey);
  const setYKey = useProductionDataStore((s) => s.setYKey);

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

      const payload = compactPayload({
        x_path: null,
        y_path: null,
        npz_path: resolvedNpzPath,
        x_key: xKey?.trim() || undefined,
        y_key: yKey?.trim() || undefined,
        expected_n_features: modelArtifact?.n_features_in ?? null,
      });

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReport(report);
      setNpzPath(resolvedNpzPath);
      setXPath(null);
      setYPath(null);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message || String(e);
      setErr(msg);
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionCompoundFileText />

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
              accept={acceptExts}
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
          placeholder={defaultXKey}
        />
        <TextInput
          label="y key (labels) (optional)"
          value={yKey || ''}
          onChange={(e) => setYKey(e.currentTarget.value)}
          placeholder={defaultYKey}
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

export default function ProductionDataUploadCard() {
  const { data: filesConstraints } = useFilesConstraintsQuery();
  const acceptExts = Array.isArray(filesConstraints?.allowed_exts)
    ? filesConstraints.allowed_exts.join(',')
    : undefined;
  const defaultXKey = filesConstraints?.data_default_keys?.x_key ?? 'X';
  const defaultYKey = filesConstraints?.data_default_keys?.y_key ?? 'y';

  const inspectReport = useProductionDataStore((s) => s.inspectReport);

  const xPath = useProductionDataStore((s) => s.xPath);
  const yPath = useProductionDataStore((s) => s.yPath);
  const npzPath = useProductionDataStore((s) => s.npzPath);

  const modelArtifact = useModelArtifactStore((s) =>
  s?.artifact || s?.activeArtifact || s?.modelArtifact || null
);

  // Task shown can be inferred from the inspected production data
  const effectiveTask = inspectReport?.task_inferred || null;

  // NEW persisted display
  const xDisplay = useProductionDataStore((s) => s.xDisplay);
  const yDisplay = useProductionDataStore((s) => s.yDisplay);
  const npzDisplay = useProductionDataStore((s) => s.npzDisplay);
  const setXDisplay = useProductionDataStore((s) => s.setXDisplay);
  const setYDisplay = useProductionDataStore((s) => s.setYDisplay);
  const setNpzDisplay = useProductionDataStore((s) => s.setNpzDisplay);

  return (
    <Stack gap="md">
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Box style={{ width: 90 }} />
            <Text fw={700} size="lg" align="center" style={{ flex: 1 }}>
              Production data
            </Text>
            {inspectReport ? <Badge color="green">Ready</Badge> : <Badge color="gray">Not loaded</Badge>}
          </Group>

          <ProductionDataIntroText />

          <Tabs defaultValue="individual" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="individual">Individual files</Tabs.Tab>
              <Tabs.Tab value="compound">Compound file</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="individual" pt="md">
              <IndividualFilesTab acceptExts={acceptExts} setXDisplayGlobal={setXDisplay} setYDisplayGlobal={setYDisplay} modelArtifact={modelArtifact} />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <CompoundFileTab acceptExts={acceptExts} defaultXKey={defaultXKey} defaultYKey={defaultYKey} setNpzDisplayGlobal={setNpzDisplay} modelArtifact={modelArtifact} />
            </Tabs.Panel>
          </Tabs>
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
        modelArtifact={modelArtifact}
        showSuggestion={false}
      />
    </Stack>
  );
}
