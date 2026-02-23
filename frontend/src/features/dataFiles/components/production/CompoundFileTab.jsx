import { useEffect, useState } from 'react';
import { Stack, Button, Alert, Group, Text, TextInput } from '@mantine/core';

import FilePathInput from '../common/FilePathInput.jsx';
import { useInspectProductionDataMutation } from '../../state/useInspectDataMutation.js';
import { useProductionDataStore } from '../../state/useProductionDataStore.js';
import { uploadFile } from '../../api/filesApi.js';
import { compactPayload } from '../../../../shared/utils/compactPayload.js';
import { toErrorText } from '../../../../shared/utils/errors.js';
import { formatDisplayNameFromUpload } from '../../utils/fileDisplay.js';

import { ProductionCompoundFileText } from '../../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function ProductionCompoundFileTab({
  acceptExts,
  defaultXKey,
  defaultYKey,
  setNpzDisplayGlobal,
  modelArtifact,
  initialNpzPath,
  initialNpzDisplay,
}) {
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

  useEffect(() => {
    if (!npzPathDisplay) {
      const v = (initialNpzDisplay || initialNpzPath || '').trim();
      if (v) {
        setNpzPathDisplay(v);
        setNpzBackendPath(initialNpzPath ? initialNpzPath : null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialNpzPath, initialNpzDisplay]);

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
      setErr(toErrorText(e));
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionCompoundFileText />

      <FilePathInput
        label="Compound dataset (.npz)"
        acceptExts={acceptExts}
        value={npzPathDisplay}
        onTextChange={(v) => {
          setNpzPathDisplay(v);
          setNpzBackendPath(v?.trim() || null);
          setNpzLocalFile(null);
          setNpzUploadInfo(null);
        }}
        onBrowseFile={(file) => {
          setNpzLocalFile(file);
          setNpzUploadInfo(null);
          setNpzBackendPath(null);
          if (!file) setNpzPathDisplay('');
        }}
        uploadInfo={npzUploadInfo}
      />

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
