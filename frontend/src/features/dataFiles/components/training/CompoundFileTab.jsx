import { useEffect, useState } from 'react';
import { Stack, Button, Alert, Group, Text, TextInput } from '@mantine/core';

import FilePathInput from '../common/FilePathInput.jsx';
import { useInspectDataMutation } from '../../state/useInspectDataMutation.js';
import { uploadFile } from '../../api/filesApi.js';
import { compactPayload } from '../../../../shared/utils/compactPayload.js';
import { toErrorText } from '../../../../shared/utils/errors.js';
import { formatDisplayNameFromUpload } from '../../utils/fileDisplay.js';

import { TrainingCompoundFileText } from '../../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function TrainingCompoundFileTab({
  acceptExts,
  defaultXKey,
  defaultYKey,
  setInspectReportGlobal,
  setXPathGlobal,
  setYPathGlobal,
  setNpzPathGlobal,
  xKey,
  yKey,
  setXKey,
  setYKey,
  setNpzDisplayGlobal,
  initialNpzPath,
  initialNpzDisplay,
}) {
  const inspectMutation = useInspectDataMutation();

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
      });

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);
      setNpzPathGlobal(resolvedNpzPath);
      setXPathGlobal(null);
      setYPathGlobal(null);
    } catch (e) {
      setErr(toErrorText(e));
      setInspectReportGlobal(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <TrainingCompoundFileText />

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
          label="y key (labels)"
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
