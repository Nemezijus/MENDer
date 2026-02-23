import { useEffect, useState } from 'react';
import { Stack, Button, Alert, Group, Text } from '@mantine/core';

import FilePathInput from '../common/FilePathInput.jsx';
import { useInspectDataMutation } from '../../state/useInspectDataMutation.js';
import { uploadFile } from '../../api/filesApi.js';
import { compactPayload } from '../../../../shared/utils/compactPayload.js';
import { toErrorText } from '../../../../shared/utils/errors.js';
import { formatDisplayNameFromUpload } from '../../utils/fileDisplay.js';

import { TrainingIndividualFilesText } from '../../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function TrainingIndividualFilesTab({
  acceptExts,
  setInspectReportGlobal,
  setXPathGlobal,
  setYPathGlobal,
  setNpzPathGlobal,
  xKeyGlobal,
  yKeyGlobal,
  setXDisplayGlobal,
  setYDisplayGlobal,
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
        // Only set backendPath for raw paths; if it's a friendly "[hash] name" label, keep backendPath null.
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

      const payload = compactPayload({
        x_path: resolvedXPath,
        y_path: resolvedYPath,
        npz_path: null,
        x_key: xKeyGlobal?.trim() || undefined,
        y_key: yKeyGlobal?.trim() || undefined,
      });

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReportGlobal(report);

      // Keep store in sync with backend paths (for dev quick-start + for summary + for later runs)
      setXPathGlobal(resolvedXPath);
      setYPathGlobal(resolvedYPath);
      setNpzPathGlobal(null);
    } catch (e) {
      setErr(toErrorText(e));
      setInspectReportGlobal(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <TrainingIndividualFilesText />

      <FilePathInput
        label="Feature matrix (X)"
        acceptExts={acceptExts}
        value={xPathDisplay}
        onTextChange={(v) => {
          setXPathDisplay(v);
          // Treat typed values as backend paths (dev quick-start is this case)
          setXBackendPath(v?.trim() || null);
          // Clear file-upload state when user types
          setXLocalFile(null);
          setXUploadInfo(null);
        }}
        onBrowseFile={(file) => {
          setXLocalFile(file);
          setXUploadInfo(null);
          setXBackendPath(null);
          if (!file) setXPathDisplay('');
        }}
        uploadInfo={xUploadInfo}
      />

      <FilePathInput
        label="Label vector (y)"
        acceptExts={acceptExts}
        value={yPathDisplay}
        onTextChange={(v) => {
          setYPathDisplay(v);
          setYBackendPath(v?.trim() || null);
          setYLocalFile(null);
          setYUploadInfo(null);
        }}
        onBrowseFile={(file) => {
          setYLocalFile(file);
          setYUploadInfo(null);
          setYBackendPath(null);
          if (!file) setYPathDisplay('');
        }}
        uploadInfo={yUploadInfo}
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
