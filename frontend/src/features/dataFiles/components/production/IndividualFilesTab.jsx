import { useEffect, useState } from 'react';
import { Stack, Button, Alert, Group, Text, Badge } from '@mantine/core';

import FilePathInput from '../common/FilePathInput.jsx';
import { useInspectProductionDataMutation } from '../../state/useInspectDataMutation.js';
import { useProductionDataStore } from '../../state/useProductionDataStore.js';
import { uploadFile } from '../../api/filesApi.js';
import { compactPayload } from '../../../../shared/utils/compactPayload.js';
import { toErrorText } from '../../../../shared/utils/errors.js';
import { formatDisplayNameFromUpload } from '../../utils/fileDisplay.js';
import { optNumber, optString } from '../../utils/optionalFields.js';

import { ProductionIndividualFilesText } from '../../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function ProductionIndividualFilesTab({
  acceptExts,
  setXDisplayGlobal,
  setYDisplayGlobal,
  modelArtifact,
  initialXPath,
  initialYPath,
  initialXDisplay,
  initialYDisplay,
}) {
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

  useEffect(() => {
    if (!xPathDisplay) {
      const v = (initialXDisplay || initialXPath || '').trim();
      if (v) {
        setXPathDisplay(v);
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

      const x_path = optString(resolvedXPath);
      const y_path = optString(resolvedYPath);
      const expected_n_features = optNumber(modelArtifact?.n_features_in);

      const payload = compactPayload({
        ...(x_path ? { x_path } : {}),
        ...(y_path ? { y_path } : {}),
        x_key: optString(xKey),
        y_key: optString(yKey),
        expected_n_features,
      });

      const report = await inspectMutation.mutateAsync(payload);

      setInspectReport(report);
      setXPath(resolvedXPath);
      setYPath(resolvedYPath);
      setNpzPath(null);
    } catch (e) {
      setErr(toErrorText(e));
      setInspectReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="sm">
      <ProductionIndividualFilesText />

      <FilePathInput
        label="Feature matrix (X)"
        acceptExts={acceptExts}
        value={xPathDisplay}
        onTextChange={(v, meta) => {
          const vv = v ?? '';
          setXPathDisplay(vv);

          // If the value came from the file picker (local://...), keep the File object.
          if (meta?.source === 'browse' || String(vv).startsWith('local://')) {
            setXBackendPath(null);
            return;
          }

          // Typed paths are treated as backend paths.
          setXBackendPath(vv.trim() || null);
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
        label="Label vector (y) (optional)"
        acceptExts={acceptExts}
        value={yPathDisplay}
        onTextChange={(v, meta) => {
          const vv = v ?? '';
          setYPathDisplay(vv);

          if (meta?.source === 'browse' || String(vv).startsWith('local://')) {
            setYBackendPath(null);
            return;
          }

          setYBackendPath(vv.trim() || null);
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
