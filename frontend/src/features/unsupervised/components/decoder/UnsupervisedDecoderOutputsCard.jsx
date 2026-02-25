import { useMemo, useState } from 'react';
import { Card, Stack, Group, Text, Tooltip, Button, Divider, Alert } from '@mantine/core';

import { exportDecoderOutputs } from '../../../modelArtifacts/api/modelsApi.js';
import { downloadBlob } from '../../../../shared/utils/download.js';
import { toErrorText } from '../../../../shared/utils/errors.js';

import UnsupervisedTrainingDecoderResults from '../results/UnsupervisedTrainingDecoderResults.jsx';
import DecoderPreviewTable from './DecoderPreviewTable.jsx';

import { pickColumns } from '../../utils/previewTable.js';

export default function UnsupervisedDecoderOutputsCard({ trainResult }) {
  if (!trainResult) return null;

  const artifactUid = trainResult?.artifact?.uid || null;
  const previewRows = trainResult.unsupervised_outputs?.preview_rows || [];
  const nTotal = trainResult.unsupervised_outputs?.n_rows_total ?? null;

  const previewColumns = useMemo(() => pickColumns(previewRows), [previewRows]);

  const [exportErr, setExportErr] = useState(null);

  const handleExport = async () => {
    if (!artifactUid) return;
    setExportErr(null);
    try {
      const { blob, filename } = await exportDecoderOutputs({
        artifactUid,
        filename: `decoder_outputs_${artifactUid}.csv`,
      });
      downloadBlob(blob, filename);
    } catch (e) {
      setExportErr(toErrorText(e));
    }
  };

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500} size="xl" ta="center">
          Decoder outputs
        </Text>

        <Stack gap={6}>
          <Text size="sm" c="dimmed">
            Per-sample decoder outputs on the evaluation set.
          </Text>

          <Group gap="md" wrap="wrap">
            <Tooltip
              label="Whether a cluster id is available for each sample."
              multiline
              maw={320}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Cluster id:{' '}
                </Text>
                <Text span fw={700}>
                  {previewColumns.includes('cluster_id') ? 'Available' : 'Not available'}
                </Text>
              </Text>
            </Tooltip>

            <Tooltip
              label="Whether a noise indicator exists (e.g., DBSCAN)."
              multiline
              maw={320}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Noise flag:{' '}
                </Text>
                <Text span fw={700}>
                  {previewColumns.includes('is_noise') ? 'Available' : 'Not available'}
                </Text>
              </Text>
            </Tooltip>

            <Tooltip
              label="Number of rows rendered in the table. Preview may be capped for performance."
              multiline
              maw={320}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Previewed samples:{' '}
                </Text>
                <Text span fw={700}>
                  {nTotal != null ? `${previewRows.length} / ${nTotal}` : `${previewRows.length}`}
                </Text>
              </Text>
            </Tooltip>
          </Group>
        </Stack>

        <Divider />

        <UnsupervisedTrainingDecoderResults trainResult={trainResult} />

        <Group justify="flex-end" align="center" wrap="wrap">
          <Button size="xs" variant="light" onClick={handleExport} disabled={!artifactUid}>
            Export to CSV
          </Button>
        </Group>

        {exportErr ? (
          <Alert color="red" title="Export failed">
            <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
              {exportErr}
            </Text>
          </Alert>
        ) : null}

        <DecoderPreviewTable rows={previewRows} columns={previewColumns} height={360} />
      </Stack>
    </Card>
  );
}
