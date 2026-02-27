import { useMemo } from 'react';
import {
  Card,
  Stack,
  Text,
  Alert,
  Tooltip,
  Divider,
  SimpleGrid,
} from '@mantine/core';

import UnsupervisedTrainingResults from '../results/UnsupervisedTrainingResults.jsx';

import { fmt3, fmtCell } from '../../utils/format.js';
import { titleCaseFromKey, tooltipForKey } from '../../utils/labels.js';
import {
  buildClusterSummaryPairs,
  buildMetricPairs,
  buildModelDiagnosticsPairs,
  parseClusterSizes,
} from '../../utils/clusterSummary.js';

import KeyValuePairsTableSection from './KeyValuePairsTableSection.jsx';
import ClusterSizesTable from './ClusterSizesTable.jsx';

export default function UnsupervisedDiagnosticsCard({ trainResult }) {
  if (!trainResult) return null;

  const metrics = trainResult.metrics || {};
  const warnings = Array.isArray(trainResult.warnings) ? trainResult.warnings : [];
  const notes = Array.isArray(trainResult.notes) ? trainResult.notes : [];

  const clusterSummary = trainResult.cluster_summary || {};
  const modelDiag = trainResult?.diagnostics?.model_diagnostics || {};
  const embedding2d = trainResult?.diagnostics?.embedding_2d || null;

  const metricPairs = useMemo(() => buildMetricPairs(metrics), [metrics]);

  const clusterPairs = useMemo(
    () => buildClusterSummaryPairs(clusterSummary),
    [clusterSummary],
  );

  const modelDiagPairs = useMemo(
    () => buildModelDiagnosticsPairs(modelDiag, embedding2d),
    [modelDiag, embedding2d],
  );

  const clusterSizesRows = useMemo(() => {
    const rows = parseClusterSizes(clusterSummary?.cluster_sizes);
    return rows.sort((a, b) => Number(a.cluster_id) - Number(b.cluster_id));
  }, [clusterSummary]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={600} size="xl" ta="center">
          Unsupervised diagnostics
        </Text>

        {(warnings.length > 0 || notes.length > 0) && (
          <Stack gap="sm">
            {warnings.length > 0 && (
              <Alert color="yellow" title="Warnings">
                <Stack gap={4}>
                  {warnings.map((w, i) => (
                    <Text key={i} size="sm">
                      {w}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}

            {notes.length > 0 && (
              <Alert color="blue" title="Notes">
                <Stack gap={4}>
                  {notes.map((n, i) => (
                    <Text key={i} size="sm">
                      {n}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}
          </Stack>
        )}

        <Stack gap={0}>
          <Text fw={500} size="xl" ta="center">
            Summary of metrics
          </Text>

          {metricPairs.length === 0 ? (
            <Text size="sm" c="dimmed">
              No global metrics were returned.
            </Text>
          ) : (
            metricPairs.map(([k, v]) => {
              const tip = tooltipForKey(k);
              const label = titleCaseFromKey(k);
              return (
                <Text size="sm" key={k}>
                  {tip ? (
                    <Tooltip label={tip} multiline maw={360} withArrow>
                      <Text span fw={500} style={{ width: 'fit-content' }}>
                        {label}:{' '}
                      </Text>
                    </Tooltip>
                  ) : (
                    <Text span fw={500}>
                      {label}:{' '}
                    </Text>
                  )}
                  <Text span fw={700}>
                    {fmt3(v)}
                  </Text>
                </Text>
              );
            })
          )}
        </Stack>

        <Divider />

        <Stack gap="xs">
          <Text fw={500} size="xl" ta="center">
            Cluster summary &amp; diagnostics
          </Text>

          <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
            <KeyValuePairsTableSection
              title="Cluster summary"
              pairs={clusterPairs}
              emptyText="No cluster summary returned."
              rowKeyPrefix="cluster"
              renderValue={(v) => fmtCell(v)}
            />

            <KeyValuePairsTableSection
              title="Model diagnostics"
              pairs={modelDiagPairs}
              emptyText="No model diagnostics returned."
              rowKeyPrefix="diag"
              renderValue={(v) => fmtCell(v)}
            />

            <ClusterSizesTable rows={clusterSizesRows} />
          </SimpleGrid>

          <Divider my="sm" />

          <UnsupervisedTrainingResults trainResult={trainResult} />
        </Stack>
      </Stack>
    </Card>
  );
}
