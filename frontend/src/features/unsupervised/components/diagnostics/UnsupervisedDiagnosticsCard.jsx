import { useMemo } from 'react';
import {
  Card,
  Stack,
  Text,
  Alert,
  Table,
  ScrollArea,
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

  const stickyThStyle = {
    position: 'sticky',
    top: 0,
    zIndex: 2,
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
    whiteSpace: 'nowrap',
  };

  const headerTextStyle = { whiteSpace: 'nowrap', lineHeight: 1.1 };

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
            <Stack gap="xs">
              <Text size="sm" fw={600}>
                Cluster summary
              </Text>
              <Table
                withTableBorder={false}
                withColumnBorders={false}
                horizontalSpacing="xs"
                verticalSpacing="xs"
                style={{ tableLayout: 'fixed' }}
              >
                <Table.Tbody>
                  {clusterPairs.length === 0 ? (
                    <Table.Tr>
                      <Table.Td colSpan={2}>
                        <Text size="sm" c="dimmed">
                          No cluster summary returned.
                        </Text>
                      </Table.Td>
                    </Table.Tr>
                  ) : (
                    clusterPairs.map(([k, val]) => {
                      const label = titleCaseFromKey(k);
                      const tip = tooltipForKey(k);
                      return (
                        <Table.Tr key={`cluster-${k}`}>
                          <Table.Td style={{ width: '45%', paddingLeft: 0 }}>
                            {tip ? (
                              <Tooltip label={tip} multiline maw={360} withArrow>
                                <Text size="sm" c="dimmed" style={{ width: 'fit-content' }}>
                                  {label}
                                </Text>
                              </Tooltip>
                            ) : (
                              <Text size="sm" c="dimmed">
                                {label}
                              </Text>
                            )}
                          </Table.Td>
                          <Table.Td style={{ width: '55%', paddingRight: 0 }}>
                            <Text
                              size="sm"
                              fw={700}
                              style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}
                            >
                              {fmtCell(val)}
                            </Text>
                          </Table.Td>
                        </Table.Tr>
                      );
                    })
                  )}
                </Table.Tbody>
              </Table>
            </Stack>

            <Stack gap="xs">
              <Text size="sm" fw={600}>
                Model diagnostics
              </Text>
              <Table
                withTableBorder={false}
                withColumnBorders={false}
                horizontalSpacing="xs"
                verticalSpacing="xs"
                style={{ tableLayout: 'fixed' }}
              >
                <Table.Tbody>
                  {modelDiagPairs.length === 0 ? (
                    <Table.Tr>
                      <Table.Td colSpan={2}>
                        <Text size="sm" c="dimmed">
                          No model diagnostics returned.
                        </Text>
                      </Table.Td>
                    </Table.Tr>
                  ) : (
                    modelDiagPairs.map(([k, val]) => {
                      const label = titleCaseFromKey(k);
                      const tip = tooltipForKey(k);
                      return (
                        <Table.Tr key={`diag-${k}`}>
                          <Table.Td style={{ width: '45%', paddingLeft: 0 }}>
                            {tip ? (
                              <Tooltip label={tip} multiline maw={360} withArrow>
                                <Text size="sm" c="dimmed" style={{ width: 'fit-content' }}>
                                  {label}
                                </Text>
                              </Tooltip>
                            ) : (
                              <Text size="sm" c="dimmed">
                                {label}
                              </Text>
                            )}
                          </Table.Td>
                          <Table.Td style={{ width: '55%', paddingRight: 0 }}>
                            <Text
                              size="sm"
                              fw={700}
                              style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}
                            >
                              {fmtCell(val)}
                            </Text>
                          </Table.Td>
                        </Table.Tr>
                      );
                    })
                  )}
                </Table.Tbody>
              </Table>
            </Stack>

            <Stack gap="xs">
              <Text size="sm" fw={600}>
                Cluster sizes
              </Text>

              <ScrollArea h={220} type="auto" offsetScrollbars>
                <Table
                  withTableBorder={false}
                  withColumnBorders={false}
                  horizontalSpacing="xs"
                  verticalSpacing="xs"
                >
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th style={stickyThStyle}>
                        <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                          Cluster id
                        </Text>
                      </Table.Th>
                      <Table.Th style={stickyThStyle}>
                        <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                          Size
                        </Text>
                      </Table.Th>
                    </Table.Tr>
                  </Table.Thead>

                  <Table.Tbody>
                    {clusterSizesRows.length === 0 ? (
                      <Table.Tr>
                        <Table.Td colSpan={2}>
                          <Text size="sm" c="dimmed">
                            —
                          </Text>
                        </Table.Td>
                      </Table.Tr>
                    ) : (
                      clusterSizesRows.map((r, i) => {
                        const isStriped = i % 2 === 1;
                        return (
                          <Table.Tr
                            key={i}
                            style={{
                              backgroundColor: isStriped
                                ? 'var(--mantine-color-gray-1)'
                                : 'white',
                            }}
                          >
                            <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                              {fmtCell(r.cluster_id)}
                            </Table.Td>
                            <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                              {fmtCell(r.size)}
                            </Table.Td>
                          </Table.Tr>
                        );
                      })
                    )}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Stack>
          </SimpleGrid>

          <Divider my="sm" />

          <UnsupervisedTrainingResults trainResult={trainResult} />
        </Stack>
      </Stack>
    </Card>
  );
}
