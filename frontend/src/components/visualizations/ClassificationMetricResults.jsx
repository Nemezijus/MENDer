import { Card, Stack, Text, Table } from '@mantine/core';

export default function ClassificationMetricResults({
  confusion,
  metricName,
}) {
  if (!confusion) return null;

  const { overall, macro_avg, weighted_avg, per_class } = confusion;

  const fmt = (v) =>
    typeof v === 'number' ? v.toFixed(3) : v ?? '—';

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="xs">
        <Text fw={500} size="sm">
          Classification metrics{metricName ? ` (${metricName})` : ''}
        </Text>

        {overall && (
          <Text size="sm">
            <Text span fw={500}>Accuracy:</Text>{' '}
            <Text span fw={700}>{fmt(overall.accuracy)}</Text>{' '}
            <Text span fw={500}>Balanced accuracy:</Text>{' '}
            <Text span fw={700}>{fmt(overall.balanced_accuracy)}</Text>
          </Text>
        )}

        {(macro_avg || weighted_avg) && (
          <Table
            withTableBorder
            withColumnBorders
            horizontalSpacing="xs"
            verticalSpacing="xs"
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th />
                <Table.Th>Precision</Table.Th>
                <Table.Th>Recall</Table.Th>
                <Table.Th>F1</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {macro_avg && (
                <Table.Tr>
                  <Table.Td>
                    <Text size="sm" fw={500}>Macro avg</Text>
                  </Table.Td>
                  <Table.Td>{fmt(macro_avg.precision)}</Table.Td>
                  <Table.Td>{fmt(macro_avg.recall)}</Table.Td>
                  <Table.Td>{fmt(macro_avg.f1)}</Table.Td>
                </Table.Tr>
              )}
              {weighted_avg && (
                <Table.Tr>
                  <Table.Td>
                    <Text size="sm" fw={500}>Weighted avg</Text>
                  </Table.Td>
                  <Table.Td>{fmt(weighted_avg.precision)}</Table.Td>
                  <Table.Td>{fmt(weighted_avg.recall)}</Table.Td>
                  <Table.Td>{fmt(weighted_avg.f1)}</Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        )}

        {Array.isArray(per_class) && per_class.length > 0 && (
          <>
            <Text fw={500} size="sm" mt="xs">
              Per-class metrics
            </Text>
            <Table
              withTableBorder
              withColumnBorders
              horizontalSpacing="xs"
              verticalSpacing="xs"
            >
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Class</Table.Th>
                  <Table.Th>Support</Table.Th>
                  <Table.Th>Precision</Table.Th>
                  <Table.Th>Recall (TPR)</Table.Th>
                  <Table.Th>FPR</Table.Th>
                  <Table.Th>TNR</Table.Th>
                  <Table.Th>FNR</Table.Th>
                  <Table.Th>F1</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {per_class.map((c, idx) => (
                  <Table.Tr key={idx}>
                    <Table.Td>
                      <Text size="sm">{String(c.label)}</Text>
                    </Table.Td>
                    <Table.Td>{c.support}</Table.Td>
                    <Table.Td>{fmt(c.precision)}</Table.Td>
                    <Table.Td>{fmt(c.tpr)}</Table.Td>
                    <Table.Td>{fmt(c.fpr)}</Table.Td>
                    <Table.Td>{fmt(c.tnr)}</Table.Td>
                    <Table.Td>{fmt(c.fnr)}</Table.Td>
                    <Table.Td>{fmt(c.f1)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </>
        )}
      </Stack>
    </Card>
  );
}
