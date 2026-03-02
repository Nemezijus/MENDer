import { Table, Text, Tooltip } from '@mantine/core';

/**
 * Render a 2x2 pairs table:
 * rows: Array<[leftItem, rightItem]>
 * item = { label, value, tooltip }
 */
export default function MetricPairsTable({ rows, tooltipMaw = 360 }) {
  const safeRows = Array.isArray(rows) ? rows : [];

  return (
    <Table
      withTableBorder={false}
      withColumnBorders={false}
      horizontalSpacing="xs"
      verticalSpacing="xs"
      mt="xs"
      striped
    >
      <Table.Tbody>
        {safeRows.map((pair, i) => (
          <Table.Tr key={i}>
            <Table.Td className="ensMetricPairsCell">
              <Tooltip label={pair?.[0]?.tooltip} multiline maw={tooltipMaw} withArrow>
                <Text size="sm" fw={600}>
                  {pair?.[0]?.label}
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td className="ensMetricPairsCell">
              <Text size="sm" fw={700}>
                {pair?.[0]?.value}
              </Text>
            </Table.Td>

            <Table.Td className="ensMetricPairsCell">
              <Tooltip label={pair?.[1]?.tooltip} multiline maw={tooltipMaw} withArrow>
                <Text size="sm" fw={600}>
                  {pair?.[1]?.label}
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td className="ensMetricPairsCell">
              <Text size="sm" fw={700}>
                {pair?.[1]?.value}
              </Text>
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}
