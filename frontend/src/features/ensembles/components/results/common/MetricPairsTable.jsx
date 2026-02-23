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
    >
      <Table.Tbody>
        {safeRows.map((pair, i) => (
          <Table.Tr
            key={i}
            style={{
              backgroundColor: i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
            }}
          >
            <Table.Td style={{ width: '25%' }}>
              <Tooltip label={pair?.[0]?.tooltip} multiline maw={tooltipMaw} withArrow>
                <Text size="sm" fw={600}>
                  {pair?.[0]?.label}
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td style={{ width: '25%' }}>
              <Text size="sm" fw={700}>
                {pair?.[0]?.value}
              </Text>
            </Table.Td>

            <Table.Td style={{ width: '25%' }}>
              <Tooltip label={pair?.[1]?.tooltip} multiline maw={tooltipMaw} withArrow>
                <Text size="sm" fw={600}>
                  {pair?.[1]?.label}
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td style={{ width: '25%' }}>
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
