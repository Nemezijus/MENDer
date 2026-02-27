import {
  Stack,
  Text,
  Table,
  ScrollArea,
} from '@mantine/core';

import { fmtCell } from '../../utils/format.js';

export default function ClusterSizesTable({ rows, height = 220 }) {
  const safeRows = Array.isArray(rows) ? rows : [];

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
    <Stack gap="xs">
      <Text size="sm" fw={600}>
        Cluster sizes
      </Text>

      <ScrollArea h={height} type="auto" offsetScrollbars>
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
            {safeRows.length === 0 ? (
              <Table.Tr>
                <Table.Td colSpan={2}>
                  <Text size="sm" c="dimmed">
                    —
                  </Text>
                </Table.Td>
              </Table.Tr>
            ) : (
              safeRows.map((r, i) => {
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
  );
}
