import {
  Stack,
  Text,
  Table,
  ScrollArea,
} from '@mantine/core';

import { fmtCell } from '../../utils/format.js';

export default function ClusterSizesTable({ rows, height = 220 }) {
  const safeRows = Array.isArray(rows) ? rows : [];

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
          striped
        >
          <Table.Thead>
            <Table.Tr>
              <Table.Th className="unsupDiagStickyTh tableStickyTh">
                <Text size="xs" fw={600} c="white" className="tableHeaderTextCompact">
                  Cluster id
                </Text>
              </Table.Th>
              <Table.Th className="unsupDiagStickyTh tableStickyTh">
                <Text size="xs" fw={600} c="white" className="tableHeaderTextCompact">
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
              safeRows.map((r, i) => (
                <Table.Tr key={i}>
                  <Table.Td className="tableCellCenterNowrap">{fmtCell(r.cluster_id)}</Table.Td>
                  <Table.Td className="tableCellCenterNowrap">{fmtCell(r.size)}</Table.Td>
                </Table.Tr>
              ))
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Stack>
  );
}
