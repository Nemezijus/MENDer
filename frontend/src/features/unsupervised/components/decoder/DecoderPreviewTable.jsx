import { ScrollArea, Table, Tooltip, Text } from '@mantine/core';

import { fmtCell } from '../../utils/format.js';
import { headerLabel, headerTooltip } from '../../utils/labels.js';

export default function DecoderPreviewTable({ rows, columns, height = 360 }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const safeCols = Array.isArray(columns) ? columns : [];

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
    <ScrollArea h={height} type="auto">
      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
      >
        <Table.Thead style={{ position: 'sticky', top: 0, zIndex: 2 }}>
          <Table.Tr>
            {safeCols.map((c) => {
              const tip = headerTooltip(c);
              const label = headerLabel(c);
              return (
                <Table.Th key={c} style={stickyThStyle}>
                  {tip ? (
                    <Tooltip label={tip} multiline maw={360} withArrow>
                      <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                        {label}
                      </Text>
                    </Tooltip>
                  ) : (
                    <Text size="xs" fw={600} c="white">
                      {label}
                    </Text>
                  )}
                </Table.Th>
              );
            })}
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>
          {safeRows.map((r, i) => {
            const isStriped = i % 2 === 1;
            return (
              <Table.Tr
                key={i}
                style={{
                  backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
                }}
              >
                {safeCols.map((c) => (
                  <Table.Td key={c} style={{ textAlign: 'center' }}>
                    {fmtCell(r?.[c])}
                  </Table.Td>
                ))}
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
