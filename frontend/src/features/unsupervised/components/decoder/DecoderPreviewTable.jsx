import { ScrollArea, Table, Tooltip, Text } from '@mantine/core';

import { fmtCell } from '../../utils/format.js';
import { headerLabel, headerTooltip } from '../../utils/labels.js';

export default function DecoderPreviewTable({ rows, columns, height = 360 }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const safeCols = Array.isArray(columns) ? columns : [];

  return (
    <ScrollArea h={height} type="auto">
      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
        striped
      >
        <Table.Thead className="unsupPreviewThead">
          <Table.Tr>
            {safeCols.map((c) => {
              const tip = headerTooltip(c);
              const label = headerLabel(c);
              return (
                <Table.Th key={c} className="unsupPreviewTh tableStickyTh">
                  {tip ? (
                    <Tooltip label={tip} multiline maw={360} withArrow>
                      <Text size="xs" fw={600} c="white" className="tableHeaderTextCompact">
                        {label}
                      </Text>
                    </Tooltip>
                  ) : (
                    <Text size="xs" fw={600} c="white" className="tableHeaderTextCompact">
                      {label}
                    </Text>
                  )}
                </Table.Th>
              );
            })}
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>
          {safeRows.map((r, i) => (
            <Table.Tr key={i}>
              {safeCols.map((c) => (
                <Table.Td key={c} className="unsupPreviewTd">
                  {fmtCell(r?.[c])}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
