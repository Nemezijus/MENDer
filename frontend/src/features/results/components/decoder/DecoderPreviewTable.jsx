import { ScrollArea, Table, Text, Tooltip } from '@mantine/core';

import { parseNumber, fmt3 } from '../../utils/formatNumbers.js';
import { buildHeaderTooltip, prettifyHeader } from '../../utils/decoderOutputs.js';

export default function DecoderPreviewTable({ preview, columns }) {
  const rows = Array.isArray(preview) ? preview : [];
  const cols = Array.isArray(columns) ? columns : [];

  const renderCell = (col, value) => {
    if (value === null || value === undefined) return '—';

    if (col === 'correct') {
      const isTrue = value === true || value === 'true';
      return isTrue ? 'true' : 'false';
    }

    const num = parseNumber(value);
    if (num !== null) return fmt3(num);

    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
  };

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
    <ScrollArea
      h={320}
      type="auto"
      offsetScrollbars
      styles={{ viewport: { paddingRight: 8, paddingBottom: 6 }, scrollbar: { zIndex: 5 } }}
    >
      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
      >
        <Table.Thead>
          <Table.Tr>
            {cols.map((c) => {
              const tip = buildHeaderTooltip(c);
              const label = prettifyHeader(c);

              const minW =
                c === 'index'
                  ? 55
                  : c === 'fold_id'
                    ? 70
                    : c === 'y_true' || c === 'y_pred'
                      ? 85
                      : c === 'residual' || c === 'abs_error'
                        ? 95
                        : c === 'correct'
                          ? 80
                          : c === 'margin'
                            ? 80
                            : undefined;

              const thStyle = minW ? { ...stickyThStyle, minWidth: minW } : stickyThStyle;

              return (
                <Table.Th key={c} style={thStyle}>
                  {tip ? (
                    <Tooltip label={tip} multiline maw={260} withArrow>
                      <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                        {label}
                      </Text>
                    </Tooltip>
                  ) : (
                    <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                      {label}
                    </Text>
                  )}
                </Table.Th>
              );
            })}
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>
          {rows.map((row, idx) => {
            const isStriped = idx % 2 === 1;

            const curFold = row?.fold_id;
            const prevFold = idx > 0 ? rows[idx - 1]?.fold_id : null;
            const isFoldBoundary =
              curFold != null && prevFold != null && String(curFold) !== String(prevFold);

            return (
              <Table.Tr
                key={row?.index ?? idx}
                style={{
                  backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
                  borderTop: isFoldBoundary
                    ? '3px solid var(--mantine-color-gray-4)'
                    : undefined,
                }}
              >
                {cols.map((c) => {
                  const val = row?.[c];
                  const isCorrectCol = c === 'correct';
                  const isFalse = isCorrectCol && (val === false || val === 'false');

                  return (
                    <Table.Td
                      key={c}
                      style={{
                        textAlign: 'center',
                        backgroundColor: isFalse ? 'var(--mantine-color-red-1)' : undefined,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <Text size="sm" style={{ whiteSpace: 'nowrap' }}>
                        {renderCell(c, val)}
                      </Text>
                    </Table.Td>
                  );
                })}
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
