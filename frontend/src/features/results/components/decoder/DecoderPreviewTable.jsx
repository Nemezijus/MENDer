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

  return (
    <ScrollArea
      h={320}
      type="auto"
      offsetScrollbars
      classNames={{
        viewport: 'decoderPreviewViewport',
        scrollbar: 'decoderPreviewScrollbar',
      }}
    >
      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
        striped
      >
        <Table.Thead>
          <Table.Tr>
            {cols.map((c) => {
              const tip = buildHeaderTooltip(c);
              const label = prettifyHeader(c);

              const minWClass =
                c === 'index'
                  ? 'decoderThIndex'
                  : c === 'fold_id'
                    ? 'decoderThFold'
                    : c === 'y_true' || c === 'y_pred'
                      ? 'decoderThY'
                      : c === 'residual' || c === 'abs_error'
                        ? 'decoderThError'
                        : c === 'correct'
                          ? 'decoderThCorrect'
                          : c === 'margin'
                            ? 'decoderThMargin'
                            : '';

              const thClassName = [
                'decoderStickyTh',
                'tableStickyTh',
                'decoderStickyHeader',
                minWClass,
              ]
                .filter(Boolean)
                .join(' ');

              return (
                <Table.Th key={c} className={thClassName}>
                  {tip ? (
                    <Tooltip label={tip} multiline maw={260} withArrow>
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
          {rows.map((row, idx) => {
            const curFold = row?.fold_id;
            const prevFold = idx > 0 ? rows[idx - 1]?.fold_id : null;
            const isFoldBoundary =
              curFold != null && prevFold != null && String(curFold) !== String(prevFold);

            const rowClassName = isFoldBoundary ? 'decoderFoldBoundaryRow' : undefined;

            return (
              <Table.Tr
                key={row?.index ?? idx}
                className={rowClassName}
              >
                {cols.map((c) => {
                  const val = row?.[c];
                  const isCorrectCol = c === 'correct';
                  const isFalse = isCorrectCol && (val === false || val === 'false');

                  const tdClassName = [
                    'tableCellCenterNowrap',
                    isFalse ? 'decoderIncorrectCell' : null,
                  ]
                    .filter(Boolean)
                    .join(' ');

                  return (
                    <Table.Td
                      key={c}
                      className={tdClassName}
                    >
                      <Text size="sm" className="resultsNoWrap">
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
