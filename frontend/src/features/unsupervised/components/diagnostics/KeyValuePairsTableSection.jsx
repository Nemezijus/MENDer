import {
  Stack,
  Text,
  Table,
  Tooltip,
} from '@mantine/core';

import { titleCaseFromKey, tooltipForKey } from '../../utils/labels.js';

export default function KeyValuePairsTableSection({
  title,
  pairs,
  emptyText,
  rowKeyPrefix,
  renderValue,
}) {
  const safePairs = Array.isArray(pairs) ? pairs : [];
  const emptyMsg = emptyText || '—';
  const keyPrefix = rowKeyPrefix || title || 'pair';

  return (
    <Stack gap="xs">
      {title && (
        <Text size="sm" fw={600}>
          {title}
        </Text>
      )}

      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
        style={{ tableLayout: 'fixed' }}
      >
        <Table.Tbody>
          {safePairs.length === 0 ? (
            <Table.Tr>
              <Table.Td colSpan={2}>
                <Text size="sm" c="dimmed">
                  {emptyMsg}
                </Text>
              </Table.Td>
            </Table.Tr>
          ) : (
            safePairs.map(([k, val]) => {
              const label = titleCaseFromKey(k);
              const tip = tooltipForKey(k);
              return (
                <Table.Tr key={`${keyPrefix}-${k}`}>
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
                      {renderValue ? renderValue(val) : String(val)}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              );
            })
          )}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}
