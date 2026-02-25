import { Table, Text, Tooltip } from '@mantine/core';

function HeaderCell({ label, tooltip }) {
  return (
    <Table.Th style={{ textAlign: 'center' }}>
      {tooltip ? (
        <Tooltip label={tooltip} multiline maw={260} withArrow>
          <Text size="xs" fw={600} c="white">
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
}

export default function ClassificationMetricTable({
  perClass,
  macro,
  weighted,
  fmt,
  tooltips,
}) {
  const rows = Array.isArray(perClass) ? perClass : [];
  if (!rows.length) return null;

  return (
    <Table withTableBorder={false} withColumnBorders={false} horizontalSpacing="xs" verticalSpacing="xs">
      <Table.Thead>
        <Table.Tr style={{ backgroundColor: 'var(--mantine-color-gray-8)' }}>
          <HeaderCell label="Class / aggregate" />
          <HeaderCell label="Precision" tooltip={tooltips?.precisionTooltip} />
          <HeaderCell label="Recall (TPR)" tooltip={tooltips?.recallTooltip} />
          <HeaderCell label="FPR" tooltip={tooltips?.fprTooltip} />
          <HeaderCell label="TNR" tooltip={tooltips?.tnrTooltip} />
          <HeaderCell label="FNR" tooltip={tooltips?.fnrTooltip} />
          <HeaderCell label="F1" tooltip={tooltips?.f1Tooltip} />
          <HeaderCell label="MCC" tooltip={tooltips?.mccTooltip} />
        </Table.Tr>
      </Table.Thead>

      <Table.Tbody>
        {rows.map((c, idx) => {
          const isStriped = idx % 2 === 1;
          return (
            <Table.Tr
              key={idx}
              style={{
                backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
              }}
            >
              <Table.Td style={{ textAlign: 'center' }}>
                <Text size="sm" fw={600}>
                  {String(c.label)}
                </Text>
              </Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.precision)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.tpr)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.fpr)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.tnr)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.fnr)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.f1)}</Table.Td>
              <Table.Td style={{ textAlign: 'center' }}>{fmt(c.mcc)}</Table.Td>
            </Table.Tr>
          );
        })}

        <Table.Tr style={{ borderTop: '2px solid var(--mantine-color-gray-4)', height: 4 }}>
          <Table.Td colSpan={8} style={{ padding: 0 }} />
        </Table.Tr>

        <Table.Tr
          style={{
            backgroundColor: 'var(--mantine-color-gray-2)',
            borderTop: '1px solid var(--mantine-color-gray-4)',
          }}
        >
          <Table.Td style={{ textAlign: 'left' }}>
            <Tooltip label={tooltips?.macroTooltip} multiline maw={260} withArrow>
              <Text size="sm" fw={600}>
                Macro avg
              </Text>
            </Tooltip>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.precision)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.tpr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.fpr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.tnr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.fnr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.f1)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(macro?.mcc)}
            </Text>
          </Table.Td>
        </Table.Tr>

        <Table.Tr
          style={{
            backgroundColor: 'var(--mantine-color-gray-3)',
            borderTop: '1px solid var(--mantine-color-gray-4)',
          }}
        >
          <Table.Td style={{ textAlign: 'left' }}>
            <Tooltip label={tooltips?.weightedTooltip} multiline maw={260} withArrow>
              <Text size="sm" fw={600}>
                Weighted avg
              </Text>
            </Tooltip>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.precision)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.tpr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.fpr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.tnr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.fnr)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.f1)}
            </Text>
          </Table.Td>
          <Table.Td style={{ textAlign: 'center' }}>
            <Text size="sm" fw={600}>
              {fmt(weighted?.mcc)}
            </Text>
          </Table.Td>
        </Table.Tr>
      </Table.Tbody>
    </Table>
  );
}
