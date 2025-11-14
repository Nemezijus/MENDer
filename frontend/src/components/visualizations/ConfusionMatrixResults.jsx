import { Stack, Text, Table } from '@mantine/core';

export default function ConfusionMatrixResults({
  confusion,
}) {
  if (!confusion || !confusion.matrix || !confusion.labels) {
    return null;
  }

  const { matrix, labels } = confusion;

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">Confusion matrix</Text>
      <Table striped withTableBorder withColumnBorders maw={460}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th></Table.Th>
            {labels.map((lbl, j) => (
              <Table.Th key={`pred-${j}`}>
                <Text size="sm" fw={500}>Pred {String(lbl)}</Text>
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {matrix.map((row, i) => (
            <Table.Tr key={i}>
              <Table.Td>
                <Text size="sm" fw={500}>True {String(labels[i])}</Text>
              </Table.Td>
              {row.map((v, j) => (
                <Table.Td key={j}>{v}</Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}
