// MENDer/frontend/src/components/ConfusionTable.jsx
import React from 'react';
import { Table, Text, ScrollArea } from '@mantine/core';

export default function ConfusionTable({ labels, matrix }) {
  if (!labels || !matrix || labels.length === 0) {
    return null;
  }

  // Build header row: ["", "Pred <lbl0>", "Pred <lbl1>", ...]
  const headerCells = [
    <Table.Th key="empty-cell"></Table.Th>,
    ...labels.map((lbl, j) => (
      <Table.Th key={`pred-${j}`}>
        <Text size="sm" fw={500}>
          Pred {String(lbl)}
        </Text>
      </Table.Th>
    )),
  ];

  // Build body rows: first cell "True X", then row counts
  const bodyRows = matrix.map((row, i) => (
    <Table.Tr key={`row-${i}`}>
      <Table.Th>
        <Text size="sm" fw={500}>
          True {String(labels[i])}
        </Text>
      </Table.Th>
      {row.map((val, j) => (
        <Table.Td key={`cell-${i}-${j}`}>
          <Text size="sm" ta="right">
            {val}
          </Text>
        </Table.Td>
      ))}
    </Table.Tr>
  ));

  return (
    <ScrollArea style={{ maxWidth: '100%' }} mt="md">
      <Table
        striped
        highlightOnHover
        withTableBorder
        withColumnBorders
        stickyHeader
        stickyHeaderOffset={0}
      >
        <Table.Thead>
          <Table.Tr>{headerCells}</Table.Tr>
        </Table.Thead>
        <Table.Tbody>{bodyRows}</Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
