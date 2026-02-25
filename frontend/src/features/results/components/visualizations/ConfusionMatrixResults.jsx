import { Stack, Text } from '@mantine/core';

import ConfusionMatrixGrid from './confusionMatrix/ConfusionMatrixGrid.jsx';

export default function ConfusionMatrixResults({ confusion }) {
  if (!confusion || !confusion.matrix || !confusion.labels) {
    return null;
  }

  const { matrix, labels } = confusion;

  return (
    <Stack gap="xs">
      <Text fw={500} size="xl" ta="center">
        Confusion matrix
      </Text>

      <ConfusionMatrixGrid matrix={matrix} labels={labels} />
    </Stack>
  );
}
