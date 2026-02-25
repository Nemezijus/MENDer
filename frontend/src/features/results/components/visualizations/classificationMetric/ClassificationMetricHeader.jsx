import { Stack, Text } from '@mantine/core';

export default function ClassificationMetricHeader({ overall, imbalanceRatio, imbalanceDesc, fmt, fmtRatio }) {
  if (!overall) return null;

  return (
    <Stack gap={0}>
      <Text size="sm">
        <Text span fw={500}>
          Accuracy:{' '}
        </Text>
        <Text span fw={700}>
          {fmt(overall.accuracy)}
        </Text>
      </Text>

      <Text size="sm">
        <Text span fw={500}>
          Balanced accuracy:{' '}
        </Text>
        <Text span fw={700}>
          {fmt(overall.balanced_accuracy)}
        </Text>
      </Text>

      <Text size="sm">
        <Text span fw={500}>
          Class imbalance (max / min support):{' '}
        </Text>
        <Text span fw={700}>
          {fmtRatio(imbalanceRatio)}
        </Text>
        {imbalanceDesc ? (
          <>
            {' '}
            <Text span size="xs" c="dimmed">
              {imbalanceDesc}
            </Text>
          </>
        ) : null}
      </Text>
    </Stack>
  );
}
