import { Stack, Text, Group, Select, Tooltip } from '@mantine/core';

export default function DecoderClassProbabilitySummary({
  show,
  classOptions,
  selectedClass,
  setSelectedClass,
  probStats,
}) {
  if (!show) return null;

  return (
    <Stack gap="xs">
      <Stack gap={0}>
        <Text size="sm" fw={600}>
          Class probability summary
        </Text>
        <Text size="xs" c="dimmed">
          Mean/median of p(class) across the preview rows (quick inspection).
        </Text>
      </Stack>

      <Group justify="space-between" align="flex-end" wrap="wrap">
        <Select
          label="Class"
          data={classOptions}
          value={selectedClass}
          onChange={setSelectedClass}
          w={240}
        />

        <Group gap="lg" wrap="wrap">
          <Tooltip
            label="Average p(selected class) over preview rows. Range 0–1."
            multiline
            maw={280}
            withArrow
          >
            <Text size="sm">
              <Text span c="dimmed">
                Mean:{' '}
              </Text>
              <Text span fw={700}>
                {probStats?.mean != null ? probStats.mean.toFixed(3) : '—'}
              </Text>
            </Text>
          </Tooltip>

          <Tooltip
            label="Median p(selected class) over preview rows. Range 0–1."
            multiline
            maw={280}
            withArrow
          >
            <Text size="sm">
              <Text span c="dimmed">
                Median:{' '}
              </Text>
              <Text span fw={700}>
                {probStats?.median != null ? probStats.median.toFixed(3) : '—'}
              </Text>
            </Text>
          </Tooltip>

          <Tooltip
            label="Number of preview rows used for these summary values."
            multiline
            maw={280}
            withArrow
          >
            <Text size="sm">
              <Text span c="dimmed">
                Rows used:{' '}
              </Text>
              <Text span fw={700}>
                {probStats?.n ?? 0}
              </Text>
            </Text>
          </Tooltip>
        </Group>
      </Group>
    </Stack>
  );
}
