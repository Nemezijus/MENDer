import { Group, Text, Tooltip } from '@mantine/core';

export default function PlotTitle({ label, tip }) {
  if (!label) return null;

  return (
    <Group justify="center" gap={0}>
      {tip ? (
        <Tooltip label={tip} multiline maw={360} withArrow>
          <Text fw={500} size="xl" align="center" style={{ cursor: 'help' }}>
            {label}
          </Text>
        </Tooltip>
      ) : (
        <Text fw={500} size="xl" align="center">
          {label}
        </Text>
      )}
    </Group>
  );
}
