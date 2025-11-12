import { Card, Stack, Text, Checkbox, NumberInput, Tooltip } from '@mantine/core';

export default function ShuffleLabelsCard({
  title = 'Shuffle labels (control)',
  checked,
  onCheckedChange,
  nShuffles,
  onNShufflesChange,
}) {
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>{title}</Text>

        <Tooltip label="Create a baseline by shuffling labels and re-scoring. Does not affect the main training run.">
          <Checkbox
            label="Enable shuffle-baseline"
            checked={checked}
            onChange={(e) => onCheckedChange?.(e.currentTarget.checked)}
          />
        </Tooltip>

        {checked && (
          <NumberInput
            label="Number of shuffles"
            min={10}
            max={5000}
            step={10}
            value={nShuffles}
            onChange={onNShufflesChange}
          />
        )}
      </Stack>
    </Card>
  );
}