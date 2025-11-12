import { Card, Stack, Text, Select } from '@mantine/core';

export default function MetricCard({
  title = 'Metric',
  value,
  onChange,
}) {
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>{title}</Text>
        <Select
          label="Metric method"
          data={[
            { value: 'accuracy', label: 'accuracy' },
            { value: 'balanced_accuracy', label: 'balanced_accuracy' },
            { value: 'f1_macro', label: 'f1_macro' },
          ]}
          value={value}
          onChange={onChange}
        />
      </Stack>
    </Card>
  );
}