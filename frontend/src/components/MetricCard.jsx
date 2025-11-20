import { Card, Stack, Text, Select } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';

export default function MetricCard({
  title = 'Metric',
  value,
  onChange,
}) {
  const { enums } = useSchemaDefaults();
  const metricOptions = (enums?.MetricName ?? ['accuracy', 'balanced_accuracy', 'f1_macro'])
    .map((v) => ({ value: String(v), label: String(v) }));

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>{title}</Text>
        <Select
          label="Metric method"
          data={metricOptions}
          value={value}
          onChange={onChange}
        />
      </Stack>
    </Card>
  );
}
