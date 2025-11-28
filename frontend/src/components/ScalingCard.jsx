// src/components/ScalingCard.jsx
import { Card, Stack, Text, Select } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';

export default function ScalingCard({ value, onChange, title = 'Scaling' }) {
  const { enums } = useSchemaDefaults();
  const scaleOptions = (enums?.ScaleName ?? ['none', 'standard', 'robust', 'minmax', 'maxabs', 'quantile'])
    .map((v) => ({ value: String(v), label: v === 'none' ? 'None' : String(v.charAt(0).toUpperCase() + v.slice(1)) + (v.endsWith('abs') ? 'Scaler' : (v === 'quantile' ? 'Transformer' : 'Scaler')) }));

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={600}>{title}</Text>
        <Select
          label="Scaling method"
          data={scaleOptions}
          value={value}
          onChange={onChange}
        />
      </Stack>
    </Card>
  );
}
