// src/components/ScalingCard.jsx
import { Card, Stack, Text, Select } from '@mantine/core';

export default function ScalingCard({ value, onChange, title = 'Scaling' }) {
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>{title}</Text>
        <Select
          label="Scaling method"
          data={[
            { value: 'none', label: 'None' },
            { value: 'standard', label: 'StandardScaler' },
            { value: 'robust', label: 'RobustScaler' },
            { value: 'minmax', label: 'MinMaxScaler' },
            { value: 'maxabs', label: 'MaxAbsScaler' },
            { value: 'quantile', label: 'QuantileTransformer' },
          ]}
          value={value}
          onChange={onChange}
        />
      </Stack>
    </Card>
  );
}
