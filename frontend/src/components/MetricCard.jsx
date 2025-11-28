import { useEffect } from 'react';
import { Card, Stack, Text, Select } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useDataStore } from '../state/useDataStore.js';

export default function MetricCard({
  title = 'Metric',
  value,
  onChange,
}) {
  const { enums } = useSchemaDefaults();

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  const metricByTask = enums?.MetricByTask || null;

  let rawList;
  if (metricByTask && effectiveTask && Array.isArray(metricByTask[effectiveTask])) {
    // Prefer backend-provided task-specific list
    rawList = metricByTask[effectiveTask];
  } else if (Array.isArray(enums?.MetricName)) {
    // Fallback: all metrics
    rawList = enums.MetricName;
  } else {
    // Legacy fallback
    rawList = ['accuracy', 'balanced_accuracy', 'f1_macro'];
  }

  // Ensure the selected value is always in the current option list.
  useEffect(() => {
    if (!rawList || rawList.length === 0) return;

    const isValid = rawList.includes(value);
    if (!isValid) {
      const first = rawList[0];
      if (first != null && typeof onChange === 'function') {
        onChange(String(first));
      }
    }
  }, [rawList, value, onChange]);

  const metricOptions = rawList.map((v) => ({
    value: String(v),
    label: String(v),
  }));

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={600}>{title}</Text>
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
