import { useEffect } from 'react';
import { Card, Stack, Text, Select, Group, Box } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useDataStore } from '../state/useDataStore.js';
import MetricHelpText, { MetricIntroText } from './helpers/helpTexts/MetricHelpText.jsx';

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
  if (
    metricByTask &&
    effectiveTask &&
    Array.isArray(metricByTask[effectiveTask])
  ) {
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

  // Small helper describing range + direction for each metric
  const metricMeta = {
    accuracy: {
      range: '[0, 1]',
      direction: 'higher is better (1 = perfect prediction).',
    },
    balanced_accuracy: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means all classes are perfectly detected, even if imbalanced.',
    },
    f1_macro: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means perfect precision and recall averaged equally over classes.',
    },
    f1_micro: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means perfect global precision and recall over all samples.',
    },
    f1_weighted: {
      range: '[0, 1]',
      direction:
        'higher is better; weighted by how many samples each class has.',
    },
    precision_macro: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means the model almost never assigns the wrong label.',
    },
    recall_macro: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means the model almost never misses true examples of any class.',
    },
    log_loss: {
      range: '[0, ∞)',
      direction:
        'lower is better; 0 means perfectly calibrated probabilities (no penalty).',
    },
    roc_auc_ovr: {
      range: '[0, 1]',
      direction:
        'higher is better; 0.5 is random ranking, 1 means perfect separation in one-vs-rest sense.',
    },
    roc_auc_ovo: {
      range: '[0, 1]',
      direction:
        'higher is better; 0.5 is random, 1 means each pair of classes is perfectly separated.',
    },
    avg_precision_macro: {
      range: '[0, 1]',
      direction:
        'higher is better; 1 means perfect precision–recall trade-off across classes.',
    },
    r2: {
      range: '(-∞, 1]',
      direction:
        'higher is better; 1 is perfect, 0 is no better than a constant baseline, negative means worse than baseline.',
    },
    explained_variance: {
      range: '(-∞, 1]',
      direction:
        'higher is better; 1 means the model perfectly explains the variance of the target.',
    },
    mse: {
      range: '[0, ∞)',
      direction:
        'lower is better; 0 means predictions match targets exactly, large values indicate large squared errors.',
    },
    rmse: {
      range: '[0, ∞)',
      direction:
        'lower is better; same units as the target, 0 means perfect predictions.',
    },
    mae: {
      range: '[0, ∞)',
      direction:
        'lower is better; measures average absolute deviation between predictions and targets.',
    },
    mape: {
      range: '[0, ∞)',
      direction:
        'lower is better; expresses error as a percentage of the true value, but can blow up near zero targets.',
    },
  };

  const selectedMeta = value ? metricMeta[value] : null;

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* Centered, larger title */}
        <Text fw={700} size="lg" align="center">
          {title}
        </Text>

        {/* A + B: controls on the left, short intro on the right */}
        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Stack gap="xs">
              <Select
                label="Metric method"
                data={metricOptions}
                value={value}
                onChange={onChange}
                styles={{
                  input: {
                    borderWidth: 2,
                    borderColor: '#5c94ccff',
                  },
                }}
              />

              {selectedMeta && (
                <Text size="xs" c="dimmed">
                  Range:{' '}
                  <Text span fw={600}>
                    {selectedMeta.range}
                  </Text>{' '}
                  · {selectedMeta.direction}
                </Text>
              )}
            </Stack>
          </Box>

          <Box
            style={{
              flex: 1,
              minWidth: 220,
            }}
          >
            <MetricIntroText />
          </Box>
        </Group>

        {/* C: full-width detailed help text, with the selected metric highlighted */}
        <Box mt="md">
          <MetricHelpText selectedMetric={value} />
        </Box>
      </Stack>
    </Card>
  );
}
