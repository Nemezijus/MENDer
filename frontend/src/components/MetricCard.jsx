import { useMemo } from 'react';
import { Card, Stack, Text, Select, Group, Box, MultiSelect } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useDataStore } from '../state/useDataStore.js';
import MetricHelpText, { MetricIntroText } from './helpers/helpTexts/MetricHelpText.jsx';

function normalizeTaskName(t) {
  if (!t) return null;
  return t === 'clustering' ? 'unsupervised' : t;
}

export default function MetricCard({
  title = 'Metric',
  value,
  onChange,
}) {
  const { enums, eval: evalCfg, unsupervised } = useSchemaDefaults();

  const effectiveTask = useDataStore((s) => {
    const raw = s.taskSelected || s.inspectReport?.task_inferred || null;
    return normalizeTaskName(raw);
  });

  const metricByTask = enums?.MetricByTask || null;
  const isUnsupervised = effectiveTask === 'unsupervised';

  // Schema defaults
  const defaultSupervisedMetric = evalCfg?.defaults?.metric ?? null;
  const defaultUnsupervisedMetrics =
    (unsupervised?.eval?.defaults?.metrics ?? []).map(String);

  const rawList = useMemo(() => {
    if (isUnsupervised) {
      if (Array.isArray(enums?.UnsupervisedMetricName)) return enums.UnsupervisedMetricName;
      return ['silhouette', 'davies_bouldin', 'calinski_harabasz'];
    }

    if (metricByTask && effectiveTask && Array.isArray(metricByTask[effectiveTask])) {
      // Prefer backend-provided task-specific list
      return metricByTask[effectiveTask];
    }
    if (Array.isArray(enums?.MetricName)) {
      // Fallback: all supervised metrics
      return enums.MetricName;
    }
    // Legacy fallback
    return ['accuracy', 'balanced_accuracy', 'f1_macro'];
  }, [enums, metricByTask, effectiveTask, isUnsupervised]);

  // Store is overrides-only:
  // - Never auto-write defaults into state.
  // - Display schema defaults when unset.
  const arraysEqual = (a, b) => {
    if (a === b) return true;
    if (!Array.isArray(a) || !Array.isArray(b)) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (String(a[i]) !== String(b[i])) return false;
    }
    return true;
  };

  const effectiveSupervised = value ?? defaultSupervisedMetric ?? null;
  const supervisedDisplay =
    effectiveSupervised && rawList.includes(String(effectiveSupervised))
      ? String(effectiveSupervised)
      : null;

  const effectiveUnsupervised = Array.isArray(value)
    ? value.map(String)
    : defaultUnsupervisedMetrics;
  const unsupervisedDisplay = (effectiveUnsupervised ?? []).filter((m) =>
    rawList.includes(String(m)),
  );

  const handleSupervisedChange = (v) => {
    if (typeof onChange !== 'function') return;
    const next = v ? String(v) : undefined;
    if (defaultSupervisedMetric != null && next === String(defaultSupervisedMetric)) {
      onChange(undefined);
      return;
    }
    onChange(next);
  };

  const handleUnsupervisedChange = (arr) => {
    if (typeof onChange !== 'function') return;
    const next = (arr ?? []).map(String);
    if (arraysEqual(next, defaultUnsupervisedMetrics)) {
      onChange(undefined);
      return;
    }
    onChange(next);
  };

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
    silhouette: {
      range: '[-1, 1]',
      direction: 'higher is better; values near 1 indicate dense, well-separated clusters.',
    },
    davies_bouldin: {
      range: '[0, ∞)',
      direction: 'lower is better; 0 indicates perfect separation.',
    },
    calinski_harabasz: {
      range: '[0, ∞)',
      direction: 'higher is better; ratio of between-cluster to within-cluster dispersion.',
    },
  };

  const selectedSingle = isUnsupervised
    ? (unsupervisedDisplay?.[0] ?? null)
    : supervisedDisplay;

  const selectedMeta = selectedSingle ? metricMeta[selectedSingle] : null;

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
              {isUnsupervised ? (
                <MultiSelect
                  label="Metrics"
                  description="Optional. If empty, the backend will compute a default set."
                  data={metricOptions}
                  value={unsupervisedDisplay}
                  onChange={handleUnsupervisedChange}
                  searchable
                  clearable
                  styles={{
                    input: {
                      borderWidth: 2,
                      borderColor: '#5c94ccff',
                    },
                  }}
                />
              ) : (
                <Select
                  label="Metric method"
                  data={metricOptions}
                  value={supervisedDisplay}
                  onChange={handleSupervisedChange}
                  styles={{
                    input: {
                      borderWidth: 2,
                      borderColor: '#5c94ccff',
                    },
                  }}
                />
              )}

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
          <MetricHelpText selectedMetric={selectedSingle} />
        </Box>
      </Stack>
    </Card>
  );
}
