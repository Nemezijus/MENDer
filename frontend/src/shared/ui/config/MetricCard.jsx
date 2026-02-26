import { useMemo } from 'react';
import { Stack, Text, Select, MultiSelect } from '@mantine/core';
import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import { enumFromSubSchema } from '../../utils/schema/jsonSchema.js';
import { useDataStore } from '../../../features/dataFiles/state/useDataStore.js';
import MetricHelpText, { MetricIntroText } from '../../content/help/MetricHelpText.jsx';
import ConfigCardShell from './common/ConfigCardShell.jsx';

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

  const enumFromArrayItems = (schema, key) => {
    try {
      const items = schema?.properties?.[key]?.items;
      if (!items) return null;

      if (Array.isArray(items.enum)) return items.enum;

      const list = (items.anyOf ?? items.oneOf ?? []).flatMap((x) => {
        if (Array.isArray(x.enum)) return x.enum;
        if (x.const != null) return [x.const];
        return [];
      });

      return list.length ? list : null;
    } catch {
      return null;
    }
  };

  // Schema defaults
  const metricByTaskDefaults = evalCfg?.defaults?.metric_by_task ?? null;
  const defaultSupervisedMetric =
    (!isUnsupervised && metricByTaskDefaults && effectiveTask && metricByTaskDefaults[effectiveTask]
      ? String(metricByTaskDefaults[effectiveTask])
      : (evalCfg?.defaults?.metric != null ? String(evalCfg.defaults.metric) : null));
  const defaultUnsupervisedMetrics =
    (unsupervised?.eval?.defaults?.metrics ?? []).map(String);

  const rawList = useMemo(() => {
    if (isUnsupervised) {
      if (Array.isArray(enums?.UnsupervisedMetricName) && enums.UnsupervisedMetricName.length) {
        return enums.UnsupervisedMetricName;
      }

      const fromSchema = enumFromArrayItems(unsupervised?.eval?.schema, 'metrics');
      if (Array.isArray(fromSchema) && fromSchema.length) return fromSchema;

      return [];
    }

    if (
      metricByTask &&
      effectiveTask &&
      Array.isArray(metricByTask[effectiveTask]) &&
      metricByTask[effectiveTask].length
    ) {
      return metricByTask[effectiveTask];
    }

    if (Array.isArray(enums?.MetricName) && enums.MetricName.length) {
      return enums.MetricName;
    }

    const fromSchema = enumFromSubSchema(evalCfg?.schema, 'metric');
    if (Array.isArray(fromSchema) && fromSchema.length) {
      return fromSchema.filter((v) => v != null);
    }

    return [];
  }, [enums, metricByTask, effectiveTask, isUnsupervised, unsupervised, evalCfg]);


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

  const optionsUnavailable = metricOptions.length === 0;

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
    <ConfigCardShell
      title={title}
      left={(
        <Stack gap="xs">
          {isUnsupervised ? (
            <MultiSelect
              label="Metrics"
              description={optionsUnavailable ? 'Metric options unavailable (schema missing).' : 'Optional. If empty, the backend will compute a default set.'}
              data={metricOptions}
              value={unsupervisedDisplay}
              onChange={handleUnsupervisedChange}
              disabled={optionsUnavailable}
              placeholder={optionsUnavailable ? 'Schema enums unavailable' : undefined}
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
              disabled={optionsUnavailable}
              placeholder={optionsUnavailable ? 'Schema enums unavailable' : undefined}
              description={optionsUnavailable ? 'Metric options unavailable (schema missing).' : undefined}
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
      )}
      right={<MetricIntroText />}
      help={<MetricHelpText selectedMetric={selectedSingle} />}
    />
  );
}
