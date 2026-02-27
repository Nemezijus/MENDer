import { useMemo } from 'react';
import { Stack, Text, Select, MultiSelect } from '@mantine/core';
import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import { enumFromSubSchema } from '../../utils/schema/jsonSchema.js';
import { useDataStore } from '../../../features/dataFiles/state/useDataStore.js';
import MetricHelpText, { MetricIntroText } from '../../content/help/MetricHelpText.jsx';
import { getMetricMeta } from '../../constants/metrics.js';
import ConfigCardShell from './common/ConfigCardShell.jsx';

import '../styles/forms.css';

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

  const selectedSingle = isUnsupervised
    ? (unsupervisedDisplay?.[0] ?? null)
    : supervisedDisplay;

  const selectedMeta = getMetricMeta(selectedSingle);

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
              classNames={{
                input: 'configSelectInput',
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
              classNames={{
                input: 'configSelectInput',
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
