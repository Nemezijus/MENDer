import { useEffect, useMemo } from 'react';

/**
 * Compute the effective metric for a given inferred task.
 *
 * Rules:
 * - Stores are overrides-only: if `metric` is unset, fall back to the first
 *   task-specific metric from the backend schema (enums.MetricByTask).
 * - If the user has an explicit override that is invalid for the current task,
 *   clear it.
 */
export function useEffectiveMetricForTask({ enums, taskInferred, metric, setMetric }) {
  const allowedMetrics = useMemo(() => {
    const mt = enums?.MetricByTask || null;
    if (taskInferred && mt && Array.isArray(mt[taskInferred])) {
      return mt[taskInferred].map(String);
    }
    if (Array.isArray(enums?.MetricName)) return enums.MetricName.map(String);
    return [];
  }, [enums, taskInferred]);

  const defaultMetricFromSchema = allowedMetrics?.[0] ?? null;
  const effectiveMetric = metric ?? defaultMetricFromSchema;

  useEffect(() => {
    if (!metric) return;
    if (allowedMetrics.length > 0 && !allowedMetrics.includes(String(metric))) {
      setMetric?.(undefined);
    }
  }, [allowedMetrics, metric, setMetric]);

  return {
    allowedMetrics,
    defaultMetricFromSchema,
    effectiveMetric,
  };
}
