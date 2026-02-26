import { useEffect, useMemo } from 'react';

/**
 * Compute metric options + a display default for a given inferred task.
 *
 * Rules:
 * - Stores are overrides-only: `metric` is an explicit override only.
 * - The display default comes from schema defaults (eval.defaults.metric_by_task),
 *   not from "first allowed metric" heuristics.
 * - If the user has an explicit override that is invalid for the current task,
 *   clear it.
 *
 * NOTE:
 * This hook MUST NOT manufacture a metric for payloads. Callers should send
 * `metric` (override) to the backend and let the Engine choose defaults.
 */
export function useEffectiveMetricForTask({
  enums,
  taskInferred,
  metric,
  setMetric,
  evalDefaults,
}) {
  const allowedMetrics = useMemo(() => {
    const mt = enums?.MetricByTask || null;
    if (taskInferred && mt && Array.isArray(mt[taskInferred])) {
      return mt[taskInferred].map(String);
    }
    if (Array.isArray(enums?.MetricName)) return enums.MetricName.map(String);
    return [];
  }, [enums, taskInferred]);

  const defaultMetricFromSchema = useMemo(() => {
    const byTask = evalDefaults?.metric_by_task ?? null;
    if (taskInferred && byTask && byTask[taskInferred] != null) {
      return String(byTask[taskInferred]);
    }
    // Backwards compatibility (older schemas may expose a single default).
    if (evalDefaults?.metric != null) return String(evalDefaults.metric);
    return null;
  }, [evalDefaults, taskInferred]);

  // For display only (e.g. UI labels). Payloads should use `metric` override.
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
