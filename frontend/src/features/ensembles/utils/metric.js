/**
 * Metric helpers.
 *
 * The backend provides allowed metrics per task via schema enums.
 * For regression, we must avoid silently falling back to "accuracy".
 */

/**
 * @param {any} enums
 * @param {string|null|undefined} effectiveTask
 * @returns {string[]}
 */
export function getAllowedMetrics(enums, effectiveTask) {
  if (!enums) return [];

  const metricByTask = enums?.MetricByTask || null;
  if (
    metricByTask &&
    effectiveTask &&
    Array.isArray(metricByTask[effectiveTask])
  ) {
    return metricByTask[effectiveTask].map(String);
  }

  if (Array.isArray(enums?.MetricName)) return enums.MetricName.map(String);
  return [];
}

/**
 * Resolve the metric to send in the payload.
 *
 * Convention:
 * - For classification: omit when unset/invalid (engine defaults apply).
 * - For regression: if unset/invalid and schema provides allowed metrics,
 *   use the schema's first metric as a safe default.
 *
 * @param {object} args
 * @param {string|null|undefined} args.metric
 * @param {string|null|undefined} args.effectiveTask
 * @param {string[]} args.allowedMetrics
 * @returns {string|undefined}
 */
export function resolveMetricForPayload({ metric, effectiveTask, allowedMetrics }) {
  const metricOverride = metric ? String(metric) : undefined;

  const isAllowed =
    !metricOverride ||
    !Array.isArray(allowedMetrics) ||
    allowedMetrics.length === 0 ||
    allowedMetrics.includes(metricOverride);

  const defaultMetricFromSchema =
    Array.isArray(allowedMetrics) && allowedMetrics.length
      ? allowedMetrics[0]
      : undefined;

  if (effectiveTask === 'regression') {
    return isAllowed ? metricOverride ?? defaultMetricFromSchema : defaultMetricFromSchema;
  }

  // classification / unknown task: keep override if allowed, otherwise omit
  return isAllowed ? metricOverride : undefined;
}
