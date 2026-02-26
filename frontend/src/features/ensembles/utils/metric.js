/**
 * Metric helpers.
 *
 * The Engine owns metric defaults. The frontend must not invent "safe" defaults.
 * Convention: store + payload are overrides-only.
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
 * Resolve the metric override to send in the payload.
 *
 * Rule: overrides only.
 * - If unset: omit (engine defaults apply).
 * - If set but not in the allowed list: omit (engine defaults apply).
 *
 * @param {object} args
 * @param {string|null|undefined} args.metric
 * @param {string|null|undefined} args.effectiveTask
 * @param {string[]} args.allowedMetrics
 * @returns {string|undefined}
 */
export function resolveMetricForPayload({ metric, effectiveTask, allowedMetrics }) {
  const metricOverride = metric ? String(metric) : undefined;

  if (!metricOverride) return undefined;

  // If schema provides an allowed list, enforce it.
  if (Array.isArray(allowedMetrics) && allowedMetrics.length > 0) {
    return allowedMetrics.includes(metricOverride) ? metricOverride : undefined;
  }

  // No allowed list (schema not loaded) -> keep override if present.
  return metricOverride;
}
