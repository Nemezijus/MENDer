/**
 * Shared metric metadata used for lightweight UI hints.
 *
 * NOTE:
 * - This is purely presentation metadata (range + optimization direction).
 * - The backend/engine remains the source of truth for actual scoring behavior.
 */

export const METRIC_META = {
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
    direction: 'higher is better; weighted by how many samples each class has.',
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

  // Regression
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

  // Unsupervised
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

export function getMetricMeta(metricName) {
  if (!metricName) return null;
  return METRIC_META[String(metricName)] ?? null;
}
