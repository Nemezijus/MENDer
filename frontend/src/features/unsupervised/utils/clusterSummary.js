/**
 * Parse cluster_sizes from various backend-friendly shapes.
 */
export function parseClusterSizes(v) {
  if (v == null) return [];

  // dict-like: {"0": 12, "-1": 4}
  if (typeof v === 'object' && !Array.isArray(v)) {
    return Object.entries(v).map(([k, count]) => ({ cluster_id: k, size: count }));
  }

  // array of pairs: [[0, 12], [-1, 4]]
  if (Array.isArray(v) && v.length && Array.isArray(v[0]) && v[0].length >= 2) {
    return v.map(([cid, count]) => ({ cluster_id: cid, size: count }));
  }

  // array of counts: [12, 9, 7] -> ids 0..n-1
  if (Array.isArray(v)) {
    return v.map((count, i) => ({ cluster_id: i, size: count }));
  }

  return [];
}

function sortPairsByOrder(pairs, order) {
  const ord = Array.isArray(order) ? order : [];
  return [...pairs].sort((a, b) => {
    const ia = ord.indexOf(a[0]);
    const ib = ord.indexOf(b[0]);
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    return String(a[0]).localeCompare(String(b[0]));
  });
}

/**
 * Generic extraction of non-null key/value pairs.
 */
export function buildPairs(obj, { exclude = [], order = [] } = {}) {
  const out = [];
  if (!obj || typeof obj !== 'object') return out;

  Object.entries(obj).forEach(([k, v]) => {
    if (v === null || v === undefined) return;
    if (exclude.includes(k)) return;
    out.push([k, v]);
  });

  return sortPairsByOrder(out, order);
}

export function buildClusterSummaryPairs(clusterSummary) {
  return buildPairs(clusterSummary, {
    exclude: ['label_summary', 'cluster_sizes'],
    order: ['n_clusters', 'n_noise', 'noise_ratio', 'cluster_sizes'],
  });
}

export function buildModelDiagnosticsPairs(modelDiag, embedding2d) {
  const pairs = buildPairs(modelDiag, {
    exclude: ['label_summary'],
    order: [
      'inertia',
      'n_iter',
      'converged',
      'lower_bound',
      'aic',
      'bic',
      'mean_log_likelihood',
      'std_log_likelihood',
    ],
  });

  if (embedding2d) {
    pairs.push([
      'embedding_2d',
      Array.isArray(embedding2d?.x) ? `${embedding2d.x.length} points` : 'Present',
    ]);
  }

  return sortPairsByOrder(pairs, [
    'inertia',
    'n_iter',
    'converged',
    'lower_bound',
    'aic',
    'bic',
    'mean_log_likelihood',
    'std_log_likelihood',
    'embedding_2d',
  ]);
}

export function buildMetricPairs(metrics) {
  return buildPairs(metrics, {
    order: ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
  });
}
