const TITLE_MAP = {
  n_clusters: 'Number of clusters',
  n_noise: 'Noise points',
  noise_ratio: 'Noise ratio',
  cluster_sizes: 'Cluster sizes',
  inertia: 'Inertia',
  n_iter: 'Iterations',
  aic: 'AIC',
  bic: 'BIC',
  mean_log_likelihood: 'Mean log-likelihood',
  std_log_likelihood: 'Std log-likelihood',
  converged: 'Converged',
  lower_bound: 'Lower bound',
  embedding_2d: '2D embedding',
  silhouette: 'Silhouette',
  davies_bouldin: 'Davies–Bouldin',
  calinski_harabasz: 'Calinski–Harabasz',
};

const TOOLTIP_MAP = {
  silhouette:
    'Silhouette score: how well-separated clusters are (higher is better). Only defined when there are at least 2 clusters and no degenerate cases.',
  davies_bouldin:
    'Davies–Bouldin index: average similarity between each cluster and its most similar one (lower is better).',
  calinski_harabasz:
    'Calinski–Harabasz index: ratio of between-cluster dispersion to within-cluster dispersion (higher is better).',
  n_clusters: 'Number of non-noise clusters. (Noise label -1 is excluded.)',
  n_noise: 'Number of points labeled as noise (-1), if the algorithm supports noise (e.g., DBSCAN).',
  noise_ratio: 'Fraction of points labeled as noise (-1).',
  cluster_sizes: 'Sample counts per cluster id.',
  inertia:
    'Sum of squared distances of samples to their closest cluster center (KMeans). Lower usually means tighter clusters.',
  aic: 'Akaike Information Criterion (Gaussian mixture models). Lower is better (relative, not absolute).',
  bic: 'Bayesian Information Criterion (Gaussian mixture models). Lower is better (relative, not absolute).',
  mean_log_likelihood:
    'Mean per-sample log-likelihood under the fitted model (mixture models). Higher is better.',
  std_log_likelihood: 'Standard deviation of per-sample log-likelihoods.',
  converged: 'Whether the estimator reports convergence.',
  lower_bound: 'Lower bound on the log-likelihood (mixture models).',
  n_iter: 'Number of iterations run by the estimator.',
  embedding_2d: 'A downsampled 2D embedding (PCA) for plotting.',
};

const PREVIEW_HEADER_LABELS = {
  index: 'Index',
  cluster_id: 'Cluster id',
  is_noise: 'Noise',
  is_core: 'Core',
  distance_to_center: 'Distance to center',
  max_membership_prob: 'Max membership prob.',
  log_likelihood: 'Log-likelihood',
};

const PREVIEW_HEADER_TOOLTIPS = {
  index: 'Row index within the preview.',
  cluster_id: 'Assigned cluster label for this sample.',
  is_noise: 'Whether this sample was labeled as noise (-1).',
  is_core: 'Whether this sample is a core point (DBSCAN).',
  distance_to_center: 'Distance to the nearest cluster center (KMeans).',
  max_membership_prob: 'Maximum soft membership probability (mixtures).',
  log_likelihood: 'Per-sample log-likelihood under the fitted model (mixtures).',
};

export function titleCaseFromKey(key) {
  if (!key) return '';
  if (Object.prototype.hasOwnProperty.call(TITLE_MAP, key)) return TITLE_MAP[key];
  return String(key)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function tooltipForKey(key) {
  return TOOLTIP_MAP[key] || null;
}

export function headerLabel(key) {
  if (Object.prototype.hasOwnProperty.call(PREVIEW_HEADER_LABELS, key)) {
    return PREVIEW_HEADER_LABELS[key];
  }
  return titleCaseFromKey(key);
}

export function headerTooltip(key) {
  return PREVIEW_HEADER_TOOLTIPS[key] || tooltipForKey(key);
}
