import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';
import { fmtNumber as sharedFmtNumber } from '../../../shared/utils/numberFormat.js';

export function fmtNumber(val, digits = 4) {
  // Preserve prior model-artifacts behavior (Number() coercion, including booleans/empty strings).
  if (val == null) return '—';
  const n = Number(val);
  if (Number.isNaN(n)) return '—';
  return sharedFmtNumber(n, digits);
}

export function niceUnsupervisedMetricName(name) {
  if (name == null) return 'Silhouette / Davies–Bouldin / Calinski–Harabasz';
  const key = String(name);
  const map = {
    silhouette: 'Silhouette',
    davies_bouldin: 'Davies–Bouldin',
    calinski_harabasz: 'Calinski–Harabasz',
  };
  if (map[key]) return map[key];
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function friendlyClassName(classPath) {
  if (!classPath) return 'unknown';
  if (classPath === 'builtins.str') return 'none';
  const parts = String(classPath).split('.');
  return parts[parts.length - 1] || String(classPath);
}

export function inferEnsembleKindFromClassPath(classPath) {
  if (!classPath) return null;
  const name = friendlyClassName(classPath);

  if (name === 'VotingClassifier' || name === 'VotingRegressor') return 'voting';
  if (name === 'BaggingClassifier' || name === 'BaggingRegressor') return 'bagging';
  if (name === 'AdaBoostClassifier' || name === 'AdaBoostRegressor') return 'adaboost';
  if (name === 'XGBClassifier' || name === 'XGBRegressor') return 'xgboost';

  return null;
}

export function getEnsembleKindLabel(kind) {
  if (!kind) return '';
  const map = {
    voting: 'Voting',
    bagging: 'Bagging',
    adaboost: 'AdaBoost',
    xgboost: 'XGBoost',
  };
  return map[String(kind)] || String(kind);
}

export function inferPrimaryLabel(artifact) {
  const algo = artifact?.model?.algo ?? null;

  // If artifact explicitly records an ensemble, prefer ensemble_kind.
  if (algo === 'ensemble') {
    const kind =
      artifact?.model?.ensemble_kind ??
      artifact?.model?.ensemble?.kind ??
      inferEnsembleKindFromClassPath(
        Array.isArray(artifact?.pipeline) && artifact.pipeline.length
          ? artifact.pipeline[artifact.pipeline.length - 1]?.class_path
          : null,
      ) ??
      'ensemble';

    return { label: getEnsembleKindLabel(kind) || kind, raw: kind, isEnsemble: true, ensembleKind: kind };
  }

  // Normal single model: show algo from config.
  if (algo) return { label: getAlgoLabel(algo), raw: algo, isEnsemble: false, ensembleKind: null };

  // Fall back to pipeline last step class name.
  const lastStep =
    Array.isArray(artifact?.pipeline) && artifact.pipeline.length
      ? artifact.pipeline[artifact.pipeline.length - 1]
      : null;

  const ensembleKind = inferEnsembleKindFromClassPath(lastStep?.class_path);
  if (ensembleKind) {
    return {
      label: getEnsembleKindLabel(ensembleKind) || ensembleKind,
      raw: ensembleKind,
      isEnsemble: true,
      ensembleKind,
    };
  }

  const lastCls = lastStep?.class_path ? friendlyClassName(lastStep.class_path) : null;
  if (lastCls) return { label: lastCls, raw: lastCls, isEnsemble: true, ensembleKind: null };

  return { label: 'unknown', raw: 'unknown', isEnsemble: false, ensembleKind: null };
}

export function computeFeaturesText(artifact) {
  const extraStats =
    artifact && artifact.extra_stats && typeof artifact.extra_stats === 'object' ? artifact.extra_stats : {};

  const featureCfg = artifact?.features ?? null;
  const method = featureCfg?.method || 'none';
  if (method !== 'pca') return method;

  const pcaNFromStats = Object.prototype.hasOwnProperty.call(extraStats, 'pca_n_components')
    ? extraStats.pca_n_components
    : null;
  const pcaNFromCfg = Object.prototype.hasOwnProperty.call(featureCfg || {}, 'pca_n') ? featureCfg.pca_n : null;
  const pcaVarFromCfg = Object.prototype.hasOwnProperty.call(featureCfg || {}, 'pca_var') ? featureCfg.pca_var : null;

  const nComp = pcaNFromStats != null ? pcaNFromStats : pcaNFromCfg;
  if (nComp != null) return `pca (${nComp} components)`;
  if (pcaVarFromCfg != null) return `pca (var=${pcaVarFromCfg})`;
  return 'pca';
}

export function computeCompatibility({ artifact, inspectReport, effectiveTask }) {
  let compatible = true;
  let reason = '';

  if (!artifact || !inspectReport) return { compatible: true, reason: '' };

  const artifactKind = artifact.kind === 'clustering' ? 'unsupervised' : artifact.kind;
  if (artifactKind && effectiveTask && artifactKind !== effectiveTask) {
    // Allow using an unsupervised model even when y is present and the data is classified as supervised.
    const allow = artifactKind === 'unsupervised';
    if (!allow) {
      compatible = false;
      reason = reason || `Task mismatch: model is "${artifactKind}", data is "${effectiveTask}".`;
    }
  }

  if (
    artifact.n_features_in != null &&
    inspectReport.n_features != null &&
    artifact.n_features_in !== inspectReport.n_features
  ) {
    compatible = false;
    const r = `Feature count mismatch: model expects ${artifact.n_features_in}, data has ${inspectReport.n_features}.`;
    reason = reason ? `${reason} ${r}` : r;
  }

  if (
    artifact.kind === 'classification' &&
    Array.isArray(artifact.classes) &&
    Array.isArray(inspectReport.classes) &&
    artifact.classes.length !== inspectReport.classes.length
  ) {
    compatible = false;
    const r = `Number of classes changed: model trained on ${artifact.classes.length}, data has ${inspectReport.classes.length}.`;
    reason = reason ? `${reason} ${r}` : r;
  }

  return { compatible, reason };
}
