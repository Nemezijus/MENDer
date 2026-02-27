import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';
import { safeNum, fmt, fmtPct } from '../../../shared/utils/valueFormat.js';

export { safeNum, fmt, fmtPct };

// Confusion-matrix-inspired blue ramp (same idea as ConfusionMatrixResults.jsx)
export function cmBlue(t) {
  const tt = Math.max(0, Math.min(1, Number(t) || 0));
  const lightness = 100 - 55 * tt; // 100% -> 45%
  return `hsl(210, 80%, ${lightness}%)`;
}

export const HEATMAP_COLORSCALE = [
  [0.0, cmBlue(0.0)],
  [0.25, cmBlue(0.25)],
  [0.5, cmBlue(0.5)],
  [0.75, cmBlue(0.75)],
  [1.0, cmBlue(1.0)],
];




export function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

/**
 * Convert raw estimator names into compact, human-friendly labels.
 * Backend values may include suffixes like "logreg_1".
 */
export function prettyEstimatorName(raw, { task = 'classification' } = {}) {
  const base = String(raw || '').replace(/_\d+$/, '');
  const norm = base.toLowerCase().replace(/[^a-z0-9]/g, '');

  // Canonicalize common sklearn class names / aliases to backend algo keys.
  const canon =
    {
      logisticregression: 'logreg',
      logreg: 'logreg',
      svc: 'svm',
      svm: 'svm',
      svr: 'svr',
      linearsvr: 'linsvr',
      decisiontree: task === 'regression' ? 'treereg' : 'tree',
      decisiontreeregressor: 'treereg',
      decisiontreeclassifier: 'tree',
      randomforest: task === 'regression' ? 'rfreg' : 'forest',
      randomforestregressor: 'rfreg',
      randomforestclassifier: 'forest',
      rf: task === 'regression' ? 'rfreg' : 'forest',
      extratrees: task === 'regression' ? 'rfreg' : 'extratrees',
      extratreesclassifier: 'extratrees',
      extratreesregressor: 'rfreg',
      kneighborsclassifier: 'knn',
      kneighborsregressor: 'knnreg',
      knn: task === 'regression' ? 'knnreg' : 'knn',
      gaussiannaivebayes: 'gnb',
      gnb: 'gnb',
      xgboost: 'xgboost',
      xgb: 'xgboost',
      linearregression: 'linreg',
      ridge: task === 'regression' ? 'ridgereg' : 'ridge',
      ridgeregression: 'ridgereg',
      bayesianridge: 'bayridge',
      elasticnet: 'enet',
      lassocv: 'lassocv',
      ridgecv: 'ridgecv',
      elasticnetcv: 'enetcv',
    }[norm] || null;

  if ((task || '').toLowerCase() === 'regression') {
    const abbrev =
      {
        linreg: 'LR',
        ridgereg: 'RR',
        ridgecv: 'RR(CV)',
        enet: 'EN',
        enetcv: 'EN(CV)',
        lasso: 'Lasso',
        lassocv: 'Lasso(CV)',
        bayridge: 'BR',
        svr: 'SVR',
        linsvr: 'LinSVR',
        knnreg: 'kNN',
        treereg: 'DT',
        rfreg: 'RF',
        xgboost: 'XGB',
      }[canon || norm] || null;
    return abbrev || titleCase(base);
  }

  const short =
    {
      logreg: 'LogReg',
      svm: 'SVM',
      tree: 'Tree',
      forest: 'Forest',
      knn: 'kNN',
      gnb: 'Naive Bayes',
      xgboost: 'XGBoost',
    }[canon || norm] || null;

  return short || titleCase(base);
}

export function makeUniqueLabels(labels) {
  const counts = new Map();
  return (labels || []).map((l) => {
    const k = String(l);
    const c = (counts.get(k) || 0) + 1;
    counts.set(k, c);
    return c === 1 ? k : `${k} (${c})`;
  });
}

export function computeBarRange(means, stds) {
  const vals = (means || [])
    .map((m, i) => {
      const mm = safeNum(m);
      if (mm == null) return null;
      const ss = safeNum(stds?.[i]) ?? 0;
      return mm + ss;
    })
    .filter((v) => typeof v === 'number' && Number.isFinite(v));

  if (!vals.length) return null;

  const maxV = Math.max(...vals);
  const pad = Math.max(0.02, maxV * 0.08);
  const upper = maxV <= 1.2 ? Math.min(1, maxV + pad) : maxV + pad;
  return [0, upper];
}

export function normalize01(vals) {
  const nums = (vals || [])
    .map((v) => safeNum(v))
    .filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (!nums.length) return (vals || []).map(() => 0.5);

  const minV = Math.min(...nums);
  const maxV = Math.max(...nums);
  const denom = maxV - minV;

  return (vals || []).map((v) => {
    const n = safeNum(v);
    if (n == null) return 0.5;
    return denom > 0 ? (n - minV) / denom : 0.5;
  });
}

/**
 * Rebin a histogram defined by (edges, counts) onto a new set of equally spaced bins.
 * Counts are distributed by overlap length (assumes uniform density within each old bin).
 */
export function rebinHistogram(edges, counts, newEdges) {
  const oldEdges = (edges || []).map(Number);
  const oldCounts = (counts || []).map(Number);
  const outCounts = new Array(Math.max(0, newEdges.length - 1)).fill(0);

  for (let i = 0; i < oldCounts.length; i++) {
    const a0 = oldEdges[i];
    const a1 = oldEdges[i + 1];
    const c = oldCounts[i];
    if (!Number.isFinite(a0) || !Number.isFinite(a1) || !Number.isFinite(c)) continue;

    const aLen = a1 - a0;
    if (aLen <= 0) continue;

    for (let j = 0; j < outCounts.length; j++) {
      const b0 = newEdges[j];
      const b1 = newEdges[j + 1];
      const overlap = Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
      if (overlap > 0) outCounts[j] += c * (overlap / aLen);
    }
  }

  return outCounts;
}

export function histToBarTrace(edges, counts, opts = {}) {
  const {
    color,
    xLabel,
    hoverLabel,
    isIntegerBins = false,
    hideTickLabels = false,
    xRange = null,
    xTickmode = null,
    xTick0 = null,
    xDtick = null,
  } = opts;

  if (!Array.isArray(edges) || !Array.isArray(counts) || edges.length < 2 || counts.length < 1) {
    return null;
  }
  if (edges.length !== counts.length + 1) return null;

  const e = edges.map((v) => safeNum(v));
  const c = counts.map((v) => safeNum(v));
  if (e.some((v) => v == null) || c.some((v) => v == null)) return null;

  const mids = e.slice(0, -1).map((a, i) => (a + e[i + 1]) / 2);
  const widths = e.slice(0, -1).map((a, i) => (e[i + 1] - a) * 0.9);

  const binW = e.length >= 2 ? e[1] - e[0] : null;

  const layoutX = {
    title: { text: xLabel },
    automargin: true,
    showgrid: false,
    zeroline: false,
    showticklabels: !hideTickLabels,
  };

  if (xRange) layoutX.range = xRange;
  if (xTickmode) layoutX.tickmode = xTickmode;
  if (xTick0 != null) layoutX.tick0 = xTick0;
  if (xDtick != null) layoutX.dtick = xDtick;

  if (isIntegerBins && !xTickmode && Number.isFinite(binW) && binW > 0) {
    layoutX.tickmode = 'linear';
    layoutX.tick0 = mids[0];
    layoutX.dtick = binW;
  }

  return {
    trace: {
      type: 'bar',
      x: mids,
      y: c,
      width: widths,
      marker: { color: color || cmBlue(0.75) },
      hovertemplate: `${hoverLabel}: %{x}<br>count: %{y}<extra></extra>`,
    },
    layoutX,
  };
}

export function inferSquareLabels(n) {
  const m = Math.max(0, Number(n) || 0);
  return Array.from({ length: m }, (_, i) => `est_${i + 1}`);
}

export function niceEstimatorLabel({ name, algo } = {}, { fallback = '—' } = {}) {
  const key = String(algo || '').toLowerCase();
  if (key) return getAlgoLabel(key) || prettyEstimatorName(name || key) || fallback;
  if (name) return prettyEstimatorName(name) || fallback;
  return fallback;
}
