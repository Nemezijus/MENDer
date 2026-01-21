import {
  Box,
  Card,
  Divider,
  Group,
  Stack,
  Table,
  Text,
  Tooltip,
} from '@mantine/core';
import Plot from 'react-plotly.js';

// Keep labels consistent with ModelSelectionCard / VotingEnsemblePanel.
// Unknown keys fall back gracefully.
const ALGO_LABELS = {
  // classifiers
  logreg: 'Logistic Regression',
  gnb: 'Gaussian Naive Bayes',
  ridge: 'Ridge Classifier',
  sgd: 'SGD Classifier',
  svm: 'SVM (RBF)',
  linsvm: 'Linear SVM',
  knn: 'k-Nearest Neighbors',
  tree: 'Decision Tree',
  forest: 'Random Forest',
  extratrees: 'Extra Trees',
  hgb: 'Hist Gradient Boosting',
  xgboost: 'XGBoost',

  // regressors
  linreg: 'Linear Regression',
  ridgereg: 'Ridge Regression',
  ridgecv: 'Ridge Regression (CV)',
  enet: 'Elastic Net',
  enetcv: 'Elastic Net (CV)',
  lasso: 'Lasso',
  lassocv: 'Lasso (CV)',
  bayridge: 'Bayesian Ridge',
  svr: 'SVR (RBF)',
  linsvr: 'Linear SVR',
  knnreg: 'kNN Regressor',
  treereg: 'Decision Tree Regressor',
  rfreg: 'Random Forest Regressor',
};

// Confusion-matrix-inspired blue ramp (same logic as ConfusionMatrixResults.jsx)
// t in [0..1] => white -> deeper blue
function cmBlue(t) {
  const tt = Math.max(0, Math.min(1, Number(t) || 0));
  const lightness = 100 - 55 * tt; // 100% -> 45%
  return `hsl(210, 80%, ${lightness}%)`;
}

// Smooth heatmap colorscale (white -> blue), similar to confusion matrix ramp
const HEATMAP_COLORSCALE = [
  [0.0, cmBlue(0.0)],
  [0.25, cmBlue(0.25)],
  [0.5, cmBlue(0.5)],
  [0.75, cmBlue(0.75)],
  [1.0, cmBlue(1.0)],
];

function safeNum(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function fmt(x, digits = 3) {
  const n = safeNum(x);
  if (n == null) return '—';
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function prettyEstimatorName(raw) {
  const base = String(raw || '').replace(/_\d+$/, '');
  return titleCase(base);
}

function makeUniqueLabels(labels) {
  const counts = new Map();
  return labels.map((l) => {
    const k = String(l);
    const c = (counts.get(k) || 0) + 1;
    counts.set(k, c);
    return c === 1 ? k : `${k} (${c})`;
  });
}

function normalize01(vals) {
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

function computeBarRange(means, stds, { clamp01 = false } = {}) {
  const lo = [];
  const hi = [];
  for (let i = 0; i < means.length; i++) {
    const m = safeNum(means[i]);
    const s = safeNum(stds[i]);
    if (m == null) continue;
    lo.push(m - (s == null ? 0 : s));
    hi.push(m + (s == null ? 0 : s));
  }

  if (!lo.length || !hi.length) return null;

  let minY = Math.min(...lo);
  let maxY = Math.max(...hi);

  if (clamp01) {
    minY = Math.max(0, minY);
    maxY = Math.min(1, maxY);
  }

  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return null;
  if (minY === maxY) {
    const pad = Math.abs(minY) > 0 ? 0.1 * Math.abs(minY) : 0.1;
    return [minY - pad, maxY + pad];
  }

  const pad = 0.08 * (maxY - minY);
  return [minY - pad, maxY + pad];
}

function niceEstimatorLabel({ name, algo }) {
  const key = String(algo || '').toLowerCase();
  const base = ALGO_LABELS[key] || prettyEstimatorName(name || key);
  return base;
}

function labelsForMatrix(rawLabels, nameToPretty, fallbackPretty) {
  if (Array.isArray(rawLabels) && rawLabels.length) {
    return rawLabels.map((n, i) => nameToPretty.get(n) || fallbackPretty?.[i] || prettyEstimatorName(n));
  }
  return fallbackPretty || [];
}

function SummaryTable({ rows }) {
  return (
    <Table
      withTableBorder={false}
      withColumnBorders={false}
      horizontalSpacing="xs"
      verticalSpacing="xs"
      mt="xs"
    >
      <tbody>
        {rows.map((pair, idx) => (
          <tr key={idx}>
            {pair.map((item) => (
              <td key={item.label} style={{ width: '50%', verticalAlign: 'top' }}>
                <Group gap={6} align="flex-start" wrap="nowrap">
                  <Text size="sm" fw={600}>
                    {item.label}
                  </Text>
                  {item.tooltip ? (
                    <Tooltip label={item.tooltip} withArrow position="top" maw={360}>
                      <Text
                        size="sm"
                        c="dimmed"
                        style={{ cursor: 'help', userSelect: 'none' }}
                      >
                        ⓘ
                      </Text>
                    </Tooltip>
                  ) : null}
                </Group>
                <Text size="sm">{item.value}</Text>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export default function VotingEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  // Build friendly unique labels for estimators.
  const baseLabels = estimators.map((e) =>
    niceEstimatorLabel({ name: e?.name, algo: e?.algo }),
  );
  const namesPretty = makeUniqueLabels(baseLabels);

  const nameToPretty = new Map();
  estimators.forEach((e, i) => nameToPretty.set(e?.name, namesPretty[i]));

  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const hasEstimatorScores =
    means.filter((v) => typeof v === 'number' && Number.isFinite(v)).length > 0;

  const yRange = computeBarRange(means, stds, { clamp01: false });

  // Base estimator bar colors: darker for higher mean score
  const meanT = normalize01(means);
  const barColors = meanT.map((t) => cmBlue(0.25 + 0.75 * t));

  const barTrace = {
    type: 'bar',
    x: namesPretty,
    y: means.map((v) => (v == null ? null : v)),
    error_y: {
      type: 'data',
      array: stds.map((s) => (s == null ? 0 : s)),
      visible: true,
    },
    marker: { color: barColors },
    hovertemplate: `<b>%{x}</b><br>${metricName}: %{y:.4f}<extra></extra>`,
  };

  const similarity = report.similarity || {};
  const corr = Array.isArray(similarity.pairwise_corr) ? similarity.pairwise_corr : null;
  const absdiff = Array.isArray(similarity.pairwise_absdiff)
    ? similarity.pairwise_absdiff
    : null;
  const simLabelsPretty = labelsForMatrix(similarity.labels, nameToPretty, namesPretty);

  const errors = report.errors || {};
  const ensErr = errors.ensemble || {};
  const bestErr = errors.best_base || {};
  const gain = errors.gain_vs_best || {};

  const regMetricItems = [
    {
      label: 'Avg pairwise corr',
      value: fmt(similarity.pairwise_mean_corr, 3),
      tooltip:
        'Average Pearson correlation between base predictions (off-diagonal). Higher = models behave more similarly (lower diversity).',
    },
    {
      label: 'Avg pairwise |Δ|',
      value: fmt(similarity.pairwise_mean_absdiff, 4),
      tooltip:
        'Average absolute difference between base predictions (off-diagonal). Higher = more diversity in predictions.',
    },
    {
      label: 'Mean pred spread',
      value: fmt(similarity.prediction_spread_mean, 4),
      tooltip:
        'Mean per-sample standard deviation across base predictions. Higher = base estimators disagree more.',
    },
    {
      label: 'Ensemble RMSE',
      value: fmt(ensErr.rmse, 4),
      tooltip:
        'Root mean squared error of the ensemble predictions on the pooled evaluation samples.',
    },
    {
      label: 'RMSE reduction',
      value: fmt(gain.rmse_reduction, 4),
      tooltip:
        'Best-base RMSE − ensemble RMSE. Positive = ensemble improved over the best single estimator.',
    },
    {
      label: 'N samples',
      value: fmt(errors.n_total, 0),
      tooltip:
        'Total number of evaluation samples pooled across folds used for the ensemble-specific report.',
    },
  ];

  const regMetricRows = [
    [regMetricItems[0], regMetricItems[1]],
    [regMetricItems[2], regMetricItems[3]],
    [regMetricItems[4], regMetricItems[5]],
  ];

  const corrTrace = corr
    ? {
        type: 'heatmap',
        x: simLabelsPretty,
        y: simLabelsPretty,
        z: corr,
        zmin: -1,
        zmax: 1,
        colorscale: 'RdBu',
        reversescale: true,
        showscale: true,
        colorbar: {
          x: 0.75,
          xanchor: 'right',
          xpad: 0,
          thickness: 12,
          len: 0.92,
          outlinewidth: 0,
        },
        text: corr.map((row) =>
          row.map((v) => (typeof v === 'number' ? v.toFixed(2) : '')),
        ),
        texttemplate: '%{text}',
        hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{z:.3f}<extra></extra>',
      }
    : null;

  const absTrace = (() => {
    if (!absdiff) return null;
    const flat = absdiff
      .flat()
      .map((v) => safeNum(v))
      .filter((v) => v != null);
    const zmax = flat.length ? Math.max(...flat) : 1;
    return {
      type: 'heatmap',
      x: simLabelsPretty,
      y: simLabelsPretty,
      z: absdiff,
      zmin: 0,
      zmax: zmax || 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      colorbar: {
        x: 0.75,
        xanchor: 'right',
        xpad: 0,
        thickness: 12,
        len: 0.92,
        outlinewidth: 0,
      },
      text: absdiff.map((row) =>
        row.map((v) => (typeof v === 'number' ? fmt(v, 3) : '')),
      ),
      texttemplate: '%{text}',
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>|Δ|: %{z:.4f}<extra></extra>',
    };
  })();

  const bestNamePretty =
    nameToPretty.get(bestErr.name) ||
    niceEstimatorLabel({ name: bestErr.name, algo: null });

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          Voting ensemble insights
        </Text>

        <Box>
          <Text fw={500} size="lg" align="center">
            Summary metrics
          </Text>
          <SummaryTable rows={regMetricRows} />
        </Box>

        <Divider />

        <Box>
          <Text fw={500} size="lg" align="center">
            Base estimator performance
          </Text>

          {!hasEstimatorScores ? (
            <Text size="sm" c="dimmed" align="center" mt="xs">
              No base-estimator fold scores were provided.
            </Text>
          ) : (
            <Plot
              data={[barTrace]}
              layout={{
                margin: { l: 40, r: 20, t: 30, b: 70 },
                height: 320,
                yaxis: {
                  title: metricName || 'score',
                  range: yRange || undefined,
                  zeroline: true,
                },
                xaxis: { tickangle: -25 },
                showlegend: false,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          )}
        </Box>

        <Divider />

        <Box>
          <Text fw={500} size="lg" align="center">
            Prediction similarity
          </Text>

          {!corrTrace ? (
            <Text size="sm" c="dimmed" align="center" mt="xs">
              Similarity matrix not available.
            </Text>
          ) : (
            <Plot
              data={[corrTrace]}
              layout={{
                margin: { l: 90, r: 20, t: 20, b: 90 },
                height: 420,
                xaxis: { tickangle: -35 },
                yaxis: { autorange: 'reversed' },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          )}

          {absTrace ? (
            <Box mt="md">
              <Text fw={500} size="md" align="center">
                Absolute prediction differences
              </Text>
              <Plot
                data={[absTrace]}
                layout={{
                  margin: { l: 90, r: 20, t: 20, b: 90 },
                  height: 420,
                  xaxis: { tickangle: -35 },
                  yaxis: { autorange: 'reversed' },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </Box>
          ) : null}
        </Box>

        <Divider />

        <Box>
          <Text fw={500} size="lg" align="center">
            Errors
          </Text>

          <Table
            withTableBorder={false}
            withColumnBorders={false}
            horizontalSpacing="xs"
            verticalSpacing="xs"
            mt="xs"
          >
            <thead>
              <tr>
                <th />
                <th>
                  <Text size="sm" fw={600}>
                    RMSE
                  </Text>
                </th>
                <th>
                  <Text size="sm" fw={600}>
                    MAE
                  </Text>
                </th>
                <th>
                  <Text size="sm" fw={600}>
                    Median AE
                  </Text>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <Text size="sm" fw={600}>
                    Ensemble
                  </Text>
                </td>
                <td>
                  <Text size="sm">{fmt(ensErr.rmse, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(ensErr.mae, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(ensErr.median_ae, 4)}</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text size="sm" fw={600}>
                    Best base ({bestNamePretty || '—'})
                  </Text>
                </td>
                <td>
                  <Text size="sm">{fmt(bestErr.rmse, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(bestErr.mae, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(bestErr.median_ae, 4)}</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text size="sm" fw={600}>
                    Reduction vs best
                  </Text>
                </td>
                <td>
                  <Text size="sm">{fmt(gain.rmse_reduction, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(gain.mae_reduction, 4)}</Text>
                </td>
                <td>
                  <Text size="sm">{fmt(gain.median_ae_reduction, 4)}</Text>
                </td>
              </tr>
            </tbody>
          </Table>

          <Group justify="center" mt="xs">
            <Text size="sm" c="dimmed">
              Best single estimator: <Text span fw={600}>{bestNamePretty || '—'}</Text>
            </Text>
          </Group>
        </Box>
      </Stack>
    </Card>
  );
}
