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

const ALGO_ABBREV = {
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

function computeBarRange(means, stds) {
  const vals = means
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

  // if score is in [0,1], keep it tidy
  const upper = maxV <= 1.2 ? Math.min(1, maxV + pad) : maxV + pad;

  return [0, upper];
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

function niceEstimatorLabel({ name, algo }) {
  const key = String(algo || '').toLowerCase();
  return ALGO_LABELS[key] || prettyEstimatorName(name || key);
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

function algoKeyToAbbrev(algoKey) {
  if (!algoKey) return 'UNK';
  const key = String(algoKey).toLowerCase();
  return ALGO_ABBREV[key] || ALGO_LABELS[key] || key;
}

function buildLegendLines(estimators) {
  const seen = new Map(); // abbrev -> full label
  (estimators || []).forEach((e) => {
    const key = String(e?.algo || '').toLowerCase();
    if (!key) return;
    const ab = algoKeyToAbbrev(key);
    const full = ALGO_LABELS[key] || key;
    if (!seen.has(ab)) seen.set(ab, full);
  });

  // Stable order: by full label
  const entries = Array.from(seen.entries())
    .map(([ab, full]) => ({ ab, full }))
    .sort((a, b) => a.full.localeCompare(b.full));

  return entries;
}

export default function VotingEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  // friendly unique labels for estimators (long) + abbreviated labels for matrix ticks
  const baseLabelsLong = estimators.map((e) =>
    niceEstimatorLabel({ name: e?.name, algo: e?.algo }),
  );
  const namesPretty = makeUniqueLabels(baseLabelsLong);

  // abbreviations aligned to estimators (used for matrix tick labels)
  const baseAbbrev = estimators.map((e) => algoKeyToAbbrev(e?.algo));
  const matrixLabels = makeUniqueLabels(baseAbbrev);

  const nameToPretty = new Map();
  estimators.forEach((e, i) => nameToPretty.set(e?.name, namesPretty[i]));

  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const hasEstimatorScores =
    means.filter((v) => typeof v === 'number' && Number.isFinite(v)).length > 0;

  // Base estimator bar colors: darker for higher mean score (all in same blue family)
  const meanT = normalize01(means);
  const barColors = meanT.map((t) => cmBlue(0.25 + 0.75 * t));

  const yRange = computeBarRange(means, stds);

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
    hovertemplate: `<b>%{x}</b><br>${metricName || 'score'}: %{y:.4f}<extra></extra>`,
  };

  const similarity = report.similarity || {};
  const corrRaw = Array.isArray(similarity.pairwise_corr) ? similarity.pairwise_corr : null;
  const absRaw = Array.isArray(similarity.pairwise_absdiff) ? similarity.pairwise_absdiff : null;

  const labelsPretty = matrixLabels;
  const legendEntries = buildLegendLines(estimators);

  // ---- Summary metrics table (same structure as classification: 3 rows, 2 pairs each) ----
  const errors = report.errors || {};
  const ensErr = errors.ensemble || {};
  const bestErr = errors.best_base || {};
  const gain = errors.gain_vs_best || {};

  const metricItems = [
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

  const metricRows = [
    [metricItems[0], metricItems[1]],
    [metricItems[2], metricItems[3]],
    [metricItems[4], metricItems[5]],
  ];

  // ---- Matrices: match "Pairwise agreement" look, but show two side-by-side, squared ----
  // Correlation uses the same blue ramp by mapping [-1,1] -> [0,1] for colors,
  // while keeping text/hover as the original correlation.
  const corrTrace = (() => {
    if (!corrRaw) return null;
    const zColor = corrRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        if (n == null) return null;
        return (n + 1) / 2;
      }),
    );

    const text = corrRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        return n == null ? '' : n.toFixed(2);
      }),
    );

    return {
      type: 'heatmap',
      x: labelsPretty,
      y: labelsPretty,
      z: zColor,
      zmin: 0,
      zmax: 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      colorbar: {
        x: 1.02,
        xanchor: 'left',
        xpad: 0,
        thickness: 12,
        len: 0.92,
        outlinewidth: 0,
        tickvals: [0, 0.5, 1],
        ticktext: ['-1', '0', '1'],
      },
      text,
      texttemplate: '%{text}',
      hovertemplate:
        '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{text}<extra></extra>',
    };
  })();

  const absTrace = (() => {
    if (!absRaw) return null;
    const flat = absRaw
      .flat()
      .map((v) => safeNum(v))
      .filter((v) => v != null);
    const zmax = flat.length ? Math.max(...flat) : 1;

    const text = absRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        return n == null ? '' : n.toFixed(2);
      }),
    );

    return {
      type: 'heatmap',
      x: labelsPretty,
      y: labelsPretty,
      z: absRaw,
      zmin: 0,
      zmax: zmax || 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      colorbar: {
        x: 1.02,
        xanchor: 'left',
        xpad: 0,
        thickness: 12,
        len: 0.92,
        outlinewidth: 0,
      },
      text,
      texttemplate: '%{text}',
      hovertemplate:
        '<b>%{y}</b> vs <b>%{x}</b><br>|Δ|: %{z:.4f}<extra></extra>',
    };
  })();

  // ---- Errors table (same visual style as Summary metrics) ----
  const bestNamePretty = bestErr?.name
    ? (nameToPretty.get(bestErr.name) || niceEstimatorLabel({ name: bestErr.name, algo: bestErr.algo }))
    : '—';

  const errorItems = [
    {
      label: 'Ensemble RMSE',
      value: fmt(ensErr.rmse, 4),
      tooltip: 'Root mean squared error of ensemble predictions.',
    },
    {
      label: 'Ensemble MAE',
      value: fmt(ensErr.mae, 4),
      tooltip: 'Mean absolute error of ensemble predictions.',
    },
    {
      label: 'Best base RMSE',
      value: fmt(bestErr.rmse, 4),
      tooltip: `RMSE of the best single estimator (${bestNamePretty}).`,
    },
    {
      label: 'Best base MAE',
      value: fmt(bestErr.mae, 4),
      tooltip: `MAE of the best single estimator (${bestNamePretty}).`,
    },
    {
      label: 'RMSE reduction',
      value: fmt(gain.rmse_reduction, 4),
      tooltip: 'Best-base RMSE − ensemble RMSE. Positive = improvement.',
    },
    {
      label: 'MAE reduction',
      value: fmt(gain.mae_reduction, 4),
      tooltip: 'Best-base MAE − ensemble MAE. Positive = improvement.',
    },
  ];

  const errorRows = [
    [errorItems[0], errorItems[1]],
    [errorItems[2], errorItems[3]],
    [errorItems[4], errorItems[5]],
  ];

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          Voting ensemble insights
        </Text>

        {/* Summary metrics */}
        <Box>
          <Text fw={500} size="lg" align="center">
            Summary metrics
          </Text>

          <Table
            withTableBorder={false}
            withColumnBorders={false}
            horizontalSpacing="xs"
            verticalSpacing="xs"
            mt="xs"
          >
            <Table.Tbody>
              {metricRows.map((pair, i) => (
                <Table.Tr
                  key={i}
                  style={{
                    backgroundColor:
                      i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
                  }}
                >
                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[0].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[0].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[0].value}
                    </Text>
                  </Table.Td>

                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[1].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[1].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[1].value}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>

          <Text size="sm" c="dimmed" mt="xs" align="center">
            Voting: <b>{report.voting}</b> • Estimators: <b>{report.n_estimators}</b>
          </Text>
        </Box>

        <Divider my="xs" />

        {/* Base estimators */}
        <Box>
          <Tooltip
            label="Per-estimator performance across folds (mean ± std). Helps you spot strong/weak and stable/unstable base models."
            multiline
            maw={360}
            withArrow
          >
            <Text size="lg" fw={500} align="center" mb={6}>
              Base estimators ({metricName || 'score'})
            </Text>
          </Tooltip>

          {!hasEstimatorScores ? (
            <Text size="sm" c="dimmed" align="center" mt="xs">
              Base estimator scores unavailable.
            </Text>
          ) : (
            <Box style={{ maxWidth: 560, margin: '0 auto' }}>
              <Plot
                data={[barTrace]}
                layout={{
                  autosize: true,
                  height: 300,
                  margin: { l: 70, r: 18, t: 10, b: 90 },
                  xaxis: {
                    tickangle: -25,
                    title: { text: 'Estimator' },
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  yaxis: {
                    title: { text: metricName || 'score' },
                    range: yRange || undefined,
                    automargin: true,
                    showgrid: true,
                    zeroline: false,
                  },
                  bargap: 0.25,
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: '#ffffff',
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </Box>
          )}
        </Box>

        <Divider my="xs" />

        {/* Matrices */}
        <Box>
          <Tooltip
            label="Pairwise relationships between base estimators. Correlation close to 1 means very similar predictions; |Δ| highlights how far predictions differ on average."
            multiline
            maw={420}
            withArrow
          >
            <Text size="lg" fw={500} align="center" mb={6}>
              Pairwise prediction structure
            </Text>
          </Tooltip>

          <Group align="stretch" grow wrap="wrap">
            <Box style={{ flex: 1, minWidth: 320 }}>
              <Text size="md" fw={500} align="center" mb={6}>
                Prediction similarity (corr)
              </Text>

              {corrTrace ? (
                <Plot
                  data={[corrTrace]}
                  layout={{
                    autosize: true,
                    height: 360,
                    margin: { l: 80, r: 36, t: 10, b: 90 },
                    xaxis: {
                      title: { text: 'Estimator' },
                      tickangle: -30,
                      side: 'top',
                      automargin: true,
                      showgrid: false,
                      zeroline: false,
                      constrain: 'domain',
                    },
                    yaxis: {
                      title: { text: 'Estimator' },
                      autorange: 'reversed',
                      automargin: true,
                      showgrid: false,
                      zeroline: false,
                      scaleanchor: 'x',
                      scaleratio: 1,
                      constrain: 'domain',
                    },
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#ffffff',
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : (
                <Text size="sm" c="dimmed" align="center">
                  Correlation matrix unavailable.
                </Text>
              )}
            </Box>

            <Box style={{ flex: 1, minWidth: 320 }}>
              <Text size="md" fw={500} align="center" mb={6}>
                Absolute prediction differences (|Δ|)
              </Text>

              {absTrace ? (
                <Plot
                  data={[absTrace]}
                  layout={{
                    autosize: true,
                    height: 360,
                    margin: { l: 80, r: 36, t: 10, b: 90 },
                    xaxis: {
                      title: { text: 'Estimator' },
                      tickangle: -30,
                      side: 'top',
                      automargin: true,
                      showgrid: false,
                      zeroline: false,
                      constrain: 'domain',
                    },
                    yaxis: {
                      title: { text: 'Estimator' },
                      autorange: 'reversed',
                      automargin: true,
                      showgrid: false,
                      zeroline: false,
                      scaleanchor: 'x',
                      scaleratio: 1,
                      constrain: 'domain',
                    },
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#ffffff',
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : (
                <Text size="sm" c="dimmed" align="center">
                  Absolute-difference matrix unavailable.
                </Text>
              )}
            </Box>
          </Group>
          {legendEntries.length > 0 && (
            <Box mt="xs">
              <Text size="sm" c="dimmed" align="center">
                {legendEntries.map((e, idx) => (
                  <span key={e.ab}>
                    <b>{e.ab}</b> — {e.full}
                    {idx < legendEntries.length - 1 ? '   •   ' : ''}
                  </span>
                ))}
              </Text>
            </Box>
          )}
        </Box>

        <Divider my="xs" />

        {/* Errors */}
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
            <Table.Tbody>
              {errorRows.map((pair, i) => (
                <Table.Tr
                  key={i}
                  style={{
                    backgroundColor:
                      i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
                  }}
                >
                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[0].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[0].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[0].value}
                    </Text>
                  </Table.Td>

                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[1].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[1].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[1].value}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>

          <Text size="sm" c="dimmed" mt="xs" align="center">
            Best single estimator: <b>{bestNamePretty}</b>
          </Text>
        </Box>
      </Stack>
    </Card>
  );
}
