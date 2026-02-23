import { Box, Card, Divider, Group, Stack, Table, Text, Tooltip } from '@mantine/core';
import Plot from 'react-plotly.js';

// Confusion-matrix-inspired blue ramp (same idea as ConfusionMatrixResults.jsx)
function cmBlue(t) {
  const tt = Math.max(0, Math.min(1, Number(t) || 0));
  const lightness = 100 - 55 * tt; // 100% -> 45%
  return `hsl(210, 80%, ${lightness}%)`;
}

const HEATMAP_COLORSCALE = [
  [0.0, cmBlue(0.0)],
  [0.25, cmBlue(0.25)],
  [0.5, cmBlue(0.5)],
  [0.75, cmBlue(0.75)],
  [1.0, cmBlue(1.0)],
];

function safeNum(x) {
  if (x === null || x === undefined || x === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function fmtPct(x, digits = 1) {
  const n = safeNum(x);
  if (n == null) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}

function fmt(x, digits = 3) {
  const n = safeNum(x);
  if (n == null) return '—';
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

function histToBarTrace(edges, counts, opts = {}) {
  const { color, xLabel, hoverLabel, xRange = null, hideTickLabels = false } = opts;

  if (!Array.isArray(edges) || !Array.isArray(counts) || edges.length < 2) return null;
  if (edges.length !== counts.length + 1) return null;

  const e = edges.map((v) => safeNum(v));
  const c = counts.map((v) => safeNum(v));
  if (e.some((v) => v == null) || c.some((v) => v == null)) return null;

  const mids = e.slice(0, -1).map((a, i) => (a + e[i + 1]) / 2);
  const widths = e.slice(0, -1).map((a, i) => (e[i + 1] - a) * 0.9);

  const layoutX = {
    title: { text: xLabel },
    automargin: true,
    showgrid: false,
    zeroline: false,
    showticklabels: !hideTickLabels,
  };
  if (xRange) layoutX.range = xRange;

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

function inferSquareLabels(n) {
  const m = Math.max(0, Number(n) || 0);
  return Array.from({ length: m }, (_, i) => `est_${i + 1}`);
}

export default function BaggingEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const metricName = report.metric_name || '';

  const bag = report.bagging || {};
  const oob = report.oob || {};
  const sim = report.similarity || report.diversity || {};
  const errors = report.errors || {};
  const baseScores = report.base_estimator_scores || {};

  const nEstimators = safeNum(bag.n_estimators);
  const bootstrap = typeof bag.bootstrap === 'boolean' ? bag.bootstrap : null;
  const bootstrapFeatures =
    typeof bag.bootstrap_features === 'boolean' ? bag.bootstrap_features : null;

  // ------------------------
  // Summary metrics
  // ------------------------
  const metricItems = [
    {
      label: 'OOB score',
      value: oob.score == null ? '—' : fmt(oob.score, 4),
      tooltip:
        'Out-of-bag (OOB) score: estimated generalization performance using only samples not used to train each bootstrap estimator (requires oob_score=True).',
    },
    {
      label: 'OOB coverage',
      value: oob.coverage_rate == null ? '—' : fmtPct(oob.coverage_rate, 1),
      tooltip:
        'Fraction of training samples that received an OOB prediction. Low coverage can make OOB score noisy.',
    },
    {
      label: 'Avg pairwise corr',
      value: sim.pairwise_mean_corr == null ? '—' : fmt(sim.pairwise_mean_corr, 3),
      tooltip:
        'Average Pearson correlation between base predictions (off-diagonal). Higher = more similarity (lower diversity).',
    },
    {
      label: 'Avg pairwise |Δ|',
      value:
        sim.pairwise_mean_absdiff == null ? '—' : fmt(sim.pairwise_mean_absdiff, 4),
      tooltip:
        'Average absolute difference between base predictions (off-diagonal). Higher = more diversity.',
    },
    {
      label: 'Mean pred spread',
      value:
        sim.prediction_spread_mean == null ? '—' : fmt(sim.prediction_spread_mean, 4),
      tooltip:
        'Mean per-sample standard deviation across base predictions. Higher = more disagreement.',
    },
    {
      label: 'N samples',
      value: errors.n_total == null ? '—' : fmt(errors.n_total, 0),
      tooltip:
        'Total number of evaluation samples pooled across folds used for bagging-specific reporting.',
    },
  ];

  const metricRows = [
    [metricItems[0], metricItems[1]],
    [metricItems[2], metricItems[3]],
    [metricItems[4], metricItems[5]],
  ];

  // ------------------------
  // Matrices (corr + absdiff)
  // ------------------------
  const corrRaw =
    Array.isArray(sim.pairwise_corr)
      ? sim.pairwise_corr
      : Array.isArray(sim.corr_matrix)
      ? sim.corr_matrix
      : null;

  const absRaw =
    Array.isArray(sim.pairwise_absdiff)
      ? sim.pairwise_absdiff
      : Array.isArray(sim.absdiff_matrix)
      ? sim.absdiff_matrix
      : null;

  const labelsRaw =
    Array.isArray(sim.labels)
      ? sim.labels
      : Array.isArray(sim.estimator_labels)
      ? sim.estimator_labels
      : null;

  const sizeFromMatrix =
    corrRaw && Array.isArray(corrRaw) && Array.isArray(corrRaw[0]) ? corrRaw.length : null;

  const labels = labelsRaw || inferSquareLabels(sizeFromMatrix || nEstimators || 0);
  const showMatrixText = Array.isArray(labels) ? labels.length < 10 : true;

  const corrTrace = (() => {
    if (!corrRaw || !labels?.length) return null;
    const corrVals = corrRaw;

    // map [-1,1] -> [0,1] for colors, but keep text as correlation
    const zColor = corrRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        if (n == null) return null;
        return (n + 1) / 2;
      }),
    );

    const text = showMatrixText
      ? corrRaw.map((row) =>
          row.map((v) => {
            const n = safeNum(v);
            return n == null ? '' : n.toFixed(2);
          }),
        )
      : undefined;

    return {
      type: 'heatmap',
      x: labels,
      y: labels,
      z: zColor,
      zmin: 0,
      zmax: 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      customdata: corrVals,
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
      text: showMatrixText ? text : undefined,
      texttemplate: showMatrixText ? '%{text}' : undefined,
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{customdata:.2f}<extra></extra>',
    };
  })();

  const absTrace = (() => {
    if (!absRaw || !labels?.length) return null;
    const flat = absRaw
      .flat()
      .map((v) => safeNum(v))
      .filter((v) => v != null);
    const zmax = flat.length ? Math.max(...flat) : 1;

    const text = showMatrixText
      ? absRaw.map((row) =>
          row.map((v) => {
            const n = safeNum(v);
            return n == null ? '' : n.toFixed(2);
          }),
        )
      : undefined;

    return {
      type: 'heatmap',
      x: labels,
      y: labels,
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
      texttemplate: showMatrixText ? '%{text}' : undefined,
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>|Δ|: %{z:.4f}<extra></extra>',
    };
  })();

  // ------------------------
  // Base estimator score histogram
  // ------------------------
  const scoreHist = baseScores.hist || {};
  const scorePlot =
    scoreHist && scoreHist.edges && scoreHist.counts
      ? histToBarTrace(scoreHist.edges, scoreHist.counts, {
          color: cmBlue(0.7),
          xLabel: `Base-estimator ${metricName || 'score'}`,
          hoverLabel: 'score bin',
        })
      : null;

  // ------------------------
  // Errors table (optional)
  // ------------------------
  const ensErr = errors.ensemble || {};
  const errItems = [
    {
      label: 'Ensemble RMSE',
      value: ensErr.rmse == null ? '—' : fmt(ensErr.rmse, 4),
      tooltip: 'Root mean squared error of the bagging ensemble predictions.',
    },
    {
      label: 'Ensemble MAE',
      value: ensErr.mae == null ? '—' : fmt(ensErr.mae, 4),
      tooltip: 'Mean absolute error of the bagging ensemble predictions.',
    },
    {
      label: 'Ensemble R²',
      value: ensErr.r2 == null ? '—' : fmt(ensErr.r2, 4),
      tooltip: 'Coefficient of determination (R²) for the bagging ensemble.',
    },
    {
      label: '—',
      value: '—',
      tooltip: '',
    },
  ];

  const errRows = [
    [errItems[0], errItems[1]],
    [errItems[2], errItems[3]],
  ];

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          Bagging ensemble insights
        </Text>

        {/* Summary metrics */}
        <Box>
          <Text fw={500} size="lg" align="center">
            Summary metrics
          </Text>

          <Table withTableBorder={false} withColumnBorders={false} horizontalSpacing="xs" verticalSpacing="xs" mt="xs">
            <Table.Tbody>
              {metricRows.map((pair, i) => (
                <Table.Tr
                  key={i}
                  style={{
                    backgroundColor: i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
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
            Estimators: <b>{nEstimators == null ? '—' : String(nEstimators)}</b> • Bootstrap:{' '}
            <b>{bootstrap == null ? '—' : bootstrap ? 'on' : 'off'}</b> • Feature bootstrap:{' '}
            <b>{bootstrapFeatures == null ? '—' : bootstrapFeatures ? 'on' : 'off'}</b>
          </Text>
        </Box>

        <Divider my="xs" />

        {/* Pairwise matrices */}
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
        </Box>

        <Divider my="xs" />

        {/* Score histogram */}
        <Box>
          <Tooltip
            label="Distribution of base-estimator scores (computed per fold on the evaluation split), if the backend provides it."
            multiline
            maw={420}
            withArrow
          >
            <Text size="lg" fw={500} align="center" mb={6}>
              Base-estimator score distribution
            </Text>
          </Tooltip>

          {scorePlot?.trace ? (
            <Plot
              data={[scorePlot.trace]}
              layout={{
                autosize: true,
                height: 300,
                margin: { l: 60, r: 12, t: 10, b: 60 },
                xaxis: { ...(scorePlot.layoutX || {}) },
                yaxis: {
                  title: { text: 'Count' },
                  automargin: true,
                  showgrid: true,
                  zeroline: false,
                },
                bargap: 0.05,
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          ) : (
            <Text size="sm" c="dimmed" align="center">
              Base-estimator score histogram unavailable.
            </Text>
          )}
        </Box>

        <Divider my="xs" />

        {/* Errors */}
        <Box>
          <Text fw={500} size="lg" align="center">
            Errors
          </Text>

          <Table withTableBorder={false} withColumnBorders={false} horizontalSpacing="xs" verticalSpacing="xs" mt="xs">
            <Table.Tbody>
              {errRows.map((pair, i) => (
                <Table.Tr
                  key={i}
                  style={{
                    backgroundColor: i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
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
                    {pair[1].tooltip ? (
                      <Tooltip label={pair[1].tooltip} multiline maw={360} withArrow>
                        <Text size="sm" fw={600}>
                          {pair[1].label}
                        </Text>
                      </Tooltip>
                    ) : (
                      <Text size="sm" fw={600}>
                        {pair[1].label}
                      </Text>
                    )}
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
        </Box>
      </Stack>
    </Card>
  );
}
