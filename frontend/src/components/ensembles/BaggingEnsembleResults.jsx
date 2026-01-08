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

function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function histToBarTrace(edges, counts, opts = {}) {
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

  // Show a tick for each integer-bin bar if requested (useful for margins)
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

export default function BaggingEnsembleResults({ report }) {
  if (!report || report.kind !== 'bagging') return null;

  const metricName = report.metric_name || '';

  const bag = report.bagging || {};
  const oob = report.oob || {};
  const diversity = report.diversity || {};
  const vote = report.vote || {};
  const baseScores = report.base_estimator_scores || {};

  const baseAlgo = bag.base_algo ? titleCase(bag.base_algo) : '—';
  const nEstimators = safeNum(bag.n_estimators);
  const hideDenseLabels = (nEstimators != null && nEstimators > 20);

  const bootstrap = typeof bag.bootstrap === 'boolean' ? bag.bootstrap : null;
  const bootstrapFeatures =
    typeof bag.bootstrap_features === 'boolean' ? bag.bootstrap_features : null;

  const isBalanced = !!bag.balanced;

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
      label: 'All-agree',
      value:
        diversity.all_agree_rate == null ? '—' : fmtPct(diversity.all_agree_rate, 1),
      tooltip:
        'Fraction of evaluation samples where ALL base estimators predicted the same label.',
    },
    {
      label: 'Avg pairwise agreement',
      value:
        diversity.pairwise_mean_agreement == null
          ? '—'
          : fmtPct(diversity.pairwise_mean_agreement, 1),
      tooltip:
        'Average agreement between estimator pairs. Higher = lower diversity (more redundancy).',
    },
    {
      label: 'Tie rate',
      value: vote.tie_rate == null ? '—' : fmtPct(vote.tie_rate, 1),
      tooltip:
        'Fraction of samples where the vote was tied for the top label (weak consensus).',
    },
    {
      label: 'Mean margin',
      value: vote.mean_margin == null ? '—' : fmt(vote.mean_margin, 3),
      tooltip:
        'Average vote margin: (top vote count − runner-up vote count). Larger = clearer majorities.',
    },
  ];

  const metricRows = [
    [metricItems[0], metricItems[1]],
    [metricItems[2], metricItems[3]],
    [metricItems[4], metricItems[5]],
  ];

  // ------------------------
  // Agreement heatmap
  // ------------------------
  const matrix = Array.isArray(diversity.matrix) ? diversity.matrix : null;
  let labels = Array.isArray(diversity.labels) ? diversity.labels : null;

  // If backend drops labels for large N, generate placeholders
  if (!labels && matrix && Array.isArray(matrix) && Array.isArray(matrix[0])) {
    const m = matrix.length;
    labels = Array.from({ length: m }, (_, i) => `est_${i + 1}`);
  }

  const heatmapTrace =
    matrix && labels
      ? {
          type: 'heatmap',
          x: labels,
          y: labels,
          z: matrix,
          zmin: 0,
          zmax: 1,
          colorscale: HEATMAP_COLORSCALE,
          showscale: true,
          colorbar: {
            // your settings
            x: 1.3,
            xanchor: 'right',
            xpad: 0,
            thickness: 12,
            len: 0.7,
            outlinewidth: 0,
          },
          hovertemplate:
            '<b>%{y}</b> vs <b>%{x}</b><br>agreement: %{z:.3f}<extra></extra>',
        }
      : null;

  // ------------------------
  // Histograms (from backend)
  // ------------------------
  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};
  const scoreHist = baseScores.hist || {};

  const marginPlot = histToBarTrace(marginHist.edges, marginHist.counts, {
    color: cmBlue(0.75),
    xLabel: 'Margin (top − runner-up)',
    hoverLabel: 'margin bin',
    isIntegerBins: true,
    hideTickLabels: hideDenseLabels, // drop x tick labels if > 20 estimators
  });

  const strengthPlot = histToBarTrace(strengthHist.edges, strengthHist.counts, {
    color: cmBlue(0.75),
    xLabel: 'Strength (top / total)',
    hoverLabel: 'strength bin',
    xRange: [0, 1],
    xTickmode: 'linear',
    xTick0: 0,
    xDtick: 0.1,
  });

  const scorePlot = histToBarTrace(scoreHist.edges, scoreHist.counts, {
    color: cmBlue(0.7),
    xLabel: `Base-estimator ${metricName || 'score'}`,
    hoverLabel: 'score bin',
    xRange: [0, 1],
  });

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
            Base estimator: <b>{baseAlgo}</b> • Estimators:{' '}
            <b>{nEstimators == null ? '—' : String(nEstimators)}</b> • Bootstrap:{' '}
            <b>{bootstrap == null ? '—' : bootstrap ? 'on' : 'off'}</b> • Feature
            bootstrap:{' '}
            <b>
              {bootstrapFeatures == null ? '—' : bootstrapFeatures ? 'on' : 'off'}
            </b>
            {isBalanced ? (
              <>
                {' '}
                • <b>Balanced</b>
              </>
            ) : null}
          </Text>
        </Box>

        <Divider my="xs" />

        <Group align="stretch" grow wrap="wrap">
          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Agreement between bagged estimators (0–1). High agreement means redundancy; lower agreement indicates diversity."
              multiline
              maw={380}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Estimator agreement
              </Text>
            </Tooltip>

            {heatmapTrace ? (
              <Plot
                data={[heatmapTrace]}
                layout={{
                  autosize: true,
                  height: 460,
                  margin: { l: 50, r: 10, t: 10, b: 50 },
                  xaxis: {
                    title: { text: hideDenseLabels ? '' : 'Estimator' },
                    tickangle: -30,
                    side: 'top',
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                    constrain: 'domain',
                    showticklabels: !hideDenseLabels,
                  },
                  yaxis: {
                    title: { text: hideDenseLabels ? '' : 'Estimator' },
                    autorange: 'reversed',
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                    scaleanchor: 'x',
                    scaleratio: 1,
                    constrain: 'domain',
                    showticklabels: !hideDenseLabels,
                  },
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: '#ffffff',
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : (
              <Text size="sm" c="dimmed" align="center">
                Agreement matrix unavailable.
              </Text>
            )}
          </Box>

          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Distribution of base-estimator scores (computed per fold on the evaluation split)."
              multiline
              maw={380}
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
                  height: 460,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    ...(scorePlot.layoutX || {}),
                  },
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
        </Group>

        <Divider my="xs" />

        <Group align="stretch" grow wrap="wrap">
          <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Distribution of vote margins (top votes − runner-up votes)."
              multiline
              maw={360}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Vote margins
              </Text>
            </Tooltip>

            {marginPlot?.trace ? (
              <Plot
                data={[marginPlot.trace]}
                layout={{
                  autosize: true,
                  height: 240,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    ...(marginPlot.layoutX || {}),
                  },
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
                Margin histogram unavailable.
              </Text>
            )}
          </Card>

          <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Distribution of vote strength (top votes / total estimators). With N estimators, strengths are typically discrete: k/N."
              multiline
              maw={380}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Vote strength
              </Text>
            </Tooltip>

            {strengthPlot?.trace ? (
              <Plot
                data={[strengthPlot.trace]}
                layout={{
                  autosize: true,
                  height: 240,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    ...(strengthPlot.layoutX || {}),
                  },
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
                Strength histogram unavailable.
              </Text>
            )}
          </Card>
        </Group>
      </Stack>
    </Card>
  );
}
