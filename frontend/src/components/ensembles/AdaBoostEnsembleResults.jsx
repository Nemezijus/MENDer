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

// Confusion-matrix-inspired blue ramp
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
    xRange = null,
    hideTickLabels = false,
  } = opts;

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

function fmtIntish(x) {
  const n = safeNum(x);
  if (n == null) return '—';
  return String(Math.round(n));
}

export default function AdaBoostEnsembleResults({ report }) {
  if (!report || report.kind !== 'adaboost') return null;

  const metricName = report.metric_name || '';
  const ada = report.adaboost || {};
  const vote = report.vote || {};
  const weights = report.weights || {};
  const errors = report.errors || {};
  const baseScores = report.base_estimator_scores || {};
  const stages = report.stages || {};

  const baseAlgo = ada.base_algo ? titleCase(ada.base_algo) : '—';
  const nEstimatorsConfigured = safeNum(ada.n_estimators);
  const learningRate = safeNum(ada.learning_rate);
  const algorithm = ada.algorithm ? String(ada.algorithm) : null;

  const nEstimatorsFittedMean = safeNum(stages.n_estimators_fitted_mean);
  const nNontrivialMean = safeNum(stages.n_nontrivial_weights_mean);
  const weightEps = safeNum(stages.weight_eps);

  const topkMean = stages.weight_mass_topk_mean || null; // keys are strings like "5","10","20"
  const top10Mass = topkMean ? safeNum(topkMean['10']) : null;
  const top20Mass = topkMean ? safeNum(topkMean['20']) : null;

  const hideDenseLabels =
    (nEstimatorsConfigured != null && nEstimatorsConfigured > 20) ||
    (nEstimatorsFittedMean != null && nEstimatorsFittedMean > 20);

  const effectiveN = safeNum(weights.effective_n_mean);

  const summaryItems = [
    {
      label: 'Mean margin',
      value: vote.mean_margin == null ? '—' : fmt(vote.mean_margin, 3),
      tooltip: 'Mean weighted vote margin: (top weight − runner-up weight) / total weight.',
    },
    {
      label: 'Mean strength',
      value: vote.mean_strength == null ? '—' : fmtPct(vote.mean_strength, 1),
      tooltip: 'Mean weighted vote strength: top weight / total weight.',
    },
    {
      label: 'Tie rate',
      value: vote.tie_rate == null ? '—' : fmtPct(vote.tie_rate, 1),
      tooltip: 'Fraction of samples where top two labels have equal weight (weak consensus).',
    },
    {
      label: 'Eff. # estimators',
      value: effectiveN == null ? '—' : fmt(effectiveN, 2),
      tooltip:
        'Effective number of estimators (ESS) derived from estimator weights. Lower means a few stages dominate.',
    },
  ];

  const rows = [
    [summaryItems[0], summaryItems[1]],
    [summaryItems[2], summaryItems[3]],
  ];

  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};
  const wHist = weights.hist || null;
  const eHist = errors.hist || null;
  const scoreHist = baseScores.hist || null;

  const marginPlot = histToBarTrace(marginHist.edges, marginHist.counts, {
    color: cmBlue(0.75),
    xLabel: 'Weighted margin',
    hoverLabel: 'margin bin',
    xRange: [0, 1],
  });

  const strengthPlot = histToBarTrace(strengthHist.edges, strengthHist.counts, {
    color: cmBlue(0.75),
    xLabel: 'Weighted strength',
    hoverLabel: 'strength bin',
    xRange: [0, 1],
  });

  const weightPlot =
    wHist && wHist.edges && wHist.counts
      ? histToBarTrace(wHist.edges, wHist.counts, {
          color: cmBlue(0.65),
          xLabel: 'Estimator weight',
          hoverLabel: 'weight bin',
          hideTickLabels: hideDenseLabels,
        })
      : null;

  const errorPlot =
    eHist && eHist.edges && eHist.counts
      ? histToBarTrace(eHist.edges, eHist.counts, {
          color: cmBlue(0.65),
          xLabel: 'Estimator error',
          hoverLabel: 'error bin',
          xRange: [0, 1],
        })
      : null;

  const baseScorePlot =
    scoreHist && scoreHist.edges && scoreHist.counts
      ? histToBarTrace(scoreHist.edges, scoreHist.counts, {
          color: cmBlue(0.7),
          xLabel: `Stage ${metricName || 'score'}`,
          hoverLabel: 'score bin',
          xRange: [0, 1],
        })
      : null;

  const showConcentrationHint =
    effectiveN != null &&
    ((nEstimatorsConfigured != null && effectiveN < 0.25 * nEstimatorsConfigured) ||
      (nEstimatorsFittedMean != null && effectiveN < 0.25 * nEstimatorsFittedMean));

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          AdaBoost ensemble insights
        </Text>

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
              {rows.map((pair, i) => (
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
            <b>{nEstimatorsConfigured == null ? '—' : String(nEstimatorsConfigured)}</b> • Learning rate:{' '}
            <b>{learningRate == null ? '—' : fmt(learningRate, 3)}</b>
            {algorithm ? (
              <>
                {' '}
                • Algorithm: <b>{algorithm}</b>
              </>
            ) : null}
          </Text>

          <Text size="sm" c="dimmed" mt={6} align="center">
            <Tooltip
              label="Configured = requested n_estimators. Fitted = actual number of stage estimators trained. Non-trivial weight counts how many stages have weight greater than ε (default 1e-6)."
              multiline
              maw={520}
              withArrow
            >
              <span>
                Stages: configured <b>{fmtIntish(nEstimatorsConfigured)}</b>
                {nEstimatorsFittedMean != null ? (
                  <>
                    {' '}
                    • fitted <b>{fmtIntish(nEstimatorsFittedMean)}</b>
                  </>
                ) : null}
                {nNontrivialMean != null ? (
                  <>
                    {' '}
                    • non-trivial weight{' '}
                    <b>
                      {fmtIntish(nNontrivialMean)}
                      {weightEps != null ? ` (ε=${weightEps})` : ''}
                    </b>
                  </>
                ) : null}
                {top10Mass != null ? (
                  <>
                    {' '}
                    • top-10 weight mass <b>{fmtPct(top10Mass, 1)}</b>
                  </>
                ) : null}
                {top20Mass != null ? (
                  <>
                    {' '}
                    • top-20 weight mass <b>{fmtPct(top20Mass, 1)}</b>
                  </>
                ) : null}
              </span>
            </Tooltip>
          </Text>

          {showConcentrationHint ? (
            <Text size="sm" c="dimmed" mt={6} align="center">
              Note: weights are highly concentrated (effective # is much smaller than configured). Some histograms may collapse into few bins.
            </Text>
          ) : null}
        </Box>

        <Divider my="xs" />

        <Group align="stretch" grow wrap="wrap">
          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Weighted vote margin distribution. Higher means clearer weighted majorities."
              multiline
              maw={360}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Weighted vote margins
              </Text>
            </Tooltip>

            {marginPlot?.trace ? (
              <Plot
                data={[marginPlot.trace]}
                layout={{
                  autosize: true,
                  height: 260,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: { ...(marginPlot.layoutX || {}) },
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
          </Box>

          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Weighted vote strength distribution: (top weight / total weight)."
              multiline
              maw={360}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Weighted vote strength
              </Text>
            </Tooltip>

            {strengthPlot?.trace ? (
              <Plot
                data={[strengthPlot.trace]}
                layout={{
                  autosize: true,
                  height: 260,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: { ...(strengthPlot.layoutX || {}) },
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
          </Box>
        </Group>

        <Divider my="xs" />

        <Group align="stretch" grow wrap="wrap">
          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Distribution of boosting stage weights. If weights concentrate, effective estimator count decreases."
              multiline
              maw={380}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Estimator weights
              </Text>
            </Tooltip>

            {weightPlot?.trace ? (
              <Plot
                data={[weightPlot.trace]}
                layout={{
                  autosize: true,
                  height: 320,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: { ...(weightPlot.layoutX || {}) },
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
                Weight histogram unavailable.
              </Text>
            )}
          </Box>

          <Box style={{ flex: 1, minWidth: 340 }}>
            <Tooltip
              label="Stage errors (if available). Lower is better; a spread indicates varying weak-learner performance."
              multiline
              maw={380}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Estimator errors
              </Text>
            </Tooltip>

            {errorPlot?.trace ? (
              <Plot
                data={[errorPlot.trace]}
                layout={{
                  autosize: true,
                  height: 320,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: { ...(errorPlot.layoutX || {}) },
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
                Error histogram unavailable (not provided by backend).
              </Text>
            )}
          </Box>
        </Group>

        <Divider my="xs" />

        <Box>
          <Tooltip
            label="Optional distribution of per-stage scores on the evaluation split (if the backend provides it)."
            multiline
            maw={380}
            withArrow
          >
            <Text size="lg" fw={500} align="center" mb={6}>
              Stage score distribution
            </Text>
          </Tooltip>

          {baseScorePlot?.trace ? (
            <Plot
              data={[baseScorePlot.trace]}
              layout={{
                autosize: true,
                height: 260,
                margin: { l: 60, r: 12, t: 10, b: 60 },
                xaxis: { ...(baseScorePlot.layoutX || {}) },
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
              No per-stage score distribution available (backend not provided yet).
            </Text>
          )}
        </Box>
      </Stack>
    </Card>
  );
}
