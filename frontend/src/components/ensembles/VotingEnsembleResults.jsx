// frontend/src/components/ensembles/VotingEnsembleResults.jsx
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

// Confusion-matrix-inspired blue scale (white → blue)
const BLUE_MAIN = 'hsl(210, 80%, 45%)';
const BLUE_DARK = 'hsl(210, 80%, 35%)';
const HEATMAP_COLORSCALE = [
  [0.0, '#ffffff'],
  [1.0, BLUE_MAIN],
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

function prettyEstimatorName(raw) {
  const base = String(raw || '').replace(/_\d+$/, '');
  const key = base.toLowerCase();

  const map = {
    logreg: 'LogReg',
    logisticregression: 'LogReg',
    svm: 'SVM',
    svc: 'SVM',
    tree: 'Tree',
    decisiontree: 'Tree',
    forest: 'Forest',
    randomforest: 'Forest',
    rf: 'Forest',
    knn: 'kNN',
    naive_bayes: 'Naive Bayes',
    nb: 'Naive Bayes',
    xgboost: 'XGBoost',
  };

  return map[key] || titleCase(base);
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

  // most classification metrics are in [0,1]; keep it tidy if that's the case
  const upper = maxV <= 1.2 ? Math.min(1, maxV + pad) : maxV + pad;

  return [0, upper];
}

export default function VotingEnsembleResults({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  const namesRaw = estimators.map((e) => e?.name ?? '');
  const namesPretty = makeUniqueLabels(namesRaw.map((n) => prettyEstimatorName(n)));

  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const agreement = report.agreement || {};
  const matrix = Array.isArray(agreement.matrix) ? agreement.matrix : null;

  const labelsRaw = Array.isArray(agreement.labels)
    ? agreement.labels
    : namesRaw;

  const labelsPretty = makeUniqueLabels(labelsRaw.map((n) => prettyEstimatorName(n)));

  const vote = report.vote || {};
  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};

  const change = report.change_vs_best || {};
  const bestNamePretty = prettyEstimatorName(change.best_name || '');

  // ------------------------
  // Summary metrics “table”
  // ------------------------
  const metricItems = [
    {
      label: 'All-agree',
      value: fmtPct(agreement.all_agree_rate),
      tooltip:
        'Fraction of samples where ALL base estimators predicted the same label. Range 0–100%. Higher = more consensus (not necessarily higher accuracy).',
    },
    {
      label: 'Avg pairwise agreement',
      value: fmtPct(agreement.pairwise_mean_agreement),
      tooltip:
        'Average agreement between all estimator pairs. Range 0–100%. Higher = models behave more similarly (lower diversity).',
    },
    {
      label: 'Tie rate',
      value: fmtPct(vote.tie_rate),
      tooltip:
        'Fraction of samples where the vote resulted in a tie. Range 0–100%. Lower is better; ties mean low consensus.',
    },
    {
      label: 'Mean margin',
      value: fmt(vote.mean_margin, 3),
      tooltip:
        'Average vote margin: (top vote count − runner-up vote count). Larger margins mean clearer majorities. For N estimators, values are roughly 0…N.',
    },
    {
      label: 'Mean strength',
      value: fmt(vote.mean_strength, 3),
      tooltip:
        'Average vote strength: (top vote count / total estimators). Range 0…1. Higher = winning label got a larger share of votes.',
    },
    {
      label: 'Net vs best',
      value: fmt(change.net, 0),
      tooltip:
        'Net change vs the best single estimator: corrected − harmed (counts). Positive = ensemble helped more than it hurt.',
    },
  ];

  const metricRows = [
    [metricItems[0], metricItems[1]],
    [metricItems[2], metricItems[3]],
    [metricItems[4], metricItems[5]],
  ];

  // ------------------------
  // Plot traces
  // ------------------------
  const yRange = computeBarRange(means, stds);
  const hasEstimatorScores =
    means.filter((v) => typeof v === 'number' && Number.isFinite(v)).length > 0;

  const barTrace = {
    type: 'bar',
    x: namesPretty,
    y: means.map((v) => (v == null ? null : v)),
    error_y: {
      type: 'data',
      array: stds.map((s) => (s == null ? 0 : s)),
      visible: true,
    },
    marker: { color: BLUE_MAIN },
    hovertemplate: `<b>%{x}</b><br>${metricName}: %{y:.4f}<extra></extra>`,
  };

  const changeTrace = {
    type: 'bar',
    x: ['Corrected', 'Harmed', 'Net'],
    y: [
      safeNum(change.corrected) ?? 0,
      safeNum(change.harmed) ?? 0,
      safeNum(change.net) ?? 0,
    ],
    marker: { color: [BLUE_MAIN, 'var(--mantine-color-gray-5)', BLUE_DARK] },
    hovertemplate: '<b>%{x}</b><br>count: %{y}<extra></extra>',
  };

  const heatmapTrace = matrix
    ? {
        type: 'heatmap',
        x: labelsPretty,
        y: labelsPretty,
        z: matrix,
        zmin: 0,
        zmax: 1,
        colorscale: HEATMAP_COLORSCALE,
        showscale: true,
        // pull colorbar closer + reduce whitespace
        colorbar: {
          x: 1.02,
          xpad: 6,
          thickness: 12,
          len: 0.92,
        },
        text: matrix.map((row) =>
          row.map((v) => (typeof v === 'number' ? v.toFixed(2) : '')),
        ),
        texttemplate: '%{text}',
        hovertemplate:
          '<b>%{y}</b> vs <b>%{x}</b><br>agreement: %{z:.3f}<extra></extra>',
      }
    : null;

  const marginTrace = (() => {
    const edges = Array.isArray(marginHist.edges) ? marginHist.edges : null;
    const counts = Array.isArray(marginHist.counts) ? marginHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;
    const mids = edges.slice(0, -1).map((a, i) => (a + edges[i + 1]) / 2);
    return {
      type: 'bar',
      x: mids,
      y: counts,
      marker: { color: BLUE_MAIN },
      hovertemplate: 'margin bin: %{x}<br>count: %{y}<extra></extra>',
    };
  })();

  const strengthTrace = (() => {
    const edges = Array.isArray(strengthHist.edges) ? strengthHist.edges : null;
    const counts = Array.isArray(strengthHist.counts) ? strengthHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;
    const mids = edges.slice(0, -1).map((a, i) => (a + edges[i + 1]) / 2);
    return {
      type: 'bar',
      x: mids,
      y: counts,
      marker: { color: BLUE_MAIN },
      hovertemplate: 'strength bin: %{x}<br>count: %{y}<extra></extra>',
    };
  })();

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
                    // lighter alternate tint (less "highlighty")
                    backgroundColor:
                      i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
                  }}
                >
                  {/* metric 1 */}
                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip
                      label={pair[0].tooltip}
                      multiline
                      maw={320}
                      withArrow
                    >
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

                  {/* metric 2 */}
                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip
                      label={pair[1].tooltip}
                      multiline
                      maw={320}
                      withArrow
                    >
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
            {bestNamePretty ? (
              <>
                {' '}
                • Best estimator: <b>{bestNamePretty}</b>
              </>
            ) : null}
          </Text>
        </Box>

        <Divider my="xs" />

        {/* Base estimators + Changed vs best */}
        <Group align="stretch" grow wrap="wrap">
          <Box style={{ flex: 1, minWidth: 320 }}>
            <Tooltip
              label="Per-estimator performance across folds (mean ± std). Helps you spot strong/weak and stable/unstable base models."
              multiline
              maw={340}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Base estimators ({metricName})
              </Text>
            </Tooltip>

            {/* Invisible subtitle line to match header height of the right panel (baseline alignment) */}
            <Text
              size="sm"
              c="dimmed"
              align="center"
              mb={6}
              style={{ visibility: 'hidden' }}
            >
              placeholder
            </Text>

            {hasEstimatorScores ? (
              <Plot
                data={[barTrace]}
                layout={{
                  autosize: true,
                  height: 280,
                  // keep same top margin as the right plot so plot baselines align
                  margin: { l: 60, r: 12, t: 10, b: 90 },
                  xaxis: {
                    tickangle: -25,
                    title: 'Estimator',
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  yaxis: {
                    title: metricName,
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
            ) : (
              <Text size="sm" c="dimmed" align="center">
                Base estimator scores unavailable.
              </Text>
            )}
          </Box>

          <Box style={{ flex: 1, minWidth: 320 }}>
            <Tooltip
              label="Counts of samples where the ensemble differs from the best single estimator. “Corrected” = ensemble right / best wrong; “Harmed” = ensemble wrong / best right; “Net” = corrected − harmed."
              multiline
              maw={360}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Changed vs best estimator
              </Text>
            </Tooltip>

            <Text size="sm" c="dimmed" align="center" mb={6}>
              Best: <b>{bestNamePretty || '—'}</b> • total samples:{' '}
              <b>{change.total ?? '—'}</b>
            </Text>

            <Plot
              data={[changeTrace]}
              layout={{
                autosize: true,
                height: 280,
                margin: { l: 60, r: 12, t: 10, b: 60 },
                xaxis: {
                  title: 'Outcome',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                },
                yaxis: {
                  title: 'Count',
                  automargin: true,
                  showgrid: true,
                  zeroline: false,
                },
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </Box>
        </Group>

        <Divider my="xs" />

        {/* Agreement heatmap */}
        <Box>
          <Tooltip
            label="Agreement between each pair of base estimators (0–1). 1.0 = always match. Useful to see redundancy (very high agreement) vs diversity (lower agreement)."
            multiline
            maw={360}
            withArrow
          >
            <Text size="lg" fw={500} align="center" mb={6}>
              Pairwise agreement
            </Text>
          </Tooltip>

          {heatmapTrace ? (
            <Plot
              data={[heatmapTrace]}
              layout={{
                autosize: true,
                height: 360,
                // tighter margins so y labels sit closer, and colorbar is closer
                margin: { l: 70, r: 55, t: 10, b: 90 },
                xaxis: {
                  title: 'Estimator',
                  tickangle: -30,
                  side: 'top',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  constrain: 'domain',
                },
                yaxis: {
                  title: 'Estimator',
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
              Agreement matrix unavailable.
            </Text>
          )}
        </Box>

        <Divider my="xs" />

        {/* Histograms */}
        <Group align="stretch" grow wrap="wrap">
          <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 320 }}>
            <Tooltip
              label="Distribution of vote margins (top votes − runner-up votes). Larger margins mean clearer majorities."
              multiline
              maw={320}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Vote margins
              </Text>
            </Tooltip>

            {marginTrace ? (
              <Plot
                data={[marginTrace]}
                layout={{
                  autosize: true,
                  height: 220,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    title: 'Margin (top − runner-up)',
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  yaxis: {
                    title: 'Count',
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

          <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 320 }}>
            <Tooltip
              label="Distribution of vote strengths (top votes / total estimators). Values closer to 1 mean stronger consensus."
              multiline
              maw={340}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Vote strength
              </Text>
            </Tooltip>

            {strengthTrace ? (
              <Plot
                data={[strengthTrace]}
                layout={{
                  autosize: true,
                  height: 220,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    title: 'Strength (top / total)',
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  yaxis: {
                    title: 'Count',
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
