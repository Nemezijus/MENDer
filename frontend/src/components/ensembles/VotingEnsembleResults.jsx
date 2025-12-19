// frontend/src/components/ensembles/VotingEnsembleResults.jsx
import { Card, Group, Stack, Text, Divider } from '@mantine/core';
import Plot from 'react-plotly.js';

function safeNum(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function fmtPct(x, digits = 1) {
  const n = safeNum(x);
  if (n == null) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}

function fmt(x, digits = 4) {
  const n = safeNum(x);
  if (n == null) return '—';
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

export default function VotingEnsembleResults({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  const names = estimators.map((e) => e?.name ?? '');
  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const agreement = report.agreement || {};
  const matrix = Array.isArray(agreement.matrix) ? agreement.matrix : null;
  const labels = Array.isArray(agreement.labels) ? agreement.labels : names;

  const vote = report.vote || {};
  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};

  const change = report.change_vs_best || {};

  const barTrace = {
    type: 'bar',
    x: names,
    y: means,
    error_y: { type: 'data', array: stds.map((s) => (s == null ? 0 : s)), visible: true },
  };

  const heatmapTrace = matrix
    ? {
        type: 'heatmap',
        x: labels,
        y: labels,
        z: matrix,
        zmin: 0,
        zmax: 1,
      }
    : null;

  const histTrace = (() => {
    const edges = Array.isArray(marginHist.edges) ? marginHist.edges : null;
    const counts = Array.isArray(marginHist.counts) ? marginHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;
    const mids = edges.slice(0, -1).map((a, i) => (a + edges[i + 1]) / 2);
    return { type: 'bar', x: mids, y: counts };
  })();

  const strengthTrace = (() => {
    const edges = Array.isArray(strengthHist.edges) ? strengthHist.edges : null;
    const counts = Array.isArray(strengthHist.counts) ? strengthHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;
    const mids = edges.slice(0, -1).map((a, i) => (a + edges[i + 1]) / 2);
    return { type: 'bar', x: mids, y: counts };
  })();

  const changeTrace = {
    type: 'bar',
    x: ['corrected', 'harmed', 'net'],
    y: [safeNum(change.corrected) ?? 0, safeNum(change.harmed) ?? 0, safeNum(change.net) ?? 0],
  };

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Group justify="space-between" align="center">
          <Text fw={600}>Voting ensemble insights</Text>
          <Text size="sm" c="dimmed">
            {report.voting} voting • {report.n_estimators} estimators
          </Text>
        </Group>

        <Group gap="xl" wrap="wrap">
          <Text size="sm">
            <b>All-agree:</b> {fmtPct(agreement.all_agree_rate)}
          </Text>
          <Text size="sm">
            <b>Avg pairwise agreement:</b> {fmtPct(agreement.pairwise_mean_agreement)}
          </Text>
          <Text size="sm">
            <b>Tie rate:</b> {fmtPct(vote.tie_rate)}
          </Text>
          <Text size="sm">
            <b>Mean vote margin:</b> {fmt(vote.mean_margin, 3)}
          </Text>
          <Text size="sm">
            <b>Mean vote strength:</b> {fmt(vote.mean_strength, 3)}
          </Text>
        </Group>

        <Divider my="xs" />

        <Group align="stretch" grow wrap="wrap">
          <Card withBorder radius="md" p="sm">
            <Text size="sm" fw={600} mb={6}>
              Base estimators ({metricName})
            </Text>
            <Plot
              data={[barTrace]}
              layout={{
                autosize: true,
                height: 260,
                margin: { l: 40, r: 10, t: 10, b: 80 },
                xaxis: { tickangle: -35 },
                yaxis: { title: metricName },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </Card>

          <Card withBorder radius="md" p="sm">
            <Text size="sm" fw={600} mb={6}>
              Changed vs best estimator
            </Text>
            <Text size="sm" c="dimmed" mb={6}>
              Best: {change.best_name || '—'} • total: {change.total ?? '—'}
            </Text>
            <Plot
              data={[changeTrace]}
              layout={{
                autosize: true,
                height: 260,
                margin: { l: 40, r: 10, t: 10, b: 40 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </Card>
        </Group>

        <Group align="stretch" grow wrap="wrap">
          <Card withBorder radius="md" p="sm">
            <Text size="sm" fw={600} mb={6}>
              Pairwise agreement
            </Text>
            {heatmapTrace ? (
              <Plot
                data={[heatmapTrace]}
                layout={{
                  autosize: true,
                  height: 280,
                  margin: { l: 60, r: 10, t: 10, b: 60 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : (
              <Text size="sm" c="dimmed">
                Agreement matrix unavailable.
              </Text>
            )}
          </Card>

          <Card withBorder radius="md" p="sm">
            <Text size="sm" fw={600} mb={6}>
              Vote margins / confidence
            </Text>
            <Group grow wrap="wrap">
              <div style={{ width: '100%' }}>
                <Text size="xs" c="dimmed" mb={4}>
                  Margin histogram
                </Text>
                {histTrace ? (
                  <Plot
                    data={[histTrace]}
                    layout={{
                      autosize: true,
                      height: 170,
                      margin: { l: 40, r: 10, t: 10, b: 40 },
                      xaxis: { title: 'margin' },
                      yaxis: { title: 'count' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                ) : (
                  <Text size="sm" c="dimmed">
                    Margin histogram unavailable.
                  </Text>
                )}
              </div>

              <div style={{ width: '100%' }}>
                <Text size="xs" c="dimmed" mb={4}>
                  Strength histogram
                </Text>
                {strengthTrace ? (
                  <Plot
                    data={[strengthTrace]}
                    layout={{
                      autosize: true,
                      height: 170,
                      margin: { l: 40, r: 10, t: 10, b: 40 },
                      xaxis: { title: 'top vote / total' },
                      yaxis: { title: 'count' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                ) : (
                  <Text size="sm" c="dimmed">
                    Strength histogram unavailable.
                  </Text>
                )}
              </div>
            </Group>
          </Card>
        </Group>
      </Stack>
    </Card>
  );
}
