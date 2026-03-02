import { Box, Group, Text } from '@mantine/core';
import { useMantineTheme } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  cmBlue,
  computeBarRange,
  makeUniqueLabels,
  normalize01,
  prettyEstimatorName,
  safeNum,
} from '../../../utils/resultsFormat.js';

export default function VotingClsBaseAndChangeSection({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  const namesRaw = estimators.map((e) => e?.name ?? '');
  const namesPretty = makeUniqueLabels(namesRaw.map((n) => prettyEstimatorName(n)));

  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const yRange = computeBarRange(means, stds);
  const hasEstimatorScores =
    means.filter((v) => typeof v === 'number' && Number.isFinite(v)).length > 0;

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

  const change = report.change_vs_best || {};
  const bestNamePretty = prettyEstimatorName(change.best_name || '');

  const theme = useMantineTheme();

  const changeTrace = {
    type: 'bar',
    x: ['Corrected', 'Harmed', 'Net'],
    y: [
      safeNum(change.corrected) ?? 0,
      safeNum(change.harmed) ?? 0,
      safeNum(change.net) ?? 0,
    ],
    marker: { color: [cmBlue(0.65), theme.colors.gray[5], cmBlue(0.9)] },
    hovertemplate: '<b>%{x}</b><br>count: %{y}<extra></extra>',
  };

  return (
    <Group align="stretch" grow wrap="wrap">
      <Box className="ensFlexMin320">
        <SectionTitle
          title={`Base estimators (${metricName})`}
          tooltip="Per-estimator performance across folds (mean ± std). Helps you spot strong/weak and stable/unstable base models."
          maw={340}
        />

        {/* keep placeholder line to match original vertical spacing */}
        <Text size="sm" c="dimmed" align="center" mb={6} className="ensHidden">
          placeholder
        </Text>

        {hasEstimatorScores ? (
          <Plot
            data={[barTrace]}
            layout={{
              autosize: true,
              height: 280,
              margin: { l: 60, r: 12, t: 10, b: 90 },
              xaxis: {
                tickangle: -25,
                title: { text: 'Estimator' },
                automargin: true,
                showgrid: false,
                zeroline: false,
              },
              yaxis: {
                title: { text: metricName },
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
            className="ensPlotFullWidth"
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Base estimator scores unavailable.
          </Text>
        )}
      </Box>

      <Box className="ensFlexMin320">
        <SectionTitle
          title="Changed vs best estimator"
          tooltip="Counts of samples where the ensemble differs from the best single estimator. “Corrected” = ensemble right / best wrong; “Harmed” = ensemble wrong / best right; “Net” = corrected − harmed."
          maw={360}
        />

        <Text size="sm" c="dimmed" align="center" mb={6}>
          Best: <b>{bestNamePretty || '—'}</b> • total samples: <b>{change.total ?? '—'}</b>
        </Text>

        <Plot
          data={[changeTrace]}
          layout={{
            autosize: true,
            height: 280,
            margin: { l: 60, r: 12, t: 10, b: 60 },
            xaxis: {
              title: { text: 'Outcome' },
              automargin: true,
              showgrid: false,
              zeroline: false,
            },
            yaxis: {
              title: { text: 'Count' },
              automargin: true,
              showgrid: true,
              zeroline: false,
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="ensPlotFullWidth"
        />
      </Box>
    </Group>
  );
}
