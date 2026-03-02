import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  cmBlue,
  computeBarRange,
  fmt,
  makeUniqueLabels,
  niceEstimatorLabel,
  normalize01,
  safeNum,
} from '../../../utils/resultsFormat.js';

export default function VotingRegBaseEstimatorsSection({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const metricName = report.metric_name || '';

  const baseLabelsLong = estimators.map((e) =>
    niceEstimatorLabel({ name: e?.name, algo: e?.algo }),
  );
  const namesPretty = makeUniqueLabels(baseLabelsLong);

  const means = estimators.map((e) => safeNum(e?.mean));
  const stds = estimators.map((e) => safeNum(e?.std));

  const hasEstimatorScores =
    means.filter((v) => typeof v === 'number' && Number.isFinite(v)).length > 0;

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

  return (
    <Box>
      <SectionTitle
        title={`Base estimators (${metricName || 'score'})`}
        tooltip="Per-estimator performance across folds (mean ± std). Helps you spot strong/weak and stable/unstable base models."
        maw={360}
      />

      {!hasEstimatorScores ? (
        <Text size="sm" c="dimmed" align="center" mt="xs">
          Base estimator scores unavailable.
        </Text>
      ) : (
        <Box className="votingPlotNarrow">
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
            className="ensPlotFullWidth"
          />
        </Box>
      )}
    </Box>
  );
}
