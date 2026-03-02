import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace } from '../../../utils/resultsFormat.js';

export default function BaggingRegScoreHistogramSection({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const metricName = report.metric_name || '';
  const baseScores = report.base_estimator_scores || {};

  const scoreHist = baseScores.hist || {};
  const scorePlot =
    scoreHist && scoreHist.edges && scoreHist.counts
      ? histToBarTrace(scoreHist.edges, scoreHist.counts, {
          color: cmBlue(0.7),
          xLabel: `Base-estimator ${metricName || 'score'}`,
          hoverLabel: 'score bin',
        })
      : null;

  return (
    <Box>
      <SectionTitle
        title="Base-estimator score distribution"
        tooltip="Distribution of base-estimator scores (computed per fold on the evaluation split), if the backend provides it."
        maw={420}
      />

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
          className="ensPlotFullWidth"
        />
      ) : (
        <Text size="sm" c="dimmed" align="center">
          Base-estimator score histogram unavailable.
        </Text>
      )}
    </Box>
  );
}
