import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace } from '../../../utils/resultsFormat.js';

export default function AdaBoostClsStageScoreSection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const metricName = report.metric_name || '';
  const baseScores = report.base_estimator_scores || {};

  const scoreHist = baseScores.hist || null;

  const baseScorePlot =
    scoreHist && scoreHist.edges && scoreHist.counts
      ? histToBarTrace(scoreHist.edges, scoreHist.counts, {
          color: cmBlue(0.7),
          xLabel: `Stage ${metricName || 'score'}`,
          hoverLabel: 'score bin',
          xRange: [0, 1],
        })
      : null;

  return (
    <Box>
      <SectionTitle
        title="Stage score distribution"
        tooltip="Optional distribution of per-stage scores on the evaluation split (if the backend provides it)."
        maw={380}
      />

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
          className="ensPlotFullWidth"
        />
      ) : (
        <Text size="sm" c="dimmed" align="center">
          No per-stage score distribution available (backend not provided yet).
        </Text>
      )}
    </Box>
  );
}
