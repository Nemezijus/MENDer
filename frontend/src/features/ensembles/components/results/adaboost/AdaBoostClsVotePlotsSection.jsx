import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace } from '../../../utils/resultsFormat.js';

export default function AdaBoostClsVotePlotsSection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const vote = report.vote || {};

  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};

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

  return (
    <Group align="stretch" grow wrap="wrap">
      <Box className="ensPlotColWide">
        <SectionTitle
          title="Weighted vote margins"
          tooltip="Weighted vote margin distribution. Higher means clearer weighted majorities."
          maw={360}
        />

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
            className="ensPlotFullWidth"
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Margin histogram unavailable.
          </Text>
        )}
      </Box>

      <Box className="ensPlotColWide">
        <SectionTitle
          title="Weighted vote strength"
          tooltip="Weighted vote strength distribution: (top weight / total weight)."
          maw={360}
        />

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
            className="ensPlotFullWidth"
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Strength histogram unavailable.
          </Text>
        )}
      </Box>
    </Group>
  );
}
