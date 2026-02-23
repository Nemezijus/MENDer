import { Card, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace, safeNum } from '../../../utils/resultsFormat.js';

export default function BaggingClsVoteHistogramsSection({ report }) {
  if (!report || report.kind !== 'bagging') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const bag = report.bagging || {};
  const vote = report.vote || {};

  const nEstimators = safeNum(bag.n_estimators);
  const hideDenseLabels = nEstimators != null && nEstimators > 20;

  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};

  const marginPlot = histToBarTrace(marginHist.edges, marginHist.counts, {
    color: cmBlue(0.75),
    xLabel: 'Margin (top − runner-up)',
    hoverLabel: 'margin bin',
    isIntegerBins: true,
    hideTickLabels: hideDenseLabels,
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

  return (
    <Group align="stretch" grow wrap="wrap">
      <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Vote margins"
          tooltip="Distribution of vote margins (top votes − runner-up votes)."
          maw={360}
        />

        {marginPlot?.trace ? (
          <Plot
            data={[marginPlot.trace]}
            layout={{
              autosize: true,
              height: 240,
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
      </Card>

      <Card withBorder={false} radius="md" p="sm" style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Vote strength"
          tooltip="Distribution of vote strength (top votes / total estimators). With N estimators, strengths are typically discrete: k/N."
          maw={380}
        />

        {strengthPlot?.trace ? (
          <Plot
            data={[strengthPlot.trace]}
            layout={{
              autosize: true,
              height: 240,
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
      </Card>
    </Group>
  );
}
