import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace } from '../../../utils/resultsFormat.js';

export default function AdaBoostRegWeightsAndErrorsSection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const weights = report.weights || {};
  const estErrors = report.errors || {};

  const wHist = weights.hist || null;
  const eHist = estErrors.hist || null;

  const weightPlot =
    wHist && wHist.edges && wHist.counts
      ? histToBarTrace(wHist.edges, wHist.counts, {
          color: cmBlue(0.65),
          xLabel: 'Estimator weight',
          hoverLabel: 'weight bin',
        })
      : null;

  const errorPlot =
    eHist && eHist.edges && eHist.counts
      ? histToBarTrace(eHist.edges, eHist.counts, {
          color: cmBlue(0.65),
          xLabel: 'Estimator error',
          hoverLabel: 'error bin',
        })
      : null;

  return (
    <Group align="stretch" grow wrap="wrap">
      <Box style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Estimator weights"
          tooltip="Distribution of boosting stage weights. If weights concentrate, effective estimator count decreases."
          maw={380}
        />

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

      <Box style={{ flex: 1, minWidth: 340 }} maw={380}>
        <SectionTitle
          title="Estimator errors"
          tooltip='Estimator "error" is algorithm-defined; for AdaBoostRegressor this is typically based on loss.'

        />

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
            Error histogram unavailable.
          </Text>
        )}
      </Box>
    </Group>
  );
}
