import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { cmBlue, histToBarTrace, safeNum } from '../../../utils/resultsFormat.js';

export default function AdaBoostClsWeightsAndErrorsSection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const ada = report.adaboost || {};
  const stages = report.stages || {};
  const weights = report.weights || {};
  const errors = report.errors || {};

  const nEstimatorsConfigured = safeNum(ada.n_estimators);
  const nEstimatorsFittedMean = safeNum(stages.n_estimators_fitted_mean);

  const hideDenseLabels =
    (nEstimatorsConfigured != null && nEstimatorsConfigured > 20) ||
    (nEstimatorsFittedMean != null && nEstimatorsFittedMean > 20);

  const wHist = weights.hist || null;
  const eHist = errors.hist || null;

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

      <Box style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Estimator errors"
          tooltip="Stage errors (if available). Lower is better; a spread indicates varying weak-learner performance."
          maw={380}
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
            Error histogram unavailable (not provided by backend).
          </Text>
        )}
      </Box>
    </Group>
  );
}
