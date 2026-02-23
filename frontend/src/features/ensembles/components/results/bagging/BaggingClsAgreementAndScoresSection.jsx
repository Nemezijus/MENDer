import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  HEATMAP_COLORSCALE,
  cmBlue,
  histToBarTrace,
  safeNum,
} from '../../../utils/resultsFormat.js';

export default function BaggingClsAgreementAndScoresSection({ report }) {
  if (!report || report.kind !== 'bagging') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const metricName = report.metric_name || '';

  const bag = report.bagging || {};
  const diversity = report.diversity || {};
  const baseScores = report.base_estimator_scores || {};

  const nEstimators = safeNum(bag.n_estimators);
  const hideDenseLabels = nEstimators != null && nEstimators > 20;

  const matrix = Array.isArray(diversity.matrix) ? diversity.matrix : null;
  let labels = Array.isArray(diversity.labels) ? diversity.labels : null;

  if (!labels && matrix && Array.isArray(matrix) && Array.isArray(matrix[0])) {
    const m = matrix.length;
    labels = Array.from({ length: m }, (_, i) => `est_${i + 1}`);
  }

  const heatmapTrace =
    matrix && labels
      ? {
          type: 'heatmap',
          x: labels,
          y: labels,
          z: matrix,
          zmin: 0,
          zmax: 1,
          colorscale: HEATMAP_COLORSCALE,
          showscale: true,
          colorbar: {
            x: 1.3,
            xanchor: 'right',
            xpad: 0,
            thickness: 12,
            len: 0.7,
            outlinewidth: 0,
          },
          hovertemplate:
            '<b>%{y}</b> vs <b>%{x}</b><br>agreement: %{z:.3f}<extra></extra>',
        }
      : null;

  const scoreHist = baseScores.hist || {};
  const scorePlot = histToBarTrace(scoreHist.edges, scoreHist.counts, {
    color: cmBlue(0.7),
    xLabel: `Base-estimator ${metricName || 'score'}`,
    hoverLabel: 'score bin',
    xRange: [0, 1],
  });

  return (
    <Group align="stretch" grow wrap="wrap">
      <Box style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Estimator agreement"
          tooltip="Agreement between bagged estimators (0–1). High agreement means redundancy; lower agreement indicates diversity."
          maw={380}
        />

        {heatmapTrace ? (
          <Plot
            data={[heatmapTrace]}
            layout={{
              autosize: true,
              height: 460,
              margin: { l: 50, r: 10, t: 10, b: 50 },
              xaxis: {
                title: { text: hideDenseLabels ? '' : 'Estimator' },
                tickangle: -30,
                side: 'top',
                automargin: true,
                showgrid: false,
                zeroline: false,
                constrain: 'domain',
                showticklabels: !hideDenseLabels,
              },
              yaxis: {
                title: { text: hideDenseLabels ? '' : 'Estimator' },
                autorange: 'reversed',
                automargin: true,
                showgrid: false,
                zeroline: false,
                scaleanchor: 'x',
                scaleratio: 1,
                constrain: 'domain',
                showticklabels: !hideDenseLabels,
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

      <Box style={{ flex: 1, minWidth: 340 }}>
        <SectionTitle
          title="Base-estimator score distribution"
          tooltip="Distribution of base-estimator scores (computed per fold on the evaluation split)."
          maw={380}
        />

        {scorePlot?.trace ? (
          <Plot
            data={[scorePlot.trace]}
            layout={{
              autosize: true,
              height: 460,
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
            style={{ width: '100%' }}
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Base-estimator score histogram unavailable.
          </Text>
        )}
      </Box>
    </Group>
  );
}
