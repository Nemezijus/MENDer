import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import { inferSquareLabels, safeNum } from '../../../utils/resultsFormat.js';
import {
  buildAbsDiffHeatmapTrace,
  buildCorrHeatmapTrace,
} from '../../../utils/resultHelpers.js';

export default function BaggingRegMatricesSection({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const bag = report.bagging || {};
  const sim = report.similarity || report.diversity || {};

  const nEstimators = safeNum(bag.n_estimators);

  const corrRaw =
    Array.isArray(sim.pairwise_corr)
      ? sim.pairwise_corr
      : Array.isArray(sim.corr_matrix)
      ? sim.corr_matrix
      : null;

  const absRaw =
    Array.isArray(sim.pairwise_absdiff)
      ? sim.pairwise_absdiff
      : Array.isArray(sim.absdiff_matrix)
      ? sim.absdiff_matrix
      : null;

  const labelsRaw =
    Array.isArray(sim.labels)
      ? sim.labels
      : Array.isArray(sim.estimator_labels)
      ? sim.estimator_labels
      : null;

  const sizeFromMatrix =
    corrRaw && Array.isArray(corrRaw) && Array.isArray(corrRaw[0]) ? corrRaw.length : null;

  const labels = labelsRaw || inferSquareLabels(sizeFromMatrix || nEstimators || 0);
  const showMatrixText = Array.isArray(labels) ? labels.length < 10 : true;

  const corrTrace = buildCorrHeatmapTrace({
    matrix: corrRaw,
    labels,
    showText: showMatrixText,
    hoverValueSource: 'customdata',
  });

  const absTrace = buildAbsDiffHeatmapTrace({
    matrix: absRaw,
    labels,
    showText: showMatrixText,
  });

  return (
    <Box>
      <SectionTitle
        title="Pairwise prediction structure"
        tooltip="Pairwise relationships between base estimators. Correlation close to 1 means very similar predictions; |Δ| highlights how far predictions differ on average."
        maw={420}
      />

      <Group align="stretch" grow wrap="wrap">
        <Box className="ensFlexMin320">
          <Text size="md" fw={500} align="center" mb={6}>
            Prediction similarity (corr)
          </Text>

          {corrTrace ? (
            <Plot
              data={[corrTrace]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 80, r: 36, t: 10, b: 90 },
                xaxis: {
                  title: { text: 'Estimator' },
                  tickangle: -30,
                  side: 'top',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  constrain: 'domain',
                },
                yaxis: {
                  title: { text: 'Estimator' },
                  autorange: 'reversed',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  scaleanchor: 'x',
                  scaleratio: 1,
                  constrain: 'domain',
                },
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="ensPlotFullWidth"
            />
          ) : (
            <Text size="sm" c="dimmed" align="center">
              Correlation matrix unavailable.
            </Text>
          )}
        </Box>

        <Box className="ensFlexMin320">
          <Text size="md" fw={500} align="center" mb={6}>
            Absolute prediction differences (|Δ|)
          </Text>

          {absTrace ? (
            <Plot
              data={[absTrace]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 80, r: 36, t: 10, b: 90 },
                xaxis: {
                  title: { text: 'Estimator' },
                  tickangle: -30,
                  side: 'top',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  constrain: 'domain',
                },
                yaxis: {
                  title: { text: 'Estimator' },
                  autorange: 'reversed',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  scaleanchor: 'x',
                  scaleratio: 1,
                  constrain: 'domain',
                },
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="ensPlotFullWidth"
            />
          ) : (
            <Text size="sm" c="dimmed" align="center">
              Absolute-difference matrix unavailable.
            </Text>
          )}
        </Box>
      </Group>
    </Box>
  );
}
