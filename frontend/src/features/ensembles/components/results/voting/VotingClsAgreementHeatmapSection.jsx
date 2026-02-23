import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  HEATMAP_COLORSCALE,
  makeUniqueLabels,
  prettyEstimatorName,
} from '../../../utils/resultsFormat.js';

export default function VotingClsAgreementHeatmapSection({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const agreement = report.agreement || {};
  const matrix = Array.isArray(agreement.matrix) ? agreement.matrix : null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];
  const namesRaw = estimators.map((e) => e?.name ?? '');

  const labelsRaw = Array.isArray(agreement.labels) ? agreement.labels : namesRaw;
  const labelsPretty = makeUniqueLabels(labelsRaw.map((n) => prettyEstimatorName(n)));

  const heatmapTrace = matrix
    ? {
        type: 'heatmap',
        x: labelsPretty,
        y: labelsPretty,
        z: matrix,
        zmin: 0,
        zmax: 1,
        colorscale: HEATMAP_COLORSCALE,
        showscale: true,
        colorbar: {
          x: 0.75,
          xanchor: 'right',
          xpad: 0,
          thickness: 12,
          len: 0.92,
          outlinewidth: 0,
        },
        text: matrix.map((row) =>
          row.map((v) => (typeof v === 'number' ? v.toFixed(2) : '')),
        ),
        texttemplate: '%{text}',
        hovertemplate:
          '<b>%{y}</b> vs <b>%{x}</b><br>agreement: %{z:.3f}<extra></extra>',
      }
    : null;

  return (
    <Box>
      <SectionTitle
        title="Pairwise agreement"
        tooltip="Agreement between each pair of base estimators (0–1). 1.0 = always match. Useful to see redundancy (very high agreement) vs diversity (lower agreement)."
        maw={360}
      />

      {heatmapTrace ? (
        <Plot
          data={[heatmapTrace]}
          layout={{
            autosize: true,
            height: 360,
            margin: { l: 70, r: 28, t: 10, b: 90 },
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
          style={{ width: '100%' }}
        />
      ) : (
        <Text size="sm" c="dimmed" align="center">
          Agreement matrix unavailable.
        </Text>
      )}
    </Box>
  );
}
