import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import {
  AXIS_TITLE,
  MENDER_BLUE_SCALE,
  PLOT_BG,
  PLOT_MARGIN_STD,
} from '../../../utils/plotly.js';

import { PLOT_HEIGHT, TILE_STACK_STYLE } from '../common/styles.js';

export default function SeparationMatrixSection({ sepMatrix, clusterLabel }) {
  if (!sepMatrix) return null;
  const ids = sepMatrix?.cluster_ids;
  const z = sepMatrix?.values;
  if (!Array.isArray(ids) || !Array.isArray(z) || !z.length) return null;

  return (
    <Stack gap={4} style={TILE_STACK_STYLE}>
      <PlotHeader
        title="Pairwise cluster separation matrix"
        help="Pairwise distances between cluster centroids (in feature space)."
      />

      <div style={{ width: '100%', marginTop: 'auto' }}>
        <div style={{ width: '100%', aspectRatio: '1 / 1', minHeight: PLOT_HEIGHT }}>
          <Plot
            data={[
              {
                type: 'heatmap',
                z,
                x: ids.map((c) => clusterLabel(c)),
                y: ids.map((c) => clusterLabel(c)),
                hovertemplate: '%{y} vs %{x}<br>Distance=%{z:.3f}<extra></extra>',
                colorscale: MENDER_BLUE_SCALE,
                showscale: false,
              },
            ]}
            layout={{
              margin: { ...PLOT_MARGIN_STD, l: 60, t: 14, b: 50 },
              xaxis: {
                title: AXIS_TITLE('Cluster'),
                tickfont: { size: 10 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                automargin: true,
                constrain: 'domain',
              },
              yaxis: {
                title: AXIS_TITLE('Cluster'),
                tickfont: { size: 10 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                automargin: true,
                scaleanchor: 'x',
                scaleratio: 1,
                constrain: 'domain',
              },
              ...PLOT_BG,
            }}
            config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </Stack>
  );
}
